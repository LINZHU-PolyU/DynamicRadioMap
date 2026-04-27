"""Dataset and DataLoader for dynamic radio map prediction."""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
#  Building-level split
# ---------------------------------------------------------------------------
def split_buildings(
    num_tiles: int = 5,
    num_subtiles: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Tuple[int, int]]]:
    """Shuffle all (tile_idx, subtile_idx) building pairs and split into train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    buildings = [(tile_idx, subtile_idx) for tile_idx in range(num_tiles) for subtile_idx in range(num_subtiles)]
    rng = random.Random(seed)
    rng.shuffle(buildings)

    n_total = len(buildings)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    return {
        "train": buildings[:n_train],
        "val": buildings[n_train:n_train + n_val],
        "test": buildings[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
#  Augmentation
# ---------------------------------------------------------------------------
def random_spatial_rotation(
    building_map: np.ndarray,
    radio_map: np.ndarray,
    trajectory: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate map, sequence, and trajectory by a random multiple of 90 degrees."""
    k = rng.randint(0, 3)
    if k == 0:
        return building_map, radio_map, trajectory

    building_map = np.rot90(building_map, k=k, axes=(0, 1)).copy()
    radio_map = np.rot90(radio_map, k=k, axes=(1, 2)).copy()

    x, y = trajectory[:, 0].copy(), trajectory[:, 1].copy()
    if k == 1:
        trajectory = np.stack([y, 1.0 - x], axis=-1)
    elif k == 2:
        trajectory = np.stack([1.0 - x, 1.0 - y], axis=-1)
    else:
        trajectory = np.stack([1.0 - y, x], axis=-1)

    return building_map, radio_map, trajectory


def random_spatial_flip(
    building_map: np.ndarray,
    radio_map: np.ndarray,
    trajectory: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random horizontal / vertical flips applied consistently to all modalities."""
    # MODIFIED: add flips to enlarge the augmentation group beyond rotations.
    if rng.random() < 0.5:
        building_map = np.flip(building_map, axis=1).copy()
        radio_map = np.flip(radio_map, axis=2).copy()
        trajectory = trajectory.copy()
        trajectory[:, 0] = 1.0 - trajectory[:, 0]

    if rng.random() < 0.5:
        building_map = np.flip(building_map, axis=0).copy()
        radio_map = np.flip(radio_map, axis=1).copy()
        trajectory = trajectory.copy()
        trajectory[:, 1] = 1.0 - trajectory[:, 1]

    return building_map, radio_map, trajectory


def temporal_reversal(
    trajectory: np.ndarray,
    radio_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse the time axis of the trajectory and radio map."""
    trajectory = trajectory[::-1].copy()
    radio_map = radio_map[::-1].copy()
    return trajectory, radio_map


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class DynamicRadioMapDataset(Dataset):
    """PyTorch Dataset for the Dynamic Radio Map prediction task."""

    def __init__(
        self,
        buildings: List[Tuple[int, int]],
        data_root: str,
        num_trajs: int = 30,
        num_frames: int = 32,
        augment: bool = False,
        aug_rotation: bool = True,
        aug_temporal: bool = True,
        aug_flip: bool = True,
        dataset_stats: dict = None,
        seed: int = 42,
    ):
        super().__init__()
        required = {
            "building_height_max",
            "area_size",
            "threshold",
            "global_rm_max",
            "global_rm_mean",
            "global_rm_std",
        }
        if dataset_stats is None or not required.issubset(dataset_stats):
            missing = required if dataset_stats is None else required - set(dataset_stats.keys())
            raise ValueError(f"dataset_stats is missing required keys: {missing}")

        self.buildings = buildings
        self.data_root = data_root
        self.num_trajs = num_trajs
        self.num_frames = num_frames
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_temporal = aug_temporal
        self.aug_flip = aug_flip  # MODIFIED: support mirror augmentations.
        self.dataset_stats = dataset_stats

        # Build flat index: list of (tile_idx, subtile_idx, traj_idx)
        self.samples: List[Tuple[int, int, int]] = []
        for tile_idx, subtile_idx in buildings:
            for traj_idx in range(num_trajs):
                self.samples.append((tile_idx, subtile_idx, traj_idx))

        # Load trajectories array once into memory (shared across samples)
        trajs_path = os.path.join(data_root, "trajs_array.npy")
        self.trajs_array: np.ndarray = np.load(trajs_path)

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_idx, subtile_idx, traj_idx = self.samples[idx]

        # --- Load building map ---
        bld_path = os.path.join(self.data_root, "buildings", f"{tile_idx}_{subtile_idx}.npy")
        building_map: np.ndarray = np.load(bld_path)

        # --- Trajectory ---
        trajectory: np.ndarray = self.trajs_array[tile_idx, subtile_idx, traj_idx].copy()

        # --- Load radio map ---
        rm_path = os.path.join(self.data_root, "raw_radio_maps", f"{tile_idx}_{subtile_idx}_{traj_idx}.npy")
        radio_map: np.ndarray = np.load(rm_path)

        # --- Normalize inputs ---
        building_map = building_map / self.dataset_stats['building_height_max']
        trajectory = trajectory / self.dataset_stats['area_size']

        # --- Normalize & standardize output ---
        threshold = self.dataset_stats['threshold']
        global_radio_map_max = self.dataset_stats['global_rm_max']
        global_radio_map_mean = self.dataset_stats['global_rm_mean']
        global_radio_map_std = self.dataset_stats['global_rm_std']
        radio_map[radio_map <= threshold] = threshold
        radio_map = (radio_map - threshold) / (global_radio_map_max - threshold)
        radio_map = (radio_map - global_radio_map_mean) / global_radio_map_std

        # --- Augmentation ---
        if self.augment:
            if self.aug_flip:
                building_map, radio_map, trajectory = random_spatial_flip(
                    building_map, radio_map, trajectory, self._rng,
                )
            if self.aug_rotation:
                building_map, radio_map, trajectory = random_spatial_rotation(
                    building_map, radio_map, trajectory, self._rng,
                )
            if self.aug_temporal and self._rng.random() < 0.5:
                trajectory, radio_map = temporal_reversal(trajectory, radio_map)

        # --- Convert to tensors ---
        building_map_t = torch.from_numpy(building_map).unsqueeze(0).float()
        trajectory_t = torch.from_numpy(trajectory).float()
        radio_map_t = torch.from_numpy(radio_map).unsqueeze(1).float()

        # --- Truncate frames ---
        total_frames = trajectory_t.shape[0]
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        trajectory_t = trajectory_t[indices]
        radio_map_t = radio_map_t[indices]

        return building_map_t, trajectory_t, radio_map_t


# ---------------------------------------------------------------------------
#  Worker init
# ---------------------------------------------------------------------------
def _worker_init_fn(worker_id: int):
    """Re-seed the dataset's internal RNG per worker so augmentations differ."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._rng = random.Random(worker_info.seed % (2 ** 32))


# ---------------------------------------------------------------------------
#  DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_root: str = "/home/user/datasets/DynamicRadioMap",
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    num_trajs: int = 30,
    num_frames: int = 32,
    pin_memory: bool = True,
    augment_train: bool = True,
    dataset_stats: dict = None,
    aug_rotation: bool = True,
    aug_temporal: bool = True,
    aug_flip: bool = True,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    splits = split_buildings(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    loaders = {}
    for split_name in ("train", "val", "test"):
        is_train = split_name == "train"
        ds = DynamicRadioMapDataset(
            buildings=splits[split_name],
            data_root=data_root,
            num_trajs=num_trajs,
            num_frames=num_frames,
            augment=is_train and augment_train,
            aug_rotation=aug_rotation,
            aug_temporal=aug_temporal,
            aug_flip=aug_flip,
            dataset_stats=dataset_stats,
            seed=seed,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            worker_init_fn=_worker_init_fn,
            generator=generator if is_train else None,
            persistent_workers=num_workers > 0,
        )

    return loaders


# ---------------------------------------------------------------------------
#  Quick verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    dataset_stats = {
        'building_height_max': 40.0,
        'area_size': 128.0,
        'threshold': -110.0,
        'global_rm_max': -83.9375,
        'global_rm_mean': 0.46968572689996413,
        'global_rm_std': 0.37251425724702003
    }

    parser = argparse.ArgumentParser(description="DynamicRadioMap Dataset verification")
    parser.add_argument("--data_root", type=str, default="/home/user/datasets/DynamicRadioMap")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 60)
    print("  DynamicRadioMap Dataset — Verification")
    print("=" * 60)

    loaders = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dataset_stats=dataset_stats,
    )

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        bld, traj, rm = batch
        print(f"[{split_name}]  batch shapes:")
        print(f"    building_map : {bld.shape}   dtype={bld.dtype}")
        print(f"    trajectory   : {traj.shape}  dtype={traj.dtype}")
        print(f"    radio_map    : {rm.shape}    dtype={rm.dtype}")
        print(f"    dataset size : {len(loader.dataset)}")
        print()

    print("All loaders created and one batch fetched successfully.")
