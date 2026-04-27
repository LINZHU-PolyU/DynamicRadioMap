"""Dataset and DataLoader for RadioUNet static radio map prediction."""

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
#  Dataset
# ---------------------------------------------------------------------------
class DynamicRadioMapDataset(Dataset):
    """PyTorch Dataset for the Dynamic Radio Map prediction task."""

    def __init__(
        self,
        buildings: List[Tuple[int, int]],
        data_root: str,
        num_trajs: int = 30,
        num_waypoints: int = 64,
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
        self.num_waypoints = num_waypoints
        self.dataset_stats = dataset_stats

        # Build flat index: list of (tile_idx, subtile_idx, traj_idx, waypoint_idx)
        self.samples: List[Tuple[int, int, int, int]] = []
        for tile_idx, subtile_idx in buildings:
            for traj_idx in range(num_trajs):
                for waypoint_idx in range(num_waypoints):
                    self.samples.append((tile_idx, subtile_idx, traj_idx, waypoint_idx))

        # Load trajectories array once into memory (shared across samples)
        trajs_path = os.path.join(data_root, "trajs_array.npy")
        self.trajs_array: np.ndarray = np.load(trajs_path)

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_idx, subtile_idx, traj_idx, waypoint_idx = self.samples[idx]

        # --- Load building map ---
        bld_path = os.path.join(self.data_root, "buildings", f"{tile_idx}_{subtile_idx}.npy")
        building_map: np.ndarray = np.load(bld_path)

        # --- Load radio map ---
        rm_path = os.path.join(self.data_root, "raw_radio_maps", f"{tile_idx}_{subtile_idx}_{traj_idx}.npy")
        radio_map: np.ndarray = np.load(rm_path)
        radio_map = radio_map[waypoint_idx, :, :]

        # --- Transmitter location (one-hot, peak of raw radio map) ---
        tx_loc_map = np.zeros_like(radio_map, dtype=np.float32)
        max_idx = np.unravel_index(np.argmax(radio_map), radio_map.shape)
        tx_loc_map[max_idx] = 1.0

        # --- Normalize inputs ---
        building_map = building_map / self.dataset_stats['building_height_max']

        # --- Normalize & standardize output ---
        threshold = self.dataset_stats['threshold']
        global_radio_map_max = self.dataset_stats['global_rm_max']
        global_radio_map_mean = self.dataset_stats['global_rm_mean']
        global_radio_map_std = self.dataset_stats['global_rm_std']
        radio_map[radio_map <= threshold] = threshold
        radio_map = (radio_map - threshold) / (global_radio_map_max - threshold)
        radio_map = (radio_map - global_radio_map_mean) / global_radio_map_std

        # --- Convert to tensors ---
        building_map_t = torch.from_numpy(building_map).unsqueeze(0).float()  # (1, 128, 128)
        tx_loc_map_t = torch.from_numpy(tx_loc_map).unsqueeze(0).float()      # (1, 128, 128)
        inputs = torch.cat([building_map_t, tx_loc_map_t], dim=0)      # (2, 128, 128)
        radio_map_t = torch.from_numpy(radio_map).unsqueeze(0).float()        # (1, 128, 128)

        return inputs, radio_map_t


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
    num_waypoints: int = 64,
    pin_memory: bool = True,
    dataset_stats: dict = None,
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
            num_waypoints=num_waypoints,
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
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dataset_stats=dataset_stats,
    )

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        inputs, rm = batch
        print(f"[{split_name}]  batch shapes:")
        print(f"    inputs       : {inputs.shape}   dtype={inputs.dtype}")
        print(f"    building_map : {inputs[:, 0].shape}   dtype={inputs[:, 0].dtype}")
        print(f"    tx_loc_map   : {inputs[:, 1].shape}  dtype={inputs[:, 1].dtype}")
        print(f"    radio_map    : {rm.shape}    dtype={rm.dtype}")
        print(f"    dataset size : {len(loader.dataset)}")
        print()

    print("All loaders created and one batch fetched successfully.")
