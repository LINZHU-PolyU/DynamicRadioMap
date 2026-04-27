"""Parallel numerical metrics for interpolation-based evaluation."""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from Models.Proposed import DynamicRadioMapNet, rasterize_tx_controls

warnings.filterwarnings("ignore")


DEFAULT_DATASET_STATS = {
    "building_height_max": 40.0,
    "area_size": 128.0,
    "threshold": -110.0,
    "global_rm_max": -83.9375,
    "global_rm_mean": 0.46969,
    "global_rm_std": 0.37251,
}

METRIC_NAMES = ("RMSE", "SSIM", "CIoU@-100", "CIoU@-95", "CIoU@-90", "DM-RMSE")


class TestIndexDataset(Dataset):
    """Dataset backed by the same Utils/test_idx.npy list used elsewhere in the repo."""

    def __init__(
        self,
        dataset_dir: str,
        idx_file: str,
        dataset_stats: Dict[str, float],
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.idx_file = Path(idx_file)
        self.dataset_stats = dataset_stats

        if not self.idx_file.is_file():
            raise FileNotFoundError(f"Test index file not found: {self.idx_file}")
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.idx_list = np.load(self.idx_file)
        traj_file = self.dataset_dir / "trajs_array.npy"
        if not traj_file.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
        self.trajs_array = np.load(traj_file)

    def __len__(self) -> int:
        return int(self.idx_list.shape[0])

    def __getitem__(self, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_idx, subtile_idx, traj_idx = (int(v) for v in self.idx_list[sample_idx])

        building_file = self.dataset_dir / "buildings" / f"{tile_idx}_{subtile_idx}.npy"
        radio_file = self.dataset_dir / "raw_radio_maps" / f"{tile_idx}_{subtile_idx}_{traj_idx}.npy"

        building_map = np.load(building_file).astype(np.float32)
        trajectory = self.trajs_array[tile_idx, subtile_idx, traj_idx].astype(np.float32).copy()
        radio_map = np.load(radio_file).astype(np.float32)

        building_map = building_map / self.dataset_stats["building_height_max"]
        trajectory = trajectory / self.dataset_stats["area_size"]
        radio_map = normalize_radio_map(radio_map, self.dataset_stats)

        building_map_t = torch.from_numpy(building_map).unsqueeze(0).float()
        trajectory_t = torch.from_numpy(trajectory).float()
        radio_map_t = torch.from_numpy(radio_map).unsqueeze(1).float()
        return building_map_t, trajectory_t, radio_map_t


def normalize_radio_map(radio_map: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    threshold = stats["threshold"]
    global_radio_map_max = stats["global_rm_max"]
    global_radio_map_mean = stats["global_rm_mean"]
    global_radio_map_std = stats["global_rm_std"]

    radio_map = radio_map.copy()
    radio_map[radio_map <= threshold] = threshold
    radio_map = (radio_map - threshold) / (global_radio_map_max - threshold)
    radio_map = (radio_map - global_radio_map_mean) / global_radio_map_std
    return radio_map.astype(np.float32, copy=False)


def denormalize_standardized_radio_map(x: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    x = x * stats["global_rm_std"] + stats["global_rm_mean"]
    x = torch.clamp(x, 0.0, 1.0)
    return stats["threshold"] + x * (stats["global_rm_max"] - stats["threshold"])


def load_dataset_stats(stats_json: Optional[str]) -> Dict[str, float]:
    if stats_json is None:
        return dict(DEFAULT_DATASET_STATS)
    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    missing = set(DEFAULT_DATASET_STATS) - set(stats)
    if missing:
        raise ValueError(f"Dataset stats JSON is missing keys: {sorted(missing)}")
    return stats


def load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = DynamicRadioMapNet(
        scene_widths=(32, 64, 128, 256, 256),
        control_widths=(8, 16, 24, 32, 48),
        film_hidden=128,
        temporal_depth=2,
        temporal_kernel=5,
        tx_sigma_px=1.5,
    ).to(device)
    load_state_dict(model, args.model_proposed_path, device)
    model.eval()
    return model


def make_sparse_indices(
    total_frames: int,
    num_anchors: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    idx = np.linspace(0, total_frames - 1, num_anchors, dtype=int)
    return torch.from_numpy(idx).to(device=device, dtype=torch.long)


def compute_hybrid_radial_prior(
    model: DynamicRadioMapNet,
    coords: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    controls = rasterize_tx_controls(
        coords,
        height,
        width,
        sigma_px=model.tx_sigma_px,
    )
    batch_size, num_frames, _, _, _ = controls.shape
    controls_bt = controls.reshape(batch_size * num_frames, 2, height, width)
    logdist = controls_bt[:, 1:2]
    prior_bt = model.radial_prior(logdist)
    return prior_bt.reshape(batch_size, num_frames, 1, height, width)


def temporal_linear_interpolate(
    sparse_values: torch.Tensor,
    anchor_idx: torch.Tensor,
    target_frames: int,
    fixed_alpha: Optional[float] = None,
) -> torch.Tensor:
    if sparse_values.ndim != 5:
        raise ValueError("sparse_values must have shape [B, K, C, H, W].")

    _, num_anchors, _, _, _ = sparse_values.shape
    if anchor_idx.ndim != 1 or anchor_idx.numel() != num_anchors:
        raise ValueError("anchor_idx must have shape [K].")
    if anchor_idx[0].item() != 0 or anchor_idx[-1].item() != target_frames - 1:
        raise ValueError("anchor_idx must include endpoints 0 and target_frames - 1.")
    if not torch.all(anchor_idx[1:] > anchor_idx[:-1]):
        raise ValueError("anchor_idx must be strictly increasing.")

    device = sparse_values.device
    dtype = sparse_values.dtype

    anchor_pos = anchor_idx.to(device=device, dtype=dtype)
    target_pos = torch.arange(target_frames, device=device, dtype=dtype)

    right = torch.searchsorted(anchor_pos, target_pos, right=False)
    right = right.clamp(min=0, max=num_anchors - 1)
    left = (right - 1).clamp(min=0, max=num_anchors - 1)

    t_left = anchor_pos[left]
    t_right = anchor_pos[right]
    denom = (t_right - t_left).clamp_min(1e-6)

    if fixed_alpha is None:
        alpha = ((target_pos - t_left) / denom).view(1, target_frames, 1, 1, 1)
        same_anchor = (right == left).view(1, target_frames, 1, 1, 1)
        alpha = torch.where(same_anchor, torch.zeros_like(alpha), alpha)
    else:
        alpha = torch.full((1, target_frames, 1, 1, 1), fixed_alpha, device=device, dtype=dtype)

    left_values = sparse_values[:, left, :, :, :]
    right_values = sparse_values[:, right, :, :, :]
    return (1.0 - alpha) * left_values + alpha * right_values


def predict_linear_interpolation(
    model: DynamicRadioMapNet,
    layout: torch.Tensor,
    coords_full: torch.Tensor,
    anchor_counts: Sequence[int],
) -> List[torch.Tensor]:
    total_frames = coords_full.shape[1]
    outputs: List[torch.Tensor] = []
    for num_anchors in anchor_counts:
        anchor_idx = make_sparse_indices(total_frames, num_anchors, device=coords_full.device)
        coords_sparse = coords_full.index_select(dim=1, index=anchor_idx)
        sparse_pred = model(layout, coords_sparse)
        dense_pred = temporal_linear_interpolate(
            sparse_values=sparse_pred,
            anchor_idx=anchor_idx,
            target_frames=total_frames,
            fixed_alpha=None,
        )
        outputs.append(dense_pred)
    return outputs


def compute_ciou_batch(
    target: np.ndarray,
    predictions: np.ndarray,
    gamma: float,
    eps: float = 1e-6,
) -> np.ndarray:
    target_mask = target >= gamma
    pred_mask = predictions >= gamma

    intersection = np.logical_and(target_mask[:, None], pred_mask).sum(axis=(3, 4))
    union = np.logical_or(target_mask[:, None], pred_mask).sum(axis=(3, 4))
    return (intersection / (union + eps)).mean(axis=2)


def compute_sample_ssim(payload: Tuple[np.ndarray, np.ndarray]) -> float:
    target, prediction = payload
    data_range = float(target.max() - target.min())
    if data_range <= 0.0:
        return 1.0 if np.allclose(target, prediction) else 0.0

    target_norm = (target - target.min()) / data_range
    prediction_norm = (prediction - target.min()) / data_range
    return float(
        np.mean(
            [ssim(target_norm[t], prediction_norm[t], data_range=1.0) for t in range(target.shape[0])]
        )
    )


def compute_ssim_batch(
    target: np.ndarray,
    predictions: np.ndarray,
    executor: Optional[ThreadPoolExecutor],
) -> np.ndarray:
    batch_size, num_models = predictions.shape[:2]
    payloads = [
        (target[sample_idx], predictions[sample_idx, model_idx])
        for sample_idx in range(batch_size)
        for model_idx in range(num_models)
    ]
    if executor is None:
        values = [compute_sample_ssim(payload) for payload in payloads]
    else:
        values = list(executor.map(compute_sample_ssim, payloads))
    return np.asarray(values, dtype=np.float64).reshape(batch_size, num_models)


def compute_batch_metrics(
    target: np.ndarray,
    model_predictions: Sequence[np.ndarray],
    executor: Optional[ThreadPoolExecutor],
) -> Dict[str, np.ndarray]:
    predictions = np.stack(model_predictions, axis=1)
    diff = target[:, None] - predictions

    metrics = {
        "RMSE": np.sqrt(np.mean(diff ** 2, axis=(2, 3, 4))),
        "SSIM": compute_ssim_batch(target, predictions, executor),
        "CIoU@-100": compute_ciou_batch(target, predictions, gamma=-100.0),
        "CIoU@-95": compute_ciou_batch(target, predictions, gamma=-95.0),
        "CIoU@-90": compute_ciou_batch(target, predictions, gamma=-90.0),
    }

    target_delta = np.diff(target, axis=1)
    pred_delta = np.diff(predictions, axis=2)
    metrics["DM-RMSE"] = np.sqrt(np.mean((target_delta[:, None] - pred_delta) ** 2, axis=(2, 3, 4)))
    return metrics


def tensor_to_metric_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()[:, :, 0, :, :]


def append_metric_chunks(
    metric_chunks: Dict[str, List[np.ndarray]],
    batch_metrics: Dict[str, np.ndarray],
) -> None:
    for metric_name in METRIC_NAMES:
        metric_chunks[metric_name].append(batch_metrics[metric_name])


def summarize_metrics(metric_chunks: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {metric_name: np.concatenate(chunks, axis=0) for metric_name, chunks in metric_chunks.items()}


def print_average_table(results: Dict[str, np.ndarray], model_names: Sequence[str]) -> None:
    header = f"{'Metric':<14}" + "".join(f"{name:>18}" for name in model_names)
    print("\nAveraged metrics across test samples")
    print(header)
    print("-" * len(header))
    for metric_name in METRIC_NAMES:
        means = results[metric_name].mean(axis=0)
        row = f"{metric_name:<14}" + "".join(f"{value:>18.6f}" for value in means)
        print(row)


def print_mean_std_table(results: Dict[str, np.ndarray], model_names: Sequence[str]) -> None:
    num_samples = next(iter(results.values())).shape[0]
    ddof = 1 if num_samples > 1 else 0

    header = f"{'Metric':<14}" + "".join(f"{name:>28}" for name in model_names)
    print("\nMean +/- std over test samples")
    print(header)
    print("-" * len(header))
    for metric_name in METRIC_NAMES:
        values = results[metric_name]
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=ddof)
        row = f"{metric_name:<14}" + "".join(
            f"{mean:.6f} +/- {std:.6f}".rjust(28)
            for mean, std in zip(means, stds)
        )
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute interpolation metrics in parallel.")

    parser.add_argument("--dataset_dir", type=str, default="/home/user/datasets/DynamicRadioMap")
    parser.add_argument("--idx_file", type=str, default="Utils/test_idx.npy")
    parser.add_argument("--dataset_stats_json", type=str, default=None)

    parser.add_argument("--model_proposed_path", type=str, default="Checkpoints/Proposed/best.pth")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--metric_workers",
        type=int,
        default=0,
        help="Threads for per-sample SSIM. 0 disables SSIM threading.",
    )
    parser.add_argument(
        "--anchor_counts",
        type=int,
        nargs="+",
        default=[32, 16, 8],
        help="Sparse anchor counts to evaluate. Dense 64-frame inference is always included.",
    )
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_stats = load_dataset_stats(args.dataset_stats_json)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {args.device} to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    dataset = TestIndexDataset(
        dataset_dir=args.dataset_dir,
        idx_file=args.idx_file,
        dataset_stats=dataset_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and not args.no_pin_memory),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args, device)
    model_names = [f"{count} anchors" for count in args.anchor_counts] + ["64 dense"]

    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
    print(f"Test samples: {len(dataset)}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Data workers: {args.num_workers}")
    print(f"SSIM workers: {args.metric_workers}")
    print(f"Variants    : {', '.join(model_names)}")
    print()

    metric_chunks: Dict[str, List[np.ndarray]] = {metric_name: [] for metric_name in METRIC_NAMES}
    executor = ThreadPoolExecutor(max_workers=args.metric_workers) if args.metric_workers > 0 else None

    try:
        progress = tqdm(loader, desc="Computing metrics", ncols=120)
        with torch.inference_mode():
            for building_map, trajectory, radio_map in progress:
                building_map = building_map.to(device, non_blocking=True)
                trajectory = trajectory.to(device, non_blocking=True)
                radio_map = radio_map.to(device, non_blocking=True)

                sparse_predictions = predict_linear_interpolation(
                    model=model,
                    layout=building_map,
                    coords_full=trajectory,
                    anchor_counts=args.anchor_counts,
                )
                dense_prediction = model(building_map, trajectory)

                target_denorm = denormalize_standardized_radio_map(radio_map, dataset_stats)
                prediction_arrays = [
                    tensor_to_metric_array(denormalize_standardized_radio_map(pred, dataset_stats))
                    for pred in sparse_predictions
                ]
                prediction_arrays.append(
                    tensor_to_metric_array(denormalize_standardized_radio_map(dense_prediction, dataset_stats))
                )

                batch_metrics = compute_batch_metrics(
                    target=tensor_to_metric_array(target_denorm),
                    model_predictions=prediction_arrays,
                    executor=executor,
                )
                append_metric_chunks(metric_chunks, batch_metrics)

                completed = sum(chunk.shape[0] for chunk in metric_chunks["RMSE"])
                progress.set_postfix(samples=f"{completed}/{len(dataset)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    results = summarize_metrics(metric_chunks)
    print_average_table(results, model_names)
    print_mean_std_table(results, model_names)


if __name__ == "__main__":
    main()
