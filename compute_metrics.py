"""Parallel numerical metrics for the DynamicRadioMap test set."""

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

from Models.Proposed import DynamicRadioMapNet
from Models.baselines import ConvLSTMRadioMap, Full3DUNet
from Models.RadioUNet import RadioWNet

warnings.filterwarnings("ignore")


MODEL_NAMES = ("Proposed", "ConvLSTM", "FullUNet-3D", "RadioUNet")
METRIC_NAMES = ("RMSE", "SSIM", "CIoU@-100", "CIoU@-95", "CIoU@-90", "DM-RMSE")

DEFAULT_DATASET_STATS = {
    "building_height_max": 40.0,
    "area_size": 128.0,
    "threshold": -110.0,
    "global_rm_max": -83.9375,
    "global_rm_mean": 0.46969,
    "global_rm_std": 0.37251,
}


class TestIndexDataset(Dataset):
    """Dataset backed by the same Utils/test_idx.npy list used by compute_metrics.py."""

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

    def __getitem__(self, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_idx, subtile_idx, traj_idx = (int(v) for v in self.idx_list[sample_idx])

        building_file = self.dataset_dir / "buildings" / f"{tile_idx}_{subtile_idx}.npy"
        radio_file = self.dataset_dir / "raw_radio_maps" / f"{tile_idx}_{subtile_idx}_{traj_idx}.npy"

        building_map = np.load(building_file).astype(np.float32)  # (128, 128)
        trajectory = self.trajs_array[tile_idx, subtile_idx, traj_idx].astype(np.float32).copy()
        radio_map = np.load(radio_file).astype(np.float32)  # (64, 128, 128)

        tx_loc_map = make_tx_location_map(radio_map)

        building_map = building_map / self.dataset_stats["building_height_max"]
        trajectory = trajectory / self.dataset_stats["area_size"]
        radio_map = normalize_radio_map(radio_map, self.dataset_stats)

        building_map_t = torch.from_numpy(building_map).unsqueeze(0).float()
        trajectory_t = torch.from_numpy(trajectory).float()
        tx_loc_map_t = torch.from_numpy(tx_loc_map).float()
        radio_map_t = torch.from_numpy(radio_map).unsqueeze(1).float()

        return building_map_t, trajectory_t, tx_loc_map_t, radio_map_t


def make_tx_location_map(radio_map: np.ndarray) -> np.ndarray:
    """Create one-hot transmitter maps from each frame's peak radio value."""
    num_frames, height, width = radio_map.shape
    flat_peak_idx = np.argmax(radio_map.reshape(num_frames, -1), axis=1)
    rows = flat_peak_idx // width
    cols = flat_peak_idx % width

    tx_loc_map = np.zeros_like(radio_map, dtype=np.float32)
    tx_loc_map[np.arange(num_frames), rows, cols] = 1.0
    return tx_loc_map


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


def denormalize_radiounet_output(x: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
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


def build_models(args: argparse.Namespace, device: torch.device) \
        -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    model_proposed = DynamicRadioMapNet(
        scene_widths=(32, 64, 128, 256, 256),
        control_widths=(8, 16, 24, 32, 48),
        film_hidden=128,
        temporal_depth=2,
        temporal_kernel=5,
        tx_sigma_px=1.5,
    ).to(device)
    model_lstm = ConvLSTMRadioMap(
        widths=(48, 96, 192, 192),
        hidden_ch=256,
        convlstm_kernel=3,
        use_prev_frame=False,
    ).to(device)
    model_unet3d = Full3DUNet(
        widths=(32, 64, 128, 192, 192),
    ).to(device)
    model_radiounet = RadioWNet(inputs=2, phase="secondU").to(device)

    load_state_dict(model_proposed, args.model_proposed_path, device)
    load_state_dict(model_lstm, args.model_lstm_path, device)
    load_state_dict(model_unet3d, args.model_unet_3d_path, device)
    load_state_dict(model_radiounet, args.model_radiounet_path, device)

    model_proposed.eval()
    model_lstm.eval()
    model_unet3d.eval()
    model_radiounet.eval()
    return model_proposed, model_lstm, model_unet3d, model_radiounet


def run_radiounet_batched(
    model: torch.nn.Module,
    building_map: torch.Tensor,
    tx_loc_map: torch.Tensor,
    frame_batch_size: int,
) -> torch.Tensor:
    """Run RadioUNet on all B*T frames, chunked only to avoid GPU OOM."""
    batch_size, num_frames, height, width = tx_loc_map.shape
    building_frames = (
        building_map.unsqueeze(1)
        .expand(-1, num_frames, -1, -1, -1)
        .reshape(batch_size * num_frames, 1, height, width)
    )
    tx_frames = tx_loc_map.unsqueeze(2).reshape(batch_size * num_frames, 1, height, width)
    inputs = torch.cat([building_frames, tx_frames], dim=1)

    if frame_batch_size <= 0:
        frame_batch_size = inputs.shape[0]

    outputs: List[torch.Tensor] = []
    for start in range(0, inputs.shape[0], frame_batch_size):
        end = min(start + frame_batch_size, inputs.shape[0])
        _, pred = model(inputs[start:end])
        outputs.append(pred)

    pred_all = torch.cat(outputs, dim=0)
    return pred_all.reshape(batch_size, num_frames, 1, height, width)


def compute_ciou_batch(
    target: np.ndarray,
    predictions: np.ndarray,
    gamma: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return per-sample, per-model CIoU with shape (B, M)."""
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
            [
                ssim(target_norm[t], prediction_norm[t], data_range=1.0)
                for t in range(target.shape[0])
            ]
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
    """Compute per-sample metrics for a batch.

    Args:
        target: Ground truth array, shape (B, T, H, W).
        model_predictions: One array per model, each shape (B, T, H, W).

    Returns:
        Dict mapping metric name to array of shape (B, M).
    """
    predictions = np.stack(model_predictions, axis=1)  # (B, M, T, H, W)
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


def print_average_table(results: Dict[str, np.ndarray]) -> None:
    header = f"{'Metric':<14}" + "".join(f"{name:>18}" for name in MODEL_NAMES)
    print("\nAveraged metrics across test samples")
    print(header)
    print("-" * len(header))
    for metric_name in METRIC_NAMES:
        means = results[metric_name].mean(axis=0)
        row = f"{metric_name:<14}" + "".join(f"{value:>18.6f}" for value in means)
        print(row)


def print_mean_std_table(results: Dict[str, np.ndarray]) -> None:
    num_samples = next(iter(results.values())).shape[0]
    ddof = 1 if num_samples > 1 else 0

    header = f"{'Metric':<14}" + "".join(f"{name:>28}" for name in MODEL_NAMES)
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
    parser = argparse.ArgumentParser(description="Compute DynamicRadioMap metrics in parallel.")

    parser.add_argument("--dataset_dir", type=str, default="/home/user/datasets/DynamicRadioMap")
    parser.add_argument("--idx_file", type=str, default="Utils/test_idx.npy")
    parser.add_argument("--dataset_stats_json", type=str, default=None)

    parser.add_argument("--model_proposed_path", type=str, default="Checkpoints/Proposed/best.pth")
    parser.add_argument("--model_lstm_path", type=str, default="Checkpoints/ConvLSTM/best.pth")
    parser.add_argument("--model_unet_3d_path", type=str, default="Checkpoints/Full3DUNet/best.pth")
    parser.add_argument("--model_radiounet_path", type=str, default="Checkpoints/RadioUNet/best.pth")

    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--metric_workers", type=int, default=0,
                        help="Threads for per-sample SSIM. 0 disables SSIM threading.")
    parser.add_argument("--radiounet_frame_batch_size", type=int, default=64,
                        help="Number of RadioUNet frames evaluated per forward pass. <=0 uses all B*T frames.")
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

    model_proposed, model_lstm, model_unet3d, model_radiounet = build_models(args, device)

    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
    print(f"Test samples: {len(dataset)}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Data workers: {args.num_workers}")
    print(f"SSIM workers: {args.metric_workers}")
    print()

    metric_chunks: Dict[str, List[np.ndarray]] = {metric_name: [] for metric_name in METRIC_NAMES}
    executor = ThreadPoolExecutor(max_workers=args.metric_workers) if args.metric_workers > 0 else None

    try:
        progress = tqdm(loader, desc="Computing metrics", ncols=120)
        with torch.inference_mode():
            for building_map, trajectory, tx_loc_map, radio_map in progress:
                building_map = building_map.to(device, non_blocking=True)
                trajectory = trajectory.to(device, non_blocking=True)
                tx_loc_map = tx_loc_map.to(device, non_blocking=True)
                radio_map = radio_map.to(device, non_blocking=True)

                pred_proposed = model_proposed(building_map, trajectory)
                pred_lstm = model_lstm(building_map, trajectory)
                pred_unet3d = model_unet3d(building_map, trajectory)
                pred_radiounet = run_radiounet_batched(
                    model_radiounet,
                    building_map,
                    tx_loc_map,
                    frame_batch_size=args.radiounet_frame_batch_size,
                )

                target_denorm = denormalize_standardized_radio_map(radio_map, dataset_stats)
                proposed_denorm = denormalize_standardized_radio_map(pred_proposed, dataset_stats)
                lstm_denorm = denormalize_standardized_radio_map(pred_lstm, dataset_stats)
                unet3d_denorm = denormalize_standardized_radio_map(pred_unet3d, dataset_stats)
                radiounet_denorm = denormalize_radiounet_output(pred_radiounet, dataset_stats)

                batch_metrics = compute_batch_metrics(
                    target=tensor_to_metric_array(target_denorm),
                    model_predictions=(
                        tensor_to_metric_array(proposed_denorm),
                        tensor_to_metric_array(lstm_denorm),
                        tensor_to_metric_array(unet3d_denorm),
                        tensor_to_metric_array(radiounet_denorm),
                    ),
                    executor=executor,
                )
                append_metric_chunks(metric_chunks, batch_metrics)

                completed = sum(chunk.shape[0] for chunk in metric_chunks["RMSE"])
                progress.set_postfix(samples=f"{completed}/{len(dataset)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    results = summarize_metrics(metric_chunks)
    print_average_table(results)
    print_mean_std_table(results)


if __name__ == "__main__":
    main()
