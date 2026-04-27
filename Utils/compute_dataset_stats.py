"""Compute and cache global statistics over the DynamicRadioMap dataset."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
RADIO_MAPS_ROOT = "/home/user/datasets/DynamicRadioMap/raw_radio_maps"

NUM_TILES    = 5
NUM_SUBTILES = 64
NUM_TRAJS    = 30


# ---------------------------------------------------------------------------
#  Worker functions
# ---------------------------------------------------------------------------
def _radio_map_file_stats(args):
    """Load one radio-map .npy file and return (min, max, n_valid, sum, sum_sq)."""
    tile_idx, subtile_idx, traj_idx, radio_maps_root, global_max, threshold = args
    path = Path(radio_maps_root) / f"{tile_idx}_{subtile_idx}_{traj_idx}.npy"
    arr = np.load(path).astype(np.float32)
    arr[arr <= threshold] = threshold
    arr = (arr - threshold) / (global_max - threshold)
    valid = arr[~np.isnan(arr)]
    n_valid  = int(valid.size)
    f_sum    = float(valid.sum())
    f_sum_sq = float((valid.astype(np.float64) ** 2).sum())   # float64 for precision
    f_min    = float(valid.min()) if n_valid > 0 else float("inf")
    f_max    = float(valid.max()) if n_valid > 0 else float("-inf")
    return f_min, f_max, n_valid, f_sum, f_sum_sq


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    dataset_stats: dict = {}
    num_workers = os.cpu_count()

    # -----------------------------------------------------------------------
    # Radio-map  –  global min / max / mean / var
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Radio-map statistics")
    print("=" * 60)

    threshold = -110
    global_max = -83.9375
    rm_args = [
        (tile_idx, subtile_idx, traj_idx, RADIO_MAPS_ROOT, global_max, threshold)
        for tile_idx    in range(NUM_TILES)
        for subtile_idx in range(NUM_SUBTILES)
        for traj_idx    in range(NUM_TRAJS)
    ]

    global_rm_min  = float("inf")
    global_rm_max  = float("-inf")
    global_count   = 0
    global_sum     = 0.0
    global_sum_sq  = 0.0          # Σ x²  (float64 accumulator)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_radio_map_file_stats, arg): arg
            for arg in rm_args
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Radio Maps", unit="file"):
            f_min, f_max, n_valid, f_sum, f_sum_sq = future.result()
            if f_min < global_rm_min:
                global_rm_min = f_min
            if f_max > global_rm_max:
                global_rm_max = f_max
            global_count  += n_valid
            global_sum    += f_sum
            global_sum_sq += f_sum_sq

    # E[X] and Std[X] = sqrt(E[X²] − E[X]²)
    global_rm_mean = global_sum    / global_count
    global_rm_std  = float(np.sqrt(global_sum_sq / global_count - global_rm_mean ** 2))

    dataset_stats["global_rm_min"]  = global_rm_min
    dataset_stats["global_rm_max"]  = global_rm_max
    dataset_stats["global_rm_mean"] = global_rm_mean
    dataset_stats["global_rm_std"]  = global_rm_std

    print(f"  global_rm_min  = {global_rm_min}")
    print(f"  global_rm_max  = {global_rm_max}")
    print(f"  global_rm_mean = {global_rm_mean}")
    print(f"  global_rm_std  = {global_rm_std}")
    print(f"  (computed over {global_count:,} valid elements)")

    # -----------------------------------------------------------------------
    # 3.  Summary & save
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("dataset_stats =", dataset_stats)
    print("=" * 60)


if __name__ == "__main__":
    main()

