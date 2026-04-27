"""Computational-Efficiency Comparison."""

import time
import gc
import warnings
from typing import Callable, Tuple

import torch
import torch.nn as nn
import numpy as np

from Models.Proposed import DynamicRadioMapNet
from Models.baselines import ConvLSTMRadioMap, Full3DUNet
from Models.RadioUNet import RadioWNet
from compute_interpolation_metrics import predict_linear_interpolation

from thop import profile

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_radiounet_single_input(
    building_map: torch.Tensor,  # (1,1,128,128)
    x_norm: float,
    y_norm: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a single (1,2,128,128) RadioUNet input for one waypoint."""
    mask = (building_map[0, 0] > 0).float()  # (128,128)
    inp = torch.zeros(1, 2, 128, 128, device=device)
    inp[0, 0] = mask
    col = int(min(max(x_norm * 127, 0), 127))
    row = int(min(max(y_norm * 127, 0), 127))
    inp[0, 1, row, col] = 1.0
    return inp


def reset_memory(device: torch.device):
    """Free cached GPU memory and reset peak stats."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def benchmark(
    fn: Callable,
    device: torch.device,
    n_warmup: int = 10,
    n_runs:   int = 20,
) -> Tuple[float, float]:
    """Return (mean_ms, std_ms) over *n_runs* timed iterations."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            fn()
            end_ev.record()
            end_ev.synchronize()  # wait for this specific event to complete
            times.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def fmt_params(n):
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    return f"{n/1e3:.1f} K"

def fmt_flops(f):
    if f is None:
        return "OOM"
    if f >= 1e12:
        return f"{f/1e12:.2f} TFLOPs"
    if f >= 1e9:
        return f"{f/1e9:.2f} GFLOPs"
    return f"{f/1e6:.2f} MFLOPs"


# ---------------------------------------------------------------------
#  Per-configuration evaluation functions
# ---------------------------------------------------------------------
def evaluate_proposed(device):
    """Evaluate Proposed Model: 1 pass -> 64 maps."""
    print("  [1/5] Evaluating Proposed Model ...")
    model = DynamicRadioMapNet(
        scene_widths=(32, 64, 128, 256, 256),
        control_widths=(8, 16, 24, 32, 48),
        film_hidden=128,
        temporal_depth=2,
        temporal_kernel=5,
        tx_sigma_px=1.5,
    ).to(device).eval()
    total, trainable = count_parameters(model)

    num_anchors = 64
    bmap = torch.rand(1, 1, 128, 128, device=device)
    traj = torch.rand(1, num_anchors, 2, device=device)
    traj_full = torch.rand(1, 64, 2, device=device)

    # FLOPs (thop needs fresh tensors)
    flops, _ = profile(model, inputs=(bmap.clone(), traj.clone()), verbose=False)

    # Inference time
    # def fn():
    #     with torch.no_grad():
    #         model(bmap, traj)
    def fn():
        with torch.no_grad():
            dense_pred = predict_linear_interpolation(
                model=model,
                layout=bmap,
                coords_full=traj_full,
                anchor_counts=[num_anchors]
            )
    mean_ms, std_ms = benchmark(fn, device)

    # Peak GPU memory
    reset_memory(device)
    with torch.no_grad():
        _ = model(bmap, traj)
    mem = get_peak_memory_mb(device)

    # Cleanup
    del model, bmap, traj
    reset_memory(device)

    return dict(total=total, trainable=trainable, flops=flops,
                time_mean=mean_ms, time_std=std_ms, mem=mem)


def evaluate_lstm(device):
    """Evaluate ConvLSTM: 1 pass -> 64 maps."""
    print("  [2/5] Evaluating ConvLSTM ...")
    model = ConvLSTMRadioMap(
        widths=(48, 96, 192, 192),
        hidden_ch=256,
        convlstm_kernel=3,
        use_prev_frame=True,
    ).to(device).eval()
    total, trainable = count_parameters(model)

    bmap = torch.rand(1, 1, 128, 128, device=device)
    traj = torch.rand(1, 64, 2, device=device)

    # FLOPs (thop needs fresh tensors)
    flops, _ = profile(model, inputs=(bmap.clone(), traj.clone()), verbose=False)

    # Inference time
    def fn():
        with torch.no_grad():
            model(bmap, traj)
    mean_ms, std_ms = benchmark(fn, device)

    # Peak GPU memory
    reset_memory(device)
    with torch.no_grad():
        _ = model(bmap, traj)
    mem = get_peak_memory_mb(device)

    # Cleanup
    del model, bmap, traj
    reset_memory(device)

    return dict(total=total, trainable=trainable, flops=flops,
                time_mean=mean_ms, time_std=std_ms, mem=mem)


def _trilinear_flops_hook(module, input, output):
    """Custom thop hook for nn.Upsample with mode='trilinear'.

    thop does not implement trilinear upsampling natively and silently returns
    zero FLOPs for it.  Trilinear interpolation reads 8 neighbouring voxels per
    output element, so we count output_elements * 8 multiply-add operations.
    """
    module.__flops__ = getattr(module, "__flops__", 0) + output.numel() * 8


def evaluate_full3dunet(device):
    """Evaluate Full3DUNet: 1 pass -> 64 maps."""
    print("  [3/5] Evaluating Full3DUNet ...")
    model = Full3DUNet(
        widths=(32, 64, 128, 192, 192),
    ).to(device).eval()
    total, trainable = count_parameters(model)

    bmap = torch.rand(1, 1, 128, 128, device=device)
    traj = torch.rand(1, 64, 2, device=device)

    # Register a custom thop hook for every trilinear Upsample layer so that
    # thop does not silently count those ops as zero FLOPs.
    custom_ops = {nn.Upsample: _trilinear_flops_hook}

    # FLOPs (thop needs fresh tensors)
    flops, _ = profile(model, inputs=(bmap.clone(), traj.clone()),
                       custom_ops=custom_ops, verbose=False)

    # Inference time
    def fn():
        with torch.no_grad():
            model(bmap, traj)
    mean_ms, std_ms = benchmark(fn, device)

    # Peak GPU memory
    reset_memory(device)
    with torch.no_grad():
        _ = model(bmap, traj)
    mem = get_peak_memory_mb(device)

    # Cleanup
    del model, bmap, traj
    reset_memory(device)

    return dict(total=total, trainable=trainable, flops=flops,
                time_mean=mean_ms, time_std=std_ms, mem=mem)


def evaluate_radiounet_sequential(device):
    """Evaluate RadioUNet sequential: 64 x (B=1) passes."""
    print("  [4/5] Evaluating RadioUNet sequential ...")
    model = RadioWNet(inputs=2, phase="secondU").to(device).eval()
    total, trainable = count_parameters(model)

    bmap = torch.rand(1, 1, 128, 128, device=device)
    traj = torch.rand(1, 64, 2, device=device)

    # FLOPs for a single (B=1) call, then multiply by 64
    single_inp = build_radiounet_single_input(bmap, 0.5, 0.5, device)
    flops_one, _ = profile(model, inputs=(single_inp.clone(),), verbose=False)
    flops_total = flops_one * 64

    # Pre-build all 64 single inputs
    inputs_list = []
    for t in range(64):
        x_n = traj[0, t, 0].item()
        y_n = traj[0, t, 1].item()
        inputs_list.append(build_radiounet_single_input(bmap, x_n, y_n, device))

    # Inference time
    def fn():
        with torch.no_grad():
            for inp in inputs_list:
                model(inp)
    mean_ms, std_ms = benchmark(fn, device)

    # Peak GPU memory
    reset_memory(device)
    with torch.no_grad():
        for inp in inputs_list:
            _ = model(inp)
    mem = get_peak_memory_mb(device)

    del model, bmap, traj, inputs_list
    reset_memory(device)

    return dict(total=total, trainable=trainable, flops=flops_total,
                time_mean=mean_ms, time_std=std_ms, mem=mem)


def evaluate_radiounet_batched(device):
    """ Evaluate RadioUNet batched: 1 pass with B=64."""
    print("  [5/5] Evaluating RadioUNet batched (B=64) ...")
    model = RadioWNet(inputs=2, phase="secondU").to(device).eval()
    total, trainable = count_parameters(model)

    bmap = torch.rand(1, 1, 128, 128, device=device)
    traj = torch.rand(1, 64, 2, device=device)

    # Build the (64, 2, 128, 128) batch
    mask = (bmap[0, 0] > 0).float()
    batch_inp = torch.zeros(64, 2, 128, 128, device=device)
    for t in range(64):
        batch_inp[t, 0] = mask
        c = int(min(max(traj[0, t, 0].item() * 127, 0), 127))
        r = int(min(max(traj[0, t, 1].item() * 127, 0), 127))
        batch_inp[t, 1, r, c] = 1.0

    import psutil
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    if device.type == "cpu" and avail_gb < 6.0:
        print(f"    WARNING: only {avail_gb:.1f} GB RAM available; "
              "skipping batched run (would OOM).")
        # FLOPs can still be estimated: 64 x single-sample
        single_inp = batch_inp[0:1].clone()
        flops_one, _ = profile(model, inputs=(single_inp,), verbose=False)
        flops = flops_one * 64
        del model, bmap, traj, batch_inp, single_inp
        reset_memory(device)
        return dict(total=total, trainable=trainable, flops=flops,
                    time_mean=None, time_std=None,
                    mem=None, note="OOM_estimated")

    try:
        # FLOPs
        flops, _ = profile(model, inputs=(batch_inp.clone(),), verbose=False)

        # Inference time
        def fn():
            with torch.no_grad():
                model(batch_inp)
        mean_ms, std_ms = benchmark(fn, device)

        # Peak GPU memory
        reset_memory(device)
        with torch.no_grad():
            _ = model(batch_inp)
        mem = get_peak_memory_mb(device)

    except (RuntimeError, MemoryError) as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower() \
                or isinstance(e, MemoryError):
            print("    WARNING: OOM -- RadioUNet batched (B=64) exceeds available memory.")
            del model, bmap, traj, batch_inp
            reset_memory(device)
            return dict(total=total, trainable=trainable, flops=None,
                        time_mean=None, time_std=None, mem=None)
        raise

    del model, bmap, traj, batch_inp
    reset_memory(device)

    return dict(total=total, trainable=trainable, flops=flops,
                time_mean=mean_ms, time_std=std_ms, mem=mem)


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
    print()

    # --- Run evaluations one at a time ---
    proposed = evaluate_proposed(device)
    conv_lstm = evaluate_lstm(device)
    full_3dunet = evaluate_full3dunet(device)
    radio_unet_seq = evaluate_radiounet_sequential(device)
    radio_unet_batch = evaluate_radiounet_batched(device)

    # Assign short aliases used in the table below
    drmn   = proposed        # DynamicRadioMapNet (proposed)
    lstm = conv_lstm       # ARConvLSTMRadioMap
    unet3d = full_3dunet   # Full3DUNet
    rbt  = radio_unet_batch
    rsq  = radio_unet_seq

    # --- Print the comparison table ---
    W = 24  # column width

    def col(val, fallback="OOM"):
        s = str(val) if val is not None else fallback
        return s.ljust(W)

    sep  = "-" * (30 + 5 * W)
    sep2 = "=" * (30 + 5 * W)

    print()
    print(sep2)
    print("  COMPUTATIONAL EFFICIENCY COMPARISON")
    print("  Task: produce 64-frame dynamic radio map (B=1)")
    print(sep2)
    print()

    hdr = (f"{'Metric':<30}{col('DynamicRadioMapNet')}{col('ConvLSTM')}"
           f"{col('Full3DUNet')}{col('RadioUNet (batched)')}{col('RadioUNet (seq.)')}")
    print(hdr)
    print(sep)

    # -- Parameters --
    print(f"{'Total params':<30}"
          f"{col(fmt_params(drmn['total']))}"
          f"{col(fmt_params(lstm['total']))}"
          f"{col(fmt_params(unet3d['total']))}"
          f"{col(fmt_params(rbt['total']))}"
          f"{col(fmt_params(rsq['total']))}")
    print(f"{'Trainable params':<30}"
          f"{col(fmt_params(drmn['trainable']))}"
          f"{col(fmt_params(lstm['trainable']))}"
          f"{col(fmt_params(unet3d['trainable']))}"
          f"{col(fmt_params(rbt['trainable']))}"
          f"{col(fmt_params(rsq['trainable']))}")
    print()

    # -- FLOPs --
    print(f"{'FLOPs (64 maps)':<30}"
          f"{col(fmt_flops(drmn['flops']))}"
          f"{col(fmt_flops(lstm['flops']))}"
          f"{col(fmt_flops(unet3d['flops']))}"
          f"{col(fmt_flops(rbt['flops']))}"
          f"{col(fmt_flops(rsq['flops']))}")
    print()

    # -- Inference time --
    def time_str(r):
        if r['time_mean'] is None:
            return "OOM"
        return f"{r['time_mean']:.2f} +/- {r['time_std']:.2f}"

    def per_frame(r):
        if r['time_mean'] is None:
            return "OOM"
        return f"{r['time_mean']/64:.3f}"

    print(f"{'Inference time (ms)':<30}"
          f"{col(time_str(drmn))}{col(time_str(lstm))}{col(time_str(unet3d))}"
          f"{col(time_str(rbt))}{col(time_str(rsq))}")
    print(f"{'  per frame (ms)':<30}"
          f"{col(per_frame(drmn))}{col(per_frame(lstm))}{col(per_frame(unet3d))}"
          f"{col(per_frame(rbt))}{col(per_frame(rsq))}")
    print()

    # -- Memory --
    def mem_str(r, dev):
        if dev.type != "cuda":
            return "N/A (CPU)"
        if r['mem'] is None:
            return "OOM"
        return f"{r['mem']:.1f}"

    print(f"{'Peak GPU memory (MiB)':<30}"
          f"{col(mem_str(drmn, device))}"
          f"{col(mem_str(lstm, device))}"
          f"{col(mem_str(unet3d, device))}"
          f"{col(mem_str(rbt, device))}"
          f"{col(mem_str(rsq, device))}")
    print()

    # -- Speed-up ratios --
    print(sep)
    print("  SPEED-UP RATIOS  (relative to RadioUNet sequential)")
    print(sep)
    if rsq['time_mean'] and rsq['time_mean'] > 0:
        t_ref = rsq['time_mean']
        if drmn['time_mean']:
            print(f"  DynamicRadioMapNet  : {t_ref / drmn['time_mean']:.2f}x faster")
        if lstm['time_mean']:
            print(f"  ConvLSTM            : {t_ref / lstm['time_mean']:.2f}x faster")
        if unet3d['time_mean']:
            print(f"  Full3DUNet          : {t_ref / unet3d['time_mean']:.2f}x faster")
        if rbt['time_mean']:
            print(f"  RadioUNet (batched) : {t_ref / rbt['time_mean']:.2f}x faster")
    print()

    # -- FLOPs ratios --
    print(sep)
    print("  FLOPs RATIOS  (relative to RadioUNet sequential)")
    print(sep)
    if rsq['flops'] and rsq['flops'] > 0:
        f_ref = rsq['flops']
        for name, r in [("DynamicRadioMapNet", drmn), ("ConvLSTM", lstm), ("Full3DUNet", unet3d)]:
            if r['flops']:
                ratio = f_ref / r['flops']
                if ratio >= 1:
                    print(f"  {name:<20}: {ratio:.2f}x fewer FLOPs")
                else:
                    print(f"  {name:<20}: {1/ratio:.2f}x more FLOPs")
        if rbt['flops']:
            print(f"  {'RadioUNet (batched)':<20}: {f_ref / rbt['flops']:.2f}x "
                  "(same compute, better parallelism)")
    print()


if __name__ == "__main__":
    main()