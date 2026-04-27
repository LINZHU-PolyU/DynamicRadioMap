"""Training script for RadioUNet static radio map prediction."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from Models.RadioUNet import RadioWNet
from Utils.build_dataloader_RadioUNet import build_dataloaders
from Utils.trainer_RadioUNet import (
    EarlyStopper,
    WarmupThenReduceOnPlateau,
    append_log,
    eval_epoch,
    init_log,
    plot_training_curves,
    train_epoch,
)


# ---------------------------------------------------------------------------
#  Dataset statistics
# ---------------------------------------------------------------------------
DATASET_STATS = {
    'building_height_max': 40.0,
    'area_size': 128.0,
    'threshold': -110.0,
    'global_rm_max': -83.9375,
    'global_rm_mean': 0.0,
    'global_rm_std': 1.0,
}


def seed_everything(seed: int):
    # MODIFIED: make the regularization / early-stop comparisons reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_stats(stats_json: str = None) -> dict:
    # MODIFIED: allow loading train-only normalization stats from JSON instead
    # of hard-coding global numbers in the training script.
    if stats_json is None:
        return DATASET_STATS
    with open(stats_json, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train WavePropNet")

    # ------ Paths ------
    parser.add_argument("--data_root", type=str, default="/home/user/datasets/DynamicRadioMap")
    parser.add_argument("--out_dir", type=str, default="Results/RadioUNet")
    parser.add_argument("--dataset_stats_json", type=str, default=None,
                        help="Optional JSON containing TRAIN-split-only normalization stats")

    # ------ Hardware ------
    parser.add_argument("--device", type=str, default="cuda:0")

    # ------ Training ------
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # ------ Data ------
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # ------ Optimizer ------
    parser.add_argument("--lr", type=float, default=2e-4)         # MODIFIED: slightly lower LR.
    parser.add_argument("--weight_decay", type=float, default=5e-4)  # MODIFIED: stronger weight decay.
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # ------ Scheduler ------
    parser.add_argument("--warmup_epochs", type=int, default=10)  # MODIFIED: shorter warmup.
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # ------ Early stopping ------
    parser.add_argument("--early_stop_patience", type=int, default=20)   # MODIFIED: stop closer to the plateau.
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    # ------ Checkpoint ------
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # --- Set random seed ---
    seed_everything(args.seed)

    # --- Define device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Output directory & log ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    init_log(log_path)

    # --- Data ---
    dataset_stats = load_dataset_stats(args.dataset_stats_json)

    loaders = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,   # MODIFIED: actually honor CLI split args.
        val_ratio=args.val_ratio,       # MODIFIED: actually honor CLI split args.
        test_ratio=args.test_ratio,     # MODIFIED: actually honor CLI split args.
        dataset_stats=dataset_stats,
    )
    print(
        f"Train: {len(loaders['train'].dataset)} samples/epoch  |  "
        f"Val: {len(loaders['val'].dataset)} samples  |  "
        f"Test: {len(loaders['test'].dataset)} samples"
    )

    # --- Model & optimizer ---
    model = RadioWNet(inputs=2, phase="secondU").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupThenReduceOnPlateau(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    early_stopper = EarlyStopper(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WavePropNet — trainable parameters: {n_params:,}")

    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "early_stopper" in ckpt:
            early_stopper.load_state_dict(ckpt["early_stopper"])
        elif "best_val_loss" in ckpt:
            early_stopper.best = ckpt["best_val_loss"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']})")

    print(f"\n{'=' * 70}")
    print("Starting Training: Building + Trajectory -> Dynamic Radio Map")
    print(f"{'=' * 70}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_tot, tr_rec = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )
        vl_tot, vl_rec = eval_epoch(
            model,
            loaders["val"],
            device,
        )
        scheduler.step(vl_tot)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()
        improved, should_stop = early_stopper.step(vl_tot)

        print(
            f"Train total: {tr_tot:.4f} | recon: {tr_rec:.4f}"
        )
        print(
            f"Val   total: {vl_tot:.4f} | recon: {vl_rec:.4f} | lr: {lr_now:.2e}"
        )

        append_log(log_path, dict(
            epoch=epoch,
            train_total=tr_tot,
            train_recon=tr_rec,
            val_total=vl_tot,
            val_recon=vl_rec,
            lr=lr_now,
            elapsed_s=round(elapsed, 1),
        ))

        # MODIFIED: save the updated early-stopping / best-loss state so resume
        # behaves consistently and the best checkpoint records the correct metric.
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopper": early_stopper.state_dict(),
            "best_val_loss": early_stopper.best,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pth")
        if improved:
            torch.save(ckpt, out_dir / "best.pth")
            print(f"         ↳ New best val loss: {early_stopper.best:.4f}")

        plot_training_curves(log_path, out_dir)

        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best val loss: {early_stopper.best:.4f})."
            )
            break

    print(f"\n{'=' * 70}")
    print("Training complete.")
    print(f"{'=' * 70}")
    print(f"Best val loss : {early_stopper.best:.4f}")
    print(f"Checkpoints   : {out_dir / 'best.pth'},  {out_dir / 'last.pth'}")
    print(f"Log           : {log_path}")
    print(f"Figures       : {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
