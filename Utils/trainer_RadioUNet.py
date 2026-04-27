"""Trainer utilities: epoch loops, LR scheduling, logging, and training curve plots."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
#  1. LOSS COMPONENTS
# ---------------------------------------------------------------------------
def _loss_components(pred, target):
    """Decompose the WavePropLoss into its parts (recon)."""
    l_recon = F.mse_loss(pred, target)

    combined = l_recon

    return l_recon, combined


# ---------------------------------------------------------------------------
#  2. EPOCH LOOPS
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device, grad_clip_norm: float = 1.0):
    """Run one training epoch."""
    model.train()
    tot, rec, n_samples = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  Train", leave=False, ncols=120)
    for inputs, rm in pbar:
        inputs = inputs.to(device)
        rm = rm.to(device)

        optimizer.zero_grad()
        [_, pred] = model(inputs)
        l_rec, loss = _loss_components(pred, rm)
        loss.backward()

        # MODIFIED: clip gradients to reduce late-epoch instability that often
        # makes overfitting look worse once validation has plateaued.
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = rm.size(0)
        n_samples += batch_size
        tot += loss.item() * batch_size
        rec += l_rec.item() * batch_size
        pbar.set_postfix(
            loss=f"{tot / n_samples:.4f}",
            recon=f"{rec / n_samples:.4f}",
        )

    return tot / n_samples, rec / n_samples


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Run one validation epoch."""
    model.eval()
    tot, rec, n_samples = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=120)
    for inputs, rm in pbar:
        inputs = inputs.to(device)
        rm = rm.to(device)
        [_, pred] = model(inputs)
        l_rec, loss = _loss_components(pred, rm)

        batch_size = rm.size(0)
        n_samples += batch_size
        tot += loss.item() * batch_size
        rec += l_rec.item() * batch_size
        pbar.set_postfix(
            loss=f"{tot / n_samples:.4f}",
            recon=f"{rec / n_samples:.4f}",
        )

    return tot / n_samples, rec / n_samples


@torch.no_grad()
def test_epoch(model, loader, device):
    """Run one test epoch with full metrics (MSE, RMSE, MAE)."""
    model.eval()
    tot, rec, n_samples = 0.0, 0.0, 0
    total_mse, total_mae, n_elements = 0.0, 0.0, 0

    for inputs, rm in tqdm(loader, desc="Testing", leave=False, ncols=120):
        inputs = inputs.to(device)
        rm = rm.to(device)
        [_, pred] = model(inputs)
        l_rec, loss = _loss_components(pred, rm)

        batch_size = rm.size(0)
        n_samples += batch_size
        tot += loss.item() * batch_size
        rec += l_rec.item() * batch_size
        total_mse += F.mse_loss(pred, rm, reduction='sum').item()
        total_mae += (pred - rm).abs().sum().item()
        n_elements += pred.numel()

    mse = total_mse / n_elements
    return {
        "loss": tot / n_samples,
        "recon": rec / n_samples,
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": total_mae / n_elements,
    }


# ---------------------------------------------------------------------------
#  4. LR SCHEDULER — linear warmup then ReduceLROnPlateau
# ---------------------------------------------------------------------------
class WarmupThenReduceOnPlateau:
    """Linear LR warmup for the first `warmup_epochs`, then ReduceLROnPlateau."""

    def __init__(self, optimizer, warmup_epochs: int, base_lr: float, **plateau_kwargs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_kwargs)
        self._epoch = 0
        if warmup_epochs > 0:
            self._set_lr(base_lr / warmup_epochs)

    def step(self, val_loss: float):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            self._set_lr(self.base_lr * self._epoch / self.warmup_epochs)
        else:
            self.plateau.step(val_loss)

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> dict:
        return {'epoch': self._epoch, 'plateau': self.plateau.state_dict()}

    def load_state_dict(self, d: dict):
        self._epoch = d['epoch']
        self.plateau.load_state_dict(d['plateau'])

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


# ---------------------------------------------------------------------------
#  5. EARLY STOPPING
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Track validation loss with a minimum required improvement."""

    # MODIFIED: min_delta avoids running for many epochs on tiny noisy changes
    # after validation has effectively plateaued.
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float):
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        should_stop = self.counter >= self.patience
        return improved, should_stop

    def state_dict(self) -> dict:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "counter": self.counter,
        }

    def load_state_dict(self, state: dict):
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.best = state["best"]
        self.counter = state["counter"]


# ---------------------------------------------------------------------------
#  6. LOGGING
# ---------------------------------------------------------------------------
_LOG_FIELDS = (
    "epoch",
    "train_total", "train_recon",
    "val_total", "val_recon",
    "lr", "elapsed_s",
)


def init_log(log_path: Path):
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(_LOG_FIELDS)


def append_log(log_path: Path, row: dict):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([row[k] for k in _LOG_FIELDS])


def read_log(log_path: Path) -> dict:
    data = {k: [] for k in _LOG_FIELDS}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            # Skip rows that are missing any expected field (e.g. stale log
            # from a previous run with a different schema, or a truncated line).
            if any(row.get(k) is None or row.get(k) == "" for k in _LOG_FIELDS):
                continue
            for k in _LOG_FIELDS:
                data[k].append(float(row[k]))
    return data


# ---------------------------------------------------------------------------
#  7. PLOTTING
# ---------------------------------------------------------------------------
def plot_training_curves(log_path: Path, out_dir: Path):
    """Read the CSV log and save training_curves.png to out_dir."""
    if not log_path.exists():
        return
    d = read_log(log_path)
    if len(d["epoch"]) < 2:
        return  # too few points to plot meaningfully

    epochs = d["epoch"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("WavePropNet — Training Curves", fontsize=13, y=0.99)

    # --- Panel 1: Total loss ---
    ax = axes[0]
    ax.plot(epochs, d["train_total"], color="steelblue", label="Train total")
    ax.plot(epochs, d["val_total"], color="tomato", label="Val total", linestyle="--")
    ax.set_ylabel("Total Loss")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Loss components ---
    ax = axes[1]
    ax.plot(epochs, d["train_recon"], color="steelblue", label="Train recon")
    ax.plot(epochs, d["val_recon"], color="tomato", label="Val recon", linestyle="--")
    ax.set_ylabel("Loss Component")
    ax.legend(framealpha=0.8, ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Learning rate ---
    ax = axes[2]
    ax.plot(epochs, d["lr"], color="seagreen")
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
