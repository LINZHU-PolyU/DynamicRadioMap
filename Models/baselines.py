"""
Baseline models for dynamic radio map construction.

Each baseline is a self-contained nn.Module with the same interface:
    Input:  layout [B, 1, H, W],  coords [B, T, 2]
    Output: maps   [B, T, 1, H, W]

Baselines
---------
1. ConvLSTMRadioMap    — Causal ConvLSTM sequential prediction
2. Full3DUNet          — Naïve 3D U-Net treating the task as video-to-video
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Baseline 1: ConvLSTMRadioNet
# -----------------------------------------------------------------------------
def coords_to_onehot(
    coords: torch.Tensor,
    height: int = 128,
    width: int = 128,
) -> torch.Tensor:
    """Convert normalized (x, y) coordinates to one-hot 2-D tensors.

    Args:
        coords: [B, T, 2] with values in [0, 1].
                 coords[..., 0] is x (horizontal / column index),
                 coords[..., 1] is y (vertical   / row    index).
        height, width: spatial resolution of the output grid.

    Returns:
        onehot: [B, T, 1, H, W]  with exactly one 1.0 per (b, t) slice.
    """
    B, T, _ = coords.shape
    device, dtype = coords.device, coords.dtype

    # Map [0, 1] → integer pixel indices, clamped to valid range.
    col = (coords[..., 0] * (width  - 1)).round().long().clamp(0, width  - 1)  # [B, T]
    row = (coords[..., 1] * (height - 1)).round().long().clamp(0, height - 1)  # [B, T]

    onehot = torch.zeros(B, T, 1, height, width, device=device, dtype=dtype)
    # Advanced indexing: set the transmitter pixel to 1.
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)
    t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
    onehot[b_idx, t_idx, 0, row, col] = 1.0

    return onehot


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell operating on 2-D feature maps."""

    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(
            input_ch + hidden_ch,
            4 * hidden_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def init_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_ch, height, width, device=device, dtype=dtype)
        c = torch.zeros_like(h)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            h_prev, c_prev = self.init_state(b, h, w, x.device, x.dtype)
        else:
            h_prev, c_prev = state

        gates = self.gates(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class EncBlock(nn.Module):
    """Down-sample with strided conv → BN → ReLU → conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True) -> None:
        super().__init__()
        layers = []
        if downsample:
            # Strided conv for spatial down-sampling (factor 2).
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False))
        layers += [
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    """Up-sample with ConvTranspose2d, concatenate skip, then conv → BN → ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvLSTMRadioMap(nn.Module):
    """Autoregressive ConvLSTM for dynamic radio-map construction."""

    def __init__(
        self,
        widths: Tuple[int, int, int, int] = (48, 96, 192, 192),
        hidden_ch: int = 256,
        convlstm_kernel: int = 3,
        use_prev_frame: bool = False,
    ) -> None:
        super().__init__()
        self.use_prev_frame = use_prev_frame

        c0, c1, c2, c3 = widths
        # layout (1) + one-hot Tx (1) + optional prev frame (1)
        in_ch = 1 + 1 + (1 if use_prev_frame else 0)

        # --- Encoder ---
        self.enc0 = EncBlock(in_ch, c0, downsample=False)  # H    -> H
        self.enc1 = EncBlock(c0, c1, downsample=True)      # H    -> H/2
        self.enc2 = EncBlock(c1, c2, downsample=True)      # H/2  -> H/4
        self.enc3 = EncBlock(c2, c3, downsample=True)      # H/4  -> H/8

        # --- Recurrent bottleneck ---
        self.recurrent = ConvLSTMCell(c3, hidden_ch, kernel_size=convlstm_kernel)
        self.post_recurrent = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        self.dec2 = DecBlock(hidden_ch, c2, c2)  # H/8 -> H/4
        self.dec1 = DecBlock(c2, c1, c1)          # H/4 -> H/2
        self.dec0 = DecBlock(c1, c0, c0)          # H/2 -> H

        # ---- Output head (linear, no activation — target is standardised) ----
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(
        self,
        layout: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = layout.shape
        T = coords.shape[1]

        # --- Build one-hot Tx maps: [B, T, 1, H, W] ---
        tx_onehot = coords_to_onehot(coords, H, W)

        # --- Autoregressive loop ---
        prev_frame = torch.zeros(B, 1, H, W, device=layout.device, dtype=layout.dtype)
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        outputs = []

        for t in range(T):
            # Assemble per-step input.
            parts = [layout, tx_onehot[:, t]]          # each [B, 1, H, W]
            if self.use_prev_frame:
                parts.append(prev_frame)
            x_t = torch.cat(parts, dim=1)              # [B, in_ch, H, W]

            # Encode.
            s0 = self.enc0(x_t)   # [B, c0, H,   W  ]
            s1 = self.enc1(s0)    # [B, c1, H/2, W/2]
            s2 = self.enc2(s1)    # [B, c2, H/4, W/4]
            s3 = self.enc3(s2)    # [B, c3, H/8, W/8]

            # Recurrent update.
            h_t, c_t = self.recurrent(s3, state)
            state = (h_t, c_t)
            z_t = self.post_recurrent(h_t)

            # Decode.
            y = self.dec2(z_t, s2)
            y = self.dec1(y, s1)
            y = self.dec0(y, s0)
            y = self.head(y)      # [B, 1, H, W]
            outputs.append(y)

            # Autoregressive feedback.
            if self.use_prev_frame:
                prev_frame = y

        return torch.stack(outputs, dim=1)  # [B, T, 1, H, W]


# -----------------------------------------------------------------------------
# Baseline 2: Full3DUNet
# -----------------------------------------------------------------------------
class EncBlock3D(nn.Module):
    """Two-layer conv block with optional spatial-only downsampling."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True) -> None:
        super().__init__()
        layers = []
        if downsample:
            # Strided conv: keep time, halve H and W.
            layers.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3),
                          stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
            )
        else:
            layers.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3),
                          stride=1, padding=1, bias=False)
            )
        layers += [
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock3D(nn.Module):
    """ConvTranspose3d upsample (spatial only) + skip concat + two-layer conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # Upsample H, W by 2; keep T unchanged.
        self.up = nn.ConvTranspose3d(
            in_ch, out_ch,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=False,
        )
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=(3, 3, 3),
                      padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3),
                      padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Full3DUNet(nn.Module):
    """3-D U-Net for spatiotemporal radio-map prediction."""

    def __init__(
            self,
            widths: Tuple[int, ...] = (32, 64, 128, 192, 192),
    ) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = widths
        in_ch = 2  # layout + one-hot Tx

        # --- Encoder ---
        self.stem = EncBlock3D(in_ch, c0, downsample=False)  # T, H,   W
        self.d1 = EncBlock3D(c0, c1, downsample=True)  # T, H/2, W/2
        self.d2 = EncBlock3D(c1, c2, downsample=True)  # T, H/4, W/4
        self.d3 = EncBlock3D(c2, c3, downsample=True)  # T, H/8, W/8
        self.d4 = EncBlock3D(c3, c4, downsample=True)  # T, H/16, W/16

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv3d(c4, c4, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(c4),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        self.u3 = DecBlock3D(c4, c3, c3)  # H/16 -> H/8
        self.u2 = DecBlock3D(c3, c2, c2)  # H/8  -> H/4
        self.u1 = DecBlock3D(c2, c1, c1)  # H/4  -> H/2
        self.u0 = DecBlock3D(c1, c0, c0)  # H/2  -> H

        # --- Output head (linear — target is standardised) ---
        self.head = nn.Conv3d(c0, 1, kernel_size=1)

    def forward(self, layout: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, _, H, W = layout.shape
        T = coords.shape[1]

        # --- Build one-hot Tx maps: [B, T, 1, H, W] ---
        tx_onehot = coords_to_onehot(coords, H, W)

        # --- Tile layout across time: [B, T, 1, H, W] ---
        layout_exp = layout[:, None].expand(B, T, 1, H, W)

        # --- Assemble input: [B, T, 2, H, W] -> [B, 2, T, H, W] ---
        x = torch.cat([layout_exp, tx_onehot], dim=2)  # [B, T, 2, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, 2, T, H, W]

        # --- 3-D Encoder ---
        s0 = self.stem(x)  # [B, c0, T, H,    W   ]
        s1 = self.d1(s0)  # [B, c1, T, H/2,  W/2 ]
        s2 = self.d2(s1)  # [B, c2, T, H/4,  W/4 ]
        s3 = self.d3(s2)  # [B, c3, T, H/8,  W/8 ]
        s4 = self.d4(s3)  # [B, c4, T, H/16, W/16]

        # --- Bottleneck ---
        x = self.bottleneck(s4)

        # --- 3-D Decoder ---
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)

        out = self.head(x)  # [B, 1, T, H, W]
        return out.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, 1, H, W]


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import sys

    # Copy HybridRadioMapNet to current dir for import
    sys.path.insert(0, "/mnt/user-data/uploads")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, T, H, W = 2, 64, 128, 128
    layout = torch.randn(B, 1, H, W, device=device)
    coords = torch.rand(B, T, 2, device=device)

    models = {
        "ConvLSTMRadioMap": ConvLSTMRadioMap(),
        "Full3DUNet": Full3DUNet(),
    }

    for name, model in models.items():
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"{name} — {n_params:,} parameters")
        print(f"{'='*60}")

        try:
            with torch.no_grad():
                out = model(layout, coords)
            print(f"  Output shape: {out.shape}")
            assert out.shape == (B, T, 1, H, W), f"Shape mismatch: {out.shape}"
            print(f"  ✓ Shape verified")

            # Timing
            with torch.no_grad():
                _ = model(layout, coords)
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            n_iters = 5
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(layout, coords)
            if device.type == "cuda":
                torch.cuda.synchronize()
            avg_ms = (time.time() - t0) / n_iters * 1000
            print(f"  Avg inference: {avg_ms:.1f} ms  ({avg_ms/T:.2f} ms/frame)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*60}")
    print("All baselines verified.")
    print(f"{'='*60}")
