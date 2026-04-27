import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def rasterize_tx_controls(
    coords: torch.Tensor,
    height: int,
    width: int,
    sigma_px: float = 1.5,
) -> torch.Tensor:
    """Build transmitter-centric control maps.

    Args:
        coords: [B, T, 2] normalized (x, y) coordinates in [0, 1].
        height, width: output map size.
        sigma_px: Gaussian std for the transmitter heatmap, in pixels.

    Returns:
        control_maps: [B, T, 2, H, W]
            channel 0 -> Gaussian transmitter heatmap
            channel 1 -> normalized log-distance map
    """
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape [B, T, 2].")

    device, dtype = coords.device, coords.dtype
    B, T, _ = coords.shape

    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)

    tx_x = coords[..., 0].view(B, T, 1, 1)
    tx_y = coords[..., 1].view(B, T, 1, 1)

    dist = torch.sqrt((xx - tx_x) ** 2 + (yy - tx_y) ** 2 + 1e-8)
    sigma = sigma_px / max(height, width)
    heat = torch.exp(-0.5 * (dist / sigma) ** 2)
    logdist = torch.log1p(32.0 * dist) / math.log1p(32.0 * math.sqrt(2.0))

    return torch.stack([heat, logdist], dim=2)


class EdgeMagnitude(nn.Module):
    """Fixed Sobel edge magnitude from the building-height map."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 4.0
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.res = ResBlock2D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.pool(x))


class SceneEncoder(nn.Module):
    """Encode the static layout once.

    Input:
        layout: [B, 1, H, W]

    Output:
        [s0, s1, s2, s3, s4]
        s0: [B, C0, H,    W]
        s1: [B, C1, H/2,  W/2]
        s2: [B, C2, H/4,  W/4]
        s3: [B, C3, H/8,  W/8]
        s4: [B, C4, H/16, W/16]
    """

    def __init__(self, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        c0, c1, c2, c3, c4 = widths
        self.edge = EdgeMagnitude()
        self.stem = ResBlock2D(2, c0)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.down4 = DownBlock(c3, c4)

    def forward(self, layout: torch.Tensor) -> List[torch.Tensor]:
        edge = self.edge(layout)
        x = torch.cat([layout, edge], dim=1)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        return [s0, s1, s2, s3, s4]


class ControlPyramid(nn.Module):
    """Encode per-frame transmitter control maps.

    Input:
        control_maps: [B*T, 2, H, W]

    Output:
        [c0, c1, c2, c3, c4] with the same scales as SceneEncoder.
    """

    def __init__(self, widths=(8, 16, 24, 32, 48)):
        super().__init__()
        c0, c1, c2, c3, c4 = widths
        self.l0 = ResBlock2D(2, c0)
        self.l1 = DownBlock(c0, c1)
        self.l2 = DownBlock(c1, c2)
        self.l3 = DownBlock(c2, c3)
        self.l4 = DownBlock(c3, c4)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c0 = self.l0(x)
        c1 = self.l1(c0)
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        return [c0, c1, c2, c3, c4]


class FiLM2D(nn.Module):
    """Feature-wise affine modulation.

    cond: [N, cond_dim] -> gamma, beta in R^{C}
    """

    def __init__(self, cond_dim: int, feat_ch: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * feat_ch),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class UpFuseBlock(nn.Module):
    """Upsample -> FiLM -> concat scene skip + control skip -> residual fuse."""

    def __init__(
        self,
        in_ch: int,
        scene_skip_ch: int,
        ctrl_skip_ch: int,
        out_ch: int,
        cond_dim: int,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
        )
        self.film = FiLM2D(cond_dim, out_ch, hidden=film_hidden)
        self.fuse = ResBlock2D(out_ch + scene_skip_ch + ctrl_skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        scene_skip: torch.Tensor,
        ctrl_skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        x = self.film(x, cond)
        x = torch.cat([x, scene_skip, ctrl_skip], dim=1)
        return self.fuse(x)


class SymmetricTemporalMix(nn.Module):
    """Exactly time-reversal-equivariant temporal mixing."""

    def __init__(self, channels: int, kernel_t: int = 5):
        super().__init__()
        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_t, 1, 1),
            padding=(kernel_t // 2, 0, 0),
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def _op(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self._op(x)
        y_rev = torch.flip(self._op(torch.flip(x, dims=[2])), dims=[2])
        return 0.5 * (y_fwd + y_rev)


class SeparableTemporalBlock(nn.Module):
    """Cheap low-resolution spatio-temporal refinement.

    Input / output: [B, C, T, H, W]
    """

    def __init__(self, channels: int, kernel_t: int = 5):
        super().__init__()
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.spatial_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.temporal = SymmetricTemporalMix(channels, kernel_t=kernel_t)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial_pw(self.spatial_dw(x))
        y = self.temporal(y)
        y = self.norm(y)
        return self.act(x + y)


class TemporalRefiner(nn.Module):
    """Refine [B*T, C, H, W] latent sequences using non-causal symmetric blocks."""

    def __init__(self, channels: int, depth: int = 2, kernel_t: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SeparableTemporalBlock(channels, kernel_t=kernel_t) for _ in range(depth)]
        )

    def forward(self, x_btchw: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
        bt, c, h, w = x_btchw.shape
        x = x_btchw.reshape(batch_size, steps, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 1, 3, 4).contiguous().reshape(bt, c, h, w)


class LearnableRadialPrior(nn.Module):
    """Learnable affine transform of the log-distance prior."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, logdist: torch.Tensor) -> torch.Tensor:
        return self.scale * logdist + self.bias


class DynamicRadioMapNet(nn.Module):
    """Shared-scene, transmitter-conditioned, temporally refined radio-map network.

    Inputs:
        layout: [B, 1, H, W]     static building-height map
        coords: [B, T, 2]        normalized waypoint coordinates

    Output:
        maps:   [B, T, 1, H, W]  predicted radio maps
    """

    def __init__(
        self,
        scene_widths=(32, 64, 128, 256, 256),
        control_widths=(8, 16, 24, 32, 48),
        film_hidden: int = 128,
        temporal_depth: int = 2,
        temporal_kernel: int = 5,
        tx_sigma_px: float = 1.5,
    ):
        super().__init__()
        self.tx_sigma_px = tx_sigma_px

        self.scene_encoder = SceneEncoder(scene_widths)
        self.control_encoder = ControlPyramid(control_widths)

        cond_dim = sum(control_widths)
        s0, s1, s2, s3, s4 = scene_widths
        c0, c1, c2, c3, c4 = control_widths

        self.bottleneck = ResBlock2D(s4 + c4, s4)
        self.up3 = UpFuseBlock(s4, s3, c3, s3, cond_dim, film_hidden=film_hidden)
        self.up2 = UpFuseBlock(s3, s2, c2, s2, cond_dim, film_hidden=film_hidden)
        self.temporal = TemporalRefiner(s2, depth=temporal_depth, kernel_t=temporal_kernel)
        self.up1 = UpFuseBlock(s2, s1, c1, s1, cond_dim, film_hidden=film_hidden)
        self.up0 = UpFuseBlock(s1, s0, c0, s0, cond_dim, film_hidden=film_hidden)

        self.residual_head = nn.Sequential(
            ResBlock2D(s0, s0),
            nn.Conv2d(s0, 1, kernel_size=1),
        )
        self.radial_prior = LearnableRadialPrior()

    def forward(self, layout: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if layout.ndim != 4 or layout.shape[1] != 1:
            raise ValueError("layout must have shape [B, 1, H, W].")
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError("coords must have shape [B, T, 2].")

        B, _, H, W = layout.shape
        T = coords.shape[1]

        # 1) Static scene features, encoded once.
        scene_feats = self.scene_encoder(layout)

        # 2) Per-frame transmitter control maps.
        controls = rasterize_tx_controls(coords, H, W, sigma_px=self.tx_sigma_px)  # [B, T, 2, H, W]
        controls_bt = controls.reshape(B * T, 2, H, W)

        # 3) Per-frame control pyramid + global control descriptor.
        ctrl_feats = self.control_encoder(controls_bt)
        ctrl_desc = torch.cat(
            [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in ctrl_feats],
            dim=1,
        )  # [B*T, sum(control_widths)]

        # Broadcast the static scene pyramid across time.
        scene_bt = [
            feat[:, None].expand(B, T, *feat.shape[1:]).reshape(B * T, *feat.shape[1:])
            for feat in scene_feats
        ]

        # 4) Decode each frame down to 1/4 resolution.
        x = self.bottleneck(torch.cat([scene_bt[-1], ctrl_feats[-1]], dim=1))
        x = self.up3(x, scene_bt[-2], ctrl_feats[-2], ctrl_desc)
        x = self.up2(x, scene_bt[-3], ctrl_feats[-3], ctrl_desc)

        # 5) Non-causal, reversal-equivariant temporal refinement in latent space.
        x = self.temporal(x, batch_size=B, steps=T)

        # 6) Finish decoding to full resolution.
        x = self.up1(x, scene_bt[-4], ctrl_feats[-4], ctrl_desc)
        x = self.up0(x, scene_bt[-5], ctrl_feats[-5], ctrl_desc)

        residual = self.residual_head(x)          # [B*T, 1, H, W]
        logdist = controls_bt[:, 1:2]             # [B*T, 1, H, W]
        prior = self.radial_prior(logdist)        # [B*T, 1, H, W]

        out = residual + prior
        return out.reshape(B, T, 1, H, W)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = DynamicRadioMapNet(
        scene_widths=(32, 64, 128, 256, 256),
        control_widths=(8, 16, 24, 32, 48),
        film_hidden=128,
        temporal_depth=2,
        temporal_kernel=5,
        tx_sigma_px=1.5,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DyRadioNet v2 — trainable parameters: {n_params:,}")

    B = 2
    building_map = torch.randn(B, 1, 128, 128, device=device)
    trajectory = torch.randn(B, 64, 2, device=device)

    print("\nRunning forward pass...")
    with torch.no_grad():
        radio_maps = model(building_map, trajectory)

    print(f"Output shape: {radio_maps.shape}")
    assert radio_maps.shape == (B, 64, 1, 128, 128), "Shape mismatch!"
    print("✓ Output shape verified: (B, 64, 1, 128, 128)")

    # --- Speed estimate ---
    import time

    # Warm-up
    with torch.no_grad():
        _ = model(building_map, trajectory)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    n_iters = 5
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(building_map, trajectory)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) / n_iters * 1000
    print(f"\nAvg inference time (B={B}, 64 frames): {avg_ms:.1f} ms")
    print(f"  → {avg_ms / 64:.2f} ms/frame (amortized)")
    print(f"  → A frame-independent baseline would need ~64× the single-frame cost")

    print("\n" + "=" * 65)
    print("WavePropNet verification complete.")
    print("=" * 65)

