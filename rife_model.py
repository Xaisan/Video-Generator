#!/usr/bin/env python3
"""
rife_model.py — RIFE v4.7 neural network for frame interpolation
================================================================

Pure PyTorch implementation of IFNet (Real-Time Intermediate Flow Estimation).
No project dependencies — this is a leaf module.

Architecture: coarse-to-fine optical flow with v4.7 feature encoding.
Model file: rife47.pth (21 MB)

Adapted from:
  - https://github.com/hzwer/Practical-RIFE
  - https://github.com/HolyWu/vs-rife (IFNet_HDv3_v4_7.py)

Dependencies (project-internal): none
External: torch, numpy

Public API:
  Classes:
    IFNet                  — RIFE v4.7 network (4 IFBlock stages + encoder)
    IFBlock                — single flow estimation block
    ResConv                — residual convolution with learnable beta

  Functions:
    load_rife_model()      — load .pth file -> IFNet on device
    interpolate_frame()    — interpolate single frame between two inputs
    interpolate_sequence() — interpolate full sequence to target fps
    warp()                 — optical flow warping via grid_sample

Used by: upscale.py -> upscale_video()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── Grid cache for flow warping ─────────────────────────────────
_backwarp_grid_cache: dict = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp an image tensor using optical flow (grid_sample based)."""
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in _backwarp_grid_cache:
        tenHor = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVer = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        _backwarp_grid_cache[k] = torch.cat([tenHor, tenVer], 1)

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
    ], 1)

    g = (_backwarp_grid_cache[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput, grid=g, mode="bilinear",
        padding_mode="border", align_corners=True,
    )


# ─── Building blocks ────────────────────────────────────────────

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        # 4*6=24 channels → PixelShuffle(2) → 6 channels at 2× resolution
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear",
                          align_corners=False)
        if flow is not None:
            flow = (F.interpolate(flow, scale_factor=1.0 / scale,
                                  mode="bilinear", align_corners=False)
                    * (1.0 / scale))
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear",
                            align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    """RIFE v4.7 IFNet — coarse-to-fine optical flow with feature encoding."""

    def __init__(self):
        super().__init__()
        # block0: img0(3) + img1(3) + f0(4) + f1(4) + timestep(1) = 15
        self.block0 = IFBlock(7 + 8, c=192)
        # blocks 1-3: warped0(3) + warped1(3) + wf0(4) + wf1(4) + timestep(1) + mask(1) + flow(4) = 20
        self.block1 = IFBlock(8 + 4 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8, c=64)
        # v4.7-specific shallow feature encoder
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 4, 2, 1),
        )

    def forward(self, img0, img1, timestep=0.5, scale_list=None):
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        n, c, h, w = img0.shape
        # Pad to multiple of 64
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        # v4.7 feature encoding
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow = None
        mask = None
        warped_img0 = img0
        warped_img1 = img1
        merged = None
        blocks = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None, scale=scale_list[i],
                )
            else:
                fd, mask = blocks[i](
                    torch.cat((
                        warped_img0[:, :3], warped_img1[:, :3],
                        warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]),
                        timestep, mask,
                    ), 1),
                    flow, scale=scale_list[i],
                )
                flow = flow + fd

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        return merged[:, :, :h, :w]


# ─── Public API ──────────────────────────────────────────────────

def load_rife_model(model_path: str, device: torch.device) -> IFNet | None:
    """Load a RIFE v4.7 model from a .pth file."""
    try:
        model = IFNet()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        # Handle DDP-wrapped keys (module.xxx → xxx)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval().to(device)
        print(f"  Loaded RIFE model: {model_path} → {device}")
        return model
    except Exception as e:
        print(f"  ⚠  Failed to load RIFE model: {e}")
        return None


def interpolate_frame(model: IFNet, img0: np.ndarray, img1: np.ndarray,
                      timestep: float, device: torch.device) -> np.ndarray:
    """
    Interpolate a single frame between img0 and img1.

    Args:
        model:    Loaded IFNet model
        img0:     HWC uint8 numpy array (first frame)
        img1:     HWC uint8 numpy array (second frame)
        timestep: 0.0 = img0, 1.0 = img1, 0.5 = midpoint
        device:   torch device

    Returns:
        HWC uint8 numpy array (interpolated frame)
    """
    t0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        result = model(t0, t1, timestep=timestep)

    result = result.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return (result * 255).to(torch.uint8).cpu().numpy()


def interpolate_sequence(model: IFNet, frames: list[np.ndarray],
                         target_fps: int, source_fps: int,
                         device: torch.device,
                         progress_fn=None) -> list[np.ndarray]:
    """
    Interpolate a sequence of frames to reach a target FPS.

    Uses time-accurate resampling: each output frame is placed at its exact
    time position, and we interpolate between the two nearest source frames.
    This preserves the original video duration exactly.

    Args:
        model:      Loaded IFNet model
        frames:     List of HWC uint8 numpy arrays
        target_fps: Desired output framerate
        source_fps: Original framerate
        device:     torch device
        progress_fn: Optional callback(current, total) for progress

    Returns:
        List of HWC uint8 numpy arrays at the target framerate
    """
    if target_fps <= source_fps:
        return frames  # No interpolation needed

    n_src = len(frames)
    if n_src < 2:
        return frames

    # Compute exact duration and target frame count
    duration = (n_src - 1) / source_fps  # time from first to last frame
    n_target = round(duration * target_fps) + 1  # +1 for the first frame at t=0

    result = []
    total_interp = 0

    for i in range(n_target):
        # Time position of this output frame
        t = i / target_fps
        # Corresponding fractional source frame index
        src_pos = t * source_fps
        src_lo = int(src_pos)
        src_hi = min(src_lo + 1, n_src - 1)
        blend = src_pos - src_lo

        if blend < 0.001 or src_lo == src_hi:
            # Close enough to an original frame — use it directly
            result.append(frames[min(src_lo, n_src - 1)])
        elif blend > 0.999:
            # Close enough to the next frame
            result.append(frames[src_hi])
        else:
            # Interpolate between src_lo and src_hi
            mid = interpolate_frame(model, frames[src_lo], frames[src_hi],
                                    blend, device)
            result.append(mid)
            total_interp += 1

        if progress_fn and ((i + 1) % 10 == 0 or (i + 1) == n_target):
            progress_fn(i + 1, n_target)

    print(f"    Resampled {n_src} frames @ {source_fps}fps → "
          f"{len(result)} frames @ {target_fps}fps "
          f"({total_interp} interpolated, {len(result) - total_interp} original)")
    print(f"    Duration preserved: {duration:.3f}s → "
          f"{(len(result) - 1) / target_fps:.3f}s")

    return result
