#!/usr/bin/env python3
"""
Post-generation 4× upscale using Real-ESRGAN style model.
Uses the 4x_NMKD-Siax_200k.pth model you already have.
Processes frame-by-frame to stay within VRAM budget.
"""

import os
import torch
import numpy as np
from pathlib import Path


def load_upscale_model(model_path: str, device: torch.device):
    """
    Load a Real-ESRGAN / NMKD upscale model.
    Falls back to simple bicubic if the model format is unsupported.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=256,           # tile to save VRAM
            tile_pad=10,
            half=True,          # fp16 inference
            device=device,
        )
        return upsampler
    except ImportError:
        print("  ⚠  basicsr/realesrgan not installed — falling back to bicubic 4×")
        return None


def upscale_frame(upsampler, frame_np: np.ndarray) -> np.ndarray:
    """Upscale a single HWC uint8 frame."""
    if upsampler is None:
        # Bicubic fallback
        from PIL import Image
        img = Image.fromarray(frame_np)
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.BICUBIC)
        return np.array(img)

    output, _ = upsampler.enhance(frame_np, outscale=4)
    return output


def upscale_video(input_path: str, output_path: str, cfg: dict):
    """Upscale every frame of a video file and re-encode."""
    import imageio.v3 as iio

    model_path = cfg.get("upscale_model", "models/upscale_models/4x_NMKD-Siax_200k.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  🔍 Upscaling with {Path(model_path).name}")
    upsampler = load_upscale_model(model_path, device)

    # Read all frames
    frames = iio.imread(input_path, plugin="pyav")
    fps = cfg.get("fps", 16)

    upscaled = []
    for i, frame in enumerate(frames):
        up = upscale_frame(upsampler, frame)
        upscaled.append(up)
        if (i + 1) % 10 == 0:
            print(f"    Frame {i+1}/{len(frames)}")

    # Write
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    iio.imwrite(output_path, np.stack(upscaled), fps=fps, codec="libx264")
    print(f"  ✅ Upscaled video: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python upscale.py input.mp4 output.mp4")
        sys.exit(1)
    upscale_video(sys.argv[1], sys.argv[2], {})
