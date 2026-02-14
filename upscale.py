#!/usr/bin/env python3
"""
Post-generation upscale pipeline:
  1. RIFE frame interpolation — increase framerate (e.g. 16fps → 24fps)
  2. Spatial 4× upscale — via spandrel (ESRGAN/NMKD model)

Processes frame-by-frame with tiling to stay within VRAM budget.
RIFE model is loaded/freed separately from the spatial upscale model.
"""

import gc
import os
import torch
import numpy as np
from pathlib import Path


# ─── Spatial upscale (spandrel) ─────────────────────────────────

def load_upscale_model(model_path: str, device: torch.device):
    """
    Load an upscale model via spandrel (supports ESRGAN, SwinIR, etc).
    Returns (model, scale_factor) or (None, 4) on failure.
    """
    try:
        from spandrel import ModelLoader
        model = ModelLoader().load_from_file(model_path)
        model = model.to(device).eval()
        if hasattr(model, 'scale'):
            scale = model.scale
        else:
            scale = 4
        print(f"  Loaded upscale model: {Path(model_path).name} (scale={scale})")
        return model, scale
    except Exception as e:
        print(f"  ⚠  Failed to load upscale model: {e}")
        print("  ⚠  Falling back to bicubic 4×")
        return None, 4


def tile_upscale(model, img_tensor: torch.Tensor, tile_size: int = 256,
                 tile_pad: int = 16) -> torch.Tensor:
    """Upscale a BCHW tensor using tiled inference to save VRAM."""
    b, c, h, w = img_tensor.shape
    scale = model.scale if hasattr(model, 'scale') else 4

    out_h, out_w = h * scale, w * scale
    output = torch.zeros(b, c, out_h, out_w, device=img_tensor.device,
                         dtype=img_tensor.dtype)

    tiles_y = max(1, (h + tile_size - 1) // tile_size)
    tiles_x = max(1, (w + tile_size - 1) // tile_size)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            py0 = max(0, y0 - tile_pad)
            px0 = max(0, x0 - tile_pad)
            py1 = min(h, y1 + tile_pad)
            px1 = min(w, x1 + tile_pad)

            tile_in = img_tensor[:, :, py0:py1, px0:px1]

            with torch.no_grad():
                tile_out = model(tile_in)

            crop_y0 = (y0 - py0) * scale
            crop_x0 = (x0 - px0) * scale
            crop_y1 = crop_y0 + (y1 - y0) * scale
            crop_x1 = crop_x0 + (x1 - x0) * scale

            out_y0 = y0 * scale
            out_x0 = x0 * scale
            out_y1 = y1 * scale
            out_x1 = x1 * scale

            output[:, :, out_y0:out_y1, out_x0:out_x1] = \
                tile_out[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

    return output


def upscale_frame(model, frame_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Upscale a single HWC uint8 frame."""
    if model is None:
        from PIL import Image
        img = Image.fromarray(frame_np)
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.BICUBIC)
        return np.array(img)

    img_t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(device=device, dtype=torch.float32) / 255.0

    result = tile_upscale(model, img_t, tile_size=192, tile_pad=16)

    result = result.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    result = (result * 255).to(torch.uint8).cpu().numpy()
    return result


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Main upscale pipeline ──────────────────────────────────────

def upscale_video(input_path: str, output_path: str, cfg: dict):
    """
    Full upscale pipeline:
      1. Read input video frames
      2. (Optional) RIFE frame interpolation to increase framerate
      3. Spatial 4× upscale each frame
      4. Write output video at the target fps

    Config keys used:
      fps           — source generation fps (default 16)
      output_fps    — desired output framerate (default = fps, no interpolation)
      upscale_model — path to spatial upscale model
      rife_model    — path to RIFE .pth (default: models/upscale_models/rife47.pth)
    """
    import imageio.v3 as iio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Read frames ──
    frames_raw = iio.imread(input_path, plugin="pyav")
    frames = [frames_raw[i] for i in range(len(frames_raw))]
    del frames_raw

    source_fps = int(cfg.get("fps", 16))
    output_fps = int(cfg.get("output_fps", source_fps))

    # ── Phase 1: RIFE frame interpolation ──
    if output_fps > source_fps:
        rife_path = cfg.get("rife_model", "models/upscale_models/rife47.pth")
        if os.path.isfile(rife_path):
            print(f"  🎞  Frame interpolation: {source_fps}fps → {output_fps}fps using RIFE")
            from rife_model import load_rife_model, interpolate_sequence

            rife = load_rife_model(rife_path, device)
            if rife is not None:
                def rife_progress(cur, total):
                    if cur % 10 == 0 or cur == total:
                        print(f"    RIFE interpolation: {cur}/{total} pairs")

                frames = interpolate_sequence(
                    rife, frames,
                    target_fps=output_fps,
                    source_fps=source_fps,
                    device=device,
                    progress_fn=rife_progress,
                )
                print(f"    RIFE done: {len(frames)} frames @ {output_fps}fps")

                # Free RIFE model before spatial upscale
                del rife
                _flush()
            else:
                print("  ⚠  RIFE model failed to load, skipping frame interpolation")
                output_fps = source_fps
        else:
            print(f"  ⚠  RIFE model not found at {rife_path}, skipping frame interpolation")
            output_fps = source_fps
    else:
        if output_fps < source_fps:
            print(f"  ℹ  Output FPS ({output_fps}) ≤ source ({source_fps}), no interpolation")
        output_fps = source_fps  # Don't reduce fps

    # ── Phase 2: Spatial 4× upscale ──
    model_path = cfg.get("upscale_model", "models/upscale_models/4x_NMKD-Siax_200k.pth")
    print(f"  🔍 Spatial upscale with {Path(model_path).name}")
    model, scale = load_upscale_model(model_path, device)

    upscaled = []
    total = len(frames)
    for i, frame in enumerate(frames):
        up = upscale_frame(model, frame, device)
        upscaled.append(up)
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"    Spatial upscale: frame {i+1}/{total}")

    # Free spatial model
    del model
    _flush()

    # ── Write output ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    iio.imwrite(output_path, np.stack(upscaled), fps=output_fps, codec="libx264")
    print(f"  ✅ Upscaled video: {output_path} "
          f"({scale}× spatial, {output_fps}fps, {len(upscaled)} frames)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python upscale.py input.mp4 output.mp4 [output_fps]")
        sys.exit(1)
    out_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    upscale_video(sys.argv[1], sys.argv[2], {"output_fps": out_fps})
