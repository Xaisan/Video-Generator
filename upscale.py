#!/usr/bin/env python3
"""
upscale.py — Post-generation video upscale + frame interpolation
================================================================

Two-phase post-processing pipeline (professional order):
  Phase 1: Spatial 4x upscale — via spandrel (ESRGAN/DAT/SwinIR/etc.)
  Phase 2: RIFE frame interpolation — at high resolution for best quality

Upscale-first-then-interpolate rationale:
  - RIFE gets 4x more spatial detail to analyse motion
  - Only real frames get upscaled (fewer tiles to process)
  - RIFE is lightweight (21 MB) — running it at 4x resolution is fast

Target duration support:
  - target_duration > source -> slow-motion (more interpolated frames)
  - target_duration < source -> speed-up (fewer frames)
  - target_duration == 0     -> preserve original duration (standard fps conversion)

Dependencies (project-internal):
  -> rife_model.py  (load_rife_model, interpolate_frame)

Public API:
  upscale_video()      — main entry: read video -> upscale -> interpolate -> write
  load_upscale_model() — load spatial model via spandrel
  tile_upscale()       — tiled inference on BCHW tensor
  upscale_frame()      — upscale single HWC uint8 frame

Called from: pipeline_engine.py -> PipelineEngine.step_upscale()
             generate.py -> generate() (if enable_upscale=True)

CLI usage: python upscale.py input.mp4 output.mp4 [output_fps] [target_duration]
"""

import gc
import os
import torch
import numpy as np
from pathlib import Path


# ─── Spatial upscale (spandrel) ─────────────────────────────────

def load_upscale_model(model_path: str, device: torch.device):
    """
    Load an upscale model via spandrel (supports ESRGAN, SwinIR, DAT, etc).
    Returns (model, scale_factor, arch_name) or (None, 4, "none") on failure.
    """
    try:
        from spandrel import ModelLoader
        model = ModelLoader().load_from_file(model_path)
        model = model.to(device).eval()
        if hasattr(model, 'scale'):
            scale = model.scale
        else:
            scale = 4

        # spandrel wraps the actual model in model.model
        if hasattr(model, 'model'):
            arch_name = type(model.model).__name__
            num_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        else:
            # Fallback if no .model attribute (e.g. if it's a bare module)
            arch_name = type(model).__name__
            num_params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  Loaded upscale model: {Path(model_path).name}")
        print(f"    Architecture: {arch_name}, Scale: {scale}×, Params: {num_params:.1f}M")
        return model, scale, arch_name
    except Exception as e:
        print(f"  ⚠  Failed to load upscale model: {e}")
        print("  ⚠  Falling back to bicubic 4×")
        return None, 4, "none"


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


def upscale_frame(model, frame_np: np.ndarray, device: torch.device,
                  tile_size: int = 384, tile_pad: int = 32) -> np.ndarray:
    """Upscale a single HWC uint8 frame."""
    if model is None:
        from PIL import Image
        img = Image.fromarray(frame_np)
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.BICUBIC)
        return np.array(img)

    img_t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(device=device, dtype=torch.float32) / 255.0

    result = tile_upscale(model, img_t, tile_size=tile_size, tile_pad=tile_pad)

    result = result.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    result = (result * 255).to(torch.uint8).cpu().numpy()
    return result


def _choose_tile_size(arch_name: str) -> tuple:
    """
    Choose tile_size and tile_pad based on model architecture.
    Transformer-based architectures (DAT, SwinIR, HAT, etc.) need smaller
    tiles than CNN-based ones (ESRGAN/RRDB) due to self-attention memory.
    """
    # Transformer / attention-based — use smaller tiles to avoid OOM
    attn_archs = {"DAT", "SwinIR", "Swin2SR", "HAT", "SRFormer", "ATD",
                  "RGT", "DRCT", "GRL", "DITN", "Restormer", "IPT"}
    for a in attn_archs:
        if a.lower() in arch_name.lower():
            return 256, 24
    # CNN-based (ESRGAN, RRDB, SRVGGNet, SPAN, etc.) — can use larger tiles
    return 384, 32


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Main upscale pipeline ──────────────────────────────────────

def upscale_video(input_path: str, output_path: str, cfg: dict,
                  progress_fn=None):
    """
    Full upscale pipeline (professional order):
      1. Read input video frames
      2. Spatial 4× upscale each frame (at source resolution — fewer frames)
      3. RIFE frame interpolation at upscaled resolution (best motion quality)
      4. Write output video

    Config keys used:
      fps             — source generation fps (default 16)
      output_fps      — desired output framerate (default = fps, no interpolation)
      target_duration — desired final duration in seconds (0 = preserve original)
                        > source duration: slow-motion effect
                        < source duration: speed-up effect
      upscale_model   — path to spatial upscale model
      rife_model      — path to RIFE .pth (default: models/upscale_models/rife47.pth)

    progress_fn: optional callback(percent: int, msg: str) for UI updates
    """
    import imageio.v3 as iio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Read frames ──
    frames_raw = iio.imread(input_path, plugin="pyav")
    frames = [frames_raw[i] for i in range(len(frames_raw))]
    del frames_raw

    source_fps = int(cfg.get("fps", 16))
    output_fps = int(cfg.get("output_fps", source_fps))
    target_duration = float(cfg.get("target_duration", 0))

    n_src = len(frames)
    source_duration = (n_src - 1) / source_fps if n_src > 1 else 0
    print(f"  📹 Source: {n_src} frames @ {source_fps}fps = {source_duration:.2f}s")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: Spatial 4× upscale (on original frames — fewer tiles)
    # ══════════════════════════════════════════════════════════════
    model_path = cfg.get("upscale_model", "models/upscale_models/4xRealWebPhoto_v4_dat2.pth")
    print(f"  🔍 Phase 1: Spatial upscale with {Path(model_path).name}")
    if progress_fn:
        progress_fn(2, "Loading upscale model…")
    model, scale, arch_name = load_upscale_model(model_path, device)
    tile_size, tile_pad = _choose_tile_size(arch_name)
    print(f"    Tile size: {tile_size}px, pad: {tile_pad}px (arch: {arch_name})")
    print(f"    Upscaling {n_src} frames (source resolution)")

    upscaled = []
    total = n_src
    for i, frame in enumerate(frames):
        up = upscale_frame(model, frame, device,
                           tile_size=tile_size, tile_pad=tile_pad)
        upscaled.append(up)
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"    Spatial upscale: frame {i+1}/{total}")
        if progress_fn:
            pct = 2 + int(58 * (i + 1) / total)
            progress_fn(pct, f"Spatial {scale}×: {i+1}/{total} frames")

    # Free spatial model before RIFE
    del model, frames
    _flush()

    src_h, src_w = upscaled[0].shape[:2] if upscaled else (0, 0)
    print(f"    ✓ Upscale done: {src_w}×{src_h} ({scale}×)")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2: RIFE frame interpolation (at upscaled resolution)
    # ══════════════════════════════════════════════════════════════
    #
    # RIFE runs on the high-res upscaled frames, giving it 4× more
    # spatial detail for motion estimation → much better quality.
    #
    # Target duration logic:
    #   - Compute how many output frames we need
    #   - RIFE time-accurately resamples to produce exactly that many frames
    #   - target_duration == 0 or matches source → standard fps conversion
    #   - target_duration > source → slow motion (more interp frames)
    #   - target_duration < source → speed up (subsample frames)

    need_rife = False
    effective_target_duration = target_duration if target_duration > 0 else source_duration

    if target_duration > 0 and abs(target_duration - source_duration) > 0.05:
        # Duration change requested
        need_rife = True
        speed_factor = source_duration / target_duration
        print(f"  🎞  Phase 2: Duration retiming {source_duration:.2f}s → "
              f"{target_duration:.2f}s @ {output_fps}fps")
        if speed_factor > 1:
            print(f"    Speed up: {speed_factor:.2f}× faster")
        else:
            print(f"    Slow motion: {1/speed_factor:.2f}× slower")
    elif output_fps > source_fps:
        # Standard fps upsampling
        need_rife = True
        print(f"  🎞  Phase 2: Frame interpolation {source_fps}fps → "
              f"{output_fps}fps (duration preserved)")

    if need_rife:
        rife_path = cfg.get("rife_model", "models/upscale_models/rife47.pth")
        if os.path.isfile(rife_path):
            if progress_fn:
                progress_fn(62, "Loading RIFE model…")
            from rife_model import load_rife_model, interpolate_frame

            rife = load_rife_model(rife_path, device)
            if rife is not None:
                # Calculate target frame count
                n_target = round(effective_target_duration * output_fps) + 1

                def rife_progress(cur, total):
                    if cur % 10 == 0 or cur == total:
                        print(f"    RIFE interpolation: {cur}/{total} frames")
                    if progress_fn:
                        pct = 62 + int(30 * cur / total)
                        progress_fn(pct, f"RIFE: {cur}/{total} frames")

                upscaled = _interpolate_to_count(
                    rife, upscaled,
                    n_target=n_target,
                    device=device,
                    progress_fn=rife_progress,
                )
                final_dur = (len(upscaled) - 1) / output_fps if len(upscaled) > 1 else 0
                print(f"    RIFE done: {len(upscaled)} frames @ {output_fps}fps "
                      f"= {final_dur:.2f}s")

                del rife
                _flush()
            else:
                print("  ⚠  RIFE model failed to load, skipping interpolation")
                output_fps = source_fps
        else:
            print(f"  ⚠  RIFE model not found at {rife_path}, skipping")
            output_fps = source_fps
    else:
        if output_fps <= source_fps:
            output_fps = source_fps
        print(f"  ℹ  No interpolation needed ({source_fps}fps, "
              f"duration={source_duration:.2f}s)")

    # ── Write output ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if progress_fn:
        progress_fn(94, "Encoding final video…")

    import imageio
    h, w = upscaled[0].shape[:2]
    writer = imageio.get_writer(
        output_path, fps=output_fps, codec="libx264",
        quality=8, pixelformat="yuv420p",
    )
    for frame in upscaled:
        writer.append_data(frame)
    writer.close()

    # Report
    final_duration = (len(upscaled) - 1) / output_fps if len(upscaled) > 1 else 0
    print(f"  ✅ Upscaled video: {output_path}")
    print(f"     {src_w}×{src_h} ({scale}× spatial)")
    print(f"     {output_fps}fps, {len(upscaled)} frames")
    print(f"     Duration: {source_duration:.2f}s → {final_duration:.2f}s")
    if target_duration > 0 and abs(target_duration - source_duration) > 0.05:
        speed = source_duration / final_duration if final_duration > 0 else 1
        print(f"     Speed: {speed:.2f}× ({'faster' if speed > 1 else 'slower'})")


def _interpolate_to_count(model, frames, n_target, device, progress_fn=None):
    """
    Interpolate/resample a frame sequence to produce exactly n_target frames.

    Generalised time-accurate resampling:
      - Speed-up (n_target < n_source): subsamples + interpolates precisely
      - Slow-motion (n_target > n_source): creates many interpolated frames
      - Standard fps conversion: works via time-position mapping

    Each output frame is placed at its exact time position, and we
    interpolate between the two nearest source frames using RIFE.
    """
    from rife_model import interpolate_frame

    n_src = len(frames)
    if n_src < 2 or n_target < 2:
        return frames

    result = []
    total_interp = 0

    for i in range(n_target):
        # Map output frame index to source frame position
        src_pos = i * (n_src - 1) / (n_target - 1)
        src_lo = int(src_pos)
        src_hi = min(src_lo + 1, n_src - 1)
        blend = src_pos - src_lo

        if blend < 0.001 or src_lo == src_hi:
            result.append(frames[min(src_lo, n_src - 1)])
        elif blend > 0.999:
            result.append(frames[src_hi])
        else:
            mid = interpolate_frame(model, frames[src_lo], frames[src_hi],
                                    blend, device)
            result.append(mid)
            total_interp += 1

        if progress_fn and ((i + 1) % 10 == 0 or (i + 1) == n_target):
            progress_fn(i + 1, n_target)

    print(f"    Resampled {n_src} → {len(result)} frames "
          f"({total_interp} interpolated, {len(result) - total_interp} original)")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python upscale.py input.mp4 output.mp4 [output_fps] [target_duration]")
        sys.exit(1)
    out_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    tgt_dur = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    upscale_video(sys.argv[1], sys.argv[2],
                  {"output_fps": out_fps, "target_duration": tgt_dur})
