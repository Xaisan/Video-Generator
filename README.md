# Wan 2.2 I2V — AMD ROCm AI Video Generator

> **Image-to-Video generation** powered by [Wan 2.2 (14B)](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers), optimised for **AMD GPUs via ROCm** (tested on RX 7900 XTX / gfx1100).

## What This App Does

Takes a **single input image** + a **text prompt** and generates a short video (typically 5 seconds at 16 fps → 81 frames) using a two-stage diffusion transformer pipeline. The output can optionally be **4× upscaled** (spatial) and **frame-interpolated** (temporal) via RIFE.

## Quick Start

```bash
# Docker (recommended — handles all ROCm deps)
./run.fish                    # Web UI at http://localhost:7860
./run.fish --cli -i input/photo.jpg -p "camera zooms in"  # CLI mode

# Or with docker-compose
docker compose up --build
```

### Prerequisites

| Requirement | Notes |
|---|---|
| AMD GPU with ROCm | RX 7900 XTX (24 GB) recommended; 12 GB GPUs work with `force_vae_cpu: true` |
| Docker + `/dev/kfd` `/dev/dri` | AMD GPU passthrough |
| Models in `./models/` | GGUF transformers, LoRAs, VAE, upscale models (see below) |
| Input image in `./input/` | Any resolution — auto-resized to VAE-compatible dims |

### Model Files Required

```
models/
├── unet/
│   ├── Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf   # ~10 GB, high-noise transformer
│   └── Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf    # ~10 GB, low-noise transformer
├── loras/                                       # Quality + distillation LoRAs
├── text_encoders/                               # UMT5-XXL (optional local cache)
├── vae/
│   └── wan_2.1_vae.safetensors                 # VAE (float32)
├── upscale_models/
│   ├── 4xRealWebPhoto_v4_dat2.pth             # Spatial upscaler (DAT-2)
│   └── rife47.pth                              # Frame interpolation (RIFE v4.7)
└── vae_approx/
    └── taew2_1.pth                             # Tiled VAE approximator
```

## Architecture Overview

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full technical deep-dive.

### File Map

| File | Lines | Purpose |
|---|---|---|
| `app.py` | ~600 | Flask web server, REST API, SSE broadcasting, log capture |
| `pipeline_engine.py` | ~1960 | Core pipeline: 5-step engine (encode → denoise → VAE decode → export → upscale), VRAM management, chunked SDPA, session logging |
| `session_manager.py` | ~270 | Session CRUD, checkpoint persistence (`.pt` files), step status tracking |
| `generate.py` | ~490 | Standalone CLI generator (builds full pipeline in memory, no checkpointing) |
| `upscale.py` | ~380 | Post-generation: spatial 4× upscale (spandrel) + RIFE temporal interpolation |
| `rife_model.py` | ~290 | RIFE v4.7 neural network architecture (IFNet) + inference helpers |
| `amd_tune.py` | ~120 | AMD GPU detection, ROCm env vars, matmul benchmark |
| `config.yaml` | ~230 | All generation parameters, model paths, memory settings, AMD tuning |
| `models/server.py` | ~110 | Utility: simple HTTP server to download model folders as ZIP |
| `run.fish` | ~90 | Docker launch script (fish shell) |
| `templates/index.html` | ~470 | Web UI template (Jinja2) |
| `static/app.js` | ~1260 | Frontend: SSE client, form handling, session management |
| `static/style.css` | ~1200 | UI styles (dark/light theme) |

### Two Entry Points

1. **Web UI** (`app.py`) — Flask server with real-time SSE progress, session management, resume/checkpoint support. This is the primary interface.
2. **CLI** (`generate.py`) — Standalone script that loads everything into one pipeline and runs end-to-end. No checkpointing, no web UI. Useful for batch/scripting.

### Pipeline Steps (Web UI mode)

```
encode → denoise → vae_decode → export → upscale (optional)
```

Each step loads only the components it needs, then frees them from VRAM. Intermediate results are checkpointed to disk (`sessions/<id>/*.pt`) so any step can be resumed.

### Key Design Decisions

- **GGUF quantized transformers** — 10 GB each instead of 28 GB FP16. Loaded via `from_single_file()`.
- **Sequential transformer loading** — Only ONE transformer in VRAM at a time during denoise (high-noise first, then low-noise).
- **LoRAs stay unfused** — GGUF uint-quantized weights have incompatible shapes for fusion. LoRAs are loaded BEFORE group offloading.
- **Chunked SDPA** — Replaces PyTorch's SDPA on AMD GPUs (no flash/mem-efficient backend) with head-by-head query-chunked attention. Prevents 160+ GB attention matrix OOM.
- **Group offloading** — Moves transformer blocks CPU↔GPU on-the-fly. Peak VRAM ~14 GB for a 14B model.
- **Two generation modes**: Quality (20 steps, CFG=5.0) and Distill/Fast (4 steps, CFG=1.0 via LightX2V).

## Configuration

All settings are in `config.yaml`. See inline comments there for every parameter. Key sections:

- **Model paths** — GGUF transformers, LoRAs, VAE
- **Generation** — prompt, resolution, frames, steps, CFG, seed
- **Memory** — offload type, VAE tiling/slicing, force CPU fallback
- **Attention** — chunked SDPA mode (auto/on/off)
- **AMD tuning** — HSA GFX version, HIP alloc config, torch.compile

## API Reference

See **[API_REFERENCE.md](API_REFERENCE.md)** for the full REST API documentation.

## For AI Agents

See **[AGENTS.md](AGENTS.md)** for a concise context file optimised for AI coding assistants working on this codebase.
