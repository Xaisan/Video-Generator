# Architecture — Wan 2.2 I2V AMD Video Generator

This document describes the system architecture in detail. It is intended for developers and AI agents who need to understand the codebase before making changes.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│                                                                          │
│  Browser ◄──SSE──► Flask (app.py :7860)                                 │
│  └─ static/app.js        │                                              │
│  └─ templates/index.html  │  REST API (/api/generate, /api/resume, …)   │
│  └─ static/style.css      │                                              │
│                            ▼                                              │
│                    PipelineEngine (pipeline_engine.py)                    │
│                    ├─ step_encode()     → embeddings.pt                  │
│                    ├─ step_denoise()    → latents.pt                     │
│                    ├─ step_vae_decode() → frames.pt + preview.mp4       │
│                    ├─ step_export()     → output.mp4                    │
│                    └─ step_upscale()    → output_upscaled.mp4           │
│                            │                                              │
│                    SessionManager (session_manager.py)                    │
│                    └─ sessions/<id>/session.json + *.pt checkpoints      │
│                                                                          │
│  CLI alternative: generate.py (standalone, no checkpointing)             │
│                                                                          │
│  Post-processing: upscale.py → rife_model.py                            │
│  GPU tuning:      amd_tune.py                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
app.py
 ├── session_manager.py     (SessionManager, StepStatus, STEP_ORDER)
 ├── pipeline_engine.py     (PipelineEngine, vram_stats, flush_vram, load_config)
 └── amd_tune.py            (apply_amd_optimisations, detect_amd_gpu)

pipeline_engine.py
 ├── session_manager.py     (SessionManager, StepStatus, STEP_ORDER)
 ├── upscale.py             (upscale_video)  — imported lazily in step_upscale()
 └── [diffusers, transformers, torch, PIL, yaml, numpy]

generate.py                 — standalone, duplicates some logic from pipeline_engine.py
 ├── upscale.py             (upscale_video)
 └── [diffusers, transformers, torch, PIL, yaml, numpy]

upscale.py
 └── rife_model.py          (load_rife_model, interpolate_frame)

rife_model.py               — pure PyTorch, no project dependencies

amd_tune.py                 — pure PyTorch, no project dependencies

models/server.py            — standalone utility, no project dependencies
```

---

## Data Flow: Full Generation (Web UI)

### 1. User submits generation request

```
Browser → POST /api/generate → app.py
    ├─ Validates input image exists
    ├─ Snaps dimensions to mod-16
    ├─ Creates session via SessionManager
    ├─ Spawns background thread
    └─ Returns session_id immediately
```

### 2. Pipeline execution (background thread)

```
PipelineEngine.run_from_step(session_id, "encode")
    │
    ├─ STEP 1: encode
    │   ├─ Load UMT5-XXL text encoder (strategy: auto/cpu/gpu)
    │   ├─ Tokenize prompt + negative prompt
    │   ├─ Forward pass → prompt_embeds, negative_prompt_embeds
    │   ├─ Free text encoder from memory
    │   └─ Save embeddings.pt checkpoint
    │
    ├─ STEP 2: denoise (the BIG step, ~80% of total time)
    │   ├─ PHASE 1 — VAE encode input image
    │   │   ├─ Load VAE (float32, tiling+slicing enabled)
    │   │   ├─ prepare_latents() → latents tensor + condition tensor
    │   │   └─ Free VAE completely
    │   │
    │   ├─ PHASE 2 — High-noise transformer pass
    │   │   ├─ Load GGUF transformer (Q8_0, ~10 GB)
    │   │   ├─ Load LoRAs (quality only in quality mode; +distill in fast mode)
    │   │   ├─ Apply group offloading (block_level, CPU↔GPU)
    │   │   ├─ Run timesteps >= boundary_timestep through transformer
    │   │   │   └─ Each step: latent_input → transformer → noise_pred → scheduler.step()
    │   │   │       (with CFG: run cond + uncond, blend with guidance_scale)
    │   │   └─ Free high-noise transformer completely
    │   │
    │   ├─ PHASE 3 — Low-noise transformer pass
    │   │   ├─ Load GGUF transformer (Q8_0, ~10 GB)
    │   │   ├─ Load LoRAs for transformer_2
    │   │   ├─ Apply group offloading
    │   │   ├─ Run timesteps < boundary_timestep (same scheduler, state preserved)
    │   │   └─ Free low-noise transformer completely
    │   │
    │   └─ Save latents.pt checkpoint
    │
    ├─ STEP 3: vae_decode
    │   ├─ Load VAE again
    │   ├─ Decode latents → video frames tensor
    │   ├─ Save frames.pt + preview.mp4
    │   └─ Free VAE
    │
    ├─ STEP 4: export
    │   ├─ Load frames.pt
    │   ├─ Encode → output.mp4 (libx264)
    │   └─ Copy to output/video_<session_id>.mp4
    │
    └─ STEP 5: upscale (optional, if enable_upscale=true)
        ├─ Load spatial upscale model (spandrel: ESRGAN/DAT/SwinIR)
        ├─ Tile-upscale each frame 4×
        ├─ Free spatial model
        ├─ Load RIFE v4.7 (21 MB)
        ├─ Time-accurate frame interpolation to target fps/duration
        ├─ Free RIFE
        └─ Encode → output_upscaled.mp4
```

### 3. Progress reporting

```
PipelineEngine._emit() → app.py progress_callback() → broadcast_event()
    → SSE push to all connected browser clients
    → Browser updates step progress bars + VRAM meter
```

---

## Session & Checkpoint System

Each generation creates a session directory:

```
sessions/<timestamp>_<uuid8>/
├── session.json           # Metadata: params, step statuses, timing, checkpoint list
├── input.png              # Copy of input image
├── embeddings.pt          # After encode step (prompt + negative prompt embeds)
├── latents.pt             # After denoise step (denoised latent tensor)
├── frames.pt              # After VAE decode step (uint8 video frames)
├── preview.mp4            # Quick preview (low quality)
├── output.mp4             # Final video
├── output_upscaled.mp4    # Optional upscaled video
├── generation.log         # Human-readable pipeline diary
└── vram.log               # Timestamped VRAM/RAM allocation timeline
```

**Resumability**: Any step can be resumed without re-running earlier ones, as long as the prerequisite checkpoint exists. The `get_resume_step()` method auto-detects where to resume from.

---

## Memory Management Strategy

The entire design revolves around fitting a **14B parameter model** into **24 GB VRAM**:

| Technique | What it does | VRAM savings |
|---|---|---|
| GGUF quantization | Q8_0 transformers: ~10 GB each vs ~28 GB FP16 | ~18 GB |
| Sequential loading | Only ONE transformer in memory at a time | ~10 GB |
| Group offloading | Transformer blocks shuttle CPU↔GPU during forward pass | ~6-8 GB |
| Chunked SDPA | Head-by-head attention (256 MB cap) vs full matrix (160+ GB) | Prevents OOM |
| VAE tiling/slicing | Process VAE in tiles instead of full resolution | ~4-6 GB |
| Component-level lifecycle | Each pipeline step loads/frees its own models | Peak ~14 GB |

### Peak VRAM by Step

| Step | Approximate Peak | What's in VRAM |
|---|---|---|
| encode | 4-12 GB | UMT5-XXL text encoder (depends on auto-split) |
| denoise | 12-14 GB | One transformer (offloaded blocks) + latents + embeds |
| vae_decode | 6-10 GB | VAE (float32) + latents + frames |
| export | 0 GB | CPU only (imageio) |
| upscale | 2-6 GB | Upscale model + one frame tile at a time |

---

## Attention System (Chunked SDPA)

AMD RDNA3 GPUs lack hardware flash-attention. PyTorch's SDPA falls back to the "math" backend which materialises the **full attention matrix** — for 832×480×81 frames that's **~160 GB**. Instant OOM.

The `_chunked_sdpa()` function in `pipeline_engine.py` replaces `F.scaled_dot_product_attention` and:
1. Processes **one head at a time** (not all H heads in parallel)
2. **Chunks the query dimension** so the attention score tensor is never larger than ~256 MB
3. Results are **bit-identical** to normal SDPA

Resolution logic (auto mode):
- At startup, probe GPU for flash/mem-efficient backends
- If either exists → use native SDPA (fast path)
- If only math backend → enable chunked SDPA (safe path)

---

## LoRA System

LoRAs are loaded per-transformer, per-pass:

| Role | When loaded | LoRAs |
|---|---|---|
| `quality` | Always (both modes) | NS, DR3, s_t (sparse/temporal) |
| `distill` | Only in distill mode | LightX2V, t2v_speed |

**Critical constraints**:
- LoRAs CANNOT be fused into GGUF weights (shape mismatch) — they run unfused
- LoRAs MUST be loaded BEFORE group offloading is applied
- Distill LoRAs require `CFG=1.0` + `4 steps` + `flow_shift=5.0`

Scale overrides are stored per-session in `lora_scales` and applied at load time.

---

## Two-Stage Denoising

The pipeline uses **two separate transformers** (high-noise and low-noise), split by `boundary_ratio`:

```
boundary_timestep = boundary_ratio × num_train_timesteps (1000)

timesteps ≥ boundary → high-noise transformer (compositional structure)
timesteps < boundary → low-noise transformer (detail refinement)
```

The official Wan2.2 `model_index.json` ships `boundary_ratio=0.9`, meaning ~90% of steps go to high-noise.

A **single UniPC scheduler** instance is used across both passes to preserve solver state continuity.

---

## Frontend Architecture

Single-page app served by Flask:

- `templates/index.html` — Jinja2 template, two panels: generate form + session viewer
- `static/app.js` — IIFE module, SSE client for real-time progress, session CRUD
- `static/style.css` — CSS custom properties for dark/light theming

The frontend uses **Server-Sent Events** (not WebSocket) for one-way progress streaming. VRAM stats are pushed with every progress update.

---

## Docker Setup

```
rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
    + ffmpeg, libgl1, git
    + pip install -r requirements.txt
    + Source code copied in
    + Models bind-mounted at runtime (/app/models)
```

Key runtime flags: `--device /dev/kfd --device /dev/dri --ipc host --shm-size 16g`

---

## Configuration Reference

All parameters live in `config.yaml`. The file has extensive inline comments. Key groups:

| Section | Parameters |
|---|---|
| **Model paths** | `model_id`, `gguf_transformer_high`, `gguf_transformer_low` |
| **LoRA config** | `loras[]` with path, target, adapter_name, scale, role |
| **Generation** | `prompt`, `negative_prompt`, `width`, `height`, `num_frames`, `fps`, `duration`, `output_fps` |
| **Sampler** | `num_inference_steps`, `guidance_scale`, `guidance_scale_2`, `flow_shift`, `boundary_ratio` |
| **Memory** | `enable_group_offload`, `offload_type`, `num_blocks_per_group`, `vae_tiling`, `vae_slicing`, `force_vae_cpu` |
| **Attention** | `chunked_attention` (auto/on/off) |
| **Text encoder** | `text_encoder_device` (auto/cpu/gpu), `text_encoder_gpu_headroom_gb` |
| **Upscale** | `enable_upscale`, `upscale_model`, `rife_model` |
| **AMD tuning** | `amd.hsa_override_gfx_version`, `amd.pytorch_hip_alloc_conf`, `amd.enable_torch_compile`, `amd.force_fp16` |
| **Distill mode** | `distill_lora_mode` (forces CFG=1.0, 4 steps, flow_shift=5.0) |
