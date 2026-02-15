# AGENTS.md — AI Agent Context File

> **Read this first.** This file gives an AI coding agent everything it needs to understand this codebase before making changes. It is optimised for fast comprehension, not human reading.

---

## Project Identity

- **Name**: Wan 2.2 I2V AMD Video Generator
- **Purpose**: Image-to-Video generation using Wan 2.2 14B diffusion transformers, optimised for AMD ROCm GPUs
- **Stack**: Python 3.10+, PyTorch (ROCm), HuggingFace Diffusers, Flask, Docker
- **Hardware target**: AMD Radeon RX 7900 XTX (24 GB VRAM, gfx1100/RDNA3), also supports 12 GB RDNA2
- **Primary interface**: Web UI at port 7860 (Flask + SSE)
- **Secondary interface**: CLI via `generate.py`

---

## File Inventory (what each file does)

### Root-level files

| File | Role | Key exports / entry points |
|---|---|---|
| **`app.py`** | Web server + API | Flask app, routes (`/api/generate`, `/api/resume`, `/api/events`), SSE broadcasting, log capture. Entry point: `python app.py` |
| **`pipeline_engine.py`** | **Backward-compat shim** | Re-exports everything from `pipeline/` package. Existing `from pipeline_engine import X` still works. **Do NOT add new code here.** |
| **`session_manager.py`** | Persistence layer | `SessionManager` class, `SessionInfo` dataclass, `StepStatus` enum, `STEP_ORDER` list |
| **`generate.py`** | Standalone CLI | `build_pipeline()`, `generate()`, `main()`. Loads all models at once, no checkpointing. Shares some duplicated logic with `pipeline/` |
| **`upscale.py`** | Post-processing | `upscale_video()` (main), `load_upscale_model()`, `tile_upscale()`, `upscale_frame()` |
| **`rife_model.py`** | RIFE v4.7 neural net | `IFNet` class, `load_rife_model()`, `interpolate_frame()`, `interpolate_sequence()` |
| **`amd_tune.py`** | AMD GPU helpers | `apply_amd_optimisations()`, `detect_amd_gpu()`, `benchmark_matmul()` |
| **`config.yaml`** | Configuration | All model paths, generation params, memory settings, AMD tuning. Extensively commented |
| **`models/server.py`** | Utility | Simple HTTP server to download model folders as ZIP (not part of main app) |
| **`run.fish`** | Launch script | Docker run with AMD GPU passthrough flags |
| **`templates/index.html`** | Web UI template | Jinja2, two-panel layout (generate form + session viewer) |
| **`static/app.js`** | Frontend JS | SSE event handling, form sync, session list, progress bars |
| **`static/style.css`** | Styles | CSS custom properties, dark/light theme |

### `pipeline/` package (split from old `pipeline_engine.py`)

| File | Lines | Role | Key exports |
|---|---|---|---|
| **`__init__.py`** | ~70 | Re-export hub | All public names from submodules |
| **`vram_utils.py`** | ~50 | VRAM/RAM monitoring | `flush_vram()`, `vram_stats()`, `ram_stats()` |
| **`amd_env.py`** | ~25 | AMD env vars | `apply_amd_env()` |
| **`sdpa.py`** | ~170 | Chunked SDPA (AMD workaround) | `_chunked_sdpa()`, `patched_sdpa()`, `_resolve_chunked_attention()` |
| **`config.py`** | ~35 | Config + image prep | `load_config()`, `prepare_image()`, `snap_to_mod()` |
| **`loaders.py`** | ~160 | Model loaders | `load_text_encoder()`, `load_vae()`, `load_single_transformer()`, `load_scheduler()` |
| **`lora.py`** | ~100 | LoRA adapter loading | `apply_loras_to_transformer()`, `apply_loras()` |
| **`offload.py`** | ~50 | Group offloading | `setup_offloading()`, `remove_offloading()` |
| **`session_logger.py`** | ~250 | Per-session logging | `SessionLogger` class |
| **`engine.py`** | ~750 | Pipeline orchestration | `PipelineEngine` class |

---

## Critical Concepts an Agent Must Know

### 1. Pipeline Steps
The generation pipeline has 5 sequential steps. Each step is a separate method on `PipelineEngine`:
```
encode → denoise → vae_decode → export → upscale
```
Steps are independently resumable via checkpoints (`sessions/<id>/*.pt`).

### 2. Two-Stage Denoise
The denoise step loads TWO transformers **sequentially** (never both at once):
- **High-noise** (timesteps ≥ boundary) → compositional structure
- **Low-noise** (timesteps < boundary) → detail refinement

A single `UniPCMultistepScheduler` is shared across both passes.

### 3. GGUF Quantization
Transformers are stored as GGUF Q8_0 files (~10 GB each). Loaded via `WanTransformer3DModel.from_single_file()`. This means:
- LoRAs **cannot** be fused (GGUF uint weights have incompatible shapes)
- LoRAs must be loaded BEFORE group offloading

### 4. Memory Management
The app is designed to fit a 14B model in 24 GB VRAM:
- Group offloading (CPU↔GPU block-level shuttle)
- Chunked SDPA (AMD has no flash-attention → manual chunking)
- Component lifecycle (load → use → free per step)
- VAE tiling/slicing

### 5. Two Generation Modes
- **Quality mode** (`distill_lora_mode=false`): 20 steps, CFG=5.0, quality LoRAs only
- **Distill/Fast mode** (`distill_lora_mode=true`): 4 steps, CFG=1.0 (baked), flow_shift=5.0, ~10× faster

---

## Common Modification Patterns

### Adding a new generation parameter
1. Add to `config.yaml` with inline comment
2. Add field to `SessionInfo` dataclass in `session_manager.py`
3. Add to `create_session()` in `SessionManager`
4. Add to `api_generate()` params dict in `app.py`
5. Add to `api_resume()` update block in `app.py`
6. Add to `api_clone_session()` clone dict in `app.py`
7. Add UI control in `templates/index.html`
8. Add hybrid input sync in `static/app.js` (if slider+number)
9. Read from `info` object in the relevant `pipeline/engine.py` → `PipelineEngine.step_*()` method

### Adding a new pipeline step
1. Add step name to `STEP_ORDER` in `session_manager.py`
2. Add `step_<name>()` method to `PipelineEngine` in `pipeline/engine.py`
3. Add to `step_funcs` dict in `PipelineEngine.run_from_step()`
4. Update `get_resume_step()` logic
5. Add step UI in `templates/index.html` and `static/app.js`

### Adding a new LoRA
1. Place `.safetensors` file in `models/loras/`
2. Add entry to `loras:` list in `config.yaml` with: `path`, `target`, `adapter_name`, `scale`, `role`
3. The pipeline picks it up automatically — no code changes needed

### Modifying the upscale pipeline
- Spatial upscale: `upscale.py` → `load_upscale_model()`, `tile_upscale()`, `upscale_frame()`
- Frame interpolation: `rife_model.py` → `IFNet`, `interpolate_frame()`
- Called from: `upscale.py` → `upscale_video()`
- Integrated in: `pipeline/engine.py` → `PipelineEngine.step_upscale()`

---

## Gotchas & Traps

| Issue | Details |
|---|---|
| **LoRA fusion crashes** | GGUF weights are uint-quantized with different shapes. Never call `pipe.fuse_lora()`. |
| **use_stream must be False** | CUDA streams are broken on ROCm/HIP. Always `use_stream=False` in group offloading. |
| **Attention OOM on AMD** | Without chunked SDPA, the math backend materialises a 160+ GB attention matrix. Always ensure chunked SDPA is available. |
| **Text encoder device** | The UMT5-XXL encoder is ~12 GB. `device_map="auto"` splits it across GPU+CPU. If you force it to GPU, it may OOM. |
| **generate.py vs pipeline/** | These have duplicated logic (LoRA loading, offloading, pipeline building). `generate.py` is the older standalone version; `pipeline/` is the production step-by-step engine. Changes to core generation logic should go in the appropriate `pipeline/*.py` submodule. |
| **boundary_ratio** | Official Wan2.2 default is 0.9 (90% high-noise steps). Changing this significantly alters output quality. |
| **Scheduler state** | The UniPC scheduler is stateful — it must persist across the high→low pass boundary. Never reset or recreate it between passes. |
| **VAE must be float32** | Using float16/bfloat16 for VAE decode causes severe quality loss. |

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | AMD GFX ISA override (set to match GPU arch) |
| `HIP_VISIBLE_DEVICES` | `0` | Which AMD GPU to use |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | Avoids VRAM fragmentation OOMs |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Same (PyTorch 2.9+ name) |
| `TOKENIZERS_PARALLELISM` | `false` | Prevents HF tokenizer threading overhead |
| `HF_HOME` | `/app/hf_cache` | HuggingFace cache directory |

---

## Testing Locally (without GPU)

The app requires a ROCm-capable AMD GPU. Without one:
- `torch.cuda.is_available()` returns False
- All GPU-dependent steps will fail
- You can still test Flask routes and session management

---

## Directory Layout Convention

```
project root/
├── app.py                 # Flask web server (entry point)
├── pipeline_engine.py     # Backward-compat shim → imports from pipeline/
├── pipeline/              # Core generation engine (split into modules)
│   ├── __init__.py        # Re-exports all public names
│   ├── engine.py          # PipelineEngine class (5-step orchestration)
│   ├── vram_utils.py      # VRAM/RAM helpers
│   ├── sdpa.py            # Chunked SDPA (AMD attention workaround)
│   ├── config.py          # load_config(), image prep
│   ├── loaders.py         # Model component loaders
│   ├── lora.py            # LoRA adapter loading
│   ├── offload.py         # Group offloading setup
│   ├── amd_env.py         # AMD environment variables
│   └── session_logger.py  # Per-session file logger
├── session_manager.py     # Session persistence (dataclass + CRUD)
├── generate.py            # Legacy standalone CLI
├── upscale.py             # Post-processing (spatial + RIFE)
├── rife_model.py          # RIFE v4.7 frame interpolation
├── amd_tune.py            # AMD GPU detection + tuning
├── config.yaml            # Single config file
├── templates/             # Jinja2 HTML templates
├── static/                # JS + CSS served by Flask
├── models/                # Model weights (gitignored, bind-mounted)
├── input/                 # User-uploaded images
├── output/                # Generated videos
├── sessions/              # Per-generation checkpoints + logs
└── hf_cache/              # HuggingFace model cache
```

Most Python files are at the root level. The `pipeline/` package is the one exception — it was split from the 2000-line `pipeline_engine.py` for readability. A backward-compat shim (`pipeline_engine.py`) re-exports everything so existing imports still work.
