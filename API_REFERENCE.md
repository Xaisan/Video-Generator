# API Reference — Wan 2.2 I2V Video Generator

Base URL: `http://localhost:7860`

---

## Pages

| Method | Path | Description |
|---|---|---|
| GET | `/` | Main web UI (renders `index.html`) |

---

## REST API

### Sessions

| Method | Path | Description |
|---|---|---|
| GET | `/api/sessions` | List all sessions (newest first, max 100) |
| GET | `/api/sessions/<session_id>` | Get session details (params, step statuses, timing) |
| DELETE | `/api/sessions/<session_id>` | Delete a session and all its files |
| POST | `/api/sessions/<session_id>/clone` | Get generation params for cloning to a new generation |

### Generation

| Method | Path | Description |
|---|---|---|
| POST | `/api/generate` | Start a new generation. Returns `{ session_id, status }`. Only one generation can run at a time (409 if busy). |
| POST | `/api/resume/<session_id>` | Resume a session from a specific step. Body: `{ from_step, ...param_overrides }` |
| POST | `/api/cancel` | Cancel the currently running generation |

### Upload

| Method | Path | Description |
|---|---|---|
| POST | `/api/upload` | Upload an input image. Multipart form with field `image`. Returns `{ path, filename }`. |

### System

| Method | Path | Description |
|---|---|---|
| GET | `/api/vram` | Current VRAM stats `{ available, name, allocated_gb, reserved_gb, total_gb }` |
| GET | `/api/config` | Current config.yaml as JSON |
| GET | `/api/active` | Currently running session `{ session_id, running }` |
| GET | `/api/logs?n=100&since=0` | Recent log lines (ring buffer, last N entries since timestamp) |

### Session Files

| Method | Path | Description |
|---|---|---|
| GET | `/sessions/<session_id>/<filename>` | Serve session file (mp4, png, json only) |
| GET | `/input/<filename>` | Serve uploaded input image |
| GET | `/api/sessions/<session_id>/generation_log` | Get generation.log content |
| GET | `/api/sessions/<session_id>/vram_log` | Get vram.log content |

---

## SSE Events

Endpoint: `GET /api/events` (Server-Sent Events stream)

### Event Types

| Event | Data Fields | When |
|---|---|---|
| `connected` | `{ client_id }` | On SSE connection established |
| `progress` | `{ session_id, step, percent, message, vram }` | During generation (per-step progress) |
| `generation_complete` | `{ session_id, session }` | When generation finishes (success or failure) |

`percent = -1` indicates an error in the current step.

---

## POST /api/generate — Request Body

All fields are optional; defaults come from `config.yaml`.

```json
{
  "prompt": "string",
  "negative_prompt": "string",
  "input_image": "string (path from /api/upload)",
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "num_inference_steps": 20,
  "guidance_scale": 5.0,
  "guidance_scale_2": 5.0,
  "flow_shift": 8.0,
  "seed": 42,
  "fps": 16,
  "duration": 5.0,
  "output_fps": 24,
  "target_duration": 0,
  "enable_upscale": false,
  "upscale_model": "models/upscale_models/4xRealWebPhoto_v4_dat2.pth",
  "lora_scales": [{"adapter_name": "ns_high", "scale": 0.9}],
  "boundary_ratio": 0.9,
  "distill_lora_mode": false,

  "offload_type": "block_level",
  "num_blocks_per_group": 1,
  "enable_group_offload": true,
  "vae_tiling": true,
  "vae_slicing": true,
  "force_vae_cpu": false
}
```

---

## POST /api/resume/<session_id> — Request Body

```json
{
  "from_step": "denoise",
  "prompt": "optional override",
  "seed": 123,
  "num_inference_steps": 30,
  "guidance_scale": 7.0,
  "enable_upscale": true
}
```

If `from_step` is omitted, the engine auto-detects the resume point based on available checkpoints.

---

## Session Object Structure

Returned by GET `/api/sessions/<id>` and in SSE `generation_complete` events:

```json
{
  "session_id": "1771129720_9be2b16f",
  "created_at": 1771129720.123,
  "updated_at": 1771129780.456,
  "status": "done",
  "error_message": "",
  "prompt": "...",
  "negative_prompt": "...",
  "input_image": "sessions/1771129720_9be2b16f/input.png",
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "num_inference_steps": 20,
  "guidance_scale": 5.0,
  "guidance_scale_2": 5.0,
  "flow_shift": 8.0,
  "seed": 42,
  "fps": 16,
  "duration": 5.0,
  "output_fps": 24,
  "target_duration": 0,
  "enable_upscale": false,
  "upscale_model": "...",
  "boundary_ratio": 0.9,
  "distill_lora_mode": false,
  "lora_scales": [],
  "steps": {
    "encode": "done",
    "denoise": "done",
    "vae_decode": "done",
    "export": "done",
    "upscale": "skipped"
  },
  "step_times": {
    "encode": 12.34,
    "denoise": 456.78,
    "vae_decode": 89.01,
    "export": 2.34
  },
  "checkpoints": ["embeddings", "latents", "frames", "preview.mp4", "output.mp4"]
}
```

### Step Status Values
`pending` | `running` | `done` | `failed` | `skipped`

### Session Status Values
`created` | `running` | `done` | `failed` | `cancelled`
