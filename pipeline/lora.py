"""
pipeline/lora.py — LoRA adapter loading (per-transformer target)
================================================================

With GGUF transformers we CANNOT fuse (GGUF stores uint-quantized weights
with different shapes from bfloat16 originals), so LoRAs stay active
during inference.

Public API:
  apply_loras_to_transformer(pipe, cfg, target, ...)  — load LoRAs for ONE target
  apply_loras(pipe, cfg, lora_overrides)              — load all (legacy wrapper)

No project-internal dependencies (pure diffusers/peft operations).
"""

import os


def apply_loras_to_transformer(pipe, cfg: dict, target: str,
                               lora_overrides: list | None = None,
                               load_as_primary: bool = False,
                               distill_lora_mode: bool = False):
    """Load LoRAs for a SINGLE transformer target.

    target: "transformer" or "transformer_2" — used to FILTER config entries.
    load_as_primary: if True, load into pipe.transformer even when target is
                     "transformer_2".  Needed when the low-noise transformer
                     is mounted as pipe.transformer (because load_lora_weights
                     requires a non-None .transformer to read its config).
    distill_lora_mode: if False, skip LoRAs with role="distill" (LightX2V, t2v_speed).
                       If True, load all LoRAs including distillation ones.

    This avoids loading all LoRAs at once (halves RAM pressure) and
    prevents the duplicate peft_config warning.
    """
    import copy
    lora_cfgs = copy.deepcopy(cfg.get("loras", []))
    if not lora_cfgs:
        return

    # Apply scale overrides from session
    if lora_overrides:
        if lora_overrides and isinstance(lora_overrides[0], (int, float)):
            for i, scale_val in enumerate(lora_overrides):
                if i < len(lora_cfgs):
                    lora_cfgs[i]["scale"] = float(scale_val)
        else:
            override_map = {o["adapter_name"]: o["scale"] for o in lora_overrides}
            for lora in lora_cfgs:
                if lora["adapter_name"] in override_map:
                    lora["scale"] = override_map[lora["adapter_name"]]

    # Filter to only LoRAs for this target
    target_loras = [l for l in lora_cfgs if l.get("target", "transformer") == target]

    # Filter by role: skip distill LoRAs in quality mode
    if not distill_lora_mode:
        skipped = [l["adapter_name"] for l in target_loras if l.get("role") == "distill"]
        if skipped:
            print(f"  ⏭ Quality mode: skipping distill LoRAs: {skipped}")
        target_loras = [l for l in target_loras if l.get("role") != "distill"]

    if not target_loras:
        print(f"  No LoRAs to load for {target}")
        return

    adapters = []
    scales = {}

    for lora in target_loras:
        path = lora["path"]
        name = lora["adapter_name"]
        scale = lora.get("scale", 1.0)

        if not os.path.isfile(path):
            print(f"  ⚠ LoRA not found, skipping: {path}")
            continue

        print(f"  📦 Loading LoRA: {name} → {target} (scale={scale})")
        try:
            if target == "transformer_2" and not load_as_primary:
                pipe.load_lora_weights(path, adapter_name=name,
                                       load_into_transformer_2=True)
            else:
                pipe.load_lora_weights(path, adapter_name=name)
            adapters.append(name)
            scales[name] = scale
        except Exception as e:
            print(f"  ❌ LoRA {name}: {e}")

    if adapters:
        if load_as_primary:
            model = pipe.transformer
        else:
            model = pipe.transformer if target == "transformer" else pipe.transformer_2
        if model is not None:
            model.set_adapters(adapters, weights=[scales[a] for a in adapters])
            print(f"  ✅ Active adapters on {target}: {adapters}")


def apply_loras(pipe, cfg: dict, lora_overrides: list | None = None):
    """Load LoRAs into the pipeline (legacy wrapper — both targets)."""
    apply_loras_to_transformer(pipe, cfg, "transformer", lora_overrides)
    if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
        apply_loras_to_transformer(pipe, cfg, "transformer_2", lora_overrides)
