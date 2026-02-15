/*
 * preset-editor.js — Standalone preset editor logic
 * ===================================================
 * Runs inside the separate preset editor window (preset_editor.html).
 * Handles create/edit preset, model dropdowns, LoRA rows.
 * Communicates back to the main window via window.opener.postMessage().
 */

"use strict";

const PresetEditor = (() => {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  let editingPresetId = null;
  let availableModels = {};

  function basename(path) {
    if (!path) return "";
    return path.split("/").pop();
  }

  function toast(message, type = "info", duration = 3000) {
    const container = $("#toast-container");
    if (!container) return;
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.classList.add("toast-out");
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  // ─── Initialize ───────────────────────────────────────────────

  async function init() {
    // Apply theme from parent or localStorage
    const theme = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", theme);

    // Get preset ID from URL params
    const params = new URLSearchParams(window.location.search);
    editingPresetId = params.get("id") || null;

    // Fetch available models
    try {
      const r = await fetch("/api/models/scan");
      availableModels = await r.json();
    } catch (e) {
      console.error("Failed to scan models:", e);
    }

    populateModelDropdowns();

    if (editingPresetId) {
      // Load existing preset
      try {
        const r = await fetch(`/api/presets/${editingPresetId}`);
        if (r.ok) {
          const preset = await r.json();
          fillForm(preset);
          $("#editor-title").textContent = `Edit: ${preset.name}`;
          $("#btn-duplicate").style.display = "inline-flex";
        } else {
          toast("Preset not found", "error");
        }
      } catch (e) {
        toast("Failed to load preset: " + e.message, "error");
      }
    } else {
      $("#editor-title").textContent = "New Model Preset";
      $("#btn-duplicate").style.display = "none";
    }

    // Bind buttons
    $("#btn-save").addEventListener("click", save);
    $("#btn-cancel").addEventListener("click", () => window.close());
    $("#btn-duplicate").addEventListener("click", duplicate);
    $("#btn-add-lora").addEventListener("click", () => addLoraRow());
  }

  // ─── Form population ─────────────────────────────────────────

  function fillForm(preset) {
    $("#pe-name").value = preset.name || "";
    $("#pe-desc").value = preset.description || "";
    $("#pe-high").value = preset.gguf_transformer_high || "";
    $("#pe-low").value = preset.gguf_transformer_low || "";
    $("#pe-vae").value = preset.vae_path || "";
    $("#pe-text-enc").value = preset.text_encoder_path || "";

    const defs = preset.defaults || {};
    $("#pe-steps").value = defs.num_inference_steps || 20;
    $("#pe-cfg").value = defs.guidance_scale || 5.0;
    $("#pe-cfg2").value = defs.guidance_scale_2 || 5.0;
    $("#pe-flow").value = defs.flow_shift || 8.0;
    $("#pe-boundary").value = defs.boundary_ratio || 0.9;
    $("#pe-distill").checked = defs.distill_lora_mode || false;

    // Populate LoRA rows
    const loraList = $("#lora-list");
    loraList.innerHTML = "";
    (preset.loras || []).forEach(lora => addLoraRow(lora));
  }

  function populateModelDropdowns() {
    populateSelect("#pe-high", availableModels.unet || [], "— Use config.yaml default —");
    populateSelect("#pe-low", availableModels.unet || [], "— Use config.yaml default —");
    populateSelect("#pe-vae", availableModels.vae || [], "— Use config.yaml default —");
    populateSelect("#pe-text-enc", availableModels.text_encoders || [], "— Use config.yaml default —");
  }

  function populateSelect(selector, files, emptyLabel) {
    const sel = $(selector);
    if (!sel) return;
    const currentVal = sel.value;
    sel.innerHTML = `<option value="">${emptyLabel}</option>`;
    for (const f of files) {
      const opt = document.createElement("option");
      opt.value = f.path;
      opt.textContent = `${f.filename} (${f.size_mb} MB)`;
      sel.appendChild(opt);
    }
    sel.value = currentVal;
  }

  // ─── LoRA rows ────────────────────────────────────────────────

  function addLoraRow(lora) {
    const list = $("#lora-list");
    if (!list) return;
    const loraFiles = availableModels.loras || [];

    const row = document.createElement("div");
    row.className = "lora-row";

    // File selector
    const fileSel = document.createElement("select");
    fileSel.className = "lora-file-select";
    fileSel.innerHTML = '<option value="">— Select LoRA —</option>';
    for (const f of loraFiles) {
      const opt = document.createElement("option");
      opt.value = f.path;
      opt.textContent = f.filename;
      fileSel.appendChild(opt);
    }
    if (lora?.path) fileSel.value = lora.path;

    // Adapter name
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "lora-name-input";
    nameInput.placeholder = "adapter_name";
    nameInput.value = lora?.adapter_name || "";

    // Auto-fill adapter name from filename
    fileSel.addEventListener("change", () => {
      if (!nameInput.value && fileSel.value) {
        const fname = fileSel.value.split("/").pop().replace(/\.safetensors$/, "");
        nameInput.value = fname.replace(/[^a-zA-Z0-9_]/g, "_").substring(0, 30);
      }
    });

    // Target select
    const targetSel = document.createElement("select");
    targetSel.className = "lora-target-select";
    targetSel.innerHTML = `
      <option value="transformer">HIGH</option>
      <option value="transformer_2">LOW</option>
    `;
    if (lora?.target) targetSel.value = lora.target;

    // Role select
    const roleSel = document.createElement("select");
    roleSel.className = "lora-role-select";
    roleSel.innerHTML = `
      <option value="quality">Quality</option>
      <option value="distill">Distill</option>
    `;
    if (lora?.role) roleSel.value = lora.role;

    // Scale
    const scaleInput = document.createElement("input");
    scaleInput.type = "number";
    scaleInput.className = "lora-scale-input";
    scaleInput.min = "0";
    scaleInput.max = "5";
    scaleInput.step = "0.05";
    scaleInput.value = lora?.scale ?? 1.0;

    // Remove button
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn-remove-lora";
    removeBtn.textContent = "✕";
    removeBtn.title = "Remove this LoRA";
    removeBtn.addEventListener("click", () => row.remove());

    row.appendChild(fileSel);
    row.appendChild(nameInput);
    row.appendChild(targetSel);
    row.appendChild(roleSel);
    row.appendChild(scaleInput);
    row.appendChild(removeBtn);
    list.appendChild(row);
  }

  // ─── Gather form data ────────────────────────────────────────

  function gatherData() {
    const loras = [];
    $$("#lora-list .lora-row").forEach(row => {
      const selects = row.querySelectorAll("select");
      const inputs = row.querySelectorAll("input");
      const path = selects[0]?.value || "";
      const adapterName = inputs[0]?.value || "";
      const target = selects[1]?.value || "transformer";
      const role = selects[2]?.value || "quality";
      const scale = parseFloat(inputs[1]?.value) || 1.0;
      if (path) {
        loras.push({ path, adapter_name: adapterName, target, role, scale });
      }
    });

    return {
      name: $("#pe-name").value.trim(),
      description: $("#pe-desc").value.trim(),
      gguf_transformer_high: $("#pe-high").value,
      gguf_transformer_low: $("#pe-low").value,
      vae_path: $("#pe-vae").value,
      text_encoder_path: $("#pe-text-enc").value,
      loras: loras,
      defaults: {
        num_inference_steps: parseInt($("#pe-steps").value) || 20,
        guidance_scale: parseFloat($("#pe-cfg").value) || 5.0,
        guidance_scale_2: parseFloat($("#pe-cfg2").value) || 5.0,
        flow_shift: parseFloat($("#pe-flow").value) || 8.0,
        boundary_ratio: parseFloat($("#pe-boundary").value) || 0.9,
        distill_lora_mode: $("#pe-distill").checked || false,
      },
    };
  }

  // ─── Save ─────────────────────────────────────────────────────

  async function save() {
    const data = gatherData();
    if (!data.name) {
      toast("Preset name is required", "warning");
      return;
    }

    try {
      let r;
      if (editingPresetId) {
        r = await fetch(`/api/presets/${editingPresetId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      } else {
        r = await fetch("/api/presets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      }
      const result = await r.json();
      if (r.ok) {
        toast(`Preset "${data.name}" saved`, "success");
        // Notify parent window
        if (window.opener) {
          window.opener.postMessage({
            type: "preset-saved",
            presetId: result.id || editingPresetId,
          }, "*");
        }
        // If creating new, switch to edit mode
        if (!editingPresetId && result.id) {
          editingPresetId = result.id;
          $("#editor-title").textContent = `Edit: ${data.name}`;
          $("#btn-duplicate").style.display = "inline-flex";
          // Update URL without reload
          history.replaceState(null, "", `/presets/editor?id=${result.id}`);
        }
      } else {
        toast("Save failed: " + (result.error || "unknown"), "error");
      }
    } catch (e) {
      toast("Save error: " + e.message, "error");
    }
  }

  // ─── Duplicate ────────────────────────────────────────────────

  async function duplicate() {
    if (!editingPresetId) return;
    try {
      const r = await fetch(`/api/presets/${editingPresetId}/duplicate`, { method: "POST" });
      if (r.ok) {
        const result = await r.json();
        toast("Preset duplicated — editing copy", "success");
        editingPresetId = result.id;
        fillForm(result);
        $("#editor-title").textContent = `Edit: ${result.name}`;
        history.replaceState(null, "", `/presets/editor?id=${result.id}`);
        if (window.opener) {
          window.opener.postMessage({ type: "preset-saved", presetId: result.id }, "*");
        }
      } else {
        toast("Duplicate failed", "error");
      }
    } catch (e) {
      toast("Duplicate error: " + e.message, "error");
    }
  }

  document.addEventListener("DOMContentLoaded", init);

  return { init };
})();
