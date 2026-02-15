/*
 * presets.js — Model preset selector & application
 * ==================================================
 * Handles the preset dropdown in the main generate panel:
 * select, apply, info display, LoRA grid rebuild.
 *
 * The preset EDITOR (create/edit) lives in a separate page
 * opened via window.open() — see preset-editor.js.
 */

"use strict";

const Presets = (() => {
  const { $, $$ } = App;

  function setup() {
    const sel = $("#preset-select");
    const btnApply = $("#btn-preset-apply");
    const btnEdit = $("#btn-preset-edit");
    const btnNew = $("#btn-preset-new");
    const btnDel = $("#btn-preset-delete");

    if (!sel) return;

    sel.addEventListener("change", () => {
      const pid = sel.value;
      updateInfo(pid);
      btnDel.style.display = pid ? "inline-flex" : "none";
    });

    btnApply.addEventListener("click", () => apply(sel.value));
    btnEdit.addEventListener("click", () => {
      const pid = sel.value;
      if (pid) openEditor(pid);
      else App.toast("Select a preset to edit", "warning");
    });
    btnNew.addEventListener("click", () => openEditor(null));
    btnDel.addEventListener("click", () => deletePreset(sel.value));

    updateInfo(sel.value);
  }

  function updateInfo(presetId) {
    const infoDiv = $("#preset-info");
    const badge = $("#preset-active-name");
    if (!presetId) {
      if (infoDiv) infoDiv.style.display = "none";
      if (badge) badge.textContent = "";
      return;
    }

    const preset = App.presets.find(p => p.id === presetId);
    if (!preset) {
      if (infoDiv) infoDiv.style.display = "none";
      if (badge) badge.textContent = "";
      return;
    }

    if (badge) badge.textContent = `(${preset.name})`;
    if (infoDiv) {
      infoDiv.style.display = "flex";
      $("#preset-info-high").textContent = App.basename(preset.gguf_transformer_high) || "— default —";
      $("#preset-info-low").textContent = App.basename(preset.gguf_transformer_low) || "— default —";
      $("#preset-info-vae").textContent = App.basename(preset.vae_path) || "— default —";
      const loraCount = (preset.loras || []).length;
      $("#preset-info-loras").textContent = loraCount ? `${loraCount} adapter(s)` : "— config.yaml defaults —";
      $("#preset-info-desc").textContent = preset.description || "—";
    }
  }

  function apply(presetId) {
    if (!presetId) {
      $("#preset-id").value = "";
      $("#preset-active-name").textContent = "";
      rebuildLoraGrid(App.configLoras);
      App.toast("Using config.yaml defaults", "info", 2000);
      return;
    }

    const preset = App.presets.find(p => p.id === presetId);
    if (!preset) { App.toast("Preset not found", "error"); return; }

    $("#preset-id").value = presetId;
    $("#preset-active-name").textContent = `(${preset.name})`;

    const defs = preset.defaults || {};
    if (defs.num_inference_steps) Controls.setParam("steps", defs.num_inference_steps);
    if (defs.guidance_scale) Controls.setParam("cfg", defs.guidance_scale);
    if (defs.guidance_scale_2) Controls.setParam("cfg2", defs.guidance_scale_2);
    if (defs.flow_shift) Controls.setParam("flow-shift", defs.flow_shift);
    if (defs.boundary_ratio) Controls.setParam("boundary-ratio", defs.boundary_ratio);

    if (defs.distill_lora_mode !== undefined) {
      Generate.applyMode(defs.distill_lora_mode ? "fast" : "quality", true);
    }

    if (preset.loras && preset.loras.length > 0) {
      rebuildLoraGrid(preset.loras);
    } else {
      rebuildLoraGrid(App.configLoras);
    }

    Controls.updateTotalSteps();
    Controls.updateFramesFromDuration();

    App.toast(`Preset "${preset.name}" applied`, "success", 2000);
  }

  function rebuildLoraGrid(loras) {
    const grid = $("#lora-grid");
    if (!grid) return;

    grid.innerHTML = "";
    loras.forEach((lora, idx) => {
      const isDistill = lora.role === "distill";
      const isHigh = (lora.target || "transformer") === "transformer";
      const disabledClass = (isDistill && App.currentMode === "quality") ? " lora-disabled" : "";

      const card = document.createElement("div");
      card.className = `lora-card${isDistill ? " lora-distill" : ""}${disabledClass}`;
      card.dataset.index = idx;
      card.dataset.role = lora.role || "quality";
      card.innerHTML = `
        <div class="lora-top">
          <span class="lora-name" title="${lora.adapter_name || ""}">${lora.adapter_name || ""}</span>
          <span class="lora-badge ${isHigh ? "badge-high" : "badge-low"}">
            ${isHigh ? "HIGH" : "LOW"}
          </span>
          ${isDistill ? '<span class="lora-badge badge-distill" title="Distill LoRA — only active in Fast mode">⚡</span>' : ""}
        </div>
        <div class="lora-control">
          <input type="range" class="lora-scale" data-lora="${idx}"
                 value="${lora.scale || 1.0}" min="0" max="5" step="0.05" />
          <input type="number" class="lora-scale-num" data-lora="${idx}"
                 value="${lora.scale || 1.0}" min="0" max="5" step="0.05" />
        </div>
        <div class="lora-file" title="${lora.path || ""}">${App.basename(lora.path)}</div>
      `;
      grid.appendChild(card);
    });

    Controls.setupLoraHybridInputs();
  }

  // ─── Preset Editor (separate window) ──────────────────────────

  function openEditor(presetId) {
    const url = presetId ? `/presets/editor?id=${presetId}` : "/presets/editor";
    const w = 720, h = 700;
    const left = (screen.width - w) / 2;
    const top = (screen.height - h) / 2;
    const win = window.open(url, "preset_editor",
      `width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`);
    if (win) win.focus();
  }

  // ─── Delete ───────────────────────────────────────────────────

  async function deletePreset(presetId) {
    if (!presetId) return;
    const preset = App.presets.find(p => p.id === presetId);
    const name = preset?.name || presetId;
    if (!confirm(`Delete preset "${name}"?`)) return;

    try {
      const r = await fetch(`/api/presets/${presetId}`, { method: "DELETE" });
      if (r.ok) {
        App.toast(`Preset "${name}" deleted`, "info");
        await refreshList();
        $("#preset-id").value = "";
        updateInfo("");
      } else {
        App.toast("Delete failed", "error");
      }
    } catch (e) {
      App.toast("Delete error: " + e.message, "error");
    }
  }

  // ─── Refresh preset list ──────────────────────────────────────

  async function refreshList() {
    try {
      const r = await fetch("/api/presets");
      App.presets = await r.json();
      const sel = $("#preset-select");
      if (!sel) return;
      const currentVal = sel.value;
      sel.innerHTML = '<option value="">— Use config.yaml defaults —</option>';
      for (const p of App.presets) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
      }
      sel.value = currentVal;
      const btnDel = $("#btn-preset-delete");
      if (btnDel) btnDel.style.display = sel.value ? "inline-flex" : "none";
    } catch (e) {
      console.error("Failed to refresh presets:", e);
    }
  }

  // Listen for messages from the preset editor window
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "preset-saved") {
      refreshList().then(() => {
        if (e.data.presetId) {
          const sel = $("#preset-select");
          if (sel) {
            sel.value = e.data.presetId;
            updateInfo(e.data.presetId);
          }
        }
      });
    }
  });

  return {
    setup,
    updateInfo,
    apply,
    rebuildLoraGrid,
    refreshList,
  };
})();
