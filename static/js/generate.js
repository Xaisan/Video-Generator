/*
 * generate.js — Generation, resume, cancel, clone & mode toggle
 * ===============================================================
 * Handles the generate/resume/cancel workflow, parameter gathering,
 * quality/fast mode toggle, and session cloning.
 */

"use strict";

const Generate = (() => {
  const { $, $$ } = App;

  // Mode defaults
  const QUALITY_DEFAULTS = { steps: 20, cfg: 5.0, cfg2: 5.0, flowShift: 8.0, boundaryRatio: 0.9 };
  const FAST_DEFAULTS    = { steps: 4,  cfg: 1.0, cfg2: 1.0, flowShift: 5.0, boundaryRatio: 0.9 };

  // ─── Mode Toggle ─────────────────────────────────────────────

  function setupModeToggle() {
    const qualBtn = $("#mode-quality");
    const fastBtn = $("#mode-fast");
    if (!qualBtn || !fastBtn) return;
    qualBtn.addEventListener("click", () => applyMode("quality"));
    fastBtn.addEventListener("click", () => applyMode("fast"));
    applyMode("quality", true);
  }

  function applyMode(mode, silent) {
    App.currentMode = mode;
    const qualBtn = $("#mode-quality");
    const fastBtn = $("#mode-fast");
    const hint = $("#mode-hint");

    if (qualBtn) qualBtn.classList.toggle("active", mode === "quality");
    if (fastBtn) fastBtn.classList.toggle("active", mode === "fast");

    if (hint) {
      hint.textContent = mode === "quality"
        ? "Quality defaults: 20 steps, CFG=5.0 — adjust freely"
        : "Fast defaults: 4 steps, CFG=1.0 (raise CFG for stronger prompt adherence)";
    }

    const preset = mode === "fast" ? FAST_DEFAULTS : QUALITY_DEFAULTS;
    Controls.setParam("steps", preset.steps);
    Controls.setParam("cfg", preset.cfg);
    Controls.setParam("cfg2", preset.cfg2);
    Controls.setParam("flow-shift", preset.flowShift);
    Controls.setParam("boundary-ratio", preset.boundaryRatio);

    $$(".lora-card").forEach(card => {
      if (card.dataset.role === "distill") {
        card.classList.toggle("lora-disabled", mode === "quality");
      }
    });

    Controls.updateTotalSteps();

    if (!silent) {
      App.toast(mode === "quality"
        ? "🎨 Quality defaults applied — all params adjustable"
        : "⚡ Fast defaults applied — increase CFG for more prompt control",
        "info", 3000);
    }
  }

  // ─── Parameter Gathering ──────────────────────────────────────

  function gatherParams() {
    const loraRanges = $$(".lora-scale");
    const loraScales = [];
    loraRanges.forEach(inp => loraScales.push(parseFloat(inp.value) || 0));

    const duration = parseFloat($("#duration").value) || 5;
    const fps = parseInt($("#fps").value) || 16;
    const numFrames = Controls.calcFrames4k1(duration, fps);

    const params = {
      prompt:              $("#prompt").value,
      negative_prompt:     $("#negative-prompt").value,
      input_image:         $("#input-image-path").value,
      width:               parseInt($("#width").value),
      height:              parseInt($("#height").value),
      num_frames:          numFrames,
      fps:                 fps,
      duration:            duration,
      output_fps:          parseInt($("#output-fps").value) || fps,
      target_duration:     parseFloat($("#target-duration")?.value) || 0,
      num_inference_steps: parseInt($("#steps").value),
      guidance_scale:      parseFloat($("#cfg").value),
      guidance_scale_2:    parseFloat($("#cfg2").value),
      flow_shift:          parseFloat($("#flow-shift").value),
      boundary_ratio:      parseFloat($("#boundary-ratio").value),
      seed:                parseInt($("#seed").value),
      enable_upscale:      $("#enable-upscale").checked,
      upscale_model:       $("#upscale-model") ? $("#upscale-model").value : "",
      lora_scales:         loraScales,
      distill_lora_mode:   App.currentMode === "fast",
      // Memory settings
      offload_type:        $("#offload-type") ? $("#offload-type").value : "block_level",
      num_blocks_per_group: parseInt($("#num-blocks-per-group")?.value) || 1,
      enable_group_offload: $("#enable-group-offload")?.checked ?? true,
      vae_tiling:          $("#vae-tiling")?.checked ?? true,
      vae_slicing:         $("#vae-slicing")?.checked ?? true,
      force_vae_cpu:       $("#force-vae-cpu")?.checked ?? false,
    };

    const presetId = $("#preset-id")?.value || "";
    if (presetId) params.preset_id = presetId;

    // User assignment
    if (App.activeUserId) {
      params.user_id = App.activeUserId;
      params.user_name = App.activeUserName;
    }

    return params;
  }

  // ─── Start Generation ─────────────────────────────────────────

  async function start() {
    const params = gatherParams();
    if (!params.input_image) {
      App.toast("Please upload an input image first.", "warning");
      return;
    }
    if (!params.prompt.trim()) {
      App.toast("Please enter a prompt.", "warning");
      return;
    }

    const btnGenerate = $("#btn-generate");
    btnGenerate.disabled = true;
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (d.error) {
        App.toast("Error: " + d.error, "error");
        btnGenerate.disabled = false;
        return;
      }

      App.toast("Generation started!", "info");
      App.stepProgressData = {};
      App.currentSessionId = d.session_id;
      await Sessions.refresh();
      Sessions.show(d.session_id);
      Timer.start();
      Sessions.startAutoRefresh();
      Logs.startPolling();
      Logs.startSessionLogPolling();
    } catch (e) {
      App.toast("Request failed: " + e.message, "error");
      btnGenerate.disabled = false;
    }
  }

  // ─── Resume from Step ─────────────────────────────────────────

  async function resumeFromStep(step) {
    if (!App.currentSessionId) return;
    const stepOrder = ["encode", "denoise", "vae_decode", "export", "upscale"];
    const stepIdx = stepOrder.indexOf(step);
    const downstreamSteps = stepOrder.slice(stepIdx).join(", ");
    const confirmed = confirm(
      `Resume from "${step}"?\n\nThis will re-run: ${downstreamSteps}\nAny existing checkpoints for these steps will be overwritten.`
    );
    if (!confirmed) return;
    try {
      const payload = { from_step: step };
      const upscaleCheckbox = $("#enable-upscale");

      if (step === "upscale") {
        payload.enable_upscale = true;
        if (upscaleCheckbox) upscaleCheckbox.checked = true;
      } else if (upscaleCheckbox) {
        payload.enable_upscale = upscaleCheckbox.checked;
      }

      const stepUpscaleSelect = $('.step-card[data-step="upscale"] select');
      const mainUpscaleSelect = $("#upscale-model");
      if (stepUpscaleSelect) payload.upscale_model = stepUpscaleSelect.value;
      else if (mainUpscaleSelect) payload.upscale_model = mainUpscaleSelect.value;

      const stepFpsInput = $('.step-card[data-step="upscale"] .upscale-step-fps');
      const stepDurInput = $('.step-card[data-step="upscale"] .upscale-step-duration');
      payload.output_fps = parseInt(stepFpsInput?.value) || parseInt($("#output-fps")?.value) || 24;
      payload.target_duration = parseFloat(stepDurInput?.value) || parseFloat($("#target-duration")?.value) || 0;

      const r = await fetch(`/api/resume/${App.currentSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) { App.toast("Error: " + d.error, "error"); return; }
      App.toast(`Resuming from ${step}…`, "info");
      App.stepProgressData = {};
      const btnCancel = $("#btn-cancel");
      if (btnCancel) btnCancel.style.display = "inline-flex";
      Timer.start();
      Sessions.startAutoRefresh();
      Logs.startPolling();
      Logs.startSessionLogPolling();
      Sessions.show(App.currentSessionId);
    } catch (e) {
      App.toast("Resume failed: " + e.message, "error");
    }
  }

  // ─── Cancel ───────────────────────────────────────────────────

  async function cancel() {
    try { await fetch("/api/cancel", { method: "POST" }); App.toast("Cancel requested…", "warning"); }
    catch (e) { console.error(e); }
  }

  // ─── Delete Session ───────────────────────────────────────────

  async function deleteSession() {
    if (!App.currentSessionId) return;
    if (!confirm("Delete this session and all its files?")) return;
    try {
      await fetch(`/api/sessions/${App.currentSessionId}`, { method: "DELETE" });
      App.toast("Session deleted", "info");
      App.currentSessionId = null;
      await Sessions.refresh();
      if (typeof Dashboard !== "undefined") {
        Dashboard.show();
      } else {
        showGeneratePanel();
      }
    } catch (e) {
      App.toast("Delete failed: " + e.message, "error");
    }
  }

  // ─── Clone Session ────────────────────────────────────────────

  async function cloneSession() {
    if (!App.currentSessionId) return;
    try {
      const r = await fetch(`/api/sessions/${App.currentSessionId}/clone`, { method: "POST" });
      const d = await r.json();
      if (d.error) { App.toast("Clone failed: " + d.error, "error"); return; }

      $("#prompt").value = d.prompt || "";
      $("#negative-prompt").value = d.negative_prompt || "";
      Controls.setParam("width", d.width || 832);
      Controls.setParam("height", d.height || 480);
      Controls.setParam("duration", d.duration || 5.0);
      Controls.setParam("fps", d.fps || 16);
      Controls.setParam("steps", d.num_inference_steps || 20);
      Controls.setParam("cfg", d.guidance_scale || 5.0);
      Controls.setParam("cfg2", d.guidance_scale_2 || 5.0);
      Controls.setParam("flow-shift", d.flow_shift || 8.0);
      Controls.setParam("output-fps", d.output_fps || 24);
      Controls.setParam("target-duration", d.target_duration || 0);
      Controls.setParam("boundary-ratio", d.boundary_ratio || 0.9);
      $("#seed").value = d.seed || 42;
      $("#enable-upscale").checked = d.enable_upscale || false;
      if (d.upscale_model && $("#upscale-model")) {
        $("#upscale-model").value = d.upscale_model;
      }

      applyMode(d.distill_lora_mode ? "fast" : "quality", true);

      if (d.preset_id) {
        const presetSel = $("#preset-select");
        if (presetSel) presetSel.value = d.preset_id;
        $("#preset-id").value = d.preset_id;
        Presets.updateInfo(d.preset_id);
      } else {
        const presetSel = $("#preset-select");
        if (presetSel) presetSel.value = "";
        $("#preset-id").value = "";
        Presets.updateInfo("");
      }

      Controls.updateFramesFromDuration();
      Controls.updateInterpHint();
      Controls.updateMegapixels();
      Controls.updateTotalSteps();
      Controls.updateTargetDurationHint();

      if (d.lora_scales && d.lora_scales.length) {
        const loraRanges = $$(".lora-scale");
        const loraNums = $$(".lora-scale-num");
        d.lora_scales.forEach((val, i) => {
          if (loraRanges[i]) loraRanges[i].value = val;
          if (loraNums[i]) loraNums[i].value = val;
        });
      }

      if (d.input_image_url) {
        const previewImg = $("#preview-img");
        const dropText = $("#drop-text");
        previewImg.src = d.input_image_url;
        previewImg.style.display = "block";
        dropText.style.display = "none";
        $("#input-image-path").value = d.input_image || "";
        if (!d.input_image) {
          App.toast("Parameters cloned — please upload/confirm input image", "info", 5000);
        }
      }

      showGeneratePanel();
      App.toast("Session parameters cloned", "success");
    } catch (e) {
      App.toast("Clone error: " + e.message, "error");
    }
  }

  // ─── Navigation ───────────────────────────────────────────────

  function showGeneratePanel() {
    const dashPanel = $("#panel-dashboard");
    if (dashPanel) dashPanel.style.display = "none";
    if (typeof Dashboard !== "undefined") Dashboard.hide();
    $("#panel-generate").style.display = "block";
    $("#panel-session").style.display = "none";
    App.currentSessionId = null;
    Sessions.stopAutoRefresh();
    Logs.stopPolling();
    Logs.stopSessionLogPolling();
    Timer.stop();
    Sessions.renderList();
    const logSec = $("#session-logs-section");
    if (logSec) logSec.style.display = "none";
  }

  // ─── Keyboard Shortcuts ───────────────────────────────────────

  function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      const tag = e.target.tagName.toLowerCase();
      const isInput = tag === "input" || tag === "textarea" || tag === "select";

      if (e.key === "Enter" && !isInput && $("#panel-generate").style.display !== "none") {
        e.preventDefault();
        start();
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        cancel();
        return;
      }
      if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        showGeneratePanel();
        return;
      }
    });
  }

  return {
    setupModeToggle,
    applyMode,
    gatherParams,
    start,
    resumeFromStep,
    cancel,
    deleteSession,
    cloneSession,
    showGeneratePanel,
    setupKeyboardShortcuts,
  };
})();
