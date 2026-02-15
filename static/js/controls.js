/*
 * controls.js — Hybrid slider/number inputs & computed values
 * ============================================================
 * Manages the bidirectional sync between range sliders and number inputs,
 * and updates computed display values (frames, megapixels, interp hints).
 */

"use strict";

const Controls = (() => {
  const { $, $$ } = App;

  const HYBRID_PAIRS = [
    ["width",              "width-num"],
    ["height",             "height-num"],
    ["duration",           "duration-num"],
    ["fps",                "fps-num"],
    ["output-fps",         "output-fps-num"],
    ["steps",              "steps-num"],
    ["cfg",                "cfg-num"],
    ["cfg2",               "cfg2-num"],
    ["flow-shift",         "flow-shift-num"],
    ["boundary-ratio",     "boundary-ratio-num"],
    ["num-blocks-per-group","num-blocks-per-group-num"],
    ["target-duration",    "target-duration-num"],
  ];

  function setupHybridInputs() {
    for (const [rangeId, numId] of HYBRID_PAIRS) {
      const range = $(`#${rangeId}`);
      const num = $(`#${numId}`);
      if (!range || !num) continue;

      range.addEventListener("input", () => {
        num.value = range.value;
        onParamChange(rangeId);
      });

      num.addEventListener("input", () => {
        range.value = num.value;
        onParamChange(rangeId);
      });

      num.addEventListener("blur", () => {
        let v = parseFloat(num.value);
        const mn = parseFloat(range.min);
        const mx = parseFloat(range.max);
        if (isNaN(v)) v = parseFloat(range.value);
        v = Math.max(mn, Math.min(mx, v));
        num.value = v;
        range.value = v;
        onParamChange(rangeId);
      });
    }

    setupLoraHybridInputs();
  }

  function setupLoraHybridInputs() {
    $$(".lora-scale").forEach(range => {
      const idx = range.dataset.lora;
      const num = $(`.lora-scale-num[data-lora="${idx}"]`);
      if (!num) return;

      range.addEventListener("input", () => { num.value = range.value; });
      num.addEventListener("input", () => { range.value = num.value; });
      num.addEventListener("blur", () => {
        let v = parseFloat(num.value);
        if (isNaN(v)) v = parseFloat(range.value);
        v = Math.max(0, Math.min(5, v));
        num.value = v;
        range.value = v;
      });
    });
  }

  function setParam(id, val) {
    const range = $(`#${id}`);
    const num = $(`#${id}-num`);
    if (range) range.value = val;
    if (num) num.value = val;
  }

  function onParamChange(id) {
    if (id === "duration" || id === "fps") updateFramesFromDuration();
    if (id === "fps" || id === "output-fps") updateInterpHint();
    if (id === "width" || id === "height") { updateMegapixels(); updateResDimLabel(); }
    if (id === "steps") updateTotalSteps();
    if (id === "target-duration" || id === "duration" || id === "fps") updateTargetDurationHint();
  }

  // ─── Computed values ──────────────────────────────────────────

  function calcFrames4k1(duration, fps) {
    let raw = Math.round(duration * fps);
    if (raw < 5) raw = 5;
    let frames = Math.round((raw - 1) / 4) * 4 + 1;
    if (frames < 5) frames = 5;
    return frames;
  }

  function updateFramesFromDuration() {
    const duration = parseFloat($("#duration").value) || 5;
    const fps = parseInt($("#fps").value) || 16;
    const frames = calcFrames4k1(duration, fps);
    const el = $("#frames-computed");
    if (el) el.textContent = frames;
  }

  function updateInterpHint() {
    const fps = parseInt($("#fps").value) || 16;
    const outputFps = parseInt($("#output-fps").value) || fps;
    const targetDur = parseFloat($("#target-duration")?.value) || 0;
    const hint = $("#interp-hint");
    if (!hint) return;

    if (targetDur > 0) {
      const duration = parseFloat($("#duration").value) || 5;
      const sourceDur = (calcFrames4k1(duration, fps) - 1) / fps;
      const speed = sourceDur / targetDur;
      if (Math.abs(speed - 1) < 0.05) {
        hint.textContent = `RIFE: ${fps}→${outputFps}fps`;
        hint.style.color = "var(--success)";
      } else if (speed > 1) {
        hint.textContent = `RIFE: ${speed.toFixed(1)}× faster → ${targetDur}s`;
        hint.style.color = "var(--warning, #d29922)";
      } else {
        hint.textContent = `RIFE: ${(1/speed).toFixed(1)}× slower → ${targetDur}s`;
        hint.style.color = "var(--info, #58a6ff)";
      }
    } else if (outputFps > fps) {
      const ratio = Math.round(outputFps / fps);
      hint.textContent = `RIFE: ${fps}→${outputFps} (~${ratio}×)`;
      hint.style.color = "var(--success)";
    } else {
      hint.textContent = `No interp (${fps}fps)`;
      hint.style.color = "var(--text-muted)";
    }
  }

  function updateTargetDurationHint() {
    const hint = $("#target-duration-hint");
    if (!hint) return;
    const targetDur = parseFloat($("#target-duration")?.value) || 0;
    if (targetDur <= 0) {
      hint.textContent = "0 = keep original duration";
      hint.style.color = "";
      return;
    }
    const fps = parseInt($("#fps").value) || 16;
    const duration = parseFloat($("#duration").value) || 5;
    const sourceDur = (calcFrames4k1(duration, fps) - 1) / fps;
    const speed = sourceDur / targetDur;
    if (Math.abs(speed - 1) < 0.05) {
      hint.textContent = `≈ original (${sourceDur.toFixed(1)}s)`;
      hint.style.color = "";
    } else if (speed > 1) {
      hint.textContent = `${speed.toFixed(2)}× speed up (${sourceDur.toFixed(1)}s → ${targetDur}s)`;
      hint.style.color = "var(--warning, #d29922)";
    } else {
      hint.textContent = `${(1/speed).toFixed(2)}× slow motion (${sourceDur.toFixed(1)}s → ${targetDur}s)`;
      hint.style.color = "var(--info, #58a6ff)";
    }
  }

  function updateMegapixels() {
    const w = parseInt($("#width").value) || 832;
    const h = parseInt($("#height").value) || 480;
    const mp = (w * h / 1e6).toFixed(2);
    const el = $("#mpx-computed");
    if (el) el.textContent = mp;
  }

  function updateTotalSteps() {
    // Steps value is total steps — no computation needed
  }

  function updateResDimLabel() {
    const el = $("#res-scale-dim");
    if (el) el.textContent = `${$("#width").value}×${$("#height").value}`;
  }

  // ─── Resolution scaling ───────────────────────────────────────

  function scaleResolution(pct) {
    const curW = parseInt($("#width").value) || 832;
    const curH = parseInt($("#height").value) || 480;
    const factor = 1 + pct / 100;

    const MIN_W = 256, MAX_W = 1920;
    const MIN_H = 256, MAX_H = 1080;
    const STEP = 16;

    let newW = Math.round(curW * factor / STEP) * STEP;
    let newH = Math.round(curH * factor / STEP) * STEP;

    newW = Math.max(MIN_W, Math.min(MAX_W, newW));
    newH = Math.max(MIN_H, Math.min(MAX_H, newH));

    const aspect = curW / curH;
    if (newW === MIN_W || newW === MAX_W) {
      newH = Math.round(newW / aspect / STEP) * STEP;
      newH = Math.max(MIN_H, Math.min(MAX_H, newH));
    } else if (newH === MIN_H || newH === MAX_H) {
      newW = Math.round(newH * aspect / STEP) * STEP;
      newW = Math.max(MIN_W, Math.min(MAX_W, newW));
    }

    setParam("width", newW);
    setParam("height", newH);
    onParamChange("width");
    onParamChange("height");
    updateResDimLabel();
  }

  function setupResScaleButtons() {
    document.querySelectorAll(".res-scale-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        scaleResolution(parseInt(btn.dataset.scale));
      });
    });
  }

  // ─── Public API ───────────────────────────────────────────────
  return {
    setupHybridInputs,
    setupLoraHybridInputs,
    setupResScaleButtons,
    setParam,
    onParamChange,
    calcFrames4k1,
    updateFramesFromDuration,
    updateInterpHint,
    updateTargetDurationHint,
    updateMegapixels,
    updateTotalSteps,
    updateResDimLabel,
  };
})();
