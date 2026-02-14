/* ── Wan 2.2 I2V — Web UI Frontend ──────────────────────────────── */

(() => {
  "use strict";

  // ─── State ──────────────────────────────────────────────────
  let currentSessionId = null;
  let sessions = [];
  let evtSource = null;
  let generationStartTime = null;
  let timerInterval = null;
  let autoRefreshInterval = null;
  let logPollInterval = null;
  let lastLogTs = 0;
  let stepProgressData = {};

  // ─── DOM refs ───────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const panelGen = $("#panel-generate");
  const panelSession = $("#panel-session");
  const sessionList = $("#session-list");
  const btnNew = $("#btn-new");
  const btnGenerate = $("#btn-generate");
  const btnCancel = $("#btn-cancel");
  const btnBack = $("#btn-back");
  const btnDelete = $("#btn-delete-session");
  const btnClone = $("#btn-clone");
  const btnTheme = $("#btn-theme");
  const dropZone = $("#drop-zone");
  const fileInput = $("#file-input");
  const previewImg = $("#preview-img");
  const dropText = $("#drop-text");

  // ─── Hybrid slider ↔ number sync ───────────────────────────
  // Each hybrid pair: range slider + number input stay in sync.
  // The number input allows precise typing; the slider gives quick drag.

  const HYBRID_PAIRS = [
    // [rangeSliderId, numberInputId]
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
  ];

  function setupHybridInputs() {
    for (const [rangeId, numId] of HYBRID_PAIRS) {
      const range = $(`#${rangeId}`);
      const num = $(`#${numId}`);
      if (!range || !num) continue;

      // Slider → number
      range.addEventListener("input", () => {
        num.value = range.value;
        onParamChange(rangeId);
      });

      // Number → slider
      num.addEventListener("input", () => {
        range.value = num.value;
        onParamChange(rangeId);
      });

      // On blur, clamp number to slider min/max
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

    // LoRA hybrid pairs
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

  // Helper to set both range + number input for a param
  function setParam(id, val) {
    const range = $(`#${id}`);
    const num = $(`#${id}-num`);
    if (range) range.value = val;
    if (num) num.value = val;
  }

  // Called when any parameter slider/number changes
  function onParamChange(id) {
    if (id === "duration" || id === "fps") updateFramesFromDuration();
    if (id === "fps" || id === "output-fps") updateInterpHint();
    if (id === "width" || id === "height") updateMegapixels();
    if (id === "steps") updateTotalSteps();
  }

  // ─── Computed values ────────────────────────────────────────

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
    const hint = $("#interp-hint");
    if (!hint) return;
    if (outputFps > fps) {
      const ratio = Math.round(outputFps / fps);
      hint.textContent = `RIFE: ${fps}→${outputFps} (~${ratio}×)`;
      hint.style.color = "var(--success)";
    } else {
      hint.textContent = `No interp (${fps}fps)`;
      hint.style.color = "var(--text-muted)";
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
    // Steps value is now total steps (split between high & low by boundary ratio)
    // No calculation needed — the UI value IS the total
  }

  // ─── Toast notification system ──────────────────────────────
  function toast(message, type = "info", duration = 4000) {
    const container = $("#toast-container");
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.classList.add("toast-out");
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  // ─── Theme toggle ──────────────────────────────────────────
  function initTheme() {
    const saved = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
    updateThemeButton(saved);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateThemeButton(next);
  }

  function updateThemeButton(theme) {
    btnTheme.textContent = theme === "dark" ? "🌙" : "☀️";
    btnTheme.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  }

  // ─── SSE ────────────────────────────────────────────────────
  function connectSSE() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource("/api/events");

    evtSource.addEventListener("connected", (e) => {
      console.log("SSE connected:", JSON.parse(e.data));
    });

    evtSource.addEventListener("progress", (e) => {
      handleProgress(JSON.parse(e.data));
    });

    evtSource.addEventListener("generation_complete", (e) => {
      handleComplete(JSON.parse(e.data));
    });

    evtSource.onerror = () => {
      console.warn("SSE error, reconnecting in 3s…");
      setTimeout(connectSSE, 3000);
    };
  }

  function handleProgress(d) {
    const pFill = $("#progress-fill");
    const pMsg = $("#progress-msg");
    const pSec = $("#progress-section");
    if (pFill && pSec) {
      pSec.style.display = "block";
      pFill.style.width = d.percent + "%";
      pMsg.textContent = `[${d.step}] ${d.message}  (${d.percent.toFixed(1)}%)`;
    }

    const card = $(`.step-card[data-step="${d.step}"]`);
    if (card) {
      card.setAttribute("data-state", "running");
      card.querySelector(".step-status").textContent = "running";
      stepProgressData[d.step] = d.percent;
      const fill = card.querySelector(".step-progress-fill");
      if (fill) fill.style.width = Math.min(100, d.percent) + "%";
    }

    if (d.vram && d.vram.allocated_gb !== undefined) {
      updateVRAM(d.vram);
      updateVRAMDetail(d.vram);
    }
  }

  function handleComplete(d) {
    stopTimer();
    stopAutoRefresh();
    stopSessionLogPolling();

    // Final fetch of session logs to capture summary
    fetchSessionLogs();

    const session = d.session || {};
    const status = session.status || "done";

    if (status === "done") toast("Generation complete! 🎉", "success");
    else if (status === "failed") toast("Generation failed: " + (session.error_message || "unknown error"), "error", 8000);
    else if (status === "cancelled") toast("Generation cancelled", "warning");

    refreshSessions().then(() => {
      if (d.session_id === currentSessionId) showSession(d.session_id);
    });
    btnCancel.style.display = "none";
    btnGenerate.disabled = false;
  }

  // ─── VRAM ───────────────────────────────────────────────────
  function updateVRAM(v) {
    const fill = $(".vram-fill");
    const label = $(".vram-label");
    if (!fill) return;
    const total = v.total_gb || 24;
    const used = v.allocated_gb || 0;
    const pct = Math.min(100, (used / total) * 100);
    fill.style.width = pct + "%";
    label.textContent = `VRAM: ${used.toFixed(1)} / ${total.toFixed(1)} GB`;
    if (pct > 85) fill.style.background = "linear-gradient(90deg, #f85149, #da3633)";
    else if (pct > 60) fill.style.background = "linear-gradient(90deg, #d29922, #e3b341)";
    else fill.style.background = "";
  }

  function updateVRAMDetail(v) {
    const total = v.total_gb || 24;
    const allocated = v.allocated_gb || 0;
    const reserved = v.reserved_gb || 0;
    const allocPct = Math.min(100, (allocated / total) * 100);
    const resvPct = Math.min(100, (reserved / total) * 100);
    const allocBar = $(".vram-detail-allocated");
    const resvBar = $(".vram-detail-reserved");
    if (allocBar) allocBar.style.width = allocPct + "%";
    if (resvBar) resvBar.style.width = resvPct + "%";
    const al = $(".vram-alloc-label");
    const rl = $(".vram-reserved-label");
    const tl = $(".vram-total-label");
    if (al) al.textContent = `Allocated: ${allocated.toFixed(2)} GB`;
    if (rl) rl.textContent = `Reserved: ${reserved.toFixed(2)} GB`;
    if (tl) tl.textContent = `Total: ${total.toFixed(1)} GB`;
  }

  function pollVRAM() {
    fetch("/api/vram").then(r => r.json()).then(v => {
      updateVRAM(v);
      updateVRAMDetail(v);
    }).catch(() => {});
  }

  // ─── Timer ──────────────────────────────────────────────────
  function startTimer() {
    generationStartTime = Date.now();
    const timerBar = $("#timer-bar");
    if (timerBar) timerBar.style.display = "flex";
    updateTimerDisplay();
    timerInterval = setInterval(updateTimerDisplay, 1000);
  }

  function stopTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = null;
  }

  function updateTimerDisplay() {
    if (!generationStartTime) return;
    const elapsed = Math.floor((Date.now() - generationStartTime) / 1000);
    const el = $("#timer-elapsed");
    if (el) el.textContent = `⏱ ${formatDuration(elapsed)}`;
    const pFill = $("#progress-fill");
    if (pFill) {
      const pct = parseFloat(pFill.style.width) || 0;
      if (pct > 5 && elapsed > 10) {
        const totalEst = (elapsed / pct) * 100;
        const remaining = Math.max(0, Math.floor(totalEst - elapsed));
        const etaEl = $("#timer-eta");
        if (etaEl) etaEl.textContent = `ETA: ~${formatDuration(remaining)}`;
      }
    }
  }

  function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  // ─── Auto-refresh ───────────────────────────────────────────
  function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshInterval = setInterval(() => {
      if (currentSessionId) {
        fetch(`/api/sessions/${currentSessionId}`)
          .then(r => r.json())
          .then(s => {
            if (s.session_id) renderSessionDetail(s);
            if (s.status !== "running") stopAutoRefresh();
          }).catch(() => {});
      }
    }, 3000);
  }

  function stopAutoRefresh() {
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = null;
  }

  // ─── Log viewer ─────────────────────────────────────────────
  function startLogPolling() {
    stopLogPolling();
    lastLogTs = 0;
    const logContent = $("#log-content");
    if (logContent) logContent.innerHTML = "";
    logPollInterval = setInterval(fetchLogs, 2000);
    fetchLogs();
  }

  function stopLogPolling() {
    if (logPollInterval) clearInterval(logPollInterval);
    logPollInterval = null;
  }

  function fetchLogs() {
    fetch(`/api/logs?since=${lastLogTs}&n=50`)
      .then(r => r.json())
      .then(lines => {
        if (!lines.length) return;
        const logContent = $("#log-content");
        if (!logContent) return;
        for (const line of lines) {
          const span = document.createElement("span");
          span.className = "log-line";
          if (line.level === "WARNING") span.className += " log-line-warn";
          if (line.level === "ERROR") span.className += " log-line-error";
          span.textContent = line.msg;
          logContent.appendChild(span);
          logContent.appendChild(document.createTextNode("\n"));
          if (line.ts > lastLogTs) lastLogTs = line.ts;
        }
        logContent.scrollTop = logContent.scrollHeight;
      }).catch(() => {});
  }

  // ─── Session Log Files (generation.log / vram.log) ──────────
  let sessionLogPollInterval = null;

  function setupSessionLogTabs() {
    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("session-log-tab")) {
        const tab = e.target.getAttribute("data-logtab");
        $$(".session-log-tab").forEach(t => t.classList.toggle("active", t.getAttribute("data-logtab") === tab));
        $$(".session-log-content").forEach(c => c.classList.toggle("active", c.getAttribute("data-logtab") === tab));
      }
    });

    const refreshBtn = $("#btn-refresh-logs");
    if (refreshBtn) refreshBtn.addEventListener("click", () => fetchSessionLogs());
  }

  function fetchSessionLogs() {
    if (!currentSessionId) return;

    fetch(`/api/sessions/${currentSessionId}/generation_log`)
      .then(r => r.json())
      .then(d => {
        const el = $("#gen-log-content");
        if (el) {
          el.textContent = d.content || "(empty)";
          // Auto-scroll to bottom if user hasn't scrolled up
          if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
            el.scrollTop = el.scrollHeight;
          }
        }
        // Show section if log exists
        const sec = $("#session-logs-section");
        if (sec && d.exists) sec.style.display = "block";
      }).catch(() => {});

    fetch(`/api/sessions/${currentSessionId}/vram_log`)
      .then(r => r.json())
      .then(d => {
        const el = $("#vram-log-content");
        if (el) {
          el.textContent = d.content || "(empty)";
          if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
            el.scrollTop = el.scrollHeight;
          }
        }
        const sec = $("#session-logs-section");
        if (sec && d.exists) sec.style.display = "block";
      }).catch(() => {});
  }

  function startSessionLogPolling() {
    stopSessionLogPolling();
    fetchSessionLogs();
    sessionLogPollInterval = setInterval(fetchSessionLogs, 5000);
  }

  function stopSessionLogPolling() {
    if (sessionLogPollInterval) clearInterval(sessionLogPollInterval);
    sessionLogPollInterval = null;
  }

  function updateSessionLogDownloadLinks(sid) {
    const genDl = $("#btn-dl-gen-log");
    const vramDl = $("#btn-dl-vram-log");
    if (genDl) {
      genDl.href = `/sessions/${sid}/generation.log`;
      genDl.download = `${sid.slice(0, 8)}_generation.log`;
    }
    if (vramDl) {
      vramDl.href = `/sessions/${sid}/vram.log`;
      vramDl.download = `${sid.slice(0, 8)}_vram.log`;
    }
  }

  // ─── Sessions ───────────────────────────────────────────────
  async function refreshSessions() {
    try {
      const r = await fetch("/api/sessions");
      sessions = await r.json();
      renderSessionList();
    } catch (e) { console.error("Failed to refresh sessions:", e); }
  }

  function renderSessionList() {
    sessionList.innerHTML = "";
    for (const s of sessions) {
      const div = document.createElement("div");
      div.className = "session-item" + (s.session_id === currentSessionId ? " active" : "");
      const date = new Date(s.created_at * 1000);
      const dateStr = date.toLocaleString();
      const prompt = s.prompt || "(no prompt)";
      const thumbUrl = `/sessions/${s.session_id}/input.png`;
      const thumbHtml = `<img class="si-thumb" src="${thumbUrl}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="si-thumb-placeholder" style="display:none">🎬</div>`;
      div.innerHTML = `
        ${thumbHtml}
        <div class="si-details">
          <div class="si-top">
            <span class="si-id">${s.session_id.slice(0, 8)}</span>
            <span class="si-status ${s.status}">${s.status}</span>
          </div>
          <div class="si-prompt" title="${escapeHtml(prompt)}">${escapeHtml(prompt.slice(0, 50))}</div>
          <div class="si-date">${dateStr}</div>
        </div>
      `;
      div.addEventListener("click", () => showSession(s.session_id));
      sessionList.appendChild(div);
    }
  }

  async function showSession(sid) {
    currentSessionId = sid;
    renderSessionList();
    try {
      const r = await fetch(`/api/sessions/${sid}`);
      if (!r.ok) return;
      const s = await r.json();
      renderSessionDetail(s);
      panelGen.style.display = "none";
      panelSession.style.display = "block";

      // Update log download links and fetch logs
      updateSessionLogDownloadLinks(sid);
      fetchSessionLogs();

      if (s.status === "running") {
        startAutoRefresh();
        startLogPolling();
        startSessionLogPolling();
        startTimer();
      } else {
        stopAutoRefresh();
        stopLogPolling();
        stopSessionLogPolling();
        stopTimer();
      }
    } catch (e) { console.error(e); }
  }

  function renderSessionDetail(s) {
    $("#session-title").textContent = `Session ${s.session_id.slice(0, 8)}`;

    const steps = s.steps || {};
    $$(".step-card").forEach(card => {
      const step = card.getAttribute("data-step");
      const status = steps[step] || "pending";
      card.setAttribute("data-state", status);
      card.querySelector(".step-status").textContent = status;
      const fill = card.querySelector(".step-progress-fill");
      if (fill) {
        if (status === "done" || status === "skipped") fill.style.width = "100%";
        else if (status === "running" && stepProgressData[step] !== undefined) fill.style.width = Math.min(100, stepProgressData[step]) + "%";
        else if (status === "pending") fill.style.width = "0%";
      }
      const timeEl = card.querySelector(".step-time");
      if (s.step_times && s.step_times[step]) timeEl.textContent = formatStepTime(s.step_times[step]);
      else timeEl.textContent = "";

      // Ensure upscale model dropdown reflects session state ONLY if not focused
      // Prevents overwriting user selection while they're changing it
      if (step === "upscale" && s.upscale_model) {
        const uSelect = card.querySelector("select");
        if (uSelect && document.activeElement !== uSelect) {
            uSelect.value = s.upscale_model;
        }
      }
    });

    $("#info-prompt").textContent = s.prompt || "—";
    const durStr = s.duration ? `${s.duration}s` : "";
    const outFpsStr = s.output_fps && s.output_fps !== s.fps ? ` → ${s.output_fps}fps` : "";
    $("#info-size").textContent = `${s.width}×${s.height}, ${s.num_frames}f ${durStr} @ ${s.fps}fps${outFpsStr}`;
    const modeStr = s.distill_lora_mode ? "⚡ Fast" : "🎨 Quality";
    $("#info-steps").textContent = `${s.num_inference_steps} (${modeStr})`;
    $("#info-seed").textContent = s.seed;
    $("#info-cfg").textContent = `${s.guidance_scale} / ${s.guidance_scale_2 || "—"}`;
    $("#info-status").textContent = s.status;

    const errorRow = $("#info-error-row");
    const errorEl = $("#info-error");
    if (s.error_message) {
      errorRow.style.display = "";
      errorEl.textContent = s.error_message;
    } else {
      errorRow.style.display = "none";
    }

    if (s.status === "running") {
      $("#progress-section").style.display = "block";
      btnCancel.style.display = "inline-flex";
      $("#timer-bar").style.display = "flex";
    } else {
      $("#progress-section").style.display = "none";
      btnCancel.style.display = "none";
      $("#timer-bar").style.display = "none";
    }

    const inputImgSection = $("#input-image-section");
    const sessionImg = $("#session-input-img");
    const inputImgUrl = `/sessions/${s.session_id}/input.png`;
    sessionImg.src = inputImgUrl;
    sessionImg.onerror = () => { inputImgSection.style.display = "none"; };
    sessionImg.onload = () => { inputImgSection.style.display = "block"; };

    renderVideoSection(s);
  }

  function renderVideoSection(s) {
    const videoSec = $("#video-section");
    const hasOutput = (s.checkpoints || []).includes("output.mp4");
    const hasPreview = (s.checkpoints || []).includes("preview.mp4");
    const hasUpscaled = (s.checkpoints || []).includes("output_upscaled.mp4");

    if (!hasOutput && !hasPreview) {
      videoSec.style.display = "none";
      return;
    }

    videoSec.style.display = "block";
    const videoFile = hasOutput ? "output.mp4" : "preview.mp4";
    const origSrc = `/sessions/${s.session_id}/${videoFile}`;
    const upSrc = `/sessions/${s.session_id}/output_upscaled.mp4`;

    $("#video-player").src = origSrc;
    const dlBtn = $("#video-download");
    dlBtn.href = origSrc;
    dlBtn.download = `${s.session_id.slice(0,8)}_${videoFile}`;

    const tabUp = $("#tab-upscaled");
    const tabCompare = $("#tab-compare");
    if (hasUpscaled) {
      tabUp.style.display = "";
      tabCompare.style.display = "";
      $("#video-player-upscaled").src = upSrc;
      const dlUp = $("#video-download-upscaled");
      dlUp.href = upSrc;
      dlUp.download = `${s.session_id.slice(0,8)}_upscaled.mp4`;
      $("#video-compare-orig").src = origSrc;
      $("#video-compare-up").src = upSrc;
    } else {
      tabUp.style.display = "none";
      tabCompare.style.display = "none";
      activateVideoTab("original");
    }
  }

  function activateVideoTab(tabName) {
    $$(".video-tab").forEach(t => t.classList.toggle("active", t.getAttribute("data-tab") === tabName));
    $$(".video-tab-content").forEach(c => c.classList.toggle("active", c.getAttribute("data-tab") === tabName));
  }

  function formatStepTime(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + "s";
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(0);
    return `${m}m ${s}s`;
  }

  // ─── Image Upload ───────────────────────────────────────────
  function setupDropZone() {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });
  }

  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("image", file);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const d = await r.json();
      if (d.path) {
        $("#input-image-path").value = d.path;
        previewImg.src = `/input/${d.filename}`;
        previewImg.style.display = "block";
        dropText.style.display = "none";
        toast("Image uploaded", "success", 2000);
      } else {
        toast("Upload failed: " + (d.error || "unknown"), "error");
      }
    } catch (e) {
      toast("Upload error: " + e.message, "error");
    }
  }

  // ─── Clone session ─────────────────────────────────────────
  async function cloneSession() {
    if (!currentSessionId) return;
    try {
      const r = await fetch(`/api/sessions/${currentSessionId}/clone`, { method: "POST" });
      const d = await r.json();
      if (d.error) { toast("Clone failed: " + d.error, "error"); return; }

      $("#prompt").value = d.prompt || "";
      $("#negative-prompt").value = d.negative_prompt || "";
      setParam("width", d.width || 832);
      setParam("height", d.height || 480);
      setParam("duration", d.duration || 5.0);
      setParam("fps", d.fps || 16);
      setParam("steps", d.num_inference_steps || 20);
      setParam("cfg", d.guidance_scale || 5.0);
      setParam("cfg2", d.guidance_scale_2 || 5.0);
      setParam("flow-shift", d.flow_shift || 8.0);
      setParam("output-fps", d.output_fps || 24);
      setParam("boundary-ratio", d.boundary_ratio || 0.9);
      $("#seed").value = d.seed || 42;
      $("#enable-upscale").checked = d.enable_upscale || false;
      if (d.upscale_model && $("#upscale-model")) {
        $("#upscale-model").value = d.upscale_model;
      }

      // Restore mode toggle based on session's distill_lora_mode
      applyMode(d.distill_lora_mode ? "fast" : "quality", /* silent */ true);

      updateFramesFromDuration();
      updateInterpHint();
      updateMegapixels();
      updateTotalSteps();

      if (d.lora_scales && d.lora_scales.length) {
        const loraRanges = $$(".lora-scale");
        const loraNums = $$(".lora-scale-num");
        d.lora_scales.forEach((val, i) => {
          if (loraRanges[i]) loraRanges[i].value = val;
          if (loraNums[i]) loraNums[i].value = val;
        });
      }

      if (d.input_image_url) {
        previewImg.src = d.input_image_url;
        previewImg.style.display = "block";
        dropText.style.display = "none";
        $("#input-image-path").value = "";
        toast("Parameters cloned — please upload/confirm input image", "info", 5000);
      }

      showGeneratePanel();
      toast("Session parameters cloned", "success");
    } catch (e) {
      toast("Clone error: " + e.message, "error");
    }
  }

  // ─── Mode Toggle (Quality / Fast) ──────────────────────────
  // Quality mode: no distill LoRAs, full CFG, 20 steps
  // Fast mode: LightX2V distill LoRAs, CFG=1.0, 4 steps, flow_shift=5.0
  let currentMode = "quality";

  // Presets store the values that were active before switching
  const QUALITY_DEFAULTS = { steps: 20, cfg: 5.0, cfg2: 5.0, flowShift: 8.0, boundaryRatio: 0.9 };
  const FAST_DEFAULTS    = { steps: 4,  cfg: 1.0, cfg2: 1.0, flowShift: 5.0, boundaryRatio: 0.9 };

  function setupModeToggle() {
    const qualBtn = $("#mode-quality");
    const fastBtn = $("#mode-fast");
    if (!qualBtn || !fastBtn) return;

    qualBtn.addEventListener("click", () => applyMode("quality"));
    fastBtn.addEventListener("click", () => applyMode("fast"));

    // Initial state: quality mode, disable distill LoRA cards
    applyMode("quality", /* silent */ true);
  }

  function applyMode(mode, silent) {
    currentMode = mode;
    const qualBtn = $("#mode-quality");
    const fastBtn = $("#mode-fast");
    const hint = $("#mode-hint");

    // Toggle active buttons
    if (qualBtn) qualBtn.classList.toggle("active", mode === "quality");
    if (fastBtn) fastBtn.classList.toggle("active", mode === "fast");

    // Update hint text
    if (hint) {
      if (mode === "quality") {
        hint.textContent = "Quality mode: 20 steps, CFG=5.0, quality LoRAs only";
      } else {
        hint.textContent = "Fast mode: 4 steps, no CFG (baked in), LightX2V distill — ~10× faster";
      }
    }

    // Apply preset values
    const preset = mode === "fast" ? FAST_DEFAULTS : QUALITY_DEFAULTS;
    setParam("steps", preset.steps);
    setParam("cfg", preset.cfg);
    setParam("cfg2", preset.cfg2);
    setParam("flow-shift", preset.flowShift);
    setParam("boundary-ratio", preset.boundaryRatio);

    // Enable/disable distill LoRA cards
    $$(".lora-card").forEach(card => {
      if (card.dataset.role === "distill") {
        if (mode === "quality") {
          card.classList.add("lora-disabled");
        } else {
          card.classList.remove("lora-disabled");
        }
      }
    });

    // Update computed displays
    updateTotalSteps();

    if (!silent) {
      toast(mode === "quality"
        ? "🎨 Quality mode — distill LoRAs disabled, full CFG"
        : "⚡ Fast mode — LightX2V distill, CFG=1.0, 4 steps",
        "info", 3000);
    }
  }

  // ─── Generate ───────────────────────────────────────────────
  function gatherParams() {
    const loraRanges = $$(".lora-scale");
    const loraScales = [];
    loraRanges.forEach(inp => loraScales.push(parseFloat(inp.value) || 0));

    const duration = parseFloat($("#duration").value) || 5;
    const fps = parseInt($("#fps").value) || 16;
    const numFrames = calcFrames4k1(duration, fps);

    return {
      prompt:              $("#prompt").value,
      negative_prompt:     $("#negative-prompt").value,
      input_image:         $("#input-image-path").value,
      width:               parseInt($("#width").value),
      height:              parseInt($("#height").value),
      num_frames:          numFrames,
      fps:                 fps,
      duration:            duration,
      output_fps:          parseInt($("#output-fps").value) || fps,
      num_inference_steps: parseInt($("#steps").value),
      guidance_scale:      parseFloat($("#cfg").value),
      guidance_scale_2:    parseFloat($("#cfg2").value),
      flow_shift:          parseFloat($("#flow-shift").value),
      boundary_ratio:      parseFloat($("#boundary-ratio").value),
      seed:                parseInt($("#seed").value),
      enable_upscale:      $("#enable-upscale").checked,
      upscale_model:       $("#upscale-model") ? $("#upscale-model").value : "",
      lora_scales:         loraScales,
      distill_lora_mode:   currentMode === "fast",
      // Memory settings
      offload_type:        $("#offload-type") ? $("#offload-type").value : "block_level",
      num_blocks_per_group: parseInt($("#num-blocks-per-group")?.value) || 1,
      enable_group_offload: $("#enable-group-offload")?.checked ?? true,
      vae_tiling:          $("#vae-tiling")?.checked ?? true,
      vae_slicing:         $("#vae-slicing")?.checked ?? true,
      force_vae_cpu:       $("#force-vae-cpu")?.checked ?? false,
    };
  }

  async function startGeneration() {
    const params = gatherParams();
    if (!params.input_image) {
      toast("Please upload an input image first.", "warning");
      return;
    }
    if (!params.prompt.trim()) {
      toast("Please enter a prompt.", "warning");
      return;
    }

    btnGenerate.disabled = true;
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (d.error) {
        toast("Error: " + d.error, "error");
        btnGenerate.disabled = false;
        return;
      }

      toast("Generation started!", "info");
      stepProgressData = {};
      currentSessionId = d.session_id;
      await refreshSessions();
      showSession(d.session_id);
      startTimer();
      startAutoRefresh();
      startLogPolling();
      startSessionLogPolling();
    } catch (e) {
      toast("Request failed: " + e.message, "error");
      btnGenerate.disabled = false;
    }
  }

  async function resumeFromStep(step) {
    if (!currentSessionId) return;
    const stepOrder = ["encode", "denoise", "vae_decode", "export", "upscale"];
    const stepIdx = stepOrder.indexOf(step);
    const downstreamSteps = stepOrder.slice(stepIdx).join(", ");
    const confirmed = confirm(
      `Resume from "${step}"?\n\nThis will re-run: ${downstreamSteps}\nAny existing checkpoints for these steps will be overwritten.`
    );
    if (!confirmed) return;
    try {
      // Always send current UI state for enable_upscale so the session
      // gets updated — covers the case where the user forgot to check
      // upscale initially but wants it on resume.
      const payload = { from_step: step };
      const upscaleCheckbox = $("#enable-upscale");

      // If resuming specifically from "upscale", force it on — the user
      // clearly wants to upscale if they clicked play on that step.
      if (step === "upscale") {
        payload.enable_upscale = true;
        if (upscaleCheckbox) upscaleCheckbox.checked = true;
      } else if (upscaleCheckbox) {
        payload.enable_upscale = upscaleCheckbox.checked;
      }

      // Also send the current upscale model selection
      // Priority: Dropdown in the upscale step card > Main panel dropdown
      const stepUpscaleSelect = $('.step-card[data-step="upscale"] select');
      const mainUpscaleSelect = $("#upscale-model");
      
      if (stepUpscaleSelect) {
        payload.upscale_model = stepUpscaleSelect.value;
      } else if (mainUpscaleSelect) {
        payload.upscale_model = mainUpscaleSelect.value;
      }

      const r = await fetch(`/api/resume/${currentSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) { toast("Error: " + d.error, "error"); return; }
      toast(`Resuming from ${step}…`, "info");
      stepProgressData = {};
      btnCancel.style.display = "inline-flex";
      startTimer();
      startAutoRefresh();
      startLogPolling();
      startSessionLogPolling();
      showSession(currentSessionId);
    } catch (e) {
      toast("Resume failed: " + e.message, "error");
    }
  }

  async function cancelGeneration() {
    try { await fetch("/api/cancel", { method: "POST" }); toast("Cancel requested…", "warning"); }
    catch (e) { console.error(e); }
  }

  async function deleteSession() {
    if (!currentSessionId) return;
    if (!confirm("Delete this session and all its files?")) return;
    try {
      await fetch(`/api/sessions/${currentSessionId}`, { method: "DELETE" });
      toast("Session deleted", "info");
      currentSessionId = null;
      await refreshSessions();
      showGeneratePanel();
    } catch (e) {
      toast("Delete failed: " + e.message, "error");
    }
  }

  // ─── Navigation ─────────────────────────────────────────────
  function showGeneratePanel() {
    panelGen.style.display = "block";
    panelSession.style.display = "none";
    currentSessionId = null;
    stopAutoRefresh();
    stopLogPolling();
    stopSessionLogPolling();
    stopTimer();
    renderSessionList();
    const logSec = $("#session-logs-section");
    if (logSec) logSec.style.display = "none";
  }

  // ─── Keyboard shortcuts ─────────────────────────────────────
  function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      const tag = e.target.tagName.toLowerCase();
      const isInput = tag === "input" || tag === "textarea" || tag === "select";

      if (e.key === "Enter" && !isInput && panelGen.style.display !== "none") {
        e.preventDefault();
        startGeneration();
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        cancelGeneration();
        return;
      }
      if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        showGeneratePanel();
        return;
      }
    });
  }

  // ─── Video tab switching ────────────────────────────────────
  function setupVideoTabs() {
    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("video-tab")) {
        const tab = e.target.getAttribute("data-tab");
        activateVideoTab(tab);
        if (tab === "compare") {
          const orig = $("#video-compare-orig");
          const up = $("#video-compare-up");
          if (orig && up) { orig.currentTime = 0; up.currentTime = 0; }
        }
      }
    });
  }

  // ─── Helpers ────────────────────────────────────────────────
  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ─── Init ───────────────────────────────────────────────────
  function init() {
    initTheme();
    connectSSE();
    setupDropZone();
    setupHybridInputs();
    setupModeToggle();
    setupKeyboardShortcuts();
    setupVideoTabs();
    setupSessionLogTabs();

    // Initialize computed values
    updateFramesFromDuration();
    updateInterpHint();
    updateMegapixels();
    updateTotalSteps();

    btnNew.addEventListener("click", showGeneratePanel);
    btnGenerate.addEventListener("click", startGeneration);
    btnCancel.addEventListener("click", cancelGeneration);
    btnBack.addEventListener("click", showGeneratePanel);
    btnDelete.addEventListener("click", deleteSession);
    btnClone.addEventListener("click", cloneSession);
    btnTheme.addEventListener("click", toggleTheme);

    $$(".btn-resume").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        resumeFromStep(btn.getAttribute("data-step"));
      });
    });

    pollVRAM();
    setInterval(pollVRAM, 10000);
    refreshSessions();
    setInterval(refreshSessions, 15000);

    fetch("/api/active").then(r => r.json()).then(d => {
      if (d.running && d.session_id) {
        currentSessionId = d.session_id;
        showSession(d.session_id);
      }
    }).catch(() => {});
  }

  document.addEventListener("DOMContentLoaded", init);
})();
