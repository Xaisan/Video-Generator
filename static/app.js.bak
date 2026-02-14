/* ── Wan 2.2 I2V — Web UI Frontend ──────────────────────────────── */

(() => {
  "use strict";

  // ─── State ──────────────────────────────────────────────────
  let currentSessionId = null;
  let sessions = [];
  let evtSource = null;
  let generationStartTime = null;      // epoch ms when generation started
  let timerInterval = null;             // setInterval for elapsed timer
  let autoRefreshInterval = null;       // polling during running session
  let logPollInterval = null;           // log polling interval
  let lastLogTs = 0;                    // last log timestamp fetched
  let stepProgressData = {};            // { step: pct } for per-step mini-bars

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

  // ─── Slider value display + duration↔frames ─────────────────
  function setupSliders() {
    // Map: slider id → value display span id
    const sliderMap = {
      "width": "width-val",
      "height": "height-val",
      "duration": "duration-val",
      "fps": "fps-val",
      "steps": "steps-val",
      "cfg": "cfg-val",
      "cfg2": "cfg2-val",
      "flow-shift": "flow-shift-val",
      "output-fps": "output-fps-val",
    };

    for (const [sliderId, labelId] of Object.entries(sliderMap)) {
      const slider = $(`#${sliderId}`);
      const label = $(`#${labelId}`);
      if (slider && label) {
        slider.addEventListener("input", () => {
          label.textContent = slider.value;
          // Recalculate frames when duration or fps change
          if (sliderId === "duration" || sliderId === "fps") {
            updateFramesFromDuration();
          }
          if (sliderId === "fps" || sliderId === "output-fps") {
            updateInterpHint();
          }
        });
      }
    }

    // LoRA slider value displays
    $$(".lora-scale").forEach(slider => {
      slider.addEventListener("input", () => {
        const valSpan = slider.parentElement.querySelector(".lora-val");
        if (valSpan) valSpan.textContent = slider.value;
      });
    });

    // Initialize computed values
    updateFramesFromDuration();
    updateInterpHint();
  }

  function calcFrames4k1(duration, fps) {
    // Wan 2.2 frame count rule: frames must be 4k+1
    // frames = round(duration * fps), then snap to nearest 4k+1
    let raw = Math.round(duration * fps);
    if (raw < 5) raw = 5;
    // Snap to 4k+1: frames = ((raw - 1) / 4) * 4 + 1
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
      hint.textContent = `RIFE: ${fps}→${outputFps}fps (~${ratio}×)`;
      hint.style.color = "var(--success)";
    } else {
      hint.textContent = `No interpolation (${fps}fps)`;
      hint.style.color = "var(--text-muted)";
    }
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
      const d = JSON.parse(e.data);
      handleProgress(d);
    });

    evtSource.addEventListener("generation_complete", (e) => {
      const d = JSON.parse(e.data);
      handleComplete(d);
    });

    evtSource.onerror = () => {
      console.warn("SSE error, reconnecting in 3s…");
      setTimeout(connectSSE, 3000);
    };
  }

  function handleProgress(d) {
    // Update progress bar
    const pFill = $("#progress-fill");
    const pMsg = $("#progress-msg");
    const pSec = $("#progress-section");
    if (pFill && pSec) {
      pSec.style.display = "block";
      pFill.style.width = d.percent + "%";
      pMsg.textContent = `[${d.step}] ${d.message}  (${d.percent.toFixed(1)}%)`;
    }

    // Update step card
    const card = $(`.step-card[data-step="${d.step}"]`);
    if (card) {
      card.setAttribute("data-state", "running");
      card.querySelector(".step-status").textContent = "running";

      // Per-step mini progress bar
      stepProgressData[d.step] = d.percent;
      const fill = card.querySelector(".step-progress-fill");
      if (fill) {
        // Map the overall percent range to step-local 0-100
        fill.style.width = Math.min(100, d.percent) + "%";
      }
    }

    // Update VRAM bars (sidebar + detail)
    if (d.vram && d.vram.allocated_gb !== undefined) {
      updateVRAM(d.vram);
      updateVRAMDetail(d.vram);
    }
  }

  function handleComplete(d) {
    stopTimer();
    stopAutoRefresh();

    const session = d.session || {};
    const status = session.status || "done";

    if (status === "done") {
      toast("Generation complete! 🎉", "success");
    } else if (status === "failed") {
      toast("Generation failed: " + (session.error_message || "unknown error"), "error", 8000);
    } else if (status === "cancelled") {
      toast("Generation cancelled", "warning");
    }

    // Refresh session list and detail
    refreshSessions().then(() => {
      if (d.session_id === currentSessionId) {
        showSession(d.session_id);
      }
    });
    btnCancel.style.display = "none";
    btnGenerate.disabled = false;
  }

  // ─── VRAM (sidebar) ────────────────────────────────────────
  function updateVRAM(v) {
    const fill = $(".vram-fill");
    const label = $(".vram-label");
    if (!fill) return;
    const total = v.total_gb || 24;
    const used = v.allocated_gb || 0;
    const pct = Math.min(100, (used / total) * 100);
    fill.style.width = pct + "%";
    label.textContent = `VRAM: ${used.toFixed(1)} / ${total.toFixed(1)} GB`;

    // Color code
    if (pct > 85) fill.style.background = "linear-gradient(90deg, #f85149, #da3633)";
    else if (pct > 60) fill.style.background = "linear-gradient(90deg, #d29922, #e3b341)";
    else fill.style.background = "";
  }

  // ─── VRAM detail bar (in session view) ──────────────────────
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

  // ─── Timer (elapsed + ETA) ─────────────────────────────────
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
    const elapsedStr = formatDuration(elapsed);
    const el = $("#timer-elapsed");
    if (el) el.textContent = `⏱ ${elapsedStr}`;

    // Estimate ETA from overall progress fill
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

  // ─── Auto-refresh during running session ────────────────────
  function startAutoRefresh() {
    stopAutoRefresh();
    autoRefreshInterval = setInterval(() => {
      if (currentSessionId) {
        fetch(`/api/sessions/${currentSessionId}`)
          .then(r => r.json())
          .then(s => {
            if (s.session_id) renderSessionDetail(s);
            if (s.status !== "running") {
              stopAutoRefresh();
            }
          })
          .catch(() => {});
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
    fetchLogs(); // immediate first fetch
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

        // Auto-scroll to bottom
        logContent.scrollTop = logContent.scrollHeight;
      })
      .catch(() => {});
  }

  // ─── Sessions ───────────────────────────────────────────────
  async function refreshSessions() {
    try {
      const r = await fetch("/api/sessions");
      sessions = await r.json();
      renderSessionList();
    } catch (e) {
      console.error("Failed to refresh sessions:", e);
    }
  }

  function renderSessionList() {
    sessionList.innerHTML = "";
    for (const s of sessions) {
      const div = document.createElement("div");
      div.className = "session-item" + (s.session_id === currentSessionId ? " active" : "");
      const date = new Date(s.created_at * 1000);
      const dateStr = date.toLocaleString();
      const prompt = s.prompt || "(no prompt)";

      // Thumbnail: check if input.png exists in session
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

      // Start auto-refresh and log polling if running
      if (s.status === "running") {
        startAutoRefresh();
        startLogPolling();
        startTimer();
      } else {
        stopAutoRefresh();
        stopLogPolling();
        stopTimer();
      }
    } catch (e) {
      console.error(e);
    }
  }

  function renderSessionDetail(s) {
    $("#session-title").textContent = `Session ${s.session_id.slice(0, 8)}`;

    // Steps
    const steps = s.steps || {};
    $$(".step-card").forEach(card => {
      const step = card.getAttribute("data-step");
      const status = steps[step] || "pending";
      card.setAttribute("data-state", status);
      card.querySelector(".step-status").textContent = status;

      // Per-step mini progress: set to 100% if done, 0% if pending
      const fill = card.querySelector(".step-progress-fill");
      if (fill) {
        if (status === "done" || status === "skipped") {
          fill.style.width = "100%";
        } else if (status === "running" && stepProgressData[step] !== undefined) {
          fill.style.width = Math.min(100, stepProgressData[step]) + "%";
        } else if (status === "pending") {
          fill.style.width = "0%";
        }
      }

      // Show timing if available
      const timeEl = card.querySelector(".step-time");
      if (s.step_times && s.step_times[step]) {
        timeEl.textContent = formatStepTime(s.step_times[step]);
      } else {
        timeEl.textContent = "";
      }
    });

    // Info
    $("#info-prompt").textContent = s.prompt || "—";
    const durStr = s.duration ? `${s.duration}s` : "";
    const outFpsStr = s.output_fps && s.output_fps !== s.fps ? ` → ${s.output_fps}fps` : "";
    $("#info-size").textContent = `${s.width}×${s.height}, ${s.num_frames}f ${durStr} @ ${s.fps}fps${outFpsStr}`;
    $("#info-steps").textContent = s.num_inference_steps;
    $("#info-seed").textContent = s.seed;
    $("#info-cfg").textContent = `${s.guidance_scale} / ${s.guidance_scale_2 || "—"}`;
    $("#info-status").textContent = s.status;

    // Error
    const errorRow = $("#info-error-row");
    const errorEl = $("#info-error");
    if (s.error_message) {
      errorRow.style.display = "";
      errorEl.textContent = s.error_message;
    } else {
      errorRow.style.display = "none";
    }

    // Progress
    if (s.status === "running") {
      $("#progress-section").style.display = "block";
      btnCancel.style.display = "inline-flex";
      $("#timer-bar").style.display = "flex";
    } else {
      $("#progress-section").style.display = "none";
      btnCancel.style.display = "none";
      $("#timer-bar").style.display = "none";
    }

    // Input image in session detail
    const inputImgSection = $("#input-image-section");
    const sessionImg = $("#session-input-img");
    const inputImgUrl = `/sessions/${s.session_id}/input.png`;
    sessionImg.src = inputImgUrl;
    sessionImg.onerror = () => { inputImgSection.style.display = "none"; };
    sessionImg.onload = () => { inputImgSection.style.display = "block"; };

    // Video section with tabs
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

    // Original player
    const player = $("#video-player");
    player.src = origSrc;
    const dlBtn = $("#video-download");
    dlBtn.href = origSrc;
    dlBtn.download = `${s.session_id.slice(0,8)}_${videoFile}`;

    // Upscaled tab + player
    const tabUp = $("#tab-upscaled");
    const tabCompare = $("#tab-compare");
    if (hasUpscaled) {
      tabUp.style.display = "";
      tabCompare.style.display = "";

      const playerUp = $("#video-player-upscaled");
      playerUp.src = upSrc;
      const dlUp = $("#video-download-upscaled");
      dlUp.href = upSrc;
      dlUp.download = `${s.session_id.slice(0,8)}_upscaled.mp4`;

      // Comparison videos
      $("#video-compare-orig").src = origSrc;
      $("#video-compare-up").src = upSrc;
    } else {
      tabUp.style.display = "none";
      tabCompare.style.display = "none";
      // Reset to original tab if on hidden tab
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
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
      }
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
      if (d.error) {
        toast("Clone failed: " + d.error, "error");
        return;
      }

      // Fill the generate form with cloned params
      $("#prompt").value = d.prompt || "";
      $("#negative-prompt").value = d.negative_prompt || "";

      // Update sliders + their value labels
      function setSlider(id, val) {
        const el = $(`#${id}`);
        if (el) el.value = val;
        const label = $(`#${id}-val`);
        if (label) label.textContent = val;
      }
      setSlider("width", d.width || 832);
      setSlider("height", d.height || 480);
      setSlider("duration", d.duration || 5.0);
      setSlider("fps", d.fps || 16);
      setSlider("steps", d.num_inference_steps || 8);
      setSlider("cfg", d.guidance_scale || 2.0);
      setSlider("cfg2", d.guidance_scale_2 || 2.0);
      setSlider("flow-shift", d.flow_shift || 8.0);
      setSlider("output-fps", d.output_fps || 24);
      $("#seed").value = d.seed || 42;
      $("#enable-upscale").checked = d.enable_upscale || false;

      // Update computed values
      updateFramesFromDuration();
      updateInterpHint();

      // Update LoRA scales
      if (d.lora_scales && d.lora_scales.length) {
        const loraInputs = $$(".lora-scale");
        d.lora_scales.forEach((val, i) => {
          if (loraInputs[i]) {
            loraInputs[i].value = val;
            const valSpan = loraInputs[i].parentElement.querySelector(".lora-val");
            if (valSpan) valSpan.textContent = val;
          }
        });
      }

      // If there's an input image URL, show it
      if (d.input_image_url) {
        previewImg.src = d.input_image_url;
        previewImg.style.display = "block";
        dropText.style.display = "none";
        // Note: input_image_path needs to be set from the original session
        // For clone, we keep the path empty — user needs to re-upload or
        // we copy the image from the session
        $("#input-image-path").value = "";
        toast("Parameters cloned — please upload/confirm input image", "info", 5000);
      }

      showGeneratePanel();
      toast("Session parameters cloned", "success");
    } catch (e) {
      toast("Clone error: " + e.message, "error");
    }
  }

  // ─── Generate ───────────────────────────────────────────────
  function gatherParams() {
    const loraInputs = $$(".lora-scale");
    const loraScales = [];
    loraInputs.forEach(inp => {
      loraScales.push(parseFloat(inp.value) || 0);
    });

    const duration = parseFloat($("#duration").value) || 5;
    const fps = parseInt($("#fps").value) || 16;
    const numFrames = calcFrames4k1(duration, fps);

    return {
      prompt: $("#prompt").value,
      negative_prompt: $("#negative-prompt").value,
      input_image: $("#input-image-path").value,
      width: parseInt($("#width").value),
      height: parseInt($("#height").value),
      num_frames: numFrames,
      fps: fps,
      duration: duration,
      output_fps: parseInt($("#output-fps").value) || fps,
      num_inference_steps: parseInt($("#steps").value),
      guidance_scale: parseFloat($("#cfg").value),
      guidance_scale_2: parseFloat($("#cfg2").value),
      flow_shift: parseFloat($("#flow-shift").value),
      seed: parseInt($("#seed").value),
      enable_upscale: $("#enable-upscale").checked,
      lora_scales: loraScales,
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

      // Switch to session view
      currentSessionId = d.session_id;
      await refreshSessions();
      showSession(d.session_id);
      startTimer();
      startAutoRefresh();
      startLogPolling();
    } catch (e) {
      toast("Request failed: " + e.message, "error");
      btnGenerate.disabled = false;
    }
  }

  async function resumeFromStep(step) {
    if (!currentSessionId) return;

    // Confirmation before overwrite — warn about downstream checkpoints
    const stepOrder = ["encode", "denoise", "vae_decode", "export", "upscale"];
    const stepIdx = stepOrder.indexOf(step);
    const downstreamSteps = stepOrder.slice(stepIdx).join(", ");
    const confirmed = confirm(
      `Resume from "${step}"?\n\n` +
      `This will re-run: ${downstreamSteps}\n` +
      `Any existing checkpoints for these steps will be overwritten.`
    );
    if (!confirmed) return;

    try {
      const r = await fetch(`/api/resume/${currentSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_step: step }),
      });
      const d = await r.json();
      if (d.error) {
        toast("Error: " + d.error, "error");
        return;
      }
      toast(`Resuming from ${step}…`, "info");
      stepProgressData = {};
      btnCancel.style.display = "inline-flex";
      startTimer();
      startAutoRefresh();
      startLogPolling();
      showSession(currentSessionId);
    } catch (e) {
      toast("Resume failed: " + e.message, "error");
    }
  }

  async function cancelGeneration() {
    try {
      await fetch("/api/cancel", { method: "POST" });
      toast("Cancel requested…", "warning");
    } catch (e) {
      console.error(e);
    }
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
    stopTimer();
    renderSessionList();
  }

  // ─── Keyboard shortcuts ─────────────────────────────────────
  function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      // Don't trigger shortcuts when typing in inputs
      const tag = e.target.tagName.toLowerCase();
      const isInput = tag === "input" || tag === "textarea" || tag === "select";

      // Enter to generate (only when generate panel is visible and not in input)
      if (e.key === "Enter" && !isInput && panelGen.style.display !== "none") {
        e.preventDefault();
        startGeneration();
        return;
      }

      // Escape to cancel
      if (e.key === "Escape") {
        e.preventDefault();
        cancelGeneration();
        return;
      }

      // Ctrl+N to show new generation panel
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

        // Sync playback for comparison
        if (tab === "compare") {
          const orig = $("#video-compare-orig");
          const up = $("#video-compare-up");
          if (orig && up) {
            orig.currentTime = 0;
            up.currentTime = 0;
          }
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
    setupSliders();
    setupKeyboardShortcuts();
    setupVideoTabs();

    btnNew.addEventListener("click", showGeneratePanel);
    btnGenerate.addEventListener("click", startGeneration);
    btnCancel.addEventListener("click", cancelGeneration);
    btnBack.addEventListener("click", showGeneratePanel);
    btnDelete.addEventListener("click", deleteSession);
    btnClone.addEventListener("click", cloneSession);
    btnTheme.addEventListener("click", toggleTheme);

    // Resume buttons
    $$(".btn-resume").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        resumeFromStep(btn.getAttribute("data-step"));
      });
    });

    // Poll VRAM every 10s
    pollVRAM();
    setInterval(pollVRAM, 10000);

    // Load sessions immediately, then refresh every 15s
    refreshSessions();
    setInterval(refreshSessions, 15000);

    // Check if there's an active generation and auto-show it
    fetch("/api/active").then(r => r.json()).then(d => {
      if (d.running && d.session_id) {
        currentSessionId = d.session_id;
        showSession(d.session_id);
      }
    }).catch(() => {});
  }

  document.addEventListener("DOMContentLoaded", init);
})();
