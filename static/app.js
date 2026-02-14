/* ── Wan 2.2 I2V — Web UI Frontend ──────────────────────────────── */

(() => {
  "use strict";

  // ─── State ──────────────────────────────────────────────────
  let currentSessionId = null;
  let sessions = [];
  let evtSource = null;

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
  const dropZone = $("#drop-zone");
  const fileInput = $("#file-input");
  const previewImg = $("#preview-img");
  const dropText = $("#drop-text");

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
    }

    // Update VRAM bar
    if (d.vram && d.vram.allocated_gb !== undefined) {
      updateVRAM(d.vram);
    }
  }

  function handleComplete(d) {
    // Refresh session list and detail
    refreshSessions().then(() => {
      if (d.session_id === currentSessionId) {
        showSession(d.session_id);
      }
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

    // Color code
    if (pct > 85) fill.style.background = "linear-gradient(90deg, #f85149, #da3633)";
    else if (pct > 60) fill.style.background = "linear-gradient(90deg, #d29922, #e3b341)";
    else fill.style.background = "";
  }

  function pollVRAM() {
    fetch("/api/vram").then(r => r.json()).then(updateVRAM).catch(() => {});
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
      div.innerHTML = `
        <div class="si-top">
          <span class="si-id">${s.session_id.slice(0, 8)}</span>
          <span class="si-status ${s.status}">${s.status}</span>
        </div>
        <div class="si-prompt" title="${escapeHtml(prompt)}">${escapeHtml(prompt.slice(0, 60))}</div>
        <div class="si-date">${dateStr}</div>
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

      // Show timing if available
      const timeEl = card.querySelector(".step-time");
      if (s.step_times && s.step_times[step]) {
        timeEl.textContent = s.step_times[step].toFixed(1) + "s";
      } else {
        timeEl.textContent = "";
      }
    });

    // Info
    $("#info-prompt").textContent = s.prompt || "—";
    $("#info-size").textContent = `${s.width}×${s.height}, ${s.num_frames} frames`;
    $("#info-steps").textContent = s.num_inference_steps;
    $("#info-seed").textContent = s.seed;
    $("#info-cfg").textContent = `${s.guidance_scale} / ${s.guidance_scale_2 || "—"}`;
    $("#info-status").textContent = s.status;
    $("#info-error").textContent = s.error_message || "—";

    // Progress
    if (s.status === "running") {
      $("#progress-section").style.display = "block";
      btnCancel.style.display = "inline-flex";
    } else {
      $("#progress-section").style.display = "none";
      btnCancel.style.display = "none";
    }

    // Video
    const videoSec = $("#video-section");
    const player = $("#video-player");
    const dlBtn = $("#video-download");
    const dlBtnUp = $("#video-download-upscaled");

    // Check which files exist via checkpoints
    const hasOutput = (s.checkpoints || []).includes("output.mp4");
    const hasPreview = (s.checkpoints || []).includes("preview.mp4");
    const hasUpscaled = (s.checkpoints || []).includes("output_upscaled.mp4");

    if (hasOutput || hasPreview) {
      videoSec.style.display = "block";
      const videoFile = hasOutput ? "output.mp4" : "preview.mp4";
      const src = `/sessions/${s.session_id}/${videoFile}`;
      player.src = src;
      dlBtn.href = src;
      dlBtn.download = `${s.session_id.slice(0,8)}_${videoFile}`;

      if (hasUpscaled) {
        dlBtnUp.style.display = "inline-flex";
        dlBtnUp.href = `/sessions/${s.session_id}/output_upscaled.mp4`;
        dlBtnUp.download = `${s.session_id.slice(0,8)}_upscaled.mp4`;
      } else {
        dlBtnUp.style.display = "none";
      }
    } else {
      videoSec.style.display = "none";
    }
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
      } else {
        alert("Upload failed: " + (d.error || "unknown"));
      }
    } catch (e) {
      alert("Upload error: " + e.message);
    }
  }

  // ─── Generate ───────────────────────────────────────────────
  function gatherParams() {
    const loraInputs = $$(".lora-scale");
    const loraScales = [];
    loraInputs.forEach(inp => {
      loraScales.push(parseFloat(inp.value) || 0);
    });

    return {
      prompt: $("#prompt").value,
      negative_prompt: $("#negative-prompt").value,
      input_image: $("#input-image-path").value,
      width: parseInt($("#width").value),
      height: parseInt($("#height").value),
      num_frames: parseInt($("#num-frames").value),
      fps: parseInt($("#fps").value),
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
      alert("Please upload an input image first.");
      return;
    }
    if (!params.prompt.trim()) {
      alert("Please enter a prompt.");
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
        alert("Error: " + d.error);
        btnGenerate.disabled = false;
        return;
      }

      // Switch to session view
      currentSessionId = d.session_id;
      await refreshSessions();
      showSession(d.session_id);
    } catch (e) {
      alert("Request failed: " + e.message);
      btnGenerate.disabled = false;
    }
  }

  async function resumeFromStep(step) {
    if (!currentSessionId) return;

    try {
      const r = await fetch(`/api/resume/${currentSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_step: step }),
      });
      const d = await r.json();
      if (d.error) {
        alert("Error: " + d.error);
        return;
      }
      btnCancel.style.display = "inline-flex";
      showSession(currentSessionId);
    } catch (e) {
      alert("Resume failed: " + e.message);
    }
  }

  async function cancelGeneration() {
    try {
      await fetch("/api/cancel", { method: "POST" });
    } catch (e) {
      console.error(e);
    }
  }

  async function deleteSession() {
    if (!currentSessionId) return;
    if (!confirm("Delete this session and all its files?")) return;

    try {
      await fetch(`/api/sessions/${currentSessionId}`, { method: "DELETE" });
      currentSessionId = null;
      await refreshSessions();
      showGeneratePanel();
    } catch (e) {
      alert("Delete failed: " + e.message);
    }
  }

  // ─── Navigation ─────────────────────────────────────────────
  function showGeneratePanel() {
    panelGen.style.display = "block";
    panelSession.style.display = "none";
    currentSessionId = null;
    renderSessionList();
  }

  // ─── Helpers ────────────────────────────────────────────────
  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ─── Init ───────────────────────────────────────────────────
  function init() {
    connectSSE();
    setupDropZone();

    btnNew.addEventListener("click", showGeneratePanel);
    btnGenerate.addEventListener("click", startGeneration);
    btnCancel.addEventListener("click", cancelGeneration);
    btnBack.addEventListener("click", showGeneratePanel);
    btnDelete.addEventListener("click", deleteSession);

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

    // Refresh sessions every 15s
    setInterval(refreshSessions, 15000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
