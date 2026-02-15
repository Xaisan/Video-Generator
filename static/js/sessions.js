/*
 * sessions.js — Session list, detail view & video tabs
 * ======================================================
 * Handles session list rendering, session detail display,
 * video section with tabs (original/upscaled/compare).
 */

"use strict";

const Sessions = (() => {
  const { $, $$ } = App;

  async function refresh() {
    try {
      const url = (typeof Users !== "undefined") ? Users.getSessionsUrl() : "/api/sessions";
      const r = await fetch(url);
      App.sessions = await r.json();
      renderList();
    } catch (e) { console.error("Failed to refresh sessions:", e); }
  }

  function renderList() {
    const sessionList = $("#session-list");
    if (!sessionList) return;
    sessionList.innerHTML = "";
    for (const s of App.sessions) {
      const div = document.createElement("div");
      div.className = "session-item" + (s.session_id === App.currentSessionId ? " active" : "");
      const date = new Date(s.created_at * 1000);
      const dateStr = date.toLocaleString();
      const prompt = s.prompt || "(no prompt)";
      const thumbUrl = `/sessions/${s.session_id}/input.png`;
      const thumbHtml = `<img class="si-thumb" src="${thumbUrl}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="si-thumb-placeholder" style="display:none">🎬</div>`;
      const userTag = s.user_name ? `<span class="si-user" title="${App.escapeHtml(s.user_name)}">${App.escapeHtml(s.user_name)}</span>` : "";
      div.innerHTML = `
        ${thumbHtml}
        <div class="si-details">
          <div class="si-top">
            <span class="si-id">${s.session_id.slice(0, 8)}</span>
            ${userTag}
            <span class="si-status ${s.status}">${s.status}</span>
          </div>
          <div class="si-prompt" title="${App.escapeHtml(prompt)}">${App.escapeHtml(prompt.slice(0, 50))}</div>
          <div class="si-date">${dateStr}</div>
        </div>
      `;
      div.addEventListener("click", () => show(s.session_id));
      sessionList.appendChild(div);
    }
  }

  async function show(sid) {
    App.currentSessionId = sid;
    renderList();
    try {
      const r = await fetch(`/api/sessions/${sid}`);
      if (!r.ok) return;
      const s = await r.json();
      renderDetail(s);
      const dashPanel = $("#panel-dashboard");
      if (dashPanel) dashPanel.style.display = "none";
      if (typeof Dashboard !== "undefined") Dashboard.hide();
      $("#panel-generate").style.display = "none";
      $("#panel-session").style.display = "block";

      Logs.updateDownloadLinks(sid);
      Logs.fetchSessionLogs();

      if (s.status === "running") {
        startAutoRefresh();
        Logs.startPolling();
        Logs.startSessionLogPolling();
        Timer.start();
      } else {
        stopAutoRefresh();
        Logs.stopPolling();
        Logs.stopSessionLogPolling();
        Timer.stop();
      }
    } catch (e) { console.error(e); }
  }

  function renderDetail(s) {
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
        else if (status === "running" && App.stepProgressData[step] !== undefined)
          fill.style.width = Math.min(100, App.stepProgressData[step]) + "%";
        else if (status === "pending") fill.style.width = "0%";
      }
      const timeEl = card.querySelector(".step-time");
      if (s.step_times && s.step_times[step])
        timeEl.textContent = App.formatStepTime(s.step_times[step]);
      else timeEl.textContent = "";

      if (step === "upscale" && s.upscale_model) {
        const uSelect = card.querySelector("select");
        if (uSelect && document.activeElement !== uSelect) uSelect.value = s.upscale_model;
        const fpsInput = card.querySelector(".upscale-step-fps");
        if (fpsInput && document.activeElement !== fpsInput) fpsInput.value = s.output_fps || 24;
        const durInput = card.querySelector(".upscale-step-duration");
        if (durInput && document.activeElement !== durInput) durInput.value = s.target_duration || 0;
      }
    });

    $("#info-prompt").textContent = s.prompt || "—";
    const durStr = s.duration ? `${s.duration}s` : "";
    const outFpsStr = s.output_fps && s.output_fps !== s.fps ? ` → ${s.output_fps}fps` : "";
    const tgtDurStr = s.target_duration && s.target_duration > 0 ? ` (target: ${s.target_duration}s)` : "";
    $("#info-size").textContent = `${s.width}×${s.height}, ${s.num_frames}f ${durStr} @ ${s.fps}fps${outFpsStr}${tgtDurStr}`;
    const modeStr = s.distill_lora_mode ? "⚡ Fast" : "🎨 Quality";
    const presetStr = s.preset_name ? ` | Preset: ${s.preset_name}` : "";
    $("#info-steps").textContent = `${s.num_inference_steps} (${modeStr}${presetStr})`;
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

    // User info
    const userRow = $("#info-user-row");
    if (userRow) {
      if (s.user_name) {
        userRow.style.display = "";
        $("#info-user").textContent = s.user_name;
      } else {
        userRow.style.display = "none";
      }
    }

    if (s.status === "running") {
      $("#progress-section").style.display = "block";
      const btnCancel = $("#btn-cancel");
      if (btnCancel) btnCancel.style.display = "inline-flex";
      $("#timer-bar").style.display = "flex";
    } else {
      $("#progress-section").style.display = "none";
      const btnCancel = $("#btn-cancel");
      if (btnCancel) btnCancel.style.display = "none";
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
    dlBtn.download = `${s.session_id.slice(0, 8)}_${videoFile}`;

    const tabUp = $("#tab-upscaled");
    const tabCompare = $("#tab-compare");
    if (hasUpscaled) {
      tabUp.style.display = "";
      tabCompare.style.display = "";
      $("#video-player-upscaled").src = upSrc;
      const dlUp = $("#video-download-upscaled");
      dlUp.href = upSrc;
      dlUp.download = `${s.session_id.slice(0, 8)}_upscaled.mp4`;
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

  // ─── Auto-refresh for running sessions ────────────────────────

  function startAutoRefresh() {
    stopAutoRefresh();
    App.autoRefreshInterval = setInterval(() => {
      if (App.currentSessionId) {
        fetch(`/api/sessions/${App.currentSessionId}`)
          .then(r => r.json())
          .then(s => {
            if (s.session_id) renderDetail(s);
            if (s.status !== "running") stopAutoRefresh();
          }).catch(() => {});
      }
    }, 3000);
  }

  function stopAutoRefresh() {
    if (App.autoRefreshInterval) clearInterval(App.autoRefreshInterval);
    App.autoRefreshInterval = null;
  }

  return {
    refresh,
    renderList,
    show,
    renderDetail,
    setupVideoTabs,
    startAutoRefresh,
    stopAutoRefresh,
  };
})();
