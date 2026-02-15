/*
 * dashboard.js — Main dashboard overview
 * ========================================
 * Landing page showing system stats, GPU info, VRAM gauge,
 * active generation status, recent sessions gallery,
 * and quick-action buttons for all app features.
 */

"use strict";

const Dashboard = (() => {
  const { $, $$ } = App;
  let statsInterval = null;

  // ─── Load & render everything ─────────────────────────────────

  async function init() {
    await Promise.all([
      loadStats(),
      loadRecentSessions(),
      loadActiveGeneration(),
    ]);
    startPolling();
  }

  function show() {
    $("#panel-dashboard").style.display = "block";
    $("#panel-generate").style.display = "none";
    $("#panel-session").style.display = "none";
    App.currentSessionId = null;
    Sessions.stopAutoRefresh();
    Logs.stopPolling();
    Logs.stopSessionLogPolling();
    Timer.stop();
    Sessions.renderList();
    startPolling();
  }

  function hide() {
    stopPolling();
  }

  // ─── Polling ──────────────────────────────────────────────────

  function startPolling() {
    stopPolling();
    statsInterval = setInterval(() => {
      loadStats();
      loadActiveGeneration();
      loadRecentSessions();
    }, 8000);
  }

  function stopPolling() {
    if (statsInterval) clearInterval(statsInterval);
    statsInterval = null;
  }

  // ─── System Stats ────────────────────────────────────────────

  async function loadStats() {
    try {
      const [statsR, vramR] = await Promise.all([
        fetch("/api/stats"),
        fetch("/api/vram"),
      ]);
      const stats = await statsR.json();
      const vram = await vramR.json();
      renderStats(stats, vram);
    } catch (e) {
      console.error("Dashboard stats failed:", e);
    }
  }

  function renderStats(stats, vram) {
    // GPU card
    const gpuName = $("#dash-gpu-name");
    const gpuArch = $("#dash-gpu-arch");
    const gpuPytorch = $("#dash-gpu-pytorch");
    if (gpuName) gpuName.textContent = stats.gpu?.device_name || "No GPU detected";
    if (gpuArch) gpuArch.textContent = stats.gpu?.gfx_arch || "—";
    if (gpuPytorch) gpuPytorch.textContent = stats.gpu?.pytorch_version || "—";

    // VRAM gauge
    const total = vram.total_gb || stats.gpu?.total_vram_gb || 24;
    const used = vram.allocated_gb || 0;
    const reserved = vram.reserved_gb || 0;
    const pct = Math.min(100, (used / total) * 100);

    const gauge = $("#dash-vram-gauge-fill");
    const gaugeText = $("#dash-vram-gauge-text");
    const vramDetail = $("#dash-vram-detail");
    if (gauge) {
      // SVG arc dasharray is 157 (half circle). Offset from 157 (empty) to 0 (full)
      const offset = 157 - (pct / 100) * 157;
      gauge.style.strokeDashoffset = offset;
      if (pct > 85) gauge.setAttribute("data-level", "critical");
      else if (pct > 60) gauge.setAttribute("data-level", "warning");
      else gauge.setAttribute("data-level", "normal");
    }
    if (gaugeText) gaugeText.textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
    if (vramDetail) vramDetail.textContent = `Reserved: ${reserved.toFixed(1)} GB`;

    // Session counts
    const counts = stats.sessions || {};
    setCount("#dash-count-total", counts.total || 0);
    setCount("#dash-count-done", counts.done || 0);
    setCount("#dash-count-running", counts.running || 0);
    setCount("#dash-count-failed", counts.failed || 0);

    // Presets count
    setCount("#dash-count-presets", stats.preset_count || 0);

    // Users count
    setCount("#dash-count-users", stats.user_count || 0);

    // Disk usage
    const diskEl = $("#dash-disk-usage");
    if (diskEl && stats.disk) {
      diskEl.textContent = formatBytes(stats.disk.sessions_bytes || 0);
    }

    const diskOutputEl = $("#dash-disk-output");
    if (diskOutputEl && stats.disk) {
      diskOutputEl.textContent = formatBytes(stats.disk.output_bytes || 0);
    }
  }

  function setCount(selector, value) {
    const el = $(selector);
    if (el) el.textContent = value;
  }

  // ─── Active Generation ───────────────────────────────────────

  async function loadActiveGeneration() {
    try {
      const r = await fetch("/api/active");
      const d = await r.json();
      const section = $("#dash-active-section");
      if (!section) return;

      if (d.running && d.session_id) {
        section.style.display = "block";
        const sr = await fetch(`/api/sessions/${d.session_id}`);
        const s = await sr.json();
        renderActiveGeneration(s);
      } else {
        section.style.display = "none";
      }
    } catch (e) {
      const section = $("#dash-active-section");
      if (section) section.style.display = "none";
    }
  }

  function renderActiveGeneration(s) {
    const prompt = $("#dash-active-prompt");
    const status = $("#dash-active-status");
    const steps = $("#dash-active-steps");
    const sid = $("#dash-active-sid");

    if (prompt) prompt.textContent = s.prompt || "(no prompt)";
    if (status) {
      status.textContent = s.status;
      status.className = `si-status ${s.status}`;
    }
    if (sid) sid.textContent = s.session_id?.slice(0, 8) || "—";

    if (steps) {
      steps.innerHTML = "";
      const stepOrder = ["encode", "denoise", "vae_decode", "export", "upscale"];
      const stepData = s.steps || {};
      for (const step of stepOrder) {
        const st = stepData[step] || "pending";
        const dot = document.createElement("span");
        dot.className = `dash-step-dot dash-step-${st}`;
        dot.title = `${step}: ${st}`;
        dot.textContent = stepIcon(st);
        const label = document.createElement("span");
        label.className = "dash-step-label";
        label.textContent = step;
        const wrap = document.createElement("div");
        wrap.className = "dash-step-item";
        wrap.appendChild(dot);
        wrap.appendChild(label);
        steps.appendChild(wrap);
      }
    }
  }

  function stepIcon(status) {
    switch (status) {
      case "done": return "✓";
      case "running": return "●";
      case "failed": return "✕";
      case "skipped": return "—";
      default: return "○";
    }
  }

  // ─── Recent Sessions (Gallery) ───────────────────────────────

  async function loadRecentSessions() {
    try {
      const url = (typeof Users !== "undefined") ? Users.getSessionsUrl() : "/api/sessions";
      const r = await fetch(url);
      const sessions = await r.json();
      renderGallery(sessions.slice(0, 12));
    } catch (e) {
      console.error("Failed to load recent sessions:", e);
    }
  }

  function renderGallery(sessions) {
    const grid = $("#dash-gallery-grid");
    if (!grid) return;

    if (sessions.length === 0) {
      grid.innerHTML = `<div class="dash-gallery-empty">
        <div class="dash-gallery-empty-icon">🎬</div>
        <p>No sessions yet</p>
        <p class="text-muted">Start your first generation to see results here</p>
      </div>`;
      return;
    }

    grid.innerHTML = "";
    for (const s of sessions) {
      const card = document.createElement("div");
      card.className = "dash-gallery-card";
      card.setAttribute("data-status", s.status);

      const hasVideo = (s.checkpoints || []).includes("output.mp4") ||
                       (s.checkpoints || []).includes("output_upscaled.mp4");
      const hasPreview = (s.checkpoints || []).includes("preview.mp4");
      const hasUpscaled = (s.checkpoints || []).includes("output_upscaled.mp4");

      const thumbUrl = `/sessions/${s.session_id}/input.png`;
      const prompt = s.prompt || "(no prompt)";
      const date = new Date(s.created_at * 1000);
      const dateStr = date.toLocaleDateString() + " " + date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
      const modeStr = s.distill_lora_mode ? "⚡Fast" : "🎨Quality";
      const durStr = s.duration ? `${s.duration}s` : "";
      const userStr = s.user_name ? `<span class="dash-gallery-user">${App.escapeHtml(s.user_name)}</span>` : "";

      card.innerHTML = `
        <div class="dash-gallery-thumb">
          <img src="${thumbUrl}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
          <div class="dash-gallery-placeholder" style="display:none">🎬</div>
          ${hasVideo || hasPreview ? '<div class="dash-gallery-play">▶</div>' : ""}
          ${hasUpscaled ? '<span class="dash-gallery-badge-up">4K</span>' : ""}
          <span class="dash-gallery-status si-status ${s.status}">${s.status}</span>
        </div>
        <div class="dash-gallery-info">
          <div class="dash-gallery-prompt" title="${App.escapeHtml(prompt)}">${App.escapeHtml(prompt.slice(0, 60))}</div>
          <div class="dash-gallery-meta">
            <span>${s.width}×${s.height}</span>
            <span>${durStr}</span>
            <span>${modeStr}</span>
          </div>
          <div class="dash-gallery-footer">
            <span class="dash-gallery-date">${dateStr}</span>
            ${userStr}
            <span class="dash-gallery-id">${s.session_id.slice(0, 8)}</span>
          </div>
        </div>
      `;

      card.addEventListener("click", () => {
        Dashboard.hide();
        Sessions.show(s.session_id);
      });

      grid.appendChild(card);
    }
  }

  // ─── Utilities ────────────────────────────────────────────────

  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  return { init, show, hide, loadStats, loadActiveGeneration, loadRecentSessions, startPolling, stopPolling };
})();
