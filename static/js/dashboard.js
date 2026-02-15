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
  let activeSessionId = null;   // track displayed active session for View Details

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
      const [statsR, vramR, sysR] = await Promise.all([
        fetch("/api/stats"),
        fetch("/api/vram"),
        fetch("/api/system"),
      ]);
      const stats = await statsR.json();
      const vram = await vramR.json();
      const sys = await sysR.json();
      renderStats(stats, vram, sys);
    } catch (e) {
      console.error("Dashboard stats failed:", e);
    }
  }

  function renderStats(stats, vram, sys) {
    // GPU card
    const gpuName = $("#dash-gpu-name");
    const gpuArch = $("#dash-gpu-arch");
    const gpuPytorch = $("#dash-gpu-pytorch");
    if (gpuName) gpuName.textContent = stats.gpu?.device_name || "No GPU detected";
    if (gpuArch) gpuArch.textContent = stats.gpu?.gfx_arch || "—";
    if (gpuPytorch) gpuPytorch.textContent = stats.gpu?.pytorch_version || "—";

    // VRAM gauge — use reserved_gb as the primary "used" value
    const total = vram.total_gb || stats.gpu?.total_vram_gb || 24;
    const allocated = vram.allocated_gb || 0;
    const reserved = vram.reserved_gb || 0;
    // Use reserved (actual OS-level VRAM usage) as primary
    const used = reserved;
    const pct = Math.min(100, (used / total) * 100);

    const gauge = $("#dash-vram-gauge-fill");
    const gaugeText = $("#dash-vram-gauge-text");
    const vramDetail = $("#dash-vram-detail");
    if (gauge) {
      const offset = 157 - (pct / 100) * 157;
      gauge.style.strokeDashoffset = offset;
      if (pct > 85) gauge.setAttribute("data-level", "critical");
      else if (pct > 60) gauge.setAttribute("data-level", "warning");
      else gauge.setAttribute("data-level", "normal");
    }
    if (gaugeText) gaugeText.textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
    if (vramDetail) vramDetail.textContent = `Allocated: ${allocated.toFixed(1)} GB`;

    // System monitor card
    renderSystemMonitor(sys);

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

    // Energy stats
    const e = stats.energy || {};
    const energyTotal = $("#dash-energy-total");
    const energyGpu = $("#dash-energy-gpu");
    const energyPeak = $("#dash-energy-peak");
    const energyCost = $("#dash-energy-cost");
    if (energyTotal) {
      if (e.total_wh > 0) {
        energyTotal.textContent = e.total_wh >= 1000
          ? `${(e.total_wh / 1000).toFixed(2)} kWh`
          : `${e.total_wh.toFixed(1)} Wh`;
      } else {
        energyTotal.textContent = "—";
      }
    }
    if (energyGpu) energyGpu.textContent = e.gpu_wh > 0
      ? (e.gpu_wh >= 1000 ? `${(e.gpu_wh / 1000).toFixed(2)} kWh` : `${e.gpu_wh.toFixed(1)} Wh`)
      : "—";
    if (energyPeak) energyPeak.textContent = e.peak_gpu_w > 0 ? `${e.peak_gpu_w.toFixed(0)}W` : "—";
    if (energyCost) {
      if (e.total_cost > 0) {
        energyCost.textContent = `$${e.total_cost.toFixed(4)} across ${e.sessions_tracked} session${e.sessions_tracked !== 1 ? "s" : ""} @ $${e.cost_kwh}/kWh`;
      } else {
        energyCost.textContent = `${e.sessions_tracked || 0} sessions tracked`;
      }
    }
  }

  function renderSystemMonitor(sys) {
    const panel = $("#dash-system-monitor");
    if (!panel) return;

    let html = "";

    // GPU section
    const g = sys.gpu || {};
    if (g.power_w != null || g.temp_edge_c != null) {
      html += '<div class="sysmon-section"><div class="sysmon-section-title">🖥️ GPU</div><div class="sysmon-grid">';
      if (g.power_w != null)         html += sysItem("Power Draw", `${g.power_w.toFixed(0)} W`, powerColor(g.power_w));
      if (g.gpu_use_pct != null)     html += sysItem("GPU Load", `${g.gpu_use_pct.toFixed(0)}%`, pctColor(g.gpu_use_pct));
      if (g.temp_junction_c != null) html += sysItem("Junction", `${g.temp_junction_c.toFixed(0)}°C`, tempColor(g.temp_junction_c));
      if (g.temp_edge_c != null)     html += sysItem("Edge Temp", `${g.temp_edge_c.toFixed(0)}°C`, tempColor(g.temp_edge_c));
      if (g.temp_memory_c != null)   html += sysItem("VRAM Temp", `${g.temp_memory_c.toFixed(0)}°C`, tempColor(g.temp_memory_c));
      if (g.fan_pct != null)         html += sysItem("Fan", `${g.fan_pct.toFixed(0)}%`, null);
      if (g.sclk)                    html += sysItem("GPU Clock", g.sclk.replace(/[()]/g, ""), null);
      if (g.mclk)                    html += sysItem("Mem Clock", g.mclk.replace(/[()]/g, ""), null);
      if (g.pcie)                    html += sysItem("PCIe", g.pcie, null);
      html += '</div></div>';
    }

    // CPU section
    const c = sys.cpu || {};
    if (c.model || c.usage_pct != null) {
      html += '<div class="sysmon-section"><div class="sysmon-section-title">🧠 CPU</div>';
      if (c.model) html += `<div class="sysmon-cpu-model">${App.escapeHtml(c.model)}</div>`;
      html += '<div class="sysmon-grid">';
      if (c.usage_pct != null) html += sysItem("Usage", `${c.usage_pct.toFixed(0)}%`, pctColor(c.usage_pct));
      if (c.temp_c != null)    html += sysItem("Temperature", `${c.temp_c.toFixed(0)}°C`, tempColor(c.temp_c));
      if (c.power_w != null)   html += sysItem("Power Draw", `${c.power_w.toFixed(0)} W`, null);
      if (c.cores)             html += sysItem("Threads", `${c.cores}`, null);
      if (c.freq_mhz != null)  html += sysItem("Frequency", `${c.freq_mhz.toFixed(0)} MHz`, null);
      if (c.load_avg) {
        const [l1, l5, l15] = c.load_avg;
        html += sysItem("Load (1/5/15m)", `${l1.toFixed(1)} / ${l5.toFixed(1)} / ${l15.toFixed(1)}`, null);
      }
      html += '</div></div>';
    }

    // RAM section
    const r = sys.ram || {};
    const sw = sys.swap || {};
    if (r.total_gb) {
      html += '<div class="sysmon-section"><div class="sysmon-section-title">🧮 Memory</div><div class="sysmon-grid">';
      html += sysItem("RAM", `${r.used_gb} / ${r.total_gb} GB (${r.percent}%)`, pctColor(r.percent));
      html += sysItem("Available", `${r.available_gb} GB`, null);
      if (sw.total_gb > 0) html += sysItem("Swap", `${sw.used_gb} / ${sw.total_gb} GB (${sw.percent}%)`, pctColor(sw.percent));
      html += '</div></div>';
    }

    // Uptime
    if (sys.uptime_seconds != null) {
      const up = formatUptime(sys.uptime_seconds);
      html += `<div class="sysmon-section"><div class="sysmon-section-title">⏱️ Uptime</div><div class="sysmon-uptime">${up}</div></div>`;
    }

    panel.innerHTML = html || '<div class="text-muted" style="text-align:center;padding:12px">No system data available</div>';
  }

  function sysItem(label, value, color) {
    const style = color ? ` style="color:${color}"` : "";
    return `<div class="sysmon-item"><span class="sysmon-label">${label}</span><span class="sysmon-value"${style}>${value}</span></div>`;
  }

  function tempColor(c) {
    if (c >= 90) return "var(--danger)";
    if (c >= 70) return "var(--warning)";
    return null;
  }

  function powerColor(w) {
    if (w >= 250) return "var(--danger)";
    if (w >= 150) return "var(--warning)";
    return null;
  }

  function pctColor(p) {
    if (p >= 90) return "var(--danger)";
    if (p >= 70) return "var(--warning)";
    return null;
  }

  function formatUptime(seconds) {
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const parts = [];
    if (d > 0) parts.push(`${d}d`);
    if (h > 0) parts.push(`${h}h`);
    parts.push(`${m}m`);
    return parts.join(" ");
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
        activeSessionId = d.session_id;
        const sr = await fetch(`/api/sessions/${d.session_id}`);
        const s = await sr.json();
        renderActiveGeneration(s);
      } else {
        section.style.display = "none";
        activeSessionId = null;
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
      let energyStr = "";
      if (s.energy_wh > 0) {
        energyStr = `<span class="dash-gallery-energy" title="Estimated wall energy">⚡${s.energy_wh.toFixed(1)}Wh</span>`;
      }

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
            ${energyStr}
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

  function getActiveSessionId() { return activeSessionId; }

  return { init, show, hide, loadStats, loadActiveGeneration, loadRecentSessions, startPolling, stopPolling, getActiveSessionId };
})();
