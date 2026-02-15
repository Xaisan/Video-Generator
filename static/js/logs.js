/*
 * logs.js — Log viewer & session log polling
 * =============================================
 * Pipeline log ring buffer polling and session log files
 * (generation.log / vram.log) display.
 */

"use strict";

const Logs = (() => {
  const { $, $$ } = App;

  // ─── Pipeline log ring buffer ─────────────────────────────────

  function startPolling() {
    stopPolling();
    App.lastLogTs = 0;
    const logContent = $("#log-content");
    if (logContent) logContent.innerHTML = "";
    App.logPollInterval = setInterval(fetchLogs, 2000);
    fetchLogs();
  }

  function stopPolling() {
    if (App.logPollInterval) clearInterval(App.logPollInterval);
    App.logPollInterval = null;
  }

  function fetchLogs() {
    fetch(`/api/logs?since=${App.lastLogTs}&n=50`)
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
          if (line.ts > App.lastLogTs) App.lastLogTs = line.ts;
        }
        logContent.scrollTop = logContent.scrollHeight;
      }).catch(() => {});
  }

  // ─── Session log files (generation.log / vram.log) ────────────

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
    if (!App.currentSessionId) return;

    fetch(`/api/sessions/${App.currentSessionId}/generation_log`)
      .then(r => r.json())
      .then(d => {
        const el = $("#gen-log-content");
        if (el) {
          el.textContent = d.content || "(empty)";
          if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
            el.scrollTop = el.scrollHeight;
          }
        }
        const sec = $("#session-logs-section");
        if (sec && d.exists) sec.style.display = "block";
      }).catch(() => {});

    fetch(`/api/sessions/${App.currentSessionId}/vram_log`)
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
    App.sessionLogPollInterval = setInterval(fetchSessionLogs, 5000);
  }

  function stopSessionLogPolling() {
    if (App.sessionLogPollInterval) clearInterval(App.sessionLogPollInterval);
    App.sessionLogPollInterval = null;
  }

  function updateDownloadLinks(sid) {
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

  return {
    startPolling,
    stopPolling,
    setupSessionLogTabs,
    fetchSessionLogs,
    startSessionLogPolling,
    stopSessionLogPolling,
    updateDownloadLinks,
  };
})();
