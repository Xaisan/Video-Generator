/*
 * sse.js — Server-Sent Events & progress handling
 * =================================================
 * Manages the SSE connection to /api/events and dispatches
 * progress + completion events to update the UI.
 */

"use strict";

const SSE = (() => {
  const { $ } = App;

  function connect() {
    if (App.evtSource) App.evtSource.close();
    App.evtSource = new EventSource("/api/events");

    App.evtSource.addEventListener("connected", (e) => {
      console.log("SSE connected:", JSON.parse(e.data));
    });

    App.evtSource.addEventListener("progress", (e) => {
      handleProgress(JSON.parse(e.data));
    });

    App.evtSource.addEventListener("generation_complete", (e) => {
      handleComplete(JSON.parse(e.data));
    });

    App.evtSource.onerror = () => {
      console.warn("SSE error, reconnecting in 3s…");
      setTimeout(connect, 3000);
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
      App.stepProgressData[d.step] = d.percent;
      const fill = card.querySelector(".step-progress-fill");
      if (fill) fill.style.width = Math.min(100, d.percent) + "%";
    }

    if (d.vram && d.vram.allocated_gb !== undefined) {
      VRAM.update(d.vram);
      VRAM.updateDetail(d.vram);
    }
  }

  function handleComplete(d) {
    Timer.stop();
    Sessions.stopAutoRefresh();
    Logs.stopSessionLogPolling();
    Logs.fetchSessionLogs();

    const session = d.session || {};
    const status = session.status || "done";

    if (status === "done") App.toast("Generation complete! 🎉", "success");
    else if (status === "failed") App.toast("Generation failed: " + (session.error_message || "unknown error"), "error", 8000);
    else if (status === "cancelled") App.toast("Generation cancelled", "warning");

    Sessions.refresh().then(() => {
      if (d.session_id === App.currentSessionId) Sessions.show(d.session_id);
    });
    const btnCancel = $("#btn-cancel");
    if (btnCancel) btnCancel.style.display = "none";
    const btnGenerate = $("#btn-generate");
    if (btnGenerate) btnGenerate.disabled = false;
  }

  return { connect, handleProgress, handleComplete };
})();
