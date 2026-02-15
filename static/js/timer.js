/*
 * timer.js — Generation timer & ETA
 * ====================================
 * Elapsed time display and ETA estimation during generation.
 */

"use strict";

const Timer = (() => {
  const { $ } = App;

  function start() {
    App.generationStartTime = Date.now();
    const timerBar = $("#timer-bar");
    if (timerBar) timerBar.style.display = "flex";
    updateDisplay();
    App.timerInterval = setInterval(updateDisplay, 1000);
  }

  function stop() {
    if (App.timerInterval) clearInterval(App.timerInterval);
    App.timerInterval = null;
  }

  function updateDisplay() {
    if (!App.generationStartTime) return;
    const elapsed = Math.floor((Date.now() - App.generationStartTime) / 1000);
    const el = $("#timer-elapsed");
    if (el) el.textContent = `⏱ ${App.formatDuration(elapsed)}`;
    const pFill = $("#progress-fill");
    if (pFill) {
      const pct = parseFloat(pFill.style.width) || 0;
      if (pct > 5 && elapsed > 10) {
        const totalEst = (elapsed / pct) * 100;
        const remaining = Math.max(0, Math.floor(totalEst - elapsed));
        const etaEl = $("#timer-eta");
        if (etaEl) etaEl.textContent = `ETA: ~${App.formatDuration(remaining)}`;
      }
    }
  }

  return { start, stop };
})();
