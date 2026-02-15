/*
 * vram.js — VRAM monitoring & display
 * =====================================
 * Updates VRAM bars in sidebar and session detail panel.
 */

"use strict";

const VRAM = (() => {
  const { $ } = App;

  function update(v) {
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

  function updateDetail(v) {
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

  function poll() {
    fetch("/api/vram").then(r => r.json()).then(v => {
      update(v);
      updateDetail(v);
    }).catch(() => {});
  }

  return { update, updateDetail, poll };
})();
