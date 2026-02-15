/*
 * init.js — Application initialization
 * =======================================
 * Wires up all modules, binds event listeners, starts polling.
 * Must be loaded LAST after all other modules.
 */

"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const { $, $$ } = App;

  // Theme
  App.initTheme();

  // SSE
  SSE.connect();

  // Drop zone & upload
  Upload.setup();

  // Hybrid slider/number inputs + resolution scale buttons
  Controls.setupHybridInputs();
  Controls.setupResScaleButtons();

  // Mode toggle (quality/fast)
  Generate.setupModeToggle();

  // Keyboard shortcuts
  Generate.setupKeyboardShortcuts();

  // Video tabs
  Sessions.setupVideoTabs();

  // Session log tabs
  Logs.setupSessionLogTabs();

  // Presets
  Presets.setup();

  // Users
  Users.setup();

  // Initialize computed values
  Controls.updateFramesFromDuration();
  Controls.updateInterpHint();
  Controls.updateMegapixels();
  Controls.updateTotalSteps();
  Controls.updateTargetDurationHint();

  // ─── Sidebar navigation ──────────────────────────────────────
  function setActiveNav(btnId) {
    $$(".sidebar-nav-btn").forEach(b => b.classList.remove("active"));
    const btn = $(`#${btnId}`);
    if (btn) btn.classList.add("active");
  }

  $("#btn-dashboard").addEventListener("click", () => {
    setActiveNav("btn-dashboard");
    Dashboard.show();
  });

  // Sidebar title also navigates to dashboard
  const sidebarTitle = $("#sidebar-title");
  if (sidebarTitle) sidebarTitle.addEventListener("click", () => {
    setActiveNav("btn-dashboard");
    Dashboard.show();
  });

  $("#btn-new").addEventListener("click", () => {
    setActiveNav("btn-new");
    Dashboard.hide();
    Generate.showGeneratePanel();
  });

  $("#btn-presets-nav").addEventListener("click", () => {
    Presets.refreshList();
    // Open presets editor in separate window
    const url = "/presets/editor";
    const w = 720, h = 700;
    const left = (screen.width - w) / 2;
    const top = (screen.height - h) / 2;
    const win = window.open(url, "preset_editor",
      `width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`);
    if (win) win.focus();
  });

  // Dashboard hero button
  const heroBtn = $("#btn-new-hero");
  if (heroBtn) heroBtn.addEventListener("click", () => {
    setActiveNav("btn-new");
    Dashboard.hide();
    Generate.showGeneratePanel();
  });

  // Dashboard quick actions
  const viewActiveBtn = $("#btn-dash-view-active");
  if (viewActiveBtn) viewActiveBtn.addEventListener("click", () => {
    fetch("/api/active").then(r => r.json()).then(d => {
      if (d.running && d.session_id) {
        setActiveNav("");
        Dashboard.hide();
        Sessions.show(d.session_id);
      }
    }).catch(() => {});
  });

  const dashPresetsBtn = $("#btn-dash-presets");
  if (dashPresetsBtn) dashPresetsBtn.addEventListener("click", () => {
    const url = "/presets/editor";
    const w = 720, h = 700;
    const left = (screen.width - w) / 2;
    const top = (screen.height - h) / 2;
    const win = window.open(url, "preset_editor",
      `width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`);
    if (win) win.focus();
  });

  const dashManageUsersBtn = $("#btn-dash-manage-users");
  if (dashManageUsersBtn) dashManageUsersBtn.addEventListener("click", () => {
    $("#btn-manage-users")?.click();
  });

  const viewAllBtn = $("#btn-dash-view-all");
  if (viewAllBtn) viewAllBtn.addEventListener("click", () => {
    // Scroll sidebar sessions into view or click first session
    const firstSession = $(".session-item");
    if (firstSession) firstSession.click();
  });

  // ─── Existing button bindings ──────────────────────────────────
  $("#btn-generate").addEventListener("click", Generate.start);
  $("#btn-cancel").addEventListener("click", Generate.cancel);

  // Back button returns to dashboard
  $("#btn-back").addEventListener("click", () => {
    setActiveNav("btn-dashboard");
    Dashboard.show();
  });

  // Back to dashboard from generate panel
  const backToDash = $("#btn-back-to-dash");
  if (backToDash) backToDash.addEventListener("click", () => {
    setActiveNav("btn-dashboard");
    Dashboard.show();
  });

  $("#btn-delete-session").addEventListener("click", Generate.deleteSession);
  $("#btn-clone").addEventListener("click", () => {
    setActiveNav("btn-new");
    Generate.cloneSession();
  });
  $("#btn-theme").addEventListener("click", App.toggleTheme);

  // Resume buttons on step cards
  $$(".btn-resume").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      Generate.resumeFromStep(btn.getAttribute("data-step"));
    });
  });

  // VRAM polling
  VRAM.poll();
  setInterval(VRAM.poll, 10000);

  // Session list refresh
  Sessions.refresh();
  setInterval(() => Sessions.refresh(), 15000);

  // ─── Dashboard: show by default ───────────────────────────────
  // Check for active generation on page load
  fetch("/api/active").then(r => r.json()).then(d => {
    if (d.running && d.session_id) {
      // If something is running, show its session detail
      App.currentSessionId = d.session_id;
      setActiveNav("");
      Dashboard.hide();
      Sessions.show(d.session_id);
    } else {
      // Show dashboard
      Dashboard.init();
    }
  }).catch(() => {
    Dashboard.init();
  });
});
