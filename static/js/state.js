/*
 * state.js — Shared application state
 * =====================================
 * Central state object and DOM helper utilities used by all modules.
 * Must be loaded FIRST before any other module.
 */

"use strict";

const App = {
  // ─── Shared state ────────────────────────────────────────────
  currentSessionId: null,
  sessions: [],
  evtSource: null,
  generationStartTime: null,
  timerInterval: null,
  autoRefreshInterval: null,
  logPollInterval: null,
  lastLogTs: 0,
  stepProgressData: {},
  currentMode: "quality",
  sessionLogPollInterval: null,

  // Preset state
  availableModels: window.__AVAILABLE_MODELS || {},
  presets: window.__PRESETS || [],
  configLoras: window.__CONFIG_LORAS || [],

  // User state
  activeUserId: "",
  activeUserName: "",

  // ─── DOM helpers ─────────────────────────────────────────────
  $: (s) => document.querySelector(s),
  $$: (s) => document.querySelectorAll(s),

  // ─── Utility functions ───────────────────────────────────────
  escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  },

  basename(path) {
    if (!path) return "";
    return path.split("/").pop();
  },

  formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  },

  formatStepTime(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + "s";
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(0);
    return `${m}m ${s}s`;
  },

  // ─── Toast notification system ───────────────────────────────
  toast(message, type = "info", duration = 4000) {
    const container = App.$("#toast-container");
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.classList.add("toast-out");
      setTimeout(() => el.remove(), 300);
    }, duration);
  },

  // ─── Theme ───────────────────────────────────────────────────
  initTheme() {
    const saved = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
    App.updateThemeButton(saved);
  },

  toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    App.updateThemeButton(next);
  },

  updateThemeButton(theme) {
    const btn = App.$("#btn-theme");
    if (!btn) return;
    btn.textContent = theme === "dark" ? "🌙" : "☀️";
    btn.title = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  },
};
