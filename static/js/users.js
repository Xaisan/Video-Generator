/*
 * users.js — User profile management
 * =====================================
 * Handles user profile switching, creation, editing, deletion.
 * Users provide optional session separation — sessions created
 * while a user is selected are tagged with that user's ID.
 *
 * The "global" space (no user selected) shows all sessions.
 * Selecting a user filters the sidebar to their sessions only.
 */

"use strict";

const Users = (() => {
  const { $, $$ } = App;

  // Available avatars loaded from server
  let avatarOptions = window.__AVATAR_OPTIONS || [];
  let users = window.__USERS || [];

  function setup() {
    const switcher = $("#user-switcher");
    if (!switcher) return;

    // User switcher change
    switcher.addEventListener("change", () => {
      const uid = switcher.value;
      setActiveUser(uid);
    });

    // Create user button
    const btnCreate = $("#btn-create-user");
    if (btnCreate) btnCreate.addEventListener("click", showCreateDialog);

    // Manage users button
    const btnManage = $("#btn-manage-users");
    if (btnManage) btnManage.addEventListener("click", showManageDialog);

    // Restore last selected user from localStorage
    const saved = localStorage.getItem("activeUserId") || "";
    if (saved && users.find(u => u.id === saved)) {
      switcher.value = saved;
      App.activeUserId = saved;
      App.activeUserName = (users.find(u => u.id === saved) || {}).name || "";
    } else {
      App.activeUserId = "";
      App.activeUserName = "";
    }

    updateUserBadge();
  }

  function setActiveUser(userId) {
    App.activeUserId = userId;
    const user = users.find(u => u.id === userId);
    App.activeUserName = user ? user.name : "";
    localStorage.setItem("activeUserId", userId);

    updateUserBadge();

    // Refresh sessions to show only this user's (or all for global)
    Sessions.refresh();

    // Refresh dashboard if visible
    if (typeof Dashboard !== "undefined") {
      Dashboard.loadRecentSessions();
    }

    const label = user ? `${user.avatar || "👤"} ${user.name}` : "🌐 Global";
    App.toast(`Switched to ${label}`, "info", 2000);
  }

  function updateUserBadge() {
    const badge = $("#user-active-badge");
    if (!badge) return;

    if (App.activeUserId) {
      const user = users.find(u => u.id === App.activeUserId);
      if (user) {
        badge.textContent = `${user.avatar || "👤"} ${user.name}`;
        badge.style.display = "inline-flex";
      } else {
        badge.textContent = "";
        badge.style.display = "none";
      }
    } else {
      badge.textContent = "🌐 Global";
      badge.style.display = "inline-flex";
    }
  }

  function getSessionsUrl() {
    // Always pass user_id — empty string means "global (unowned) sessions only"
    return `/api/sessions?user_id=${encodeURIComponent(App.activeUserId || "")}`;
  }

  // ─── Create User Dialog ──────────────────────────────────────

  function showCreateDialog() {
    const overlay = document.createElement("div");
    overlay.className = "user-dialog-overlay";
    overlay.innerHTML = `
      <div class="user-dialog">
        <h3>Create Profile</h3>
        <div class="user-dialog-field">
          <label>Name</label>
          <input type="text" id="new-user-name" maxlength="50" placeholder="Your name" autofocus />
        </div>
        <div class="user-dialog-field">
          <label>Avatar</label>
          <div class="avatar-grid" id="avatar-grid"></div>
        </div>
        <div class="user-dialog-actions">
          <button class="btn btn-ghost btn-sm" id="btn-cancel-user">Cancel</button>
          <button class="btn btn-primary btn-sm" id="btn-save-user">Create</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    // Populate avatar grid
    const grid = overlay.querySelector("#avatar-grid");
    let selectedAvatar = "👤";
    avatarOptions.forEach(av => {
      const btn = document.createElement("button");
      btn.className = "avatar-option" + (av === selectedAvatar ? " selected" : "");
      btn.textContent = av;
      btn.addEventListener("click", () => {
        grid.querySelectorAll(".avatar-option").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        selectedAvatar = av;
      });
      grid.appendChild(btn);
    });

    overlay.querySelector("#btn-cancel-user").addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    overlay.querySelector("#btn-save-user").addEventListener("click", async () => {
      const name = overlay.querySelector("#new-user-name").value.trim();
      if (!name) {
        App.toast("Please enter a name", "warning");
        return;
      }
      try {
        const r = await fetch("/api/users", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, avatar: selectedAvatar }),
        });
        if (!r.ok) {
          const d = await r.json();
          App.toast(d.error || "Failed to create user", "error");
          return;
        }
        const user = await r.json();
        users.push(user);
        refreshSwitcher();
        setActiveUser(user.id);
        const switcher = $("#user-switcher");
        if (switcher) switcher.value = user.id;
        App.toast(`Profile "${user.name}" created!`, "success");
        overlay.remove();
      } catch (e) {
        App.toast("Error creating profile: " + e.message, "error");
      }
    });

    // Enter key to submit
    overlay.querySelector("#new-user-name").addEventListener("keydown", (e) => {
      if (e.key === "Enter") overlay.querySelector("#btn-save-user").click();
    });
  }

  // ─── Manage Users Dialog ─────────────────────────────────────

  function showManageDialog() {
    const overlay = document.createElement("div");
    overlay.className = "user-dialog-overlay";

    function renderList() {
      let html = `
        <div class="user-dialog user-dialog-wide">
          <h3>👥 Manage Profiles</h3>
          <div class="manage-users-list">
      `;
      if (users.length === 0) {
        html += `<div class="manage-user-empty">No profiles created yet</div>`;
      } else {
        for (const u of users) {
          const isActive = u.id === App.activeUserId;
          html += `
            <div class="manage-user-row${isActive ? " active" : ""}" data-uid="${u.id}">
              <span class="manage-user-avatar">${u.avatar || "👤"}</span>
              <span class="manage-user-name">${App.escapeHtml(u.name)}</span>
              ${isActive ? '<span class="manage-user-active-tag">active</span>' : ""}
              <div class="manage-user-actions">
                <button class="btn btn-xs btn-ghost btn-edit-user" data-uid="${u.id}" title="Edit">✏️</button>
                <button class="btn btn-xs btn-danger btn-delete-user" data-uid="${u.id}" title="Delete">🗑</button>
              </div>
            </div>
          `;
        }
      }
      html += `
          </div>
          <div class="user-dialog-actions">
            <button class="btn btn-ghost btn-sm" id="btn-close-manage">Close</button>
            <button class="btn btn-primary btn-sm" id="btn-add-from-manage">+ New Profile</button>
          </div>
        </div>
      `;
      overlay.innerHTML = html;

      // Event bindings
      overlay.querySelector("#btn-close-manage").addEventListener("click", () => overlay.remove());
      overlay.querySelector("#btn-add-from-manage").addEventListener("click", () => {
        overlay.remove();
        showCreateDialog();
      });
      overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

      overlay.querySelectorAll(".btn-delete-user").forEach(btn => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const uid = btn.dataset.uid;
          const user = users.find(u => u.id === uid);
          if (!confirm(`Delete profile "${user?.name}"?\n\nSessions created by this user will be preserved and become unowned (visible in Global space).`)) return;
          try {
            await fetch(`/api/users/${uid}`, { method: "DELETE" });
            users = users.filter(u => u.id !== uid);
            if (App.activeUserId === uid) {
              setActiveUser("");
              const switcher = $("#user-switcher");
              if (switcher) switcher.value = "";
            }
            refreshSwitcher();
            renderList();
            App.toast(`Profile deleted`, "info");
          } catch (e) {
            App.toast("Delete failed: " + e.message, "error");
          }
        });
      });

      overlay.querySelectorAll(".btn-edit-user").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          overlay.remove();
          showEditDialog(btn.dataset.uid);
        });
      });
    }

    renderList();
    document.body.appendChild(overlay);
  }

  // ─── Edit User Dialog ────────────────────────────────────────

  function showEditDialog(userId) {
    const user = users.find(u => u.id === userId);
    if (!user) return;

    const overlay = document.createElement("div");
    overlay.className = "user-dialog-overlay";
    overlay.innerHTML = `
      <div class="user-dialog">
        <h3>Edit Profile</h3>
        <div class="user-dialog-field">
          <label>Name</label>
          <input type="text" id="edit-user-name" maxlength="50" value="${App.escapeHtml(user.name)}" />
        </div>
        <div class="user-dialog-field">
          <label>Avatar</label>
          <div class="avatar-grid" id="edit-avatar-grid"></div>
        </div>
        <div class="user-dialog-actions">
          <button class="btn btn-ghost btn-sm" id="btn-cancel-edit-user">Cancel</button>
          <button class="btn btn-primary btn-sm" id="btn-save-edit-user">Save</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const grid = overlay.querySelector("#edit-avatar-grid");
    let selectedAvatar = user.avatar || "👤";
    avatarOptions.forEach(av => {
      const btn = document.createElement("button");
      btn.className = "avatar-option" + (av === selectedAvatar ? " selected" : "");
      btn.textContent = av;
      btn.addEventListener("click", () => {
        grid.querySelectorAll(".avatar-option").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        selectedAvatar = av;
      });
      grid.appendChild(btn);
    });

    overlay.querySelector("#btn-cancel-edit-user").addEventListener("click", () => {
      overlay.remove();
      showManageDialog();
    });
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    overlay.querySelector("#btn-save-edit-user").addEventListener("click", async () => {
      const name = overlay.querySelector("#edit-user-name").value.trim();
      if (!name) {
        App.toast("Please enter a name", "warning");
        return;
      }
      try {
        const r = await fetch(`/api/users/${userId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, avatar: selectedAvatar }),
        });
        if (!r.ok) {
          const d = await r.json();
          App.toast(d.error || "Failed to update user", "error");
          return;
        }
        const updated = await r.json();
        const idx = users.findIndex(u => u.id === userId);
        if (idx >= 0) users[idx] = updated;
        if (App.activeUserId === userId) {
          App.activeUserName = updated.name;
        }
        refreshSwitcher();
        updateUserBadge();
        App.toast(`Profile updated`, "success");
        overlay.remove();
        showManageDialog();
      } catch (e) {
        App.toast("Error updating profile: " + e.message, "error");
      }
    });
  }

  // ─── Refresh Switcher Dropdown ────────────────────────────────

  function refreshSwitcher() {
    const switcher = $("#user-switcher");
    if (!switcher) return;
    const currentVal = switcher.value;
    switcher.innerHTML = '<option value="">🌐 Global</option>';
    for (const u of users) {
      const opt = document.createElement("option");
      opt.value = u.id;
      opt.textContent = `${u.avatar || "👤"} ${u.name}`;
      switcher.appendChild(opt);
    }
    // Restore selection if still valid
    if (users.find(u => u.id === currentVal)) {
      switcher.value = currentVal;
    } else {
      switcher.value = "";
    }
  }

  // ─── Public API ───────────────────────────────────────────────

  return {
    setup,
    setActiveUser,
    getSessionsUrl,
    updateUserBadge,
    refreshSwitcher,
    get users() { return users; },
  };
})();
