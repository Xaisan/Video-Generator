#!/usr/bin/env python3
"""
user_manager.py — Optional user profile management
====================================================

Provides lightweight user profiles for session separation.
No authentication — just a named identity that groups sessions.

Profiles are stored in users.json at the project root.
Each profile has:
  - id: unique identifier (timestamp + short uuid)
  - name: display name
  - avatar: emoji avatar (selected by user)
  - created_at: creation timestamp

The "global" space (no user selected) has no profile and sees
all sessions regardless of owner. Sessions created without a
user are "unowned" and always visible in the global space.

Public API:
  UserManager          — CRUD operations for user profiles
  USER_GLOBAL_ID       — sentinel value for "no user" / global space

Dependencies: none (leaf module)
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any


USERS_FILE = Path("users.json")
USER_GLOBAL_ID = ""  # Empty string = global / no user selected

# Default avatar options
AVATAR_OPTIONS = [
    "👤", "👩", "👨", "🧑", "👧", "👦",
    "🐱", "🐶", "🦊", "🐻", "🐼", "🐨",
    "🦁", "🐯", "🐸", "🐵", "🐰", "🦄",
    "🌟", "🎨", "🎬", "🚀", "🔥", "💎",
    "🎯", "🎮", "🎵", "🌈", "⚡", "🌙",
]


class UserManager:
    """CRUD operations for user profiles stored in users.json."""

    def __init__(self, path: str | Path = USERS_FILE):
        self.path = Path(path)
        self._users: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._users = data.get("users", {})
            except (json.JSONDecodeError, KeyError):
                self._users = {}
        else:
            self._users = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"users": self._users}, f, indent=2)

    def list_users(self) -> list[dict]:
        """Return all users sorted by name."""
        result = []
        for uid, user in self._users.items():
            u = dict(user)
            u["id"] = uid
            result.append(u)
        return sorted(result, key=lambda x: x.get("name", "").lower())

    def get_user(self, user_id: str) -> dict | None:
        """Get a single user by ID."""
        user = self._users.get(user_id)
        if user:
            u = dict(user)
            u["id"] = user_id
            return u
        return None

    def create_user(self, data: dict) -> dict:
        """Create a new user profile. Returns the created user with ID."""
        user_id = f"u_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        user = {
            "name": str(data.get("name", "User")).strip()[:50],
            "avatar": str(data.get("avatar", "👤")),
            "created_at": time.time(),
        }
        if not user["name"]:
            user["name"] = "User"
        self._users[user_id] = user
        self._save()
        u = dict(user)
        u["id"] = user_id
        return u

    def update_user(self, user_id: str, data: dict) -> dict | None:
        """Update an existing user profile."""
        if user_id not in self._users:
            return None
        existing = self._users[user_id]
        if "name" in data:
            existing["name"] = str(data["name"]).strip()[:50]
        if "avatar" in data:
            existing["avatar"] = str(data["avatar"])
        self._users[user_id] = existing
        self._save()
        u = dict(existing)
        u["id"] = user_id
        return u

    def delete_user(self, user_id: str) -> bool:
        """Delete a user profile. Sessions are NOT deleted — they become unowned."""
        if user_id in self._users:
            del self._users[user_id]
            self._save()
            return True
        return False

    def get_session_count(self, user_id: str, session_list: list) -> int:
        """Count sessions belonging to a specific user."""
        return sum(1 for s in session_list
                   if getattr(s, 'user_id', '') == user_id)
