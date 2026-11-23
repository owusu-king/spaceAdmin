"""Feature switches: explicit, audited, off by default.

A switch lets a *person* perform an action the agents are never allowed to perform (the
agents' LOCKED_ACTIONS in app/agents/base.py stay locked whatever the switches say). Two
keys are needed to turn a switch on:

  1. The machine allows it: FAI_ALLOW_UNLOCK=true in .env (set by whoever runs the server).
  2. The integration behind it is configured (e.g. outgoing mail settings for send_email).

Turning a switch off is always allowed. A switch can turn itself off after a chosen time.
Every change is written to the audit log with who made it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from app.logging_setup import get_logger

log = get_logger("settings")
AUTO_OFF_MINUTES = [15, 60, 480]          # offered in the UI; None = stays on until turned off


def unlock_allowed() -> bool:
    return os.environ.get("FAI_ALLOW_UNLOCK", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Switch:
    key: str
    label: str
    description: str
    group: str
    requires: str = ""                    # plain-language prerequisite shown in the UI
    depends_on: str | None = None         # another switch that must be on first
    needs_unlock: bool = True             # locked actions need FAI_ALLOW_UNLOCK; local features do not


SWITCHES: list[Switch] = [
    # Short on purpose: the Settings page shows one line per switch; the reason a switch is unavailable is the tooltip.
    Switch("voice_assistant", "Voice assistant", "Talk to the Space agent in a new space request. Runs on this computer.",
           "Voice", "The local models chosen below", needs_unlock=False),
    Switch("send_email", "Send email", "Send approved drafts from a case.", "Actions",
           "Outgoing mail settings (FAI_SMTP_*) in .env"),
    Switch("email_live_recipients", "Real recipients", "When off, emails go to the test address only.", "Actions",
           depends_on="send_email"),
    Switch("write_tririga", "Write to TRIRIGA", "Update outage records.", "Actions", "A TRIRIGA integration (not built)"),
    Switch("assign_space", "Assign space", "Record space assignments.", "Actions",
           "A space-management integration (not built)"),
    Switch("approve_outage", "Approve outages", "Record outage approvals.", "Actions", "A TRIRIGA integration (not built)"),
]
BY_KEY = {s.key: s for s in SWITCHES}


class SettingsStore:
    """Switch state in the application database (same connection and lock as the case store)."""

    def __init__(self, store, checks: dict[str, Callable[[], tuple[bool, str]]] | None = None) -> None:
        self.store, self.db = store, store.db
        self.checks = checks or {}         # key -> () -> (integration ready?, reason if not)
        self.db.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value INTEGER, expires TEXT,
            updated TEXT, updated_by TEXT)""")
        self.db.commit()

    # ------------------------------------------------------------------ queries
    def _row(self, key: str) -> dict | None:
        # Locked: background threads (voice preloading) read choices on the same SQLite connection.
        with self.store._lock:
            r = self.db.execute("SELECT * FROM settings WHERE key=?", (key,)).fetchone()
        return dict(r) if r else None

    def availability(self, key: str) -> tuple[bool, str]:
        """Whether the switch may be turned on now, and why not."""
        sw = BY_KEY[key]
        if sw.needs_unlock and not unlock_allowed():
            return False, "Unlocking is disabled on this machine (set FAI_ALLOW_UNLOCK=true in .env)"
        if sw.depends_on and not self.is_on(sw.depends_on):
            return False, f"Turn on '{BY_KEY[sw.depends_on].label}' first"
        check = self.checks.get(key)
        if check is None and sw.requires:
            return False, f"Needs {sw.requires[0].lower()}{sw.requires[1:]}"
        if check:
            ok, why = check()
            if not ok:
                return False, why
        return True, ""

    def is_on(self, key: str) -> bool:
        r = self._row(key)
        if not r or not r["value"]:
            return False
        if r["expires"] and datetime.fromisoformat(r["expires"]) <= datetime.now():
            self._write(key, False, None, "system (timer)")
            self.store.audit("system", "switch_expired", detail=key)
            return False
        if not self.availability(key)[0]:      # prerequisites removed since it was turned on
            return False
        return True

    def list(self) -> list[dict]:
        out = []
        for sw in SWITCHES:
            on = self.is_on(sw.key)
            ok, why = self.availability(sw.key)
            r = self._row(sw.key) or {}
            out.append({"key": sw.key, "label": sw.label, "description": sw.description, "group": sw.group,
                        "requires": sw.requires, "depends_on": sw.depends_on, "on": on, "available": ok,
                        "reason": why, "expires": r.get("expires") if on else None,
                        "updated": r.get("updated"), "updated_by": r.get("updated_by")})
        return out

    # ------------------------------------------------------------------ changes
    def set(self, key: str, on: bool, actor: str, minutes: int | None = None) -> dict:
        if key not in BY_KEY:
            raise KeyError(key)
        if on:
            ok, why = self.availability(key)
            if not ok:
                raise PermissionError(why)
            if minutes is not None and minutes not in AUTO_OFF_MINUTES:
                raise ValueError(f"minutes must be one of {AUTO_OFF_MINUTES} or empty")
        expires = (datetime.now() + timedelta(minutes=minutes)).isoformat(timespec="seconds") if on and minutes else None
        self._write(key, on, expires, actor)
        if not on:                             # switches that depend on this one go off with it
            for dep in (s.key for s in SWITCHES if s.depends_on == key):
                if self._row(dep) and self._row(dep)["value"]:
                    self._write(dep, False, None, actor)
                    self.store.audit(actor, "switch_off", detail=f"{dep} (with {key})")
        self.store.audit(actor, "switch_on" if on else "switch_off", detail=key + (f" until {expires}" if expires else ""))
        log.info(f"switch key={key} on={on} expires={expires} by={actor}")
        return next(s for s in self.list() if s["key"] == key)

    # ------------------------------------------------------------------ choices (e.g. which installed model to use)
    def get_choice(self, key: str) -> str | None:
        r = self._row("choice:" + key)
        return r["value"] if r else None

    def set_choice(self, key: str, value: str, actor: str) -> None:
        with self.store._lock:
            self.db.execute("INSERT INTO settings (key, value, expires, updated, updated_by) VALUES (?,?,?,?,?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated, "
                            "updated_by=excluded.updated_by",
                            ("choice:" + key, value, None, datetime.now().isoformat(timespec="seconds"), actor))
            self.db.commit()
        self.store.audit(actor, "setting_changed", detail=f"{key} = {value}")

    def _write(self, key: str, on: bool, expires: str | None, actor: str) -> None:
        with self.store._lock:
            self.db.execute("INSERT INTO settings (key, value, expires, updated, updated_by) VALUES (?,?,?,?,?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, expires=excluded.expires, "
                            "updated=excluded.updated, updated_by=excluded.updated_by",
                            (key, int(on), expires, datetime.now().isoformat(timespec="seconds"), actor))
            self.db.commit()
