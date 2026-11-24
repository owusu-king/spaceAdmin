"""Keeping the space and people records current without moving files by hand.

    source ──version?──► changed? ──► download ──► validate ──► compare ──► snapshot ──► the app switches to it
    (cheap check)                     (only then)   (loader)     (what changed)  data/records/<time>/

Sources (FAI_RECORDS_SOURCE)
  folder     a folder the reports land in by themselves: a OneDrive/SharePoint folder synced to this computer, or the
             target of a scheduled TRIRIGA/HR export (FAI_RECORDS_DIR; default datares/). File names are
             FAI_RECORDS_SPACE_FILE / FAI_RECORDS_PEOPLE_FILE (default SpaceRecords.xlsx, Report_PeopleRecords.xlsx).
  microsoft  the two reports in SharePoint or OneDrive, read through Microsoft Graph with the app's Microsoft 365
             connection (FAI_RECORDS_SPACE_URL, FAI_RECORDS_PEOPLE_URL: the files' links). Only file metadata is
             read until a file changes.

A new version is accepted only if both files pass the same validation as start-up (required columns, rows).
A failed export never replaces good data. Every accepted version is kept as a snapshot, with a summary of what
changed (rooms added / removed / changed, people added / removed / moved). FAI_RECORDS_REFRESH_MINUTES schedules
the check (0 = only when "Check now" is pressed).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import DATA_DIR, PEOPLE_FILE, SOURCE_DIR, SPACE_FILE
from app.logging_setup import get_logger, kv

log = get_logger("data.records")
RECORDS_DIR = DATA_DIR / "records"
CURRENT = RECORDS_DIR / "current.json"
SPACE_NAME, PEOPLE_NAME = SPACE_FILE.name, PEOPLE_FILE.name
WATCHED = ("seat_status", "allocation", "area", "space_class", "plan_reservation")


def active_files() -> tuple[Path, Path]:
    """The record files the app uses: the newest accepted snapshot, or the original exports in datares/."""
    if CURRENT.exists():
        cur = json.loads(CURRENT.read_text(encoding="utf-8"))
        space, people = Path(cur["space_file"]), Path(cur["people_file"])
        if space.exists() and people.exists():
            return space, people
    return SPACE_FILE, PEOPLE_FILE


# ------------------------------------------------------------------------------ sources
class FolderSource:
    kind = "folder"

    def __init__(self, folder: Path | None = None, names: tuple[str, str] | None = None) -> None:
        e = os.environ
        self.folder = Path(folder or e.get("FAI_RECORDS_DIR") or SOURCE_DIR)
        self.names = names or (e.get("FAI_RECORDS_SPACE_FILE") or SPACE_NAME, e.get("FAI_RECORDS_PEOPLE_FILE") or PEOPLE_NAME)

    def describe(self) -> dict:
        return {"kind": self.kind, "where": str(self.folder), "files": list(self.names)}

    def versions(self) -> dict:
        out = {}
        for role, name in zip(("space", "people"), self.names):
            f = self.folder / name
            if not f.exists():
                raise FileNotFoundError(f"{name} not found in {self.folder}")
            st = f.stat()
            out[role] = {"name": name, "etag": f"{st.st_mtime_ns}-{st.st_size}",
                         "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
        return out

    def download(self, role: str, target: Path) -> None:
        shutil.copy2(self.folder / self.names[0 if role == "space" else 1], target)


class MicrosoftSource:
    kind = "microsoft"

    def __init__(self, auth, urls: dict | None = None) -> None:
        from app.integrations.msgraph import GraphFile
        e = os.environ
        self.urls = urls or {"space": e.get("FAI_RECORDS_SPACE_URL", ""), "people": e.get("FAI_RECORDS_PEOPLE_URL", "")}
        self.files = {k: GraphFile(auth, u) for k, u in self.urls.items() if u}

    def describe(self) -> dict:
        return {"kind": self.kind, "where": "SharePoint / OneDrive", "files": [u.split("/")[-1][:60] for u in self.urls.values() if u]}

    def versions(self) -> dict:
        if len(self.files) < 2:
            raise ValueError("Set FAI_RECORDS_SPACE_URL and FAI_RECORDS_PEOPLE_URL in .env")
        return {k: f.version() for k, f in self.files.items()}

    def download(self, role: str, target: Path) -> None:
        target.write_bytes(self.files[role].download())


class UploadSource:
    """Files James uploaded in the app. Automatic checks are off for this source: nothing replaces an upload but
    another upload (or choosing a different source)."""
    kind = "upload"

    def __init__(self, space: bytes | None = None, people: bytes | None = None) -> None:
        self.data = {"space": space, "people": people}

    def describe(self) -> dict:
        return {"kind": self.kind, "where": "Uploaded in the app", "files": [SPACE_NAME, PEOPLE_NAME]}

    def versions(self) -> dict:
        import hashlib
        if not (self.data["space"] and self.data["people"]):
            raise ValueError("Upload both reports")
        return {k: {"name": k, "etag": hashlib.sha1(v).hexdigest()} for k, v in self.data.items()}

    def download(self, role: str, target: Path) -> None:
        target.write_bytes(self.data[role])


def records_config(settings=None) -> dict:
    """Where the records come from: what James chose in the app, else .env."""
    e, c = os.environ, (settings.get_choice if settings else (lambda k: None))
    return {"source": (c("records_source") or e.get("FAI_RECORDS_SOURCE") or "folder").lower(),
            "dir": c("records_dir") or e.get("FAI_RECORDS_DIR") or str(SOURCE_DIR),
            "space_url": c("records_space_url") or e.get("FAI_RECORDS_SPACE_URL", ""),
            "people_url": c("records_people_url") or e.get("FAI_RECORDS_PEOPLE_URL", ""),
            "refresh_minutes": int(c("records_refresh_minutes") or e.get("FAI_RECORDS_REFRESH_MINUTES") or 0)}


def build_records_source(graph_auth=None, settings=None):
    cfg = records_config(settings)
    if cfg["source"] == "microsoft":
        from app.integrations.msgraph import GraphAuth
        return MicrosoftSource(graph_auth or GraphAuth(), {"space": cfg["space_url"], "people": cfg["people_url"]})
    if cfg["source"] == "upload":
        return UploadSource()
    return FolderSource(Path(cfg["dir"]))


# ------------------------------------------------------------------------------ compare
def compare(old, new) -> dict:
    """What changed between two datasets: rooms (by space id) and people (by NetID)."""
    o, n = old.spaces.set_index("space_id"), new.spaces.set_index("space_id")
    added, removed = sorted(set(n.index) - set(o.index)), sorted(set(o.index) - set(n.index))
    common = n.index.intersection(o.index)
    changed = []
    for col in WATCHED:
        a, b = o.loc[common, col].astype("string").fillna(""), n.loc[common, col].astype("string").fillna("")
        for sid in a.index[a != b]:
            changed.append({"space_id": sid, "field": col, "from": a[sid], "to": b[sid]})
    po = old.people.dropna(subset=["netid"]).drop_duplicates("netid").set_index("netid")
    pn = new.people.dropna(subset=["netid"]).drop_duplicates("netid").set_index("netid")
    p_added, p_removed = sorted(set(pn.index) - set(po.index)), sorted(set(po.index) - set(pn.index))
    both = pn.index.intersection(po.index)
    ra, rb = po.loc[both, "room_key"].astype("string").fillna(""), pn.loc[both, "room_key"].astype("string").fillna("")
    moved = [{"netid": k, "name": pn.loc[k, "name"], "from": ra[k], "to": rb[k]} for k in ra.index[ra != rb]]
    return {"spaces": {"added": len(added), "removed": len(removed), "changed": len({c["space_id"] for c in changed}),
                       "examples": {"added": added[:8], "removed": removed[:8], "changed": changed[:10]}},
            "people": {"added": len(p_added), "removed": len(p_removed), "moved": len(moved),
                       "examples": {"moved": moved[:8]}},
            "unchanged": not (added or removed or changed or p_added or p_removed or moved)}


# ------------------------------------------------------------------------------ pipeline
class RecordsPipeline:
    """check() looks for a new version; refresh() takes it in. on_new_dataset(ds) lets the app switch to the new data."""

    def __init__(self, source, current_ds_getter, on_new_dataset=None, state_file: Path | None = None) -> None:
        self.source, self.current, self.on_new = source, current_ds_getter, on_new_dataset
        self.state_file = state_file or RECORDS_DIR / "state.json"
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.exists() else {}
        self._lock = threading.Lock()

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")

    def status(self) -> dict:
        ds = self.current()
        return {"source": self.source.describe(), "active": [str(p) for p in active_files()],
                "records": {"spaces": int(len(ds.spaces)), "people": int(len(ds.people))},
                **{k: self.state.get(k) for k in ("last_check", "last_refresh", "versions", "last_diff", "last_error",
                                                  "without_model_history")}}

    def refresh(self, force: bool = False, source=None) -> dict:
        """Download, validate, compare and switch, but only when a file changed (or force).
        source: use this source once (an upload) instead of the configured one."""
        from app.data.loader import DataValidationError, load_dataset
        src = source or self.source
        with self._lock:
            self.state["last_check"] = datetime.now().isoformat(timespec="seconds")
            try:
                versions = src.versions()
            except Exception as exc:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._save()
                return {"changed": False, "error": self.state["last_error"]}
            if not force and versions == self.state.get("versions"):
                self._save()
                return {"changed": False, "reason": "no new version"}
            snap = RECORDS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
            snap.mkdir(parents=True, exist_ok=True)
            space_f, people_f = snap / SPACE_NAME, snap / PEOPLE_NAME
            try:
                src.download("space", space_f)
                src.download("people", people_f)
                new = load_dataset(use_cache=False, space_file=space_f, people_file=people_f, write_cache=False)
            except (DataValidationError, Exception) as exc:   # a broken export never replaces good data
                shutil.rmtree(snap, ignore_errors=True)
                self.state["last_error"] = f"Rejected the new files: {type(exc).__name__}: {exc}"
                self._save()
                log.warning("records refresh rejected " + kv(error=repr(exc)[:200]))
                return {"changed": False, "error": self.state["last_error"]}
            diff = compare(self.current(), new)
            CURRENT.write_text(json.dumps({"space_file": str(space_f), "people_file": str(people_f),
                                           "accepted": datetime.now().isoformat(timespec="seconds"), "versions": versions}),
                               encoding="utf-8")
            self.state.update(versions=versions, last_refresh=datetime.now().isoformat(timespec="seconds"),
                              last_diff=diff, last_error=None)
            self._save()
            log.info("records refreshed " + kv(spaces=len(new.spaces), people=len(new.people),
                                               rooms_changed=diff["spaces"]["changed"], people_moved=diff["people"]["moved"]))
        if self.on_new:
            info = self.on_new(new)
            if info:
                self.state.update(info)
                self._save()
        return {"changed": True, "diff": diff}
