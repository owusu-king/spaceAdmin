"""Building resolution: crosswalk between floor-plan codes, full names and export names.

Rules (architecture §7.5): exact aliases first, then normalized matching; an alias that
fits several buildings returns `ambiguous` and is never auto-resolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import AMBIGUOUS_ALIASES, BUILDINGS, FLOOR_PLAN_DIRS


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‑", "-").lower()
    return re.sub(r"[^a-z0-9' ]+", " ", text).strip()


@dataclass
class BuildingMatch:
    status: str                 # resolved | ambiguous | not_found
    export_name: str | None = None
    code: str | None = None
    name: str | None = None
    matched_alias: str | None = None
    candidates: list[str] | None = None


class BuildingResolver:
    def __init__(self, export_names: list[str] | None = None) -> None:
        self.by_export = {b["export"]: b for b in BUILDINGS}
        self.by_code = {b["code"]: b for b in BUILDINGS}
        aliases: dict[str, str] = {}
        for b in BUILDINGS:
            for alias in [b["name"], b["export"], *b["aliases"]]:
                aliases[_norm(alias)] = b["export"]
        for name in export_names or []:
            aliases.setdefault(_norm(name), name)
        # longest alias first so "brockman hall for physics" wins over shorter forms
        self._aliases = sorted(aliases.items(), key=lambda kv: -len(kv[0]))
        self._codes = {b["code"]: b["export"] for b in BUILDINGS}

    def info(self, export_name: str) -> dict:
        b = self.by_export.get(export_name)
        return {"export_name": export_name, "code": b["code"] if b else None,
                "name": b["name"] if b else export_name, "map": b["map"] if b else None,
                "in_crosswalk": b is not None}

    def resolve(self, text: str) -> BuildingMatch:
        """Resolve one building reference (a name, alias or code)."""
        n = _norm(text)
        if text.strip().upper() in self._codes:
            return self._hit(self._codes[text.strip().upper()], text.strip().upper())
        for alias, export in self._aliases:
            if n == alias:
                return self._hit(export, alias)
        if n in AMBIGUOUS_ALIASES:
            return BuildingMatch("ambiguous", candidates=AMBIGUOUS_ALIASES[n], matched_alias=n)
        return BuildingMatch("not_found")

    def find_in_text(self, text: str) -> list[BuildingMatch]:
        """Find every crosswalk building mentioned in free text (notice bodies, subjects)."""
        n = " " + _norm(text) + " "
        found: dict[str, BuildingMatch] = {}
        consumed = n
        for alias, export in self._aliases:
            if export not in self.by_export or len(alias) < 3:
                continue
            if f" {alias} " in consumed:
                found.setdefault(export, self._hit(export, alias))
                consumed = consumed.replace(f" {alias} ", " ")
        for code, export in self._codes.items():
            if re.search(rf"\b{code}\b", text):
                found.setdefault(export, self._hit(export, code))
        for word, cands in AMBIGUOUS_ALIASES.items():
            if f" {word} " in consumed and not any(c in found for c in cands):
                found[f"?{word}"] = BuildingMatch("ambiguous", candidates=cands, matched_alias=word)
        return list(found.values())

    def _hit(self, export: str, alias: str) -> BuildingMatch:
        b = self.by_export.get(export)
        return BuildingMatch("resolved", export_name=export, code=b["code"] if b else None,
                             name=b["name"] if b else export, matched_alias=alias)


def floor_plan_files() -> list[dict]:
    """Index floor-plan PDFs as (code, floor, snapshot, path). Filenames are kept verbatim."""
    out = []
    for root in FLOOR_PLAN_DIRS:
        for pdf in sorted(Path(root).rglob("*.pdf")):
            parent = pdf.parent.name.upper()
            m = re.match(r"([A-Z]+)\s+(\w+)\s+([A-Z]{3})\s+(\d{4})", pdf.stem.upper())
            if m:
                code, floor, snap = m.group(1), m.group(2), f"{m.group(3).title()} {m.group(4)}"
            else:  # HRZ folder holds bare floor names such as "01.pdf"
                code = parent.split()[0]
                floor = pdf.stem.upper()
                ms = re.search(r"([A-Z]{3})\s+(\d{4})", parent)
                snap = f"{ms.group(1).title()} {ms.group(2)}" if ms else "unknown"
            out.append({"code": code, "floor": floor, "snapshot": snap,
                        "path": str(pdf), "file": pdf.name})
    return out
