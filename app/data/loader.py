"""Load, validate and normalize the space and people exports.

Original string values are preserved (leading zeros, suffixes, basement prefixes); derived
columns are added next to them. The result is cached in data/ keyed by source mtimes so
the app starts quickly after the first run.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import (CLASS_GROUP_BY_PREFIX, CLASS_GROUPS, DATA_DIR, MANAGEMENT_ROLE_PREFIXES,
                        OCCUPANT_GROUPS, PEOPLE_FILE, ROLE_GROUPS, SPACE_FILE)
from app.logging_setup import get_logger, timed

log = get_logger("data")

SPACE_COLUMNS = ["Plan Rsvn", "Seat Stat", "Avble Seats", "Building", "Floor", "Space", "Area",
                 "W/S Area", "Space Class", "SSA", "spaceStandard", "occupancyAllocationData",
                 "chargebackAllocationData"]
PEOPLE_COLUMNS = ["SSA", "Net ID", "Name", "Title", "Department", "Functional Role",
                  "Building", "Floor", "Space"]
CACHE_VERSION = 3


class DataValidationError(ValueError):
    """Raised when a source file does not have the expected structure."""


def class_group(space_class: str | float) -> str:
    """Map a Space Class such as '250 - Research/Nonclass Laboratory' to its functional group."""
    if not isinstance(space_class, str) or not space_class.strip():
        return "Unknown"
    code = space_class.split(" - ")[0].strip()
    return CLASS_GROUPS.get(code) or CLASS_GROUP_BY_PREFIX.get(code[:1], "Special & General Use")


def role_group(role: str | float) -> str:
    if not isinstance(role, str):
        return "staff"
    return ROLE_GROUPS.get(role, "staff")


def room_key(building: pd.Series, floor: pd.Series, room: pd.Series) -> pd.Series:
    return building.str.strip() + " · " + floor.str.strip() + " · " + room.str.strip()


def _validate(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")
    if df.empty:
        raise DataValidationError(f"{name}: no rows")


def _clean(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


def normalize_spaces(raw: pd.DataFrame) -> pd.DataFrame:
    _validate(raw, SPACE_COLUMNS, "SpaceRecords")
    df = pd.DataFrame({
        "building": _clean(raw["Building"]), "floor": _clean(raw["Floor"]), "room": _clean(raw["Space"]),
        "area": pd.to_numeric(raw["Area"], errors="coerce"),
        "ws_area": pd.to_numeric(raw["W/S Area"], errors="coerce"),
        "space_class": _clean(raw["Space Class"]), "space_standard": _clean(raw["spaceStandard"]),
        "ssa": _clean(raw["SSA"]), "seat_status": _clean(raw["Seat Stat"]),
        "available_seats": pd.to_numeric(raw["Avble Seats"], errors="coerce"),
        "plan_reservation": _clean(raw["Plan Rsvn"]).str.upper().eq("TRUE").fillna(False),
        "allocation": _clean(raw["occupancyAllocationData"]),
        "chargeback": _clean(raw["chargebackAllocationData"]),
    })
    bad = df["building"].isna() | df["floor"].isna() | df["room"].isna()
    if bad.any():
        log.warning(f"dropping {int(bad.sum())} space rows without building/floor/room")
        df = df[~bad].copy()
    df["room_key"] = room_key(df["building"], df["floor"], df["room"])
    dup_rank = df.groupby("room_key").cumcount()
    df["duplicate_key"] = df["room_key"].duplicated(keep=False)
    # Repeated room keys are investigated, not merged: each row keeps its own id.
    df["space_id"] = np.where(dup_rank > 0, df["room_key"] + " #" + (dup_rank + 1).astype(str), df["room_key"])
    df["class_group"] = df["space_class"].map(class_group)
    df["allocation_primary"] = df["allocation"].str.split(", ").str[0]
    df["allocation_count"] = df["allocation"].str.count(", ").fillna(-1).astype(int) + 1
    return df.reset_index(drop=True)


def normalize_people(raw: pd.DataFrame) -> pd.DataFrame:
    _validate(raw, PEOPLE_COLUMNS, "PeopleRecords")
    df = pd.DataFrame({
        "netid": _clean(raw["Net ID"]), "name": _clean(raw["Name"]), "title": _clean(raw["Title"]),
        "department": _clean(raw["Department"]), "role": _clean(raw["Functional Role"]),
        "ssa": _clean(raw["SSA"]), "building": _clean(raw["Building"]),
        "floor": _clean(raw["Floor"]), "room": _clean(raw["Space"]),
    })
    df = df[df["netid"].notna()].copy()
    df["email"] = df["netid"] + "@rice.edu"     # rule stated by James: NetID + @rice.edu
    df["role_group"] = df["role"].map(role_group)
    df["is_management"] = df["role"].fillna("").str.startswith(MANAGEMENT_ROLE_PREFIXES)
    located = df["building"].notna() & df["floor"].notna() & df["room"].notna()
    df["room_key"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[located, "room_key"] = room_key(df.loc[located, "building"], df.loc[located, "floor"],
                                           df.loc[located, "room"])
    return df.reset_index(drop=True)


def occupant_profile(spaces: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Count recorded occupants per room key by occupant group (people records join)."""
    located = people.dropna(subset=["room_key"])
    counts = pd.crosstab(located["room_key"], located["role_group"])
    counts = counts.reindex(columns=OCCUPANT_GROUPS, fill_value=0).add_prefix("occ_")
    prof = spaces[["room_key"]].join(counts, on="room_key").fillna(0)
    prof["occ_total"] = prof[[f"occ_{g}" for g in OCCUPANT_GROUPS]].sum(axis=1)
    return prof.drop(columns="room_key").astype(int)


@dataclass
class Dataset:
    spaces: pd.DataFrame
    people: pd.DataFrame
    report: dict = field(default_factory=dict)

    def space(self, space_id: str) -> pd.Series | None:
        hit = self.spaces[self.spaces["space_id"] == space_id]
        return None if hit.empty else hit.iloc[0]

    def occupants(self, room_key: str) -> pd.DataFrame:
        return self.people[self.people["room_key"] == room_key]


def _quality_report(spaces: pd.DataFrame, people: pd.DataFrame) -> dict:
    people_located = people["room_key"].notna()
    matched = people.loc[people_located, "room_key"].isin(set(spaces["room_key"]))
    return {
        "space_rows": int(len(spaces)), "people_rows": int(len(people)),
        "buildings": int(spaces["building"].nunique()),
        "duplicate_room_keys": int(spaces.loc[spaces["duplicate_key"], "room_key"].nunique()),
        "spaces_missing_area": int(spaces["area"].isna().sum()),
        "spaces_missing_standard": int(spaces["space_standard"].isna().sum()),
        "spaces_missing_allocation": int(spaces["allocation"].isna().sum()),
        "planning_reservations": int(spaces["plan_reservation"].sum()),
        "people_with_location": int(people_located.sum()),
        "people_location_unmatched": int((~matched).sum()),
    }


def load_dataset(use_cache: bool = True, space_file=None, people_file=None, write_cache: bool = True) -> Dataset:
    """Load and validate the records. By default the active version (newest accepted snapshot from the records
    pipeline, else the original exports in datares/); explicit files are used to validate a new version."""
    if space_file is None or people_file is None:
        from app.data.records import active_files
        space_file, people_file = active_files()
    space_file, people_file = Path(space_file), Path(people_file)
    for f in (space_file, people_file):
        if not f.exists():
            raise FileNotFoundError(f"Source file not found: {f}")
    stamp = (CACHE_VERSION, str(space_file), space_file.stat().st_mtime, str(people_file), people_file.stat().st_mtime)
    cache = DATA_DIR / "dataset_cache.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as fh:
            cached = pickle.load(fh)
        if cached.get("stamp") == stamp:
            log.info(f"dataset loaded from cache spaces={len(cached['ds'].spaces)} people={len(cached['ds'].people)}")
            return cached["ds"]
    with timed(log, "dataset loaded from source"):
        spaces = normalize_spaces(pd.read_excel(space_file, dtype=str))
        people = normalize_people(pd.read_excel(people_file, dtype=str))
        spaces = pd.concat([spaces, occupant_profile(spaces, people)], axis=1)
        ds = Dataset(spaces, people, _quality_report(spaces, people))
    log.info("data quality " + " ".join(f"{k}={v}" for k, v in ds.report.items()))
    if not write_cache:
        return ds
    DATA_DIR.mkdir(exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump({"stamp": stamp, "ds": ds}, fh)
    return ds
