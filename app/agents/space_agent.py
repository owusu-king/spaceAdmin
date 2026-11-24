"""Space Administrator Agent (Space Assignment Support).

Objective: find suitable space and facilitate assignment review. It does NOT schedule or
book rooms, and it never assigns space: it returns ranked candidates, constraints,
verification questions and the approvals the institution must still give.

Decision pipeline
  resolve department/buildings -> discover candidates -> constraints (hard) -> six weighted
  criteria -> model corroboration (deep learning, inference) -> status -> questions/approvals

Suitability and evidence coverage are reported separately (architecture §10.4) so that
missing information can never produce a misleadingly high score.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.agents.base import Agent, Evidence, ToolResult, tool
from app.agents.schemas import SpaceRequest
from app.config import AGENTS, NATURAL_SCIENCES_DEPARTMENTS
from app.data.buildings import BuildingResolver, floor_plan_files
from app.data.loader import Dataset
from app.ml.inference import SpacePredictor

COMPATIBLE = {
    "Office": ["Office"], "Workstation": ["Workstation"], "Research Lab": ["Research Lab"],
    "Teaching Lab": ["Teaching Lab"], "Meeting": ["Meeting"],
    "Support & Storage": ["Support & Storage", "Office Service"],
}
WORKPOINT_FUNCTIONS = {"Office", "Workstation"}
OPEN_SEAT_STATUS = {"Vacant", "Under Capacity"}
LAB_UNRECORDED = ["ventilation and air-change rate", "laboratory utilities (gases, vacuum, water, power)",
                  "safety and hazardous-materials requirements", "bench and equipment space"]


class SpaceAdministratorAgent(Agent):
    name = "space_admin"
    objectives = [
        "Find spaces that fit a need.",
        "Separate recorded vacancy from availability.",
        "Flag rooms whose use may differ from the record.",
        "Prepare questions and approvals; never assign.",
    ]
    permissions = {"resolve_building", "get_department", "get_person", "find_person", "measure_proximity", "search_spaces", "get_space_details",
                   "get_space_occupancy", "check_planning_reservations", "get_floor_plan",
                   "model_space_assessment", "evaluate_space_constraints", "rank_candidate_spaces"}
    data_scopes = ["Space records (read-only)", "People records: name, NetID, department, role, location",
                   "Floor-plan index", "Space-function model predictions"]

    def __init__(self, ds: Dataset, predictor: SpacePredictor, resolver: BuildingResolver) -> None:
        self.ds, self.predictor, self.resolver = ds, predictor, resolver
        depts = set(ds.people["department"].dropna())
        for a in ds.spaces["allocation"].dropna().unique():
            depts.update(x.strip() for x in a.split(", "))
        self.departments = sorted(depts)
        self.plans = floor_plan_files()
        super().__init__()

    # =========================================================================== tools
    @tool("resolve_building")
    def resolve_building(self, text: str) -> ToolResult:
        """Resolve a building name, alias or code to the space-export name."""
        m = self.resolver.resolve(text)
        if m.status == "resolved":
            return ToolResult("resolve_building", "success", m.export_name,
                              [Evidence(f"'{text}' resolves to {m.name}", "recorded", "config", f"alias:{m.matched_alias}")])
        if m.status == "ambiguous":
            return ToolResult("resolve_building", "ambiguous", m.candidates,
                              warnings=[f"'{text}' matches several buildings: {', '.join(m.candidates)}. Choose one."])
        return ToolResult("resolve_building", "not_found", None, missing_information=[f"Building '{text}' not on record"])

    @tool("get_department")
    def get_department(self, name: str) -> ToolResult:
        """Match a department name against people and allocation records."""
        low = name.lower().strip()
        exact = [d for d in self.departments if d.lower() == low]
        hits = exact or [d for d in self.departments if low in d.lower()]
        if len(hits) == 1:
            in_ns = hits[0] in NATURAL_SCIENCES_DEPARTMENTS
            return ToolResult("get_department", "success", {"name": hits[0], "natural_sciences": in_ns},
                              [Evidence(f"Department '{hits[0]}' found in institutional records", "recorded", "people_records")])
        if hits:
            return ToolResult("get_department", "ambiguous", hits[:10], warnings=[f"'{name}' matches {len(hits)} departments"])
        return ToolResult("get_department", "not_found", None, missing_information=[f"Department '{name}' not on record"])

    @tool("get_person")
    def get_person(self, netid: str) -> ToolResult:
        """Look up a person by NetID and their recorded locations."""
        p = self.ds.people[self.ds.people["netid"].str.lower() == netid.lower().strip()]
        if p.empty:
            return ToolResult("get_person", "not_found", None, missing_information=[f"NetID {netid} not on record"])
        r = p.iloc[0]
        rooms = [k for k in p["room_key"].dropna().unique()]
        return ToolResult("get_person", "success",
                          {"netid": r.netid, "name": r["name"], "department": r.department, "role": r.role,
                           "title": r.title, "email": r.email, "rooms": rooms},
                          [Evidence(f"{r['name']} recorded in {r.department} as {r.role}", "recorded", "people_records", r.netid)]
                          + [Evidence(f"{r['name']} recorded at {k}", "recorded", "people_records", k) for k in rooms])

    @tool("find_person")
    def find_person(self, query: str) -> ToolResult:
        """Find a person by NetID or name and return their recorded office location(s)."""
        q = query.strip().lower()
        people = self.ds.people
        hits = people[people["netid"].str.lower() == q]
        if hits.empty:
            tokens = [t for t in re.split(r"[\s,.]+", q) if t and t not in ("professor", "prof", "dr", "dr.")]
            mask = pd.Series(True, index=people.index)
            for t in tokens:
                mask &= people["name"].fillna("").str.lower().str.contains(t, regex=False)
            hits = people[mask] if tokens else people.iloc[0:0]
        if hits.empty:
            return ToolResult("find_person", "not_found", None, missing_information=[f"No person matching '{query}' in people records"])
        if hits["netid"].nunique() > 1 and re.search(r"\b(prof|professor|dr)\b", q):
            faculty = hits[hits["role"].fillna("").str.contains("Faculty")]
            if faculty["netid"].nunique() == 1:      # "Professor X": the only faculty match is the reference
                hits = faculty
        if hits["netid"].nunique() > 1:
            opts = [f"{r['name']} ({r.netid}, {r.department})" for _, r in hits.drop_duplicates("netid").head(6).iterrows()]
            return ToolResult("find_person", "ambiguous", opts, warnings=[f"'{query}' matches {hits['netid'].nunique()} people"])
        r = hits.iloc[0]
        rooms = sorted(hits["room_key"].dropna().unique())
        ev = [Evidence(f"{r['name']} ({r.department}, {r.role}) recorded at {k}", "recorded", "people_records", k) for k in rooms]
        return ToolResult("find_person", "success" if rooms else "partial",
                          {"name": r["name"], "netid": r.netid, "department": r.department, "role": r.role, "rooms": rooms},
                          ev, missing_information=[] if rooms else [f"{r['name']} has no recorded office location"])

    @tool("measure_proximity")
    def measure_proximity(self, candidates: pd.DataFrame, anchors: list[str]) -> ToolResult:
        """Estimate closeness of each candidate to the reference office(s) from building, floor and room numbering."""
        refs = self.ds.spaces[self.ds.spaces["room_key"].isin(anchors)][["building", "floor", "room"]].drop_duplicates()
        out = {}
        for idx, r in candidates.iterrows():
            best = max((_proximity(r.building, r.floor, r.room, a.building, a.floor, a.room) for a in refs.itertuples()),
                       key=lambda x: x[0], default=(None, "Reference office not on record"))
            out[idx] = {"score": best[0], "label": best[1]}
        return ToolResult("measure_proximity", "success", out,
                          [Evidence("Proximity estimated from building, floor and room numbering; floor plans are not "
                                    "machine-readable, so walking distance is not measured", "inferred", "rule:proximity")])

    @tool("search_spaces")
    def search_spaces(self, function: str, buildings: list[str], open_only: bool) -> ToolResult:
        """Retrieve spaces whose recorded class is compatible with the requested function."""
        s = self.ds.spaces
        mask = s["class_group"].isin(COMPATIBLE[function])
        if buildings:
            mask &= s["building"].isin(buildings)
        if open_only and function in WORKPOINT_FUNCTIONS:
            mask &= s["seat_status"].isin(OPEN_SEAT_STATUS)
        hits = s[mask]
        status = "success" if len(hits) else "not_found"
        return ToolResult("search_spaces", status, hits.index.tolist(),
                          [Evidence(f"{len(hits)} records with a compatible recorded class", "recorded", "space_records")],
                          missing_information=[] if len(hits) else ["No compatible space on record in the searched scope"])

    @tool("get_space_details")
    def get_space_details(self, space_id: str) -> ToolResult:
        """Return the recorded attributes of one space."""
        row = self.ds.space(space_id)
        if row is None:
            return ToolResult("get_space_details", "not_found", None, missing_information=[f"{space_id} not on record"])
        return ToolResult("get_space_details", "success", _space_dict(row))

    @tool("get_space_occupancy")
    def get_space_occupancy(self, room_key: str) -> ToolResult:
        """Recorded occupants of a room (people records)."""
        occ = self.ds.occupants(room_key)
        people = occ[["name", "netid", "department", "role"]].to_dict("records")
        return ToolResult("get_space_occupancy", "success", people,
                          [Evidence(f"{len(people)} recorded occupant(s) at {room_key}", "recorded", "people_records", room_key)])

    @tool("check_planning_reservations")
    def check_planning_reservations(self, indices: list[int]) -> ToolResult:
        """Flag spaces earmarked for an upcoming space change (Plan Rsvn)."""
        s = self.ds.spaces.loc[indices]
        flagged = s[s["plan_reservation"]]["space_id"].tolist()
        return ToolResult("check_planning_reservations", "success", flagged,
                          [Evidence(f"{sid} carries a planning reservation (future space change)", "recorded", "space_records", sid)
                           for sid in flagged])

    @tool("get_floor_plan")
    def get_floor_plan(self, building: str, floor: str) -> ToolResult:
        """Find the floor-plan PDF for a building floor."""
        code = self.resolver.info(building)["code"]
        hits = [p for p in self.plans if p["code"] == code and p["floor"] == floor]
        if not hits:
            return ToolResult("get_floor_plan", "not_found", None, missing_information=[f"No floor plan on file for {building} floor {floor}"])
        return ToolResult("get_floor_plan", "success", hits[-1],
                          [Evidence(f"Floor plan {hits[-1]['file']} ({hits[-1]['snapshot']}). Room location not extracted; verify visually.",
                                    "historical", "floor_plans", hits[-1]["file"])])

    @tool("model_space_assessment")
    def model_space_assessment(self, space_ids: list[str], function: str) -> ToolResult:
        """Deep-learning view of candidates: p(requested function) and actual-use consistency."""
        out, ev = {}, []
        for sid in space_ids:
            a = self.predictor.assessment(sid)
            if a["status"] != "success":
                continue
            a["p_function"] = round(float(self.predictor.probability([sid], function).iloc[0]), 3)
            out[sid] = a
            if a["verify_flag"]:
                ev.append(Evidence(f"Model infers {a['predicted_group']} ({a['confidence']:.0%}) for {sid}, recorded as "
                                   f"{a['recorded_group']}. Recorded class may not reflect actual use.", "inferred", "model", sid))
        return ToolResult("model_space_assessment", "success", out, ev)

    @tool("evaluate_space_constraints")
    def evaluate_space_constraints(self, candidates: pd.DataFrame, req: SpaceRequest) -> ToolResult:
        """Hard constraints: function, minimum area, recorded capacity."""
        res = {}
        for idx, r in candidates.iterrows():
            failed, unknown = [], []
            if req.min_area:
                if pd.isna(r.area):
                    unknown.append("Area not on record")
                elif r.area < req.min_area:
                    failed.append(f"Area {r.area:.0f} sf below required {req.min_area:.0f} sf")
            if req.function in WORKPOINT_FUNCTIONS and req.request_type != "vacancy_review":
                if r.seat_status in ("At Capacity", "Over Capacity"):
                    failed.append(f"Seating status recorded as {r.seat_status}")
                elif r.seat_status == "Under Capacity":
                    seats = r.available_seats if pd.notna(r.available_seats) else None
                    if seats is not None and seats < req.seats:
                        failed.append(f"{int(seats)} recorded open seat(s), {req.seats} needed")
            res[idx] = {"failed": failed, "unknown": unknown}
        return ToolResult("evaluate_space_constraints", "success", res)

    @tool("rank_candidate_spaces")
    def rank_candidate_spaces(self, candidates: pd.DataFrame, req: SpaceRequest, dept_buildings: set[str],
                              model: dict, constraints: dict, use_model: bool = True,
                              proximity: dict | None = None) -> ToolResult:
        """Weighted suitability S = sum(w_i s_i) over known criteria, with separate evidence coverage.
        With a reference office, 'proximity' replaces the generic location criterion and carries more weight."""
        w = dict(AGENTS.ranking_weights)
        if proximity is not None:
            w.pop("location")
            w["proximity"] = AGENTS.proximity_weight
            total = sum(w.values())
            w = {k: v / total for k, v in w.items()}
        rows = []
        for idx, r in candidates.iterrows():
            m = model.get(r.space_id, {})
            crit: dict[str, float | None] = {}
            p_fn = m.get("p_function") if use_model else None
            crit["functional"] = 1.0 if p_fn is None else 0.6 + 0.4 * p_fn
            crit["area"] = _area_score(r.area, req)
            crit["occupancy"] = _occupancy_score(r)
            crit["department"] = _department_score(r.allocation, req.department)
            if proximity is not None:
                crit["proximity"] = proximity[idx]["score"]
            elif req.preferred_buildings:
                crit["location"] = 1.0 if r.building in req.preferred_buildings else 0.4
            else:
                crit["location"] = 0.8 if r.building in dept_buildings else (0.5 if dept_buildings else None)
            crit["planning"] = 0.0 if r.plan_reservation else 1.0
            known = {k: v for k, v in crit.items() if v is not None}
            coverage = sum(w[k] for k in known)
            suit = sum(w[k] * v for k, v in known.items()) / coverage if coverage else 0.0
            rows.append({"idx": idx, "criteria": crit, "suitability": round(suit, 3),
                         "coverage": round(coverage, 3), "constraints": constraints[idx],
                         "proximity": proximity[idx] if proximity is not None else None})
        rows.sort(key=lambda x: (-bool(not x["constraints"]["failed"]), -x["suitability"], -x["coverage"]))
        return ToolResult("rank_candidate_spaces", "success", rows)

    # =========================================================================== workflow
    def run(self, req: SpaceRequest, use_model: bool = True) -> dict:
        """Execute the full assessment for one request. Returns a structured report."""
        self.reset()
        self.log.info(f"workflow start type={req.request_type} function={req.function} dept={req.department}")
        findings: list[Evidence] = []
        questions: list[str] = []
        requirements = [{"field": "Request type", "value": req.request_type.replace("_", " "), "source": "stated"},
                        {"field": "Function", "value": req.function, "source": "stated"}]
        dept = None
        person = None
        if req.request_type == "change_assignment":
            if not req.netid:
                questions.append("Which person (NetID) needs the assignment change?")
            else:
                pr = self.call("get_person", netid=req.netid)
                findings += pr.evidence
                if pr.status == "success":
                    person = pr.data
                    req = req.model_copy(update={"department": req.department or person["department"]})
                    requirements.append({"field": "Person", "value": f"{person['name']} ({person['netid']})", "source": "stated"})
                else:
                    questions.append(f"NetID {req.netid} is not on record. Which NetID?")
        anchor = None
        if req.near:
            fr = self.call("find_person", query=req.near)
            findings += fr.evidence
            if fr.status in ("success", "partial"):
                anchor = fr.data
                if not req.department:
                    req = req.model_copy(update={"department": anchor["department"]})
                where = ", ".join(anchor["rooms"]) or "office not on record"
                requirements.append({"field": "Close to", "value": f"{anchor['name']} ({anchor['netid']}) · {where}", "source": "stated"})
                if not anchor["rooms"]:
                    questions.append(f"Where is {anchor['name']}'s office?")
            elif fr.status == "ambiguous":
                questions.append(f"Which person is meant by '{req.near}': {'; '.join(fr.data)}?")
            else:
                questions.append(f"'{req.near}' is not on record. What is their NetID?")
        if req.near and anchor is None:
            # Without the reference office a proximity ranking would be meaningless: ask first.
            findings.append(Evidence(f"Reference person '{req.near}' not resolved; ranking deferred", "unknown", "people_records"))
            report = self._report(req, requirements, [], [], findings, questions, None)
            report["summary"] = "Ranking deferred: which person is meant?"
            return report
        if req.department:
            dr = self.call("get_department", name=req.department)
            findings += dr.evidence
            if dr.status == "success":
                dept = dr.data["name"]
                requirements.append({"field": "Department", "value": dept, "source": "stated"})
            elif dr.status == "ambiguous":
                questions.append(f"Which department is meant: {', '.join(dr.data[:5])}?")
            else:
                questions.append(f"Department '{req.department}' is not on record. What is the official name?")
        else:
            questions.append("Which department is this for?")

        preferred = self._resolve_list(req.preferred_buildings, findings, questions)
        excluded = set(self._resolve_list(req.exclude_buildings, findings, questions))
        req = req.model_copy(update={"preferred_buildings": preferred, "department": dept})
        dept_buildings = self._department_buildings(dept)
        if req.min_area:
            requirements.append({"field": "Minimum area", "value": f"{req.min_area:.0f} sf", "source": "stated"})
        if req.seats > 1:
            requirements.append({"field": "Seats", "value": str(req.seats), "source": "stated"})
        anchor_buildings = sorted({k.split(" · ")[0] for k in (anchor or {}).get("rooms", [])})
        scope = preferred or sorted(set(anchor_buildings) | dept_buildings) or []
        requirements.append({"field": "Search scope", "value": ", ".join(scope) if scope else "All buildings on record",
                             "source": "stated" if preferred else "inferred"})

        if req.request_type == "vacancy_review":
            return self._vacancy_review(req, dept, scope, findings, questions, requirements)

        sr = self.call("search_spaces", function=req.function, buildings=scope, open_only=True)
        pool = self.ds.spaces.loc[sr.data or []]
        pool = pool[~pool["building"].isin(excluded)]
        if person:
            pool = pool[~pool["room_key"].isin(person["rooms"])]
        if pool.empty:
            findings.append(Evidence("No compatible space with recorded open seating in the search scope", "unknown", "space_records"))
            return self._report(req, requirements, [], [], findings, questions, person)

        model = {}
        if use_model:
            mr = self.call("model_space_assessment", space_ids=pool["space_id"].tolist(), function=req.function)
            model = mr.data
        cons = self.call("evaluate_space_constraints", candidates=pool, req=req).data
        prox = None
        if anchor and anchor["rooms"]:
            prox = self.call("measure_proximity", candidates=pool, anchors=anchor["rooms"]).data
        ranked = self.call("rank_candidate_spaces", candidates=pool, req=req, dept_buildings=dept_buildings,
                           model=model, constraints=cons, use_model=use_model, proximity=prox).data
        top = ranked[: AGENTS.max_candidates]
        self.call("check_planning_reservations", indices=[r["idx"] for r in top])
        candidates = [self._candidate(r, req, model) for r in top]
        alternatives = [self._candidate(r, req, model, brief=True) for r in ranked[AGENTS.max_candidates: AGENTS.max_candidates + 5]]
        for c in candidates[:3]:
            fp = self.call("get_floor_plan", building=c["building"], floor=c["floor"])
            c["floor_plan"] = {k: v for k, v in fp.data.items() if k != "path"} if fp.data else None
        return self._report(req, requirements, candidates, alternatives, findings, questions, person, pool_size=len(pool))

    # ---------------------------------------------------------------------------- helpers
    def _resolve_list(self, names: list[str], findings: list, questions: list) -> list[str]:
        out = []
        for b in names:
            r = self.call("resolve_building", text=b)
            if r.status == "success":
                out.append(r.data)
                findings += r.evidence
            elif r.status == "ambiguous":
                questions.append(f"Which building is meant by '{b}': {', '.join(r.data)}?")
            else:
                questions.append(f"Building '{b}' is not on record. Which building?")
        return out

    def _department_buildings(self, dept: str | None) -> set[str]:
        if not dept:
            return set()
        s = self.ds.spaces
        return set(s.loc[s["allocation"].fillna("").str.contains(dept, regex=False), "building"].unique())

    def _candidate(self, r: dict, req: SpaceRequest, model: dict, brief: bool = False) -> dict:
        row = self.ds.spaces.loc[r["idx"]]
        m = model.get(row.space_id, {})
        failed, unknown = r["constraints"]["failed"], list(r["constraints"]["unknown"])
        conditions = []
        if row.plan_reservation:
            conditions.append("Planning reservation")
        if req.department and (pd.isna(row.allocation) or req.department not in str(row.allocation)):
            conditions.append("Allocated to another unit: needs reallocation approval"
                              if pd.notna(row.allocation) else "Allocation not on record")
        if m.get("verify_flag"):
            conditions.append(f"Model sees {m['predicted_group']} ({m['confidence']:.0%}): verify use")
        if req.function in ("Research Lab", "Teaching Lab"):
            unknown += [f"{x.capitalize()} not on record" for x in LAB_UNRECORDED]
        if req.function not in WORKPOINT_FUNCTIONS and row.occ_total == 0:
            unknown.append("Current use not on record")
        notes = []
        if row.seat_status == "Vacant":
            notes.append("Recorded vacant: does not authorize assignment")
        if failed:
            status = "incompatible"
        elif r["coverage"] < AGENTS.coverage_threshold:
            status = "insufficient_information"
        elif conditions or unknown:
            status = "conditional"
        else:
            status = "potentially_suitable"
        out = {"space_id": row.space_id, "room_key": row.room_key, "building": row.building, "floor": row.floor,
               "room": row.room, "area": None if pd.isna(row.area) else round(float(row.area), 1),
               "space_class": row.space_class, "space_standard": _na(row.space_standard),
               "seat_status": row.seat_status, "allocation": _na(row.allocation),
               "plan_reservation": bool(row.plan_reservation), "status": status,
               "suitability": r["suitability"], "evidence_coverage": r["coverage"],
               "provisional": r["coverage"] < AGENTS.coverage_threshold}
        if brief:
            return out
        out.update({
            "criteria": r["criteria"], "constraints_failed": failed, "missing_information": unknown,
            "conditions": conditions, "notes": notes, "occupants": int(row.occ_total),
            "proximity": r.get("proximity"),
            "model": {k: m.get(k) for k in ("predicted_group", "confidence", "p_function", "verify_flag", "top3")} if m else None,
        })
        return out

    def _vacancy_review(self, req, dept, scope, findings, questions, requirements) -> dict:
        s = self.ds.spaces
        mask = s["seat_status"].isin(OPEN_SEAT_STATUS)
        if dept:
            mask &= s["allocation"].fillna("").str.contains(dept, regex=False)
        elif scope:
            mask &= s["building"].isin(scope)
        else:
            mask &= s["allocation"].fillna("").apply(lambda a: any(d in a for d in NATURAL_SCIENCES_DEPARTMENTS))
            requirements.append({"field": "Department group", "value": "Natural Sciences", "source": "inferred"})
        hits = s[mask]
        mr = self.call("model_space_assessment", space_ids=hits["space_id"].tolist(), function="Office")
        rows = []
        for _, r in hits.iterrows():
            m = mr.data.get(r.space_id, {})
            flags = []
            if r.plan_reservation:
                flags.append("Planning reservation")
            if pd.isna(r.allocation) or r.allocation == "Unassigned":
                flags.append("No departmental allocation")
            if m.get("verify_flag"):
                flags.append(f"Model infers {m['predicted_group']}")
            if r.occ_total > 0 and r.seat_status == "Vacant":
                flags.append(f"{int(r.occ_total)} occupant(s) in people records")
            rows.append({"space_id": r.space_id, "room_key": r.room_key, "building": r.building, "floor": r.floor,
                         "room": r.room, "area": None if pd.isna(r.area) else round(float(r.area), 1),
                         "space_class": r.space_class, "seat_status": r.seat_status,
                         "available_seats": None if pd.isna(r.available_seats) else int(r.available_seats),
                         "allocation": _na(r.allocation), "plan_reservation": bool(r.plan_reservation),
                         "occupants": int(r.occ_total), "flags": flags,
                         "status": "conditional" if flags else "recorded_vacant"})
        rows.sort(key=lambda x: (x["building"], x["floor"], x["room"]))
        findings.append(Evidence(f"{len(rows)} spaces recorded as vacant or under capacity in scope", "recorded", "space_records"))
        conflicts = [r for r in rows if any("occupant(s) in people records" in f for f in r["flags"])]
        if conflicts:
            findings.append(Evidence(f"{len(conflicts)} space(s) are recorded vacant but have occupants in people records",
                                     "conflicting", "space_records+people_records"))
        findings += mr.evidence[:10]
        questions.append("Which vacancies has the department confirmed in use review?")
        report = self._report(req, requirements, [], [], findings, questions, None)
        report["vacancies"] = rows
        report["summary"] = f"{len(rows)} recorded vacancies, {sum(1 for r in rows if r['flags'])} flagged."
        return report

    def _report(self, req, requirements, candidates, alternatives, findings, questions, person, pool_size=0) -> dict:
        approvals = ["Department", "School leadership"]
        if any(c.get("allocation") and req.department and req.department not in c["allocation"] for c in candidates):
            approvals.append("Space Management (allocation change)")
        if req.function in ("Research Lab", "Teaching Lab"):
            approvals += ["Facilities (technical review)", "Environmental Health & Safety"]
            questions.append("What equipment, utilities and safety needs must the lab meet?")
        if any(c["status"] == "conditional" or c.get("notes") for c in candidates):
            questions.append("Can the department confirm current use of the top rooms?")
        implications = []
        if person and person.get("rooms"):
            implications.append(f"Review current room(s) {', '.join(person['rooms'])} after the change")
        if req.request_type == "temporary_relocation":
            implications.append("Temporary use: confirm duration and return date")
        n_ok = sum(c["status"] == "potentially_suitable" for c in candidates)
        n_cond = sum(c["status"] == "conditional" for c in candidates)
        summary = (f"{len(candidates)} candidates: {n_ok} suitable, {n_cond} conditional." if candidates else
                   "No candidates yet.")
        self.log.info(f"workflow done candidates={len(candidates)} suitable={n_ok} conditional={n_cond} questions={len(questions)}")
        return {"case_type": "space_assignment", "agent": self.name, "request": req.model_dump(),
                "requirements": requirements, "summary": summary, "candidates": candidates,
                "alternative_spaces": alternatives, "findings": [vars(e) for e in findings],
                "verification_questions": list(dict.fromkeys(questions)), "required_approvals": approvals,
                "implications": implications, "review_status": "pending", "trace": self.trace,
                "decision_note": "Recommendation pending review. Candidates are not assignments."}


def _na(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else str(v)


def _space_dict(row: pd.Series) -> dict:
    keys = ["space_id", "room_key", "building", "floor", "room", "area", "space_class", "class_group",
            "space_standard", "ssa", "seat_status", "available_seats", "plan_reservation", "allocation",
            "chargeback", "duplicate_key", "occ_total"]
    out = {}
    for k in keys:
        v = row[k]
        out[k] = None if (v is pd.NA or (isinstance(v, float) and np.isnan(v))) else (
            v.item() if hasattr(v, "item") else v)
    return out


_FLOOR_ORDER = {"B3": -3, "B2": -2, "B1": -1, "B": -1, "A": 0, "M": 1.5, "PH": 99}


def _floor_level(floor: str) -> float | None:
    f = str(floor).upper()
    if f in _FLOOR_ORDER:
        return _FLOOR_ORDER[f]
    return float(f) if f.isdigit() else None


def _room_parts(room: str) -> tuple[str, int | None]:
    m = re.match(r"^([A-Za-z]*)(\d+)", str(room))
    return (m.group(1).upper(), int(m.group(2))) if m else ("", None)


def _proximity(b, f, r, ab, af, ar) -> tuple[float, str]:
    """Closeness score in [0,1] with a readable label. Same floor > same building > other building."""
    if b != ab:
        return 0.1, "Other building"
    lf, laf = _floor_level(f), _floor_level(af)
    if f != af:
        diff = abs(lf - laf) if lf is not None and laf is not None else 3
        return max(0.25, 0.65 - 0.1 * diff), (f"{int(diff)} floor{'s' if diff != 1 else ''} away" if diff == int(diff) else "Other floor")
    (w1, n1), (w2, n2) = _room_parts(r), _room_parts(ar)
    if n1 is None or n2 is None:
        return 0.8, "Same floor"
    gap = abs(n1 - n2)
    if gap == 0 and w1 == w2:
        return 1.0, f"Same floor · {w1}{n1}"
    wing = "" if w1 == w2 else ", other wing"
    score = max(0.7, 1.0 - gap / 100 - (0.1 if wing else 0))
    return round(score, 3), f"Same floor{' · other wing' if wing else ''} · {gap} rooms"


def _area_score(area, req: SpaceRequest) -> float | None:
    if pd.isna(area):
        return None
    if req.min_area and area < req.min_area:
        return 0.0
    if req.max_area and area > req.max_area:
        return max(0.2, req.max_area / area)
    return 1.0


def _occupancy_score(r) -> float | None:
    return {"Vacant": 1.0, "Under Capacity": 0.6, "At Capacity": 0.1, "Over Capacity": 0.0}.get(r.seat_status) \
        if r.seat_status != "Not Available for Seating" else (None if r.occ_total == 0 else 0.3)


def _department_score(allocation, dept: str | None) -> float | None:
    if pd.isna(allocation):
        return None
    if dept and dept in allocation:
        return 1.0
    if allocation == "Unassigned":
        return 0.6
    return 0.2
