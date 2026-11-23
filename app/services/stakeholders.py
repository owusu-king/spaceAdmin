"""Stakeholder resolution with three independently verified roles (architecture §11.6).

* Formal approver:       holds authority to approve the activity. Comes from TRIRIGA; the
                         supplied exports do not contain an approver roster -> "Not on record".
* Coordination contact:  gives departmental input. Candidates come from correspondence
                         (config.DEPARTMENT_CONTACTS) and from recorded management roles.
* Notification recipient: people recorded in affected spaces.

A recorded occupant is never treated as an approver.
"""
from __future__ import annotations

import pandas as pd

from app.config import DEPARTMENT_CONTACTS, NATURAL_SCIENCES_DEPARTMENTS


def resolve_stakeholders(people: pd.DataFrame, spaces: pd.DataFrame, buildings: list[str],
                         departments: list[str]) -> dict:
    in_scope = people[people["building"].isin(buildings)]
    result, gaps = [], []
    for dept in departments:
        contacts = [{"name": c["name"], "email": f"{c['netid']}@rice.edu", "role": "coordination_contact",
                     "verification": "candidate", "basis": c["basis"], "classification": "recorded"}
                    for c in DEPARTMENT_CONTACTS if c["department"] == dept]
        mgmt = in_scope[(in_scope["department"] == dept) & in_scope["is_management"]]
        for _, p in mgmt.head(4).iterrows():
            contacts.append({"name": p["name"], "email": p["email"], "role": "coordination_contact",
                             "verification": "unverified",
                             "basis": f"Recorded {p['role']} located at {p['room_key']}; role must be confirmed",
                             "classification": "inferred"})
        occupants = in_scope[in_scope["department"] == dept]
        entry = {
            "department": dept,
            "natural_sciences": dept in NATURAL_SCIENCES_DEPARTMENTS,
            "formal_approver": {"status": "not_on_record",
                                "note": "Confirm the approver for this building in the TRIRIGA approval roster"},
            "coordination_contacts": contacts,
            "notification_recipients": {"count": int(len(occupants)),
                                        "sample": occupants["email"].head(5).tolist(),
                                        "basis": "Recorded occupants in affected building (people records)"},
        }
        if not contacts:
            gaps.append(f"No coordination contact on record for {dept}")
        result.append(entry)
    return {"departments": result, "gaps": gaps}
