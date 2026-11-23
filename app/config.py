"""Central configuration: paths, institutional reference data and hyperparameters.

Everything a team member may need to tune lives here, so behaviour is not scattered
across modules. Institutional values (building crosswalk, department group) come from
James's written clarifications in `datares/Space and Outage Agents_ Proposed Scope and
Workflows.docx`; values marked ASSUMPTION still need domain confirmation.

Secrets and deployment variables (FAI_*) live in the project's `.env` file, loaded here once
when the app starts; `.env.example` lists every variable. A variable already set in the
terminal wins over the file. Never put secrets in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE, override=False)
SOURCE_DIR = ROOT / "datares"          # original materials, never modified
DATA_DIR = ROOT / "data"               # derived/cache files and app database
MODEL_DIR = ROOT / "models"            # trained artifacts
WEB_DIR = ROOT / "web"

SPACE_FILE = SOURCE_DIR / "SpaceRecords.xlsx"
PEOPLE_FILE = SOURCE_DIR / "Report_PeopleRecords.xlsx"
FLOOR_PLAN_DIRS = [SOURCE_DIR / "May 2026", SOURCE_DIR / "November 2026"]
# Case database. FAI_DB_PATH lets a second copy of the app (e.g. for testing) use its own database.
DB_PATH = Path(os.environ["FAI_DB_PATH"]) if os.environ.get("FAI_DB_PATH") else DATA_DIR / "facilities.db"

SEED = 42

# Building crosswalk supplied by James (code, full name, name in space export, map no.).
BUILDINGS: list[dict] = [
    {"code": "ABL", "name": "M.D. Anderson Biological Laboratories", "export": "Biology", "map": "22",
     "aliases": ["MDA Biology", "Anderson Biological", "BIO basement", "Biology building"]},
    {"code": "BRC", "name": "BioScience Research Collaborative", "export": "BRC", "map": "63",
     "aliases": ["Bioscience Research Collaborative", "BRC"]},
    {"code": "BRK", "name": "Brockman Hall for Physics", "export": "Brockman Hall", "map": "72",
     "aliases": ["Brockman Hall for Physics"]},
    {"code": "DBH", "name": "Dell Butcher Hall", "export": "Dell Butcher", "map": "48", "aliases": ["Dell Butcher"]},
    {"code": "GRB", "name": "George R. Brown Hall", "export": "George R. Brown", "map": "45",
     "aliases": ["George R. Brown", "GRB East", "GRB West"]},
    {"code": "HBH", "name": "Herman Brown Hall for Mathematical Sciences", "export": "Herman Brown", "map": "34",
     "aliases": ["Herman Brown"]},
    {"code": "HRZ", "name": "Herzstein Hall", "export": "Herzstein", "map": "4", "aliases": ["Herzstein"]},
    {"code": "KECK", "name": "Keck Hall", "export": "Keck", "map": "8", "aliases": ["Keck Hall"]},
    {"code": "KWG", "name": "Keith-Wiess Geological Laboratories", "export": "Geology", "map": "23",
     "aliases": ["Keith-Wiess", "Keith Wiess", "Geology building", "Geology elevator"]},
    {"code": "OES", "name": "Ralph S. O'Connor Building for Engineering and Science",
     "export": "O'Connor Engineering and Science", "map": None,
     "aliases": ["O'Connor Building", "O'Connor Engineering"]},
    {"code": "SST", "name": "Space Science and Technology Building", "export": "Space Science", "map": "32",
     "aliases": ["Space Science"]},
    {"code": "TUDOR", "name": "Tudor Fieldhouse and Youngkin Center", "export": "Tudor Fieldhouse", "map": "14",
     "aliases": ["Tudor Fieldhouse"]},
]

# Bare words that match more than one building and must never auto-resolve.
AMBIGUOUS_ALIASES: dict[str, list[str]] = {
    "brockman": ["Brockman Hall", "Brockman Hall for Opera"],
}

# James's "Natural Sciences group". ASSUMPTION: membership inferred from department
# names in the people/space exports; confirm with James.
NATURAL_SCIENCES_DEPARTMENTS: list[str] = [
    "BioSciences",
    "Chemistry",
    "Physics and Astronomy",
    "Earth Environmental Planetary Sciences",
    "Mathematics",
    "Nat Sci - Dean's Office",
]

# Departmental coordination contacts that appear in the supplied correspondence.
# Status is always "candidate": an email mention is not a verified role assignment.
DEPARTMENT_CONTACTS: list[dict] = [
    {"department": "BioSciences", "name": "Juan Sanchez", "netid": "cjds",
     "basis": "Signature 'Facilities Manager, Biosciences'; asked to notify groups in ABL (ABL thread, Oct 22 2026)"},
    {"department": "Chemistry", "name": "Abby Vacek", "netid": "akv1",
     "basis": "Added as 'CHEM facilities' and asked to notify groups in ABL (ABL thread, Oct 22 2026)"},
]

# Space Class code -> functional group (the deep learning target).
CLASS_GROUPS: dict[str, str] = {
    "110": "Classroom", "115": "Classroom",
    "210": "Teaching Lab", "215": "Teaching Lab", "220": "Teaching Lab", "225": "Teaching Lab",
    "250": "Research Lab", "255": "Research Lab",
    "310": "Office", "320": "Office", "340": "Office", "360": "Office",
    "315": "Office Service",
    "330": "Workstation",
    "350": "Meeting", "355": "Meeting", "680": "Meeting", "685": "Meeting",
}
CLASS_GROUP_BY_PREFIX: dict[str, str] = {
    "0": "Building Service", "4": "Study & Library", "5": "Special & General Use",
    "6": "Special & General Use", "7": "Support & Storage", "8": "Special & General Use",
    "9": "Residential", "W": "Building Service", "X": "Building Service", "Y": "Building Service",
}

# People "Functional Role" -> occupant group used as model features.
ROLE_GROUPS: dict[str, str] = {
    "Tenured/Tenure Track Faculty": "faculty", "Non Tenure Track Faculty": "faculty",
    "Adjunct and Unpaid Faculty": "faculty", "Retiree: Emeritus Faculty": "faculty", "Dean": "faculty",
    "Postdoctoral Associate": "postdoc", "Postdoctoral Fellow: Limited Benefits": "postdoc",
    "Active Student (Grad)": "grad", "Graduate Assistant": "grad",
    "Student: Non-Employee Graduate Fellowship": "grad",
    "Active Student (Undergrad)": "undergrad", "Hourly Student Employee - UG/GR": "undergrad",
    "Undergraduate Student Employee": "undergrad", "Student Officer": "undergrad",
    "Visitor: On Campus": "visitor", "Visitor: Remote": "visitor", "Visiting Student": "visitor",
}
OCCUPANT_GROUPS = ["faculty", "postdoc", "grad", "undergrad", "staff", "visitor"]
MANAGEMENT_ROLE_PREFIXES = ("Management -", "Executive -", "Dean")


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for SpaceFunctionNet (see app/ml/model.py)."""
    seed: int = SEED
    val_size: float = 0.15
    test_size: float = 0.15
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.25
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    min_category_count: int = 5     # rarer categorical values map to <UNK>
    cv_folds: int = 5               # out-of-fold predictions served to the agents
    # Selected by the ablation study on VALIDATION macro-F1 (app/ml/ablation.py, models/ablation.json):
    # one-hot categorical inputs without class weighting beat learned embeddings + class weights.
    use_embeddings: bool = False
    class_weighting: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """Bounded-execution limits and decision thresholds shared by both agents."""
    max_tool_calls: int = 40
    max_candidates: int = 8
    coverage_threshold: float = 0.6        # below this a suitability score is "Provisional"
    mismatch_confidence: float = 0.70      # model confidence needed to flag a record for verification
    lab_like_threshold: float = 0.50       # p(lab) for a non-lab record to count as potential lab use
    relocation_min_hours: float = 4.0      # outages at least this long trigger relocation support
    proximity_weight: float = 0.35         # weight of closeness to a reference office when one is given
    ranking_weights: dict = field(default_factory=lambda: {
        # Initial engineering defaults from the architecture document (Figure 13); James must validate.
        "functional": 0.30, "area": 0.20, "occupancy": 0.20,
        "department": 0.15, "location": 0.10, "planning": 0.05,
    })


TRAIN = TrainConfig()
AGENTS = AgentConfig()
