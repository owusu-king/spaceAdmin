import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Tests never write to the real databases: cases, audit, inbox and the calendar go to a throwaway folder.
# (Set before any app module is imported; app.config reads it once.)
os.environ["FAI_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="fai-tests-")) / "facilities.db")

AS_OF = datetime(2026, 11, 22)


@pytest.fixture(scope="session")
def system():
    """One fully initialized system (data, model, corpus, both agents) shared by the tests."""
    from app.agents.coordinator import build_system
    return build_system(as_of=AS_OF)


@pytest.fixture(scope="session", autouse=True)
def isolated_records(tmp_path_factory):
    """Tests never touch the real records snapshots or their status (data/records): the app reads the original
    exports and any refresh made by a test lands in a temporary folder."""
    import app.data.records as rec
    tmp = tmp_path_factory.mktemp("records")
    saved = rec.RECORDS_DIR, rec.CURRENT
    rec.RECORDS_DIR, rec.CURRENT = tmp, tmp / "current.json"
    yield
    rec.RECORDS_DIR, rec.CURRENT = saved
