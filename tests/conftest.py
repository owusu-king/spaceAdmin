import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AS_OF = datetime(2026, 11, 22)


@pytest.fixture(scope="session")
def system():
    """One fully initialized system (data, model, corpus, both agents) shared by the tests."""
    from app.agents.coordinator import build_system
    return build_system(as_of=AS_OF)
