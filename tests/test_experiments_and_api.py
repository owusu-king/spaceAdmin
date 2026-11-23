"""Cooperation, competition, evaluation and the HTTP API."""
import pytest
from fastapi.testclient import TestClient

from app.evaluation import evaluate_agents
from app.experiments import competition_experiment, cooperation_experiment


def test_cooperation_experiment(system):
    r = cooperation_experiment(system)
    s = r["summary"]
    assert s["outages_evaluated"] > 0
    assert s["coop_share_with_options"] > s["solo_share_with_options"]
    assert s["constraint_compliance"] == 1.0


def test_competition_is_reproducible(system):
    a, b = competition_experiment(system, n=300), competition_experiment(system, n=300)
    assert a["results"] == b["results"]
    assert set(a["results"]) == {"A_rule_based", "B_model_based", "C_combined"}


def test_evaluation_passes(system):
    r = evaluate_agents(system)
    failed = [(s["scenario"], c["check"]) for s in r["scenarios"] for c in s["checks"] if not c["passed"]]
    assert not failed, failed


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import api
    from app.services.cases import CaseStore
    with TestClient(api.app) as c:
        from app.integrations.email import EmailIntake, ReplaySource
        api.STATE["store"] = CaseStore(tmp_path_factory.mktemp("db") / "test.db")   # keep tests off the app DB
        api.STATE["intake"] = EmailIntake(ReplaySource(api.STATE["coord"].corpus), api.STATE["coord"], api.STATE["store"])
        yield c


def test_read_endpoints(client):
    for path in ["/api/v1/overview", "/api/v1/buildings", "/api/v1/spaces?limit=5", "/api/v1/outages",
                 "/api/v1/sources", "/api/v1/model", "/api/v1/agents", "/api/v1/activity"]:
        assert client.get(path).status_code == 200, path


def test_space_detail_and_floor_plan(client):
    sid = client.get("/api/v1/spaces?building=Keck&limit=1").json()["rows"][0]["space_id"]
    d = client.get(f"/api/v1/spaces/{sid}").json()
    assert d["space"]["space_id"] == sid and d["model"]["status"] == "success"
    assert client.get(d["floor_plan"]["url"]).headers["content-type"] == "application/pdf"
    assert client.get("/api/v1/floorplans/../../etc").status_code == 404


def test_case_lifecycle(client):
    c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-OES-EXHAUST"}}).json()
    assert c["status"] == "ReadyForReview" and c["rev_letter"] == "A"
    assert client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "decline", "rationale": "", "rev": 1}).status_code == 409
    r = client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "return", "rationale": "Need contacts", "rev": 1}).json()
    assert r["status"] == "NeedsRevision"
    r = client.post(f"/api/v1/cases/{c['id']}/rerun", json={}).json()
    assert r["rev_letter"] == "B" and r["reviews"][0]["invalidated"] == 1
    assert client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "approve", "rationale": "", "rev": 1}).status_code == 409
    assert client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "approve", "rationale": "", "rev": 2}).json()["status"] == "Approved"


def test_invalid_requests(client):
    assert client.post("/api/v1/cases", json={"kind": "space", "input": {"function": "Garage"}}).status_code == 422
    assert client.post("/api/v1/cases", json={"kind": "weather", "input": {}}).status_code == 422
    assert client.post("/api/v1/actions/send_email").status_code == 403
    assert client.get("/api/v1/cases/XX-9999").status_code == 404


def test_inbox_and_recipients(client):
    assert client.get("/api/v1/inbox").json()["source"]["mode"] == "replay"
    r = client.post("/api/v1/inbox/poll?limit=5").json()
    assert r["fetched"] == 5
    c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-OES-EXHAUST"}}).json()
    csv_text = client.get(f"/api/v1/cases/{c['id']}/recipients.csv").text
    assert csv_text.splitlines()[0].startswith("name,email,netid") and "@rice.edu" in csv_text
    assert client.get("/api/v1/ablation").status_code == 200
