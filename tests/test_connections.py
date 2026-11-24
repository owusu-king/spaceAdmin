"""Case deletion, the Microsoft 365 reader (fake Graph), and the records pipeline (small generated exports)."""
import json
from datetime import datetime

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.integrations.msgraph import GraphAuth, GraphFile, GraphMailSource, share_id

ENV = {"tenant": "contoso", "client_id": "abc", "auth": "device", "secret": "", "mailbox": "me", "folder": "Inbox", "since_days": 14}


# ------------------------------------------------------------------------------ case deletion
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import api
    from app.services.cases import CaseStore
    with TestClient(api.app) as c:
        api.STATE["store"] = CaseStore(tmp_path_factory.mktemp("db") / "del.db")
        api.attach_switches(api.STATE["store"])
        yield c


def test_delete_case_removes_it_and_never_reuses_its_number(client):
    a = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-OES-EXHAUST"}}).json()
    b = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-GRB-ELEVATOR"}}).json()
    client.post(f"/api/v1/cases/{b['id']}/review", json={"decision": "approve", "rationale": "", "rev": 1})
    r = client.delete(f"/api/v1/cases/{b['id']}")
    assert r.status_code == 200 and r.json()["deleted"] == b["id"]
    assert client.get(f"/api/v1/cases/{b['id']}").status_code == 404
    assert client.delete(f"/api/v1/cases/{b['id']}").status_code == 404
    c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-GRB-ELEVATOR"}}).json()
    assert not c.get("existing")                                                          # deleted: a new case
    assert int(c["id"].split("-")[1]) > int(b["id"].split("-")[1]) > int(a["id"].split("-")[1])   # no reuse
    audit = client.get("/api/v1/activity").json()["audit"]
    assert any(e["action"] == "case_deleted" and e["case_id"] == b["id"] for e in audit)


def test_one_case_per_outage_request(client):
    """Clicking "Review with Outage Agent" again, or the same notice from another message, opens the existing case."""
    before = len(client.get("/api/v1/cases").json())
    a = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-ABL-STEAM"}}).json()
    again = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-ABL-STEAM"}}).json()
    assert again["id"] == a["id"] and again["existing"] and not a.get("existing")
    sr = a["output"]["outage"]["request_reference"]
    assert sr and sr.startswith("SR-")
    msgs = [x for x in client.get("/api/v1/sources").json()["messages"] if x["sr"] == sr]
    assert msgs                                              # the same request reached through each of its emails
    for m in msgs:
        via_mail = client.post("/api/v1/cases", json={"kind": "outage", "input": {"message_id": m["id"]}}).json()
        assert via_mail["id"] == a["id"] and via_mail["existing"]
    assert len(client.get("/api/v1/cases").json()) == before + 1


def test_double_clicks_record_one_decision(client):
    c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": "OUT-BRK-ELEVATOR"}}).json()
    ret = {"decision": "return", "rationale": "Check the dates", "rev": c["rev"]}
    assert client.post(f"/api/v1/cases/{c['id']}/review", json=ret).status_code == 200
    second = client.post(f"/api/v1/cases/{c['id']}/review", json=ret)
    assert second.status_code in (409, 422) and "already" in second.json()["detail"]
    assert len(client.get(f"/api/v1/cases/{c['id']}").json()["reviews"]) == 1


def test_events_are_not_duplicated(client):
    body = {"title": "Open house", "start": "2027-02-01T10:00:00", "end": "2027-02-01T12:00:00", "building": "Keck"}
    a = client.post("/api/v1/events", json=body).json()
    b = client.post("/api/v1/events", json=body).json()
    assert b["id"] == a["id"] and b["duplicate"] and not a.get("duplicate")
    ics = "\n".join(["BEGIN:VCALENDAR", "BEGIN:VEVENT", "SUMMARY:Seminar", "DTSTART:20270203T150000",
                     "DTEND:20270203T160000", "LOCATION:Keck Hall", "END:VEVENT", "END:VCALENDAR", ""])
    assert client.post("/api/v1/events/import", json={"ics": ics}).json() == {"imported": 1, "skipped": 0}
    assert client.post("/api/v1/events/import", json={"ics": ics}).json() == {"imported": 0, "skipped": 1}


# ------------------------------------------------------------------------------ Microsoft 365 (fake Graph)
class FakeMicrosoft:
    def __init__(self):
        self.requests, self.token_calls = [], 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        url = str(req.url)
        if url.endswith("/devicecode"):
            return httpx.Response(200, json={"device_code": "dev", "user_code": "ABCD-1234", "interval": 0,
                                             "verification_uri": "https://microsoft.com/devicelogin", "expires_in": 900})
        if url.endswith("/token"):
            self.token_calls += 1
            if self.token_calls == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(200, json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600})
        if "/me?" in url or url.endswith("/me"):
            return httpx.Response(200, json={"userPrincipalName": "james@rice.edu"})
        if "messages/delta" in url and "page2" not in url and "deltatoken" not in url:
            return httpx.Response(200, json={"value": [
                {"id": "1", "internetMessageId": "<m1@rice.edu>", "subject": "Steam outage ABL Friday",
                 "from": {"emailAddress": {"name": "Plant Ops", "address": "ops@rice.edu"}},
                 "receivedDateTime": "2026-11-20T15:00:00Z", "body": {"content": "Steam off 6 to 10."}},
                {"id": "x", "@removed": {"reason": "deleted"}}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages/delta?page2"})
        if "page2" in url:
            return httpx.Response(200, json={"value": [
                {"id": "2", "internetMessageId": "<m2@rice.edu>", "subject": "Lunch?",
                 "from": {"emailAddress": {"name": "Doug", "address": "d@rice.edu"}},
                 "receivedDateTime": "2026-11-20T16:00:00Z", "body": {"content": "Free Friday?"}}],
                "@odata.deltaLink": "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages/delta?deltatoken=t1"})
        if "deltatoken=t1" in url:
            return httpx.Response(200, json={"value": [], "@odata.deltaLink": url})
        if "/shares/" in url and url.endswith("/content"):
            return httpx.Response(200, content=b"xlsx-bytes")
        if "/shares/" in url:
            return httpx.Response(200, json={"name": "SpaceRecords.xlsx", "eTag": "\"v2\"", "lastModifiedDateTime": "2026-11-20T10:00:00Z", "size": 9})
        return httpx.Response(404, json={})


def _auth(tmp_path, fake):
    return GraphAuth(env=dict(ENV), token_file=tmp_path / "tok.json", http=httpx.Client(transport=httpx.MockTransport(fake)))


def test_device_sign_in_stores_a_token_never_a_password(tmp_path):
    import time
    fake = FakeMicrosoft()
    auth = _auth(tmp_path, fake)
    st = auth.start_device_flow()
    assert st["status"] == "signing_in" and st["user_code"] == "ABCD-1234"
    for _ in range(50):
        if auth.status()["status"] == "connected":
            break
        time.sleep(0.05)
    assert auth.status()["status"] == "connected" and auth.status()["account"] == "james@rice.edu"
    saved = json.loads((tmp_path / "tok.json").read_text())
    assert saved["refresh_token"] == "rt" and "password" not in json.dumps(saved)
    auth.sign_out()
    assert auth.status()["status"] == "not_connected" and not (tmp_path / "tok.json").exists()


def test_mail_reader_syncs_with_delta_and_only_reads(tmp_path):
    fake = FakeMicrosoft()
    auth = _auth(tmp_path, fake)
    (tmp_path / "tok.json").write_text(json.dumps({"access_token": "at", "refresh_token": "rt", "expires_at": 9e12}))
    auth._tok = json.loads((tmp_path / "tok.json").read_text())
    src = GraphMailSource(auth, state_file=tmp_path / "state.json")
    first = src.fetch(25)
    assert [m.subject for m in first] == ["Steam outage ABL Friday", "Lunch?"]          # removed items skipped
    assert first[0].message_id == "<m1@rice.edu>" and first[0].sender == "Plant Ops <ops@rice.edu>"
    assert "deltatoken=t1" in json.loads((tmp_path / "state.json").read_text())["delta_link"]
    assert src.fetch(25) == []                                                           # nothing new since last sync
    graph_calls = [r for r in fake.requests if "graph.microsoft.com" in str(r.url)]
    assert graph_calls and all(r.method == "GET" for r in graph_calls)                   # read-only by construction
    assert "receivedDateTime ge" in str(graph_calls[0].url).replace("%20", " ").replace("+", " ")
    with pytest.raises(PermissionError):
        src.execute("mark_read")


def test_sharepoint_file_version_and_download(tmp_path):
    fake = FakeMicrosoft()
    auth = _auth(tmp_path, fake)
    auth._tok = {"access_token": "at", "refresh_token": "rt", "expires_at": 9e12}
    f = GraphFile(auth, "https://rice.sharepoint.com/:x:/s/fac/SpaceRecords.xlsx")
    assert share_id("https://a.b/c").startswith("u!") and "=" not in share_id("https://a.b/c")
    assert f.version()["etag"] == "\"v2\"" and f.download() == b"xlsx-bytes"


# ------------------------------------------------------------------------------ records pipeline
def _exports(folder, rooms, people):
    space_cols = ["Plan Rsvn", "Seat Stat", "Avble Seats", "Building", "Floor", "Space", "Area", "W/S Area", "Space Class",
                  "SSA", "spaceStandard", "occupancyAllocationData", "chargebackAllocationData"]
    people_cols = ["SSA", "Net ID", "Name", "Title", "Department", "Functional Role", "Building", "Floor", "Space"]
    pd.DataFrame([{c: "" for c in space_cols} | r for r in rooms], columns=space_cols).to_excel(folder / "SpaceRecords.xlsx", index=False)
    pd.DataFrame([{c: "" for c in people_cols} | p for p in people], columns=people_cols).to_excel(folder / "Report_PeopleRecords.xlsx", index=False)


ROOM = {"Plan Rsvn": "FALSE", "Building": "Keck", "Floor": "01", "Area": "120", "Space Class": "310 - Office",
        "Seat Stat": "Vacant", "occupancyAllocationData": "Chemistry"}


def test_records_pipeline_validates_compares_and_switches(tmp_path, monkeypatch):
    import app.data.records as rec
    from app.data.loader import load_dataset
    monkeypatch.setattr(rec, "RECORDS_DIR", tmp_path / "records")
    monkeypatch.setattr(rec, "CURRENT", tmp_path / "records" / "current.json")
    src_dir = tmp_path / "exports"
    src_dir.mkdir()
    _exports(src_dir, [ROOM | {"Space": "101"}, ROOM | {"Space": "102"}],
             [{"Net ID": "ab1", "Name": "Ann B", "Building": "Keck", "Floor": "01", "Space": "101"}])
    current = {"ds": load_dataset(use_cache=False, space_file=src_dir / "SpaceRecords.xlsx",
                                  people_file=src_dir / "Report_PeopleRecords.xlsx", write_cache=False)}
    switched = []
    pipe = rec.RecordsPipeline(rec.FolderSource(src_dir), lambda: current["ds"],
                               lambda ds: switched.append(ds) or {"without_model_history": 0}, state_file=tmp_path / "state.json")
    assert pipe.refresh()["changed"]                       # first look: takes the files in
    current["ds"] = switched[-1]
    assert pipe.refresh() == {"changed": False, "reason": "no new version"}              # unchanged: nothing downloaded
    _exports(src_dir, [ROOM | {"Space": "101", "Seat Stat": "At Capacity"}, ROOM | {"Space": "102"}, ROOM | {"Space": "103"}],
             [{"Net ID": "ab1", "Name": "Ann B", "Building": "Keck", "Floor": "01", "Space": "102"}])
    res = pipe.refresh()
    assert res["changed"] and res["diff"]["spaces"]["added"] == 1 and res["diff"]["spaces"]["changed"] >= 1
    assert res["diff"]["people"]["moved"] == 1 and len(switched) == 2
    assert rec.active_files()[0].parent.parent == tmp_path / "records"                  # the app now reads the snapshot
    current["ds"] = switched[-1]
    good_space = (src_dir / "SpaceRecords.xlsx").read_bytes()
    pd.DataFrame({"Building": ["Keck"]}).to_excel(src_dir / "SpaceRecords.xlsx", index=False)   # a broken export
    res = pipe.refresh()
    assert not res["changed"] and "Rejected" in res["error"] and len(switched) == 2      # good data kept
    # James uploads the two reports in the app: same validation, compare and switch
    up = rec.UploadSource(good_space, (src_dir / "Report_PeopleRecords.xlsx").read_bytes())
    assert pipe.refresh(force=True, source=up)["changed"] and len(switched) == 3


def test_records_setup_in_the_app_validates_input(client):
    r = client.put("/api/v1/records/config", json={"source": "microsoft", "space_url": "https://x/a.xlsx"})
    assert r.status_code == 422 and "both" in r.json()["detail"]
    r = client.put("/api/v1/records/config", json={"source": "folder", "dir": "Z:/no/such/folder"})
    assert r.status_code == 422
    import base64
    bad = base64.b64encode(b"not an excel file").decode()
    r = client.post("/api/v1/records/upload", json={"space_b64": bad, "people_b64": bad}).json()
    assert not r["changed"] and "Rejected" in r["error"]                                  # current records untouched


def test_new_rooms_get_model_predictions(system):
    from app.ml.inference import SpacePredictor
    sp = system.ds.spaces
    new = sp.iloc[[0]].copy()
    new["space_id"] = new["room_key"] = "Keck · 09 · 999Z"
    p = SpacePredictor(pd.concat([sp, new], ignore_index=True))
    assert p.without_oof == 1 and p.assessment("Keck · 09 · 999Z")["status"] == "success"
