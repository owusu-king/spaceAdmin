"""Settings switches and sending approved drafts: off by default, two keys to unlock, every guard enforced."""
from datetime import datetime, timedelta
from email import message_from_string

import pytest
from fastapi.testclient import TestClient

from app.integrations.mailer import SmtpMailer

MAIL_ENV = {"FAI_SMTP_HOST": "smtp.test", "FAI_SMTP_PORT": "587", "FAI_SMTP_FROM": "facilities-ai@test.edu",
            "FAI_EMAIL_REDIRECT_TO": "demo@test.edu"}


class FakeSMTP:
    """Stands in for smtplib.SMTP; records every message instead of delivering it."""
    outbox: list = []

    def __init__(self, host, port):
        self.host = host

    def login(self, user, password):
        pass

    def send_message(self, msg, to_addrs):
        FakeSMTP.outbox.append({"headers": message_from_string(msg.as_string()), "to_addrs": to_addrs, "body": msg.get_content()})

    def quit(self):
        pass


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import api
    from app.services.cases import CaseStore
    with TestClient(api.app) as c:
        api.STATE["store"] = CaseStore(tmp_path_factory.mktemp("db") / "test.db")   # keep tests off the app DB
        api.attach_switches(api.STATE["store"])
        api.STATE["mailer"] = SmtpMailer(env={}, client_factory=FakeSMTP)           # unconfigured until a test says so
        yield c


@pytest.fixture
def mail(monkeypatch):
    """Unlock allowed and outgoing mail configured; resets switches and the fake outbox."""
    from app import api
    monkeypatch.setenv("FAI_ALLOW_UNLOCK", "true")
    api.STATE["mailer"] = SmtpMailer(env=MAIL_ENV, client_factory=FakeSMTP)
    FakeSMTP.outbox.clear()
    yield api
    for key in ("email_live_recipients", "send_email"):
        api.STATE["settings"].set(key, False, "test")


def _sw(settings: dict, key: str) -> dict:
    return next(x for x in settings["switches"] if x["key"] == key)


def _fresh_case(client, scenario="OUT-ABL-STEAM"):
    """A new case for the request. One request has one case, so an earlier test's case for it is removed first."""
    c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": scenario}}).json()
    if c.get("existing"):
        client.delete(f"/api/v1/cases/{c['id']}")
        c = client.post("/api/v1/cases", json={"kind": "outage", "input": {"scenario": scenario}}).json()
    return c


def _approved_case(client, scenario="OUT-ABL-STEAM"):
    c = _fresh_case(client, scenario)
    client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "approve", "rationale": "", "rev": 1})
    return c


def test_switches_are_off_and_locked_without_the_machine_key(client, monkeypatch):
    monkeypatch.delenv("FAI_ALLOW_UNLOCK", raising=False)
    s = client.get("/api/v1/settings").json()
    assert s["unlock_allowed"] is False
    locked = [x for x in s["switches"] if x["group"] == "Actions"]
    assert locked and all(not x["on"] and not x["available"] for x in locked)
    assert all(not x["on"] for x in s["switches"])       # local features (voice) are off by default too
    r = client.put("/api/v1/settings/send_email", json={"on": True})
    assert r.status_code == 409 and "FAI_ALLOW_UNLOCK" in r.json()["detail"]
    assert client.put("/api/v1/settings/no_such_switch", json={"on": True}).status_code == 404


def test_send_email_needs_mail_configuration(client, monkeypatch):
    from app import api
    monkeypatch.setenv("FAI_ALLOW_UNLOCK", "true")
    api.STATE["mailer"] = SmtpMailer(env={}, client_factory=FakeSMTP)
    r = client.put("/api/v1/settings/send_email", json={"on": True})
    assert r.status_code == 409 and "FAI_SMTP_HOST" in r.json()["detail"]


def test_unbuilt_integrations_cannot_be_switched_on(client, mail):
    for key in ("write_tririga", "assign_space", "approve_outage"):
        assert client.put(f"/api/v1/settings/{key}", json={"on": True}).status_code == 409


def test_send_guards_and_test_mode(client, mail):
    c = _fresh_case(client)
    send = {"draft": "coordination", "rev": 1, "confirm": True}
    assert client.post(f"/api/v1/cases/{c['id']}/send", json=send).status_code == 403            # switch off
    assert _sw(client.put("/api/v1/settings/send_email", json={"on": True}).json(), "send_email")["on"] is True
    assert client.post(f"/api/v1/cases/{c['id']}/send", json=send).status_code == 409            # not approved
    client.post(f"/api/v1/cases/{c['id']}/review", json={"decision": "approve", "rationale": "", "rev": 1})
    assert client.post(f"/api/v1/cases/{c['id']}/send", json={**send, "confirm": False}).status_code == 422
    assert client.post(f"/api/v1/cases/{c['id']}/send", json={**send, "rev": 2}).status_code == 409
    r = client.post(f"/api/v1/cases/{c['id']}/send", json=send)
    assert r.status_code == 200 and r.json()["redirected"] == 1
    sent = FakeSMTP.outbox[-1]
    assert sent["to_addrs"] == ["demo@test.edu"]                                                  # test address only
    assert sent["headers"]["Subject"].startswith("[TEST]") and "[TEST SEND" in sent["body"]
    assert "cjds@rice.edu" in sent["body"]                                                         # real list shown inside
    assert client.post(f"/api/v1/cases/{c['id']}/send", json=send).status_code == 409            # once per revision
    case = client.get(f"/api/v1/cases/{c['id']}").json()
    assert case["sent"][0]["draft"] == "coordination" and case["sending"]["enabled"] is True
    assert any(a["action"] == "email_sent" for a in client.get("/api/v1/activity").json()["audit"])


def test_live_delivery_needs_send_switch_and_uses_bcc(client, mail):
    assert client.put("/api/v1/settings/email_live_recipients", json={"on": True}).status_code == 409   # depends on send_email
    client.put("/api/v1/settings/send_email", json={"on": True})
    client.put("/api/v1/settings/email_live_recipients", json={"on": True})
    c = _approved_case(client)
    r = client.post(f"/api/v1/cases/{c['id']}/send", json={"draft": "notification", "rev": 1, "confirm": True})
    assert r.status_code == 200 and r.json()["redirected"] == 0
    sent = FakeSMTP.outbox[-1]
    assert sent["headers"]["To"] == "facilities-ai@test.edu" and "Bcc" not in sent["headers"]   # people never see each other
    assert len(sent["to_addrs"]) == r.json()["intended"] + 1
    # turning Send email off also turns live delivery off
    s = client.put("/api/v1/settings/send_email", json={"on": False}).json()
    assert not any(x["on"] for x in s["switches"])


def test_test_mode_without_test_address_is_refused(client, mail):
    mail.STATE["mailer"] = SmtpMailer(env={k: v for k, v in MAIL_ENV.items() if k != "FAI_EMAIL_REDIRECT_TO"},
                                      client_factory=FakeSMTP)
    client.put("/api/v1/settings/send_email", json={"on": True})
    c = _approved_case(client)
    r = client.post(f"/api/v1/cases/{c['id']}/send", json={"draft": "coordination", "rev": 1, "confirm": True})
    assert r.status_code == 409 and "FAI_EMAIL_REDIRECT_TO" in r.json()["detail"]
    assert not FakeSMTP.outbox


def test_switch_turns_itself_off(client, mail):
    s = client.put("/api/v1/settings/send_email", json={"on": True, "minutes": 15}).json()
    assert _sw(s, "send_email")["expires"]
    assert client.put("/api/v1/settings/send_email", json={"on": True, "minutes": 7}).status_code == 422
    settings = mail.STATE["settings"]
    with settings.store._lock:
        settings.db.execute("UPDATE settings SET expires=? WHERE key='send_email'",
                            ((datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds"),))
        settings.db.commit()
    assert settings.is_on("send_email") is False
    assert any(a["action"] == "switch_expired" for a in client.get("/api/v1/activity").json()["audit"])


def test_agents_stay_locked_even_when_sending_is_switched_on(client, mail):
    client.put("/api/v1/settings/send_email", json={"on": True})
    assert mail.STATE["coord"].outage.call("send_email").status == "unauthorized"
    assert client.post("/api/v1/actions/send_email").status_code == 403
