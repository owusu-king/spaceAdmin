"""Agent initialization, responsibilities, permissions, communication and workflows."""
import pytest

from app.agents.base import AgentMessage, ExecutionLimitReached, PermissionDenied
from app.agents.schemas import OutageRequest, SpaceRequest
from app.scenarios import resolve_payload, scenarios


def _scenario(system, sid):
    return resolve_payload(next(s for s in scenarios() if s.id == sid), system.corpus)


def test_agents_are_distinct(system):
    s, o = system.space.describe(), system.outage.describe()
    assert s["name"] != o["name"]
    assert not set(s["permissions"]) & set(o["permissions"])      # no shared tools
    assert "rank_candidate_spaces" in s["permissions"] and "parse_outage_notice" in o["permissions"]


def test_permission_boundary(system):
    with pytest.raises(PermissionDenied):
        system.outage.call("search_spaces", function="Office", buildings=[], open_only=True)
    with pytest.raises(PermissionDenied):
        system.space.call("draft_communication")


@pytest.mark.parametrize("action", ["send_email", "write_tririga", "assign_space", "approve_outage"])
def test_locked_actions_refused(system, action):
    assert system.space.call(action).status == "unauthorized"


def test_execution_limit(system):
    agent = system.space
    agent.reset()
    agent.max_tool_calls, saved = 2, agent.max_tool_calls
    try:
        agent.call("resolve_building", text="BRC")
        agent.call("resolve_building", text="BRC")
        with pytest.raises(ExecutionLimitReached):
            agent.call("resolve_building", text="BRC")
    finally:
        agent.max_tool_calls = saved
        agent.reset()


def test_tool_errors_are_contained(system):
    r = system.space.call("get_space_details", space_id=None)
    assert r.status in ("error", "not_found")


def test_space_office_request(system):
    r = system.run_space(SpaceRequest(request_type="new_office", department="Chemistry", function="Office", min_area=120))
    assert r["candidates"] and r["review_status"] == "pending"
    for c in r["candidates"]:
        assert c["status"] != "incompatible" or c["constraints_failed"]
        assert c["area"] is None or c["area"] >= 120 or c["status"] == "incompatible"
        assert 0 <= c["suitability"] <= 1 and 0 <= c["evidence_coverage"] <= 1


def test_space_lab_request_is_conditional(system):
    r = system.run_space(SpaceRequest(request_type="department_function", department="BioSciences",
                                      function="Research Lab", min_area=400))
    assert all(c["status"] != "potentially_suitable" for c in r["candidates"])
    assert any("Safety" in a for a in r["required_approvals"])


def test_space_unknown_department_asks(system):
    r = system.run_space(SpaceRequest(department="Department of Nothing", function="Office"))
    assert any("not on record" in q for q in r["verification_questions"])


def test_outage_exclusions_preserved(system):
    r = system.run_outage(OutageRequest(**_scenario(system, "OUT-OES-EXHAUST")))
    excl = " ".join(i["statement"] for i in r["impacts"]["explicitly_unaffected"]).lower()
    assert "fume hoods" in excl and "snorkels" in excl
    assert r["outage"]["proposed_start"] == "2026-11-19T07:00:00"


def test_outage_preliminary_vs_confirmed(system):
    r = system.run_outage(OutageRequest(**_scenario(system, "OUT-ABL-STEAM")))
    prelim = {p["statement"].lower(): p["later_confirmed"] for p in r["impacts"]["preliminary_superseded"]}
    assert any("heating" in k and not v for k, v in prelim.items())
    assert any("autoclave" in i["statement"].lower() for i in r["impacts"]["confirmed"])
    assert r["incident"]["set_aside"] > 0


def test_outage_uses_model_predictions(system):
    r = system.run_outage(OutageRequest(**_scenario(system, "OUT-OES-EXHAUST")))
    assert any(i["classification"] == "inferred" and i["source"].startswith("model") for i in r["impacts"]["potential"])


def test_cross_agent_communication(system):
    r = system.run_outage(OutageRequest(**_scenario(system, "OUT-ABL-STEAM")), cooperate=True)
    intents = [m["intent"] for m in r["messages"]]
    assert intents == ["temporary_space_request", "temporary_space_options"]
    opts = r["space_support"]["reply"]["payload"]["candidates"]
    assert opts and all(c["building"] != "Biology" for c in opts)


def test_unpermitted_intent_rejected(system):
    reply = system._relay(AgentMessage("space_admin", "outage", "send_everything", {}))
    assert reply.intent == "rejected"


def test_temporary_space_is_on_demand_only(system):
    r = system.run_outage(OutageRequest(**_scenario(system, "OUT-ABL-STEAM")))
    assert r["space_support"] is None and r["messages"] == []


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        SpaceRequest(function="Garage")
    with pytest.raises(ValueError):
        OutageRequest(text="short")
