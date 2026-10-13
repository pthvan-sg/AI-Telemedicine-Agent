import json

import pytest

from app import create_app
from fakes import GOOD_NOTE, ScriptedLLM, call, say
from medscribe import tools
from medscribe.agent import ConsultationAgent
from medscribe.llm import Turn
from medscribe.mcp_backend import MCPBackend, make_backend
from medscribe.samples import SAMPLE_TRANSCRIPT


# ── MCP ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mcp_backend():
    backend = MCPBackend()
    yield backend
    backend.close()


def test_mcp_server_lists_the_same_tools(mcp_backend):
    names = {s["function"]["name"] for s in mcp_backend.specs()}
    assert names == set(tools.REGISTRY)
    spec = next(s for s in mcp_backend.specs() if s["function"]["name"] == "lookup_drug_interactions")
    assert "drugs" in spec["function"]["parameters"]["properties"]


def test_mcp_calls_match_local_calls(mcp_backend):
    args = {"drugs": ["Zoloft", "Imitrex"]}
    assert mcp_backend.call("lookup_drug_interactions", args) == tools.lookup_drug_interactions(**args)
    assert mcp_backend.call("check_allergy_conflicts", {"allergies": ["penicillin"], "drugs": ["amoxicillin"]}) \
        == tools.check_allergy_conflicts(["penicillin"], ["amoxicillin"])
    assert mcp_backend.call("get_patient_history", {"patient_id": "P-1003"})["allergies"] == ["sulfa drugs"]


def test_mcp_tool_error_raises(mcp_backend):
    with pytest.raises(tools.ToolError):
        mcp_backend.call("lookup_icd10", {})


def test_agent_runs_end_to_end_over_mcp(mcp_backend):
    llm = ScriptedLLM([say(call("lookup_drug_interactions", drugs=["ibuprofen", "sumatriptan"])),
                       say(call("submit_note", **GOOD_NOTE))])
    result = ConsultationAgent(llm, mcp_backend, use_critic=False).run(SAMPLE_TRANSCRIPT)
    assert result.status == "accepted" and result.trajectory["tool_success_rate"] == 1.0


def test_make_backend_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_backend("carrier-pigeon")


# ── Flask API ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    llm = ScriptedLLM([say(call("submit_note", **GOOD_NOTE))])
    app = create_app(llm_factory=lambda: llm, backend_factory=tools.LocalBackend)
    app.config["LLM"] = llm
    return app.test_client()


def test_health_and_ready_without_key():
    c = create_app().test_client()
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.get("/").get_json()["version"] == "2.0.0"


def test_analyze_returns_note_alerts_and_run_stats(client):
    r = client.post("/analyze", json={"transcript": SAMPLE_TRANSCRIPT})
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "success"
    assert body["summary"]["prescriptions"] == ["sumatriptan"]      # old clients read "summary"
    assert body["verification"]["ok"] and body["run"]["llm_calls"] == 1
    assert "trace" not in body


def test_analyze_can_include_trace(client):
    body = client.post("/analyze", json={"transcript": SAMPLE_TRANSCRIPT, "include_trace": True}).get_json()
    assert any(e["type"] == "verify" for e in body["trace"])


@pytest.mark.parametrize("payload,code", [
    (None, 400), ({}, 400), ({"transcript": ""}, 400), ({"transcript": 5}, 400),
    ({"transcript": "x", "patient_id": 7}, 400), ({"transcript": "x", "save": True}, 400),
])
def test_analyze_validation(client, payload, code):
    r = client.post("/analyze", json=payload) if payload is not None else client.post("/analyze", data="nope")
    assert r.status_code == code and r.get_json()["status"] == "error"


def test_oversized_transcript_rejected(client, monkeypatch):
    from medscribe import config
    monkeypatch.setattr(config, "MAX_TRANSCRIPT_CHARS", 50)
    assert client.post("/analyze", json={"transcript": "x" * 51}).status_code == 413


def test_agent_failure_is_a_502(client):
    llm = ScriptedLLM([say(text="a"), say(text="b"), say(text="c")])
    c = create_app(llm_factory=lambda: llm, backend_factory=tools.LocalBackend).test_client()
    r = c.post("/analyze", json={"transcript": SAMPLE_TRANSCRIPT})
    assert r.status_code == 502 and r.get_json()["status"] == "error"


def test_missing_api_key_is_503_not_500(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = create_app(backend_factory=tools.LocalBackend).test_client()
    r = c.post("/analyze", json={"transcript": "Doctor: hi"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.get_json()["message"]


def test_unexpected_error_does_not_leak_details():
    class Boom:
        def chat(self, *a, **k):
            raise ValueError("secret internal detail")
    # the agent captures LLM errors, so break the backend instead
    class BadBackend(tools.LocalBackend):
        def specs(self):
            raise RuntimeError("secret internal detail")
    c = create_app(llm_factory=Boom, backend_factory=BadBackend).test_client()
    r = c.post("/analyze", json={"transcript": "Doctor: hi"})
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)


def test_baseline_endpoint_makes_one_json_mode_call():
    seen = {}

    class OneShot:
        def chat(self, messages, tools=None, max_tokens=None, json_mode=False):
            seen.update(tools=tools, json_mode=json_mode)
            return Turn(content=json.dumps({"patient_chief_complaint": "headache", "symptoms": ["headache"]}))
    c = create_app(llm_factory=OneShot).test_client()
    r = c.post("/analyze/baseline", json={"transcript": SAMPLE_TRANSCRIPT})
    assert r.status_code == 200 and r.get_json()["summary"]["symptoms"] == ["headache"]
    assert seen == {"tools": None, "json_mode": True}


def test_sample_endpoint(client):
    assert client.post("/analyze/sample").status_code == 200


def test_patient_endpoint(client):
    assert client.get("/patients/P-1002").get_json()["medications"][0].startswith("warfarin")
    assert client.get("/patients/zzz").status_code == 404


def test_api_key_protection(client, monkeypatch):
    monkeypatch.setenv("MEDSCRIBE_API_KEY", "s3cret")
    assert client.get("/health").status_code == 200
    assert client.post("/analyze", json={"transcript": "x"}).status_code == 401
    assert client.get("/patients/P-1001").status_code == 401
    assert client.get("/patients/P-1001", headers={"X-API-Key": "s3cret"}).status_code == 200
