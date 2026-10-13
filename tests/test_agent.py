import json

import pytest

from fakes import GOOD_NOTE, ScriptedLLM, call, ev, note_with, say
from medscribe import tools
from medscribe.agent import ConsultationAgent
from medscribe.context import ContextManager
from medscribe.samples import SAMPLE_TRANSCRIPT


def run(script, transcript=SAMPLE_TRANSCRIPT, patient_id=None, save=False, **agent_kw):
    llm = ScriptedLLM(script)
    agent = ConsultationAgent(llm, use_critic=False, **agent_kw)
    return agent.run(transcript, patient_id, save=save), llm


def tool_replies(llm_call):
    return [json.loads(m["content"]) for m in llm_call["messages"] if m["role"] == "tool"]


def test_happy_path_uses_tools_then_submits():
    code = tools.lookup_icd10("migraine")["results"][0]["code"]
    note = note_with(suggested_icd10=[{"code": code, "description": "x"}])
    result, llm = run([
        say(call("lookup_drug_interactions", drugs=["ibuprofen", "sumatriptan"]), call("lookup_icd10", condition="migraine")),
        say(call("submit_note", **note)),
    ])
    assert result.status == "accepted"
    assert result.verification["ok"] and result.verification["pruned"] == []
    assert result.trajectory["tool_calls"] == 2 and result.trajectory["tool_success_rate"] == 1.0
    assert result.trajectory["llm_calls"] == 2 and result.trajectory["revisions"] == 0
    assert result.alerts == []
    assert result.note["suggested_icd10"][0]["code"] == code
    # prescription written, allergy status never discussed: reported as a gap, not invented
    assert any(g["field"] == "allergies" for g in result.documentation_gaps)
    assert result.note["allergies"] == []


def test_fabricated_item_is_rejected_then_fixed():
    bad = note_with(current_medications=["ibuprofen", "warfarin"])
    bad["evidence"].append(ev("current_medications", "warfarin", "I take warfarin every day"))
    result, llm = run([say(call("submit_note", **bad)), say(call("submit_note", **GOOD_NOTE))])
    assert result.status == "accepted" and result.trajectory["revisions"] == 1
    rejection = tool_replies(llm.agent_calls[1])[-1]
    assert rejection["status"] == "rejected"
    assert {i["item"] for i in rejection["issues"]} == {"warfarin"}
    assert "warfarin" not in result.note["current_medications"]


def test_pruning_when_revision_budget_is_spent():
    bad = note_with(symptoms=["headache", "fever"], assessment="meningitis")
    bad["evidence"] += [ev("symptoms", "fever", "fever for two days"), ev("assessment", "meningitis", "this is meningitis")]
    result, _ = run([say(call("submit_note", **bad))], max_revisions=0)
    assert result.status == "accepted_with_pruning"
    assert "fever" not in result.note["symptoms"] and result.note["assessment"] == ""
    assert {p["item"] for p in result.verification["pruned"]} == {"fever", "meningitis"}
    assert result.verification["ok"]


def test_safety_alerts_do_not_depend_on_the_model_calling_tools():
    transcript = ("Doctor: What's the problem?\nPatient: My knee hurts when I climb stairs.\n"
                  "Doctor: Any other medicines?\nPatient: Just my usual ones.\n"
                  "Doctor: For the knee, take ibuprofen 400 mg three times a day for a week.")
    note = {"patient_chief_complaint": "knee pain", "symptoms": ["knee pain"],
            "prescriptions": ["ibuprofen 400 mg three times a day"],
            "evidence": [ev("patient_chief_complaint", "knee pain", "My knee hurts when I climb stairs"),
                         ev("symptoms", "knee pain", "My knee hurts when I climb stairs"),
                         ev("prescriptions", "ibuprofen 400 mg three times a day", "take ibuprofen 400 mg three times a day")]}
    # P-1002 is on warfarin and sertraline in stored history. The model never calls a safety tool.
    result, _ = run([say(call("submit_note", **note))], transcript, patient_id="P-1002")
    keys = {a["key"] for a in result.alerts}
    assert {"interaction:ibuprofen+warfarin", "interaction:ibuprofen+sertraline", "interaction:sertraline+warfarin"} <= keys
    assert result.alerts[0]["severity"] == "major"
    assert result.note["safety_alerts"] == result.alerts


def test_allergy_conflict_comes_from_stored_history():
    transcript = "Doctor: You have a urinary infection. I'll start amoxicillin 500 mg three times a day."
    note = {"patient_chief_complaint": "urinary infection", "assessment": "urinary infection",
            "prescriptions": ["amoxicillin 500 mg three times a day"],
            "evidence": [ev("patient_chief_complaint", "urinary infection", "You have a urinary infection"),
                         ev("assessment", "urinary infection", "You have a urinary infection"),
                         ev("prescriptions", "amoxicillin 500 mg three times a day", "I'll start amoxicillin 500 mg three times a day")]}
    result, _ = run([say(call("submit_note", **note))], transcript, patient_id="P-1001")
    assert [a["key"] for a in result.alerts] == ["allergy:penicillin->amoxicillin"]
    assert not any(g["field"] == "allergies" for g in result.documentation_gaps)  # history documents it


def test_history_backed_medication_is_accepted():
    transcript = "Doctor: How is the blood pressure?\nPatient: Fine, I take my usual pill.\nDoctor: Good, carry on."
    note = {"patient_chief_complaint": "blood pressure review", "current_medications": ["lisinopril 10 mg daily"],
            "evidence": [ev("patient_chief_complaint", "blood pressure review", "How is the blood pressure"),
                         ev("current_medications", "lisinopril 10 mg daily", "medications: lisinopril 10 mg daily", "history")]}
    result, llm = run([say(call("get_patient_history", patient_id="P-1001")), say(call("submit_note", **note))],
                      transcript, patient_id="P-1001")
    assert result.status == "accepted" and result.note["current_medications"] == ["lisinopril 10 mg daily"]
    assert tool_replies(llm.agent_calls[1])[0]["found"] is True


def test_bad_tool_arguments_are_reported_to_the_model_and_counted():
    result, llm = run([say(call("lookup_icd10")), say(call("submit_note", **GOOD_NOTE))])
    assert "error" in tool_replies(llm.agent_calls[1])[0]
    assert result.status == "accepted"
    assert result.trajectory["tool_errors"] == 1 and result.trajectory["tool_success_rate"] == 0.0


def test_unknown_tool_is_an_error_not_a_crash():
    result, llm = run([say(call("delete_everything")), say(call("submit_note", **GOOD_NOTE))])
    assert "unknown tool" in tool_replies(llm.agent_calls[1])[0]["error"]


def test_search_transcript_returns_context():
    result, llm = run([say(call("search_transcript", query="sumatriptan", k=1)), say(call("submit_note", **GOOD_NOTE))])
    hit = tool_replies(llm.agent_calls[1])[0]["results"][0]
    assert "sumatriptan" in hit["text"] and hit["previous"].startswith("Patient:")


def test_patient_history_tool_hidden_without_patient_id():
    _, llm = run([say(call("submit_note", **GOOD_NOTE))])
    names = {t["function"]["name"] for t in llm.agent_calls[0]["tools"]}
    assert "get_patient_history" not in names and {"submit_note", "search_transcript", "lookup_icd10"} <= names
    _, llm2 = run([say(call("submit_note", **GOOD_NOTE))], patient_id="P-1001")
    assert "get_patient_history" in {t["function"]["name"] for t in llm2.agent_calls[0]["tools"]}


def test_gives_up_cleanly_if_model_never_submits():
    result, _ = run([say(text="Here is a summary"), say(text="Still no tool"), say(text="Nope")])
    assert result.status == "failed" and "without submitting" in result.error
    assert result.note["symptoms"] == []


def test_step_limit():
    result, _ = run([say(call("lookup_icd10", condition="x"))] * 3, max_steps=3)
    assert result.status == "failed" and "3 steps" in result.error


def test_llm_exception_is_captured():
    class Boom:
        def chat(self, *a, **k):
            raise ConnectionError("network down")
    result = ConsultationAgent(Boom(), use_critic=False).run(SAMPLE_TRANSCRIPT)
    assert result.status == "failed" and "network down" in result.error


def test_long_transcript_is_condensed_but_verified_against_the_full_text():
    filler = "\n".join(f"Doctor: Question {i}?\nPatient: Answer number {i}, nothing special." for i in range(60))
    transcript = SAMPLE_TRANSCRIPT + "\n" + filler
    llm = ScriptedLLM([say(call("submit_note", **GOOD_NOTE))], summary="Headache for 5 days, migraine suspected")
    agent = ConsultationAgent(llm, use_critic=False)
    agent.context = ContextManager(200)
    result = agent.run(transcript)
    assert result.condensed and llm.aux_calls and result.status == "accepted"
    prompt = llm.agent_calls[0]["messages"][1]["content"]
    assert "Running summary" in prompt and "Headache for 5 days" in prompt
    assert "Question 59" in prompt and "Question 0?" not in prompt
    assert len(prompt) < len(transcript)


def test_save_writes_verified_note_to_memory():
    note = note_with()
    result, _ = run([say(call("submit_note", **note))], patient_id="P-SAVE-1", save=True)
    saved = tools.get_store().get_patient("P-SAVE-1")
    assert "sumatriptan" in saved["medications"] and saved["recent_visits"][0]["assessment"] == "migraines"


def test_failed_run_never_saves():
    run([say(text="x"), say(text="y"), say(text="z")], patient_id="P-SAVE-2", save=True)
    assert tools.get_store().get_patient("P-SAVE-2") is None


def test_trajectory_is_written_when_trace_dir_is_set(tmp_path, monkeypatch):
    from medscribe import config
    monkeypatch.setattr(config, "TRACE_DIR", str(tmp_path))
    run([say(call("submit_note", **GOOD_NOTE))])
    line = json.loads((tmp_path / "trajectories.jsonl").read_text().splitlines()[0])
    assert line["run"]["status"] == "accepted" and any(e["type"] == "verify" for e in line["events"])


def test_reward_prefers_clean_runs():
    clean, _ = run([say(call("submit_note", **GOOD_NOTE))])
    bad = note_with(symptoms=["headache", "fever"])
    bad["evidence"].append(ev("symptoms", "fever", "fever for two days"))
    messy, _ = run([say(call("submit_note", **bad)), say(call("submit_note", **GOOD_NOTE))])
    failed, _ = run([say(text="a"), say(text="b"), say(text="c")])
    assert clean.trajectory["reward"] > messy.trajectory["reward"] > failed.trajectory["reward"]
