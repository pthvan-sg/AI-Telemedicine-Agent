import json

from eval import run_eval
from eval.metrics import check_traps, item_matches, score_case, score_field
from medscribe import tools

CASES = run_eval.load_cases(run_eval.HERE / "cases.jsonl", None)


def gold_note(case):
    """A note that says exactly what the gold says (first alias of each group)."""
    note = {f: [g[0] for g in groups] for f, groups in case["gold"].items()}
    for f in ("symptoms", "current_medications", "prescriptions", "allergies", "red_flags", "medical_history"):
        note.setdefault(f, [])
    note["assessment"] = "" if case["traps"].get("assessment_must_be_empty") else "x"
    return note


def test_every_case_is_well_formed():
    assert len(CASES) >= 9 and len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        assert c["transcript"].strip() and "symptoms" in c["gold"]
        assert isinstance(c["expected_alerts"], list)


def test_gold_note_scores_perfectly_on_every_case():
    for c in CASES:
        s = score_case(c, gold_note(c), [{"key": k} for k in c["expected_alerts"]])
        assert all(x.fp == 0 and x.fn == 0 for x in s.fields.values()), c["id"]
        assert s.violations == [] and s.alerts.fn == 0 and s.alerts.fp == 0, c["id"]


def test_expected_alert_keys_are_what_the_tools_really_emit():
    """Guards the gold: if the drug KB changes, the eval cases must be updated with it."""
    kb = tools.get_kb()
    for c in CASES:
        if not c["expected_alerts"]:
            continue
        meds = kb.extract_all([c["transcript"]]) + kb.extract_all(
            (tools.get_patient_history(c["patient_id"]).get("medications", []) if c["patient_id"] else []))
        allergies = tools.get_patient_history(c["patient_id"]).get("allergies", []) if c["patient_id"] else []
        got = {a["key"] for a in kb.interactions(meds)} | {a["key"] for a in kb.allergy_conflicts(allergies, meds)}
        assert set(c["expected_alerts"]) <= got, (c["id"], got)


def test_alias_matching():
    assert item_matches("sensitivity to light", ["light", "photophobia"])
    assert item_matches("Heart racing", ["heart rac"])
    assert item_matches("lisinopril 10 mg daily", ["lisinopril"])
    assert not item_matches("headache", ["nausea"])


def test_precision_recall_and_duplicates():
    c = score_field(["headache", "migraine headache", "fever"], [["headache"], ["nausea"]])
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)   # the duplicate headache is ignored, fever is a false positive
    assert round(c.f1, 2) == 0.5
    assert score_field([], []).f1 == 1.0 and score_field([], []).precision == 1.0
    assert score_field(["x"], []).fp == 1


def test_traps():
    traps = {"allergies_must_be_empty": True, "prescriptions_no_dose": True, "assessment_must_be_empty": True,
             "forbidden": {"symptoms": ["fever"]}}
    note = {"allergies": ["latex"], "prescriptions": ["sumatriptan 50 mg"], "assessment": "flu",
            "symptoms": ["high fever"]}
    checked, bad = check_traps(note, traps)
    assert checked == 4 and len(bad) == 4
    assert check_traps({"allergies": [], "prescriptions": ["sumatriptan"], "assessment": "", "symptoms": []}, traps)[1] == []


def test_evaluate_with_stub_arms(monkeypatch):
    def perfect(llm, case, backend):
        return gold_note(case), [{"key": k} for k in case["expected_alerts"]], {
            "llm_calls": 2, "tool_calls": 4, "tool_errors": 1, "revisions": 1, "elapsed_ms": 2000}, []

    def awful(llm, case, backend):
        return {"symptoms": ["invented thing"], "allergies": ["invented"], "prescriptions": ["x 5 mg"]}, [], \
               {"llm_calls": 1, "tool_calls": 0, "tool_errors": 0, "revisions": 0, "elapsed_ms": 1000}, []
    monkeypatch.setitem(run_eval.ARMS, "perfect", perfect)
    monkeypatch.setitem(run_eval.ARMS, "awful", awful)
    res = run_eval.evaluate(None, CASES[:3], ["perfect", "awful"], repeats=2)
    p, a = res["perfect"], res["awful"]
    assert p["all_cases"]["extraction"]["f1"] == 1.0 and p["all_cases"]["hallucination"]["violations"] == 0
    assert p["agent_stats"]["tool_success_rate"] == 0.75 and p["agent_stats"]["avg_revisions"] == 1
    assert a["all_cases"]["extraction"]["f1"] == 0.0 and a["all_cases"]["hallucination"]["violation_rate"] > 0
    assert p["all_cases"]["cases"] == 6                       # 3 cases x 2 repeats
    md = run_eval.to_markdown(res, {"date": "d", "model": "m", "cases": 3, "repeats": 2})
    assert "| Extraction F1 (all cases) | 1.00 | 0.00 |" in md
    json.dumps(res)                                           # results must be serialisable


def test_one_crashing_run_does_not_abort_the_eval(monkeypatch):
    def boom(llm, case, backend):
        raise RuntimeError("api down")
    monkeypatch.setitem(run_eval.ARMS, "boom", boom)
    res = run_eval.evaluate(None, CASES[:1], ["boom"])
    assert res["boom"]["agent_stats"]["failed_runs"] == 1
