import json

from fakes import GOOD_NOTE, ScriptedLLM, ev, note_with
from medscribe import tools
from medscribe.llm import Turn
from medscribe.samples import SAMPLE_TRANSCRIPT
from medscribe.schema import normalize_note
from medscribe.verifier import Source, Verifier, prune

V = Verifier(tools.get_kb())


def verify(note, transcript=SAMPLE_TRANSCRIPT, **kw):
    return V.verify(normalize_note(note), transcript, **kw)


def kinds(report):
    return {(i.field, i.kind) for i in report.blocking}


def test_good_note_passes_with_no_blocking_issues():
    r = verify(GOOD_NOTE)
    assert r.ok and r.grounded_ratio == 1.0 and r.n_items > 15


def test_missing_evidence_is_blocking():
    note = note_with()
    note["evidence"] = [e for e in note["evidence"] if e["item"] != "ibuprofen"]
    assert ("current_medications", "no_evidence") in kinds(verify(note))


def test_fabricated_quote_is_blocking():
    note = note_with(symptoms=["headache", "fever"])
    note["evidence"].append(ev("symptoms", "fever", "I have had a fever since Monday"))
    assert ("symptoms", "quote_not_found") in kinds(verify(note))


def test_quote_matching_ignores_case_and_punctuation():
    s = Source(SAMPLE_TRANSCRIPT)
    assert s.locate("LIGHT bothers me a lot!!") is not None
    assert s.locate("light ... nauseous") is not None       # ellipsis pieces must appear in order
    assert s.locate("nauseous ... light") is None
    assert s.locate("ab") is None                            # too short to mean anything


def test_invented_drug_is_blocking():
    note = note_with(current_medications=["ibuprofen", "warfarin"])
    note["evidence"].append(ev("current_medications", "warfarin", "Just ibuprofen, but it's not really helping"))
    assert ("current_medications", "drug_not_in_source") in kinds(verify(note))


def test_invented_dose_is_blocking():
    note = note_with(prescriptions=["sumatriptan 100 mg"])
    note["evidence"].append(ev("prescriptions", "sumatriptan 100 mg",
                               "I'm going to prescribe you sumatriptan for acute attacks"))
    assert ("prescriptions", "bad_number") in kinds(verify(note))


def test_spoken_numbers_match_digits():
    note = note_with(follow_up="return in 2 weeks if it does not improve")
    note["evidence"].append(ev("follow_up", "return in 2 weeks if it does not improve",
                               "Come back in two weeks if it doesn't improve"))
    assert verify(note).ok


def test_patient_speech_cannot_back_a_prescription():
    transcript = ("Doctor: What brings you in?\nPatient: A cough. Can I get antibiotics?\n"
                  "Doctor: Let's wait and see, rest and fluids.")
    note = {"patient_chief_complaint": "cough", "symptoms": ["cough"], "prescriptions": ["antibiotics"],
            "evidence": [ev("patient_chief_complaint", "cough", "A cough"), ev("symptoms", "cough", "A cough"),
                         ev("prescriptions", "antibiotics", "Can I get antibiotics")]}
    assert ("prescriptions", "wrong_speaker") in kinds(verify(note, transcript))


def test_history_evidence_checked_against_history_text():
    hist = "medications: lisinopril 10 mg daily"
    note = {"patient_chief_complaint": "x", "current_medications": ["lisinopril 10 mg daily"],
            "evidence": [ev("patient_chief_complaint", "x", "Doctor: hello"),
                         ev("current_medications", "lisinopril 10 mg daily", "lisinopril 10 mg daily", "history")]}
    r = verify(note, "Doctor: hello\nPatient: hi", history_text=hist)
    assert ("current_medications", "no_evidence") not in kinds(r) and ("current_medications", "quote_not_found") not in kinds(r)
    assert not verify(note, "Doctor: hello\nPatient: hi", history_text="").ok


def test_icd_codes_and_guideline_ids_must_come_from_tools():
    note = note_with(suggested_icd10=[{"code": "G43.909", "description": "x"}], guideline_references=["migraine#2"])
    assert {("suggested_icd10", "unknown_code"), ("guideline_references", "unknown_reference")} <= kinds(verify(note))
    ok = verify(note, seen_icd={"G43.909"}, seen_refs={"migraine#2"})
    assert ok.ok


def test_summary_cannot_add_drugs_or_numbers():
    r = verify(note_with(visit_summary="Prescribed warfarin 40 mg."))
    assert ("visit_summary", "summary_unsupported") in kinds(r)


def test_weak_support_is_a_warning_without_critic():
    r = verify(GOOD_NOTE)
    assert r.ok and any(i.kind == "weak_support" for i in r.warnings)  # "nausea" vs "nauseous"


def test_critic_can_reject_a_weak_item_and_approve_another():
    def critic(messages):
        pairs = messages[-1]["content"]
        out = []
        for line in pairs.splitlines():
            i = int(line.split(".")[0])
            out.append({"i": i, "supported": "nausea" in line})
        return json.dumps({"results": out})

    note = note_with(symptoms=["headache", "nausea", "dizziness"])
    note["evidence"].append(ev("symptoms", "dizziness", "I felt nauseous yesterday"))
    llm = ScriptedLLM([], critic=critic)
    r = Verifier(tools.get_kb(), llm).verify(normalize_note(note), SAMPLE_TRANSCRIPT)
    assert ("symptoms", "quote_does_not_support") in kinds(r)
    assert not any(i.item == "nausea" for i in r.blocking)


def test_critic_failure_degrades_to_warning():
    class Broken:
        def chat(self, *a, **k):
            return Turn(content="not json")
    r = Verifier(tools.get_kb(), Broken()).verify(normalize_note(GOOD_NOTE), SAMPLE_TRANSCRIPT)
    assert r.ok


def test_prune_removes_only_bad_items_and_their_evidence():
    note = normalize_note(note_with(symptoms=["headache", "fever"], assessment="stroke"))
    note["evidence"].append(ev("symptoms", "fever", "fever since Monday"))
    r = V.verify(note, SAMPLE_TRANSCRIPT)
    clean, removed = prune(note, r)
    assert "fever" not in clean["symptoms"] and "headache" in clean["symptoms"]
    assert clean["assessment"] == ""
    assert {x["item"] for x in removed} == {"fever", "stroke"}
    assert all(e["item"] not in ("fever", "stroke") for e in clean["evidence"])
    assert V.verify(clean, SAMPLE_TRANSCRIPT).ok
