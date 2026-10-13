from medscribe import tools
from medscribe.memory import PatientStore
from medscribe.rag import GuidelineIndex
from medscribe import config


def test_guideline_search_finds_relevant_chunk():
    hits = tools.search_guidelines("triptan with an SSRI serotonin syndrome", k=2)["results"]
    assert hits and hits[0]["id"].startswith("migraine#")


def test_icd_lookup():
    codes = [r["code"] for r in tools.lookup_icd10("migraine")["results"]]
    assert any(c.startswith("G43") for c in codes)


def test_interactions_use_aliases_and_doses():
    out = tools.lookup_drug_interactions(["Imitrex 50 mg", "Zoloft"])
    assert out["recognized"] == ["sumatriptan", "sertraline"]
    assert out["interactions"][0]["key"] == "interaction:sertraline+sumatriptan"


def test_no_interaction_for_ibuprofen_and_sumatriptan():
    assert tools.lookup_drug_interactions(["ibuprofen", "sumatriptan"])["interactions"] == []


def test_contraindicated_sorts_first():
    out = tools.lookup_drug_interactions(["nitroglycerin", "Viagra", "ibuprofen", "warfarin"])
    assert out["interactions"][0]["severity"] == "contraindicated"


def test_penicillin_allergy_covers_the_class():
    out = tools.check_allergy_conflicts(["penicillin"], ["amoxicillin"])
    assert out["conflicts"][0]["key"] == "allergy:penicillin->amoxicillin"


def test_sulfa_allergy():
    out = tools.check_allergy_conflicts(["sulfa drugs"], ["Bactrim"])
    assert out["conflicts"] and out["conflicts"][0]["drug"] == "trimethoprim-sulfamethoxazole"


def test_missing_info_flags_allergy_gap_with_prescription():
    gaps = tools.flag_missing_info({"prescriptions": ["sumatriptan"], "allergies": []})["missing"]
    fields = {g["field"] for g in gaps}
    assert "allergies" in fields and "prescriptions.dose" in fields


def test_patient_history_tool():
    assert tools.get_patient_history("P-1001")["allergies"] == ["penicillin"]
    assert tools.get_patient_history("nope") == {"found": False, "patient_id": "nope"}


def test_every_registered_tool_has_a_valid_schema():
    for spec in tools.REGISTRY.values():
        assert spec.parameters["type"] == "object"
        assert spec.description


def test_guideline_chunk_ids_are_unique():
    ids = [c["id"] for c in GuidelineIndex.from_dir(config.DATA_DIR / "guidelines").chunks]
    assert len(ids) == len(set(ids))


def test_memory_seed_is_idempotent_and_record_visit_merges(tmp_path):
    store = PatientStore(str(tmp_path / "m.db"))
    assert store.seed_from(config.DATA_DIR / "patients_seed.json") == 3
    assert store.seed_from(config.DATA_DIR / "patients_seed.json") == 0
    store.record_visit("P-1001", {"allergies": ["Penicillin", "latex"], "prescriptions": ["lisinopril 10 mg daily", "x"],
                                  "current_medications": [], "medical_history": [], "assessment": "a",
                                  "visit_summary": "s"})
    p = store.get_patient("P-1001")
    assert p["allergies"] == ["penicillin", "latex"]  # no duplicate, new item added
    assert p["medications"].count("lisinopril 10 mg daily") == 1 and "x" in p["medications"]
    assert len(p["recent_visits"]) == 2


def test_record_visit_creates_new_patient(tmp_path):
    store = PatientStore(str(tmp_path / "n.db"))
    store.record_visit("P-NEW", {"allergies": [], "prescriptions": ["sumatriptan"], "current_medications": [],
                                 "medical_history": [], "assessment": "", "visit_summary": ""})
    assert store.get_patient("P-NEW")["medications"] == ["sumatriptan"]
