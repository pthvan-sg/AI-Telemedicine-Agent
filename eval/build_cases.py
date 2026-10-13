"""Writes cases.jsonl. Run once after editing the cases below: python eval/build_cases.py

All transcripts are synthetic. `gold` lists, per field, groups of acceptable surface forms (any
alias in a group counts as a match). A field that is missing from `gold` is not scored for that
case. `traps` are things that must NOT appear: they measure hallucination directly.
"""
import json
from pathlib import Path

CASES = [
    {
        "id": "c01-migraine",
        "patient_id": None,
        "transcript": """Doctor: Good morning! How are you feeling today?
Patient: Not great, doctor. I've had this persistent headache for about 5 days now. It's mostly on the left side.
Doctor: On a scale of 1 to 10, how would you rate the pain?
Patient: Around a 6 or 7. It gets worse in the morning and when I stare at screens too long.
Doctor: Any nausea, vomiting, or sensitivity to light?
Patient: Yes, actually. Light bothers me a lot, and I felt nauseous yesterday.
Doctor: Any history of migraines in your family?
Patient: My mother has them, yes.
Doctor: Are you currently taking any medications?
Patient: Just ibuprofen, but it's not really helping.
Doctor: I see. I'm going to prescribe you sumatriptan for acute attacks, and I'd like you to keep a headache diary. Avoid known triggers like caffeine and irregular sleep. Come back in two weeks if it doesn't improve.
Patient: Should I be worried about anything serious?
Doctor: Given the pattern and family history, this looks like migraines. But if you experience sudden severe headache, vision changes, or weakness, go to the ER immediately.
Patient: Okay, thank you doctor.""",
        "gold": {
            "symptoms": [["headache"], ["light", "photophobia"], ["nausea", "nauseous"]],
            "current_medications": [["ibuprofen"]],
            "prescriptions": [["sumatriptan"]],
            "allergies": [],
            "red_flags": [["sudden severe headache", "thunderclap"], ["vision"], ["weakness"]],
        },
        "traps": {"allergies_must_be_empty": True, "prescriptions_no_dose": True},
        "expected_alerts": [],
    },
    {
        "id": "c02-uti-penicillin-allergy",
        "patient_id": "P-1001",
        "transcript": """Doctor: What brings you in today?
Patient: It burns when I pee, and I need to go all the time. Started three days ago.
Doctor: Any fever or back pain?
Patient: No fever, no back pain.
Doctor: Are you taking anything at the moment?
Patient: Just my blood pressure pill.
Doctor: This sounds like a urinary tract infection. I'll start you on amoxicillin 500 mg three times a day for five days.
Patient: Okay. Should I drink more water?
Doctor: Yes, plenty of fluids. If you get a fever, back pain or vomiting, come to the clinic the same day.
Patient: Alright.""",
        "gold": {
            "symptoms": [["burn", "dysuria"], ["frequen", "all the time"]],
            "current_medications": [["lisinopril", "blood pressure"]],
            "prescriptions": [["amoxicillin"]],
            "allergies": [["penicillin"]],
            "red_flags": [["fever"], ["back pain"], ["vomiting"]],
        },
        "traps": {"forbidden": {"symptoms": ["fever", "back pain"]}},
        "expected_alerts": ["allergy:penicillin->amoxicillin"],
    },
    {
        "id": "c03-knee-pain-warfarin",
        "patient_id": "P-1002",
        "transcript": """Doctor: What can I help you with?
Patient: My right knee has been sore for about two weeks, mostly when I climb stairs.
Doctor: Any swelling or injury?
Patient: Some swelling in the evening. No injury that I remember.
Doctor: Are you still on your usual medicines?
Patient: Yes, the same as always.
Doctor: Okay. This looks like early osteoarthritis. For the pain, take ibuprofen 400 mg three times a day for a week, and try some gentle exercises.
Patient: Will that be enough?
Doctor: Let's see how you get on. Book a follow-up in two weeks.""",
        "gold": {
            "symptoms": [["knee"], ["swelling"]],
            "current_medications": [["warfarin"], ["sertraline"]],
            "prescriptions": [["ibuprofen"]],
            "allergies": [],
        },
        "traps": {"allergies_must_be_empty": True, "forbidden": {"symptoms": ["injury"]}},
        "expected_alerts": ["interaction:ibuprofen+warfarin", "interaction:ibuprofen+sertraline",
                            "interaction:sertraline+warfarin"],
    },
    {
        "id": "c04-cold-denials",
        "patient_id": None,
        "transcript": """Doctor: Good afternoon. What's been going on?
Patient: I've had a runny nose and a sore throat since Sunday. Mostly a dry cough at night.
Doctor: Any fever, chest pain, or trouble breathing?
Patient: No fever. No chest pain and I'm breathing fine.
Doctor: Have you been around anyone sick?
Patient: My sister had strep throat last week, but she is on antibiotics. I'm not.
Doctor: Do you take any regular medication?
Patient: No, nothing. I'm allergic to latex though.
Doctor: Thanks. This looks like a viral upper respiratory infection. No antibiotics needed. Rest, fluids, and honey for the cough. If the sore throat lasts beyond a week or you develop a high fever, see us again.
Patient: Great, thanks.""",
        "gold": {
            "symptoms": [["runny nose"], ["sore throat"], ["cough"]],
            "current_medications": [],
            "prescriptions": [],
            "allergies": [["latex"]],
        },
        "traps": {"forbidden": {"symptoms": ["fever", "chest pain", "breathing", "strep"],
                                "medical_history": ["strep"], "prescriptions": ["antibiotic"],
                                "current_medications": ["antibiotic"]}},
        "expected_alerts": [],
    },
    {
        "id": "c05-sinusitis-sulfa-allergy",
        "patient_id": "P-1003",
        "transcript": """Doctor: How have you been since your last visit?
Patient: Diabetes is fine, but for ten days I've had facial pressure, thick nasal discharge and a stuffy nose.
Doctor: Any fever?
Patient: A mild one, 38 degrees on Monday.
Doctor: Alright. This sounds like bacterial sinusitis since it has lasted more than ten days. I'll prescribe Bactrim, one tablet twice a day for seven days.
Patient: Okay.
Doctor: Keep taking your metformin as usual and check your sugars. Come back if you are not better in a week.""",
        "gold": {
            "symptoms": [["facial pressure"], ["nasal discharge"], ["stuffy", "congestion"], ["fever"]],
            "current_medications": [["metformin"]],
            "prescriptions": [["bactrim", "trimethoprim", "sulfamethoxazole"]],
            "allergies": [["sulfa"]],
        },
        "traps": {},
        "expected_alerts": ["allergy:sulfa drugs->trimethoprim-sulfamethoxazole"],
    },
    {
        "id": "c06-chest-pain-sildenafil",
        "patient_id": None,
        "transcript": """Doctor: Tell me what happened.
Patient: I get a tight pressure in my chest when I walk uphill, for about a month. It goes away when I rest.
Doctor: Does it spread anywhere?
Patient: Sometimes to my left arm.
Doctor: Do you take any medicines?
Patient: Viagra now and then, and a statin every night, atorvastatin.
Doctor: I'm concerned this is angina. I'll prescribe nitroglycerin spray to use when the pain comes on, and I'm arranging an ECG and stress test.
Patient: Is it serious?
Doctor: It needs checking. If the pain lasts more than 15 minutes, comes at rest, or you feel faint or short of breath, call an ambulance.""",
        "gold": {
            "symptoms": [["chest"], ["arm"]],
            "current_medications": [["viagra", "sildenafil"], ["atorvastatin", "statin"]],
            "prescriptions": [["nitroglycerin"]],
            "allergies": [],
            "red_flags": [["15 minutes"], ["at rest", "comes at rest"], ["faint"], ["breath"]],
        },
        "traps": {"allergies_must_be_empty": True, "prescriptions_no_dose": True},
        "expected_alerts": ["interaction:nitroglycerin+sildenafil"],
    },
    {
        "id": "c07-no-diagnosis-yet",
        "patient_id": None,
        "transcript": """Doctor: What brings you in?
Patient: I've been tired all the time for about two months, and I've gained some weight.
Doctor: Any changes in mood, or trouble sleeping?
Patient: Sleep is okay. Mood is a bit low sometimes.
Doctor: I don't want to jump to conclusions. Let's run blood tests including thyroid and iron, and see you in a week with the results.
Patient: Should I take iron tablets in the meantime?
Doctor: No, please wait for the results first.""",
        "gold": {
            "symptoms": [["tired", "fatigue"], ["weight"], ["mood"]],
            "current_medications": [],
            "prescriptions": [],
            "allergies": [],
        },
        "traps": {"assessment_must_be_empty": True, "allergies_must_be_empty": True,
                  "forbidden": {"prescriptions": ["iron", "thyroid", "levothyroxine"],
                                "symptoms": ["hypothyroid", "anemia", "anaemia", "depression"]}},
        "expected_alerts": [],
    },
    {
        "id": "c08-back-strain-otc",
        "patient_id": None,
        "transcript": """Doctor: What seems to be the problem?
Patient: Lower back pain for a week after moving boxes. It's a 5 out of 10 and worse when I bend forward.
Doctor: Any numbness, weakness in the legs, or trouble controlling your bladder?
Patient: None of those.
Doctor: Are you taking anything for it?
Patient: Paracetamol, 500 mg when it gets bad.
Doctor: That's fine to continue. This is a muscle strain. Stay active, use heat packs, and I'll refer you to physiotherapy.
Patient: Do I need an X-ray?
Doctor: Not at this stage. If you get numbness, leg weakness or problems with your bladder, come in straight away.""",
        "gold": {
            "symptoms": [["back pain"]],
            "current_medications": [["paracetamol", "acetaminophen"]],
            "prescriptions": [],
            "allergies": [],
            "red_flags": [["numbness"], ["weakness"], ["bladder"]],
        },
        "traps": {"allergies_must_be_empty": True,
                  "forbidden": {"symptoms": ["numbness", "weakness", "bladder"],
                                "prescriptions": ["paracetamol", "acetaminophen"]}},
        "expected_alerts": [],
    },
    {
        "id": "c09-anxiety-sertraline-triptan",
        "patient_id": None,
        "transcript": """Doctor: What would you like to talk about today?
Patient: I've been very anxious for about three months. My heart races and I can't concentrate at work.
Doctor: Any panic attacks?
Patient: A couple of times a month.
Doctor: Any regular medicines?
Patient: Sumatriptan when I get migraines. My friend takes tramadol for her back, but I don't.
Doctor: Thanks. This seems like generalized anxiety. I'd like to start sertraline 50 mg once daily in the morning, and refer you for cognitive behavioural therapy.
Patient: Any side effects?
Doctor: Some nausea at first, usually settles. Review in four weeks. If you get agitation, sweating, tremor or fever, call us the same day.""",
        "gold": {
            "symptoms": [["anxi"], ["heart rac", "palpitation"], ["concentrat"], ["panic"]],
            "current_medications": [["sumatriptan"]],
            "prescriptions": [["sertraline"]],
            "allergies": [],
            "medical_history": [["migraine"]],
            "red_flags": [["agitation"], ["sweating"], ["tremor"], ["fever"]],
        },
        "traps": {"allergies_must_be_empty": True,
                  "forbidden": {"current_medications": ["tramadol"], "symptoms": ["nausea"]}},
        "expected_alerts": ["interaction:sertraline+sumatriptan"],
    },
]

if __name__ == "__main__":
    out = Path(__file__).with_name("cases.jsonl")
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CASES) + "\n", encoding="utf-8")
    print(f"wrote {len(CASES)} cases to {out}")
