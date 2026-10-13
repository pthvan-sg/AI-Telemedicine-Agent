"""The original MedScribe design: one LLM call, JSON out. Kept as the baseline for evaluation.

There is no tool use, no retrieval, no memory and no verification here on purpose. The eval
harness compares this against ConsultationAgent on the same transcripts.
"""

import json

from .llm import LLM
from .schema import normalize_note

BASELINE_PROMPT = """You are a medical documentation AI assistant. Analyze doctor-patient
conversation transcripts and extract structured clinical information.

Always respond with ONLY a valid JSON object (no markdown, no backticks) with this structure:
{
  "patient_chief_complaint": "main reason for visit",
  "symptoms": ["list of symptoms"],
  "symptom_details": {
    "duration": "how long symptoms have been present",
    "severity": "pain scale or severity description",
    "location": "where symptoms are located",
    "aggravating_factors": ["things that make it worse"],
    "relieving_factors": ["things that make it better"]
  },
  "medical_history": ["relevant past medical history mentioned"],
  "current_medications": ["medications patient is currently taking"],
  "assessment": "doctor's diagnosis or assessment",
  "plan": ["list of treatment plan action items"],
  "prescriptions": ["new medications prescribed"],
  "follow_up": "follow-up instructions",
  "red_flags": ["warning signs that require emergency care"],
  "visit_summary": "2-3 sentence plain language summary of the visit"
}"""


def analyze_baseline(llm: LLM, transcript: str) -> dict:
    turn = llm.chat(
        [{"role": "system", "content": BASELINE_PROMPT},
         {"role": "user", "content": f"Analyze this doctor-patient transcript:\n\n{transcript}"}],
        tools=None, json_mode=True)
    note = normalize_note(json.loads(turn.content or "{}"))
    note["safety_alerts"] = []
    note["documentation_gaps"] = []
    return note
