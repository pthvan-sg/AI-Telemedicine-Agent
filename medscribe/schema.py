"""The visit note schema, the JSON Schema given to the model, and input coercion."""

from typing import Any

LIST_FIELDS = ["symptoms", "allergies", "medical_history", "current_medications",
               "plan", "prescriptions", "red_flags"]
SCALAR_FIELDS = ["patient_chief_complaint", "assessment", "follow_up"]
DETAIL_SCALARS = ["duration", "severity", "location"]
DETAIL_LISTS = ["aggravating_factors", "relieving_factors"]

_STR_LIST = {"type": "array", "items": {"type": "string"}}

NOTE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_chief_complaint": {"type": "string", "description": "Main reason for the visit, as stated."},
        "symptoms": {**_STR_LIST, "description": "Symptoms the patient reported."},
        "symptom_details": {
            "type": "object",
            "properties": {
                "duration": {"type": "string", "description": "Empty string if not stated."},
                "severity": {"type": "string", "description": "Empty string if not stated."},
                "location": {"type": "string", "description": "Empty string if not stated."},
                "aggravating_factors": _STR_LIST,
                "relieving_factors": _STR_LIST,
            },
        },
        "allergies": {**_STR_LIST, "description": "Allergies stated in the visit or in stored history. Empty if never discussed."},
        "medical_history": {**_STR_LIST, "description": "Past or family history mentioned."},
        "current_medications": {**_STR_LIST, "description": "Medications the patient currently takes, including from stored history."},
        "assessment": {"type": "string", "description": "The doctor's stated assessment. Never add a diagnosis the doctor did not state."},
        "plan": {**_STR_LIST, "description": "Plan items the doctor stated."},
        "prescriptions": {**_STR_LIST, "description": "New medications the doctor prescribed. Include a dose only if the doctor said one."},
        "follow_up": {"type": "string", "description": "Follow-up instructions as stated."},
        "red_flags": {**_STR_LIST, "description": "Warning signs the doctor told the patient to watch for."},
        "visit_summary": {"type": "string", "description": "2-3 sentence plain language summary."},
        "suggested_icd10": {
            "type": "array",
            "description": "Codes copied from lookup_icd10 results only.",
            "items": {"type": "object", "properties": {"code": {"type": "string"}, "description": {"type": "string"}},
                      "required": ["code"]},
        },
        "guideline_references": {**_STR_LIST, "description": "Chunk ids returned by search_guidelines that informed the note."},
        "evidence": {
            "type": "array",
            "description": "One entry per item in the note (not visit_summary). Every item must be backed by a verbatim quote.",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "e.g. symptoms, plan, assessment, symptom_details.duration"},
                    "item": {"type": "string", "description": "Exact copy of the note item this supports."},
                    "quote": {"type": "string", "description": "Verbatim span copied from the source. Include the preceding question if the answer alone is unclear."},
                    "source": {"type": "string", "enum": ["transcript", "history"]},
                },
                "required": ["field", "item", "quote", "source"],
            },
        },
    },
    "required": ["patient_chief_complaint", "symptoms", "evidence"],
}


def _s(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v).strip())


def _list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [_s(x) for x in v if _s(x)]


def normalize_note(raw: dict | None) -> dict:
    """Coerce whatever the model returned into the canonical note shape."""
    raw = raw if isinstance(raw, dict) else {}
    details_in = raw.get("symptom_details") if isinstance(raw.get("symptom_details"), dict) else {}
    note: dict[str, Any] = {f: _s(raw.get(f)) for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        note[f] = _list(raw.get(f))
    note["symptom_details"] = {f: _s(details_in.get(f)) for f in DETAIL_SCALARS}
    for f in DETAIL_LISTS:
        note["symptom_details"][f] = _list(details_in.get(f))
    note["visit_summary"] = _s(raw.get("visit_summary"))
    icd = []
    for e in raw.get("suggested_icd10") or []:
        if isinstance(e, dict) and _s(e.get("code")):
            icd.append({"code": _s(e["code"]), "description": _s(e.get("description"))})
    note["suggested_icd10"] = icd
    note["guideline_references"] = _list(raw.get("guideline_references"))
    ev = []
    for e in raw.get("evidence") or []:
        if isinstance(e, dict):
            ev.append({k: _s(e.get(k)) for k in ("field", "item", "quote", "source")})
    note["evidence"] = ev
    return note
