"""Turn a note into readable text or markdown."""

from datetime import datetime

DISCLAIMER = "AI-generated draft. A clinician must review it before it is used."


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "_None noted_"


def to_markdown(note: dict, alerts: list[dict] | None = None, gaps: list[dict] | None = None,
                verification: dict | None = None) -> str:
    alerts = note.get("safety_alerts", []) if alerts is None else alerts
    gaps = note.get("documentation_gaps", []) if gaps is None else gaps
    d = note.get("symptom_details", {})
    out = [f"# Consultation Summary\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n"]

    if alerts:
        out.append("## Safety alerts\n" + "\n".join(f"- **{a['severity'].upper()}**: {a['message']}" for a in alerts))
    out += [
        f"## Visit summary\n{note.get('visit_summary') or '_Not generated_'}",
        f"## Chief complaint\n{note.get('patient_chief_complaint') or '_Not stated_'}",
        f"## Assessment\n{note.get('assessment') or '_Not stated by the doctor_'}",
        "## Symptoms\n" + _bullets(note.get("symptoms", [])),
    ]
    detail_lines = []
    for key, label in (("duration", "Duration"), ("severity", "Severity"), ("location", "Location")):
        if d.get(key):
            detail_lines.append(f"- **{label}**: {d[key]}")
    for key, label in (("aggravating_factors", "Worse with"), ("relieving_factors", "Better with")):
        if d.get(key):
            detail_lines.append(f"- **{label}**: {', '.join(d[key])}")
    if detail_lines:
        out.append("### Symptom details\n" + "\n".join(detail_lines))
    out += [
        "## Allergies\n" + _bullets(note.get("allergies", [])),
        "## Current medications\n" + _bullets(note.get("current_medications", [])),
        "## Medical history\n" + _bullets(note.get("medical_history", [])),
        "## Plan\n" + _bullets(note.get("plan", [])),
        "## New prescriptions\n" + _bullets(note.get("prescriptions", [])),
        f"## Follow-up\n{note.get('follow_up') or '_Not stated_'}",
        "## Red flags\n" + _bullets(note.get("red_flags", [])),
    ]
    if note.get("suggested_icd10"):
        out.append("## Suggested ICD-10 (confirm before use)\n" +
                   "\n".join(f"- {c['code']}: {c.get('description', '')}" for c in note["suggested_icd10"]))
    if gaps:
        out.append("## Documentation gaps\n" + "\n".join(f"- {g['message']}" for g in gaps))
    if verification and verification.get("pruned"):
        out.append("## Removed during verification\n" +
                   "\n".join(f"- {p['field']}: {p['item']}" for p in verification["pruned"]))
    out.append(f"---\n_{DISCLAIMER}_")
    return "\n\n".join(out) + "\n"
