"""
MedScribe AI - Medical Consultation Summarizer
Analyzes doctor-patient transcripts and extracts structured clinical information.

Usage:
    python medical_summarizer.py                    # uses sample transcript
    python medical_summarizer.py transcript.txt     # reads from a .txt file
    python medical_summarizer.py transcript.docx    # reads from a .docx file
    python medical_summarizer.py --input            # type/paste transcript manually
"""

import json
import sys
import os
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

try:
    from docx import Document
except ImportError:
    Document = None

# ── Load API Key from .env file ───────────────────────────────────────────────

load_dotenv()  # reads OPENAI_API_KEY from .env file

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n  ERROR: OPENAI_API_KEY not found.")
    print("  Create a .env file in this folder with:")
    print("  OPENAI_API_KEY=sk-your-key-here\n")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a medical documentation AI assistant. Analyze doctor-patient 
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

SAMPLE_TRANSCRIPT = """Doctor: Good morning! How are you feeling today?
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
Patient: Okay, thank you doctor."""


# ── Core Agent ────────────────────────────────────────────────────────────────

def analyze_transcript(transcript: str) -> dict:
    """Send transcript to OpenAI and return structured summary as a dict."""

    print("\n⚕  Analyzing transcript with GPT-4o...\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Analyze this doctor-patient transcript:\n\n{transcript}"}
        ],
        temperature=0.2,       # low temperature = more consistent, factual output
        max_tokens=1500,
        response_format={"type": "json_object"}  # forces valid JSON output
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ── Display ───────────────────────────────────────────────────────────────────

def print_section(title: str, content):
    """Print a formatted section."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    if isinstance(content, list):
        if content:
            for item in content:
                print(f"  • {item}")
        else:
            print("  None noted")
    elif isinstance(content, str):
        print(f"  {content}")


def print_summary(summary: dict):
    """Pretty-print the structured summary to the terminal."""
    print("\n" + "═" * 60)
    print("  MEDSCRIBE AI — CONSULTATION SUMMARY")
    print("═" * 60)

    print_section("VISIT SUMMARY", summary.get("visit_summary", "N/A"))
    print_section("CHIEF COMPLAINT", summary.get("patient_chief_complaint", "N/A"))
    print_section("ASSESSMENT / DIAGNOSIS", summary.get("assessment", "N/A"))

    # Symptoms with details
    print(f"\n{'─' * 60}")
    print("  SYMPTOMS")
    print(f"{'─' * 60}")
    symptoms = summary.get("symptoms", [])
    for s in symptoms:
        print(f"  • {s}")
    details = summary.get("symptom_details", {})
    if details:
        if details.get("duration"):
            print(f"\n  Duration   : {details['duration']}")
        if details.get("severity"):
            print(f"  Severity   : {details['severity']}")
        if details.get("location"):
            print(f"  Location   : {details['location']}")
        if details.get("aggravating_factors"):
            print(f"  Worsened by: {', '.join(details['aggravating_factors'])}")
        if details.get("relieving_factors"):
            print(f"  Relieved by: {', '.join(details['relieving_factors'])}")

    print_section("CURRENT MEDICATIONS", summary.get("current_medications", []))
    print_section("MEDICAL HISTORY", summary.get("medical_history", []))
    print_section("TREATMENT PLAN", summary.get("plan", []))
    print_section("NEW PRESCRIPTIONS", summary.get("prescriptions", []))
    print_section("FOLLOW-UP INSTRUCTIONS", summary.get("follow_up", "N/A"))

    red_flags = summary.get("red_flags", [])
    if red_flags:
        print(f"\n{'─' * 60}")
        print("  ⚠  RED FLAGS — SEEK EMERGENCY CARE IF:")
        print(f"{'─' * 60}")
        for flag in red_flags:
            print(f"  ! {flag}")

    print("\n" + "═" * 60)
    print("  NOTE: AI-generated summary. Always verify with treating physician.")
    print("═" * 60 + "\n")


# ── File Output ───────────────────────────────────────────────────────────────

def save_to_file(summary: dict, output_path: str = None):
    """Save the summary as a markdown file."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"consultation_summary_{timestamp}.md"

    lines = [
        "# Consultation Summary\n",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        f"\n## Visit Summary\n{summary.get('visit_summary', 'N/A')}\n",
        f"\n## Chief Complaint\n{summary.get('patient_chief_complaint', 'N/A')}\n",
        f"\n## Assessment / Diagnosis\n{summary.get('assessment', 'N/A')}\n",
        "\n## Symptoms\n" + "\n".join(f"- {s}" for s in summary.get("symptoms", [])),
    ]

    details = summary.get("symptom_details", {})
    if details:
        lines.append("\n### Symptom Details")
        for k, v in details.items():
            if v:
                val = ", ".join(v) if isinstance(v, list) else v
                lines.append(f"- **{k.replace('_', ' ').title()}**: {val}")

    lines += [
        "\n## Current Medications\n" + "\n".join(f"- {m}" for m in summary.get("current_medications", [])),
        "\n## Medical History\n"     + "\n".join(f"- {h}" for h in summary.get("medical_history", [])),
        "\n## Treatment Plan\n"      + "\n".join(f"- {p}" for p in summary.get("plan", [])),
        "\n## New Prescriptions\n"   + "\n".join(f"- {rx}" for rx in summary.get("prescriptions", [])),
        f"\n## Follow-Up\n{summary.get('follow_up', 'N/A')}",
        "\n## ⚠ Red Flags\n"         + "\n".join(f"- {r}" for r in summary.get("red_flags", [])),
        "\n\n---\n_AI-generated summary. Always verify with the treating physician._"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Summary saved to: {output_path}")
    return output_path


# ── Entry Point ───────────────────────────────────────────────────────────────

def get_transcript() -> str:
    """Get transcript from CLI arg (.txt/.docx), --input flag, or use sample."""
    args = sys.argv[1:]

    # Read from file
    if args and args[0] != "--input":
        filepath = args[0]
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".docx":
            if Document is None:
                print("Error: DOCX support requires 'python-docx'.")
                print("Install it with: pip install python-docx")
                sys.exit(1)

            doc = Document(filepath)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # Manual paste mode
    if args and args[0] == "--input":
        print("Paste your transcript below. Press Enter twice when done:\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines[:-1])

    # Default: use sample
    print("No transcript provided — using sample transcript.")
    return SAMPLE_TRANSCRIPT


def main():
    transcript = get_transcript()

    if not transcript.strip():
        print("Error: Empty transcript.")
        sys.exit(1)

    summary = analyze_transcript(transcript)
    print_summary(summary)
    save_to_file(summary)


if __name__ == "__main__":
    main()
