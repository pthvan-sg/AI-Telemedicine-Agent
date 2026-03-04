"""
MedScribe AI - Flask API
Exposes doctor-patient transcript analysis as HTTP endpoints for Kubernetes deployment.

Endpoints:
    GET  /health          → health check (used by Kubernetes liveness probe)
    GET  /               → API info
    POST /analyze         → analyze a transcript, returns structured JSON summary
    POST /analyze/sample  → analyze the built-in sample transcript
"""

import json
import os
from datetime import datetime

from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file or Kubernetes secret.")

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


# ── Core Logic ────────────────────────────────────────────────────────────────

def analyze_transcript(transcript: str) -> dict:
    """Send transcript to OpenAI and return structured summary as a dict."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Analyze this doctor-patient transcript:\n\n{transcript}"}
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """API info and available endpoints."""
    return jsonify({
        "service": "MedScribe AI",
        "description": "Doctor-patient transcript analyzer",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health":         "Health check",
            "POST /analyze":        "Analyze a transcript. Body: { 'transcript': '...' }",
            "POST /analyze/sample": "Analyze the built-in sample transcript"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Kubernetes liveness/readiness probe.
    Returns 200 OK if the service is running.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze a doctor-patient transcript.

    Request body (JSON):
        { "transcript": "Doctor: ...\nPatient: ..." }

    Returns:
        Structured clinical summary as JSON.
    """
    data = request.get_json()

    # Validate input
    if not data or "transcript" not in data:
        return jsonify({
            "error": "Missing 'transcript' field in request body.",
            "example": {"transcript": "Doctor: How are you?\nPatient: I have a headache..."}
        }), 400

    transcript = data["transcript"].strip()
    if not transcript:
        return jsonify({"error": "Transcript cannot be empty."}), 400

    try:
        summary = analyze_transcript(transcript)
        return jsonify({
            "status": "success",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/analyze/sample", methods=["POST"])
def analyze_sample():
    """
    Analyze the built-in sample transcript.
    Useful for testing the deployment without needing real data.
    """
    try:
        summary = analyze_transcript(SAMPLE_TRANSCRIPT)
        return jsonify({
            "status": "success",
            "note": "This used the built-in sample transcript.",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For local dev only. In Docker/Kubernetes, gunicorn runs this instead.
    app.run(host="0.0.0.0", port=5000, debug=False)
