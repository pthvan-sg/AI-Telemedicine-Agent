"""
MedScribe AI - Flask API

Endpoints:
    GET  /                    API info
    GET  /health              liveness probe (does not need an API key)
    GET  /ready               readiness probe (503 until an API key is configured)
    POST /analyze             run the agent on a transcript
    POST /analyze/baseline    run the original single-call summarizer, for comparison
    POST /analyze/sample      run the agent on the built-in sample transcript
    GET  /patients/<id>       stored history for a synthetic patient id

POST /analyze body:
    {"transcript": "...", "patient_id": "P-1001", "save": false, "include_trace": false}
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from medscribe import __version__, config, tools
from medscribe.llm import MissingAPIKey, OpenAIChat
from medscribe.samples import SAMPLE_TRANSCRIPT

load_dotenv()
log = logging.getLogger("medscribe")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(llm_factory=None, backend_factory=None) -> Flask:
    """Build the app. Tests pass a fake llm_factory; production uses the OpenAI client."""
    app = Flask(__name__)
    state: dict = {}

    def get_llm():
        if "llm" not in state:
            state["llm"] = llm_factory() if llm_factory else OpenAIChat()
        return state["llm"]

    def get_backend():
        if "backend" not in state:
            if backend_factory:
                state["backend"] = backend_factory()
            else:
                from medscribe.mcp_backend import make_backend
                state["backend"] = make_backend()
        return state["backend"]

    @app.before_request
    def require_api_key():
        """If MEDSCRIBE_API_KEY is set, every route except the probes needs a matching X-API-Key header."""
        expected = os.getenv("MEDSCRIBE_API_KEY")
        if not expected or request.path in ("/", "/health", "/ready"):
            return None
        if request.headers.get("X-API-Key") != expected:
            return jsonify({"status": "error", "message": "Missing or invalid X-API-Key."}), 401
        return None

    def read_body(need_transcript: bool = True):
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, (jsonify({"status": "error", "message": "Send a JSON object as the request body."}), 400)
        if not need_transcript:
            return data, None
        transcript = data.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            return None, (jsonify({
                "status": "error", "message": "'transcript' must be a non-empty string.",
                "example": {"transcript": "Doctor: How are you?\nPatient: I have a headache..."}}), 400)
        if len(transcript) > config.MAX_TRANSCRIPT_CHARS:
            return None, (jsonify({"status": "error",
                                   "message": f"Transcript is over {config.MAX_TRANSCRIPT_CHARS} characters."}), 413)
        return data, None

    def run_agent(transcript: str, data: dict):
        from medscribe.agent import ConsultationAgent
        patient_id = data.get("patient_id")
        save = bool(data.get("save"))
        if patient_id is not None and not isinstance(patient_id, str):
            return jsonify({"status": "error", "message": "'patient_id' must be a string."}), 400
        if save and not patient_id:
            return jsonify({"status": "error", "message": "'save' needs a 'patient_id'."}), 400
        result = ConsultationAgent(get_llm(), get_backend()).run(transcript, patient_id, save=save)
        body = result.to_dict(include_trace=bool(data.get("include_trace")))
        status = 200 if result.status != "failed" else 502
        return jsonify({"generated_at": _now(), **body, "summary": result.note,
                        "status": "success" if status == 200 else "error"}), status

    def guarded(fn):
        try:
            return fn()
        except MissingAPIKey as e:
            return jsonify({"status": "error", "message": str(e)}), 503
        except Exception:
            log.exception("request failed")
            return jsonify({"status": "error", "message": "Internal error. See server logs."}), 500

    @app.get("/")
    def index():
        return jsonify({
            "service": "MedScribe AI", "version": __version__,
            "description": "Tool-using agent that turns doctor-patient transcripts into verified visit notes",
            "endpoints": {
                "GET  /health": "Liveness probe",
                "GET  /ready": "Readiness probe",
                "POST /analyze": "Agent. Body: {transcript, patient_id?, save?, include_trace?}",
                "POST /analyze/baseline": "Original single-call summarizer",
                "POST /analyze/sample": "Agent on the built-in sample",
                "GET  /patients/<id>": "Stored history for a synthetic patient",
            }})

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "timestamp": _now()}), 200

    @app.get("/ready")
    def ready():
        if llm_factory is None and not os.getenv("OPENAI_API_KEY"):
            return jsonify({"status": "not ready", "reason": "OPENAI_API_KEY is not set"}), 503
        return jsonify({"status": "ready"}), 200

    @app.post("/analyze")
    def analyze():
        data, err = read_body()
        if err:
            return err
        return guarded(lambda: run_agent(data["transcript"].strip(), data))

    @app.post("/analyze/sample")
    def analyze_sample():
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        return guarded(lambda: run_agent(SAMPLE_TRANSCRIPT, data))

    @app.post("/analyze/baseline")
    def analyze_baseline_route():
        data, err = read_body()
        if err:
            return err

        def go():
            from medscribe.baseline import analyze_baseline
            note = analyze_baseline(get_llm(), data["transcript"].strip())
            return jsonify({"status": "success", "mode": "baseline", "generated_at": _now(),
                            "summary": note}), 200
        return guarded(go)

    @app.get("/patients/<patient_id>")
    def patient(patient_id: str):
        record = tools.get_patient_history(patient_id)
        return (jsonify(record), 200) if record["found"] else (jsonify(record), 404)

    return app


app = create_app()

if __name__ == "__main__":
    # Local dev only. In Docker and Kubernetes gunicorn serves `app:app`.
    app.run(host="0.0.0.0", port=5000, debug=False)
