"""The MedScribe agent loop.

The model decides which tools to call (patient history, drug interactions, allergy checks,
guideline retrieval, ICD-10 lookup, transcript search) and finishes by calling submit_note.
submit_note is gated by the grounding verifier: a note with unsupported claims is rejected with
specific feedback and the model must revise it. After the revision budget is spent, anything
still unsupported is pruned, so an ungrounded claim never reaches the final note.

Safety checks (interactions, allergy conflicts, documentation gaps) are re-run in code on the
final note, so patient safety does not depend on the model choosing to call those tools.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import config, tools
from .context import ContextManager, TranscriptIndex, split_turns
from .llm import LLM, MissingAPIKey, ToolCall
from .schema import NOTE_JSON_SCHEMA, normalize_note
from .trace import Tracer, summarize, write_trajectory
from .verifier import Report, Verifier, prune

SYSTEM_PROMPT = """You are MedScribe, a clinical documentation agent. You turn a doctor-patient consultation into a structured visit note that a clinician will review.

Rules
- Use only facts stated in the transcript or returned by get_patient_history. Never infer, guess or add clinical content. If something was not said, leave it empty.
- Assessment, plan, prescriptions, follow_up and red_flags come from what the DOCTOR said. A patient's guess or request is not a prescription.
- Do not list a symptom the patient denied.
- Only write a dose if someone said it.
- Every item in the note needs an entry in "evidence" with a verbatim quote copied exactly from the transcript (source "transcript") or from the history text (source "history").
- Take ICD-10 codes only from lookup_icd10 results and guideline ids only from search_guidelines results.

Workflow
1. If a patient id is given, call get_patient_history first.
2. Read the transcript. Use search_transcript when you need the exact wording of something.
3. Safety: call lookup_drug_interactions with every medication (current, from history, and newly prescribed) and check_allergy_conflicts with allergies from the transcript and history.
4. If the doctor stated an assessment, call lookup_icd10 and search_guidelines for it.
5. Draft the note and call flag_missing_info on it. Do not invent content to fill a gap. Gaps are reported to the clinician as they are.
6. Call submit_note. If it is rejected, fix exactly the listed issues (remove the unsupported item or correct the quote) and submit again.
Finish only with a successful submit_note."""

SESSION_TOOLS = [
    {"type": "function", "function": {
        "name": "search_transcript",
        "description": "Search this consultation's transcript. Returns matching turns with the turns around them, "
                       "so you can copy exact wording for evidence quotes.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "submit_note",
        "description": "Submit the final visit note. It is checked against the transcript. If it is rejected, "
                       "you get a list of issues to fix.",
        "parameters": NOTE_JSON_SCHEMA}},
]


@dataclass
class Session:
    transcript: str
    patient_id: str | None
    index: TranscriptIndex
    history: dict | None = None
    history_text: str = ""
    seen_icd: set[str] = field(default_factory=set)
    seen_refs: set[str] = field(default_factory=set)
    revisions: int = 0
    accepted: dict | None = None
    last_report: Report | None = None
    final_report: Report | None = None
    pruned: list[dict] = field(default_factory=list)


@dataclass
class AgentResult:
    status: str  # accepted | accepted_with_pruning | failed
    note: dict
    verification: dict
    alerts: list[dict]
    documentation_gaps: list[dict]
    condensed: bool
    trajectory: dict
    trace: list[dict]
    error: str | None = None

    def to_dict(self, include_trace: bool = False) -> dict:
        out = {"status": self.status, "note": self.note, "verification": self.verification,
               "alerts": self.alerts, "documentation_gaps": self.documentation_gaps,
               "context_condensed": self.condensed, "run": self.trajectory}
        if self.error:
            out["error"] = self.error
        if include_trace:
            out["trace"] = self.trace
        return out


def _history_text(history: dict) -> str:
    lines = ["Stored history for patient " + str(history.get("patient_id", ""))]
    for key in ("allergies", "conditions", "medications"):
        for item in history.get(key, []):
            lines.append(f"{key}: {item}")
    for v in history.get("recent_visits", []):
        lines.append(f"visit {v.get('visit_date', '')}: {v.get('assessment', '')}. {v.get('summary', '')}")
    return "\n".join(lines)


class ConsultationAgent:
    def __init__(self, llm: LLM, backend: tools.ToolBackend | None = None, max_steps: int | None = None,
                 max_revisions: int | None = None, use_critic: bool = True):
        self.llm = llm
        self.backend = backend or tools.LocalBackend()
        self.max_steps = max_steps or config.MAX_STEPS
        self.max_revisions = config.MAX_REVISIONS if max_revisions is None else max_revisions
        self.verifier = Verifier(tools.get_kb(), llm if use_critic else None)
        self.context = ContextManager(config.CONTEXT_BUDGET_TOKENS)

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, transcript: str, patient_id: str | None = None, save: bool = False,
            visit_date: str | None = None) -> AgentResult:
        tracer = Tracer()
        sess = Session(transcript=transcript, patient_id=patient_id,
                       index=TranscriptIndex(split_turns(transcript)))
        prompt_text, condensed = self.context.prepare(transcript, self.llm)
        if condensed:
            tracer.add("context", action="condensed", chars=len(transcript))

        tool_specs = [s for s in self.backend.specs()
                      if patient_id or s["function"]["name"] != "get_patient_history"] + SESSION_TOOLS
        header = (f"Patient id: {patient_id}" if patient_id
                  else "No patient id was given, so there is no stored history.")
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{header}\n\nTranscript:\n{prompt_text}"},
        ]

        error = None
        nudges = 0
        for step in range(self.max_steps):
            t0 = time.perf_counter()
            try:
                turn = self.llm.chat(messages, tools=tool_specs)
            except MissingAPIKey:
                raise  # a configuration problem, not a failed run
            except Exception as e:  # network, auth, rate limit after retries
                error = f"{type(e).__name__}: {e}"
                tracer.add("llm", step=step, ok=False, error=error)
                break
            tracer.add("llm", step=step, ok=True, ms=round((time.perf_counter() - t0) * 1000),
                       usage=turn.usage, tool_calls=[tc.name for tc in turn.tool_calls])
            messages.append(turn.as_message())

            if not turn.tool_calls:
                nudges += 1
                if nudges > 2:
                    error = "The model stopped without submitting a note."
                    break
                messages.append({"role": "user", "content": "Call submit_note with the final note."})
                continue

            for tc in turn.tool_calls:
                result = self._dispatch(sess, tc, step, tracer)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            if sess.accepted is not None:
                break
        else:
            error = f"No accepted note after {self.max_steps} steps."

        return self._finish(sess, tracer, error, save, visit_date)

    # ── tool dispatch ────────────────────────────────────────────────────────

    def _dispatch(self, sess: Session, tc: ToolCall, step: int, tracer: Tracer) -> dict:
        t0 = time.perf_counter()
        ok = True
        if tc.parse_error:
            result: dict[str, Any] = {"error": tc.parse_error}
            ok = False
        else:
            try:
                if tc.name == "search_transcript":
                    k = max(1, min(int(tc.arguments.get("k", 3)), 5))
                    result = {"results": sess.index.search(str(tc.arguments["query"]), k)}
                elif tc.name == "submit_note":
                    result = self._submit(sess, tc.arguments, tracer, step)
                else:
                    result = self.backend.call(tc.name, tc.arguments)
                    self._observe(sess, tc.name, result)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
                ok = False
        if tc.name != "submit_note":
            tracer.add("tool", step=step, name=tc.name, args=_short(tc.arguments), ok=ok,
                       ms=round((time.perf_counter() - t0) * 1000),
                       result=_short(result))
        return result

    def _observe(self, sess: Session, name: str, result: dict) -> None:
        """Remember what tools returned, so the verifier can check citations against it."""
        if name == "get_patient_history" and result.get("found"):
            sess.history = result
            sess.history_text = _history_text(result)
        elif name == "lookup_icd10":
            sess.seen_icd |= {r["code"] for r in result.get("results", [])}
        elif name == "search_guidelines":
            sess.seen_refs |= {r["id"] for r in result.get("results", [])}

    def _submit(self, sess: Session, args: dict, tracer: Tracer, step: int) -> dict:
        note = normalize_note(args)
        report = self.verifier.verify(note, sess.transcript, sess.history_text, sess.seen_icd, sess.seen_refs)
        sess.last_report = report  # before any pruning, so the reward reflects what the model produced
        tracer.add("verify", step=step, ok=report.ok, revision=sess.revisions, n_items=report.n_items,
                   blocking=len(report.blocking), warnings=len(report.warnings),
                   grounded_ratio=round(report.grounded_ratio, 3))

        if report.blocking and sess.revisions < self.max_revisions:
            sess.revisions += 1
            return {
                "status": "rejected",
                "revision": f"{sess.revisions} of {self.max_revisions}",
                "issues": [i.to_dict() for i in report.blocking],
                "instruction": "Fix these issues and call submit_note again. Remove any item you cannot back "
                               "with a verbatim quote.",
            }

        if report.blocking:
            note, removed = prune(note, report)
            sess.pruned = removed
            report = self.verifier.verify(note, sess.transcript, sess.history_text, sess.seen_icd, sess.seen_refs)
        sess.accepted = note
        sess.final_report = report
        return {"status": "accepted", "pruned": len(sess.pruned)}

    # ── wrap-up ──────────────────────────────────────────────────────────────

    def _auto(self, tracer: Tracer, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            result = self.backend.call(name, args)
            ok = True
        except Exception as e:
            result, ok = {"error": f"{type(e).__name__}: {e}"}, False
        tracer.add("tool", auto=True, name=name, ok=ok, ms=round((time.perf_counter() - t0) * 1000))
        return result

    def _post_checks(self, sess: Session, note: dict, tracer: Tracer) -> tuple[list[dict], list[dict]]:
        """Deterministic safety pass on the final note. Does not depend on what the model chose to call."""
        history = sess.history
        if history is None and sess.patient_id:
            fetched = self._auto(tracer, "get_patient_history", {"patient_id": sess.patient_id})
            if fetched.get("found"):
                history = fetched
        hist_meds = (history or {}).get("medications", [])
        hist_allergies = (history or {}).get("allergies", [])

        drugs = note["current_medications"] + note["prescriptions"] + note["plan"] + hist_meds
        alerts: list[dict] = []
        if drugs:
            alerts += self._auto(tracer, "lookup_drug_interactions", {"drugs": drugs}).get("interactions", [])
        allergies = list(dict.fromkeys(note["allergies"] + hist_allergies))
        if allergies and drugs:
            alerts += self._auto(tracer, "check_allergy_conflicts",
                                 {"allergies": allergies, "drugs": drugs}).get("conflicts", [])
        seen, unique = set(), []
        for a in alerts:
            if a["key"] not in seen:
                seen.add(a["key"])
                unique.append(a)
        gaps_note = dict(note, allergies=allergies)
        gaps = self._auto(tracer, "flag_missing_info", {"note": gaps_note}).get("missing", [])
        return unique, gaps

    def _finish(self, sess: Session, tracer: Tracer, error: str | None, save: bool,
                visit_date: str | None) -> AgentResult:
        if sess.accepted is None:
            note = normalize_note({})
            status, alerts, gaps = "failed", [], []
            verification = {"ok": False, "n_items": 0, "grounded_ratio": 0.0, "issues": []}
            grounded = 0.0
        else:
            note = sess.accepted
            status = "accepted_with_pruning" if sess.pruned else "accepted"
            alerts, gaps = self._post_checks(sess, note, tracer)
            verification = sess.final_report.to_dict() if sess.final_report else {}
            verification["pruned"] = sess.pruned
            grounded = sess.last_report.grounded_ratio if sess.last_report else 1.0
            if save and sess.patient_id:
                tools.get_store().record_visit(sess.patient_id, note)
                tracer.add("memory", action="record_visit", patient_id=sess.patient_id)
        note["safety_alerts"] = alerts
        note["documentation_gaps"] = gaps
        trajectory = summarize(tracer.events, status, grounded, sess.revisions)
        trajectory["elapsed_ms"] = tracer.elapsed_ms
        write_trajectory({"patient_id": sess.patient_id, "run": trajectory, "events": tracer.events,
                          "final_note": note})
        return AgentResult(status=status, note=note, verification=verification, alerts=alerts,
                           documentation_gaps=gaps, condensed=any(e["type"] == "context" for e in tracer.events),
                           trajectory=trajectory, trace=tracer.events, error=error)


def _short(obj: Any, limit: int = 400) -> Any:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return obj if len(text) <= limit else text[:limit] + "..."
