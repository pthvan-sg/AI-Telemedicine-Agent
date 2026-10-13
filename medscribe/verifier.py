"""Grounding verifier.

Every item in a draft note must carry a verbatim quote from the transcript (or from stored
history). The checks are deterministic first, so a model cannot grade its own homework:

  no_evidence          item has no evidence entry
  quote_not_found      the quote is not in the source text
  wrong_speaker        doctor-only field (plan, prescriptions...) backed only by patient speech
  drug_not_in_source   a drug named in the item never appears in the source
  bad_number           a dose or number in the item is not in the source
  unknown_code         ICD-10 code that lookup_icd10 never returned
  unknown_reference    guideline id that search_guidelines never returned
  summary_unsupported  visit_summary mentions a drug or number that is not in the transcript
  weak_support         the quote shares almost no words with the item (sent to an optional LLM critic)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import DETAIL_LISTS, DETAIL_SCALARS
from .text import normalize, tokens

DOCTOR_FIELDS = {"assessment", "plan", "prescriptions", "follow_up", "red_flags"}
DRUG_CHECK_FIELDS = {"current_medications", "prescriptions", "plan", "allergies"}
DOSE_CHECK_FIELDS = {"current_medications", "prescriptions"}

NUMBER_WORDS = {
    "0": ["zero"], "1": ["one", "a", "an", "once"], "2": ["two", "twice", "couple"], "3": ["three"],
    "4": ["four"], "5": ["five"], "6": ["six"], "7": ["seven", "week"], "8": ["eight"], "9": ["nine"],
    "10": ["ten"], "11": ["eleven"], "12": ["twelve"], "14": ["fourteen", "fortnight"], "15": ["fifteen"],
    "20": ["twenty"], "30": ["thirty"], "40": ["forty"], "50": ["fifty"], "60": ["sixty"],
    "100": ["hundred"],
}


@dataclass
class Issue:
    field: str
    item: str
    kind: str
    message: str
    severity: str = "blocking"  # or "warning"

    def to_dict(self) -> dict:
        return {"field": self.field, "item": self.item, "kind": self.kind,
                "severity": self.severity, "message": self.message}


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    n_items: int = 0

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def grounded_ratio(self) -> float:
        if not self.n_items:
            return 1.0
        bad = {(i.field, normalize(i.item)) for i in self.blocking if i.field != "visit_summary"}
        return max(0.0, 1 - len(bad) / self.n_items)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "n_items": self.n_items, "grounded_ratio": round(self.grounded_ratio, 3),
                "issues": [i.to_dict() for i in self.issues]}


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_items(note: dict):
    """Yield (field, item) for every claim in the note that needs evidence."""
    if note.get("patient_chief_complaint"):
        yield "patient_chief_complaint", note["patient_chief_complaint"]
    for f in ("symptoms", "allergies", "medical_history", "current_medications", "plan", "prescriptions",
              "red_flags"):
        for item in note.get(f, []):
            yield f, item
    for f in ("assessment", "follow_up"):
        if note.get(f):
            yield f, note[f]
    details = note.get("symptom_details", {})
    for f in DETAIL_SCALARS:
        if details.get(f):
            yield f"symptom_details.{f}", details[f]
    for f in DETAIL_LISTS:
        for item in details.get(f, []):
            yield f"symptom_details.{f}", item


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _number_in(num: str, text_norm: str) -> bool:
    toks = set(text_norm.split())
    if num in toks:
        return True
    return any(w in toks for w in NUMBER_WORDS.get(num, []))


def _speaker(turn: str) -> str:
    m = re.match(r"^\s*([A-Za-z][A-Za-z .]{0,24}):", turn)
    if not m:
        return "unknown"
    label = m.group(1).strip().lower()
    if label.startswith(("patient", "pt")):
        return "patient"
    if label.startswith(("doctor", "dr", "physician", "clinician", "provider", "gp")):
        return "doctor"
    return "unknown"


class Source:
    """Normalised text of the transcript plus turn boundaries, used to locate quotes."""

    def __init__(self, text: str):
        self.text = text
        self.turns = [t for t in (line.strip() for line in text.splitlines()) if t]
        norm_turns = [normalize(t) for t in self.turns]
        self.roles = [_speaker(t) for t in self.turns]
        self.spans: list[tuple[int, int]] = []
        pos = 0
        parts = []
        for nt in norm_turns:
            self.spans.append((pos, pos + len(nt)))
            parts.append(nt)
            pos += len(nt) + 1
        self.norm = " ".join(parts) if parts else normalize(text)
        self.has_roles = any(r != "unknown" for r in self.roles)

    def locate(self, quote: str) -> tuple[int, int] | None:
        """Find a quote (pieces joined by '...' must appear in order). Returns (start, end) offsets."""
        pieces = [normalize(p) for p in re.split(r"\.{3}|…", quote)]
        pieces = [p for p in pieces if p]
        if not pieces or sum(len(p) for p in pieces) < 3:
            return None
        start = end = None
        cursor = 0
        for p in pieces:
            i = self.norm.find(p, cursor)
            if i < 0:
                return None
            if start is None:
                start = i
            end = i + len(p)
            cursor = end
        return start, end

    def roles_in(self, start: int, end: int) -> set[str]:
        return {self.roles[i] for i, (a, b) in enumerate(self.spans) if a < end and b > start}


# ── Verifier ─────────────────────────────────────────────────────────────────

CRITIC_PROMPT = (
    "You check clinical documentation. For each numbered pair, decide whether the quote supports the note "
    "item. A paraphrase or synonym is supported (for example 'light bothers me' supports 'photophobia'). "
    "A different fact, a denied symptom, or a statement by the wrong person is not supported. "
    'Reply with JSON only: {"results": [{"i": 0, "supported": true}]}'
)


class Verifier:
    def __init__(self, kb, llm=None, max_critic_items: int = 12):
        self.kb = kb
        self.llm = llm
        self.max_critic_items = max_critic_items

    def verify(self, note: dict, transcript: str, history_text: str = "",
               seen_icd: set[str] | None = None, seen_refs: set[str] | None = None) -> Report:
        src = {"transcript": Source(transcript), "history": Source(history_text)}
        evidence: dict[str, list[dict]] = {}
        for ev in note.get("evidence", []):
            evidence.setdefault(normalize(ev.get("item", "")), []).append(ev)

        report = Report()
        weak: list[tuple[str, str, str]] = []

        for fld, item in iter_items(note):
            report.n_items += 1
            base = fld.split(".")[0]
            entries = evidence.get(normalize(item), [])
            if not entries:
                report.issues.append(Issue(fld, item, "no_evidence",
                                           "No evidence entry. Add a verbatim quote or remove the item."))
                continue
            self._check_item(report, weak, fld, base, item, entries, src)

        self._check_codes(report, note, seen_icd, seen_refs)
        self._check_summary(report, note, transcript, history_text)
        self._run_critic(report, weak)
        return report

    # one item against its evidence entries
    def _check_item(self, report, weak, fld, base, item, entries, src):
        located = None
        for ev in entries:
            which = "history" if ev.get("source") == "history" else "transcript"
            span = src[which].locate(ev.get("quote", ""))
            if span:
                located = (ev, which, span)
                break
        if not located:
            report.issues.append(Issue(fld, item, "quote_not_found",
                                       "The quote was not found verbatim in the source. Copy it exactly "
                                       "(search_transcript can return the exact text)."))
            return
        ev, which, (start, end) = located
        source = src[which]
        quote_norm = source.norm[start:end]

        if base in DOCTOR_FIELDS and which == "transcript" and source.has_roles:
            roles = source.roles_in(start, end)
            if roles and roles <= {"patient"}:
                report.issues.append(Issue(fld, item, "wrong_speaker",
                                           f"'{base}' must come from what the doctor said, but this quote is "
                                           "patient speech."))
                return

        if base in DRUG_CHECK_FIELDS:
            source_drugs = set(self.kb.extract(source.text))
            missing = [d for d in self.kb.extract(item) if d not in source_drugs]
            if missing:
                report.issues.append(Issue(fld, item, "drug_not_in_source",
                                           f"{', '.join(missing)} is never mentioned in the source."))
                return

        nums = _numbers(item)
        if nums:
            bad = [n for n in nums if not _number_in(n, quote_norm) and not _number_in(n, source.norm)]
            if bad:
                sev = "blocking" if base in DOSE_CHECK_FIELDS else "warning"
                report.issues.append(Issue(fld, item, "bad_number",
                                           f"Number(s) {', '.join(bad)} do not appear in the source. "
                                           "Only record a dose or figure that was said.", sev))
                if sev == "blocking":
                    return

        item_tokens = set(tokens(item))
        if item_tokens and not self.kb.extract(item):
            coverage = len(item_tokens & set(tokens(quote_norm))) / len(item_tokens)
            if coverage < 0.34:
                weak.append((fld, item, ev["quote"]))

    def _check_codes(self, report, note, seen_icd, seen_refs):
        seen_icd = seen_icd or set()
        seen_refs = seen_refs or set()
        for entry in note.get("suggested_icd10", []):
            report.n_items += 1
            if entry["code"] not in seen_icd:
                report.issues.append(Issue("suggested_icd10", entry["code"], "unknown_code",
                                           "This code was not returned by lookup_icd10. Call the tool or remove it."))
        for ref in note.get("guideline_references", []):
            report.n_items += 1
            if ref not in seen_refs:
                report.issues.append(Issue("guideline_references", ref, "unknown_reference",
                                           "This id was not returned by search_guidelines in this run."))

    def _check_summary(self, report, note, transcript, history_text):
        summary = note.get("visit_summary", "")
        if not summary:
            return
        pool = f"{transcript}\n{history_text}"
        pool_norm = normalize(pool)
        pool_drugs = set(self.kb.extract(pool))
        problems = [d for d in self.kb.extract(summary) if d not in pool_drugs]
        problems += [n for n in _numbers(summary) if not _number_in(n, pool_norm)]
        if problems:
            report.issues.append(Issue("visit_summary", summary[:80], "summary_unsupported",
                                       f"The summary mentions {', '.join(problems)}, which is not in the "
                                       "transcript. Rewrite it using only stated facts."))

    def _run_critic(self, report, weak):
        if not weak:
            return
        if self.llm is None:
            for fld, item, _ in weak:
                report.issues.append(Issue(fld, item, "weak_support",
                                           "The quote shares few words with the item.", "warning"))
            return
        batch = weak[: self.max_critic_items]
        pairs = "\n".join(f'{i}. item: "{it}" | quote: "{q}"' for i, (_, it, q) in enumerate(batch))
        verdicts: dict[int, bool] = {}
        try:
            turn = self.llm.chat([{"role": "system", "content": CRITIC_PROMPT},
                                  {"role": "user", "content": pairs}], tools=None, max_tokens=600, json_mode=True)
            data = json.loads(turn.content or "{}")
            verdicts = {int(r["i"]): bool(r["supported"]) for r in data.get("results", [])}
        except Exception:
            verdicts = {}
        for i, (fld, item, _) in enumerate(weak):
            if i in verdicts and not verdicts[i]:
                report.issues.append(Issue(fld, item, "quote_does_not_support",
                                           "A reviewer judged that the quote does not support this item."))
            elif i not in verdicts:
                report.issues.append(Issue(fld, item, "weak_support",
                                           "The quote shares few words with the item.", "warning"))


# ── Pruning and feedback ─────────────────────────────────────────────────────

def prune(note: dict, report: Report) -> tuple[dict, list[dict]]:
    """Remove every item that still has a blocking issue. Returns (clean note, removed items)."""
    bad = {(i.field, normalize(i.item)) for i in report.blocking}
    removed: list[dict] = []
    out: dict[str, Any] = json.loads(json.dumps(note))

    def keep(fld: str, item: str) -> bool:
        if (fld, normalize(item)) in bad:
            removed.append({"field": fld, "item": item})
            return False
        return True

    for f in ("symptoms", "allergies", "medical_history", "current_medications", "plan", "prescriptions",
              "red_flags"):
        out[f] = [x for x in out.get(f, []) if keep(f, x)]
    for f in ("patient_chief_complaint", "assessment", "follow_up"):
        if out.get(f) and not keep(f, out[f]):
            out[f] = ""
    d = out.get("symptom_details", {})
    for f in DETAIL_SCALARS:
        if d.get(f) and not keep(f"symptom_details.{f}", d[f]):
            d[f] = ""
    for f in DETAIL_LISTS:
        d[f] = [x for x in d.get(f, []) if keep(f"symptom_details.{f}", x)]
    out["suggested_icd10"] = [e for e in out.get("suggested_icd10", []) if keep("suggested_icd10", e["code"])]
    out["guideline_references"] = [r for r in out.get("guideline_references", [])
                                   if keep("guideline_references", r)]
    gone = {normalize(r["item"]) for r in removed}
    out["evidence"] = [e for e in out.get("evidence", []) if normalize(e.get("item", "")) not in gone]
    return out, removed
