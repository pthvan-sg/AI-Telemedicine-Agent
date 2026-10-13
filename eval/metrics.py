"""Scoring for the eval harness. No LLM involved: everything is checked against hand-written gold."""

import re
from dataclasses import dataclass, field

from medscribe.text import normalize, tokens

SCORED_FIELDS = ["symptoms", "current_medications", "prescriptions", "allergies", "red_flags", "medical_history"]


def item_matches(item: str, aliases: list[str]) -> bool:
    """An item matches a gold group if any alias is a substring of it, or all alias words appear in it."""
    n_item = normalize(item)
    t_item = set(tokens(item))
    for alias in aliases:
        n_alias = normalize(alias)
        if n_alias and n_alias in n_item:
            return True
        t_alias = set(tokens(alias))
        if t_alias and t_alias <= t_item:
            return True
    return False


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_field(pred: list[str], gold_groups: list[list[str]]) -> Counts:
    matched: set[int] = set()
    c = Counts()
    for item in pred:
        hit = next((i for i, g in enumerate(gold_groups) if item_matches(item, g)), None)
        if hit is None:
            c.fp += 1
        elif hit not in matched:
            matched.add(hit)
            c.tp += 1
        # a second item matching an already-matched group is a duplicate: neither right nor wrong
    c.fn = len(gold_groups) - len(matched)
    return c


def check_traps(note: dict, traps: dict) -> tuple[int, list[str]]:
    """Return (number of traps checked, list of violations). Each violation is a hallucination."""
    checked, violations = 0, []
    if traps.get("allergies_must_be_empty"):
        checked += 1
        if note.get("allergies"):
            violations.append(f"allergies should be empty, got {note['allergies']}")
    if traps.get("prescriptions_no_dose"):
        checked += 1
        bad = [p for p in note.get("prescriptions", []) if re.search(r"\d", p)]
        if bad:
            violations.append(f"dose invented for {bad}")
    if traps.get("assessment_must_be_empty"):
        checked += 1
        if note.get("assessment"):
            violations.append(f"assessment should be empty, got '{note['assessment']}'")
    for fld, terms in (traps.get("forbidden") or {}).items():
        for term in terms:
            checked += 1
            hits = [i for i in note.get(fld, []) if normalize(term) in normalize(i)]
            if hits:
                violations.append(f"{fld} contains forbidden '{term}': {hits}")
    return checked, violations


def alert_keys(alerts: list[dict]) -> set[str]:
    return {a["key"] for a in alerts}


@dataclass
class CaseScore:
    case_id: str
    has_history: bool
    fields: dict[str, Counts] = field(default_factory=dict)
    traps_checked: int = 0
    violations: list[str] = field(default_factory=list)
    alerts: Counts = field(default_factory=Counts)


def score_case(case: dict, note: dict, alerts: list[dict]) -> CaseScore:
    s = CaseScore(case["id"], bool(case.get("patient_id")))
    for fld in SCORED_FIELDS:
        if fld in case["gold"]:
            s.fields[fld] = score_field(note.get(fld, []), case["gold"][fld])
    s.traps_checked, s.violations = check_traps(note, case.get("traps", {}))
    expected, got = set(case.get("expected_alerts", [])), alert_keys(alerts)
    s.alerts = Counts(tp=len(expected & got), fp=len(got - expected), fn=len(expected - got))
    return s


def aggregate(scores: list[CaseScore]) -> dict:
    """Micro-average over cases. Returns plain numbers ready for a table."""
    per_field: dict[str, Counts] = {}
    overall, alerts = Counts(), Counts()
    checked = violated = 0
    for s in scores:
        for fld, c in s.fields.items():
            per_field.setdefault(fld, Counts()).add(c)
            overall.add(c)
        alerts.add(s.alerts)
        checked += s.traps_checked
        violated += len(s.violations)
    r = lambda x: round(x, 3)  # noqa: E731
    return {
        "cases": len(scores),
        "extraction": {"precision": r(overall.precision), "recall": r(overall.recall), "f1": r(overall.f1)},
        "per_field_f1": {f: r(c.f1) for f, c in sorted(per_field.items())},
        "hallucination": {"traps_checked": checked, "violations": violated,
                          "violation_rate": r(violated / checked) if checked else 0.0},
        "safety_alerts": {"expected": alerts.tp + alerts.fn, "recall": r(alerts.recall), "precision": r(alerts.precision)},
    }
