"""Domain knowledge: drug normalisation, interactions, allergy conflicts and ICD-10 lookup.

The data files are small demo sets. The interfaces are what matter: replace the JSON with
RxNorm, openFDA or a licensed database without touching the agent.
"""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .rag import BM25Index

SEVERITY_ORDER = {"contraindicated": 3, "major": 2, "moderate": 1}


class DrugKB:
    def __init__(self, data: dict):
        self.classes: dict[str, set[str]] = {c: set(v) for c, v in data["classes"].items()}
        self.aliases: dict[str, str] = {k.lower(): v for k, v in data.get("aliases", {}).items()}
        self.rules: list[dict] = data["rules"]
        self.allergy_classes: dict[str, str] = data.get("allergy_classes", {})
        names = set(data.get("other_drugs", []))
        for members in self.classes.values():
            names |= members
        self.drugs = {n.lower() for n in names}
        vocab = sorted(self.drugs | set(self.aliases), key=len, reverse=True)
        pattern = "|".join(re.escape(v) for v in vocab)
        self._regex = re.compile(rf"(?<![a-z0-9])({pattern})(?![a-z0-9])")

    @classmethod
    def load(cls, path: Path) -> "DrugKB":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def extract(self, text: str) -> list[str]:
        """Canonical generic names mentioned in free text, in order of appearance."""
        found: list[str] = []
        for m in self._regex.finditer(text.lower()):
            name = self.aliases.get(m.group(1), m.group(1))
            if name not in found:
                found.append(name)
        return found

    def extract_all(self, texts: Iterable[str]) -> list[str]:
        out: list[str] = []
        for t in texts:
            for d in self.extract(t):
                if d not in out:
                    out.append(d)
        return out

    def classes_of(self, drug: str) -> set[str]:
        return {c for c, members in self.classes.items() if drug in members}

    def _matches(self, selector: str, drug: str) -> bool:
        kind, _, value = selector.partition(":")
        if kind == "drug":
            return drug == value
        return value in self.classes_of(drug)

    def interactions(self, drugs: Iterable[str]) -> list[dict]:
        unique = sorted(set(drugs))
        out, seen = [], set()
        for d1, d2 in combinations(unique, 2):
            for rule in self.rules:
                hit = (self._matches(rule["a"], d1) and self._matches(rule["b"], d2)) or (
                    self._matches(rule["a"], d2) and self._matches(rule["b"], d1))
                if not hit:
                    continue
                key = f"interaction:{d1}+{d2}"
                if (key, rule["message"]) in seen:
                    continue
                seen.add((key, rule["message"]))
                out.append({
                    "kind": "interaction", "key": key, "drugs": [d1, d2],
                    "severity": rule["severity"], "message": rule["message"],
                    "source": "drug_kb.json",
                })
        out.sort(key=lambda a: -SEVERITY_ORDER.get(a["severity"], 0))
        return out

    def allergy_conflicts(self, allergies: Iterable[str], drugs: Iterable[str]) -> list[dict]:
        out, seen = [], set()
        drugs = list(drugs)
        for allergy in allergies:
            text = allergy.lower()
            blocked: set[str] = set()
            for name in self.extract(text):
                blocked.add(name)
                for cls in self.classes_of(name):
                    if cls == "penicillin":  # a penicillin-class allergy implies the whole class
                        blocked |= self.classes[cls]
            for word, cls in self.allergy_classes.items():
                if re.search(rf"(?<![a-z]){re.escape(word)}", text):
                    blocked |= self.classes.get(cls, set())
            for drug in drugs:
                if drug in blocked:
                    key = f"allergy:{allergy.lower()}->{drug}"
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "kind": "allergy", "key": key, "drug": drug, "allergy": allergy,
                        "severity": "major",
                        "message": f"Documented allergy to '{allergy}' conflicts with {drug}.",
                        "source": "drug_kb.json",
                    })
        return out


class ICDLookup:
    def __init__(self, codes: list[dict]):
        self.codes = codes
        docs = [{**c, "index_text": f"{c['description']} {c.get('keywords', '')}"} for c in codes]
        self.index = BM25Index(docs)

    @classmethod
    def load(cls, path: Path) -> "ICDLookup":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["codes"])

    def search(self, condition: str, k: int = 3) -> list[dict]:
        return [{"code": h["code"], "description": h["description"], "score": h["score"]}
                for h in self.index.search(condition, k)]
