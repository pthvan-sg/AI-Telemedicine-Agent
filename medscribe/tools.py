"""Domain tools and the backends that run them.

Tools here are stateless, so they can run in process (LocalBackend) or behind an MCP server
(MCPBackend in mcp_backend.py). Session-bound tools such as search_transcript and
submit_note live in agent.py because they need the current consultation.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol

from . import config
from .knowledge import DrugKB, ICDLookup
from .memory import PatientStore
from .rag import GuidelineIndex
from .schema import DETAIL_SCALARS


@lru_cache(maxsize=1)
def get_kb() -> DrugKB:
    return DrugKB.load(config.DATA_DIR / "drug_kb.json")


@lru_cache(maxsize=1)
def get_guidelines() -> GuidelineIndex:
    return GuidelineIndex.from_dir(config.DATA_DIR / "guidelines")


@lru_cache(maxsize=1)
def get_icd() -> ICDLookup:
    return ICDLookup.load(config.DATA_DIR / "icd10_subset.json")


@lru_cache(maxsize=4)
def _store_for(path: str) -> PatientStore:
    store = PatientStore(path)
    store.seed_from(config.DATA_DIR / "patients_seed.json")
    return store


def get_store() -> PatientStore:
    return _store_for(config.DB_PATH)


# ── Tool implementations ─────────────────────────────────────────────────────

def search_guidelines(query: str, k: int = 3) -> dict:
    """Search the clinical guideline corpus (RAG). Returns passages with ids you can cite."""
    return {"results": get_guidelines().search(query, max(1, min(int(k), 5)))}


def lookup_drug_interactions(drugs: list[str]) -> dict:
    """Check a list of drug names (generic or brand, doses allowed) for known interactions."""
    kb = get_kb()
    recognized, unrecognized = [], []
    for entry in drugs:
        found = kb.extract(entry)
        if found:
            recognized.extend(d for d in found if d not in recognized)
        else:
            unrecognized.append(entry)
    return {"recognized": recognized, "unrecognized": unrecognized, "interactions": kb.interactions(recognized)}


def check_allergy_conflicts(allergies: list[str], drugs: list[str]) -> dict:
    """Check whether any drug conflicts with the patient's documented allergies."""
    kb = get_kb()
    names = kb.extract_all(drugs)
    return {"drugs_checked": names, "conflicts": kb.allergy_conflicts(allergies, names)}


def lookup_icd10(condition: str, k: int = 3) -> dict:
    """Suggest ICD-10-CM codes for a condition the doctor stated. Suggestions need clinician confirmation."""
    return {"results": get_icd().search(condition, max(1, min(int(k), 5)))}


def get_patient_history(patient_id: str) -> dict:
    """Fetch stored allergies, conditions, medications and recent visits for a patient id."""
    patient = get_store().get_patient(patient_id)
    if patient is None:
        return {"found": False, "patient_id": patient_id}
    return {"found": True, **patient}


def flag_missing_info(note: dict) -> dict:
    """List documentation gaps in a draft note (missing duration, dose, allergy status and so on)."""
    missing: list[dict] = []
    details = note.get("symptom_details") or {}
    for f in DETAIL_SCALARS:
        if f != "location" and not (details.get(f) or "").strip():
            missing.append({"field": f"symptom_details.{f}", "severity": "low",
                            "message": f"{f.capitalize()} of symptoms was not stated."})
    prescriptions = note.get("prescriptions") or []
    if prescriptions and not (note.get("allergies") or []):
        missing.append({"field": "allergies", "severity": "high",
                        "message": "A prescription was written but allergy status is not documented."})
    for p in prescriptions:
        if not any(ch.isdigit() for ch in p):
            missing.append({"field": "prescriptions.dose", "severity": "medium",
                            "message": f"No dose or frequency stated for '{p}'."})
    if not (note.get("current_medications") or []):
        missing.append({"field": "current_medications", "severity": "low",
                        "message": "Current medications were not documented."})
    if not (note.get("assessment") or "").strip():
        missing.append({"field": "assessment", "severity": "medium", "message": "No assessment was stated."})
    if not (note.get("follow_up") or "").strip():
        missing.append({"field": "follow_up", "severity": "low", "message": "No follow-up instructions were stated."})
    return {"missing": missing}


# ── Registry ─────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_STR_ARRAY = {"type": "array", "items": {"type": "string"}}

REGISTRY: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("get_patient_history",
             "Fetch stored allergies, conditions, medications and recent visits for a patient id. Call this first when a patient id is given.",
             _obj({"patient_id": {"type": "string"}}, ["patient_id"]), get_patient_history),
    ToolSpec("search_guidelines",
             "Search the clinical guideline corpus. Use it to check red flags and cautions for the doctor's stated assessment. Cite returned ids in guideline_references.",
             _obj({"query": {"type": "string"}, "k": {"type": "integer", "default": 3}}, ["query"]), search_guidelines),
    ToolSpec("lookup_drug_interactions",
             "Check all current and newly prescribed medications for known interactions. Pass every medication, including ones from stored history.",
             _obj({"drugs": _STR_ARRAY}, ["drugs"]), lookup_drug_interactions),
    ToolSpec("check_allergy_conflicts",
             "Check prescribed drugs against documented allergies (from the transcript or stored history).",
             _obj({"allergies": _STR_ARRAY, "drugs": _STR_ARRAY}, ["allergies", "drugs"]), check_allergy_conflicts),
    ToolSpec("lookup_icd10",
             "Suggest ICD-10-CM codes for a condition the doctor stated. Only use codes returned here.",
             _obj({"condition": {"type": "string"}, "k": {"type": "integer", "default": 3}}, ["condition"]), lookup_icd10),
    ToolSpec("flag_missing_info",
             "Run a completeness check on a draft note and list documentation gaps. Verify gaps with search_transcript before accepting them.",
             _obj({"note": {"type": "object"}}, ["note"]), flag_missing_info),
]}


class ToolBackend(Protocol):
    def specs(self) -> list[dict]: ...
    def call(self, name: str, arguments: dict) -> dict: ...
    def close(self) -> None: ...


class ToolError(Exception):
    pass


class LocalBackend:
    """Runs the domain tools in process."""

    def specs(self) -> list[dict]:
        return [s.openai() for s in REGISTRY.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> dict:
        spec = REGISTRY.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name}")
        return spec.fn(**arguments)

    def close(self) -> None:
        pass
