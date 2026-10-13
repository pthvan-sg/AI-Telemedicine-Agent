"""Long-term memory: a small SQLite patient store.

Short-term memory (the running summary of a long consultation) lives in context.py.
Only synthetic, pseudonymous patient ids belong here. Do not store real PHI.
"""

import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from .text import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    allergies TEXT NOT NULL DEFAULT '[]',
    conditions TEXT NOT NULL DEFAULT '[]',
    medications TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    visit_date TEXT NOT NULL,
    assessment TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    note_json TEXT
);
"""


def _merge(existing: list[str], new: list[str]) -> list[str]:
    seen = {normalize(x) for x in existing}
    out = list(existing)
    for item in new:
        if item and normalize(item) not in seen:
            out.append(item)
            seen.add(normalize(item))
    return out


class PatientStore:
    def __init__(self, path: str):
        self.path = path
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def seed_from(self, seed_path: Path) -> int:
        """Load synthetic patients if the table is empty. Returns number inserted."""
        with closing(self._conn()) as c:
            if c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]:
                return 0
            data = json.loads(Path(seed_path).read_text(encoding="utf-8"))["patients"]
            for p in data:
                c.execute("INSERT INTO patients VALUES (?,?,?,?)",
                          (p["patient_id"], json.dumps(p["allergies"]), json.dumps(p["conditions"]),
                           json.dumps(p["medications"])))
                for v in p.get("visits", []):
                    c.execute("INSERT INTO visits (patient_id, visit_date, assessment, summary) VALUES (?,?,?,?)",
                              (p["patient_id"], v["date"], v.get("assessment", ""), v.get("summary", "")))
            c.commit()
            return len(data)

    def get_patient(self, patient_id: str, recent: int = 3) -> dict | None:
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            if not row:
                return None
            visits = c.execute(
                "SELECT visit_date, assessment, summary FROM visits WHERE patient_id = ? "
                "ORDER BY visit_date DESC, id DESC LIMIT ?", (patient_id, recent)).fetchall()
            return {
                "patient_id": row["patient_id"],
                "allergies": json.loads(row["allergies"]),
                "conditions": json.loads(row["conditions"]),
                "medications": json.loads(row["medications"]),
                "recent_visits": [dict(v) for v in visits],
            }

    def record_visit(self, patient_id: str, note: dict, visit_date: str | None = None) -> None:
        """Merge facts from a verified note into the patient profile and log the visit."""
        existing = self.get_patient(patient_id) or {"allergies": [], "conditions": [], "medications": []}
        allergies = _merge(existing["allergies"], note.get("allergies", []))
        conditions = _merge(existing["conditions"], note.get("medical_history", []))
        meds = _merge(existing["medications"], note.get("current_medications", []) + note.get("prescriptions", []))
        with closing(self._conn()) as c:
            c.execute(
                "INSERT INTO patients (patient_id, allergies, conditions, medications) VALUES (?,?,?,?) "
                "ON CONFLICT(patient_id) DO UPDATE SET allergies=excluded.allergies, "
                "conditions=excluded.conditions, medications=excluded.medications",
                (patient_id, json.dumps(allergies), json.dumps(conditions), json.dumps(meds)))
            c.execute(
                "INSERT INTO visits (patient_id, visit_date, assessment, summary, note_json) VALUES (?,?,?,?,?)",
                (patient_id, visit_date or date.today().isoformat(), note.get("assessment", ""),
                 note.get("visit_summary", ""), json.dumps(note)))
            c.commit()
