"""Test setup: isolate the patient database and make sure no real API key is ever used."""

import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="medscribe-tests-")
os.environ["MEDSCRIBE_DB"] = str(Path(_TMP) / "test.db")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("MEDSCRIBE_API_KEY", None)
os.environ["MEDSCRIBE_TRACE_DIR"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
