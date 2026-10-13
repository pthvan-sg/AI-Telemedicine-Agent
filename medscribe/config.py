"""Central configuration. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MEDSCRIBE_DATA_DIR", ROOT / "data"))
DB_PATH = os.getenv("MEDSCRIBE_DB", str(ROOT / "medscribe.db"))

MODEL = os.getenv("MEDSCRIBE_MODEL", "gpt-4o")
# Set MEDSCRIBE_TEMPERATURE="" to omit the parameter (some newer models reject it).
_temp = os.getenv("MEDSCRIBE_TEMPERATURE", "0.2")
TEMPERATURE = float(_temp) if _temp != "" else None

MAX_STEPS = int(os.getenv("MEDSCRIBE_MAX_STEPS", "12"))
MAX_REVISIONS = int(os.getenv("MEDSCRIBE_MAX_REVISIONS", "2"))
CONTEXT_BUDGET_TOKENS = int(os.getenv("MEDSCRIBE_CONTEXT_BUDGET", "6000"))

# "local" runs domain tools in process. "mcp" runs them in a separate MCP server over stdio.
TOOL_BACKEND = os.getenv("MEDSCRIBE_TOOL_BACKEND", "local")

MAX_OUTPUT_TOKENS = int(os.getenv("MEDSCRIBE_MAX_OUTPUT_TOKENS", "3000"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MEDSCRIBE_MAX_TRANSCRIPT_CHARS", "200000"))

# If set, every run appends one JSON line (the full trajectory) to <TRACE_DIR>/trajectories.jsonl.
TRACE_DIR = os.getenv("MEDSCRIBE_TRACE_DIR", "")
