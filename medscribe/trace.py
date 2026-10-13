"""Run tracing. Every LLM call, tool call and verification is recorded as an event.

A finished run becomes a trajectory (state, action, result per step plus a scalar reward). The
reward is a plain starting point for RL or preference work: it rewards grounded notes and
penalises wasted revisions and tool calls. Change it freely.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config


@dataclass
class Tracer:
    events: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def add(self, type_: str, **data: Any) -> dict:
        event = {"type": type_, "at_ms": round((time.perf_counter() - self._t0) * 1000), **data}
        self.events.append(event)
        return event

    @property
    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)


def summarize(events: list[dict], status: str, grounded_ratio: float, revisions: int) -> dict:
    tool_events = [e for e in events if e["type"] == "tool" and not e.get("auto")]
    errors = sum(1 for e in tool_events if not e.get("ok", True))
    llm_events = [e for e in events if e["type"] == "llm"]
    reward = grounded_ratio - 0.05 * revisions - 0.01 * max(0, len(tool_events) - 10)
    if status == "failed":
        reward -= 1.0
    return {
        "status": status,
        "llm_calls": len(llm_events),
        "tool_calls": len(tool_events),
        "tool_errors": errors,
        "tool_success_rate": round(1 - errors / len(tool_events), 3) if tool_events else 1.0,
        "revisions": revisions,
        "prompt_tokens": sum(e.get("usage", {}).get("prompt_tokens", 0) for e in llm_events),
        "completion_tokens": sum(e.get("usage", {}).get("completion_tokens", 0) for e in llm_events),
        "grounded_ratio": round(grounded_ratio, 3),
        "reward": round(reward, 3),
    }


def write_trajectory(record: dict) -> None:
    """Append one run to trajectories.jsonl when MEDSCRIBE_TRACE_DIR is set."""
    if not config.TRACE_DIR:
        return
    path = Path(config.TRACE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "trajectories.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
