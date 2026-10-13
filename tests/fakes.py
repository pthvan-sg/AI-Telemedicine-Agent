"""A scripted LLM for tests: replays a fixed list of turns, so the agent loop is tested offline."""

import json

from medscribe.llm import ToolCall, Turn

_counter = [0]


def call(name: str, **arguments) -> ToolCall:
    _counter[0] += 1
    return ToolCall(id=f"call_{_counter[0]}", name=name, arguments=arguments, raw_arguments=json.dumps(arguments))


def say(*calls: ToolCall, text: str | None = None) -> Turn:
    return Turn(content=text, tool_calls=list(calls))


class ScriptedLLM:
    """Agent turns come from `script`. Summariser and critic calls are answered separately so they
    never consume agent turns."""

    def __init__(self, script, critic=None, summary="(summary of earlier turns)"):
        self.script = list(script)
        self.critic = critic
        self.summary = summary
        self.agent_calls: list[dict] = []
        self.aux_calls: list[dict] = []

    def chat(self, messages, tools=None, max_tokens=None, json_mode=False):
        if json_mode:
            self.aux_calls.append({"kind": "json", "messages": messages})
            if self.critic:
                return Turn(content=self.critic(messages))
            return Turn(content=json.dumps({"results": []}))
        if tools is None:
            self.aux_calls.append({"kind": "summary", "messages": messages})
            return Turn(content=self.summary)
        self.agent_calls.append({"messages": list(messages), "tools": tools})
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of turns")
        nxt = self.script.pop(0)
        return nxt(messages) if callable(nxt) else nxt


def ev(field, item, quote, source="transcript"):
    return {"field": field, "item": item, "quote": quote, "source": source}


GOOD_NOTE = {
    "patient_chief_complaint": "persistent headache",
    "symptoms": ["headache", "sensitivity to light", "nausea"],
    "symptom_details": {"duration": "about 5 days", "severity": "6 or 7", "location": "left side",
                        "aggravating_factors": ["screens"], "relieving_factors": []},
    "medical_history": ["mother has migraines"],
    "current_medications": ["ibuprofen"],
    "assessment": "migraines",
    "plan": ["keep a headache diary", "avoid caffeine and irregular sleep"],
    "prescriptions": ["sumatriptan"],
    "follow_up": "come back in two weeks if it doesn't improve",
    "red_flags": ["sudden severe headache", "vision changes", "weakness"],
    "visit_summary": "Patient has had a left sided headache for about 5 days. The doctor diagnosed migraines "
                     "and prescribed sumatriptan.",
    "evidence": [
        ev("patient_chief_complaint", "persistent headache", "I've had this persistent headache for about 5 days now"),
        ev("symptoms", "headache", "I've had this persistent headache for about 5 days now"),
        ev("symptoms", "sensitivity to light", "Light bothers me a lot"),
        ev("symptoms", "nausea", "I felt nauseous yesterday"),
        ev("symptom_details.duration", "about 5 days", "for about 5 days now"),
        ev("symptom_details.severity", "6 or 7", "Around a 6 or 7"),
        ev("symptom_details.location", "left side", "It's mostly on the left side"),
        ev("symptom_details.aggravating_factors", "screens", "when I stare at screens too long"),
        ev("medical_history", "mother has migraines",
           "Any history of migraines in your family?\nPatient: My mother has them"),
        ev("current_medications", "ibuprofen", "Just ibuprofen, but it's not really helping"),
        ev("assessment", "migraines", "this looks like migraines"),
        ev("plan", "keep a headache diary", "I'd like you to keep a headache diary"),
        ev("plan", "avoid caffeine and irregular sleep", "Avoid known triggers like caffeine and irregular sleep"),
        ev("prescriptions", "sumatriptan", "I'm going to prescribe you sumatriptan for acute attacks"),
        ev("follow_up", "come back in two weeks if it doesn't improve",
           "Come back in two weeks if it doesn't improve"),
        ev("red_flags", "sudden severe headache", "sudden severe headache, vision changes, or weakness"),
        ev("red_flags", "vision changes", "sudden severe headache, vision changes, or weakness"),
        ev("red_flags", "weakness", "sudden severe headache, vision changes, or weakness"),
    ],
}


def note_with(**changes):
    """GOOD_NOTE with some fields replaced (deep copy, so tests cannot leak into each other)."""
    base = json.loads(json.dumps(GOOD_NOTE))
    base.update(changes)
    return base
