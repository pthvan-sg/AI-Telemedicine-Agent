"""Context management for long consultations.

Two mechanisms:
1. A running summary. If the transcript exceeds the token budget, older chunks are
   summarised by the LLM and only the most recent chunk is kept verbatim in the prompt.
2. Transcript retrieval. The full transcript stays available to the agent through the
   search_transcript tool (BM25 over turns), so exact wording can always be recovered
   for evidence quotes even when the prompt only holds a summary.
"""

import re

from .rag import BM25Index

SUMMARY_PROMPT = (
    "You summarise parts of a doctor-patient consultation for a clinical scribe. "
    "Keep every clinically relevant fact: symptoms with onset and severity, medications with doses, "
    "allergies, history, what the doctor assessed, prescribed and advised. Keep numbers and drug names "
    "exactly as spoken. Do not add anything that was not said. Be terse."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_turns(transcript: str) -> list[str]:
    """One turn per non-empty line. Very long lines are split on sentence boundaries."""
    turns: list[str] = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 1500:
            turns.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
        else:
            turns.append(line)
    return turns


class TranscriptIndex:
    def __init__(self, turns: list[str]):
        self.turns = turns
        self.index = BM25Index([{"turn": i, "text": t, "index_text": t} for i, t in enumerate(turns)])

    def search(self, query: str, k: int = 3) -> list[dict]:
        out = []
        for hit in self.index.search(query, k):
            i = hit["turn"]
            out.append({
                "turn": i,
                "previous": self.turns[i - 1] if i > 0 else "",
                "text": self.turns[i],
                "next": self.turns[i + 1] if i + 1 < len(self.turns) else "",
            })
        return out


class ContextManager:
    def __init__(self, budget_tokens: int = 6000):
        self.budget = budget_tokens

    def prepare(self, transcript: str, llm) -> tuple[str, bool]:
        """Return (text for the prompt, whether it was condensed)."""
        if estimate_tokens(transcript) <= self.budget:
            return transcript, False
        turns = split_turns(transcript)
        chunk_budget = max(1, self.budget // 3)
        chunks: list[list[str]] = [[]]
        size = 0
        for t in turns:
            n = estimate_tokens(t)
            if size + n > chunk_budget and chunks[-1]:
                chunks.append([])
                size = 0
            chunks[-1].append(t)
            size += n
        *older, recent = chunks
        summary = ""
        for chunk in older:
            user = (f"Summary so far:\n{summary or '(none)'}\n\nNew part of the consultation:\n" + "\n".join(chunk) +
                    "\n\nReturn the updated summary only.")
            turn = llm.chat([{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}],
                            tools=None, max_tokens=900)
            summary = (turn.content or summary).strip()
        text = ("[Running summary of the earlier part of the consultation. Use search_transcript for exact wording.]\n"
                f"{summary}\n\n[Most recent turns, verbatim]\n" + "\n".join(recent))
        return text, True
