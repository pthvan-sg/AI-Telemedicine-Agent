"""Retrieval over the local guideline corpus and over the consultation transcript.

BM25 is implemented in pure Python so the project has no vector database dependency and
retrieval is deterministic and easy to unit test. The index exposes a single `search`
method, so swapping in embeddings (FAISS, Chroma, pgvector) later only touches this file.
"""

import math
import re
from collections import Counter
from pathlib import Path

from .text import tokens


class BM25Index:
    def __init__(self, docs: list[dict], text_key: str = "index_text", k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tokens = [tokens(d[text_key]) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(docs)) if docs else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for t in self.doc_tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [{**self.docs[i], "score": round(s, 3)} for s, i in scored[:k]]


class GuidelineIndex:
    """Chunks markdown files by paragraph and indexes them with BM25."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.index = BM25Index(chunks)

    @classmethod
    def from_dir(cls, directory: Path) -> "GuidelineIndex":
        chunks: list[dict] = []
        for path in sorted(Path(directory).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            doc_title = path.stem
            heading = doc_title
            n = 0
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if not block:
                    continue
                if block.startswith("# "):
                    doc_title = block[2:].strip()
                    continue
                if block.startswith("## "):
                    first, _, rest = block.partition("\n")
                    heading = first[3:].strip()
                    body = rest.strip()
                else:
                    body = block
                if not body or body.lower().startswith("demo corpus"):
                    continue
                n += 1
                chunks.append({
                    "id": f"{path.stem}#{n}",
                    "source": path.name,
                    "title": f"{doc_title}: {heading}",
                    "text": body,
                    "index_text": f"{doc_title} {heading} {body}",
                })
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        hits = self.index.search(query, k)
        return [{"id": h["id"], "title": h["title"], "text": h["text"], "score": h["score"]} for h in hits]
