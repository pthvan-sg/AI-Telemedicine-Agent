"""Small text helpers shared by retrieval, verification and evaluation."""

import re

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her his i if in into is it its
me my no not of on or our she so than that the their them then there these they this to too us was we were what
when which who will with would you your yes okay ok just really also about any some very doctor patient
""".split())


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def stem(tok: str) -> str:
    """Very light stemmer: enough to match plurals and simple verb forms."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us")):
        return tok[:-1]
    return tok


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    out = []
    for t in normalize(text).split():
        if drop_stop and t in STOPWORDS:
            continue
        out.append(stem(t))
    return out
