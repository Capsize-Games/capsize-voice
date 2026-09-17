"""Curate a few-shot exemplar pool out of a writer's raw post history.

Picks self-contained, well-formed posts a generation prompt can safely
sample from. An X archive export has no engagement counts for the
account's own posts, so scoring here relies only on what the raw text
itself shows — completeness, standalone-ness, length, and (optionally)
topic relevance and a banned-terms filter the caller supplies.
"""

import re
from dataclasses import dataclass

from capsize_voice.profile import clean

_DEPENDENT = re.compile(
    r"^(yes|no|yeah|nah|yep|nope|agreed|exactly|same|true|false|this|"
    r"lol|lmao|correct|right|wrong|thanks|thank you|congrats|nice|cool|"
    r"sure|ok|okay|i agree|i disagree|me too|and |but |or |so |because "
    r"|which |who |that's it)\b",
    re.IGNORECASE,
)
_ONLY_SYMBOLS = re.compile(r"^[\W\d\s]+$")


@dataclass(frozen=True)
class Exemplar:
    """One curated post, with the score that ranked it."""

    text: str
    score: int
    on_brand: bool


def curate(
    texts: list[str],
    banned_terms: list[str] | None = None,
    onbrand_terms: list[str] | None = None,
    min_words: int = 6,
    max_words: int = 60,
    top_n: int = 400,
) -> list[Exemplar]:
    """Filter and rank `texts`, returning the best `top_n` as exemplars.

    `banned_terms` and `onbrand_terms` are the caller's own content
    policy and topic focus — this package has no default opinion on
    either; both are treated as case-insensitive substrings.
    """
    banned = _compile_terms(banned_terms)
    onbrand = _compile_terms(onbrand_terms)

    seen: set[str] = set()
    scored: list[Exemplar] = []
    for raw in texts:
        text = clean(raw)
        words = text.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if banned and banned.search(text):
            continue
        if text.lower() in seen:
            continue
        if text.startswith(("RT @", ">")):
            continue
        if _ONLY_SYMBOLS.match(text):
            continue
        seen.add(text.lower())

        is_on_brand = bool(onbrand and onbrand.search(text))
        score = 0
        score += 3 if is_on_brand else 0
        score += 2 if not _DEPENDENT.match(text) else 0
        score += 2 if text.rstrip()[-1:] in ".!?" else 0
        score += 1 if 10 <= len(words) <= 35 else 0
        score += 1 if re.search(r"\d", text) else 0
        scored.append(Exemplar(text=text, score=score, on_brand=is_on_brand))

    scored.sort(key=lambda e: -e.score)
    return scored[:top_n]


def _compile_terms(terms: list[str] | None) -> re.Pattern[str] | None:
    if not terms:
        return None
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )
