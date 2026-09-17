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


def _passes_filters(
    text: str,
    words: list[str],
    min_words: int,
    max_words: int,
    banned: re.Pattern[str] | None,
    seen: set[str],
) -> bool:
    if not (min_words <= len(words) <= max_words):
        return False
    if banned and banned.search(text):
        return False
    if text.lower() in seen or text.startswith(("RT @", ">")):
        return False
    return not _ONLY_SYMBOLS.match(text)


def _score(
    text: str, words: list[str], onbrand: re.Pattern[str] | None
) -> tuple[int, bool]:
    is_on_brand = bool(onbrand and onbrand.search(text))
    score = 3 if is_on_brand else 0
    score += 2 if not _DEPENDENT.match(text) else 0
    score += 2 if text.rstrip()[-1:] in ".!?" else 0
    score += 1 if 10 <= len(words) <= 35 else 0
    score += 1 if re.search(r"\d", text) else 0
    return score, is_on_brand


@dataclass(frozen=True)
class _Options:
    min_words: int
    max_words: int
    banned: re.Pattern[str] | None
    onbrand: re.Pattern[str] | None


def _rank(texts: list[str], options: _Options) -> list[Exemplar]:
    seen: set[str] = set()
    scored: list[Exemplar] = []
    for raw in texts:
        text = clean(raw)
        words = text.split()
        if not _passes_filters(
            text, words, options.min_words, options.max_words,
            options.banned, seen,
        ):
            continue
        seen.add(text.lower())
        score, is_on_brand = _score(text, words, options.onbrand)
        scored.append(Exemplar(text=text, score=score, on_brand=is_on_brand))
    scored.sort(key=lambda e: -e.score)
    return scored


def curate(
    texts: list[str],
    banned_terms: list[str] | None = None,
    onbrand_terms: list[str] | None = None,
    min_words: int = 6,
    max_words: int = 60,
    top_n: int = 400,
) -> list[Exemplar]:
    """Filter and rank `texts`, returning the best `top_n` as exemplars.

    `banned_terms`/`onbrand_terms`: the caller's content policy and
    topic focus, case-insensitive substrings, no built-in default.
    """
    banned, onbrand = _compile_terms(banned_terms), _compile_terms(
        onbrand_terms
    )
    options = _Options(min_words, max_words, banned, onbrand)
    return _rank(texts, options)[:top_n]


def _compile_terms(terms: list[str] | None) -> re.Pattern[str] | None:
    if not terms:
        return None
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )
