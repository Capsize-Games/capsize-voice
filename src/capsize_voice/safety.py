"""Score text against a caller-supplied set of weighted content rules.

Ported from a script that triaged years of one person's old posts for
deletion: each category is a regex term list with a weight; a text's
score is the sum of every category it matches at least once. Kept as
plain, inspectable rules rather than a trained classifier on purpose —
predictable and auditable is what a hard safety gate needs, and it costs
no inference. This module has no opinion on what the categories are or
where the threshold sits; both are the caller's own content policy.
"""

import re
from dataclasses import dataclass

from typing_extensions import TypedDict


class CategoryData(TypedDict):
    """The shape `load_categories` expects per entry."""

    id: str
    label: str
    weight: float
    terms: list[str]


@dataclass(frozen=True)
class SafetyCategory:
    """One weighted content rule: an id/label and the regexes for it."""

    id: str
    label: str
    weight: float
    terms: list[str]


@dataclass(frozen=True)
class SafetyMatch:
    """One category that matched, for logging/explaining a flag."""

    category_id: str
    label: str
    weight: float


@dataclass(frozen=True)
class SafetyScore:
    """The total weight matched, and which categories contributed it."""

    total: float
    matches: list[SafetyMatch]


def _is_valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def load_categories(data: list[CategoryData]) -> list[SafetyCategory]:
    """Build categories from `data`.

    Drops any regex that won't compile — one bad pattern shouldn't
    disable an entire category.
    """
    categories = []
    for entry in data:
        terms = [t for t in entry["terms"] if _is_valid_regex(t)]
        if terms:
            categories.append(
                SafetyCategory(
                    entry["id"], entry["label"], entry["weight"], terms
                )
            )
    return categories


def score_text(text: str, categories: list[SafetyCategory]) -> SafetyScore:
    """Score `text`: each matching category counts once, at its weight."""
    matches = []
    total = 0.0
    for category in categories:
        if any(re.search(t, text, re.IGNORECASE) for t in category.terms):
            total += category.weight
            matches.append(
                SafetyMatch(category.id, category.label, category.weight)
            )
    return SafetyScore(total=total, matches=matches)


def is_flagged(score: SafetyScore, threshold: float) -> bool:
    """Return whether `score` meets or exceeds `threshold`."""
    return score.total >= threshold
