"""Turn a measured `StyleProfile` into an LLM-readable style guide.

Mechanical and deterministic: every number in the output traces back to
the profile. It won't read as polished as a style guide someone hand-
edits after looking at these numbers — this is the baseline to start
from and improve, not a replacement for that judgment.
"""

from capsize_voice.profile import StyleProfile


def _sections(
    profile: StyleProfile, persona: str, rules: list[str] | None
) -> list[str]:
    sections = [persona.strip()] if persona else []
    sections += [
        _length_section(profile),
        _sentence_section(profile),
        _diction_section(profile),
        _punctuation_section(profile),
    ]
    if rules:
        sections.append(_rules_section(rules))
    return sections


def render_style_guide(
    profile: StyleProfile,
    persona: str = "",
    rules: list[str] | None = None,
) -> str:
    """Render `profile` as a markdown style guide for a generation prompt.

    `persona` is a short bio/description of the writer, prepended as-is.
    `rules` are hard prohibitions specific to this writer (banned topics,
    formats to avoid) — this package has no opinion on what they should
    be, only that a caller can supply them.
    """
    return "\n\n".join(_sections(profile, persona, rules)) + "\n"


def _length_section(p: StyleProfile) -> str:
    return (
        "## Length\n\n"
        f"- Sentences per post: mean {p.sents_per_post_mean} "
        f"({p.one_sentence_post_pct}% are a single sentence).\n"
        f"- Words per sentence: mean {p.sent_len_mean}, "
        f"sd {p.sent_len_sd} — vary sentence length deliberately; a flat "
        "sentence-length distribution is a machine tell."
    )


def _sentence_section(p: StyleProfile) -> str:
    coord_vs_sub = (
        "prefers coordination (and/but/so) over subordination "
        "(because/although/which)"
        if p.coordinator_per_1k > p.subordinator_per_1k
        else "prefers subordination (because/although/which) over "
        "coordination (and/but/so)"
    )
    return (
        "## Sentence construction\n\n"
        f"- {coord_vs_sub.capitalize()} "
        f"({p.coordinator_per_1k}/1k coordinators vs. "
        f"{p.subordinator_per_1k}/1k subordinators).\n"
        f"- {p.comma_sentence_pct}% of sentences contain a comma.\n"
        f"- Part-of-speech mix: "
        + ", ".join(f"{k} {v}%" for k, v in p.pos_mix.items())
    )


def _diction_section(p: StyleProfile) -> str:
    top_words = ", ".join(w for w, _ in p.top_content_words[:20])
    return (
        "## Diction\n\n"
        f"- Mean word length {p.mean_word_len} characters; "
        f"{p.long_word_pct}% of words exceed 6 characters.\n"
        f"- Type/token ratio {p.type_token_ratio} (lexical variety).\n"
        f"- Recurring content words: {top_words}."
    )


def _punctuation_section(p: StyleProfile) -> str:
    notable = ", ".join(
        f"{mark!r} {rate}/1k" for mark, rate in p.punctuation_per_1k.items()
    )
    return f"## Punctuation\n\nObserved rates per 1,000 words: {notable}."


def _rules_section(rules: list[str]) -> str:
    bullets = "\n".join(f"- {rule}" for rule in rules)
    return f"## Hard rules\n\nNever violate these, at any rate:\n\n{bullets}"
