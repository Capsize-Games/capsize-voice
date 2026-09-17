"""Measure a writer's style from their own text.

Ported from a script that fingerprinted one person's social posts via
NLTK: sentence structure, clause construction, part-of-speech mix,
function-word rates, punctuation habits, and lexical richness. Nothing
here is specific to any one author or language variety beyond the
English function-word and stopword lists below.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict

from nltk import pos_tag, sent_tokenize, word_tokenize

from capsize_voice.nltk_setup import ensure_nltk_data

_URL = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"^(@\w+[\s,]*)+")
_WORD = re.compile(r"^[A-Za-z']+$")
_PUNCT = re.compile(r"^[^\w\s]+$")

_POS_GROUPS = {
    "nouns": ("NN", "NNS", "NNP", "NNPS"),
    "verbs": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adjectives": ("JJ", "JJR", "JJS"),
    "adverbs": ("RB", "RBR", "RBS"),
    "pronouns": ("PRP", "PRP$"),
    "determiners": ("DT",),
    "prepositions": ("IN",),
    "modals": ("MD",),
    "conjunctions": ("CC",),
}

_SUBORDINATORS = {
    "because", "although", "though", "while", "since", "unless", "if",
    "when", "whereas", "until", "after", "before", "that", "which",
    "who", "whom", "whose",
}
_COORDINATORS = {"and", "but", "or", "so", "yet", "nor"}

_FUNCTION_WORDS = (
    "the of and a to in is you that it he was for on are as with his "
    "they i at be this have from or one had by but not what all were "
    "we when your can there use an each which she do how their if "
    "will up other about out many then them these so some her would "
    "make like him into time has more go see no way could people my "
    "than been who now find long day get come may just really "
    "actually still even much very too also always never think know "
    "need want thing things im dont thats ive isnt cant youre theyre"
).split()

_STOPWORDS = set(
    (
        "the a an and or but if then of to in on at for with as is are "
        "was were be been am it its this that these those i you he "
        "she they we them his her their my your our me him us do does "
        "did have has had not no so just very too also there here "
        "what which who when where why how all any both each few more "
        "most other some such only own same than s t d ll m re ve don "
        "didn doesn isn aren wasn weren won can will would could "
        "should about from by out up down into over under again once"
    ).split()
)


def clean(text: str) -> str:
    """Strip a quoted URL and any leading @-mentions from one post."""
    text = text.replace("&gt;", ">").replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    text = _URL.sub("", text)
    text = _HANDLE.sub("", text)
    return text.strip()


@dataclass(frozen=True)
class StyleProfile:
    """Measured writing-style statistics for one corpus of short posts."""

    n_posts: int
    n_tokens: int

    sent_len_mean: float
    sent_len_sd: float
    sents_per_post_mean: float
    one_sentence_post_pct: float
    subordinator_per_1k: float
    coordinator_per_1k: float
    comma_sentence_pct: float

    pos_mix: dict[str, float] = field(default_factory=dict)

    mean_word_len: float = 0.0
    long_word_pct: float = 0.0
    type_token_ratio: float = 0.0

    function_word_per_1k: dict[str, float] = field(default_factory=dict)
    punctuation_per_1k: dict[str, float] = field(default_factory=dict)
    top_content_words: list[tuple[str, int]] = field(default_factory=list)


class _SentenceStats(TypedDict):
    sent_len_mean: float
    sent_len_sd: float
    sents_per_post_mean: float
    one_sentence_post_pct: float
    comma_sentence_pct: float


class _ClauseRates(TypedDict):
    subordinator_per_1k: float
    coordinator_per_1k: float


class _WordShapeStats(TypedDict):
    mean_word_len: float
    long_word_pct: float
    type_token_ratio: float


class _WordFrequencyStats(TypedDict):
    function_word_per_1k: dict[str, float]
    punctuation_per_1k: dict[str, float]
    top_content_words: list[tuple[str, int]]


def _tokenize_corpus(
    texts: list[str], min_words: int
) -> tuple[list[str], list[list[str]], list[list[tuple[str, str]]]]:
    """Clean, length-filter, then word/POS-tokenize a corpus."""
    cleaned = [clean(t) for t in texts]
    cleaned = [t for t in cleaned if len(t.split()) >= min_words]
    if not cleaned:
        raise ValueError("no posts left after cleaning/length filtering")
    tokenized = [word_tokenize(t) for t in cleaned]
    tagged = [pos_tag(tokens) for tokens in tokenized]
    return cleaned, tokenized, tagged


def _sentence_stats(cleaned: list[str]) -> _SentenceStats:
    sentences = [s for t in cleaned for s in sent_tokenize(t)]
    sentence_lens = [len(word_tokenize(s)) for s in sentences]
    per_post = [len(sent_tokenize(t)) for t in cleaned]
    with_comma = sum(1 for s in sentences if "," in s)
    return {
        "sent_len_mean": round(statistics.mean(sentence_lens), 2),
        "sent_len_sd": round(statistics.pstdev(sentence_lens), 2),
        "sents_per_post_mean": round(statistics.mean(per_post), 2),
        "one_sentence_post_pct": round(
            sum(1 for x in per_post if x == 1) / len(cleaned) * 100, 1
        ),
        "comma_sentence_pct": round(
            with_comma / len(sentences) * 100, 1
        ),
    }


def _clause_rates(alpha: list[str], total: int) -> _ClauseRates:
    sub = sum(1 for w in alpha if w in _SUBORDINATORS) / total * 1000
    coord = sum(1 for w in alpha if w in _COORDINATORS) / total * 1000
    return {
        "subordinator_per_1k": round(sub, 1),
        "coordinator_per_1k": round(coord, 1),
    }


def _pos_mix(tagged: list[list[tuple[str, str]]]) -> dict[str, float]:
    pos_counts = Counter(
        tag for tagged_post in tagged for _, tag in tagged_post
    )
    pos_total = sum(pos_counts.values())
    return {
        group: round(sum(pos_counts[t] for t in tags) / pos_total * 100, 2)
        for group, tags in _POS_GROUPS.items()
    }


def _word_shape_stats(alpha: list[str], total: int) -> _WordShapeStats:
    word_lens = [len(w) for w in alpha]
    return {
        "mean_word_len": round(statistics.mean(word_lens), 2),
        "long_word_pct": round(
            sum(1 for w in word_lens if w > 6) / total * 100, 1
        ),
        "type_token_ratio": round(len(set(alpha)) / total, 4),
    }


def _content_words(alpha: list[str], excluded: set[str]) -> Counter[str]:
    """Count content words, `excluded` kept out of this list only.

    It still counts everywhere else in the profile. Matters when the
    same words feed a style guide that also states a hard rule against
    the topic: showing its most common word right above "never write
    about this" reads as a contradiction to the model.
    """
    return Counter(
        w
        for w in alpha
        if w not in _STOPWORDS and w not in excluded and len(w) > 2
    )


def _word_frequency_stats(
    alpha: list[str], all_words: list[str], total: int, excluded: set[str]
) -> _WordFrequencyStats:
    function_word_rate = Counter(alpha)
    punctuation = Counter(w for w in all_words if _PUNCT.match(w))
    return {
        "function_word_per_1k": {
            w: round(function_word_rate[w] / total * 1000, 2)
            for w in _FUNCTION_WORDS
            if function_word_rate[w]
        },
        "punctuation_per_1k": {
            p: round(c / total * 1000, 2)
            for p, c in punctuation.most_common(20)
        },
        "top_content_words": _content_words(alpha, excluded).most_common(60),
    }


def _flatten(tokenized: list[list[str]]) -> tuple[list[str], list[str], int]:
    """Return (lowercased alpha words, all raw tokens, alpha word count)."""
    all_words = [w for tokens in tokenized for w in tokens]
    alpha = [w.lower() for w in all_words if _WORD.match(w)]
    return alpha, all_words, len(alpha)


def analyze(
    texts: list[str],
    min_words: int = 3,
    exclude_words: list[str] | None = None,
) -> StyleProfile:
    """Measure `texts` into a `StyleProfile`."""
    ensure_nltk_data()
    excluded = {w.lower() for w in (exclude_words or [])}
    cleaned, tokenized, tagged = _tokenize_corpus(texts, min_words)
    alpha, all_words, total = _flatten(tokenized)
    return StyleProfile(
        n_posts=len(cleaned),
        n_tokens=total,
        pos_mix=_pos_mix(tagged),
        **_sentence_stats(cleaned),
        **_clause_rates(alpha, total),
        **_word_shape_stats(alpha, total),
        **_word_frequency_stats(alpha, all_words, total, excluded),
    )
