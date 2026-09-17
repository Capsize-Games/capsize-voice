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


def analyze(
    texts: list[str],
    min_words: int = 3,
    exclude_words: list[str] | None = None,
) -> StyleProfile:
    """Measure `texts` (one person's own posts) into a `StyleProfile`.

    `min_words` drops fragments too short to carry sentence structure.
    `exclude_words` keeps specific terms out of `top_content_words` only
    — the rest of the measurement still reflects the real corpus. This
    matters when the same profile feeds a style guide that also states a
    hard rule against a topic: showing its most common word right above
    "never write about this" reads as a contradiction to the model.
    """
    ensure_nltk_data()
    excluded = {w.lower() for w in (exclude_words or [])}
    cleaned = [clean(t) for t in texts]
    cleaned = [t for t in cleaned if len(t.split()) >= min_words]
    if not cleaned:
        raise ValueError("no posts left after cleaning/length filtering")

    tokenized = [word_tokenize(t) for t in cleaned]
    tagged = [pos_tag(tokens) for tokens in tokenized]
    all_words = [w for tokens in tokenized for w in tokens]
    alpha = [w.lower() for w in all_words if _WORD.match(w)]
    total = len(alpha)

    sentences = [s for t in cleaned for s in sent_tokenize(t)]
    sentence_lens = [len(word_tokenize(s)) for s in sentences]
    per_post = [len(sent_tokenize(t)) for t in cleaned]
    with_comma = sum(1 for s in sentences if "," in s)

    sub_rate = sum(1 for w in alpha if w in _SUBORDINATORS) / total * 1000
    coord_rate = sum(1 for w in alpha if w in _COORDINATORS) / total * 1000

    pos_counts = Counter(
        tag for tagged_post in tagged for _, tag in tagged_post
    )
    pos_total = sum(pos_counts.values())
    pos_mix = {
        group: round(sum(pos_counts[t] for t in tags) / pos_total * 100, 2)
        for group, tags in _POS_GROUPS.items()
    }

    word_lens = [len(w) for w in alpha]
    types = len(set(alpha))

    function_word_rate = Counter(alpha)
    punctuation = Counter(w for w in all_words if _PUNCT.match(w))
    content_words = Counter(
        w
        for w in alpha
        if w not in _STOPWORDS and w not in excluded and len(w) > 2
    )

    return StyleProfile(
        n_posts=len(cleaned),
        n_tokens=total,
        sent_len_mean=round(statistics.mean(sentence_lens), 2),
        sent_len_sd=round(statistics.pstdev(sentence_lens), 2),
        sents_per_post_mean=round(statistics.mean(per_post), 2),
        one_sentence_post_pct=round(
            sum(1 for x in per_post if x == 1) / len(cleaned) * 100, 1
        ),
        subordinator_per_1k=round(sub_rate, 1),
        coordinator_per_1k=round(coord_rate, 1),
        comma_sentence_pct=round(with_comma / len(sentences) * 100, 1),
        pos_mix=pos_mix,
        mean_word_len=round(statistics.mean(word_lens), 2),
        long_word_pct=round(
            sum(1 for w in word_lens if w > 6) / total * 100, 1
        ),
        type_token_ratio=round(types / total, 4),
        function_word_per_1k={
            w: round(function_word_rate[w] / total * 1000, 2)
            for w in _FUNCTION_WORDS
            if function_word_rate[w]
        },
        punctuation_per_1k={
            p: round(c / total * 1000, 2)
            for p, c in punctuation.most_common(20)
        },
        top_content_words=content_words.most_common(60),
    )
