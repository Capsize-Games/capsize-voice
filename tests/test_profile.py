from capsize_voice.profile import analyze, clean


def test_clean_strips_urls_and_leading_mentions() -> None:
    text = "@alice @bob check this out https://example.com/x"
    assert clean(text) == "check this out"


SAMPLE_POSTS = [
    "I shipped a new feature today. It works.",
    "This is fine. I think it will scale.",
    "The build broke again but I fixed it in an hour.",
    "I don't know why this took so long, honestly.",
    "Good code is boring code. Nothing clever, nothing cute.",
    "We shipped 4 releases this month and nobody noticed.",
    "I like small tools that do one thing well.",
    "People assume this is hard. It really isn't.",
    "I've been doing this for 9 years and it never gets old.",
    "This bug took two days to find and five minutes to fix.",
]


def test_analyze_returns_plausible_profile() -> None:
    profile = analyze(SAMPLE_POSTS)

    assert profile.n_posts == len(SAMPLE_POSTS)
    assert profile.n_tokens > 0
    assert profile.sent_len_mean > 0
    assert 0 <= profile.one_sentence_post_pct <= 100
    assert set(profile.pos_mix) == {
        "nouns", "verbs", "adjectives", "adverbs", "pronouns",
        "determiners", "prepositions", "modals", "conjunctions",
    }
    assert profile.mean_word_len > 0
    assert profile.top_content_words


def test_analyze_drops_short_fragments() -> None:
    profile = analyze(["ok", "cool", *SAMPLE_POSTS], min_words=3)
    assert profile.n_posts == len(SAMPLE_POSTS)


def test_analyze_raises_on_empty_corpus() -> None:
    import pytest

    with pytest.raises(ValueError):
        analyze(["ok", "no"], min_words=3)


def test_analyze_excludes_words_from_top_content_words() -> None:
    posts = [*SAMPLE_POSTS, "I think politics ruins every good discussion."]
    profile = analyze(posts, exclude_words=["politics"])
    words = {w for w, _ in profile.top_content_words}
    assert "politics" not in words
