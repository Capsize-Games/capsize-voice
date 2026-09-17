from capsize_voice.exemplars import curate


def test_curate_filters_short_and_long_and_duplicates() -> None:
    texts = [
        "ok",
        "This is a good, complete, standalone sentence about shipping code.",
        "This is a good, complete, standalone sentence about shipping code.",
        " ".join(["word"] * 100),
    ]
    result = curate(texts, min_words=6, max_words=60)
    assert len(result) == 1
    assert "shipping code" in result[0].text


def test_curate_drops_banned_terms() -> None:
    texts = [
        "I have strong opinions about politics and elections this year.",
        "I shipped a small tool that saves me an hour every week.",
    ]
    result = curate(texts, banned_terms=["politics", "election"])
    assert len(result) == 1
    assert "shipped" in result[0].text


def test_curate_scores_onbrand_higher() -> None:
    texts = [
        "I went for a walk and thought about nothing in particular.",
        "I refactored the build pipeline and cut CI time in half.",
    ]
    result = curate(texts, onbrand_terms=["refactored", "pipeline", "ci"])
    assert result[0].on_brand is True
    assert result[0].score > result[1].score


def test_curate_drops_dependent_replies() -> None:
    texts = [
        "yes exactly this is the whole problem with that approach",
        "Shipping small and often beats one giant release every time.",
    ]
    result = curate(texts)
    scores = {e.text: e.score for e in result}
    assert (
        scores["Shipping small and often beats one giant release every time."]
        > scores["yes exactly this is the whole problem with that approach"]
    )
