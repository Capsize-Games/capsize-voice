from capsize_voice.safety import is_flagged, load_categories, score_text

CATEGORIES = load_categories(
    [
        {
            "id": "profanity",
            "label": "Profanity",
            "weight": 5,
            "terms": [r"\bdamn\b"],
        },
        {
            "id": "politics",
            "label": "Politics",
            "weight": 10,
            "terms": [r"\belection\b", r"\bsenator\b"],
        },
    ]
)


def test_score_text_no_match() -> None:
    score = score_text("shipped a small fix today", CATEGORIES)
    assert score.total == 0
    assert score.matches == []


def test_score_text_single_match() -> None:
    score = score_text("damn, that bug took forever", CATEGORIES)
    assert score.total == 5
    assert [m.category_id for m in score.matches] == ["profanity"]


def test_score_text_multiple_matches_sum_weights() -> None:
    score = score_text(
        "damn, the senator said something dumb about the election",
        CATEGORIES,
    )
    assert score.total == 15
    assert {m.category_id for m in score.matches} == {
        "profanity",
        "politics",
    }


def test_score_text_case_insensitive() -> None:
    score = score_text("DAMN it", CATEGORIES)
    assert score.total == 5


def test_category_counts_once_even_with_multiple_term_matches() -> None:
    score = score_text("the senator discussed the election", CATEGORIES)
    politics_matches = [
        m for m in score.matches if m.category_id == "politics"
    ]
    assert len(politics_matches) == 1


def test_is_flagged() -> None:
    score = score_text("damn it", CATEGORIES)
    assert is_flagged(score, threshold=1) is True
    assert is_flagged(score, threshold=10) is False


def test_load_categories_drops_invalid_regex() -> None:
    categories = load_categories(
        [
            {
                "id": "broken",
                "label": "Broken",
                "weight": 1,
                "terms": ["(unclosed"],
            }
        ]
    )
    assert categories == []


def test_load_categories_keeps_valid_terms_alongside_invalid() -> None:
    categories = load_categories(
        [
            {
                "id": "mixed",
                "label": "Mixed",
                "weight": 1,
                "terms": ["(unclosed", r"\bok\b"],
            }
        ]
    )
    assert len(categories) == 1
    assert categories[0].terms == [r"\bok\b"]
