from capsize_voice.profile import analyze
from capsize_voice.render import render_style_guide

SAMPLE_POSTS = [
    "I shipped a new feature today. It works.",
    "This is fine. I think it will scale.",
    "The build broke again but I fixed it in an hour.",
    "I don't know why this took so long, honestly.",
    "Good code is boring code. Nothing clever, nothing cute.",
    "We shipped 4 releases this month and nobody noticed.",
]


def test_render_includes_persona_and_rules() -> None:
    profile = analyze(SAMPLE_POSTS)
    guide = render_style_guide(
        profile,
        persona="A software engineer who writes short posts.",
        rules=["Never mention politics.", "No hashtags."],
    )

    assert "A software engineer who writes short posts." in guide
    assert "Never mention politics." in guide
    assert "No hashtags." in guide
    assert "## Length" in guide
    assert "## Sentence construction" in guide
    assert "## Diction" in guide
    assert "## Punctuation" in guide


def test_render_omits_rules_section_when_none_given() -> None:
    profile = analyze(SAMPLE_POSTS)
    guide = render_style_guide(profile)
    assert "## Hard rules" not in guide
