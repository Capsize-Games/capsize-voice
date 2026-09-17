"""capsize_voice - measure a writer's voice, then generate more of it."""

from capsize_voice.classify import should_interject
from capsize_voice.exemplars import Exemplar, curate
from capsize_voice.generate import (
    GenerationError,
    generate_candidates,
    generate_reply,
)
from capsize_voice.memory import extract_facts
from capsize_voice.profile import StyleProfile, analyze, clean
from capsize_voice.render import render_style_guide
from capsize_voice.safety import (
    CategoryData,
    SafetyCategory,
    SafetyMatch,
    SafetyScore,
    is_flagged,
    load_categories,
    score_text,
)
from capsize_voice.x_archive import (
    ArchivePost,
    load_archive,
    own_post_texts,
)

__all__ = [
    "ArchivePost",
    "CategoryData",
    "Exemplar",
    "GenerationError",
    "SafetyCategory",
    "SafetyMatch",
    "SafetyScore",
    "StyleProfile",
    "analyze",
    "clean",
    "curate",
    "extract_facts",
    "generate_candidates",
    "generate_reply",
    "is_flagged",
    "load_archive",
    "load_categories",
    "own_post_texts",
    "render_style_guide",
    "score_text",
    "should_interject",
]
