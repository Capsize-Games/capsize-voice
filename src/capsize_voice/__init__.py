"""capsize_voice - measure a writer's voice, then generate more of it."""

from capsize_voice.exemplars import Exemplar, curate
from capsize_voice.generate import GenerationError, generate_candidates
from capsize_voice.profile import StyleProfile, analyze, clean
from capsize_voice.render import render_style_guide
from capsize_voice.x_archive import (
    ArchivePost,
    load_archive,
    own_post_texts,
)

__all__ = [
    "ArchivePost",
    "Exemplar",
    "GenerationError",
    "StyleProfile",
    "analyze",
    "clean",
    "curate",
    "generate_candidates",
    "load_archive",
    "own_post_texts",
    "render_style_guide",
]
