"""Ask an LLM for candidate posts in a measured voice.

The style guide (see `render.render_style_guide`, or your own hand-tuned
text) goes in as a cached system prompt; a handful of the writer's real
posts go in as few-shot exemplars; the model is asked for a JSON array of
distinct candidates about a given context.
"""

import json
import random

import anthropic

DEFAULT_MODEL = "claude-opus-5"

_INSTRUCTION = """\
Follow the style guide above exactly - especially the length targets. \
Below are {k} real examples of this person's own writing. Match their \
rhythm and specificity. Do not imitate their subject matter.

<examples>
{exemplars}
</examples>

Write {n} distinct candidates about the context below, varying length \
across them.

Return ONLY a JSON array of strings. No commentary, no markdown fence.

<context>
{context}
</context>
"""


class GenerationError(Exception):
    """Raised when the model can't be reached or returns unusable output."""


def generate_candidates(
    api_key: str,
    style_guide: str,
    exemplars: list[str],
    context: str,
    count: int,
    seed_count: int = 20,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Return up to `count` distinct candidate posts about `context`."""
    if not api_key:
        raise GenerationError("no API key configured")
    if not exemplars:
        raise GenerationError("no exemplars supplied")

    seeds = random.sample(exemplars, min(seed_count, len(exemplars)))
    prompt = _INSTRUCTION.format(
        k=len(seeds),
        n=count,
        exemplars="\n".join(f"- {s}" for s in seeds),
        context=context,
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=[
                {
                    "type": "text",
                    "text": style_guide,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise GenerationError(str(exc)) from exc

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        candidates = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"model did not return JSON: {text[:200]}"
        ) from exc
    return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
