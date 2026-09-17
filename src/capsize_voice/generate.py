"""Ask an LLM for candidate posts in a measured voice.

The style guide (see `render.render_style_guide`, or your own hand-tuned
text) goes in as a system prompt; a handful of the writer's real posts go
in as few-shot exemplars; the model is asked for a JSON array of distinct
candidates about a given context.

Talks OpenAI-compatible chat completions, not any one vendor's own SDK —
that is the one interface every major router (OpenRouter, DeepInfra,
Together, and the providers themselves) already speaks, so pointing this
at a different `base_url` is the only thing switching providers needs.

`generate_candidates` returns several candidates for a review queue;
`generate_reply` returns one, for a live conversation that isn't going
to wait on one. Both share `model`'s no-default rule: a generic tool
defaulting to one vendor's model would pick that cost and behavior for
every caller who didn't think to override it. Both take `extra_body`,
passed straight through to the request — on OpenRouter that's where a
pinned provider order (`{"provider": {"order": [...]}}`) belongs, and
it means this module never needs to know that concept exists.
"""

import json
import random
from dataclasses import dataclass

import openai

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

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

_REPLY_INSTRUCTION = """\
Follow the style guide above exactly. Below are {k} real examples of \
this person's own writing, to match rhythm and specificity - do not \
imitate their subject matter.

<examples>
{exemplars}
</examples>

Recent conversation so far, oldest first - there may be more than one \
other speaker:

<history>
{history}
</history>

Someone named {author} just said this to you in a conversation:

<message>
{message}
</message>

Reply in character, as a single short message. Return ONLY the reply \
text - no commentary, no quotation marks, no "as an AI" framing.

Do not invent specific facts, activities, or events you have no real \
basis for (what you've been doing, working on, or experiencing right \
now) - the examples and history above show HOW this person writes, \
not what is actually true today. If asked something concrete you \
don't have real information for, answer vaguely or deflect in \
character, the way a real person would rather than making something \
up.
"""


class GenerationError(Exception):
    """Raised when the model can't be reached or returns unusable output."""


def _build_prompt(exemplars: list[str], context: str, n: int, k: int) -> str:
    seeds = random.sample(exemplars, min(k, len(exemplars)))
    return _INSTRUCTION.format(
        k=len(seeds),
        n=n,
        exemplars="\n".join(f"- {s}" for s in seeds),
        context=context,
    )


@dataclass(frozen=True)
class _Request:
    api_key: str
    base_url: str
    model: str
    extra_body: dict[str, object] | None


def _call_model(request: _Request, style_guide: str, prompt: str) -> str:
    client = openai.OpenAI(
        api_key=request.api_key, base_url=request.base_url
    )
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": style_guide},
                {"role": "user", "content": prompt},
            ],
            extra_body=request.extra_body or {},
        )
    except openai.OpenAIError as exc:
        raise GenerationError(str(exc)) from exc
    if not response.choices:
        # Confirmed live: some upstream providers return a 200 with
        # `choices: null`/`[]` instead of an error (observed via
        # OpenRouter) - most often a silent content-filter refusal.
        # Treat it the same as any other unusable response rather than
        # crashing with an unhandled TypeError on the [0] index.
        raise GenerationError(
            "model returned no choices (likely a silent content filter)"
        )
    return (response.choices[0].message.content or "").strip()


def _decode_next(
    decoder: json.JSONDecoder, text: str, idx: int
) -> tuple[object, int]:
    try:
        return decoder.raw_decode(text, idx)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"model did not return JSON: {text[:200]}"
        ) from exc


def _parse_json_arrays(text: str) -> list[object]:
    """Parse one or more whitespace-separated JSON array literals.

    Tolerates a model returning N separate 1-item arrays instead of
    one N-item array (observed with deepseek-v4.1-flash).
    """
    decoder = json.JSONDecoder()
    items: list[object] = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        value, idx = _decode_next(decoder, text, idx)
        items.extend(value if isinstance(value, list) else [value])
    return items


def _parse_candidates(text: str) -> list[str]:
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    candidates = _parse_json_arrays(text)
    return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]


def _validate(api_key: str, exemplars: list[str]) -> None:
    if not api_key:
        raise GenerationError("no API key configured")
    if not exemplars:
        raise GenerationError("no exemplars supplied")


def generate_candidates(
    api_key: str,
    style_guide: str,
    exemplars: list[str],
    context: str,
    count: int,
    model: str,
    seed_count: int = 20,
    base_url: str = DEFAULT_BASE_URL,
    extra_body: dict[str, object] | None = None,
) -> list[str]:
    """Return up to `count` distinct candidate posts about `context`.

    See the module docstring for why `model` has no default.
    """
    _validate(api_key, exemplars)
    prompt = _build_prompt(exemplars, context, count, seed_count)
    request = _Request(api_key, base_url, model, extra_body)
    text = _call_model(request, style_guide, prompt)
    return _parse_candidates(text)


def _format_history(recent_turns: list[tuple[str, str]] | None) -> str:
    if not recent_turns:
        return "(none yet)"
    return "\n".join(f"- {speaker}: {text}" for speaker, text in recent_turns)


def _build_reply_prompt(
    exemplars: list[str],
    message: str,
    author: str,
    recent_turns: list[tuple[str, str]] | None,
    k: int,
) -> str:
    seeds = random.sample(exemplars, min(k, len(exemplars)))
    return _REPLY_INSTRUCTION.format(
        k=len(seeds),
        exemplars="\n".join(f"- {s}" for s in seeds),
        history=_format_history(recent_turns),
        message=message,
        author=author,
    )


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def generate_reply(
    api_key: str,
    style_guide: str,
    exemplars: list[str],
    message: str,
    author: str,
    model: str,
    recent_turns: list[tuple[str, str]] | None = None,
    seed_count: int = 20,
    base_url: str = DEFAULT_BASE_URL,
    extra_body: dict[str, object] | None = None,
) -> str:
    """Return one reply to `message`, in the measured voice.

    `recent_turns` is a `(speaker, text)` list, oldest first - short-
    term conversational context, distinct from any durable memory the
    caller folds into `style_guide` itself. Renders as a `(none yet)`
    placeholder when empty, so a conversation's first turn never has a
    missing/malformed history block.
    """
    _validate(api_key, exemplars)
    prompt = _build_reply_prompt(
        exemplars, message, author, recent_turns, seed_count
    )
    request = _Request(api_key, base_url, model, extra_body)
    return _strip_quotes(_call_model(request, style_guide, prompt))
