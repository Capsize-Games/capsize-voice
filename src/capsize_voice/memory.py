"""Extract durable facts learned about a conversation partner.

A sibling to `generate.generate_reply`, sharing its `_Request`/
`_call_model` plumbing since both are one-shot OpenAI-compatible calls.
Generic and reusable - no opinion on what a "fact" is, how it's stored,
or how it gets folded back into a later prompt; that's the caller's own
memory model, not this module's.
"""

import json

from capsize_voice.generate import (
    DEFAULT_BASE_URL,
    GenerationError,
    _call_model,
    _Request,
)

_SYSTEM_PROMPT = (
    "You track durable facts learned about specific people in a "
    "conversation. Given a message and the facts already known about "
    "its sender, return ONLY new, durable facts learned from this "
    "message - not opinions, not small talk, not anything already "
    "known. Always refer to the sender by their given name, never as "
    "'the author' or 'the user'."
)

_INSTRUCTION = """\
Facts already known about {author}:
<known>
{known}
</known>

{author} just said:
<message>
{message}
</message>

Return ONLY a JSON array of new facts about {author}, each one using \
the name "{author}" rather than "the author" or "the user". Return \
[] if nothing new was learned. No commentary, no markdown fence.
"""


def _build_prompt(
    message: str, author: str, existing_facts: list[str]
) -> str:
    known = "\n".join(f"- {f}" for f in existing_facts) or "(none yet)"
    return _INSTRUCTION.format(author=author, known=known, message=message)


def _parse_facts(text: str) -> list[str]:
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        facts = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"model did not return JSON: {text[:200]}"
        ) from exc
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


def extract_facts(
    api_key: str,
    message: str,
    author: str,
    existing_facts: list[str],
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    extra_body: dict[str, object] | None = None,
) -> list[str]:
    """Return new durable facts learned about `author` from `message`.

    See `generate.generate_candidates` for why `model` has no default.
    """
    if not api_key:
        raise GenerationError("no API key configured")
    prompt = _build_prompt(message, author, existing_facts)
    request = _Request(api_key, base_url, model, extra_body)
    text = _call_model(request, _SYSTEM_PROMPT, prompt)
    return _parse_facts(text)
