"""Decide whether a persona should interject in an ambient conversation.

A sibling to `generate.generate_reply`/`memory.extract_facts`, sharing
their `_Request`/`_call_model` plumbing. Deliberately cheap: this runs
on every ordinary message in a channel the bot is merely present in,
not just the ones addressed to it, so it must never be routed through
the full voice-generation call.
"""

from capsize_voice.generate import DEFAULT_BASE_URL, _call_model, _Request

_SYSTEM_PROMPT = (
    "You decide whether a participant in a group conversation should "
    "speak up. Most ordinary chatter between other people should be "
    "left alone, the same way a real person in the room wouldn't "
    "comment on every single thing said. Only choose to interject "
    "when the message is a direct question to the group, a strong "
    "opinion worth responding to, or something the participant would "
    "clearly have something to add to."
)

_INSTRUCTION = """\
Recent conversation so far, oldest first:
<history>
{history}
</history>

{author} just said:
<message>
{message}
</message>

Should the participant interject? Reply with exactly one word: \
INTERJECT or SILENT. No other text.
"""


def _format_history(recent_turns: list[tuple[str, str]] | None) -> str:
    if not recent_turns:
        return "(none yet)"
    return "\n".join(f"- {speaker}: {text}" for speaker, text in recent_turns)


def _build_prompt(
    message: str, author: str, recent_turns: list[tuple[str, str]] | None
) -> str:
    return _INSTRUCTION.format(
        history=_format_history(recent_turns), author=author, message=message
    )


def _parse_decision(text: str) -> bool:
    return text.strip().upper().startswith("INTERJECT")


def should_interject(
    api_key: str,
    message: str,
    author: str,
    recent_turns: list[tuple[str, str]] | None,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    extra_body: dict[str, object] | None = None,
) -> bool:
    """Return whether the persona should reply to `message`.

    Cheap by construction: low `max_tokens`, `temperature=0`, a
    single-word output parsed by exact prefix match - no JSON, no
    retry loop. Any malformed/unexpected output is treated as SILENT
    (`_parse_decision` only returns True on an explicit INTERJECT
    prefix), never crashing the caller's ingestion loop.
    """
    body: dict[str, object] = {"temperature": 0, "max_tokens": 8}
    body.update(extra_body or {})
    prompt = _build_prompt(message, author, recent_turns)
    request = _Request(api_key, base_url, model, body)
    text = _call_model(request, _SYSTEM_PROMPT, prompt)
    return _parse_decision(text)
