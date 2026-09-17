from unittest.mock import MagicMock, patch

import pytest

from capsize_voice.generate import (
    GenerationError,
    generate_candidates,
    generate_reply,
)


def _mock_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_candidates_returns_parsed_list(
    mock_client_cls: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(
        '["post one", "post two"]'
    )
    mock_client_cls.return_value = mock_client

    result = generate_candidates(
        "key",
        "style guide text",
        ["example one", "example two"],
        "ctx",
        2,
        model="deepseek/deepseek-v4-flash-0731",
    )

    assert result == ["post one", "post two"]


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_candidates_passes_model_and_extra_body(
    mock_client_cls: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("[]")
    mock_client_cls.return_value = mock_client

    generate_candidates(
        "key",
        "style",
        ["ex"],
        "ctx",
        2,
        model="deepseek/deepseek-v4-flash-0731",
        base_url="https://openrouter.ai/api/v1",
        extra_body={"provider": {"order": ["deepinfra"]}},
    )

    mock_client_cls.assert_called_once_with(
        api_key="key", base_url="https://openrouter.ai/api/v1"
    )
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "deepseek/deepseek-v4-flash-0731"
    assert kwargs["extra_body"] == {"provider": {"order": ["deepinfra"]}}


def test_generate_candidates_requires_api_key() -> None:
    with pytest.raises(GenerationError):
        generate_candidates("", "style", ["ex"], "ctx", 2, model="m")


def test_generate_candidates_requires_exemplars() -> None:
    with pytest.raises(GenerationError):
        generate_candidates("key", "style", [], "ctx", 2, model="m")


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_candidates_rejects_non_json(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("not json at all")
    )

    with pytest.raises(GenerationError):
        generate_candidates("key", "style", ["ex"], "ctx", 2, model="m")


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_candidates_merges_split_arrays(
    mock_client_cls: MagicMock,
) -> None:
    """deepseek-v4.1-flash sometimes returns N separate 1-item arrays.

    Instead of one N-item array when asked for N candidates.
    """
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response('["post one"]\n\n["post two"]\n\n["post three"]')
    )

    result = generate_candidates(
        "key", "style", ["ex"], "ctx", 3, model="m"
    )

    assert result == ["post one", "post two", "post three"]


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_reply_returns_single_string(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("that's a fair point")
    )

    result = generate_reply(
        "key", "style", ["ex one", "ex two"], "hey what do you think?",
        "alice", model="m",
    )

    assert result == "that's a fair point"


@patch("capsize_voice.generate.openai.OpenAI")
def test_generate_reply_strips_wrapping_quotes(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response('"quoted reply"')
    )

    result = generate_reply(
        "key", "style", ["ex"], "hi", "alice", model="m"
    )

    assert result == "quoted reply"


def test_generate_reply_requires_api_key() -> None:
    with pytest.raises(GenerationError):
        generate_reply("", "style", ["ex"], "hi", "alice", model="m")
