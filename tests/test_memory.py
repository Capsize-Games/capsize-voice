from unittest.mock import MagicMock, patch

import pytest

from capsize_voice.generate import GenerationError
from capsize_voice.memory import extract_facts


def _mock_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("capsize_voice.generate.openai.OpenAI")
def test_extract_facts_returns_parsed_list(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response('["lives in Austin", "has a dog named Rex"]')
    )

    result = extract_facts(
        "key", "I just moved to Austin with my dog Rex", "alice", [],
        model="m",
    )

    assert result == ["lives in Austin", "has a dog named Rex"]


@patch("capsize_voice.generate.openai.OpenAI")
def test_extract_facts_returns_empty_when_nothing_new(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("[]")
    )

    result = extract_facts(
        "key", "cool", "alice", ["lives in Austin"], model="m"
    )

    assert result == []


@patch("capsize_voice.generate.openai.OpenAI")
def test_extract_facts_passes_known_facts_into_prompt(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("[]")
    )

    extract_facts("key", "hi again", "alice", ["lives in Austin"], model="m")

    _, kwargs = (
        mock_client_cls.return_value.chat.completions.create.call_args
    )
    user_prompt = kwargs["messages"][1]["content"]
    assert "lives in Austin" in user_prompt


@patch("capsize_voice.generate.openai.OpenAI")
def test_extract_facts_prompt_tells_model_to_use_the_real_name(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("[]")
    )

    extract_facts("key", "hi", "alice", [], model="m")

    _, kwargs = (
        mock_client_cls.return_value.chat.completions.create.call_args
    )
    system_prompt = kwargs["messages"][0]["content"]
    user_prompt = kwargs["messages"][1]["content"]
    assert "never as" in system_prompt.lower()
    assert 'the name "alice"' in user_prompt


@patch("capsize_voice.generate.openai.OpenAI")
def test_extract_facts_rejects_non_json(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("not json at all")
    )

    with pytest.raises(GenerationError):
        extract_facts("key", "hi", "alice", [], model="m")


def test_extract_facts_requires_api_key() -> None:
    with pytest.raises(GenerationError):
        extract_facts("", "hi", "alice", [], model="m")
