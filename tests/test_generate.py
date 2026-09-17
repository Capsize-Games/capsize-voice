import json
from unittest.mock import MagicMock, patch

import pytest

from capsize_voice.generate import GenerationError, generate_candidates


def _mock_response(candidates: list[str]) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(candidates)
    response = MagicMock()
    response.content = [block]
    return response


@patch("capsize_voice.generate.anthropic.Anthropic")
def test_generate_candidates_returns_parsed_list(
    mock_client_cls: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response(
        ["post one", "post two"]
    )
    mock_client_cls.return_value = mock_client

    result = generate_candidates(
        "key", "style guide text", ["example one", "example two"], "ctx", 2
    )

    assert result == ["post one", "post two"]


def test_generate_candidates_requires_api_key() -> None:
    with pytest.raises(GenerationError):
        generate_candidates("", "style", ["ex"], "ctx", 2)


def test_generate_candidates_requires_exemplars() -> None:
    with pytest.raises(GenerationError):
        generate_candidates("key", "style", [], "ctx", 2)


@patch("capsize_voice.generate.anthropic.Anthropic")
def test_generate_candidates_rejects_non_json(
    mock_client_cls: MagicMock,
) -> None:
    block = MagicMock()
    block.type = "text"
    block.text = "not json at all"
    response = MagicMock()
    response.content = [block]
    mock_client_cls.return_value.messages.create.return_value = response

    with pytest.raises(GenerationError):
        generate_candidates("key", "style", ["ex"], "ctx", 2)
