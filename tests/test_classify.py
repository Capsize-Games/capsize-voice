from unittest.mock import MagicMock, patch

from capsize_voice.classify import should_interject


def _mock_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("capsize_voice.generate.openai.OpenAI")
def test_should_interject_true_on_interject(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("INTERJECT")
    )

    result = should_interject(
        "key", "what does everyone think?", "alice", None, model="m"
    )

    assert result is True


@patch("capsize_voice.generate.openai.OpenAI")
def test_should_interject_false_on_silent(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("SILENT")
    )

    result = should_interject("key", "lol same", "bob", None, model="m")

    assert result is False


@patch("capsize_voice.generate.openai.OpenAI")
def test_should_interject_defaults_to_silent_on_malformed_output(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("uh, maybe? not sure")
    )

    result = should_interject("key", "hmm", "bob", None, model="m")

    assert result is False


@patch("capsize_voice.generate.openai.OpenAI")
def test_should_interject_sends_low_max_tokens_and_zero_temperature(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("SILENT")
    )

    should_interject("key", "hi", "alice", None, model="m")

    _, kwargs = mock_client_cls.return_value.chat.completions.create.call_args
    assert kwargs["extra_body"]["max_tokens"] == 8
    assert kwargs["extra_body"]["temperature"] == 0


@patch("capsize_voice.generate.openai.OpenAI")
def test_should_interject_includes_recent_turns_in_prompt(
    mock_client_cls: MagicMock,
) -> None:
    mock_client_cls.return_value.chat.completions.create.return_value = (
        _mock_response("SILENT")
    )

    should_interject(
        "key", "same here", "bob", [("alice", "I love pizza")], model="m"
    )

    _, kwargs = mock_client_cls.return_value.chat.completions.create.call_args
    prompt = kwargs["messages"][1]["content"]
    assert "alice: I love pizza" in prompt
