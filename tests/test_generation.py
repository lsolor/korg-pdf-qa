"""PRD R2: stream grounded answers from Claude with valid request parameters."""

import anthropic
import pytest

import main

QUESTION = "How long do we keep customer data?"
CHUNKS = [
    {
        "text": "Customer data must be retained for a minimum of 7 years.",
        "source": "data_governance.pdf",
    }
]


def test_streaming_prints_the_answer_and_returns_the_full_text(fake_claude, capsys):
    fake_claude.respond("At least ", "7 years.")

    answer = main.generate_with_context(QUESTION, CHUNKS, stream=True)

    assert answer == "At least 7 years."
    assert "At least 7 years." in capsys.readouterr().out


def test_non_streaming_returns_the_answer_text_when_thinking_comes_first(
    fake_claude,
):
    fake_claude.respond("At least 7 years.", thinking_first=True)

    answer = main.generate_with_context(QUESTION, CHUNKS)

    assert answer == "At least 7 years."


def test_an_answer_cut_off_by_the_token_limit_is_flagged(fake_claude):
    fake_claude.respond("Customer data must be retained for", stop_reason="max_tokens")

    answer = main.generate_with_context(QUESTION, CHUNKS)

    assert answer.startswith("Customer data must be retained for")
    assert "cut off" in answer.lower()


def test_a_rejected_api_key_raises_an_error_that_names_the_setting(fake_claude):
    fake_claude.fail_with(anthropic.AuthenticationError, 401)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main.generate_with_context(QUESTION, CHUNKS)
