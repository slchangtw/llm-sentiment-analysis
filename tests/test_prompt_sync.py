from unittest.mock import MagicMock, patch

import pytest

from src.prompt import PROMPT
from src.prompt.sync_prompt import sync_prompt_to_langfuse


@patch("src.prompt.sync_prompt.get_client")
def test_creates_when_remote_missing(mock_get_client, monkeypatch):
    monkeypatch.setenv("PROMPT_NAME", "imdb_review")
    client = MagicMock()
    mock_get_client.return_value = client
    client.get_prompt.side_effect = Exception("not found")

    result = sync_prompt_to_langfuse()

    assert result is True
    client.create_prompt.assert_called_once_with(
        name="imdb_review",
        type="text",
        prompt=PROMPT,
        labels=["dev"],
    )


@patch("src.prompt.sync_prompt.get_client")
def test_creates_when_remote_differs(mock_get_client, monkeypatch):
    monkeypatch.setenv("PROMPT_NAME", "imdb_review")
    client = MagicMock()
    mock_get_client.return_value = client
    remote = MagicMock()
    remote.prompt = "old prompt text"
    client.get_prompt.return_value = remote

    result = sync_prompt_to_langfuse()

    assert result is True
    client.create_prompt.assert_called_once_with(
        name="imdb_review",
        type="text",
        prompt=PROMPT,
        labels=["dev"],
    )


@patch("src.prompt.sync_prompt.get_client")
def test_skips_when_remote_matches(mock_get_client, monkeypatch):
    monkeypatch.setenv("PROMPT_NAME", "imdb_review")
    client = MagicMock()
    mock_get_client.return_value = client
    remote = MagicMock()
    remote.prompt = PROMPT
    client.get_prompt.return_value = remote

    result = sync_prompt_to_langfuse()

    assert result is False
    client.create_prompt.assert_not_called()
