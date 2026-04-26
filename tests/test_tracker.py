from unittest.mock import MagicMock, patch

import pytest

from src.config.response import SentimentResponse
from src.tracker import LangfuseTracker, load_prompt


class TestLoadPrompt:
    # load_prompt
    @patch("src.tracker.get_client")
    @patch("src.tracker.sync_prompt_to_langfuse")
    def test_calls_sync_and_get(self, mock_sync, mock_get_client, monkeypatch):
        monkeypatch.setenv("PROMPT_NAME", "imdb_review")
        client = MagicMock()
        mock_get_client.return_value = client
        prompt_obj = MagicMock()
        client.get_prompt.return_value = prompt_obj

        result = load_prompt()

        mock_sync.assert_called_once()
        client.get_prompt.assert_called_once_with(
            "imdb_review", type="text", label="dev"
        )
        assert result is prompt_obj

    @patch("src.tracker.get_client")
    @patch("src.tracker.sync_prompt_to_langfuse")
    def test_custom_label(self, mock_sync, mock_get_client, monkeypatch):
        monkeypatch.setenv("PROMPT_NAME", "imdb_review")
        client = MagicMock()
        mock_get_client.return_value = client

        load_prompt(label="production")

        client.get_prompt.assert_called_once_with(
            "imdb_review", type="text", label="production"
        )


class TestTraceLlmCall:
    # LangfuseTracker.trace_llm_call
    def _setup(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        generation = MagicMock()
        client.start_as_current_observation.return_value.__enter__ = MagicMock(
            return_value=generation
        )
        client.start_as_current_observation.return_value.__exit__ = MagicMock(
            return_value=False
        )
        return client, generation

    @patch("src.tracker.get_client")
    def test_records_sentiment(self, mock_get_client):
        client, generation = self._setup(mock_get_client)
        tracker = LangfuseTracker(session_id="s")
        response = SentimentResponse(sentiment="positive", reason="great film")
        result = tracker.trace_llm_call(
            lambda: response,
            name="sentiment-evaluation",
            model="gpt-4o",
            input=[{"role": "user", "content": "loved it"}],
        )

        client.start_as_current_observation.assert_called_once_with(
            as_type="generation",
            name="sentiment-evaluation",
            model="gpt-4o",
            input=[{"role": "user", "content": "loved it"}],
            prompt=None,
        )
        generation.update.assert_called_once_with(output="positive")
        assert result is response

    @patch("src.tracker.get_client")
    def test_links_prompt(self, mock_get_client):
        client, _ = self._setup(mock_get_client)
        prompt_obj = MagicMock()
        tracker = LangfuseTracker(session_id="s")
        tracker.trace_llm_call(
            lambda: SentimentResponse(sentiment="negative", reason="boring"),
            name="sentiment-evaluation",
            model="gpt-4o",
            input=[],
            prompt=prompt_obj,
        )

        _, kwargs = client.start_as_current_observation.call_args
        assert kwargs["prompt"] is prompt_obj

    @patch("src.tracker.get_client")
    def test_plain_string_fallback(self, mock_get_client):
        _, generation = self._setup(mock_get_client)
        tracker = LangfuseTracker(session_id="s")
        result = tracker.trace_llm_call(
            lambda: "some string", name="test", model="gpt-4o", input=[]
        )

        generation.update.assert_called_once_with(output="some string")
        assert result == "some string"

    @patch("src.tracker.get_client")
    def test_surfaces_exception_to_langfuse(self, mock_get_client):
        _, generation = self._setup(mock_get_client)
        tracker = LangfuseTracker(session_id="s")
        with pytest.raises(ValueError, match="boom"):
            tracker.trace_llm_call(
                lambda: (_ for _ in ()).throw(ValueError("boom")),
                name="test",
                model="gpt-4o",
                input=[],
            )

        generation.update.assert_called_once_with(level="ERROR", status_message="boom")


class TestScoreCorrectness:
    # LangfuseTracker.score_correctness
    @patch("src.tracker.get_client")
    def test_correct(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        tracker = LangfuseTracker(session_id="s")
        span = MagicMock()
        tracker.score_correctness(span, expected="positive", predicted="positive")
        span.score_trace.assert_called_once_with(name="correct", value=1)

    @patch("src.tracker.get_client")
    def test_wrong(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        tracker = LangfuseTracker(session_id="s")
        span = MagicMock()
        tracker.score_correctness(span, expected="positive", predicted="negative")
        span.score_trace.assert_called_once_with(name="correct", value=0)


class TestScoreMetrics:
    # LangfuseTracker.score_metrics
    @patch("src.tracker.get_client")
    def test_empty_returns_empty_dict(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        tracker = LangfuseTracker(session_id="s")

        result = tracker.score_metrics([], [])

        assert result == {}
        client.create_score.assert_not_called()

    @patch("src.tracker.get_client")
    def test_correct_values(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        tracker = LangfuseTracker(session_id="s")

        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]

        result = tracker.score_metrics(y_true, y_pred)

        assert result["accuracy"] == pytest.approx(0.75)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.5)
        assert "f1" in result

    @patch("src.tracker.get_client")
    def test_creates_scores_in_langfuse(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        tracker = LangfuseTracker(session_id="my-session")

        y_true = ["positive", "negative"]
        y_pred = ["positive", "negative"]

        tracker.score_metrics(y_true, y_pred)

        score_names = {c.kwargs["name"] for c in client.create_score.call_args_list}
        assert score_names == {"accuracy", "precision", "recall", "f1"}
        for c in client.create_score.call_args_list:
            assert c.kwargs["session_id"] == "my-session"
