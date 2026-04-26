from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.config.response import SentimentResponse


def _make_item(review: str, expected: str, sentiment: str, raise_exc=None):
    item = MagicMock()
    item.input = {"review": review}
    item.expected_output = expected

    root_span = MagicMock()

    @contextmanager
    def run_ctx(**kwargs):
        yield root_span

    item.run = MagicMock(side_effect=run_ctx)

    if raise_exc:
        item._raise = raise_exc
        item._sentiment = None
    else:
        item._raise = None
        item._sentiment = sentiment

    return item, root_span


@patch("src.experiment.get_langfuse")
@patch("src.experiment.LangfuseTracker")
@patch("src.experiment.load_prompt")
@patch("src.experiment.get_dataset")
@patch("src.experiment.SentimentEvaluator")
@patch("src.experiment.load_default_model_config")
def test_run_experiment_happy_path(
    mock_cfg, mock_evaluator_cls, mock_get_dataset, mock_load_prompt,
    mock_tracker_cls, mock_get_langfuse,
):
    config = MagicMock()
    config.name = "test-model"
    mock_cfg.return_value = config

    evaluator = MagicMock()
    evaluator.evaluate.side_effect = [
        SentimentResponse(sentiment="positive", reason="good"),
        SentimentResponse(sentiment="negative", reason="bad"),
    ]
    mock_evaluator_cls.return_value = evaluator

    item1, span1 = _make_item("great", "positive", "positive")
    item2, span2 = _make_item("awful", "negative", "negative")
    dataset = MagicMock()
    dataset.items = [item1, item2]
    mock_get_dataset.return_value = dataset

    tracker = MagicMock()
    tracker.trace_llm_call.side_effect = [
        SentimentResponse(sentiment="positive", reason="good"),
        SentimentResponse(sentiment="negative", reason="bad"),
    ]
    tracker.score_metrics.return_value = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    mock_tracker_cls.return_value = tracker

    langfuse_client = MagicMock()
    mock_get_langfuse.return_value = langfuse_client

    from src.experiment import run_experiment
    result = run_experiment("my-dataset")

    assert result["accuracy"] == 1.0
    assert tracker.score_correctness.call_count == 2
    sid = mock_tracker_cls.call_args.kwargs["session_id"]
    span1.update_trace.assert_any_call(session_id=sid, user_id="test-user", input="great")
    span2.update_trace.assert_any_call(session_id=sid, user_id="test-user", input="awful")
    langfuse_client.flush.assert_called_once()


@patch("src.experiment.get_langfuse")
@patch("src.experiment.LangfuseTracker")
@patch("src.experiment.load_prompt")
@patch("src.experiment.get_dataset")
@patch("src.experiment.SentimentEvaluator")
@patch("src.experiment.load_default_model_config")
def test_run_experiment_skips_failed_item(
    mock_cfg, mock_evaluator_cls, mock_get_dataset, mock_load_prompt,
    mock_tracker_cls, mock_get_langfuse,
):
    config = MagicMock()
    config.name = "test-model"
    mock_cfg.return_value = config

    mock_evaluator_cls.return_value = MagicMock()

    item1, span1 = _make_item("great", "positive", "positive")
    item2, span2 = _make_item("awful", "negative", "negative")
    dataset = MagicMock()
    dataset.items = [item1, item2]
    mock_get_dataset.return_value = dataset

    err = RuntimeError("LLM failed")
    tracker = MagicMock()
    tracker.trace_llm_call.side_effect = [
        err,
        SentimentResponse(sentiment="negative", reason="bad"),
    ]
    tracker.score_metrics.return_value = {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    mock_tracker_cls.return_value = tracker

    mock_get_langfuse.return_value = MagicMock()

    from src.experiment import run_experiment
    run_experiment("my-dataset")

    span1.update_trace.assert_any_call(
        user_id="test-user",
        level="ERROR",
        status_message="LLM failed",
    )
    assert tracker.score_correctness.call_count == 1
    y_true, y_pred = tracker.score_metrics.call_args.args
    assert y_true == ["negative"]
    assert y_pred == ["negative"]


@patch("src.experiment.get_langfuse")
@patch("src.experiment.LangfuseTracker")
@patch("src.experiment.load_prompt")
@patch("src.experiment.get_dataset")
@patch("src.experiment.SentimentEvaluator")
@patch("src.experiment.load_default_model_config")
def test_run_experiment_session_id_format(
    mock_cfg, mock_evaluator_cls, mock_get_dataset, mock_load_prompt,
    mock_tracker_cls, mock_get_langfuse,
):
    config = MagicMock()
    config.name = "m"
    mock_cfg.return_value = config
    mock_evaluator_cls.return_value = MagicMock()

    dataset = MagicMock()
    dataset.items = []
    mock_get_dataset.return_value = dataset

    tracker = MagicMock()
    tracker.score_metrics.return_value = {}
    mock_tracker_cls.return_value = tracker
    mock_get_langfuse.return_value = MagicMock()

    from src.experiment import run_experiment
    run_experiment("ds")

    session_id = mock_tracker_cls.call_args.kwargs["session_id"]
    assert session_id.startswith("imdb_review_")
    assert session_id[len("imdb_review_"):].isdigit()
