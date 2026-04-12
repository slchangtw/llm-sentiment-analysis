"""Integration tests for SentimentEvaluator — require OPENROUTER_API_KEY."""

import pytest

from src.config.model import load_model_configs
from src.config.response import SentimentResponse
from src.llm import SentimentEvaluator

POSITIVE_REVIEW = "This movie was absolutely fantastic. The acting was superb and I loved every minute of it."
NEGATIVE_REVIEW = "Terrible film. Boring plot, bad acting, and a complete waste of time."


@pytest.fixture(scope="module")
def evaluator() -> SentimentEvaluator:
    config = load_model_configs()[0]
    return SentimentEvaluator(config)


def test_positive_review_sentiment(evaluator: SentimentEvaluator) -> None:
    result = evaluator.evaluate(POSITIVE_REVIEW)
    assert isinstance(result, SentimentResponse)
    assert result.sentiment == "positive"
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_negative_review_sentiment(evaluator: SentimentEvaluator) -> None:
    result = evaluator.evaluate(NEGATIVE_REVIEW)
    assert isinstance(result, SentimentResponse)
    assert result.sentiment == "negative"
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0
