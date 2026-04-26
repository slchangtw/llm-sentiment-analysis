"""Integration tests for SentimentEvaluator — require OPENROUTER_API_KEY."""

import pytest

from src.config.model import load_default_model_config
from src.config.response import SentimentResponse
from src.llm import SentimentEvaluator

pytestmark = pytest.mark.integration

POSITIVE_REVIEW = "This movie was absolutely fantastic. The acting was superb and I loved every minute of it."
NEGATIVE_REVIEW = "Terrible film. Boring plot, bad acting, and a complete waste of time."


@pytest.fixture(scope="module")
def evaluator() -> SentimentEvaluator:
    return SentimentEvaluator(load_default_model_config())


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
