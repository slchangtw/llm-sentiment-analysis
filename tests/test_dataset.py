"""Tests for src/dataset.py"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

with patch("langfuse.Langfuse"):
    from src.dataset import load_reviews, metadata_for_row


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "review": [f"review_{i}" for i in range(20)],
            "sentiment": ["positive"] * 10 + ["negative"] * 10,
        }
    )
    path = tmp_path / "reviews.csv"
    df.to_csv(path, index=False)
    return path


def test_load_reviews_returns_balanced_classes(sample_csv: Path) -> None:
    df = load_reviews(sample_csv, n_per_class=5, seed=0)
    assert (df["sentiment"] == "positive").sum() == 5
    assert (df["sentiment"] == "negative").sum() == 5


def test_load_reviews_total_length(sample_csv: Path) -> None:
    df = load_reviews(sample_csv, n_per_class=5, seed=0)
    assert len(df) == 10


def test_load_reviews_is_shuffled(sample_csv: Path) -> None:
    df = load_reviews(sample_csv, n_per_class=5, seed=42)
    sentiments = df["sentiment"].tolist()
    assert sentiments != sorted(sentiments)


def test_load_reviews_seed_reproducible(sample_csv: Path) -> None:
    df1 = load_reviews(sample_csv, n_per_class=5, seed=7)
    df2 = load_reviews(sample_csv, n_per_class=5, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_reviews_different_seeds_differ(sample_csv: Path) -> None:
    df1 = load_reviews(sample_csv, n_per_class=5, seed=1)
    df2 = load_reviews(sample_csv, n_per_class=5, seed=2)
    assert not df1["review"].tolist() == df2["review"].tolist()


def test_load_reviews_resets_index(sample_csv: Path) -> None:
    df = load_reviews(sample_csv, n_per_class=5, seed=0)
    assert list(df.index) == list(range(len(df)))


def test_metadata_for_row_returns_length() -> None:
    row = pd.Series({"review": "hello", "sentiment": "positive"})
    assert metadata_for_row(row) == {"length": 5}


def test_metadata_for_row_empty_review() -> None:
    row = pd.Series({"review": "", "sentiment": "negative"})
    assert metadata_for_row(row) == {"length": 0}


def test_metadata_for_row_non_string_review() -> None:
    row = pd.Series({"review": 12345, "sentiment": "positive"})
    assert metadata_for_row(row) == {"length": len("12345")}
