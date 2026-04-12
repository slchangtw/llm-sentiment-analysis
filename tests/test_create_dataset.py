"""Tests for src/create_dataset.py"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


def _make_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review": [f"review_{i}" for i in range(n)],
            "sentiment": ["positive", "negative"] * (n // 2),
        }
    )


@pytest.fixture
def mock_langfuse():
    with patch("src.dataset.Langfuse") as MockLangfuse:
        mock = MagicMock()
        MockLangfuse.return_value = mock
        yield mock


@patch("src.dataset.Langfuse")
def test_main_raises_when_csv_missing(MockLangfuse: MagicMock, tmp_path: Path) -> None:
    MockLangfuse.return_value = MagicMock()
    from src.create_dataset import main

    with patch("src.dataset.CSV_PATH", tmp_path / "missing.csv"):
        with patch("src.create_dataset.CSV_PATH", tmp_path / "missing.csv"):
            with pytest.raises(FileNotFoundError, match="CSV not found"):
                main(dataset_name="test-ds")


@patch("src.dataset.Langfuse")
def test_main_creates_dataset(MockLangfuse: MagicMock, tmp_path: Path) -> None:
    mock_lf = MagicMock()
    MockLangfuse.return_value = mock_lf

    csv = tmp_path / "reviews.csv"
    df = pd.DataFrame(
        {
            "review": [f"r{i}" for i in range(20)],
            "sentiment": ["positive"] * 10 + ["negative"] * 10,
        }
    )
    df.to_csv(csv, index=False)

    from src.create_dataset import main

    with patch("src.create_dataset.CSV_PATH", csv):
        with patch("src.create_dataset.langfuse", mock_lf):
            with patch("src.create_dataset.load_reviews", return_value=_make_df()) as mock_load:
                main(dataset_name="my-ds", seed=0)

    mock_lf.create_dataset.assert_called_once()
    assert mock_lf.create_dataset.call_args.kwargs["name"] == "my-ds"


@patch("src.dataset.Langfuse")
def test_main_creates_correct_number_of_items(MockLangfuse: MagicMock, tmp_path: Path) -> None:
    mock_lf = MagicMock()
    MockLangfuse.return_value = mock_lf

    csv = tmp_path / "reviews.csv"
    df_data = _make_df(4)
    df_data.to_csv(csv, index=False)

    from src.create_dataset import main

    with patch("src.create_dataset.CSV_PATH", csv):
        with patch("src.create_dataset.langfuse", mock_lf):
            with patch("src.create_dataset.load_reviews", return_value=df_data):
                main(dataset_name="my-ds", seed=0)

    assert mock_lf.create_dataset_item.call_count == 4


@patch("src.dataset.Langfuse")
def test_main_item_input_and_expected_output(MockLangfuse: MagicMock, tmp_path: Path) -> None:
    mock_lf = MagicMock()
    MockLangfuse.return_value = mock_lf

    csv = tmp_path / "reviews.csv"
    df_data = pd.DataFrame({"review": ["great movie"], "sentiment": ["positive"]})
    df_data.to_csv(csv, index=False)

    from src.create_dataset import main

    with patch("src.create_dataset.CSV_PATH", csv):
        with patch("src.create_dataset.langfuse", mock_lf):
            with patch("src.create_dataset.load_reviews", return_value=df_data):
                main(dataset_name="my-ds", seed=0)

    kwargs = mock_lf.create_dataset_item.call_args.kwargs
    assert kwargs["input"] == {"review": "great movie"}
    assert kwargs["expected_output"] == "positive"
    assert kwargs["metadata"] == {"length": len("great movie")}


@patch("src.dataset.Langfuse")
def test_main_normalises_sentiment_case(MockLangfuse: MagicMock, tmp_path: Path) -> None:
    mock_lf = MagicMock()
    MockLangfuse.return_value = mock_lf

    csv = tmp_path / "reviews.csv"
    df_data = pd.DataFrame({"review": ["ok film"], "sentiment": ["Positive"]})
    df_data.to_csv(csv, index=False)

    from src.create_dataset import main

    with patch("src.create_dataset.CSV_PATH", csv):
        with patch("src.create_dataset.langfuse", mock_lf):
            with patch("src.create_dataset.load_reviews", return_value=df_data):
                main(dataset_name="my-ds", seed=0)

    kwargs = mock_lf.create_dataset_item.call_args.kwargs
    assert kwargs["expected_output"] == "positive"
