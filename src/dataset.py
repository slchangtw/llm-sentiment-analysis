"""Dataset helpers: paths, CSV loading, Langfuse client."""

from pathlib import Path
from typing import Any

import pandas as pd

from langfuse import Langfuse

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "reviews.csv"

langfuse = Langfuse()


def load_reviews(path: Path, n_per_class: int = 1000, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    pos = df[df["sentiment"] == "positive"].sample(n=n_per_class, random_state=seed)
    neg = df[df["sentiment"] == "negative"].sample(n=n_per_class, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def metadata_for_row(row: pd.Series) -> dict[str, Any]:
    return {"length": len(str(row["review"]))}


def get_dataset(name: str):
    return langfuse.get_dataset(name)
