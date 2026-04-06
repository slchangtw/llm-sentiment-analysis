"""CLI: upload reviews CSV to Langfuse dataset."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

from .config.dataset_schema import (  # noqa: E402
    DATASET_EXPECTED_OUTPUT_SCHEMA,
    DATASET_INPUT_SCHEMA,
)
from .dataset import CSV_PATH, langfuse, load_reviews, metadata_for_row  # noqa: E402


def main(*, dataset_name: str, seed: int = 42) -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    langfuse.create_dataset(
        name=dataset_name,
        description="IMDB reviews: input.review, expected_output sentiment label, metadata.length (chars)",
        input_schema=DATASET_INPUT_SCHEMA,
        expected_output_schema=DATASET_EXPECTED_OUTPUT_SCHEMA,
    )

    df = load_reviews(CSV_PATH, n_per_class=1000, seed=seed)
    for _, row in df.iterrows():
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"review": row["review"]},
            expected_output=str(row["sentiment"]).strip().lower(),
            metadata=metadata_for_row(row),
        )

    print(f"Added {len(df)} items to '{dataset_name}' (seed={seed}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload reviews CSV to Langfuse.")
    parser.add_argument("dataset_name", help="Langfuse dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()
    main(dataset_name=args.dataset_name, seed=args.seed)
