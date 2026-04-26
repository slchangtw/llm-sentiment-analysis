import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment experiment against a Langfuse dataset.")
    parser.add_argument("dataset_name", help="Langfuse dataset name")
    args = parser.parse_args()
    result = run_experiment(args.dataset_name)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
