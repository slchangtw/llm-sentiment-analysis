.DEFAULT_GOAL := help

## run_experiment: Run sentiment experiment against a Langfuse dataset.
##   Usage: make run_experiment DATASET=<dataset_name>
## create_dataset: Upload IMDB samples (1000/1000, seed 42) to Langfuse.
##   Usage: make create_dataset DATASET=<dataset_name>  [SEED=<int>]
## test: Run full test suite.
## test-no-integration: Run tests excluding API integration tests.
SEED_FLAG := $(if $(SEED),--seed $(SEED),)

.PHONY: help run_experiment create_dataset test test-no-integration

help:
	@sed -n '/^##/s/^## //p' Makefile
run_experiment:
	uv run python -m src $(DATASET)

create_dataset:
	uv run python -m src.create_dataset $(DATASET) $(SEED_FLAG)

test:
	uv run pytest

test-no-integration:
	uv run pytest -m "not integration"
