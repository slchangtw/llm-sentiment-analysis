# LLM Sentiment Analysis

Evaluate LLM performance on sentiment classification using [Langfuse](https://langfuse.com) for dataset management and experiment tracking.

The pipeline classifies IMDB movie reviews as `positive` or `negative` using an LLM, then tracks results in a self-hosted Langfuse instance.

---

## Purpose

The goal is to use Langfuse's experiment tooling to systematically compare prompts, models, and sampling strategies.

---

## Set Up the Development Environment

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync

# Copy and fill in credentials
cp .env.example .env
```

`.env` variables (copy from [`.env.example`](.env.example) and fill in):

| Variable | Description |
|---|---|
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_BASE_URL` | Langfuse API base URL (e.g. `http://localhost:3000` for local Docker) |
| `LANGFUSE_TRACING_ENVIRONMENT` | Langfuse tracing environment tag (e.g. `dev`, `staging`; see [Langfuse environments](https://langfuse.com/docs/observability/features/environments)) |
| `LANGFUSE_USER_ID` | User id attached to experiment traces in Langfuse |
| `OPENROUTER_API_KEY` | OpenRouter API key for the LLM |
| `PROMPT_NAME` | Langfuse prompt name used for evaluation (must exist or be synced in the project) |

**Make commands:** run `make help` to list targets and usage (from the [`Makefile`](Makefile)).

**Upload the dataset** (samples 1000 positive + 1000 negative, seed 42 by default):

```bash
make create_dataset DATASET=<dataset_name>
```

**Run the experiment** (Langfuse reachable, `.env` set, dataset uploaded, prompt `PROMPT_NAME` present in Langfuse):

```bash
make run_experiment DATASET=<dataset_name>
```

---

## Launch Langfuse

Langfuse runs locally via Docker Compose. The stack includes Postgres, ClickHouse, Redis, and MinIO.

```bash
docker compose -f infra/langfuse/docker-compose.yml up -d
```

Once healthy, open [http://localhost:3000](http://localhost:3000) and create a project to obtain your API keys.

**Stop the stack:**

```bash
docker compose -f infra/langfuse/docker-compose.yml down
```

> Before running in production, replace all `# CHANGEME` secrets in `infra/langfuse/docker-compose.yml` (database passwords, Redis auth, encryption key, MinIO credentials).
