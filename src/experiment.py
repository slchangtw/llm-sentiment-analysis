import os
import time

from tqdm import tqdm

from src.config.model import load_default_model_config
from src.dataset import get_dataset
from src.llm import SentimentEvaluator
from src.tracker import LangfuseTracker, get_langfuse, load_prompt


def run_experiment(dataset_name: str) -> dict[str, float]:
    trace_user_id = os.environ["LANGFUSE_USER_ID"]
    config = load_default_model_config()
    evaluator = SentimentEvaluator(config)
    prompt = load_prompt()
    dataset = get_dataset(dataset_name)

    session_id = f"imdb_review_{int(time.time())}"
    tracker = LangfuseTracker(session_id=session_id)
    y_true: list[str] = []
    y_pred: list[str] = []

    for item in tqdm(dataset.items, desc="Evaluating"):
        with item.run(
            run_name=session_id,
            run_metadata={"model": config.name},
        ) as root_span:
            review = item.input["review"]
            root_span.update_trace(
                session_id=session_id,
                user_id=trace_user_id,
                input=review,
            )
            try:
                result = tracker.trace_llm_call(
                    lambda r=review: evaluator.evaluate(r),
                    name="sentiment-evaluation",
                    model=config.name,
                    prompt=prompt,
                )
                root_span.update_trace(user_id=trace_user_id, output=result.sentiment)
                y_true.append(item.expected_output.strip().lower())
                y_pred.append(result.sentiment.strip().lower())
                tracker.score_correctness(
                    root_span,
                    expected=item.expected_output,
                    predicted=result.sentiment,
                )
            except Exception as e:
                root_span.update_trace(
                    user_id=trace_user_id,
                    level="ERROR",
                    status_message=str(e),
                )

    metrics = tracker.score_metrics(y_true, y_pred)
    get_langfuse().flush()
    return metrics
