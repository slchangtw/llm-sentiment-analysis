import os
from typing import Callable, TypeVar

from langfuse import get_client
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.prompt.sync_prompt import sync_prompt_to_langfuse

T = TypeVar("T")


def get_langfuse():
    return get_client()


def load_prompt(label: str = "dev"):
    sync_prompt_to_langfuse()
    return get_langfuse().get_prompt(
        os.environ["PROMPT_NAME"],
        type="text",
        label=label,
    )


class LangfuseTracker:
    def __init__(self, session_id: str):
        self.langfuse = get_langfuse()
        self.session_id = session_id

    def trace_llm_call(
        self,
        fn: Callable[[], T],
        *,
        name: str,
        model: str,
        prompt=None,
    ) -> T:
        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            prompt=prompt,
        ) as generation:
            try:
                result = fn()
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                raise
        return result

    def score_correctness(self, span, *, expected: str, predicted: str) -> None:
        span.score_trace(name="correct", value=int(expected == predicted))

    def score_metrics(self, y_true: list[str], y_pred: list[str]) -> dict[str, float]:
        if not y_pred:
            return {}
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label="positive", zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label="positive", zero_division=0),
            "f1": f1_score(y_true, y_pred, pos_label="positive", zero_division=0),
        }
        for metric_name, value in metrics.items():
            self.langfuse.create_score(
                session_id=self.session_id,
                name=metric_name,
                value=float(value),
            )
        return metrics
