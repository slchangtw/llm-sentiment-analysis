import re

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.model import ModelConfig
from src.config.response import SentimentResponse
from src.prompt import PROMPT

load_dotenv()


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text


class SentimentEvaluator:
    def __init__(self, config: ModelConfig):
        self.model = config.name
        self.model_params = config.model_params
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=10, max=120),
        stop=stop_after_attempt(6),
    )
    def evaluate(self, review: str) -> SentimentResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": review},
            ],
            **self.model_params,
        )
        content = response.choices[0].message.content
        return SentimentResponse.model_validate_json(_extract_json(content))
