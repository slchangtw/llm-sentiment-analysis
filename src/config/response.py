from typing import Literal

from pydantic import BaseModel


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative"]
    reason: str
