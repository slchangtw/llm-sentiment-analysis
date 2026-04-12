from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass
class ModelConfig:
    name: str
    provider: str
    base_url: str
    model_params: dict[str, Any]


def load_model_configs() -> list[ModelConfig]:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return [
        ModelConfig(
            name=m["name"],
            provider=m["provider"],
            base_url=_PROVIDER_BASE_URLS[m["provider"]],
            model_params=m.get("model_params", {}),
        )
        for m in data["models"]
    ]
