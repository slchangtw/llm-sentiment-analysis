import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


@dataclass
class ModelConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model_params: dict[str, Any]


def load_default_model_config() -> ModelConfig:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    default_name = data["default_model"]
    for m in data["models"]:
        if m["name"] == default_name:
            return ModelConfig(
                name=m["name"],
                provider=m["provider"],
                base_url=_PROVIDER_BASE_URLS[m["provider"]],
                api_key="ollama" if m["provider"] == "ollama" else os.environ["OPENROUTER_API_KEY"],
                model_params=m.get("model_params", {}),
            )
    raise ValueError(f"Default model '{default_name}' not found in config")


def load_model_configs() -> list[ModelConfig]:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return [
        ModelConfig(
            name=m["name"],
            provider=m["provider"],
            base_url=_PROVIDER_BASE_URLS[m["provider"]],
            api_key="ollama" if m["provider"] == "ollama" else os.environ["OPENROUTER_API_KEY"],
            model_params=m.get("model_params", {}),
        )
        for m in data["models"]
    ]
