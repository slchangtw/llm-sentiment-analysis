from typing import Any

DATASET_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"review": {"type": "string", "minLength": 1}},
    "required": ["review"],
    "additionalProperties": False,
}
DATASET_EXPECTED_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": ["positive", "negative"],
}
