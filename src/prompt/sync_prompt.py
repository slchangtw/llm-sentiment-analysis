import os

from dotenv import load_dotenv
from langfuse import get_client

from src.prompt import PROMPT

load_dotenv()


def sync_prompt_to_langfuse() -> bool:
    prompt_name = os.environ["PROMPT_NAME"]
    langfuse = get_client()
    try:
        remote = langfuse.get_prompt(prompt_name, type="text", label="dev")
        if remote.prompt.strip() == PROMPT.strip():
            return False
    except Exception:
        pass
    langfuse.create_prompt(
        name=prompt_name,
        type="text",
        prompt=PROMPT,
        labels=["dev"],
    )
    return True
