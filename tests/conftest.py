import pytest


@pytest.fixture(autouse=True)
def langfuse_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_USER_ID", "test-user")
