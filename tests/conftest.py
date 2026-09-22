"""Shared fixtures. Nothing here touches the robot, the network or real credentials."""

from datetime import datetime, timedelta

import pytest

from kin import memory as memory_module
from kin.memory import Memory


def _no_embeddings(text: str, is_query: bool):
    raise ConnectionError("offline")  # memory must fall back to keyword search


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Each test starts with no optional integrations configured and its own empty working directory."""
    for name in ("TELEGRAM_BOT_TOKEN", "KIN_FAMILY_CHAT_IDS", "TAVILY_API_KEY", "KIN_PERSON_NAME",
                 "KIN_REMINDER_ATTEMPTS", "KIN_RENUDGE_MINUTES", "KIN_STAY_ON_CHARGER", "KIN_PET_TO_TALK", "KIN_IDLE_HOLD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def clock(monkeypatch):
    """Memory timestamps have one-second resolution; this makes every write land a second after the last."""
    state = {"now": datetime.now().replace(microsecond=0)}

    def tick() -> str:
        state["now"] += timedelta(seconds=1)
        return state["now"].isoformat(timespec="seconds")

    monkeypatch.setattr(memory_module, "_now", tick)
    return state


@pytest.fixture
def memory(tmp_path, clock) -> Memory:
    return Memory(tmp_path / "kin.db", embed=_no_embeddings)


class FakeFamily:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, text: str, chat_id: int | None = None) -> int:
        self.sent.append(text)
        return 1


@pytest.fixture
def family() -> FakeFamily:
    return FakeFamily()
