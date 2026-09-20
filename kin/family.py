"""The family's side of Kin: a Telegram bot running on the hub.

The hub only makes outbound requests to Telegram (long polling), so no router ports are opened.
Only chats listed in KIN_FAMILY_CHAT_IDS may talk to Kin; anyone else is told their chat id and ignored,
because a message here ends up spoken aloud in someone's home.
"""

import queue
import threading
import time
from dataclasses import dataclass

import httpx

from kin import config

_POLL_SECONDS = 25


@dataclass
class FamilyMessage:
    chat_id: int
    sender: str
    text: str


def available() -> bool:
    return bool(config.telegram_token())


class Family:
    def __init__(self):
        self._url = f"https://api.telegram.org/bot{config.telegram_token()}"
        self._allowed = config.family_chat_ids()
        self.incoming: queue.Queue[FamilyMessage] = queue.Queue()
        self._offset = 0

    def start(self) -> None:
        threading.Thread(target=self._poll_forever, daemon=True, name="Telegram poller").start()

    def send(self, text: str, chat_id: int | None = None) -> int:
        """Send to one chat, or to every family chat. Returns how many were sent."""
        targets = [chat_id] if chat_id else sorted(self._allowed)
        sent = 0
        for target in targets:
            response = httpx.post(f"{self._url}/sendMessage", json={"chat_id": target, "text": text}, timeout=20)
            sent += response.is_success
        return sent

    def _poll_forever(self) -> None:
        while True:
            try:
                self._poll_once()
            except Exception:  # network blips are routine; keep polling
                time.sleep(10)

    def _poll_once(self) -> None:
        response = httpx.get(
            f"{self._url}/getUpdates",
            params={"timeout": _POLL_SECONDS, "offset": self._offset, "allowed_updates": '["message"]'},
            timeout=_POLL_SECONDS + 10,
        )
        response.raise_for_status()
        for update in response.json().get("result", []):
            self._offset = update["update_id"] + 1
            message = update.get("message") or {}
            text, chat = (message.get("text") or "").strip(), message.get("chat") or {}
            if not text or "id" not in chat:
                continue
            if chat["id"] not in self._allowed:
                self.send(
                    f"This chat is not linked to Kin yet. Its id is {chat['id']}. "
                    "Add it to KIN_FAMILY_CHAT_IDS on the hub and restart Kin.",
                    chat_id=chat["id"],
                )
                continue
            sender = (message.get("from") or {}).get("first_name") or "A family member"
            self.incoming.put(FamilyMessage(chat["id"], sender, text))
