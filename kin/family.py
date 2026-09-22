"""The family's side of Kin: a Telegram bot running on the hub.

The hub only makes outbound requests to Telegram (long polling), so no router ports are opened.
Incoming messages go straight into a durable inbox in the database and are marked handled only after
Kin has dealt with them, so a restart, crash or power cut cannot lose one.
Only chats listed in KIN_FAMILY_CHAT_IDS may talk to Kin; anyone else is told their chat id and ignored,
because a message here ends up spoken aloud in someone's home.
"""

import threading
import time

import httpx

from kin import config
from kin.memory import Memory

_POLL_SECONDS = 25


def available() -> bool:
    return bool(config.telegram_token())


class Family:
    def __init__(self):
        self._url = f"https://api.telegram.org/bot{config.telegram_token()}"
        self._allowed = config.family_chat_ids()
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
        inbox = Memory()  # this thread's own database connection; the service reads the inbox through its own
        while True:
            try:
                self._poll_once(inbox)
            except Exception:  # network blips are routine; keep polling
                time.sleep(10)

    def _poll_once(self, inbox: Memory) -> None:
        response = httpx.get(
            f"{self._url}/getUpdates",
            params={"timeout": _POLL_SECONDS, "offset": self._offset, "allowed_updates": '["message"]'},
            timeout=_POLL_SECONDS + 10,
        )
        response.raise_for_status()
        for update in response.json().get("result", []):
            self._handle_update(update, inbox)
            self._offset = update["update_id"] + 1  # only now is it acknowledged to Telegram on the next poll

    def _handle_update(self, update: dict, inbox: Memory) -> None:
        message = update.get("message") or {}
        text, chat = (message.get("text") or "").strip(), message.get("chat") or {}
        if not text or "id" not in chat:
            return
        if chat["id"] not in self._allowed:
            self.send(
                f"This chat is not linked to Kin yet. Its id is {chat['id']}. "
                "Add it to KIN_FAMILY_CHAT_IDS on the hub and restart Kin.",
                chat_id=chat["id"],
            )
            return
        sender = (message.get("from") or {}).get("first_name") or "A family member"
        # Saved before the next poll acknowledges it to Telegram, so it survives a restart.
        inbox.inbox_add(update["update_id"], chat["id"], sender, text)
