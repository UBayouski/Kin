"""Find a relative's Telegram chat id so it can be added to KIN_FAMILY_CHAT_IDS.

Run this, then message the bot from the relative's Telegram account. Stop Kin's service first:
Telegram delivers each message to only one listener.

usage: python scripts/family_pair.py [minutes]
"""

import sys
import time

import httpx

from kin import config

url = f"https://api.telegram.org/bot{config.telegram_token()}"
deadline = time.monotonic() + 60 * float(sys.argv[1] if len(sys.argv) > 1 else 5)
offset = 0
print("Waiting for a message to the bot...", flush=True)

while time.monotonic() < deadline:
    updates = httpx.get(f"{url}/getUpdates", params={"timeout": 20, "offset": offset}, timeout=30).json()
    for update in updates.get("result", []):
        offset = update["update_id"] + 1
        message = update.get("message") or {}
        chat, sender = message.get("chat") or {}, message.get("from") or {}
        if "id" not in chat:
            continue
        print(f"chat id {chat['id']}  from {sender.get('first_name', '?')}  text: {message.get('text', '')!r}", flush=True)
        httpx.post(f"{url}/sendMessage", timeout=20, json={
            "chat_id": chat["id"],
            "text": f"Hello {sender.get('first_name', '')}. Your chat id is {chat['id']}. "
                    "Add it to KIN_FAMILY_CHAT_IDS on the hub to link this chat to Kin.",
        })
