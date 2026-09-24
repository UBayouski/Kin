"""Kin's dealings with wire-pod, the server behind Vector's own voice system.

Two jobs:
  link / unlink   point wire-pod's fallback for unrecognised speech at Kin's gateway (and put it back)
  IntentWatcher   notice what the person says to Vector himself and how he reacts ("I love you", a fist
                  bump, "go to sleep"). Those stay Vector's to handle; Kin only wants to know they happened:
                  they are signs the person is around, and part of the day the family hears about.
"""

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

from kin import config
from kin.memory import Memory

BACKUP = Path("data/wirepod_kg_backup.json")

# What a built-in intent means, in words Kin and the family can use. Anything not listed is recorded by name.
REACTIONS = {
    "intent_imperative_love": "told Vector they love him",
    "intent_imperative_praise": "praised Vector (good robot)",
    "intent_imperative_scold": "scolded Vector",
    "intent_imperative_apologize": "apologised to Vector",
    "intent_greeting_hello": "said hello to Vector",
    "intent_greeting_goodmorning": "said good morning to Vector",
    "intent_greeting_goodnight": "said good night to Vector",
    "intent_greeting_goodbye": "said goodbye to Vector",
    "intent_play_fistbump": "asked Vector for a fist bump",
    "intent_imperative_dance": "asked Vector to dance",
    "intent_imperative_come": "called Vector over",
    "intent_imperative_lookatme": "asked Vector to look at them",
    "intent_photo_take_extend": "asked Vector to take a photo",
    "intent_clock_time": "asked Vector the time",
    "intent_clock_settimer_extend": "set a timer with Vector",
    "intent_weather_extend": "asked Vector about the weather",
    "intent_names_ask": "asked Vector their name",
    "intent_character_age": "asked Vector how old he is",
    "intent_system_sleep": "told Vector to go to sleep",
    "intent_system_charger": "told Vector to go to his charger",
    "intent_imperative_quiet": "told Vector to be quiet",
    "intent_imperative_volumeup": "turned Vector's volume up",
    "intent_imperative_volumedown": "turned Vector's volume down",
}
_IGNORED = {"intent_system_unmatched", "intent_system_noaudio"}  # nothing was understood; not an interaction


def describe(intent: str) -> str:
    return REACTIONS.get(intent, f"used Vector's built-in command {intent.removeprefix('intent_')}")


# --- linking wire-pod's fallback to the gateway -----------------------------------------------------------------------


def current() -> dict:
    return httpx.get(f"{config.wirepod_url()}/api/get_kg_api", timeout=10).json()


def link() -> dict:
    """Send speech wire-pod does not recognise to Kin's gateway. The previous settings are backed up once."""
    before = current()
    if not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        BACKUP.write_text(json.dumps(before, indent=2), "utf-8")
    settings = dict(before) | {
        "enable": True,
        "provider": "custom",
        "endpoint": f"http://127.0.0.1:{config.gateway_port()}/v1",
        "key": "kin-local",          # unused by the gateway; wire-pod wants something
        "model": "kin",
        "intentgraph": True,          # the point: fall back to the endpoint for anything unmatched
        "save_chat": False,           # Kin keeps its own memory
        "commands_enable": False,     # the reply is a plain phrase for Vector to say
        "openai_prompt": "",
    }
    httpx.post(f"{config.wirepod_url()}/api/set_kg_api", json=settings, timeout=10).raise_for_status()
    return settings


def unlink() -> dict:
    """Put wire-pod's fallback settings back the way they were before `link`."""
    if not BACKUP.exists():
        raise FileNotFoundError(f"no backup at {BACKUP}; nothing to restore")
    before = json.loads(BACKUP.read_text("utf-8"))
    httpx.post(f"{config.wirepod_url()}/api/set_kg_api", json=before, timeout=10).raise_for_status()
    BACKUP.unlink()
    return before


# --- taking over built-in commands whose own answer is broken -----------------------------------------------------------

# Asking Vector about the weather gets his built-in "weather API is not set up", while Kin has a real
# forecast. wire-pod tries custom intents before his built-in ones, matching by substring, so one custom
# intent claims these words for Kin. Whole phrases where a short word would catch others ("rain" is in "train").
TAKEOVER = "Kin takes over"
TAKEOVER_WORDS = ("weather", "forecast", "temperature", "umbrella", "going to rain", "is it raining",
                  "cold outside", "hot outside", "warm outside")
# Sent to Vector to end his listening quietly: it is what he gets when nobody said anything, and he simply
# goes back to what he was doing. The script then has him say Kin's acknowledgement in his own voice.
_QUIET_INTENT = "intent_system_noaudio"


def takeover_script() -> str:
    """Lua that wire-pod runs on a match. Fetching /handover is what queues the request for Kin; the reply
    is the phrase for Vector to say. If the gateway is down he stays silent rather than saying an error."""
    return (f'local ack = getHTTPRequest("http://127.0.0.1:{config.gateway_port()}/handover")\n'
            'if ack ~= nil and ack ~= "" and string.len(ack) < 80 then\n'
            "  assumeBehaviorControl(20)\n"
            "  sayText(ack)\n"
            "  releaseBehaviorControl()\n"
            "end\n")


def custom_intents() -> list[dict]:
    response = httpx.get(f"{config.wirepod_url()}/api/get_custom_intents_json", timeout=10)
    return response.json() if response.status_code == 200 else []  # wire-pod answers 500 when there are none


def take_over() -> dict:
    """Add (or refresh) the custom intent that hands weather questions to Kin."""
    give_back()
    intent = {
        "name": TAKEOVER,
        "description": "Questions Kin answers better than Vector's built-in commands (weather).",
        "utterances": list(TAKEOVER_WORDS),
        "intent": _QUIET_INTENT,
        "params": {"paramname": "", "paramvalue": ""},
        "exec": "",
        "execargs": [],
        "luascript": takeover_script(),
    }
    httpx.post(f"{config.wirepod_url()}/api/add_custom_intent", json=intent, timeout=10).raise_for_status()
    return intent


def give_back() -> bool:
    """Remove Kin's custom intent; Vector's built-in commands answer again. Other custom intents are left alone."""
    for number, intent in enumerate(custom_intents(), start=1):
        if intent.get("name") == TAKEOVER:
            httpx.post(f"{config.wirepod_url()}/api/remove_custom_intent", json={"number": number},
                       timeout=10).raise_for_status()
            return True
    return False


def last_transcript(lines: list[str], now: str, max_age_seconds: int = 30) -> str:
    """What Vector most recently heard, from wire-pod's log, if it is fresh. `now` is in the log's own
    timestamp format. The takeover script cannot pass the words along, so the gateway looks them up here."""
    for line in reversed(lines):
        if match := _HEARD.search(line):
            try:
                age = datetime.strptime(now, _STAMP_FORMAT) - datetime.strptime(line[:19], _STAMP_FORMAT)
            except ValueError:
                return ""
            return match.group(1).strip() if 0 <= age.total_seconds() <= max_age_seconds else ""
    return ""


def heard_just_now() -> str:
    text = httpx.get(f"{config.wirepod_url()}/api/get_debug_logs", timeout=10).text
    return last_transcript(text.splitlines(), datetime.now().strftime(_STAMP_FORMAT))


# --- watching Vector's built-in commands ------------------------------------------------------------------------------

_STAMP = re.compile(r"\d{4}\.\d\d\.\d\d \d\d:\d\d:\d\d")
_STAMP_FORMAT = "%Y.%m.%d %H:%M:%S"
_HEARD = re.compile(r"Bot \w+ Transcribed text: (.*)$")
_SENT = re.compile(r"Bot \w+ Intent Sent: (\S+)")
_WAKE = re.compile(r"Bot \w+ Stream type: ")  # written the moment he starts listening after the wake word


def interactions(lines: list[str]) -> list[tuple[str, str]]:
    """(what was said, intent sent) pairs found in wire-pod log lines, in order.

    Speech that matched nothing goes to Kin's gateway instead and logs no "Intent Sent", so it is not
    reported here: Kin hears about that through the gateway and records it itself.
    """
    found, said = [], None
    for line in lines:
        if match := _HEARD.search(line):
            said = match.group(1).strip()
        elif (match := _SENT.search(line)) and said is not None:
            if match.group(1) not in _IGNORED:
                found.append((said, match.group(1)))
            said = None
    return found


class IntentWatcher:
    """Polls wire-pod's log and records the person's dealings with Vector himself."""

    def __init__(self, poll_seconds: float = 1.0, start: bool = True):
        self._poll_seconds = poll_seconds
        self._mark = ""                 # timestamp of the newest line dealt with ("2026.09.20 11:02:03")
        self._at_mark: set[str] = set()  # the lines seen at exactly that second
        self._primed = False
        self.last_wake = float("-inf")  # monotonic time Vector last heard his wake word (see `spoken_to`)
        if start:
            threading.Thread(target=self._watch, daemon=True, name="wire-pod watcher").start()

    def fresh(self, lines: list[str]) -> list[str]:
        """The lines not dealt with before. The log is re-read whole each time and wire-pod may trim its
        start, so progress is a high-water mark on the timestamp each line begins with, not a line count."""
        new = []
        for line in lines:
            stamp = line[:19]
            if not _STAMP.fullmatch(stamp) or stamp < self._mark or (stamp == self._mark and line in self._at_mark):
                continue
            if stamp > self._mark:
                self._mark, self._at_mark = stamp, set()
            self._at_mark.add(line)
            new.append(line)
        if not self._primed:  # everything already in the log when Kin starts is history, not news
            self._primed = True
            return []
        if any(_WAKE.search(line) for line in new):
            self.last_wake = time.monotonic()
        return new

    def spoken_to(self, within_seconds: float = 60.0) -> bool:
        """Has Vector heard his wake word in the last little while? wire-pod logs the start of listening
        two or three seconds before it sends the intent, which is time enough to get out of his way: the
        idle hold blocked his own reactions ("I love you" got nothing, Sep 23), so the service drops it
        for a minute whenever he is spoken to."""
        return time.monotonic() - self.last_wake <= within_seconds

    def _new_lines(self) -> list[str]:
        text = httpx.get(f"{config.wirepod_url()}/api/get_debug_logs", timeout=10).text
        return self.fresh([line.rstrip() for line in text.splitlines() if line.strip()])

    def _watch(self) -> None:
        memory = Memory()  # this thread's own connection
        carried: list[str] = []  # a "Transcribed text" line whose "Intent Sent" had not been written yet
        while True:
            try:
                lines = carried + self._new_lines()
                last_heard = max((i for i, line in enumerate(lines) if _HEARD.search(line)), default=None)
                answered = last_heard is None or any(_SENT.search(line) for line in lines[last_heard:])
                carried = [] if answered or len(lines) - last_heard > 8 else lines[last_heard:]
                for said, intent in interactions(lines):
                    memory.log_turn("person", said)
                    memory.log_event("vector_intent", f"{describe(intent)}: \"{said}\"")
            except Exception:
                pass  # wire-pod restarting, or not running at all; keep trying
            time.sleep(self._poll_seconds)
