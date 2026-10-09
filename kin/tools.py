"""Robot actions exposed to the LLM as tools."""

import json
import re
import time
from typing import Callable

import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps

from kin import alerts, config, family, llm, search, weather, wirepod
from kin import robot as kin_robot
from kin.memory import Memory
from kin.robot import call
from kin.voice import ConsoleEars, Ears, Mouth, VectorMouth

MAX_DRIVE_MM = 150
MAX_TURN_DEG = 180
HEAD_UP_DEG = 35
NO_REPLY = "no reply heard"  # what `ask` returns when nobody answered
TRICK_SECONDS = 90  # how long Vector is left to himself after being asked for one of his own behaviours
REPLIED = "the person replied: "  # and how its result begins when somebody did
# With the reason: a bare "please put me on my charger" from a fully charged robot sounded like a fault.
ASK_FOR_A_LIFT = ("Ask the person, once and politely, to put you on it, and say why: you cannot get back to it "
                  "by yourself. Do not mention your battery unless it is low.")

# Friendly names the model may use, mapped to SDK animation triggers.
ANIMATIONS = {
    "greet": "GreetAfterLongTime",
    "happy": "KnowledgeGraphSuccessReaction",
    "celebrate": "PounceSuccess",
    "fist_bump": "FistBumpRequestOnce",
    "sad": "KnowledgeGraphSearchingFail",
    "timer_ring": "TimerRing",
}


def _spec(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
        },
    }


TOOL_SPECS = [
    _spec(
        "say",
        "Speak out loud. This is the only way the person can hear you. Keep it to one or two short sentences.",
        {"text": {"type": "string"}},
        ["text"],
    ),
    _spec(
        "ask",
        "Say a question out loud, then listen for the person's spoken reply. Returns what they said, or 'no reply "
        "heard'. Use this for every question; `say` does not listen. Ask one short question at a time.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _spec(
        "look_for_face",
        "Turn in place, looking up, until a human face is seen. Returns whether someone was found and their name if known.",
    ),
    _spec(
        "describe_scene",
        "Take a photo with the camera and get an answer about what is visible. The camera is 3 cm above the table, "
        "so ask the person to hold small things up in front of you.",
        {"question": {"type": "string", "description": "What you want to know about the view"}},
        ["question"],
    ),
    _spec(
        "turn",
        f"Turn in place. Positive is left (counter-clockwise). Max {MAX_TURN_DEG} degrees.",
        {"degrees": {"type": "number"}},
        ["degrees"],
    ),
    _spec(
        "drive",
        f"Drive straight. Positive is forward. Max {MAX_DRIVE_MM} mm per call. Stops by itself at table edges.",
        {"distance_mm": {"type": "number"}},
        ["distance_mm"],
    ),
    _spec(
        "play_animation",
        "Express an emotion with a body animation.",
        {"name": {"type": "string", "enum": sorted(ANIMATIONS)}},
        ["name"],
    ),
    _spec("go_home", "Drive back onto the charger. Do this when a task is finished."),
    _spec(
        "remember",
        "Save a lasting fact about the person or their life (names, family, appointments, preferences, how they "
        "are feeling). Write one self-contained sentence in the third person ('The person's granddaughter Mila...'), "
        "turning relative days like 'Tuesday' or 'tomorrow' into calendar dates. If a saved fact already covers "
        "this, use `update_fact` instead of saving a duplicate.",
        {
            "fact": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD, only if the fact is about a dated event"},
        },
        ["fact"],
    ),
    _spec(
        "recall",
        "Search your saved facts by meaning. Use before asking something the person may have told you already.",
        {"query": {"type": "string", "description": "What you want to know, or empty for the most recent facts"}},
    ),
    _spec(
        "update_fact",
        "Correct or close a saved fact, by its # number. Give new text when details changed; set status to 'done' "
        "when the event has happened, or 'retired' when the fact was wrong or no longer applies.",
        {
            "fact_id": {"type": "integer"},
            "text": {"type": "string", "description": "Replacement wording, if it changed"},
            "date": {"type": "string", "description": "New YYYY-MM-DD date, if it changed"},
            "status": {"type": "string", "enum": ["active", "done", "retired"]},
        },
        ["fact_id"],
    ),
    _spec(
        "weather",
        "Current weather plus today's and tomorrow's forecast, from a forecast service. Always use this for "
        "weather, never `web_search` and never your own knowledge. Say the temperature with its unit. Tell the "
        "person about now and today in one or two short sentences; tomorrow only if they asked about it.",
        {"place": {"type": "string", "description": "Town or city; leave empty for where the person lives"}},
    ),
    _spec(
        "set_reminder",
        "Set a repeating nudge, e.g. pills, water, a walk. You will be woken at that time to go and ask the person. "
        "Confirm the time with the person before setting it.",
        {
            "text": {"type": "string", "description": "What to nudge about, e.g. 'morning pills'"},
            "time_of_day": {"type": "string", "description": "24-hour HH:MM"},
            "days": {"type": "string", "description": "'daily' (default) or a list like 'mon,wed,fri'"},
        },
        ["text", "time_of_day"],
    ),
    _spec("list_reminders", "List the active reminders with their # numbers."),
    _spec("cancel_reminder", "Stop a reminder, by its # number.", {"reminder_id": {"type": "integer"}}, ["reminder_id"]),
    _spec(
        "reminder_outcome",
        "Record how a due reminder went. 'confirmed': they did it. 'declined': they chose not to today. "
        "'no_response': you could not find them or got no answer (you will be woken to try again later).",
        {
            "reminder_id": {"type": "integer"},
            "outcome": {"type": "string", "enum": ["confirmed", "declined", "no_response"]},
        },
        ["reminder_id", "outcome"],
    ),
]


WEB_SEARCH_SPEC = _spec(
    "web_search",
    "Search the live web for current facts: news, TV listings, opening hours, sports results, or whether "
    "a message or phone call matches a known scam. Never answer such questions from memory. When the answer "
    "depends on where they live, put the town AND its state or country in the query, exactly as given to you: "
    "'San Jose' alone once returned news from Costa Rica. Not for weather: use `weather`. Be wary of "
    "numbers and units in results; if a figure looks implausible, say you are not sure rather than repeat it.",
    {
        "query": {"type": "string"},
        "topic": {"type": "string", "enum": ["general", "news"], "description": "'news' for recent events"},
    },
    ["query"],
)


VECTOR_TRICK_SPEC = _spec(
    "vector_trick",
    "Hand the request back to Vector's own body for something only he does: fetching or playing with his "
    "cube, dancing, a fist bump, taking a photo himself, driving to his charger, going to sleep, exploring. "
    "Use it whenever the person asks for one of these, even if the words came through garbled ('french cube' "
    "is 'fetch the cube'). He does it as himself; you stand back and say nothing about it afterwards.",
    {"trick": {"type": "string", "enum": sorted(wirepod.TRICKS)}},
    ["trick"],
)


NOTIFY_FAMILY_SPEC = _spec(
    "notify_family",
    "Send a short text message to the person's family. Use it when the person asks you to pass something on, "
    "to report that a family message was delivered and what they replied, or when something worries you. "
    "Tell the person what you are sending; never message the family behind their back.",
    {"message": {"type": "string"}},
    ["message"],
)


def tool_specs() -> list[dict]:
    """The tools offered to the model. Optional tools appear only when configured, so Kin never promises them otherwise."""
    specs = list(TOOL_SPECS) + [VECTOR_TRICK_SPEC]
    if search.available():
        specs.append(WEB_SEARCH_SPEC)
    if family.available():
        specs.append(NOTIFY_FAMILY_SPEC)
    return specs


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


# The weather tool gives the planner words ("degrees Fahrenheit") and it writes symbols anyway; the voice
# then read "62°F" as "62 E F". Whatever is spoken goes through here, so no symbol reaches the voice.
_SPOKEN_UNITS = [
    (re.compile(r"\s*°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\s*°\s*C\b"), " degrees Celsius"),
    (re.compile(r"\s*°"), " degrees"),
    (re.compile(r"(?<=\d)\s*%"), " percent"),
    (re.compile(r"(?<=\d)\s*mph\b"), " miles per hour"),
    (re.compile(r"(?<=\d)\s*km/h\b"), " kilometres per hour"),
]


LISTEN_SECONDS = 12.0
CLOSING_LISTEN_SECONDS = 8.0
_CLOSING = re.compile(r"anything (else|more)|else (i can|you need|can i)", re.IGNORECASE)


def is_closing_offer(question: str) -> bool:
    """Does this question end a conversation with an offer of more? Silence is the usual answer to it."""
    return bool(_CLOSING.search(question))


# One phrase of a "no" to an offer of more: "No, thank you.", "Now I'm good" (Whisper's "No" is often "Now"),
# "That's it.", "Nothing, never mind." A decline is short and made only of such phrases; anything with a
# request in it ("No, but what's the weather?") is not one, and the planner carries on as before.
_NO_WORD = r"(?:no|nope|nah|now|not)\b\s*"
_DECLINE_CLAUSE = (r"(?:thank you(?: very much)?|thanks|i'?m (?:good|fine|ok|okay|all right|alright|all set)|"
                   r"i am (?:good|fine|okay)|everything(?:'s| is) (?:fine|ok|okay|good)|"
                   r"that(?:'s| is| will be) (?:it|all|everything)|nothing(?: else| more| at all)?|never mind|"
                   r"not (?:now|today|right now|really)|all good|i (?:do not|don'?t) (?:need anything(?: else)?|think so)|no)")
_DECLINE_PHRASE = re.compile(rf"(?:{_NO_WORD})?(?:{_DECLINE_CLAUSE}\s*){{0,2}}", re.IGNORECASE)
_SAYS_NO = re.compile(r"\b(no|nope|nah|nothing|never mind|not (now|today|really)|that('s| is| will be) (it|all|everything)|"
                      r"i('m| am) (good|fine|ok|okay|all right|alright|all set)|all good|everything('s| is) (fine|ok|okay|good)|"
                      r"(do not|don'?t) (need|think))\b", re.IGNORECASE)


def is_decline(reply: str) -> bool:
    """Is this reply to an offer of more a plain no, and nothing else?"""
    text = reply.strip()
    if "?" in text:  # the microphone caught the tail of Kin's own question first: "Anything else? No, thank you."
        before, _, after = text.rpartition("?")
        if len(before.split()) >= 2:
            text = after
    if len(text.split()) > 12:
        return False
    phrases = [p.strip() for p in re.split(r"[,.!;]+", text.lower()) if p.strip()]
    return bool(phrases) and all(_DECLINE_PHRASE.fullmatch(p) for p in phrases) and any(_SAYS_NO.search(p) for p in phrases)


def is_echo(question: str, reply: str) -> bool:
    """Did the microphone only catch the tail of Kin's own question? Live, "Could you please place me on my
    charger?" got the reply "charger." from the room's echo, and Kin thanked a person who was not there.
    A reply of one or two words, all of them from the question, is not an answer."""
    words = re.findall(r"[a-z']+", reply.lower())
    asked = set(re.findall(r"[a-z']+", question.lower()))
    return 0 < len(words) <= 2 and all(word in asked for word in words)


def for_speech(text: str) -> str:
    """Text as the voice should get it: unit symbols spelled out, odd spaces made plain."""
    for symbol, words in _SPOKEN_UNITS:
        text = symbol.sub(words, text)
    return " ".join(text.split())


class RobotTools:
    def __init__(self, robot: anki_vector.Robot, memory: Memory, mouth: Mouth | None = None, ears: Ears | None = None,
                 family_link: "family.Family | None" = None, attention: Callable[[], bool] | None = None):
        self.robot = robot
        # Returns True once when the person pets Vector. Slow moves poll it and stop, so a two-minute
        # docking attempt no longer makes him deaf to the person standing next to him.
        self.attention = attention
        self.current_reminder: int | None = None  # set by the service for the length of a reminder task
        self.memory = memory
        self.family = family_link
        self.mouth = mouth or VectorMouth(robot)
        self.ears = ears or ConsoleEars()

    def dispatch(self, name: str, arguments: str) -> str:
        """Run a tool call from the model. Always returns a string result, never raises."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"error: unknown tool {name}"
        try:
            args = json.loads(arguments) if arguments else {}
            return kin_robot.with_retry(lambda: handler(**args))
        except kin_robot.Interrupted:
            return ("interrupted: the person just petted you to get your attention. Drop what you were doing, "
                    "greet them and ask what they need.")
        except Exception as e:  # the model gets to see the failure and react
            return f"error: {type(e).__name__}: {e}"

    def _docked_to_stay(self) -> bool:
        return config.stay_on_charger() and self.robot.status.is_on_charger

    def _leave_charger(self) -> None:
        if self.robot.status.is_on_charger:
            call(self.robot.behavior.drive_off_charger)

    def _tool_say(self, text: str) -> str:
        # The planner keeps putting questions in `say` despite its instructions. Live, a check-in said "How
        # did you sleep?" and then asked it again with `ask`, and later said "How are you feeling today?" to
        # nobody listening. A question is a question: whatever tool it came through, he listens for the answer.
        if text.rstrip().endswith("?"):
            return self._tool_ask(text)
        self._speak(text)
        return "said"

    def _speak(self, text: str) -> None:
        text = for_speech(text)
        self.mouth.say(text)
        self.memory.log_turn("kin", text)

    def _tool_ask(self, question: str) -> str:
        if prepare := getattr(self.ears, "prepare", None):
            prepare()  # microphone open before he speaks, so a prompt answer is not clipped; `listen` closes it
        self._speak(question)
        # A closing offer ("anything else?") that meets silence should not keep him listening for long.
        seconds = CLOSING_LISTEN_SECONDS if is_closing_offer(question) else LISTEN_SECONDS
        with kin_robot.animating(self.robot, "VC_ListeningLoop"):  # his own "I'm listening" look
            reply = self.ears.listen(timeout=seconds)
        if not reply or is_echo(question, reply):
            return NO_REPLY
        self.memory.log_turn("person", reply)
        return REPLIED + reply

    def _tool_set_reminder(self, text: str, time_of_day: str, days: str = "daily") -> str:
        reminder_id = self.memory.add_reminder(text, time_of_day, days)
        return f"reminder #{reminder_id} set for {time_of_day} ({days})"

    def _tool_list_reminders(self) -> str:
        return "\n".join(f"#{r.id} {r.time_of_day} ({r.days}): {r.text}" for r in self.memory.reminders()) or "none set"

    def _tool_cancel_reminder(self, reminder_id: int) -> str:
        return "cancelled" if self.memory.deactivate_reminder(int(reminder_id)) else f"error: no reminder #{reminder_id}"

    def _tool_reminder_outcome(self, reminder_id: int, outcome: str) -> str:
        # The planner once logged the outcome on a made-up id (#5 for reminder #1, Sep 23), so the reminder
        # went unsettled and was asked again ten minutes later. During a reminder task the service says
        # which one is being handled, and that wins over whatever id the planner wrote.
        current = self.current_reminder
        if current is not None and int(reminder_id) != current:
            reminder_id = current
        self.memory.log_reminder(int(reminder_id), outcome)
        if alerts.report_missed_reminders(self.memory, self.family):
            return "logged; that was the last try for today, and the family has been told"
        return "logged"

    def _tool_remember(self, fact: str, date: str | None = None) -> str:
        return f"saved as #{self.memory.remember(fact, due=date)}"

    def _tool_recall(self, query: str = "") -> str:
        return "\n".join(fact.render() for fact in self.memory.recall(query)) or "nothing saved yet"

    def _tool_notify_family(self, message: str) -> str:
        if self.family is None:
            return "error: the family link is not set up"
        sent = self.family.send(f"Kin: {message}")
        self.memory.log_event("family_out", message)
        return f"sent to {sent} family member(s)" if sent else "error: the message could not be sent"

    def _tool_vector_trick(self, trick: str) -> str:
        intent = wirepod.TRICKS.get(trick.strip().lower().replace(" ", "_").replace("-", "_"))
        if intent is None:
            return f"error: '{trick}' is not one of his tricks; they are {', '.join(sorted(wirepod.TRICKS))}"
        kin_robot.let_him_be(TRICK_SECONDS)   # no control, no tasks, while his own behaviour runs
        kin_robot.hand_back(self.robot)
        if not wirepod.run_intent(intent):
            return "error: could not ask him to do it (wire-pod did not accept it)"
        self.memory.log_event("vector_intent", f"Kin asked him to {trick.replace('_', ' ')}")
        return (f"he is doing it himself now. Do not describe it or move him; if you say anything, keep it to "
                "a few words and then finish.")

    def _tool_weather(self, place: str = "") -> str:
        return weather.report(place)

    def _tool_web_search(self, query: str, topic: str = "general") -> str:
        if not search.available():
            return "error: web search is not set up"
        return search.search(query, topic)

    def _tool_update_fact(self, fact_id: int, text: str | None = None, date: str | None = None,
                          status: str | None = None) -> str:
        found = self.memory.update_fact(int(fact_id), text=text, status=status, due=date)
        return "updated" if found else f"error: there is no fact #{fact_id}"

    def _tool_look_for_face(self) -> str:
        if self._docked_to_stay():
            found = kin_robot.quiet_look(self.robot)
            return ("found a person" if found else
                    "no one in view from the charger, and you are staying on it. They may well be in the room: "
                    "call out with `ask`.")
        self._leave_charger()
        call(self.robot.behavior.set_head_angle, degrees(HEAD_UP_DEG), timeout=10)
        for _ in range(8):
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                faces = list(self.robot.world.visible_faces)
                if faces:
                    name = next((f.name for f in faces if f.name), "")
                    return f"found a person, name: {name}" if name else "found a person, name unknown"
                time.sleep(0.2)
            call(self.robot.behavior.turn_in_place, degrees(45), timeout=15, interrupt=self.attention)
        return "no one found after a full turn"

    def _tool_describe_scene(self, question: str) -> str:
        call(self.robot.behavior.set_head_angle, degrees(25), timeout=10)
        prompt = (
            "You are the eyes of a small robot whose camera is 3 cm above a table. "
            f"Answer in one or two plain sentences, no markdown. If you cannot tell, say so.\n\n{question}"
        )
        return llm.describe_image(kin_robot.take_photo(self.robot), prompt, max_tokens=120)

    def _tool_turn(self, degrees: float) -> str:  # noqa: A002 - name is the tool's argument
        if self._docked_to_stay():
            return "staying on the charger, so not turning. You can still talk and listen from here."
        self._leave_charger()
        angle = _clamp(degrees, MAX_TURN_DEG)
        call(self.robot.behavior.turn_in_place, anki_vector.util.degrees(angle), timeout=20, interrupt=self.attention)
        return f"turned {angle:.0f} degrees"

    def _tool_drive(self, distance_mm: float) -> str:  # noqa: A002
        if self._docked_to_stay():
            return "staying on the charger, so not driving. You can still talk and listen from here."
        self._leave_charger()
        dist = _clamp(distance_mm, MAX_DRIVE_MM)
        call(self.robot.behavior.drive_straight, anki_vector.util.distance_mm(dist), speed_mmps(60), timeout=20,
             interrupt=self.attention)
        return f"drove {dist:.0f} mm"

    def _tool_play_animation(self, name: str) -> str:
        trigger = ANIMATIONS.get(name)
        if trigger is None:
            return f"error: unknown animation {name}; choose from {sorted(ANIMATIONS)}"
        call(self.robot.anim.play_animation_trigger, trigger, timeout=20)
        return "played"

    def _tool_go_home(self) -> str:
        if self.robot.status.is_on_charger:
            return "already on the charger"
        if not kin_robot.charger_in_view(self.robot, self.attention):
            return "cannot see the charger from here. " + ASK_FOR_A_LIFT
        try:
            call(self.robot.behavior.drive_on_charger, timeout=60, interrupt=self.attention)
        except TimeoutError:
            pass  # the result below says whether he made it
        if self.robot.status.is_on_charger:
            return "on the charger"
        return "could not get onto the charger. " + ASK_FOR_A_LIFT
