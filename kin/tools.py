"""Robot actions exposed to the LLM as tools."""

import json
import time

import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps

from kin import config, family, llm, search
from kin import robot as kin_robot
from kin.memory import Memory
from kin.robot import call
from kin.voice import ConsoleEars, Ears, Mouth, VectorMouth

MAX_DRIVE_MM = 150
MAX_TURN_DEG = 180
HEAD_UP_DEG = 35

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
    "Search the live web for current facts: weather, news, TV listings, opening hours, sports results, or whether "
    "a message or phone call matches a known scam. Never answer such questions from memory. Include the person's "
    "town in the query when the answer depends on where they live.",
    {
        "query": {"type": "string"},
        "topic": {"type": "string", "enum": ["general", "news"], "description": "'news' for recent events"},
    },
    ["query"],
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
    specs = list(TOOL_SPECS)
    if search.available():
        specs.append(WEB_SEARCH_SPEC)
    if family.available():
        specs.append(NOTIFY_FAMILY_SPEC)
    return specs


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


class RobotTools:
    def __init__(self, robot: anki_vector.Robot, memory: Memory, mouth: Mouth | None = None, ears: Ears | None = None,
                 family_link: "family.Family | None" = None):
        self.robot = robot
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
        except Exception as e:  # the model gets to see the failure and react
            return f"error: {type(e).__name__}: {e}"

    def _leave_charger(self) -> None:
        if self.robot.status.is_on_charger:
            call(self.robot.behavior.drive_off_charger)

    def _tool_say(self, text: str) -> str:
        self.mouth.say(text)
        self.memory.log_turn("kin", text)
        return "said"

    def _tool_ask(self, question: str) -> str:
        self._tool_say(question)
        reply = self.ears.listen()
        if not reply:
            return "no reply heard"
        self.memory.log_turn("person", reply)
        return f"the person replied: {reply}"

    def _tool_set_reminder(self, text: str, time_of_day: str, days: str = "daily") -> str:
        reminder_id = self.memory.add_reminder(text, time_of_day, days)
        return f"reminder #{reminder_id} set for {time_of_day} ({days})"

    def _tool_list_reminders(self) -> str:
        return "\n".join(f"#{r.id} {r.time_of_day} ({r.days}): {r.text}" for r in self.memory.reminders()) or "none set"

    def _tool_cancel_reminder(self, reminder_id: int) -> str:
        return "cancelled" if self.memory.deactivate_reminder(int(reminder_id)) else f"error: no reminder #{reminder_id}"

    def _tool_reminder_outcome(self, reminder_id: int, outcome: str) -> str:
        self.memory.log_reminder(int(reminder_id), outcome)
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

    def _tool_web_search(self, query: str, topic: str = "general") -> str:
        if not search.available():
            return "error: web search is not set up"
        return search.search(query, topic)

    def _tool_update_fact(self, fact_id: int, text: str | None = None, date: str | None = None,
                          status: str | None = None) -> str:
        found = self.memory.update_fact(int(fact_id), text=text, status=status, due=date)
        return "updated" if found else f"error: there is no fact #{fact_id}"

    def _tool_look_for_face(self) -> str:
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
            call(self.robot.behavior.turn_in_place, degrees(45), timeout=15)
        return "no one found after a full turn"

    def _tool_describe_scene(self, question: str) -> str:
        call(self.robot.behavior.set_head_angle, degrees(25), timeout=10)
        prompt = (
            "You are the eyes of a small robot whose camera is 3 cm above a table. "
            f"Answer in one or two plain sentences, no markdown. If you cannot tell, say so.\n\n{question}"
        )
        return llm.describe_image(kin_robot.take_photo(self.robot), prompt, max_tokens=120)

    def _tool_turn(self, degrees: float) -> str:  # noqa: A002 - name is the tool's argument
        self._leave_charger()
        angle = _clamp(degrees, MAX_TURN_DEG)
        call(self.robot.behavior.turn_in_place, anki_vector.util.degrees(angle), timeout=20)
        return f"turned {angle:.0f} degrees"

    def _tool_drive(self, distance_mm: float) -> str:  # noqa: A002
        self._leave_charger()
        dist = _clamp(distance_mm, MAX_DRIVE_MM)
        call(self.robot.behavior.drive_straight, anki_vector.util.distance_mm(dist), speed_mmps(60), timeout=20)
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
        call(self.robot.behavior.drive_on_charger, timeout=120)
        return "on the charger" if self.robot.status.is_on_charger else "could not reach the charger"
