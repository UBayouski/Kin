"""Kin as an always-on service: nobody types anything; the clock and the person's touch start tasks.

    python -m kin.service                 run all day
    python -m kin.service --checkin-now   run the morning check-in immediately, then carry on
    python -m kin.service --once          exit after the first task (testing, demos)

Triggers, checked about once a second:
  - the daily check-in time (KIN_CHECKIN_TIME)
  - reminders that are due, with gentle re-nudges
  - the person petting Vector's back, to start a conversation
  - a nightly memory tidy-up
  - messages from the family (Telegram), delivered in person; a daily note and alerts go back to them

Between tasks Vector is left to his own personality; control is borrowed per task.
"""

import queue
import sys
import time
from datetime import datetime

from kin import robot as kin_robot
from kin import config, digest, family, voice
from kin.agent import Agent
from kin.consolidate import consolidate
from kin.memory import Memory, Reminder

CONSOLIDATE_AT = "03:00"
TOUCH_HOLD_SECONDS = 0.6   # a deliberate pet, not a brush
TOUCH_COOLDOWN_SECONDS = 20
RECONNECT_SECONDS = 30

CHECKIN = (
    "[Daily check-in. Find the person and greet them warmly, by name if you know it, in a way that fits "
    "the time of day. Ask how they are, and in the morning how they slept. Follow up on one thing from "
    "your facts: a recent event, a worry, a visitor. If an answer is vague, ask one gentle follow-up. "
    "Save anything new worth keeping. If web search is available, tell them the weather in one sentence. "
    "Keep it to a few exchanges, wish them a good day, then go home.]"
)
TOUCHED = (
    "[The person petted your back to get your attention. Say a short friendly hello and ask what you can "
    "do for them, then help with whatever they say. When they are done, go home.]"
)


def reminder_prompt(reminder: Reminder, attempt: int) -> str:
    again = " This is another try; they did not answer earlier, so be gentle." if attempt > 1 else ""
    return (
        f"[Reminder #{reminder.id} is due: '{reminder.text}' (set for {reminder.time_of_day}).{again} "
        "Find the person, then `ask` them about it in a warm, natural way. Record the result with "
        "`reminder_outcome`: confirmed, declined, or no_response if you cannot find them or get no answer. "
        "Then go home.]"
    )


def log(text: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {text}", flush=True)


class Service:
    def __init__(self, memory: Memory, checkin_now: bool = False, once: bool = False):
        self.memory = memory
        self.checkin_now = checkin_now
        self.once = once
        self.ears = voice.make_ears()
        self.family = family.Family() if family.available() else None
        self._touch_since: float | None = None
        self._touch_ready_at = 0.0
        self._reported_missed: set[tuple[str, int]] = set()

    # --- what is due ----------------------------------------------------------------------------------------------

    def _done_today(self, kind: str) -> bool:
        last = self.memory.last_event_time(kind)
        return bool(last and last[:10] == datetime.now().date().isoformat())

    def _checkin_due(self) -> bool:
        if self.checkin_now:
            return True
        return datetime.now().strftime("%H:%M") >= config.checkin_time() and not self._done_today("checkin")

    def _petted(self, robot) -> bool:
        now = time.monotonic()
        if not kin_robot.is_touched(robot) or now < self._touch_ready_at:
            self._touch_since = None
            return False
        self._touch_since = self._touch_since or now
        if now - self._touch_since < TOUCH_HOLD_SECONDS:
            return False
        self._touch_since, self._touch_ready_at = None, now + TOUCH_COOLDOWN_SECONDS
        return True

    # --- one pass -------------------------------------------------------------------------------------------------

    def _handle_family(self, agent: Agent) -> bool:
        """Deal with one waiting family message. Returns whether the robot was used."""
        try:
            message = self.family.incoming.get_nowait()
        except queue.Empty:
            return False
        command = message.text.split()[0].lower()
        if command in ("/start", "/help"):
            self.family.send(
                "I'm Kin. Send me any message and I will deliver it in person and tell you the reply.\n"
                "/status - how today is going\n/digest - today's note now",
                chat_id=message.chat_id,
            )
        elif command == "/status":
            self.family.send(digest.status(self.memory), chat_id=message.chat_id)
        elif command == "/digest":
            self.family.send(digest.build(self.memory), chat_id=message.chat_id)
        else:
            log(f"family message from {message.sender}")
            self.memory.log_event("family_in", f"{message.sender}: {message.text}")
            prompt = f"[Message from family member {message.sender}, to deliver in person: \"{message.text}\"]"
            log("done: " + agent.run(prompt, speaker="system"))
            return True
        return False

    def tick(self, robot, agent: Agent) -> bool:
        """Run at most one task. Returns whether one ran."""
        if self.family and self._handle_family(agent):
            return True

        if self._petted(robot):
            log("petted -> conversation")
            log("done: " + agent.run(TOUCHED, speaker="system"))
            return True

        if self._checkin_due():
            self.checkin_now = False
            log("check-in")
            summary = agent.run(CHECKIN, speaker="system")
            self.memory.log_event("checkin", summary)
            log("done: " + summary)
            return True

        for reminder in self.memory.reminders_due():
            attempt = 1 + sum(
                1 for ts, text, outcome in self.memory.reminder_history(datetime.now().date())
                if text == reminder.text and outcome == "no_response"
            )
            log(f"reminder #{reminder.id} '{reminder.text}' (attempt {attempt})")
            self.memory.log_reminder(reminder.id, "asked")
            log("done: " + agent.run(reminder_prompt(reminder, attempt), speaker="system"))
            return True  # one task per tick; the next is picked up on the following pass

        today = datetime.now().date().isoformat()
        for reminder in self.memory.reminders_missed():
            if (today, reminder.id) not in self._reported_missed:
                self._reported_missed.add((today, reminder.id))
                alert = f"No response to the reminder '{reminder.text}' after several tries today."
                self.memory.log_event("alert", alert)
                if self.family:
                    self.family.send(f"Kin: {alert} It may be worth a call.")
                log(f"gave up on reminder #{reminder.id} for today; logged an alert")

        if self.family and datetime.now().strftime("%H:%M") >= config.digest_time() and not self._done_today("digest"):
            log("sending the family digest")
            sent = self.family.send(digest.build(self.memory))
            self.memory.log_event("digest", f"sent to {sent} chat(s)")

        if datetime.now().strftime("%H:%M") >= CONSOLIDATE_AT and not self._done_today("consolidation"):
            if self.memory.turns_since(self.memory.last_event_time("consolidation")):
                log("tidying memory")
                log("done: " + consolidate(self.memory).summary())
        return False

    def run(self) -> None:
        log(f"Kin service: mouth={config.mouth()} ears={config.ears()} check-in at {config.checkin_time()} "
            f"family={'on, ' + str(len(config.family_chat_ids())) + ' chat(s)' if self.family else 'off'}")
        if self.family:
            self.family.start()
        while True:
            try:
                with kin_robot.connect(take_control=False) as robot:
                    agent = Agent(robot, self.memory, voice.make_mouth(robot), self.ears, self.family)
                    log("connected to Vector")
                    while True:
                        if self.tick(robot, agent) and self.once:
                            return
                        time.sleep(1.0)
            except KeyboardInterrupt:
                log("stopping")
                return
            except Exception as e:  # a dropped link or a bad task must not end the service
                log(f"error: {type(e).__name__}: {e}; reconnecting in {RECONNECT_SECONDS}s")
                time.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    Service(Memory(), checkin_now="--checkin-now" in sys.argv, once="--once" in sys.argv).run()
