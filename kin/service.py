"""Kin as an always-on service: nobody types anything; the clock and the person's touch start tasks.

    python -m kin.service                 run all day
    python -m kin.service --checkin-now   run the morning check-in immediately, then carry on
    python -m kin.service --once          exit after the first task (testing, demos)
    python -m kin.service --presence-now  act as if the person has not been seen by the deadline

Triggers, checked about once a second:
  - the daily check-in time (KIN_CHECKIN_TIME)
  - reminders that are due, with gentle re-nudges
  - the person speaking to Vector ("Hey Vector" or his back button; a pet too with KIN_PET_TO_TALK=on)
  - presence: quiet looks through the morning; if there is no sign of the person by a deadline he calls
    out once, then the family is told (and told again when the person turns up)
  - a nightly reflection: memory tidy-up, weekly trends for the family, and an audit of his own behaviour
  - messages from the family (Telegram), delivered in person; a daily note and alerts go back to them

Between tasks Vector is left to his own personality; control is borrowed per task.
"""

import sys
import time
import traceback
from datetime import datetime

from kin import robot as kin_robot
from kin import alerts, config, digest, family, gateway, nightly, voice, weather, wirepod
from kin.agent import Agent
from kin.memory import Memory, Reminder

NIGHTLY_AT = "03:00"

RECONNECT_SECONDS = 15
ROBOT_DOWN_MINUTES = 10  # unreachable this long, and the family is told
HEARTBEAT_SECONDS = 30

CHECKIN = (
    "[Daily check-in. Find the person and greet them warmly, by name if you know it, in a way that fits "
    "the time of day. The moment they answer your call, greet them: no lookups first (Oct 9: a weather "
    "call and a recall before the greeting left them eight seconds of silence after saying hello). "
    "Ask how they are, and in the morning how they slept. Follow up on one thing from "
    "your facts: a recent event, a worry, a visitor. If an answer is vague, ask one gentle follow-up. "
    "Always respond to what they just told you before moving on: a reply that ignores their words sounds "
    "like you were not listening. Ask what they have planned today; when they tell you, say something "
    "about it and work today's weather into that (one or two sentences: what it is like outside, and always "
    "the 'To wear' advice from the forecast, going out or not; the weather is given at the end of this note, "
    "so there is nothing to look up). The weather is mentioned once, in that remark and nowhere else, never "
    "before they have told you their plans. Save plans and anything else worth keeping. "
    "Keep it to a few exchanges. Your remark about their plans and the weather is the last one: put the "
    "offer of more in the same `ask`, as always, and after their no, wish them a good day."
)


def checkin_prompt(todays_weather: str = "", tail: str = "") -> str:
    """The check-in note with today's weather already in it, so the conversation has no lookup in the
    middle of it (Oct 9: a weather call and a recall after her hello left eight seconds of silence)."""
    given = f" Today's weather, already looked up: {todays_weather}" if todays_weather else ""
    return CHECKIN + given + tail + "]"


PRESENCE_CHECK = (
    "[You have not seen or heard the person yet today and it is getting late in the morning. Look for them, "
    "then call out once with `ask`, warmly, to check they are there. If they answer, have a short friendly "
    "exchange and close as always. If there is no answer, do not keep calling. Do not go home and do not message the family "
    "yourself: both are handled for you as soon as you finish.]"
)
LOW_BATTERY = (
    "[Your battery is low and you are off your charger. Try `go_home` once. If you cannot get there, tell "
    "the person your battery is low and ask them, once and politely, to put you on your charger. You cannot "
    "tell from their words whether they have done it: thank them and finish, and never say you are charging.]"
)
STRANDED_MINUTES = 10  # low and off the charger this long, and the family is told
GAUGE_UNREADABLE_MINUTES = 10  # off the charger with no real battery reading this long, and he asks to be put on it
GAUGE_UNREADABLE = (
    "[You are off your charger and you cannot read your own battery, so you may run flat without warning. "
    "Ask the person, once and politely, to put you on your charger to be safe. Do not mention your battery "
    "being low; you do not know. Thank them and finish.]"
)
BODY_LOST_MINUTES = 5  # his body board silent this long, and the family is told
HOLD_REFUSALS_BEFORE_RECONNECT = 10  # heartbeats (5 minutes) without the idle hold being granted: the link is bad
OFF_DOCK_GRACE_SECONDS = 120     # off the charger this long with nothing to do, and he tries to get back
RETURN_HOME_EVERY_SECONDS = 1800  # and again no sooner than this if it failed (each try is a slow turn)
ACK_SECONDS = 3.5      # how long Vector needs to say "On it!" himself before Kin takes over
HEARD_STALE_MINUTES = 3
FAMILY_COMMANDS = ("/start", "/help", "/status", "/digest", "/week")  # answered from memory; no robot needed
TOUCHED = (
    "[The person petted your back to get your attention. Greet them and ask what they need in ONE `ask` "
    "call, for example 'Hello Jennie, what can I do for you?'. Do not `say` a greeting first: that makes "
    "two questions in a row. Then help with whatever they say."
)


def going_home() -> str:
    """Added to a task's instructions only when Kin may have driven Vector off his charger. While he stays
    on it there is nowhere to go back to; and if he is off it because he wandered or was lifted, docking
    mostly fails and ends in a request to be carried. A low battery off the charger is handled on its own."""
    return "" if config.stay_on_charger() else " When you are finished, go home with `go_home`."


def reminder_prompt(reminder: Reminder, attempt: int) -> str:
    again = " This is another try; they did not answer earlier, so be gentle." if attempt > 1 else ""
    return (
        f"[Reminder #{reminder.id} is due: '{reminder.text}' (set for {reminder.time_of_day}).{again} "
        "Find the person, then `ask` them about it in a warm, natural way, ONCE: if no reply is heard, do not "
        "ask again now (you will be asked to try again later). Record the result with "
        "`reminder_outcome`: confirmed, declined, or no_response if you cannot find them or get no answer. "
        "If they answered, a few warm words back and the offer of more go in one `ask`, as always (\"Lovely, "
        "well done. Do you need anything else?\" / \"All right, another time. Anything else I can do?\"): "
        "silence after their answer sounds like you did not hear them."
        + going_home() + "]"
    )


def log(text: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {text}", flush=True)


class Service:
    def __init__(self, memory: Memory, checkin_now: bool = False, once: bool = False, presence_now: bool = False):
        self.memory = memory
        self.checkin_now = checkin_now
        self.once = once
        self.presence_now = presence_now  # testing: act as if the deadline has passed with no sign of the person
        self._faces: kin_robot.FaceWatcher | None = None
        self._next_quiet_look = 0.0
        self._last_presence_logged = 0.0
        self.ears = voice.make_ears()
        self.family = family.Family() if family.available() else None
        self._touch: kin_robot.TouchWatcher | None = None
        self._running_flat = False
        self._low_readings = 0
        self._flat_since: float | None = None
        self._next_heartbeat = 0.0
        self._battery: tuple[str, bool] | None = None
        self._volts = 0.0
        self._raw_volts = -1.0  # the previous reading, whatever it was
        self._hold_refusals = 0
        self._report_on_charger = True  # what the battery report last said; the status stream can go stale
        self._delayed: set[int] = set()  # family messages the sender has been told are waiting for the robot
        self._warmed_for: int | str | None = None  # the heard request (or "checkin") the planner was last warmed up for
        self._body_lost_since: float | None = None
        self._off_dock_since: float | None = None
        self._gauge_trusted = True
        self._gauge_stale_since: float | None = None
        self._last_gauge_plea = float("-inf")
        self._body_rechecked = False  # a silent body is believed only after a look over a fresh connection
        self._last_return_home = float("-inf")
        self._intents: wirepod.IntentWatcher | None = None
        self._last_moving = float("-inf")

    # --- what is due ----------------------------------------------------------------------------------------------

    def _done_today(self, kind: str) -> bool:
        last = self.memory.last_event_time(kind)
        return bool(last and last[:10] == datetime.now().date().isoformat())

    @staticmethod
    def _todays_weather() -> str:
        """Fetched before he calls out. If it fails he still has the `weather` tool, as before."""
        try:
            return weather.report()
        except Exception as e:
            log(f"  (weather not fetched beforehand: {type(e).__name__}; the tool is still there)")
            return ""

    def _checkin_due(self) -> bool:
        if self.checkin_now:
            return True
        return datetime.now().strftime("%H:%M") >= config.checkin_time() and not self._done_today("checkin")

    def _warm_up_for_checkin(self, agent: Agent) -> None:
        """Prime the planner in the minute before the check-in: its first call is the slow one (8.4 s live)."""
        hh, mm = (int(part) for part in config.checkin_time().split(":"))
        seconds_to_go = (datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0) - datetime.now()).total_seconds()
        if 0 < seconds_to_go <= 60 and self._warmed_for != "checkin" and not self._done_today("checkin"):
            self._warmed_for = "checkin"
            agent.warm_up()


    # --- one pass -------------------------------------------------------------------------------------------------

    def _handle_family(self, agent: Agent) -> bool:
        """Deal with one waiting family message. Returns whether the robot was used."""
        row = self.memory.inbox_next()
        if row is None:
            return False
        update_id, chat_id, sender, text, received_at = row
        # If this raises (say the robot link drops mid-delivery) the message stays unhandled and is tried
        # again after the reconnect, up to three times.
        used_robot = self._handle_family_message(agent, chat_id, sender, text, received_at)
        self.memory.inbox_done(update_id)
        return used_robot

    def _handle_family_message(self, agent: Agent, chat_id: int, sender: str, text: str, received_at: str) -> bool:
        command = text.split()[0].lower()
        if command in FAMILY_COMMANDS:
            self._run_family_command(chat_id, command)
        else:
            log(f"family message from {sender}")
            self.memory.log_event("family_in", f"{sender}: {text}")
            waited = (datetime.now() - datetime.fromisoformat(received_at)).total_seconds() / 60
            late = f" It arrived {waited:.0f} minutes ago; mention that it is from earlier." if waited > 10 else ""
            prompt = f"[Message from family member {sender}, to deliver in person: \"{text}\".{late}]"
            log("done: " + agent.run(prompt, speaker="system"))
            return True
        return False

    def _presence(self, robot, agent: Agent) -> bool:
        """Keep track of whether the person has been around today. Returns whether the robot ran a task."""
        now, clock, today = time.monotonic(), datetime.now().strftime("%H:%M"), datetime.now().date()

        if self._faces.seen() and now - self._last_presence_logged > 15 * 60:
            self._last_presence_logged = now
            self.memory.log_event("presence", "face seen")

        if note := alerts.report_all_clear(self.memory, self.family):
            log(note)

        deadline = config.presence_deadline()
        if deadline.lower() == "off":
            return False
        seen = self.memory.last_sign_of_person(today) is not None and not self.presence_now
        if seen:
            return False

        if clock >= deadline or self.presence_now:
            if self._done_today("presence_check") and not self.presence_now:
                return False
            self.presence_now = False
            log("presence check: no sign of the person today; looking and calling out")
            self._faces.seen()  # discard anything seen before the check; only what happens from now counts
            started = datetime.now().isoformat(timespec="seconds")
            summary = agent.run(PRESENCE_CHECK, speaker="system")
            self.memory.log_event("presence_check", summary)
            log("done: " + summary)
            if self._faces.seen():  # the watcher keeps looking during the task; a glimpse while turning counts
                self.memory.log_event("presence", "face seen during the presence check")
            last = self.memory.last_sign_of_person(today)
            if last and last >= started:
                log("presence check: the person is here")
            elif getattr(self.ears, "seems_deaf", bool)():
                # He called out and his own microphone gave him nothing: that is no evidence about her.
                if alert := alerts.report_deaf(self.memory, self.family):
                    log(f"alert: {alert}")
            elif alert := alerts.report_absence(self.memory, self.family):
                log(f"alert: {alert}")
            # Only now the slow part. The family should not wait on a docking manoeuvre.
            if not config.stay_on_charger():  # otherwise he looked and called from his charger and never left it
                with kin_robot.control(robot):
                    log("going home: " + agent.tools.dispatch("go_home", "{}"))
            return True

        if clock >= config.presence_start() and now >= self._next_quiet_look:
            self._next_quiet_look = now + config.presence_look_minutes() * 60
            if robot.status.is_on_charger:
                found = kin_robot.quiet_look(robot)
                log("quiet look: " + ("saw someone" if found else "no one in view"))
                if found:
                    self.memory.log_event("presence", "face seen on a quiet look")
        return False


    @staticmethod
    def _acknowledge_touch(robot) -> None:
        """React the instant a pet registers. The planner takes a couple of seconds to think, and with no
        reaction in that gap people assume the pet failed and try again."""
        try:
            with kin_robot.control(robot):
                kin_robot.call(robot.anim.play_animation_trigger, "ReactToTouchInitial", timeout=8)
        except Exception:
            pass  # a courtesy; never let it block the conversation

    def _heartbeat(self, robot) -> None:
        """Check the link every so often. A dead link raises, and the run loop reconnects.

        Without this a dropped connection (Vector rebooting, Wi-Fi loss) goes unnoticed until the next task.
        """
        now = time.monotonic()
        if now < self._next_heartbeat:
            return
        self._next_heartbeat = now + HEARTBEAT_SECONDS
        battery = kin_robot.call(robot.get_battery_state, timeout=10)
        level = {1: "low", 2: "ok", 3: "full"}.get(battery.battery_level, "?")
        state = (level, battery.is_on_charger_platform)
        # On Sep 20 he ran flat and went dark four minutes after the firmware's own "low" flag, so the
        # voltage is watched as well.
        # Two readings in a row, because one can lie: right after a boot he reported "full" at 3.61 V.
        silent = (battery.battery_level == 3 and battery.battery_volts < 3.7  # worth a second of touch samples?
                  and kin_robot.body_silent(battery.battery_level, battery.battery_volts, kin_robot.touch_samples(robot)))
        if silent and not self._body_rechecked:
            # The touch readings come over this connection's event stream, and a stream that died quietly
            # shows the same frozen value as a dead body (Sep 23, 06:29: a false alert, the robot fine on
            # a fresh connection). So the first sighting is re-checked over a new connection.
            self._body_rechecked = True
            raise RuntimeError("body looks silent; reconnecting to check over a fresh link")
        if not silent:
            self._body_rechecked = False
        body_lost = self._watch_body(silent)
        # With his body silent the battery figures are stale, so they say nothing about running flat.
        gauge_read = kin_robot.gauge_read(battery, self._raw_volts, getattr(getattr(robot, "status", None), "is_on_charger", None))
        self._raw_volts = battery.battery_volts  # every reading, not only the logged steps: a frozen gauge shows here
        self._report_on_charger = bool(battery.is_on_charger_platform)
        if not gauge_read and self._gauge_trusted:
            log(f"battery telemetry stale ({battery.battery_volts:.2f} V pinned, or the charger state disagrees "
                "with the robot's status); not acting on it")
        self._gauge_trusted = gauge_read
        if gauge_read:
            self._gauge_stale_since = None
        elif self._gauge_stale_since is None:
            self._gauge_stale_since = now
        low_now = ((level == "low" or (gauge_read and battery.battery_volts < 3.65))
                   and not battery.is_on_charger_platform and not body_lost and gauge_read)
        self._low_readings = self._low_readings + 1 if low_now else 0
        self._running_flat = self._low_readings >= 2
        if battery.is_on_charger_platform or not gauge_read:
            self._off_dock_since = None
        elif self._off_dock_since is None:
            self._off_dock_since = now
        # A dropped link releases the idle hold, and a request made while the link is still shaky can go
        # unanswered; live, he wandered off the dock in that gap. So the hold is checked every heartbeat.
        if config.idle_hold() and not kin_robot.idle_held(robot) and not self._spoken_to() and self._still_for(20):
            if kin_robot.hold_idle(robot):
                log("holding him again")
                self._hold_refusals = 0
            else:
                self._hold_refusals += 1
                if self._hold_refusals == 1:
                    log("could not take the idle hold; will retry")
                if self._hold_refusals >= HOLD_REFUSALS_BEFORE_RECONNECT:
                    # Sep 24: refused every 30 s for 100 minutes on a half-dead link that still answered
                    # battery reads, so nothing reconnected, and he wandered off the dock in the meantime.
                    self._hold_refusals = 0
                    raise RuntimeError("the idle hold has been refused for 5 minutes; reconnecting")
        # Log on any change of level or charger state, and on each 0.05 V step, so the run-up to a reboot is visible.
        if state != self._battery or abs(battery.battery_volts - self._volts) >= 0.05:
            self._battery, self._volts = state, battery.battery_volts
            log(f"battery {level} ({battery.battery_volts:.2f} V), on charger: {battery.is_on_charger_platform}")

    def _spoken_to(self) -> bool:
        return self._intents is not None and self._intents.spoken_to()

    def _watch_movement(self, robot) -> None:
        """Note when he last moved or held his cube, so the hold does not cut a fetch short (Sep 23: a fetch
        stalled mid-carry when the hold came back a minute after the command)."""
        try:
            status = robot.status
            if status.are_motors_moving or status.is_carrying_block:
                self._last_moving = time.monotonic()
        except Exception:
            pass

    def _still_for(self, seconds: float) -> bool:
        return time.monotonic() - self._last_moving >= seconds

    def _let_him_react(self, robot) -> None:
        """Drop the idle hold the moment he hears his wake word, so his own reactions ("I love you", a fist
        bump) can run; the heartbeat takes the hold back a minute after the last thing he heard."""
        if kin_robot.idle_held(robot) and self._spoken_to():
            kin_robot.drop_hold(robot)
            log("spoken to: letting him react; the hold is back in a minute")

    def _watch_body(self, silent: bool) -> bool:
        """Follow whether Vector's body board is answering, and tell the family when it has not for a while.

        Brief losses recover by themselves (two of a few minutes each came before the one that stuck), and
        he reads the same way for a moment after a boot, so only a lasting one is reported."""
        now = time.monotonic()
        if not silent:
            if self._body_lost_since is not None:
                log("his body is answering again")
            self._body_lost_since = None
            # From memory, not the timer: a restart in between must not leave the family without the all-clear.
            if note := alerts.report_body_back(self.memory, self.family):
                log(note)
            return False
        if self._body_lost_since is None:
            self._body_lost_since = now
            log("his body is not answering (battery 'full' at an impossible voltage, touch sensor frozen); watching")
        minutes = (now - self._body_lost_since) / 60
        if minutes >= BODY_LOST_MINUTES:
            if alert := alerts.report_body_down(self.memory, self.family, minutes):
                log(f"alert: {alert}")
        return True

    def _handle_heard(self, agent: Agent) -> bool:
        """A request that came through Vector's own microphone (wake word or back button)."""
        row = self.memory.heard_next()
        if row is None:
            return False
        heard_id, text, ack, received_at = row
        age = (datetime.now() - datetime.fromisoformat(received_at)).total_seconds()
        if age < ACK_SECONDS:
            if self._warmed_for != heard_id:  # use the wait: the first planner call of a task is the slow one
                self._warmed_for = heard_id
                agent.warm_up()
            self._forget_touch()  # his back button is beside the touch sensor: pressing it is not a pet
            return False  # Vector is still saying his acknowledgement; wire-pod holds him until he has
        if age > HEARD_STALE_MINUTES * 60:
            log(f"heard earlier, too old to act on now: {text!r}")
            self.memory.heard_done(heard_id)
            return False
        log(f"heard through Vector: {text!r} (he said {ack!r})")
        self.memory.log_event("presence", "spoke to Vector")
        self.memory.heard_try(heard_id)  # counted before running, so a request that crashes Kin is not retried forever
        note = (f"This request was heard through Vector's own microphone after the wake word or his back button, "
                f"and he has already answered \"{ack}\" in his own voice. Do not greet or acknowledge again; go "
                "straight to helping. His microphone is weak, so the words may be garbled: if they make no sense, "
                "ask the person to say it again. From here on you hear them through a better microphone."
                + going_home())
        log("done: " + agent.run(text, speaker="person", note=note))
        self.memory.heard_done(heard_id)
        self._forget_touch()  # a hand on his back during the request is not a new request
        return True

    def _forget_touch(self) -> None:
        if self._touch is not None:
            self._touch.discard()

    def _return_home(self, robot, agent: Agent) -> bool:
        """Off the charger with a good battery and nothing to do: drive back onto it, quietly.

        He gets off it by himself in the gaps when the link is down, or is put down somewhere after a
        restart, and cannot be trusted to find his way back when the battery is finally low: he ran flat
        that way on Sep 21. Docking works from the table in daylight (twice on Sep 22), so it is tried
        early instead, once every so often, with no words unless it works."""
        if not config.stay_on_charger() or self._off_dock_since is None or self._running_flat:
            return False
        now = time.monotonic()
        if now - self._off_dock_since < OFF_DOCK_GRACE_SECONDS or now - self._last_return_home < RETURN_HOME_EVERY_SECONDS:
            return False
        self._last_return_home = now
        log("off the charger with nothing to do: trying to get back onto it")
        with kin_robot.control(robot):
            result = agent.tools.dispatch("go_home", "{}")
        log(f"go home: {result}")
        if result == "on the charger":
            self.memory.log_event("task", "drove back onto the charger by himself")
        return True

    def _guard_unreadable_battery(self, robot, agent: Agent) -> bool:
        """Off the charger with a gauge that has not given a real reading for a while: ask to be put on it,
        and tell the family later. Sep 23: the gauge read 0.00 V from a restart at 08:56 until he went dark
        at 09:44, 40 minutes off the dock, and ignoring the unreadable gauge meant nobody was warned."""
        # Only when BOTH sources say he is off the charger. Sep 24: the status stream said "off" for six
        # hours while the battery report said "on" (and he charged to full), and he asked every hour.
        status_on = getattr(getattr(robot, "status", None), "is_on_charger", True)
        if self._gauge_stale_since is None or status_on or self._report_on_charger:
            return False
        minutes = (time.monotonic() - self._gauge_stale_since) / 60
        if minutes >= GAUGE_UNREADABLE_MINUTES and time.monotonic() - self._last_gauge_plea > 3600:
            self._last_gauge_plea = time.monotonic()
            log("battery unreadable and off the charger: asking to be put on it")
            log("done: " + agent.run(GAUGE_UNREADABLE, speaker="system"))
            return True
        if minutes >= 2 * GAUGE_UNREADABLE_MINUTES:
            if alert := alerts.report_stranded(self.memory, self.family, minutes, reason="with a battery I cannot read"):
                log(f"alert: {alert}")
        return False

    def _rescue_battery(self, agent: Agent) -> bool:
        """Off the charger with a low battery: get home or get help, before he runs flat."""
        if not self._running_flat:
            self._flat_since = None
            return False
        now = time.monotonic()
        if self._flat_since is None:
            self._flat_since = now
            log("battery low and off the charger: going home or asking for help")
            log("done: " + agent.run(LOW_BATTERY, speaker="system"))
            return True
        minutes = (now - self._flat_since) / 60
        if minutes >= STRANDED_MINUTES:
            if alert := alerts.report_stranded(self.memory, self.family, minutes):
                log(f"alert: {alert}")
        return False

    def tick(self, robot, agent: Agent) -> bool:
        """Run at most one task. Returns whether one ran."""
        self._watch_movement(robot)
        self._heartbeat(robot)
        self._let_him_react(robot)
        if kin_robot.busy_being_himself():  # one of his own behaviours is running; do not take him back
            self.desk_work()
            return False
        if self._rescue_battery(agent):
            return True
        if self._guard_unreadable_battery(robot, agent):
            return True
        if self.family and self._handle_family(agent):
            return True
        if self._handle_heard(agent):
            return True

        if self._touch.petted():
            talk = config.pet_to_talk()
            if self._touch.resting_on_something():
                log("his back has been pressed against something for a while; not a pet")
                return False
            log("petted -> conversation" if talk else "petted")
            # A pet is proof the person is here; say so at once rather than after the conversation.
            self.memory.log_event("presence", "petted")
            if note := alerts.report_all_clear(self.memory, self.family):
                log(note)
            if not talk:
                return False  # a pet is just a pet: he purrs by himself. Talking starts with "Hey Vector" or his button
            agent.warm_up()
            self._acknowledge_touch(robot)
            log("done: " + agent.run(TOUCHED + going_home() + "]", speaker="system"))
            self._touch.discard()
            return True

        if self._presence(robot, agent):
            return True
        if self._return_home(robot, agent):
            return True

        self._warm_up_for_checkin(agent)
        if self._checkin_due():
            self.checkin_now = False
            log("check-in")
            summary = agent.run(checkin_prompt(self._todays_weather(), going_home()), speaker="system")
            self.memory.log_event("checkin", summary)
            log("done: " + summary)
            return True

        for reminder in self.memory.reminders_due(
            renudge_minutes=config.renudge_minutes(), max_attempts=config.reminder_attempts()
        ):
            attempt = 1 + sum(
                1 for _, text, outcome in self.memory.reminder_history(datetime.now().date())
                if text == reminder.text and outcome == "no_response"
            )
            log(f"reminder #{reminder.id} '{reminder.text}' (attempt {attempt})")
            self.memory.log_reminder(reminder.id, "asked")
            agent.tools.current_reminder = reminder.id
            try:
                log("done: " + agent.run(reminder_prompt(reminder, attempt), speaker="system"))
            finally:
                agent.tools.current_reminder = None
            return True  # one task per tick; the next is picked up on the following pass

        for alert in alerts.report_missed_reminders(self.memory, self.family):  # normally raised by the tool already
            log(f"alert: {alert}")

        self.desk_work()
        return False

    def desk_work(self) -> None:
        """Everything that needs no robot: the family's questions, the evening note, the nightly reflection.

        Run whether or not the robot is reachable. He was unreachable for half of a three-day stretch
        (Sep 29 to Oct 2), and because all of this used to live inside the robot's connection the family
        got nothing at all on those days: no evening note, and no answer even to `/status`. The half of
        the product that still works when the hardware is sick should keep working.
        """
        if self.family:
            self._answer_family_questions()
        if self.family and datetime.now().strftime("%H:%M") >= config.digest_time() and not self._done_today("digest"):
            log("sending the family digest")
            note = digest.build(self.memory)
            sent = self.family.send(note)
            self.memory.log_event("digest_note", note)  # the weekly trends are built from these short notes
            self.memory.log_event("digest", f"sent to {sent} chat(s)")
            if datetime.now().strftime("%a").lower() == config.weekly_day():
                self._send_weekly_note()

        if datetime.now().strftime("%H:%M") >= NIGHTLY_AT and not self._done_today("nightly"):
            log("nightly reflection: memory, weekly trends, self-check")
            self.memory.log_event("nightly", "started")  # marked first, so a failure is not retried every tick
            summary = nightly.run_locally(self.memory)
            log("done: " + summary)
            for _, kind, detail in self.memory.events_on(datetime.now().date()):
                if kind == "self_check" and detail.startswith("[high]"):
                    log("SELF-CHECK " + detail)

    def _answer_family_questions(self, robot_reachable: bool = True) -> None:
        """Answer every waiting /command, and when the robot is out of reach say so once for a message
        that has to be delivered by voice, instead of leaving the sender in silence."""
        while (row := self.memory.inbox_next()) is not None:
            update_id, chat_id, sender, text, _ = row
            if text.split()[0].lower() in FAMILY_COMMANDS:
                self._run_family_command(chat_id, text.split()[0].lower())
                self.memory.inbox_done(update_id)
                continue
            if robot_reachable:
                return  # a message to deliver in person: the robot path takes it from here
            if update_id not in self._delayed:
                self._delayed.add(update_id)
                log(f"family message from {sender} held until the robot is back")
                self.family.send(
                    "Thank you, I have your message. I cannot reach my robot at the moment, so I will "
                    "deliver it as soon as he is back and tell you what she says.", chat_id=chat_id)
            return  # leave it in the inbox, in order, for delivery later

    def _run_family_command(self, chat_id: int, command: str) -> None:
        if command in ("/start", "/help"):
            self.family.send(
                "I'm Kin. Send me any message and I will deliver it in person and tell you the reply.\n"
                "/status - how today is going\n/digest - today's note now\n/week - patterns over the last week",
                chat_id=chat_id)
        elif command == "/status":
            self.family.send(digest.status(self.memory), chat_id=chat_id)
        elif command == "/digest":
            self.family.send(digest.build(self.memory), chat_id=chat_id)
        elif command == "/week":
            self._send_weekly_note(chat_id=chat_id)

    def _send_weekly_note(self, chat_id: int | None = None) -> None:
        """The weekly note, on record. Until Oct 9 the send was not logged, so memory could not tell
        whether the family had ever received one."""
        sent = self.family.send("Kin's weekly note\n\n" + nightly.latest_trends(self.memory), chat_id=chat_id)
        self.memory.log_event("weekly", f"sent to {sent} chat(s)" + (" on request" if chat_id else ""))

    def run(self) -> None:
        log(f"Kin service: mouth={config.mouth()} ears={config.ears()} check-in at {config.checkin_time()} "
            f"family={'on, ' + str(len(config.family_chat_ids())) + ' chat(s)' if self.family else 'off'}")
        for name, (in_env, in_file) in config.overridden().items():
            log(f"note: {name}={in_env} comes from the environment and overrides .env ({in_file or 'empty'})")
        if self.family:
            self.family.start()
        if config.gateway_port():
            try:
                gateway.start()
                self._intents = wirepod.IntentWatcher()
                log(f"gateway on 127.0.0.1:{config.gateway_port()} for speech wire-pod does not recognise; "
                    "watching Vector's built-in commands")
            except OSError as e:
                log(f"gateway not started: {e}")
        down_since: float | None = None
        while True:
            try:
                with kin_robot.connect(take_control=False) as robot:
                    kin_robot.call(robot.get_battery_state, timeout=10)  # a robot that connects but will not answer is not up
                    down_since = None
                    if note := alerts.report_robot_back(self.memory, self.family):
                        log(note)
                    self._touch = kin_robot.TouchWatcher(robot)
                    agent = Agent(robot, self.memory, voice.make_mouth(robot), self.ears, self.family,
                                  attention=self._touch.petted)
                    self._faces = kin_robot.FaceWatcher(robot)
                    self._next_heartbeat = 0.0
                    held = kin_robot.hold_idle(robot)
                    log("connected to Vector" + (", holding him between tasks" if held else ""))
                    # A link that dropped mid-task can leave a looped animation (and its sound) running
                    # with nothing to end it under the hold. A task that does nothing ends it on the way out.
                    with kin_robot.control(robot):
                        pass
                    try:
                        while True:
                            if self.tick(robot, agent) and self.once:
                                return
                            time.sleep(0.5)
                    finally:
                        self._touch.stop()
                        self._faces.stop()
            except KeyboardInterrupt:
                log("stopping")
                return
            except Exception as e:  # a dropped link or a bad task must not end the service
                where = " <- ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)[-3:][::-1])
                log(f"error: {type(e).__name__}: {str(e)[:120]} (at {where}); reconnecting in {RECONNECT_SECONDS}s")
                down_since = down_since or time.monotonic()
                minutes = (time.monotonic() - down_since) / 60
                if minutes >= ROBOT_DOWN_MINUTES:
                    if alert := alerts.report_robot_down(self.memory, self.family, minutes):
                        log(f"alert: {alert}")
                # The family side does not depend on the robot, so it carries on while he is out of reach.
                try:
                    self._answer_family_questions(robot_reachable=False)
                    self.desk_work()
                except Exception as desk:  # never let this stop the reconnecting
                    log(f"error (while the robot is down): {type(desk).__name__}: {str(desk)[:120]}")
                time.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    Service(
        Memory(),
        checkin_now="--checkin-now" in sys.argv,
        once="--once" in sys.argv,
        presence_now="--presence-now" in sys.argv,
    ).run()
