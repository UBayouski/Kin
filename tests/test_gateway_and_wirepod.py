import contextlib
import json
from datetime import datetime

import httpx
import pytest
from openai import OpenAI

from kin import config, gateway, wirepod
from kin.memory import Memory


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The gateway on a free port, writing to a database in the test's own directory."""
    monkeypatch.setenv("KIN_DB_PATH", str(tmp_path / "kin.db"))
    server = gateway.start(port=0)
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_vector_acknowledges_in_his_own_words_and_the_request_is_queued(served, tmp_path):
    client = OpenAI(base_url=served, api_key="kin-local")  # wire-pod talks to it the same way, streaming

    stream = client.chat.completions.create(model="kin", stream=True, messages=[
        {"role": "system", "content": "You are a helpful, animated robot called Vector."},
        {"role": "user", "content": "  what is on tv   tonight "},
    ])
    spoken = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert spoken in config.ack_phrases()
    heard_id, text, ack, _ = Memory(tmp_path / "kin.db").heard_next()
    assert text == "what is on tv tonight" and ack == spoken


def test_the_gateway_also_answers_a_plain_request(served):
    reply = OpenAI(base_url=served, api_key="x").chat.completions.create(
        model="kin", messages=[{"role": "user", "content": "tell me a joke"}])

    assert reply.choices[0].message.content in config.ack_phrases()


def test_nothing_heard_means_nothing_queued(served, tmp_path):
    reply = OpenAI(base_url=served, api_key="x").chat.completions.create(
        model="kin", messages=[{"role": "system", "content": "only a system prompt"}])

    assert reply.choices[0].message.content == "Sorry, I did not catch that."
    assert Memory(tmp_path / "kin.db").heard_next() is None


def test_acknowledgements_vary_and_can_be_customised(monkeypatch):
    said = [gateway.pick_ack() for _ in range(30)]
    assert all(a != b for a, b in zip(said, said[1:])), "never the same phrase twice running"
    assert len(set(said)) > 3

    monkeypatch.setenv("KIN_ACK_PHRASES", "Righto! | Leave it with me.")
    assert {gateway.pick_ack() for _ in range(10)} == {"Righto!", "Leave it with me."}


def test_the_ack_is_available_to_wire_pod_scripts(served):
    assert httpx.get(served.removesuffix("/v1") + "/ack").text in config.ack_phrases()


def test_a_heard_request_is_durable_and_cannot_be_retried_forever(memory):
    heard_id = memory.heard_add("what is on tv tonight", "On it!")

    assert memory.heard_next()[:3] == (heard_id, "what is on tv tonight", "On it!")
    memory.heard_try(heard_id)
    assert memory.heard_next() is not None, "one failed try leaves it queued"
    memory.heard_try(heard_id)
    assert memory.heard_next() is None, "two and it is dropped"

    other = memory.heard_add("and the news", "Got it.")
    memory.heard_done(other)
    assert memory.heard_next() is None


def test_pressing_his_back_button_is_not_also_a_pet(memory, monkeypatch):
    """The button is beside the touch sensor. Live, the answer was followed by a second, pet-started
    conversation asking what the person wanted."""
    from types import SimpleNamespace

    from kin import service

    class Touch:
        pending = True

        def discard(self):
            self.pending = False

    kin = service.Service.__new__(service.Service)  # only the heard path; no voice, robot or family
    kin.memory, kin._touch, kin._warmed_for = memory, Touch(), None
    ran = []
    agent = SimpleNamespace(warm_up=lambda: ran.append("warm"), run=lambda *a, **k: ran.append("run") or "ok")

    memory.heard_add("should i take a jacket", "Let me think.")
    assert kin._handle_heard(agent) is False, "Vector is still saying his acknowledgement"
    assert ran == ["warm"] and not kin._touch.pending, "the wait warms the planner and forgets the press"

    kin._touch.pending = True  # a hand on his back while Kin was answering
    monkeypatch.setattr(service, "ACK_SECONDS", -60)  # the test clock runs a little ahead of the real one
    assert kin._handle_heard(agent) is True
    assert ran == ["warm", "run"] and not kin._touch.pending
    assert memory.heard_next() is None


def test_a_pet_is_just_a_pet_unless_pet_to_talk_is_on(memory, monkeypatch):
    from types import SimpleNamespace

    from kin import service

    class Touch:
        def __init__(self):
            self.pets = 1

        def petted(self):
            self.pets -= 1
            return self.pets >= 0

        def resting_on_something(self):
            return False

        def discard(self):
            pass

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family, kin._touch, kin._warmed_for = memory, None, Touch(), None
    kin._gauge_stale_since, kin._last_gauge_plea = None, float("-inf")
    for quiet in ("_heartbeat", "_rescue_battery", "_presence", "_checkin_due", "_acknowledge_touch"):
        monkeypatch.setattr(kin, quiet, lambda *a, **k: False, raising=False)
    monkeypatch.setattr(memory, "reminders_due", lambda *a, **k: [])
    prompts = []
    agent = SimpleNamespace(warm_up=lambda: None, run=lambda text, **k: prompts.append(text) or "ok")

    kin.tick(None, agent)
    assert prompts == [], "no conversation"
    assert memory.last_sign_of_person(datetime.now().date()), "but it still shows she is around"

    monkeypatch.setenv("KIN_PET_TO_TALK", "on")
    kin._touch = Touch()
    assert kin.tick(None, agent) is True
    assert "go_home" not in prompts[0]
    from kin.agent import Agent
    assert "need anything else" in Agent._system_prompt(), "the closing offer is a rule for every conversation"


def test_after_a_task_that_left_the_charger_the_service_docks_him_and_asks_for_a_lift_once(memory, monkeypatch):
    """Oct 16: the check-in told the planner to go home when finished. After "No, thank you so much!" it went to
    dock instead of saying goodbye, could not see the charger, and asked her twice to carry him."""
    from types import SimpleNamespace

    from kin import service
    from kin.memory import Reminder
    from kin.tools import ASK_FOR_A_LIFT

    monkeypatch.setenv("KIN_STAY_ON_CHARGER", "off")
    assert "go_home" not in service.reminder_prompt(Reminder(id=1, text="drink water", time_of_day="18:00", days="daily"), 1)
    assert "go_home" not in service.checkin_prompt()

    robot = SimpleNamespace(status=SimpleNamespace(is_on_charger=True))
    calls = []

    def dispatch(name, arguments):
        calls.append((name, json.loads(arguments)))
        return "cannot see the charger from here. " + ASK_FOR_A_LIFT if name == "go_home" else "said"

    def task(*args):
        robot.status.is_on_charger = False  # looking for her took him off it
        return True

    kin = service.Service.__new__(service.Service)
    kin.memory = memory
    monkeypatch.setattr(kin, "_tick", task)
    monkeypatch.setattr(service.kin_robot, "control", lambda robot: contextlib.nullcontext())
    agent = SimpleNamespace(tools=SimpleNamespace(dispatch=dispatch))

    assert kin.tick(robot, agent) is True
    assert calls == [("go_home", {}), ("say", {"text": service.LIFT_PLEA})]
    assert not service.LIFT_PLEA.rstrip().endswith("?"), "a statement, so nobody is waited on"

    calls.clear()
    assert kin.tick(robot, agent) is True
    assert calls == [], "he was already off the charger when the task began: that is for the quiet return home"

    robot.status.is_on_charger = True
    monkeypatch.setenv("KIN_STAY_ON_CHARGER", "on")
    kin.tick(robot, agent)
    assert calls == [], "staying on the charger, he never leaves it on a task"


LOG = """\
2026.09.20 11:02:01: Bot 0dd19176 Stream type: OPUS
2026.09.20 11:02:03: Bot 0dd19176 Transcribed text: i love you
2026.09.20 11:02:03: Not a custom intent
2026.09.20 11:02:03: Bot 0dd19176 Intent Sent: intent_imperative_love
2026.09.20 11:02:03: Bot 0dd19176 request served.
2026.09.20 11:05:40: Bot 0dd19176 Transcribed text: what is on tv tonight
2026.09.20 11:05:40: Not a custom intent
2026.09.20 11:05:41: Making LLM request for device 0dd19176...
2026.09.20 11:07:12: Bot 0dd19176 Transcribed text: mumble
2026.09.20 11:07:12: Bot 0dd19176 Intent Sent: intent_system_unmatched
2026.09.20 11:09:30: Bot 0dd19176 Transcribed text: do a wheelie
2026.09.20 11:09:30: Bot 0dd19176 Intent Sent: intent_imperative_wheelie
""".splitlines()


def test_built_in_commands_are_recognised_in_wire_pods_log():
    assert wirepod.interactions(LOG) == [("i love you", "intent_imperative_love"),
                                         ("do a wheelie", "intent_imperative_wheelie")]
    assert wirepod.describe("intent_imperative_love") == "told Vector they love him"
    assert wirepod.describe("intent_imperative_wheelie") == "used Vector's built-in command imperative_wheelie"


def test_a_weather_question_is_handed_to_kin_with_the_words_from_wire_pods_log(served, tmp_path, monkeypatch):
    assert wirepod.last_transcript(LOG, now="2026.09.20 11:09:34") == "do a wheelie"
    assert wirepod.last_transcript(LOG, now="2026.09.20 11:12:00") == "", "stale: not what he heard just now"
    assert wirepod.last_transcript([], now="2026.09.20 11:12:00") == ""

    monkeypatch.setattr(wirepod, "heard_just_now", lambda: "what is the weather today")
    ack = httpx.get(served.removesuffix("/v1") + "/handover").text
    assert ack in config.ack_phrases(), "Vector says this himself"
    assert Memory(tmp_path / "kin.db").heard_next()[1:3] == ("what is the weather today", ack)

    monkeypatch.setattr(wirepod, "heard_just_now", lambda: "")
    assert httpx.get(served.removesuffix("/v1") + "/handover").text == "", "nothing heard: he says nothing"


def test_the_takeover_claims_weather_words_without_catching_others():
    said = "should i catch the train to the brain clinic"
    assert not any(words in said for words in wirepod.TAKEOVER_WORDS)
    assert any(words in "what's the weather like" for words in wirepod.TAKEOVER_WORDS)
    assert "/handover" in wirepod.takeover_script()


def test_the_watcher_reports_each_log_line_once_and_ignores_history():
    watcher = wirepod.IntentWatcher(start=False)

    assert watcher.fresh(LOG[:5]) == [], "whatever is in the log when Kin starts is history"
    assert watcher.fresh(LOG[:8]) == LOG[5:8]
    assert watcher.fresh(LOG[:8]) == []
    assert watcher.fresh(LOG[3:]) == LOG[8:], "wire-pod trimmed the start of its log; nothing is repeated"


def test_what_the_person_says_to_vector_himself_reaches_kins_context(memory):
    from tests.test_agent_loop import make_agent, reply

    memory.log_event("vector_intent", 'told Vector they love him: "i love you"')
    agent, _, _ = make_agent(memory, [reply(content="ok")])

    context = agent._context()

    assert 'told Vector they love him: "i love you"' in context
    assert datetime.now().strftime("%H:%M")[:2] in context


def test_a_battery_gauge_not_yet_read_is_not_a_low_battery(memory, family, monkeypatch):
    """Right after a boot he reports 0.00 V while "full" (Sep 22, 13:08); it set off the low-battery rescue
    and he asked to be put on a charger he was already near. A gauge under 1 V has simply not been read."""
    from types import SimpleNamespace

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family
    kin._low_readings, kin._running_flat, kin._body_lost_since, kin._off_dock_since = 0, False, None, None
    kin._gauge_trusted, kin._last_moving, kin._intents, kin._gauge_stale_since = True, float("-inf"), None, None
    monkeypatch.setenv("KIN_IDLE_HOLD", "off")
    kin._next_heartbeat, kin._battery, kin._volts, kin._raw_volts, kin._hold_refusals = 0.0, None, 0.0, -1.0, 0
    battery = SimpleNamespace(battery_level=3, battery_volts=0.0, is_on_charger_platform=False)
    robot = SimpleNamespace(get_battery_state=lambda: battery, touch=SimpleNamespace(last_sensor_reading=None))
    monkeypatch.setattr(service.kin_robot, "call", lambda method, *a, **k: method())
    monkeypatch.setattr(service.kin_robot, "touch_samples", lambda robot: [5470, 5473, 5469])

    for _ in range(3):
        kin._next_heartbeat = 0.0
        kin._heartbeat(robot)
    assert kin._running_flat is False

    battery.battery_volts, battery.battery_level = 3.50, 2  # a real reading that low, off the charger, still counts
    for _ in range(2):
        kin._next_heartbeat = 0.0
        kin._heartbeat(robot)
    assert kin._running_flat is True


def test_off_the_charger_with_nothing_to_do_he_drives_back_by_himself(memory, monkeypatch):
    """He gets off the dock in the gaps when the link is down (Sep 22, 13:49) and cannot be trusted to find
    it once the battery is low; docking from the table works in daylight, so it is tried early."""
    from types import SimpleNamespace

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory = memory
    kin._off_dock_since, kin._running_flat, kin._last_return_home = None, False, float("-inf")
    clock = {"now": 1000.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(service.kin_robot, "control", lambda robot: __import__("contextlib").nullcontext())
    results = ["cannot see the charger from here.", "on the charger"]
    agent = SimpleNamespace(tools=SimpleNamespace(dispatch=lambda name, args: results.pop(0)))

    assert kin._return_home(None, agent) is False, "on the charger: nothing to do"
    kin._off_dock_since = clock["now"]
    assert kin._return_home(None, agent) is False, "just got off: a grace period, someone may be holding him"
    clock["now"] += service.OFF_DOCK_GRACE_SECONDS + 1
    assert kin._return_home(None, agent) is True
    clock["now"] += 60
    assert kin._return_home(None, agent) is False, "a failed try is not repeated straight away"
    clock["now"] += service.RETURN_HOME_EVERY_SECONDS
    assert kin._return_home(None, agent) is True
    assert any("drove back onto the charger" in d for _, k, d in memory.events_on(__import__("datetime").date.today()))

    kin._running_flat = True
    clock["now"] += service.RETURN_HOME_EVERY_SECONDS
    assert kin._return_home(None, agent) is False, "a low battery is the rescue's job, with the person told"


def test_a_silent_body_is_believed_only_after_a_look_over_a_fresh_connection(memory, family, monkeypatch):
    """Sep 23, 06:29: the service's own event stream had died quietly, its touch readings froze, and the
    family was told the robot needed restarting while he slept soundly on his dock."""
    from types import SimpleNamespace

    import pytest

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family
    kin._low_readings, kin._running_flat, kin._body_lost_since, kin._off_dock_since = 0, False, None, None
    kin._gauge_trusted, kin._body_rechecked, kin._last_moving, kin._intents = True, False, float("-inf"), None
    kin._gauge_stale_since = None
    kin._next_heartbeat, kin._battery, kin._volts, kin._raw_volts, kin._hold_refusals = 0.0, None, 0.0, -1.0, 0
    monkeypatch.setenv("KIN_IDLE_HOLD", "off")
    battery = SimpleNamespace(battery_level=3, battery_volts=3.60, is_on_charger_platform=True)
    robot = SimpleNamespace(get_battery_state=lambda: battery, status=SimpleNamespace(is_on_charger=True))
    monkeypatch.setattr(service.kin_robot, "call", lambda method, *a, **k: method())
    monkeypatch.setattr(service.kin_robot, "touch_samples", lambda robot: [5491] * 20)

    with pytest.raises(RuntimeError, match="fresh link"):
        kin._heartbeat(robot)  # first sighting: reconnect and look again, no timer started
    assert kin._body_lost_since is None

    kin._next_heartbeat = 0.0
    kin._heartbeat(robot)  # still frozen over the new connection: now the clock runs
    assert kin._body_lost_since is not None

    monkeypatch.setattr(service.kin_robot, "touch_samples", lambda robot: [5455, 5458, 5463])
    kin._next_heartbeat = 0.0
    kin._heartbeat(robot)
    assert kin._body_lost_since is None and kin._body_rechecked is False, "alive again, and the next episode is re-checked too"


def test_hearing_the_wake_word_drops_the_idle_hold_for_a_minute(memory, monkeypatch):
    """The hold blocked his own reactions: "I love you" got nothing (Sep 23). wire-pod logs the start of
    listening before it sends the intent, and that line is the cue to get out of his way."""
    from types import SimpleNamespace

    from kin import service

    watcher = wirepod.IntentWatcher(start=False)
    watcher.fresh(LOG[:3])  # history
    assert not watcher.spoken_to()
    watcher.fresh(LOG[:3] + ["2026.09.23 09:10:00: Bot 0dd19176 Stream type: OPUS"])
    assert watcher.spoken_to()

    kin = service.Service.__new__(service.Service)
    kin._intents = watcher
    dropped = []
    monkeypatch.setattr(service.kin_robot, "idle_held", lambda robot: True)
    monkeypatch.setattr(service.kin_robot, "drop_hold", lambda robot: dropped.append(robot))
    kin._let_him_react("robot")
    assert dropped == ["robot"]

    monkeypatch.setattr(service.time, "monotonic", lambda: watcher.last_wake + 61)
    assert not watcher.spoken_to(), "a minute later the heartbeat may take the hold back"


def test_the_hold_waits_until_he_has_been_still_for_a_while(monkeypatch):
    """A fetch-the-cube stalled mid-carry when the hold came back (Sep 23)."""
    from types import SimpleNamespace

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin._last_moving = float("-inf")
    clock = {"now": 1000.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: clock["now"])

    kin._watch_movement(SimpleNamespace(status=SimpleNamespace(are_motors_moving=False, is_carrying_block=True)))
    assert not kin._still_for(20)
    clock["now"] += 19
    assert not kin._still_for(20)
    clock["now"] += 2
    assert kin._still_for(20)


def test_an_unreadable_battery_off_the_charger_gets_a_plea_and_then_the_family(memory, family, monkeypatch):
    """Sep 23: the gauge read 0.00 V for 40 minutes off the dock and he went dark with nobody warned."""
    from types import SimpleNamespace

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family
    kin._gauge_stale_since, kin._last_gauge_plea, kin._report_on_charger = 1000.0, float("-inf"), False
    clock = {"now": 1000.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: clock["now"])
    robot = SimpleNamespace(status=SimpleNamespace(is_on_charger=False))
    pleas = []
    agent = SimpleNamespace(run=lambda text, **k: pleas.append(text) or "asked")

    clock["now"] += 5 * 60
    assert kin._guard_unreadable_battery(robot, agent) is False
    clock["now"] += 6 * 60
    assert kin._guard_unreadable_battery(robot, agent) is True and "cannot read your own battery" in pleas[0]
    assert kin._guard_unreadable_battery(robot, agent) is False, "asked once, not every tick"
    clock["now"] += 10 * 60
    kin._guard_unreadable_battery(robot, agent)
    assert len(family.sent) == 1 and "cannot read" in family.sent[0]

    robot.status.is_on_charger = True
    clock["now"] += 3600
    assert kin._guard_unreadable_battery(robot, agent) is False, "on the charger there is nothing to ask"
    robot.status.is_on_charger, kin._report_on_charger = False, True
    clock["now"] += 3600
    assert kin._guard_unreadable_battery(robot, agent) is False, "the status stream alone is not believed (Sep 24: six false pleas)"


def test_a_hold_refused_for_five_minutes_forces_a_reconnect(memory, family, monkeypatch):
    """Sep 24: refused every 30 s for 100 minutes on a half-dead link that still answered battery reads."""
    from types import SimpleNamespace

    import pytest

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family
    kin._low_readings, kin._running_flat, kin._body_lost_since, kin._off_dock_since = 0, False, None, None
    kin._gauge_trusted, kin._body_rechecked, kin._last_moving, kin._intents = True, False, float("-inf"), None
    kin._gauge_stale_since, kin._hold_refusals = None, 0
    kin._next_heartbeat, kin._battery, kin._volts, kin._raw_volts = 0.0, None, 0.0, -1.0
    battery = SimpleNamespace(battery_level=3, battery_volts=4.05, is_on_charger_platform=True)
    robot = SimpleNamespace(get_battery_state=lambda: battery, status=SimpleNamespace(is_on_charger=True))
    monkeypatch.setattr(service.kin_robot, "call", lambda method, *a, **k: method())
    monkeypatch.setattr(service.kin_robot, "idle_held", lambda robot: False)
    monkeypatch.setattr(service.kin_robot, "hold_idle", lambda robot: False)

    for _ in range(service.HOLD_REFUSALS_BEFORE_RECONNECT - 1):
        kin._next_heartbeat = 0.0
        battery.battery_volts += 0.001
        kin._heartbeat(robot)
    kin._next_heartbeat = 0.0
    battery.battery_volts += 0.001
    with pytest.raises(RuntimeError, match="refused for 5 minutes"):
        kin._heartbeat(robot)


def desk(memory, family, monkeypatch):
    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family, kin._delayed = memory, family, set()
    monkeypatch.setenv("KIN_DIGEST_TIME", "23:59")  # not due during the test
    return kin, service


def test_the_family_still_gets_answers_while_the_robot_is_out_of_reach(memory, family, monkeypatch):
    """Sep 30: the robot was unreachable for 20 hours, and because all of this lived inside his connection
    the family got no evening note and no answer even to /status."""
    kin, service = desk(memory, family, monkeypatch)
    memory.inbox_add(1, 55, "Mila", "/status")
    memory.inbox_add(2, 55, "Mila", "Tell mum I'll call at six")

    kin._answer_family_questions(robot_reachable=False)

    assert len(family.sent) == 2
    assert "Today" in family.sent[0], "/status answered from memory, no robot involved"
    assert "cannot reach my robot" in family.sent[1], "and the message to deliver is acknowledged, not ignored"
    assert memory.inbox_next()[0] == 2, "it stays queued, in order, for delivery later"

    kin._answer_family_questions(robot_reachable=False)
    assert len(family.sent) == 2, "the sender is told once, not every reconnect"


def test_with_the_robot_reachable_a_message_is_left_for_the_voice_path(memory, family, monkeypatch):
    kin, service = desk(memory, family, monkeypatch)
    memory.inbox_add(1, 55, "Mila", "/help")
    memory.inbox_add(2, 55, "Mila", "Tell mum I'll call at six")

    kin._answer_family_questions(robot_reachable=True)

    assert len(family.sent) == 1 and "I'm Kin" in family.sent[0]
    assert memory.inbox_next()[0] == 2 and not kin._delayed


def test_a_day_kin_was_blind_is_not_reported_as_a_quiet_day(memory):
    """"A quiet day" would be a false reassurance when the truth is that he could not see her at all."""
    from datetime import date

    from kin import digest

    assert "a quiet day" in digest.build(memory, date.today())

    memory.log_event("robot_down", "I have not been able to reach the robot for 768 minutes. Reminders ...")
    note = digest.build(memory, date.today())

    assert "could not reach my robot" in note and "12 hour(s)" in note
    assert "quiet day" not in note
    assert "not a sign" in note, "and it says plainly that it is a fault at Kin's end"


def test_a_garbled_trick_request_is_handed_back_to_vectors_own_body(memory, monkeypatch):
    """His microphone turns "fetch the cube" into "french cube" (Sep 23) and the request reached Kin, who
    had no way to do it. wire-pod's own control page fires these intents, so Kin can too."""
    from types import SimpleNamespace

    from kin import robot as kin_robot
    from kin import tools, wirepod
    from kin.tools import RobotTools

    asked, released = [], []
    monkeypatch.setattr(wirepod, "run_intent", lambda intent: asked.append(intent) or True)
    monkeypatch.setattr(kin_robot, "hand_back", lambda robot: released.append(True))
    robot = SimpleNamespace(conn=SimpleNamespace(release_control=lambda timeout=0: None))
    kit = RobotTools(robot, memory)

    assert "vector_trick" in [spec["function"]["name"] for spec in tools.tool_specs()]
    result = kit.dispatch("vector_trick", json.dumps({"trick": "fetch_cube"}))

    assert asked == ["intent_imperative_fetchcube"], "the intent wire-pod's own UI uses"
    assert released == [True], "Kin lets go of control, or the behaviour cannot run"
    assert "doing it himself" in result
    assert kin_robot.busy_being_himself(), "and he is left alone while it runs"

    assert kit.dispatch("vector_trick", json.dumps({"trick": "backflip"})).startswith("error:")
    assert asked == ["intent_imperative_fetchcube"]


def test_while_he_is_doing_his_own_trick_kin_starts_no_task(memory, monkeypatch):
    from types import SimpleNamespace

    from kin import service

    kin = service.Service.__new__(service.Service)
    monkeypatch.setattr(service.kin_robot, "busy_being_himself", lambda: True)
    for quiet in ("_watch_movement", "_heartbeat", "_let_him_react", "desk_work"):
        monkeypatch.setattr(kin, quiet, lambda *a, **k: None, raising=False)
    kin._touch = SimpleNamespace(petted=lambda: pytest.fail("a task must not start mid-trick"))

    assert kin.tick(None, None) is False


def test_which_robot_to_talk_to_is_pinned_once_there_are_two(monkeypatch):
    """A second robot is being bought (Oct 3). wire-pod keeps every bot it has paired and the SDK picks one
    by itself, so without a pinned serial an intent could go to the dead one."""
    from kin import wirepod

    monkeypatch.delenv("VECTOR_SERIAL", raising=False)  # the real .env pins one; this test is about the unpinned case

    class Reply:
        def __init__(self, esns): self._esns = esns
        def json(self): return {"robots": [{"esn": e} for e in self._esns]}

    monkeypatch.setattr(wirepod.httpx, "get", lambda *a, **k: Reply(["0dd19176"]))
    assert wirepod.robot_serial() == "0dd19176", "one robot needs no setting"

    monkeypatch.setattr(wirepod.httpx, "get", lambda *a, **k: Reply(["0dd19176", "00e20abc"]))
    with pytest.raises(RuntimeError, match="set VECTOR_SERIAL"):
        wirepod.robot_serial()

    monkeypatch.setenv("VECTOR_SERIAL", "00e20abc")
    assert wirepod.robot_serial() == "00e20abc"


def test_the_checkin_carries_todays_weather_so_nothing_is_looked_up_mid_conversation(monkeypatch):
    """Oct 9, 22:29 on the Linux hub: after "Yes, I'm here" the planner fetched the weather and did a recall
    before greeting her, eight seconds of silence. The weather now travels in the note itself."""
    from kin import service

    prompt = service.checkin_prompt("San Jose. Now: 74 degrees Fahrenheit, clear sky.", " Stay on the charger.")
    assert prompt.startswith("[Daily check-in.") and prompt.endswith(" Stay on the charger.]")
    assert "Today's weather, already looked up: San Jose. Now: 74 degrees Fahrenheit, clear sky." in prompt
    assert "`weather` tool" not in prompt, "the note no longer sends him to the tool"

    assert "already looked up" not in service.checkin_prompt("", ""), "no weather, no claim of one"

    monkeypatch.setattr(service.weather, "report", lambda place="": (_ for _ in ()).throw(ConnectionError("offline")))
    assert service.Service._todays_weather() == "", "a failed fetch leaves the tool as the fallback, quietly"
    monkeypatch.setattr(service.weather, "report", lambda place="": "Now: 60 degrees.")
    assert service.Service._todays_weather() == "Now: 60 degrees."
