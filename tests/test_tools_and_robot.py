import json
from concurrent.futures import Future

import pytest

from kin import robot as kin_robot
from kin import tools
from kin.agent import Agent
from kin.tools import RobotTools


def names(specs) -> list[str]:
    return [spec["function"]["name"] for spec in specs]


def test_every_tool_offered_to_the_model_has_a_handler(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")

    for name in names(tools.tool_specs()):
        assert hasattr(RobotTools, f"_tool_{name}"), name


def test_optional_tools_are_only_offered_when_configured(monkeypatch):
    assert not {"web_search", "notify_family"} & set(names(tools.tool_specs()))
    assert "Scam check" not in Agent._system_prompt()

    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")

    assert {"web_search", "notify_family"} <= set(names(tools.tool_specs()))
    assert "Scam check" in Agent._system_prompt()
    assert "{" not in Agent._system_prompt().replace("{{", "").replace("}}", ""), "an unfilled placeholder"


def test_dispatch_never_raises_and_reports_failures_as_text(memory):
    kit = RobotTools(None, memory)

    assert kit.dispatch("teleport", "{}").startswith("error: unknown tool")
    assert kit.dispatch("remember", "not json").startswith("error:")
    assert kit.dispatch("set_reminder", json.dumps({"text": "pills", "time_of_day": "25:00"})).startswith("error:")
    assert kit.dispatch("notify_family", json.dumps({"message": "hi"})) == "error: the family link is not set up"


def test_memory_and_reminder_tools_round_trip(memory):
    kit = RobotTools(None, memory)

    assert kit.dispatch("remember", json.dumps({"fact": "Mila visits on 2026-09-20.", "date": "2026-09-20"})) == "saved as #1"
    assert "Mila visits" in kit.dispatch("recall", json.dumps({"query": "Mila"}))
    assert kit.dispatch("update_fact", json.dumps({"fact_id": 1, "status": "done"})) == "updated"
    assert kit.dispatch("recall", "{}") == "nothing saved yet"

    assert "reminder #1 set for 08:00" in kit.dispatch("set_reminder", json.dumps({"text": "pills", "time_of_day": "08:00"}))
    assert "08:00 (daily): pills" in kit.dispatch("list_reminders", "{}")
    assert kit.dispatch("cancel_reminder", json.dumps({"reminder_id": 1})) == "cancelled"
    assert kit.dispatch("list_reminders", "{}") == "none set"


def test_the_last_unanswered_try_tells_the_family_at_once(memory, family):
    kit = RobotTools(None, memory, family_link=family)
    reminder_id = memory.add_reminder("evening pills", "00:01")
    for _ in range(2):
        memory.log_reminder(reminder_id, "asked")
        assert kit.dispatch("reminder_outcome", json.dumps({"reminder_id": reminder_id, "outcome": "no_response"})) == "logged"

    memory.log_reminder(reminder_id, "asked")
    result = kit.dispatch("reminder_outcome", json.dumps({"reminder_id": reminder_id, "outcome": "no_response"}))

    assert "the family has been told" in result
    assert len(family.sent) == 1


def sdk_method(future: Future):
    """Stands in for an SDK call made with _return_future=True."""
    def method(*args, _return_future=False, **kwargs):
        assert _return_future
        return future
    return method


def test_call_returns_the_result():
    future = Future()
    future.set_result("done")

    assert kin_robot.call(sdk_method(future)) == "done"


def test_call_cancels_the_action_on_the_robot_when_it_times_out():
    future = Future()

    with pytest.raises(TimeoutError):
        kin_robot.call(sdk_method(future), timeout=0.3)
    assert future.cancelled(), "otherwise Vector would carry on after we stopped waiting"


def test_a_pet_cuts_a_slow_action_short():
    future = Future()

    with pytest.raises(kin_robot.Interrupted):
        kin_robot.call(sdk_method(future), timeout=30, interrupt=lambda: True)
    assert future.cancelled()


def test_an_interrupted_tool_tells_the_model_to_attend_to_the_person(memory, monkeypatch):
    kit = RobotTools(None, memory)

    def interrupted():
        raise kin_robot.Interrupted()

    monkeypatch.setattr(kit, "_tool_go_home", interrupted)

    assert kit.dispatch("go_home", "{}").startswith("interrupted: the person just petted you")


def docked_robot(on_charger=True):
    from types import SimpleNamespace
    return SimpleNamespace(status=SimpleNamespace(is_on_charger=on_charger),
                           behavior=SimpleNamespace(drive_straight=None, turn_in_place=None))


def test_kin_does_not_drive_vector_off_his_charger_by_default(memory, monkeypatch):
    kit = RobotTools(docked_robot(), memory)
    monkeypatch.setattr(kin_robot, "quiet_look", lambda robot: False)

    assert "staying on the charger" in kit.dispatch("drive", json.dumps({"distance_mm": 100}))
    assert "staying on the charger" in kit.dispatch("turn", json.dumps({"degrees": 90}))
    assert "call out with `ask`" in kit.dispatch("look_for_face", "{}")
    assert kit.dispatch("go_home", "{}") == "already on the charger"


def test_from_the_charger_he_still_looks_with_his_head(memory, monkeypatch):
    kit = RobotTools(docked_robot(), memory)
    monkeypatch.setattr(kin_robot, "quiet_look", lambda robot: True)

    assert kit.dispatch("look_for_face", "{}") == "found a person"


def test_staying_on_the_charger_can_be_switched_off_for_demos(memory, monkeypatch):
    monkeypatch.setenv("KIN_STAY_ON_CHARGER", "off")
    kit = RobotTools(docked_robot(), memory)
    left = []
    monkeypatch.setattr(kit, "_leave_charger", lambda: left.append(True))
    monkeypatch.setattr("kin.tools.call", lambda *a, **k: None)

    assert kit.dispatch("drive", json.dumps({"distance_mm": 100})) == "drove 100 mm"
    assert left == [True]

def test_unit_symbols_are_spelled_out_before_anything_is_spoken():
    """Live, "It's currently 62°F" came out of the voice as "62 E F"."""
    from kin.tools import for_speech

    assert for_speech("It's currently 62°F and clear.") == "It's currently 62 degrees Fahrenheit and clear."
    assert for_speech("About 70\u202f°F today, 21 °C.") == "About 70 degrees Fahrenheit today, 21 degrees Celsius."
    assert for_speech("A high of 68° with a 10% chance of rain, wind 6 mph.") == \
        "A high of 68 degrees with a 10 percent chance of rain, wind 6 miles per hour."
    assert for_speech("Take a light jacket.") == "Take a light jacket."


def test_a_silent_body_is_told_from_a_healthy_robot_by_todays_real_numbers():
    """Sep 21: button, touch, motors and battery gauge dead for hours while his face and the SDK carried on."""
    from kin.robot import body_silent

    assert body_silent(level=3, volts=3.60, touch_values=[5491] * 20), "the state he was found in"
    assert not body_silent(level=3, volts=4.10, touch_values=[5491] * 20), "really full: a calm sensor is not a fault"
    assert not body_silent(level=3, volts=3.61, touch_values=[5490, 5493, 5491] * 6), "odd reading after a boot, body alive"
    assert not body_silent(level=1, volts=3.60, touch_values=[5491] * 20), "3.6 V and 'low' is simply a low battery"


def test_the_family_hears_of_a_silent_body_once_per_outage_and_only_if_it_lasts(memory, family, monkeypatch):
    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family, kin._body_lost_since = memory, family, None
    clock = {"now": 1000.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: clock["now"])

    assert kin._watch_body(True) is True
    clock["now"] += 4 * 60
    kin._watch_body(True)
    assert family.sent == [], "two losses of a few minutes each recovered by themselves that morning"

    clock["now"] += 2 * 60
    kin._watch_body(True)
    clock["now"] += 30
    kin._watch_body(True)
    assert len(family.sent) == 1 and "body has not responded" in family.sent[0]

    assert kin._watch_body(False) is False
    assert "responding again" in family.sent[-1] and len(family.sent) == 2

    kin._watch_body(True)
    clock["now"] += 6 * 60
    kin._watch_body(True)
    assert len(family.sent) == 3, "a second outage the same day is reported too"


def test_the_tail_of_his_own_question_is_not_taken_for_an_answer():
    """Live: "Could you please place me on my charger?" was answered by the room's echo, "charger.", and
    Kin thanked a person who was not there."""
    from kin.tools import is_echo

    assert is_echo("Could you please place me on my charger?", "charger.")
    assert is_echo("How did you sleep?", "You sleep")
    assert not is_echo("Could you please place me on my charger?", "No.")
    assert not is_echo("Anything else?", "Yes")
    assert not is_echo("How did you sleep?", "I slept well, thank you.")
    assert not is_echo("How did you sleep?", "")


def test_between_tasks_kin_keeps_a_light_hold_so_vector_does_not_wander(monkeypatch):
    """He drove off his charger twice on Sep 21 and ran flat; with RESERVE_CONTROL held he stayed (Sep 22)."""
    from types import SimpleNamespace

    from anki_vector.connection import ControlPriorityLevel

    from kin import robot as kin_robot

    levels = []

    class Conn:
        behavior_control_level = None

        @property
        def requires_behavior_control(self):
            return self.behavior_control_level is not None

        def request_control(self, behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY, timeout=0):
            self.behavior_control_level = behavior_control_level
            levels.append(behavior_control_level)

        def release_control(self, timeout=0):
            self.behavior_control_level = None
            levels.append("released")

    robot = SimpleNamespace(conn=Conn())
    monkeypatch.setattr(kin_robot.time, "sleep", lambda s: None)

    assert kin_robot.hold_idle(robot) is True and kin_robot.idle_held(robot)
    with kin_robot.control(robot):
        levels.append("task")
    assert levels == [ControlPriorityLevel.RESERVE_CONTROL, "released", ControlPriorityLevel.DEFAULT_PRIORITY, "task",
                      "released", ControlPriorityLevel.RESERVE_CONTROL], \
        "the hold is dropped before task control is asked for: a request on top of it gets no grant (live)"

    monkeypatch.setenv("KIN_IDLE_HOLD", "off")
    robot.conn.behavior_control_level = None
    levels.clear()
    with kin_robot.control(robot):
        pass
    assert levels == [ControlPriorityLevel.DEFAULT_PRIORITY, "released"], "off: he is left to himself"


def test_a_looped_animation_is_closed_with_its_get_out():
    """Cancelling the loop's future stops nothing on the robot; under the idle hold he sat "listening" for
    minutes after a conversation (Sep 22). Each loop ends with its own get-out animation."""
    from types import SimpleNamespace

    from kin import robot as kin_robot

    played = []
    anim = SimpleNamespace(play_animation_trigger=lambda trigger, **k: played.append(trigger) or SimpleNamespace(cancel=lambda: played.append("cancel")))
    robot = SimpleNamespace(anim=anim)

    with kin_robot.animating(robot, "VC_ListeningLoop"):
        pass
    assert played == ["VC_ListeningLoop", "cancel", "VC_ListeningGetOut"]

    played.clear()
    with kin_robot.animating(robot, "KnowledgeGraphAnswer"):
        pass
    assert played == ["KnowledgeGraphAnswer", "cancel", "KnowledgeGraphGetOut"]


def test_a_pinned_placeholder_voltage_or_a_disagreeing_charger_state_is_not_a_reading():
    """Sep 22 evening: 3.60 V "ok", unchanging, "off the charger" while the robot's status said on it:
    the rescue ran and the family got a stranded alert for a robot sitting on his dock."""
    from types import SimpleNamespace

    from kin.robot import gauge_read

    report = lambda volts, on=False, level=2: SimpleNamespace(battery_volts=volts, is_on_charger_platform=on, battery_level=level)
    assert not gauge_read(report(3.60), previous_volts=3.60, status_on_charger=False), "pinned placeholder"
    assert not gauge_read(report(0.00), previous_volts=4.0, status_on_charger=False), "not read yet after a boot"
    assert not gauge_read(report(3.85), previous_volts=3.84, status_on_charger=True), "the two sources disagree"
    assert gauge_read(report(3.60), previous_volts=3.63, status_on_charger=False), "a real battery passing 3.60"
    assert gauge_read(report(3.52, on=True), previous_volts=3.55, status_on_charger=True)
    assert gauge_read(report(3.52), previous_volts=3.55, status_on_charger=None), "no status to compare with"
    assert not gauge_read(report(3.635, on=True, level=3), previous_volts=3.635, status_on_charger=True), "Sep 24: pinned 'full' below 3.7 V"
    assert gauge_read(report(3.635, on=True, level=3), previous_volts=3.631, status_on_charger=True), "moving: a real gauge, oddly labelled"


def test_a_closing_offer_listens_for_less_time_than_a_real_question(memory):
    """Every conversation ends with an offer of more, and silence is its usual answer; 12 s of listening
    pose after each one was too long."""
    from types import SimpleNamespace

    from kin.tools import CLOSING_LISTEN_SECONDS, LISTEN_SECONDS, RobotTools, is_closing_offer

    assert is_closing_offer("It's 57 and clear. Do you need anything else?")
    assert is_closing_offer("Lovely, well done. Anything else I can do for you?")
    assert not is_closing_offer("How did you sleep?")

    waits = []
    ears = SimpleNamespace(listen=lambda timeout=12.0: waits.append(timeout) or None)
    kit = RobotTools(SimpleNamespace(anim=SimpleNamespace(play_animation_trigger=lambda *a, **k: None)),
                     memory, mouth=SimpleNamespace(say=lambda text: None), ears=ears)
    kit.dispatch("ask", '{"question": "How did you sleep?"}')
    kit.dispatch("ask", '{"question": "Is there anything else you need?"}')
    assert waits == [LISTEN_SECONDS, CLOSING_LISTEN_SECONDS]


def test_a_reminder_outcome_lands_on_the_reminder_being_handled_whatever_id_the_planner_wrote(memory):
    """Sep 23, 18:00: the planner logged 'no_response' on a made-up #5 for reminder #1, so #1 stayed
    unsettled and was asked again ten minutes later."""
    import pytest

    kit = RobotTools(None, memory)
    reminder_id = memory.add_reminder("drink water", "18:00")

    with pytest.raises(ValueError, match="no reminder #5"):
        memory.log_reminder(5, "no_response")

    kit.current_reminder = reminder_id
    assert kit.dispatch("reminder_outcome", json.dumps({"reminder_id": 5, "outcome": "confirmed"})) == "logged"
    assert [row[0] for row in memory.db.execute("SELECT reminder_id FROM reminder_log")] == [reminder_id]

    kit.current_reminder = None
    assert kit.dispatch("reminder_outcome", json.dumps({"reminder_id": 5, "outcome": "confirmed"})).startswith("error:")


def watcher():
    """A TouchWatcher with no robot and no thread; drive it with `step`."""
    from collections import deque

    from kin.robot import TouchWatcher

    w = TouchWatcher.__new__(TouchWatcher)
    w._window, w._released, w._quiet_until, w._resting_since = deque(maxlen=TouchWatcher.WINDOW_SAMPLES), True, 0.0, None
    return w


def feed(w, touched, count, start=0.0, step=0.05):
    """`count` samples of the same reading; returns the times at which a pet fired."""
    fired = []
    for i in range(count):
        if w.step(touched, start + i * step):
            fired.append(start + i * step)
    return fired


def test_a_stroke_is_one_pet_and_the_hand_must_come_off_before_the_next():
    w = watcher()

    assert len(feed(w, True, 20)) == 1, "one stroke, one pet"
    assert feed(w, False, 20, start=1.0) == [], "letting go is not a pet"
    assert len(feed(w, True, 20, start=30.0)) == 1, "a second stroke after the cooldown"


def test_something_resting_against_his_back_is_not_a_pet_over_and_over():
    """Sep 28, in his new enclosure: a pet every 20 s for an hour, which filled memory with presence,
    made the house look permanently occupied and interrupted what he was doing."""
    w = watcher()

    fired = feed(w, True, 20 * 60 * 5)  # five minutes of unbroken contact, at 20 samples a second

    assert len(fired) == 1, "reported once, not every cooldown"
    assert w.resting_on_something(seconds=120), "and Kin can tell it is furniture, not a person"

    feed(w, False, 20, start=999.0)
    assert not w.resting_on_something(seconds=120), "released again"
