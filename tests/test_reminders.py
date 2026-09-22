from datetime import datetime, timedelta

import pytest


def at(clock, hour: int, minute: int = 0) -> datetime:
    return clock["now"].replace(hour=hour, minute=minute, second=0)


def test_a_reminder_is_due_once_its_time_has_passed(memory, clock):
    memory.add_reminder("morning pills", "08:00")

    assert memory.reminders_due(at(clock, 7, 59)) == []
    assert [r.text for r in memory.reminders_due(at(clock, 8, 0))] == ["morning pills"]


def test_weekday_reminders_only_fire_on_their_days(memory):
    memory.add_reminder("bins out", "07:00", days="mon, thu")
    monday = datetime(2026, 9, 21, 9, 0)

    assert [r.text for r in memory.reminders_due(monday)] == ["bins out"]
    assert memory.reminders_due(monday + timedelta(days=1)) == []


@pytest.mark.parametrize("outcome", ["confirmed", "declined"])
def test_a_settled_reminder_is_left_alone_for_the_day(memory, clock, outcome):
    reminder_id = memory.add_reminder("morning pills", "00:01")
    memory.log_reminder(reminder_id, "asked")
    memory.log_reminder(reminder_id, outcome)

    assert memory.reminders_due(clock["now"] + timedelta(hours=1)) == []
    assert memory.reminders_missed(clock["now"]) == []


def test_an_unanswered_reminder_waits_then_is_tried_again_then_given_up(memory, clock):
    reminder_id = memory.add_reminder("evening pills", "00:01")

    for attempt in range(3):
        assert [r.text for r in memory.reminders_due(clock["now"] + timedelta(minutes=11))] == ["evening pills"]
        memory.log_reminder(reminder_id, "asked")
        memory.log_reminder(reminder_id, "no_response")
        assert memory.reminders_due(clock["now"] + timedelta(minutes=1)) == [], "asked too recently"
        clock["now"] += timedelta(minutes=11)

    assert memory.reminders_due(clock["now"]) == [], "three tries is the limit"
    assert [r.text for r in memory.reminders_missed(clock["now"])] == ["evening pills"]


def test_a_task_that_died_before_recording_an_outcome_still_uses_up_a_try(memory, clock):
    reminder_id = memory.add_reminder("evening pills", "00:01")
    for _ in range(3):
        memory.log_reminder(reminder_id, "asked")  # no outcome ever recorded
        clock["now"] += timedelta(minutes=11)

    assert memory.reminders_due(clock["now"]) == []


def test_a_late_confirmation_clears_a_missed_reminder(memory, clock):
    reminder_id = memory.add_reminder("evening pills", "00:01")
    for _ in range(3):
        memory.log_reminder(reminder_id, "asked")
        memory.log_reminder(reminder_id, "no_response")
    assert memory.reminders_missed(clock["now"])

    memory.log_reminder(reminder_id, "confirmed")

    assert memory.reminders_missed(clock["now"]) == []


def test_cancelled_reminders_never_fire(memory, clock):
    reminder_id = memory.add_reminder("old habit", "00:01")
    assert memory.deactivate_reminder(reminder_id)

    assert memory.reminders() == []
    assert memory.reminders_due(clock["now"]) == []
    assert memory.deactivate_reminder(999) is False


@pytest.mark.parametrize("time_of_day, days", [("25:99", "daily"), ("8am", "daily"), ("08:00", "someday")])
def test_bad_times_and_days_are_rejected(memory, time_of_day, days):
    with pytest.raises(ValueError):
        memory.add_reminder("nonsense", time_of_day, days)
