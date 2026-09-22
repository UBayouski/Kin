from kin import alerts
from kin.memory import Memory


def give_up_on(memory, text="evening pills") -> int:
    reminder_id = memory.add_reminder(text, "00:01")
    for _ in range(3):
        memory.log_reminder(reminder_id, "asked")
        memory.log_reminder(reminder_id, "no_response")
    return reminder_id


def test_a_missed_reminder_alerts_the_family_exactly_once(memory, family):
    reminder_id = memory.add_reminder("evening pills", "00:01")
    for _ in range(2):
        memory.log_reminder(reminder_id, "asked")
        memory.log_reminder(reminder_id, "no_response")
        assert alerts.report_missed_reminders(memory, family) == [], "not before the last try"

    memory.log_reminder(reminder_id, "asked")
    memory.log_reminder(reminder_id, "no_response")

    assert len(alerts.report_missed_reminders(memory, family)) == 1
    assert alerts.report_missed_reminders(memory, family) == [], "a restart must not send it again"
    assert family.sent == ["Kin: No response to the reminder 'evening pills' after 3 tries today. It may be worth a call."]


def test_alerts_are_recorded_even_with_no_family_link(memory):
    give_up_on(memory)

    assert len(alerts.report_missed_reminders(memory, None)) == 1


def test_absence_is_reported_once_and_uses_the_persons_name(memory, family, monkeypatch):
    monkeypatch.setenv("KIN_PERSON_NAME", "Jennie")

    assert "I have not seen or heard Jennie today" in alerts.report_absence(memory, family)
    assert alerts.report_absence(memory, family) is None
    assert len(family.sent) == 1


def test_all_clear_needs_a_sign_of_the_person_after_the_absence_alert(memory, family):
    assert alerts.report_all_clear(memory, family) is None, "no absence, nothing to clear"

    memory.log_turn("person", "chatter from before the alert")
    alerts.report_absence(memory, family)
    assert alerts.report_all_clear(memory, family) is None, "old evidence does not count"

    memory.log_event("presence", "petted")
    assert "up and about" in alerts.report_all_clear(memory, family)
    assert alerts.report_all_clear(memory, family) is None, "once per day"


def test_inbox_survives_a_restart_until_the_message_is_handled(memory, tmp_path):
    memory.inbox_add(101, 5, "Vlad", "Dinner is at seven")

    assert memory.inbox_next()[:4] == (101, 5, "Vlad", "Dinner is at seven")
    restarted = Memory(tmp_path / "kin.db")  # the process died before marking it handled
    assert restarted.inbox_next()[0] == 101

    restarted.inbox_done(101)
    assert restarted.inbox_next() is None


def test_inbox_ignores_redelivery_and_keeps_order(memory):
    memory.inbox_add(102, 5, "Vlad", "second")
    memory.inbox_add(101, 5, "Vlad", "first")
    memory.inbox_add(101, 5, "Vlad", "first, delivered again by Telegram")

    assert memory.inbox_next()[3] == "first"
    memory.inbox_done(101)
    assert memory.inbox_next()[3] == "second"


def test_a_message_that_keeps_failing_cannot_wedge_the_service(memory):
    memory.inbox_add(103, 5, "Vlad", "poison")
    for _ in range(3):
        assert memory.inbox_next() is not None  # each try raised before inbox_done

    assert memory.inbox_next() is None


def test_a_robot_that_stays_down_is_reported_once_and_so_is_his_return(memory, family):
    assert alerts.report_robot_back(memory, family) is None, "he was never down"

    assert "not been able to reach the robot for 12 minutes" in alerts.report_robot_down(memory, family, 12)
    assert alerts.report_robot_down(memory, family, 30) is None, "once per outage"

    assert "responding again" in alerts.report_robot_back(memory, family)
    assert alerts.report_robot_back(memory, family) is None
    assert len(family.sent) == 2


def test_a_second_outage_on_the_same_day_is_reported_too(memory, family):
    alerts.report_robot_down(memory, family, 10)
    alerts.report_robot_back(memory, family)

    assert alerts.report_robot_down(memory, family, 11) is not None, "he ran flat at 09:42 and showed 899 at 11:00"
    assert alerts.report_robot_down(memory, family, 25) is None
    assert alerts.report_robot_back(memory, family) is not None
    assert len(family.sent) == 4


def test_a_robot_stranded_with_a_low_battery_is_reported_once(memory, family):
    assert "off his charger with a low battery for 10 minutes" in alerts.report_stranded(memory, family, 10)
    assert alerts.report_stranded(memory, family, 20) is None