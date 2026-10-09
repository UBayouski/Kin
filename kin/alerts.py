"""Alerts to the family. Each alert is sent once per day, and that survives a restart because
the record of what was sent lives in memory's event log, not in the process."""

from datetime import datetime

from kin import config
from kin.family import Family
from kin.memory import Memory


def _person() -> str:
    return config.person_name() or "the person"


def report_absence(memory: Memory, family: Family | None) -> str | None:
    """Tell the family, once a day, that the person has not been seen or heard. Returns the alert if raised now."""
    today = datetime.now().date()
    if any(kind == "absence" for _, kind, _ in memory.events_on(today)):
        return None
    alert = f"I have not seen or heard {_person()} today, as of {datetime.now():%H:%M}."
    memory.log_event("absence", alert)
    memory.log_event("alert", alert)
    if family:
        family.send(f"Kin: {alert} I looked and called out. It may be worth a call.")
    return alert


def report_deaf(memory: Memory, family: Family | None) -> str | None:
    """Tell the family, once a day, that Kin's microphone is giving him nothing, so he cannot say whether
    the person is about. Raised instead of an absence alert: "I have not seen or heard her today" drawn
    from a broken ear is a false claim about someone's wellbeing (Oct 3, the input device had become a
    webcam that delivers silence)."""
    today = datetime.now().date()
    if any(kind == "deaf" for _, kind, _ in memory.events_on(today)):
        return None
    alert = (f"My microphone is not picking anything up, so I cannot tell whether {_person()} is about. "
             "This is a fault at my end and says nothing about her. The microphone may be unplugged, muted "
             "or switched for another device.")
    memory.log_event("deaf", alert)
    memory.log_event("alert", alert)
    if family:
        family.send(f"Kin: {alert}")
    return alert


def report_all_clear(memory: Memory, family: Family | None) -> str | None:
    """After an absence alert, tell the family as soon as the person turns up. Once per day."""
    today = datetime.now().date()
    events = memory.events_on(today)
    raised_at = next((ts for ts, kind, _ in events if kind == "absence"), None)
    last = memory.last_sign_of_person(today)
    if not raised_at or any(kind == "all_clear" for _, kind, _ in events) or not (last and last > raised_at):
        return None
    note = f"Update: {_person()} is up and about; I saw or heard them at {datetime.now():%H:%M}."
    memory.log_event("all_clear", note)
    if family:
        family.send(f"Kin: {note}")
    return note


def report_robot_down(memory: Memory, family: Family | None, minutes: float) -> str | None:
    """Tell the family, once per outage, that the robot itself has stopped responding. While he is down
    there are no reminders, no check-ins and no presence alerts, so silence would be the worst failure."""
    today = datetime.now().date()
    kinds = [kind for _, kind, _ in memory.events_on(today)]
    if kinds.count("robot_down") > kinds.count("robot_back"):
        return None  # this outage has been reported already
    alert = (f"I have not been able to reach the robot for {minutes:.0f} minutes. Reminders and check-ins are "
             "paused until he is back. He may need restarting: hold his back button for 5 seconds, then press it once.")
    memory.log_event("robot_down", alert)
    memory.log_event("alert", alert)
    if family:
        family.send(f"Kin: {alert}")
    return alert


def report_robot_back(memory: Memory, family: Family | None) -> str | None:
    today = datetime.now().date()
    kinds = [kind for _, kind, _ in memory.events_on(today)]
    if "robot_down" not in kinds or kinds.count("robot_back") >= kinds.count("robot_down"):
        return None
    note = "The robot is responding again. Reminders and check-ins have resumed."
    memory.log_event("robot_back", note)
    if family:
        family.send(f"Kin: {note}")
    return note


def report_body_down(memory: Memory, family: Family | None, minutes: float) -> str | None:
    """Tell the family, once per outage, that the robot's body has stopped answering while his head carries
    on. He still talks and listens through the hub, so it is easy to miss: the button is dead, he cannot
    move or feel a touch, and his battery can no longer be watched."""
    today = datetime.now().date()
    kinds = [kind for _, kind, _ in memory.events_on(today)]
    if kinds.count("body_down") > kinds.count("body_back"):
        return None  # this outage has been reported already
    alert = (f"The robot's body has not responded for {minutes:.0f} minutes: his back button, touch and wheels "
             "are dead, though his face is on and I can still talk. He needs restarting: hold his back button "
             "for about 5 seconds, let go as soon as his lights and screen go dark, then press it once to "
             "start him. Do not keep holding: a longer hold on the charger puts him in recovery mode.")
    memory.log_event("body_down", alert)
    memory.log_event("alert", alert)
    if family:
        family.send(f"Kin: {alert}")
    return alert


def report_body_back(memory: Memory, family: Family | None) -> str | None:
    today = datetime.now().date()
    kinds = [kind for _, kind, _ in memory.events_on(today)]
    if "body_down" not in kinds or kinds.count("body_back") >= kinds.count("body_down"):
        return None
    note = "The robot's body is responding again."
    memory.log_event("body_back", note)
    if family:
        family.send(f"Kin: {note}")
    return note


def report_stranded(memory: Memory, family: Family | None, minutes: float,
                    reason: str = "with a low battery") -> str | None:
    """Tell the family, once a day, that the robot is off his charger with a low battery (or one Kin cannot
    read) and nobody has put him back. Left alone he runs flat and goes dark, which has happened."""
    today = datetime.now().date()
    if any(kind == "stranded" for _, kind, _ in memory.events_on(today)):
        return None
    alert = (f"The robot has been off his charger {reason} for {minutes:.0f} minutes and needs "
             "putting back on it. If he runs flat, reminders and check-ins stop.")
    memory.log_event("stranded", alert)
    memory.log_event("alert", alert)
    if family:
        family.send(f"Kin: {alert}")
    return alert


def report_missed_reminders(memory: Memory, family: Family | None) -> list[str]:
    """Tell the family about reminders Kin has given up on today. Returns the alerts raised just now."""
    today = datetime.now().date()
    already = {detail for _, kind, detail in memory.events_on(today) if kind == "alert"}
    raised = []
    for reminder in memory.reminders_missed(max_attempts=config.reminder_attempts()):
        alert = f"No response to the reminder '{reminder.text}' after {config.reminder_attempts()} tries today."
        if alert in already:
            continue
        memory.log_event("alert", alert)
        if family:
            family.send(f"Kin: {alert} It may be worth a call.")
        raised.append(alert)
    return raised
