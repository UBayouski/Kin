"""The family's daily digest, written by the large Nemotron model from the day's log.

Runs locally today; this is the function the nightly Nebius Serverless Job will call.
Only one day's log is sent to the model, never the whole memory.
"""

import re
from datetime import date, datetime, timedelta

from kin import config, llm
from kin.memory import Memory

PROMPT = """\
You write a short daily note to the family of {person}, an older adult who lives alone with Kin, a companion robot.
The note is about {day}. Write it from the log below and nothing else.

Conversation (speaker: text):
{conversation}

Reminders (time, what, outcome):
{reminders}

Other events:
{events}

Write plain text for a phone message, under 120 words, warm and matter-of-fact:
- One line on how they seemed, using only what they said themselves.
- Reminders: what was confirmed, declined or got no response.
- Anything they mentioned that family would want to know: plans, visitors, worries, aches.
- End with "Worth a call:" and one line, only if something genuinely needs attention. Otherwise leave it out.

Rules: never diagnose or interpret health; report their words ("said her knee aches"), not conclusions.
The conversation was written down by an imperfect microphone in a room. Garbled, repeated or nonsensical
lines are the microphone failing, never evidence about the person: leave them out, and never remark on how
clearly they spoke, on confusion, forgetfulness or repetition. If little was understood, say the day was
quiet. Judge nothing about their mind or mood beyond what they plainly said.
Do not invent anything. Do not mention the log, the robot's tools, or these instructions.
If the events show you could not reach your robot for an hour or more, say so plainly in one line and make
clear it says nothing about her: a silent day because you were offline is not a quiet day.
"""


def status(memory: Memory, day: date | None = None) -> str:
    """A quick factual summary with no model call, for the /status command: facts and times, no reading of them."""
    today = datetime.now().date()
    day = day or today
    person = config.person_name() or "she"
    turns = [t for t in memory.turns_on(day) if t[1] == "person"]
    events = memory.events_on(day)
    # Every settled outcome with its time, runs of the same outcome collapsed to the last one: "confirmed 12:37,
    # no response 13:10" tells the family more than the final word alone did.
    outcomes: dict[str, list[tuple[str, str]]] = {}
    for ts, text, outcome in sorted(memory.reminder_history(day)):  # by time, not by the order rows were written
        if outcome == "asked" or not ts.startswith(day.isoformat()):
            continue
        run = outcomes.setdefault(text, [])
        if run and run[-1][0] == outcome:
            run[-1] = (outcome, ts[11:16])
        else:
            run.append((outcome, ts[11:16]))
    settled = [f"{text}: " + ", ".join(f"{o.replace('_', ' ')} {at}" for o, at in run) for text, run in outcomes.items()]
    checkins = [(ts, detail) for ts, kind, detail in events if kind == "checkin"]
    answered = [ts for ts, detail in checkins if not re.search(r"no (response|reply)", detail, re.I)]
    if answered:
        checkin = f"done at {answered[-1][11:16]}"
    elif checkins:
        checkin = "tried at " + ", ".join(ts[11:16] for ts, _ in checkins) + ", no reply"
    else:
        checkin = "not yet" if day == today else "none"
    family_in = sum(kind == "family_in" for _, kind, _ in events)
    family_out = sum(kind == "family_out" for _, kind, _ in events)
    alerts = [detail for _, kind, detail in events if kind == "alert" and "reach the robot" not in detail]
    blind = blind_for(memory, day)
    last = memory.last_sign_of_person(day)
    label = "Today" if day == today else "Yesterday" if day == today - timedelta(days=1) else f"{day:%A}"
    lines = [
        f"{label} ({day:%a %d %b}):",
        f"- Last sign of {person}: {last[11:16]}" if last else f"- No sign of {person} recorded",
        f"- Check-in: {checkin}",
        f"- Spoke with Kin {len(turns)} time{'' if len(turns) == 1 else 's'}"
        + (f", last at {turns[-1][0][11:16]}" if turns else ""),
        "- Reminders: " + ("; ".join(settled) if settled else "none recorded"),
    ]
    if family_in or family_out:
        lines.append(f"- Family: {family_in} message{'' if family_in == 1 else 's'} delivered, {family_out} sent back")
    if alerts:
        lines.append("- Alerts: " + "; ".join(alerts))
    if blind:
        lines.append(f"- Robot: out of reach for {blind // 60} h {blind % 60:02d} min at the longest; "
                     f"that says nothing about {person}")
    return "\n".join(lines)


def blind_for(memory: Memory, day: date) -> int:
    """Minutes that day Kin reported being unable to reach the robot. Read from his own alerts, so it
    survives a restart. Only the longest stretch is taken; the alerts repeat as an outage goes on."""
    minutes = [int(m.group(1)) for _, kind, detail in memory.events_on(day) if kind == "robot_down"
               for m in [re.search(r"for (\d+) minutes", detail)] if m]
    return max(minutes, default=0)


def build(memory: Memory, day: date | None = None) -> str:
    day = day or datetime.now().date()
    turns, reminders, events = memory.turns_on(day), memory.reminder_history(day), memory.events_on(day)
    if not turns and not reminders:
        # "A quiet day" would be a false reassurance when the truth is that Kin could not see her at all:
        # on Sep 30 the robot was unreachable for 20 hours and the family got no note whatsoever.
        blind = blind_for(memory, day)
        if blind >= 60:
            hours = blind // 60
            return (f"Kin's note for {day:%A %d %B}\n\nI could not reach my robot for about "
                    f"{hours} hour{'' if hours == 1 else 's'} today, so I have nothing to tell you about how "
                    f"{config.person_name() or 'she'} has been. This is a fault at my end, not a sign "
                    "about her. It may be worth a call, and the robot may need restarting.")
        return f"Kin's note for {day:%A %d %B}: a quiet day. No conversations or reminders were recorded."

    prompt = PROMPT.format(
        person=config.person_name() or "a person",
        day=f"{day:%A %d %B %Y}",
        conversation="\n".join(f"[{ts[11:16]}] {who}: {text}" for ts, who, text in turns) or "(none)",
        reminders="\n".join(f"[{ts[11:16]}] {text}: {outcome}" for ts, text, outcome in reminders
                            if outcome != "asked") or "(none)",
        events="\n".join(f"[{ts[11:16]}] {kind}: {detail}" for ts, kind, detail in events
                         if kind in ("alert", "checkin")) or "(none)",
    )
    return f"Kin's note for {day:%A %d %B}\n\n{_write(prompt)}"


def _write(prompt: str, max_words: int = 200) -> str:
    """The note itself. The nightly model reasons before answering, and on Sep 28 it spent the whole
    budget thinking: the 'note' was 7600 characters of raw reasoning, which Telegram then refused. So the
    reasoning is stripped, a reply cut off by the limit is retried without thinking, and the result is
    checked for length before anyone tries to send it."""
    for thinking in (True, False):
        extra = {} if thinking else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        choice = llm.client().chat.completions.create(
            model=config.nightly_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
            **extra,
        ).choices[0]
        note = (choice.message.content or "").split("</think>")[-1].strip()
        if choice.finish_reason != "length" and note and len(note.split()) <= max_words:
            return note
    raise ValueError("the model did not produce a short note (it ran out of tokens, or wrote far too much)")
