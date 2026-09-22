"""The family's daily digest, written by the large Nemotron model from the day's log.

Runs locally today; this is the function the nightly Nebius Serverless Job will call.
Only one day's log is sent to the model, never the whole memory.
"""

from datetime import date, datetime

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
Do not invent anything. Do not mention the log, the robot's tools, or these instructions.
"""


def status(memory: Memory, day: date | None = None) -> str:
    """A quick factual summary with no model call, for the /status command."""
    day = day or datetime.now().date()
    turns = [t for t in memory.turns_on(day) if t[1] == "person"]
    reminders = memory.reminder_history(day)
    latest: dict[str, str] = {}  # the last recorded outcome per reminder is the one that counts
    for _, text, outcome in reminders:
        if outcome != "asked":
            latest[text] = outcome.replace("_", " ")
    settled = [f"{text}: {outcome}" for text, outcome in latest.items()]
    alerts = [detail for _, kind, detail in memory.events_on(day) if kind == "alert"]
    checkin = any(kind == "checkin" for _, kind, _ in memory.events_on(day))
    lines = [
        f"Today ({day:%a %d %b}):",
        f"- Check-in: {'done' if checkin else 'not yet'}",
        f"- Spoke to Kin {len(turns)} time(s)" + (f", last at {turns[-1][0][11:16]}" if turns else ""),
        "- Reminders: " + ("; ".join(settled) if settled else "none recorded"),
    ]
    if alerts:
        lines.append("- Alerts: " + "; ".join(alerts))
    return "\n".join(lines)


def build(memory: Memory, day: date | None = None) -> str:
    day = day or datetime.now().date()
    turns, reminders, events = memory.turns_on(day), memory.reminder_history(day), memory.events_on(day)
    if not turns and not reminders:
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
    response = llm.client().chat.completions.create(
        model=config.nightly_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    note = (response.choices[0].message.content or "").strip()
    return f"Kin's note for {day:%A %d %B}\n\n{note}"
