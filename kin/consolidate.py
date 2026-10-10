"""Nightly memory consolidation: a large Nemotron model tidies what Kin knows.

Merges duplicates, closes events that have passed (leaving a follow-up to ask about), retires facts that
were contradicted, and lifts lasting facts out of the day's conversation that were missed live.
Runs locally today; the same function is what the nightly Nebius Serverless Job will call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from kin import config, llm

if TYPE_CHECKING:  # only a type here: the night's container runs these modules without the database module
    from kin.memory import Memory

PROMPT = """\
You maintain the long-term memory of Kin, a companion robot for an older adult who lives alone.
Today is {today}.

Saved facts (active):
{facts}

Conversation since the last tidy-up:
{conversation}

Tidy the memory. Reply with JSON only, in this shape:
{{"operations": [
  {{"op": "update", "id": 3, "text": "...", "date": "YYYY-MM-DD or null"}},
  {{"op": "done", "id": 4, "follow_up": "Ask how ... went."}},
  {{"op": "retire", "id": 5, "reason": "..."}},
  {{"op": "add", "text": "...", "date": "YYYY-MM-DD or null"}}
]}}

Rules:
- Merge duplicates: `update` the best one to hold everything, `retire` the others.
- If the conversation contradicts or changes a fact, `update` it.
- If a fact's date is before today, mark it `done`. Add a `follow_up` only if it is something a caring \
friend would ask about afterwards (a doctor visit, a family visit), written as an instruction to Kin.
- `add` lasting facts from the conversation that are not saved yet: people, plans, preferences, worries, \
how the person has been feeling. Skip small talk and anything about the robot itself.
- Write facts as one self-contained third-person sentence with calendar dates, never 'tomorrow' or 'Tuesday'.
- Never invent. No medical interpretation or advice. If nothing needs changing, return {{"operations": []}}.
"""


@dataclass
class Report:
    updated: int = 0
    done: int = 0
    retired: int = 0
    added: int = 0
    skipped: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        text = f"updated {self.updated}, closed {self.done}, retired {self.retired}, added {self.added}"
        return text + (f", skipped {len(self.skipped)}" if self.skipped else "")


def _parse(reply: str) -> list[dict]:
    match = re.search(r"\{.*\}", reply, re.DOTALL)  # tolerate code fences or reasoning around the JSON
    if not match:
        raise ValueError(f"no JSON in model reply: {reply[:200]!r}")
    operations = json.loads(match.group(0)).get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("'operations' is not a list")
    return operations


def _date(value) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else None


def render_turns(turns) -> str:
    return "\n".join(f"[{ts[:16].replace('T', ' ')}] {who}: {text}" for ts, who, text in turns)


def propose(today: str, facts_text: str, conversation_text: str) -> list[dict]:
    """The thinking half: needs only text and Nebius credentials, so it can run anywhere (the nightly job)."""
    prompt = PROMPT.format(today=today, facts=facts_text or "(none)", conversation=conversation_text or "(none)")
    # A reasoning model thinks first, and with a few days of conversation to read it thought past 8000
    # tokens (Sep 28). Second try: thinking off. The operations are a structured job, not a judgement call.
    for thinking in (True, False):
        extra = {} if thinking else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        response = llm.client().chat.completions.create(
            model=config.nightly_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.2,
            **extra,
        )
        choice = response.choices[0]
        if choice.finish_reason != "length":
            return _parse(choice.message.content or "")
    raise ValueError("model ran out of tokens before finishing its answer, even without thinking")


def consolidate(memory: Memory, dry_run: bool = False) -> Report:
    facts = memory.active_facts(limit=200)
    turns = memory.turns_since(memory.last_event_time("consolidation"))
    if not facts and not turns:
        return Report()
    operations = propose(
        f"{datetime.now():%A, %Y-%m-%d}", "\n".join(f.render() for f in facts), render_turns(turns)
    )
    return apply(memory, operations, dry_run)


def apply(memory: Memory, operations: list[dict], dry_run: bool = False) -> Report:
    """The doing half, on the hub: validate each proposed operation against memory, then apply it."""
    report = Report(operations=operations)
    known_ids = {f.id for f in memory.active_facts(limit=500)}

    for op in report.operations:
        kind, fact_id = op.get("op"), op.get("id")
        if kind in ("update", "done", "retire") and fact_id not in known_ids:
            report.skipped.append(f"{kind}: no active fact #{fact_id}")
            continue
        if kind == "update" and op.get("text"):
            report.updated += 1
            if not dry_run:
                memory.update_fact(fact_id, text=op["text"], due=_date(op.get("date")))
        elif kind == "done":
            report.done += 1
            if not dry_run:
                memory.update_fact(fact_id, status="done")
                if op.get("follow_up"):
                    memory.remember(op["follow_up"])
        elif kind == "retire":
            report.retired += 1
            if not dry_run:
                memory.update_fact(fact_id, status="retired")
        elif kind == "add" and op.get("text"):
            report.added += 1
            if not dry_run:
                memory.remember(op["text"], due=_date(op.get("date")))
        else:
            report.skipped.append(f"unrecognized operation: {op}")

    if not dry_run:
        memory.log_event("consolidation", report.summary())
    return report
