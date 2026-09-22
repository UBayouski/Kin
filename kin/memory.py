"""Local memory: one SQLite file on the hub. The file itself never leaves the home."""

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from kin import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'done' (it happened) or 'retired' (wrong / replaced)
    due TEXT,                               -- YYYY-MM-DD when the fact is about a dated event
    embedding BLOB                          -- float32 vector; NULL if embedding was unavailable
);
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    speaker TEXT NOT NULL,  -- 'person', 'kin' or 'family'
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,     -- e.g. 'task', 'presence', 'alert', 'consolidation'
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    update_id INTEGER PRIMARY KEY,  -- Telegram's id; makes re-delivery harmless
    ts TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    sender TEXT NOT NULL,
    text TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    handled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS heard (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    text TEXT NOT NULL,             -- what Vector's own microphone heard, as transcribed by wire-pod
    ack TEXT NOT NULL DEFAULT '',   -- what Vector said back in his own voice ("On it!")
    attempts INTEGER NOT NULL DEFAULT 0,
    handled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,         -- what to nudge about, e.g. 'morning pills'
    time_of_day TEXT NOT NULL,  -- HH:MM, local time
    days TEXT NOT NULL DEFAULT 'daily',  -- 'daily' or comma-separated 'mon,wed,fri'
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS reminder_log (
    id INTEGER PRIMARY KEY,
    reminder_id INTEGER NOT NULL REFERENCES reminders(id),
    ts TEXT NOT NULL,
    outcome TEXT NOT NULL       -- 'asked', 'confirmed', 'declined' or 'no_response'
);
"""

FACT_STATUSES = ("active", "done", "retired")
REMINDER_OUTCOMES = ("asked", "confirmed", "declined", "no_response")
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "did", "do", "does", "my", "me", "you", "your", "about",
              "what", "when", "how", "who", "and", "or", "of", "to", "in", "on", "for", "with", "know", "any"}
# Calibrated on Qwen3-Embedding-8B, whose scores run low: the right fact scores 0.32-0.39, unrelated
# queries peak at 0.15-0.28. A fact counts as related above the floor and within the margin of the best hit.
_MIN_SIMILARITY = 0.30
_SIMILARITY_MARGIN = 0.08

Embedder = Callable[[str, bool], Sequence[float]]


@dataclass
class Fact:
    id: int
    day: str  # YYYY-MM-DD it was saved
    text: str
    status: str
    due: str | None

    def render(self) -> str:
        due = f" (date: {self.due})" if self.due else ""
        return f"#{self.id} [{self.day}] {self.text}{due}"


@dataclass
class Reminder:
    id: int
    text: str
    time_of_day: str
    days: str


def _attempts(outcomes: list[str]) -> int:
    """Tries made today. Counts 'asked' too, so a task that died before recording an outcome still uses one up."""
    return max(outcomes.count("no_response"), outcomes.count("asked"))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_embedder() -> Embedder:
    from kin import llm  # imported late so memory can be used without Nebius credentials

    return llm.embed


class Memory:
    def __init__(self, path: str | Path | None = None, embed: Embedder | None = None):
        self.path = Path(path or config.db_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The agent calls the planner from a worker thread, but only ever touches memory from one at a time.
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=15)
        self.db.executescript(SCHEMA)
        self._embed = embed or _default_embedder()

    def close(self) -> None:
        self.db.close()

    # --- facts ---------------------------------------------------------------------------------------------------

    def _vector(self, text: str, is_query: bool) -> bytes | None:
        """Embedding as bytes, or None when the embedding service is unreachable. Memory must work offline."""
        try:
            return np.asarray(self._embed(text, is_query), dtype=np.float32).tobytes()
        except Exception:
            return None

    def remember(self, text: str, due: str | None = None) -> int:
        text = text.strip()
        cursor = self.db.execute(
            "INSERT INTO facts (created_at, updated_at, text, due, embedding) VALUES (?, ?, ?, ?, ?)",
            (_now(), _now(), text, due or None, self._vector(text, is_query=False)),
        )
        self.db.commit()
        return cursor.lastrowid

    def update_fact(self, fact_id: int, text: str | None = None, status: str | None = None,
                    due: str | None = None) -> bool:
        """Change a fact's wording, status or date. Returns False if there is no such fact."""
        if status is not None and status not in FACT_STATUSES:
            raise ValueError(f"status must be one of {FACT_STATUSES}")
        sets, values = ["updated_at = ?"], [_now()]
        if text is not None:
            sets += ["text = ?", "embedding = ?"]
            values += [text.strip(), self._vector(text, is_query=False)]
        if status is not None:
            sets.append("status = ?")
            values.append(status)
        if due is not None:
            sets.append("due = ?")
            values.append(due or None)
        cursor = self.db.execute(f"UPDATE facts SET {', '.join(sets)} WHERE id = ?", values + [fact_id])
        self.db.commit()
        return cursor.rowcount > 0

    def active_facts(self, limit: int = 15) -> list[Fact]:
        rows = self.db.execute(
            "SELECT id, created_at, text, status, due FROM facts WHERE status = 'active' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Fact(i, ts[:10], text, status, due) for i, ts, text, status, due in rows]

    def recall(self, query: str = "", limit: int = 8) -> list[Fact]:
        """Active facts related to the query: by meaning when embeddings are available, plus keyword matches.

        With an empty query, or when nothing relates, returns the most recent facts.
        """
        query = query.strip()
        if not query:
            return self.active_facts(limit)

        rows = self.db.execute(
            "SELECT id, created_at, text, status, due, embedding FROM facts WHERE status = 'active'"
        ).fetchall()
        scores: dict[int, float] = {}

        query_vector = self._vector(query, is_query=True)
        if query_vector is not None:
            q = np.frombuffer(query_vector, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            for row in rows:
                if row[5] is None:
                    continue
                v = np.frombuffer(row[5], dtype=np.float32)
                if v.shape != q.shape:
                    continue  # saved with a different embedding model
                scores[row[0]] = float(q @ (v / (np.linalg.norm(v) or 1.0)))
            cutoff = max(_MIN_SIMILARITY, max(scores.values(), default=0.0) - _SIMILARITY_MARGIN)
            scores = {i: s for i, s in scores.items() if s >= cutoff}

        words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in _STOPWORDS and len(w) > 2]
        for row in rows:
            if any(w in row[2].lower() for w in words):
                scores[row[0]] = max(scores.get(row[0], 0.0), 1.0)  # an exact word match outranks a fuzzy one

        if not scores:
            return self.active_facts(limit)
        by_id = {row[0]: Fact(row[0], row[1][:10], row[2], row[3], row[4]) for row in rows}
        ranked = sorted(scores, key=lambda i: (scores[i], i), reverse=True)[:limit]
        return [by_id[i] for i in ranked]

    # --- conversation and events -------------------------------------------------------------------------------------

    def log_turn(self, speaker: str, text: str) -> None:
        self.db.execute("INSERT INTO conversation (ts, speaker, text) VALUES (?, ?, ?)", (_now(), speaker, text))
        self.db.commit()

    def recent_turns(self, limit: int = 12) -> list[tuple[str, str, str]]:
        """The latest conversation turns as (timestamp, speaker, text), oldest first."""
        rows = self.db.execute(
            "SELECT ts, speaker, text FROM conversation ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return rows[::-1]

    def turns_since(self, ts: str | None) -> list[tuple[str, str, str]]:
        rows = self.db.execute(
            "SELECT ts, speaker, text FROM conversation WHERE ts > ? ORDER BY id", (ts or "",)
        ).fetchall()
        return rows

    def turns_on(self, day: date) -> list[tuple[str, str, str]]:
        return self.db.execute(
            "SELECT ts, speaker, text FROM conversation WHERE ts LIKE ? ORDER BY id", (day.isoformat() + "%",)
        ).fetchall()

    def events_on(self, day: date) -> list[tuple[str, str, str]]:
        return self.db.execute(
            "SELECT ts, kind, detail FROM events WHERE ts LIKE ? ORDER BY id", (day.isoformat() + "%",)
        ).fetchall()

    def last_sign_of_person(self, day: date) -> str | None:
        """Timestamp of the latest evidence that the person was around on that day: something they said,
        a face seen, a pet, or a reminder they confirmed or declined."""
        prefix = day.isoformat() + "%"
        rows = [
            self.db.execute("SELECT max(ts) FROM conversation WHERE speaker = 'person' AND ts LIKE ?", (prefix,)),
            self.db.execute("SELECT max(ts) FROM events WHERE kind = 'presence' AND ts LIKE ?", (prefix,)),
            self.db.execute(
                "SELECT max(ts) FROM reminder_log WHERE outcome IN ('confirmed', 'declined') AND ts LIKE ?", (prefix,)
            ),
        ]
        stamps = [row.fetchone()[0] for row in rows]
        return max((s for s in stamps if s), default=None)

    def log_event(self, kind: str, detail: str) -> None:
        self.db.execute("INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)", (_now(), kind, detail))
        self.db.commit()

    def last_event_time(self, kind: str) -> str | None:
        row = self.db.execute("SELECT ts FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        return row[0] if row else None

    # --- family inbox ------------------------------------------------------------------------------------------------
    # Family messages are written here the moment they arrive and marked handled only once dealt with,
    # so a restart or crash cannot lose one.

    def inbox_add(self, update_id: int, chat_id: int, sender: str, text: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO inbox (update_id, ts, chat_id, sender, text) VALUES (?, ?, ?, ?, ?)",
            (update_id, _now(), chat_id, sender, text),
        )
        self.db.commit()

    def inbox_next(self, max_attempts: int = 3) -> tuple[int, int, str, str, str] | None:
        """The oldest unhandled message as (update_id, chat_id, sender, text, received_at), counting this as a try.

        A message that keeps failing is given up on after `max_attempts`, so it cannot wedge the service.
        """
        row = self.db.execute(
            "SELECT update_id, chat_id, sender, text, ts FROM inbox WHERE handled = 0 AND attempts < ? "
            "ORDER BY update_id LIMIT 1", (max_attempts,),
        ).fetchone()
        if row:
            self.db.execute("UPDATE inbox SET attempts = attempts + 1 WHERE update_id = ?", (row[0],))
            self.db.commit()
        return row

    def inbox_done(self, update_id: int) -> None:
        self.db.execute("UPDATE inbox SET handled = 1 WHERE update_id = ?", (update_id,))
        self.db.commit()

    # --- heard through Vector ----------------------------------------------------------------------------------------
    # Requests that came in through Vector's own microphone (wake word or back button) and that wire-pod did
    # not recognise as one of his built-in commands. Durable for the same reason as the family inbox.

    def heard_add(self, text: str, ack: str = "") -> int:
        cursor = self.db.execute("INSERT INTO heard (ts, text, ack) VALUES (?, ?, ?)", (_now(), text.strip(), ack))
        self.db.commit()
        return cursor.lastrowid

    def heard_next(self, max_attempts: int = 2) -> tuple[int, str, str, str] | None:
        """The oldest unhandled request as (id, text, ack, received_at). Does not count as a try; see heard_try."""
        return self.db.execute(
            "SELECT id, text, ack, ts FROM heard WHERE handled = 0 AND attempts < ? ORDER BY id LIMIT 1",
            (max_attempts,),
        ).fetchone()

    def heard_try(self, heard_id: int) -> None:
        self.db.execute("UPDATE heard SET attempts = attempts + 1 WHERE id = ?", (heard_id,))
        self.db.commit()

    def heard_done(self, heard_id: int) -> None:
        self.db.execute("UPDATE heard SET handled = 1 WHERE id = ?", (heard_id,))
        self.db.commit()

    # --- reminders -------------------------------------------------------------------------------------------------

    def add_reminder(self, text: str, time_of_day: str, days: str = "daily") -> int:
        datetime.strptime(time_of_day, "%H:%M")  # raises ValueError on a bad time
        days = days.lower().replace(" ", "")
        if days != "daily" and not all(d in _DAYS for d in days.split(",")):
            raise ValueError(f"days must be 'daily' or a comma-separated list of {_DAYS}")
        cursor = self.db.execute(
            "INSERT INTO reminders (created_at, text, time_of_day, days) VALUES (?, ?, ?, ?)",
            (_now(), text.strip(), time_of_day, days),
        )
        self.db.commit()
        return cursor.lastrowid

    def reminders(self) -> list[Reminder]:
        rows = self.db.execute(
            "SELECT id, text, time_of_day, days FROM reminders WHERE active = 1 ORDER BY time_of_day"
        ).fetchall()
        return [Reminder(*row) for row in rows]

    def deactivate_reminder(self, reminder_id: int) -> bool:
        cursor = self.db.execute("UPDATE reminders SET active = 0 WHERE id = ?", (reminder_id,))
        self.db.commit()
        return cursor.rowcount > 0

    def log_reminder(self, reminder_id: int, outcome: str) -> None:
        if outcome not in REMINDER_OUTCOMES:
            raise ValueError(f"outcome must be one of {REMINDER_OUTCOMES}")
        self.db.execute(
            "INSERT INTO reminder_log (reminder_id, ts, outcome) VALUES (?, ?, ?)", (reminder_id, _now(), outcome)
        )
        self.db.commit()

    def reminders_due(self, now: datetime | None = None, renudge_minutes: int = 10,
                      max_attempts: int = 3) -> list[Reminder]:
        """Reminders to act on now: their time has passed today, they are not settled (confirmed or declined),
        the last attempt was not too recent, and Kin has not already given up for the day."""
        now = now or datetime.now()
        today, weekday = now.date().isoformat(), _DAYS[now.weekday()]
        due = []
        for reminder in self.reminders():
            if reminder.days != "daily" and weekday not in reminder.days.split(","):
                continue
            if reminder.time_of_day > now.strftime("%H:%M"):
                continue
            log = self.db.execute(
                "SELECT ts, outcome FROM reminder_log WHERE reminder_id = ? AND ts >= ? ORDER BY id",
                (reminder.id, today),
            ).fetchall()
            if any(outcome in ("confirmed", "declined") for _, outcome in log):
                continue
            if _attempts([outcome for _, outcome in log]) >= max_attempts:
                continue  # gave up for today; the service reports this to the family
            if log and (now - datetime.fromisoformat(log[-1][0])).total_seconds() < renudge_minutes * 60:
                continue  # asked recently; leave them in peace for a while
            due.append(reminder)
        return due

    def reminders_missed(self, now: datetime | None = None, max_attempts: int = 3) -> list[Reminder]:
        """Reminders Kin gave up on today after `max_attempts` tries with no response."""
        today = (now or datetime.now()).date().isoformat()
        missed = []
        for reminder in self.reminders():
            rows = self.db.execute(
                "SELECT outcome FROM reminder_log WHERE reminder_id = ? AND ts >= ?", (reminder.id, today)
            ).fetchall()
            outcomes = [r[0] for r in rows]
            if _attempts(outcomes) >= max_attempts and not {"confirmed", "declined"} & set(outcomes):
                missed.append(reminder)
        return missed

    def reminder_history(self, since: date) -> list[tuple[str, str, str]]:
        """(timestamp, reminder text, outcome) since a date, for the family digest."""
        return self.db.execute(
            "SELECT l.ts, r.text, l.outcome FROM reminder_log l JOIN reminders r ON r.id = l.reminder_id "
            "WHERE l.ts >= ? ORDER BY l.id",
            (since.isoformat(),),
        ).fetchall()
