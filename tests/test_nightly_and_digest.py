import pytest

from kin import config, digest, nightly


def test_the_job_payload_and_result_survive_the_trip(capsys):
    payload = {"today": "Saturday, 2026-09-19", "person": "Jennie", "facts": "#1 a fact", "conversation": "", "week": []}
    encoded = nightly.encode_payload(payload)
    assert "\n" not in encoded and " " not in encoded, "it has to fit in one environment variable"

    result = {"operations": [{"op": "add", "text": "Émile called."}], "trends": "t", "issues": [], "errors": []}
    import base64, json, zlib
    packed = base64.b64encode(zlib.compress(json.dumps(result).encode())).decode()
    log = f"pulling image...\n{nightly.RESULT_BEGIN}\n{packed}\n{nightly.RESULT_END}\njob finished\n"

    assert nightly.decode_result(log) == result


def test_apply_validates_operations_against_memory(memory):
    kept = memory.remember("Dentist on 2026-09-15.", due="2026-09-15")
    result = {
        "operations": [
            {"op": "done", "id": kept, "follow_up": "Ask how the dentist visit went."},
            {"op": "retire", "id": 999, "reason": "the model made this id up"},
            {"op": "add", "text": "The person's knee has been aching.", "date": "not-a-date"},
            {"op": "explode"},
        ],
        "trends": "Evening pills missed three days running.",
        "issues": [{"rule": "medical", "quote": "take two ibuprofen", "severity": "high", "why": "advice"}],
        "errors": ["trends: ValueError: boom"],
    }

    summary = nightly.apply(memory, result)

    assert [f.text for f in memory.active_facts()] == [
        "The person's knee has been aching.", "Ask how the dentist visit went."]
    assert memory.active_facts()[0].due is None, "a malformed date is dropped, not stored"
    assert "closed 1" in summary and "added 1" in summary and "skipped 2" in summary
    assert "1 issue(s), 1 high" in summary and "1 part(s) failed" in summary
    assert nightly.latest_trends(memory) == "Evening pills missed three days running."
    kinds = [kind for _, kind, _ in memory.events_on(__import__("datetime").date.today())]
    assert {"self_check", "nightly_error", "trends", "consolidation"} <= set(kinds)


def test_a_part_that_keeps_failing_is_recorded_not_raised():
    errors, calls = [], []

    def flaky():
        calls.append(1)
        raise ValueError("malformed JSON")

    assert nightly._retrying("self-check", flaky, errors) is None
    assert len(calls) == 3
    assert errors == ["self-check: ValueError: malformed JSON"]


def test_a_part_that_recovers_is_not_an_error():
    errors, calls = [], []

    def second_time_lucky():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("malformed JSON")
        return ["ok"]

    assert nightly._retrying("memory", second_time_lucky, errors) == ["ok"]
    assert errors == []


def test_export_describes_the_week_without_the_conversation_history(memory):
    reminder_id = memory.add_reminder("morning pills", "00:01")
    memory.log_reminder(reminder_id, "asked")
    memory.log_reminder(reminder_id, "confirmed")
    memory.log_turn("person", "Good morning.")
    memory.log_event("checkin", "done")
    memory.log_event("digest_note", "A cheerful day.")

    payload = nightly.export(memory)

    assert len(payload["week"]) == 7
    today = payload["week"][-1]
    assert today["checkin"] and today["spoke"] == 1
    assert today["reminders"] == {"morning pills": "confirmed"}
    assert today["note"] == "A cheerful day."
    assert all(not day["spoke"] for day in payload["week"][:-1])


def test_with_under_two_active_days_trends_say_so_without_calling_a_model():
    payload = {"today": "x", "person": "Jennie", "facts": "", "conversation": "",
               "week": [{"day": "Sat", "checkin": True, "spoke": 3, "reminders": {}, "alerts": [], "note": ""}]}

    result = nightly.run(payload)  # would need credentials if it called a model

    assert result["trends"].startswith("Not enough history yet")
    assert result["operations"] == [] and result["issues"] == [] and result["errors"] == []


def test_status_shows_the_final_outcome_per_reminder(memory):
    reminder_id = memory.add_reminder("afternoon walk", "00:01")
    for _ in range(3):
        memory.log_reminder(reminder_id, "asked")
        memory.log_reminder(reminder_id, "no_response")
    memory.log_turn("person", "hello")

    text = digest.status(memory)

    assert text.count("afternoon walk") == 1 and text.count("no response") == 1, "a run of the same outcome is one entry"
    assert "afternoon walk: no response" in text
    assert "Spoke with Kin 1 time" in text and "Check-in: not yet" in text and text.startswith("Today (")


def test_status_tells_the_day_in_times(memory, monkeypatch):
    """Oct 29: /status for yesterday read as "Today" and gave the final word per reminder with no times."""
    from datetime import date, timedelta

    yesterday = date.today() - timedelta(days=1)
    now = {"ts": f"{yesterday}T12:00:00"}
    monkeypatch.setattr("kin.memory._now", lambda: now["ts"])
    pills = memory.add_reminder("take your pills", "12:00")
    for stamp, action in [("12:37", lambda: memory.log_reminder(pills, "confirmed")),
                          ("13:10", lambda: memory.log_reminder(pills, "no_response")),
                          ("12:19", lambda: memory.log_event("checkin", "Checked in: fine, plans a walk.")),
                          ("12:40", lambda: memory.log_event("family_in", "Vlad: do you need apples?"))]:
        now["ts"] = f"{yesterday}T{stamp}:00"
        action()

    text = digest.status(memory, yesterday)

    assert text.startswith("Yesterday (")
    assert "take your pills: confirmed 12:37, no response 13:10" in text
    assert "Check-in: done at 12:19" in text and "Family: 1 message delivered, 0 sent back" in text
    assert "Last sign of" in text


def test_a_quiet_day_needs_no_model_call(memory):
    assert "a quiet day" in digest.build(memory)


def test_settings_overridden_by_the_environment_are_reported_with_secrets_masked(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("KIN_MOUTH=vector\nNEBIUS_API_KEY=from-file\nKIN_EARS=hub\n", "utf-8")
    monkeypatch.setenv("KIN_MOUTH", "hub")
    monkeypatch.setenv("NEBIUS_API_KEY", "from-shell")
    monkeypatch.setenv("KIN_EARS", "hub")

    assert config.overridden() == {"KIN_MOUTH": ("hub", "vector"), "NEBIUS_API_KEY": ("***", "***")}


@pytest.mark.parametrize("value", ["loud", "speaker"])
def test_a_mistyped_choice_setting_is_an_error_not_a_silent_default(monkeypatch, value):
    monkeypatch.setenv("KIN_MOUTH", value)

    with pytest.raises(RuntimeError):
        config.mouth()


def test_the_weekly_note_is_on_record_when_sent(memory, family):
    """Until Oct 9 the Sunday send was not logged, so memory could not tell whether the family had ever
    received a weekly note. Now both the Sunday send and /week leave a 'weekly' event."""
    from datetime import date

    from kin import service

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family
    memory.log_event("trends", "Water reminders were missed on Monday and Thursday.")

    kin._send_weekly_note()
    kin._run_family_command(chat_id=42, text="/week")

    assert [s.startswith("Kin's weekly note\n\n") for s in family.sent] == [True, True]
    assert "missed on Monday and Thursday" in family.sent[0]
    weekly = [detail for _, kind, detail in memory.events_on(date.today()) if kind == "weekly"]
    assert weekly == ["sent to 1 chat(s)", "sent to 1 chat(s) on request"]


def test_the_weekly_note_is_about_the_person_not_the_robot(monkeypatch):
    """The Sep 29 and 30 notes called the person's speech 'disjointed' and 'confused' (a guess about her
    mind from a bad microphone) and most of them led with robot outages. The prompt now forbids both."""
    prompts = []
    monkeypatch.setattr(nightly, "_ask", lambda prompt, max_tokens=8000: prompts.append(prompt) or "A quiet week.")
    day = {"day": "Mon 2026-10-05", "checkin": True, "spoke": 3, "reminders": {"drink water": "confirmed"},
           "alerts": [], "note": "Jennie said her knee aches."}
    payload = {"today": "Thursday, 2026-10-08", "person": "Jennie", "facts": "", "conversation": "",
               "week": [day, dict(day, day="Tue 2026-10-06")]}

    result = nightly.run(payload)

    assert result["trends"] == "A quiet week."
    prompt = prompts[0]
    assert "It is about Jennie, not about the robot" in prompt
    assert "never remark on how clearly they spoke, on confusion, forgetfulness or repetition" in prompt
    assert "could not reach his robot, or has no record at all, says nothing about them" in prompt
    assert "at most one short sentence at the very end" in prompt
    assert "Jennie said her knee aches." in prompt, "the daily notes are the evidence it works from"


def test_a_day_with_no_record_is_shown_as_kin_not_running():
    """Oct 6 and 7 the hub was off. Shown as 'spoke 0x, no check-in' the model wrote that the person was
    silent while the robot was working; an empty day is Kin's absence, not hers."""
    empty = {"day": "Tue 2026-10-06", "checkin": False, "spoke": 0, "reminders": {}, "alerts": [], "note": ""}
    quiet = dict(empty, day="Wed 2026-10-07", checkin=True)

    text = nightly._week_text([empty, quiet])

    assert text.splitlines()[0] == "Tue 2026-10-06 | no record at all: Kin was not running that day"
    assert text.splitlines()[1].startswith("Wed 2026-10-07 | check-in: yes | spoke 0x"), "a check-in with no reply is a record"


def test_the_night_can_be_served_over_http_and_called_from_the_hub(memory, monkeypatch):
    """Oct 9: the night as a Nebius serverless container. The hub posts the payload and applies the answer;
    the container needs no database, and a bad token or payload leaves it standing."""
    import threading
    from http.server import ThreadingHTTPServer

    import httpx

    seen = {}
    def fake_run(payload):
        seen.update(payload)
        return {"operations": [], "trends": "A good week.", "issues": [], "errors": []}
    monkeypatch.setattr(nightly, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), nightly._handler("s3cret"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        assert httpx.get(url).json() == {"ok": True, "service": "kin-nightly"}
        assert httpx.post(url, json={}).status_code == 401, "no token, no night"
        assert httpx.post(url, content=b"not json", headers={"Authorization": "Bearer s3cret"}).status_code == 400

        summary = nightly.run_remotely(memory, url, "s3cret")

        assert "self-check found 0 issue(s)" in summary
        assert seen["person"] and isinstance(seen["week"], list) and len(seen["week"]) == 7, "the real export went over"
        assert nightly.latest_trends(memory) == "A good week."
    finally:
        server.shutdown()


def test_when_the_cloud_night_fails_the_hub_does_it_and_says_so(memory, monkeypatch):
    monkeypatch.setenv("KIN_NIGHTLY_URL", "http://127.0.0.1:9/")  # nothing listens there
    monkeypatch.setattr(nightly, "run_locally", lambda m: "memory updated 0; self-check found 0 issue(s), 0 high")

    summary = nightly.run_nightly(memory)

    assert summary.endswith("(on the hub; the cloud service failed: ConnectError)")
    errors = [d for _, k, d in memory.events_on(__import__("datetime").date.today()) if k == "nightly_error"]
    assert len(errors) == 1 and errors[0].startswith("cloud: ConnectError")

    monkeypatch.delenv("KIN_NIGHTLY_URL")
    assert nightly.run_nightly(memory).endswith("(on the hub)")


def test_the_container_modules_load_without_the_database_module(monkeypatch):
    """The Nebius container gets five files, not six: memory.py (18 KiB) tipped the console form over its
    size cap. The night's thinking never touches the database, so the modules must import without it."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "kin.memory", None)  # makes `import kin.memory` fail
    for name in ("kin.consolidate", "kin.nightly"):
        importlib.reload(sys.modules[name])
    assert nightly.run({"today": "x", "person": "p", "facts": "", "conversation": "", "week": []})["trends"].startswith("Not enough history")


def test_digest_can_be_asked_for_yesterday(memory, family, monkeypatch):
    """Oct 29: the evening note goes out at 20:00; "/digest" the next morning gave only the new day's
    outage line, and there was no way to ask for the full note again."""
    from datetime import date, timedelta

    from kin import digest, service

    asked = []
    monkeypatch.setattr(digest, "build", lambda mem, day=None: asked.append(day) or f"note for {day}")
    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family = memory, family

    kin._run_family_command(chat_id=42, text="/digest")
    kin._run_family_command(chat_id=42, text="/Digest yesterday")
    kin._run_family_command(chat_id=42, text="/digest 2026-10-28")
    kin._run_family_command(chat_id=42, text="/digest nonsense")
    assert asked == [None, date.today() - timedelta(days=1), date(2026, 10, 28), None]
    assert len(family.sent) == 4
