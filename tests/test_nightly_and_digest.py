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

    assert text.count("afternoon walk") == 1
    assert "afternoon walk: no response" in text
    assert "Spoke to Kin 1 time(s)" in text and "Check-in: not yet" in text


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
