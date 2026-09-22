from datetime import datetime

import httpx
import pytest
from openai import OpenAI

from kin import config, gateway, wirepod
from kin.memory import Memory


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The gateway on a free port, writing to a database in the test's own directory."""
    monkeypatch.setenv("KIN_DB_PATH", str(tmp_path / "kin.db"))
    server = gateway.start(port=0)
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_vector_acknowledges_in_his_own_words_and_the_request_is_queued(served, tmp_path):
    client = OpenAI(base_url=served, api_key="kin-local")  # wire-pod talks to it the same way, streaming

    stream = client.chat.completions.create(model="kin", stream=True, messages=[
        {"role": "system", "content": "You are a helpful, animated robot called Vector."},
        {"role": "user", "content": "  what is on tv   tonight "},
    ])
    spoken = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert spoken in config.ack_phrases()
    heard_id, text, ack, _ = Memory(tmp_path / "kin.db").heard_next()
    assert text == "what is on tv tonight" and ack == spoken


def test_the_gateway_also_answers_a_plain_request(served):
    reply = OpenAI(base_url=served, api_key="x").chat.completions.create(
        model="kin", messages=[{"role": "user", "content": "tell me a joke"}])

    assert reply.choices[0].message.content in config.ack_phrases()


def test_nothing_heard_means_nothing_queued(served, tmp_path):
    reply = OpenAI(base_url=served, api_key="x").chat.completions.create(
        model="kin", messages=[{"role": "system", "content": "only a system prompt"}])

    assert reply.choices[0].message.content == "Sorry, I did not catch that."
    assert Memory(tmp_path / "kin.db").heard_next() is None


def test_acknowledgements_vary_and_can_be_customised(monkeypatch):
    said = [gateway.pick_ack() for _ in range(30)]
    assert all(a != b for a, b in zip(said, said[1:])), "never the same phrase twice running"
    assert len(set(said)) > 3

    monkeypatch.setenv("KIN_ACK_PHRASES", "Righto! | Leave it with me.")
    assert {gateway.pick_ack() for _ in range(10)} == {"Righto!", "Leave it with me."}


def test_the_ack_is_available_to_wire_pod_scripts(served):
    assert httpx.get(served.removesuffix("/v1") + "/ack").text in config.ack_phrases()


def test_a_heard_request_is_durable_and_cannot_be_retried_forever(memory):
    heard_id = memory.heard_add("what is on tv tonight", "On it!")

    assert memory.heard_next()[:3] == (heard_id, "what is on tv tonight", "On it!")
    memory.heard_try(heard_id)
    assert memory.heard_next() is not None, "one failed try leaves it queued"
    memory.heard_try(heard_id)
    assert memory.heard_next() is None, "two and it is dropped"

    other = memory.heard_add("and the news", "Got it.")
    memory.heard_done(other)
    assert memory.heard_next() is None


def test_pressing_his_back_button_is_not_also_a_pet(memory, monkeypatch):
    """The button is beside the touch sensor. Live, the answer was followed by a second, pet-started
    conversation asking what the person wanted."""
    from types import SimpleNamespace

    from kin import service

    class Touch:
        pending = True

        def discard(self):
            self.pending = False

    kin = service.Service.__new__(service.Service)  # only the heard path; no voice, robot or family
    kin.memory, kin._touch, kin._warmed_for = memory, Touch(), None
    ran = []
    agent = SimpleNamespace(warm_up=lambda: ran.append("warm"), run=lambda *a, **k: ran.append("run") or "ok")

    memory.heard_add("should i take a jacket", "Let me think.")
    assert kin._handle_heard(agent) is False, "Vector is still saying his acknowledgement"
    assert ran == ["warm"] and not kin._touch.pending, "the wait warms the planner and forgets the press"

    kin._touch.pending = True  # a hand on his back while Kin was answering
    monkeypatch.setattr(service, "ACK_SECONDS", -60)  # the test clock runs a little ahead of the real one
    assert kin._handle_heard(agent) is True
    assert ran == ["warm", "run"] and not kin._touch.pending
    assert memory.heard_next() is None


def test_a_pet_is_just_a_pet_unless_pet_to_talk_is_on(memory, monkeypatch):
    from types import SimpleNamespace

    from kin import service

    class Touch:
        def __init__(self):
            self.pets = 1

        def petted(self):
            self.pets -= 1
            return self.pets >= 0

        def discard(self):
            pass

    kin = service.Service.__new__(service.Service)
    kin.memory, kin.family, kin._touch, kin._warmed_for = memory, None, Touch(), None
    for quiet in ("_heartbeat", "_rescue_battery", "_presence", "_checkin_due", "_acknowledge_touch"):
        monkeypatch.setattr(kin, quiet, lambda *a, **k: False, raising=False)
    monkeypatch.setattr(memory, "reminders_due", lambda *a, **k: [])
    prompts = []
    agent = SimpleNamespace(warm_up=lambda: None, run=lambda text, **k: prompts.append(text) or "ok")

    kin.tick(None, agent)
    assert prompts == [], "no conversation"
    assert memory.last_sign_of_person(datetime.now().date()), "but it still shows she is around"

    monkeypatch.setenv("KIN_PET_TO_TALK", "on")
    kin._touch = Touch()
    assert kin.tick(None, agent) is True
    assert "Anything else?" in prompts[0] and "go_home" not in prompts[0]


def test_going_home_is_only_asked_for_when_kin_may_have_left_the_charger(monkeypatch):
    from kin import service
    from kin.memory import Reminder

    reminder = Reminder(id=1, text="drink water", time_of_day="18:00", days="daily")
    assert "go_home" not in service.reminder_prompt(reminder, 1)

    monkeypatch.setenv("KIN_STAY_ON_CHARGER", "off")
    assert "go_home" in service.reminder_prompt(reminder, 1)


LOG = """\
2026.09.20 11:02:01: Bot 0dd19176 Stream type: OPUS
2026.09.20 11:02:03: Bot 0dd19176 Transcribed text: i love you
2026.09.20 11:02:03: Not a custom intent
2026.09.20 11:02:03: Bot 0dd19176 Intent Sent: intent_imperative_love
2026.09.20 11:02:03: Bot 0dd19176 request served.
2026.09.20 11:05:40: Bot 0dd19176 Transcribed text: what is on tv tonight
2026.09.20 11:05:40: Not a custom intent
2026.09.20 11:05:41: Making LLM request for device 0dd19176...
2026.09.20 11:07:12: Bot 0dd19176 Transcribed text: mumble
2026.09.20 11:07:12: Bot 0dd19176 Intent Sent: intent_system_unmatched
2026.09.20 11:09:30: Bot 0dd19176 Transcribed text: do a wheelie
2026.09.20 11:09:30: Bot 0dd19176 Intent Sent: intent_imperative_wheelie
""".splitlines()


def test_built_in_commands_are_recognised_in_wire_pods_log():
    assert wirepod.interactions(LOG) == [("i love you", "intent_imperative_love"),
                                         ("do a wheelie", "intent_imperative_wheelie")]
    assert wirepod.describe("intent_imperative_love") == "told Vector they love him"
    assert wirepod.describe("intent_imperative_wheelie") == "used Vector's built-in command imperative_wheelie"


def test_a_weather_question_is_handed_to_kin_with_the_words_from_wire_pods_log(served, tmp_path, monkeypatch):
    assert wirepod.last_transcript(LOG, now="2026.09.20 11:09:34") == "do a wheelie"
    assert wirepod.last_transcript(LOG, now="2026.09.20 11:12:00") == "", "stale: not what he heard just now"
    assert wirepod.last_transcript([], now="2026.09.20 11:12:00") == ""

    monkeypatch.setattr(wirepod, "heard_just_now", lambda: "what is the weather today")
    ack = httpx.get(served.removesuffix("/v1") + "/handover").text
    assert ack in config.ack_phrases(), "Vector says this himself"
    assert Memory(tmp_path / "kin.db").heard_next()[1:3] == ("what is the weather today", ack)

    monkeypatch.setattr(wirepod, "heard_just_now", lambda: "")
    assert httpx.get(served.removesuffix("/v1") + "/handover").text == "", "nothing heard: he says nothing"


def test_the_takeover_claims_weather_words_without_catching_others():
    said = "should i catch the train to the brain clinic"
    assert not any(words in said for words in wirepod.TAKEOVER_WORDS)
    assert any(words in "what's the weather like" for words in wirepod.TAKEOVER_WORDS)
    assert "/handover" in wirepod.takeover_script()


def test_the_watcher_reports_each_log_line_once_and_ignores_history():
    watcher = wirepod.IntentWatcher(start=False)

    assert watcher.fresh(LOG[:5]) == [], "whatever is in the log when Kin starts is history"
    assert watcher.fresh(LOG[:8]) == LOG[5:8]
    assert watcher.fresh(LOG[:8]) == []
    assert watcher.fresh(LOG[3:]) == LOG[8:], "wire-pod trimmed the start of its log; nothing is repeated"


def test_what_the_person_says_to_vector_himself_reaches_kins_context(memory):
    from tests.test_agent_loop import make_agent, reply

    memory.log_event("vector_intent", 'told Vector they love him: "i love you"')
    agent, _, _ = make_agent(memory, [reply(content="ok")])

    context = agent._context()

    assert 'told Vector they love him: "i love you"' in context
    assert datetime.now().strftime("%H:%M")[:2] in context
