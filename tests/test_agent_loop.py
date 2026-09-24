import json
from types import SimpleNamespace

import pytest

from kin.agent import Agent, speakable, thinking_continues


class StubMouth:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class StubEars:
    def __init__(self, replies):
        self.replies = list(replies)

    def listen(self, timeout=12.0):
        return self.replies.pop(0) if self.replies else None


def tool_call(name, **arguments):
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def reply(*calls, content=""):
    message = SimpleNamespace(content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_agent(memory, script, ears=()):
    """An agent whose planner replies come from `script`; records whether each wait was animated."""
    robot = SimpleNamespace(
        anim=SimpleNamespace(anim_trigger_list=["KnowledgeGraphSearching"], anim_list=["anim_knowledgegraph_searching_01"]),
        conn=SimpleNamespace(requires_behavior_control=True),  # control() then has nothing to borrow
    )
    mouth = StubMouth()
    agent = Agent(robot, memory, mouth=mouth, ears=StubEars(ears), verbose=False)
    waits = []

    def think(someone_is_waiting=True):
        waits.append(someone_is_waiting)
        return script.pop(0)

    agent._think = think
    return agent, mouth, waits


def test_no_thinking_show_after_the_last_thing_kin_says(memory):
    agent, mouth, waits = make_agent(memory, [
        reply(tool_call("weather")),                        # the person asked: they are waiting
        reply(tool_call("say", text="It's 66 and clear.")),  # still working on their answer
        reply(content="Told the person the weather."),      # only the private summary is left
    ])
    agent.tools._tool_weather = lambda place="": "66 degrees Fahrenheit, clear."

    summary = agent.run("What's the weather?")

    assert mouth.said == ["It's 66 and clear."]
    assert summary == "Told the person the weather."
    assert waits == [True, True, False]


def test_kin_shows_he_is_thinking_again_once_the_person_has_answered(memory):
    agent, mouth, waits = make_agent(memory, [
        reply(tool_call("say", text="Good morning!")),
        reply(tool_call("ask", question="How did you sleep?")),  # after a `say`: nobody is waiting
        reply(tool_call("remember", fact="The person slept badly on 2026-09-19.")),  # they answered: waiting
        reply(content="Checked in."),
    ], ears=["Badly, my knee hurt."])

    agent.run("[Daily check-in.]", speaker="system")

    assert waits == [False, False, True, False], "no thinking show before his first words: nobody is waiting yet"
    assert [text for _, who, text in memory.recent_turns() if who == "person"] == ["Badly, my knee hurt."]
    assert [f.text for f in memory.active_facts()] == ["The person slept badly on 2026-09-19."]


def test_a_check_in_starts_without_a_thinking_show_and_a_question_in_say_is_listened_to(memory):
    """The 09:00 check-in, live: two thinking shows before a word was said, then "How did you sleep?" in a
    `say` followed by the same question in an `ask`."""
    agent, mouth, waits = make_agent(memory, [
        reply(tool_call("look_for_face")),
        reply(tool_call("say", text="Good morning, Jennie! How did you sleep?")),
        reply(tool_call("say", text="Glad to hear it.")),
        reply(content="Checked in."),
    ], ears=["Very well, thank you."])
    agent.tools._tool_look_for_face = lambda: "no one in view from the charger"

    agent.run("[Daily check-in.]", speaker="system")

    assert waits[:2] == [False, False], "deciding to look, then to speak: both silent"
    assert waits[2] is True, "she answered the question in the `say`, so now she is waiting on him"
    assert [text for _, who, text in memory.recent_turns() if who == "person"] == ["Very well, thank you."]
    assert mouth.said == ["Good morning, Jennie! How did you sleep?", "Glad to hear it."]


def test_anything_else_met_by_silence_ends_the_conversation_quietly(memory):
    agent, mouth, waits = make_agent(memory, [
        reply(tool_call("say", text="Take a light jacket.")),
        reply(tool_call("ask", question="Anything else?")),
        reply(content="Answered the jacket question; they had walked away."),
    ], ears=[])

    agent.run("should I take a jacket?")

    assert mouth.said == ["Take a light jacket.", "Anything else?"], "the log line is not said to an empty room"
    assert waits == [True, False, False]


def test_a_no_to_anything_else_gets_a_short_spoken_sign_off(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Take a light jacket.")),
        reply(tool_call("ask", question="Anything else?")),
        reply(content="Okay, I'm here if you need me."),
    ], ears=["No, thank you."])

    agent.run("should I take a jacket?")

    assert mouth.said[-1] == "Okay, I'm here if you need me."


def test_system_triggers_are_not_logged_as_something_the_person_said(memory):
    agent, _, _ = make_agent(memory, [reply(content="Nothing to do.")])

    agent.run("[Reminder #1 is due.]", speaker="system")

    assert memory.recent_turns() == []


def test_a_runaway_plan_is_stopped(memory):
    agent, mouth, _ = make_agent(memory, [reply(tool_call("say", text="again")) for _ in range(40)])

    assert agent.run("loop forever") == "stopped: too many steps"
    assert len(mouth.said) == 24


@pytest.mark.parametrize("closing, spoken", [
    ("It's 59 degrees and mostly clear.", "It's 59 degrees and mostly clear."),
    ("<tool_call>\n<function=say>\nOK, I've set the reminder.\n</parameter>\n</function>\n</tool_call>", "OK, I've set the reminder."),
    ("**Done!** I saved `that` for you.", "Done! I saved that for you."),
    ("One. Two. Three. Four. Five.", "One. Two. Three."),
    ("<tool_call></tool_call>", ""),
    ("", ""),
    # notes to itself, not speech (the first was read out live, after the person had just answered him)
    ("(End of task – no further spoken response.)", ""),
    ("Task completed.", ""),
    ("The person confirmed and I logged it. No further action needed.", ""),
    ("[Closing summary: asked about the appointment]", ""),
    ("Okay, I'm here if you need me.", "Okay, I'm here if you need me."),
    ("Have a lovely day, Jennie!", "Have a lovely day, Jennie!"),
])
def test_only_plain_short_speech_reaches_the_person(closing, spoken):
    assert speakable(closing) == spoken


def test_an_answer_written_but_not_spoken_is_said_aloud(memory):
    agent, mouth, _ = make_agent(memory, [reply(content="It's 59 degrees and clear. " + "More detail. " * 10)])

    agent.run("What's the weather?")

    assert mouth.said == ["It's 59 degrees and clear. More detail. More detail."]


def test_a_verdict_written_after_a_lookup_is_said_even_though_kin_spoke_earlier(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Please don't do anything yet, let me look that up.")),
        reply(tool_call("web_search", query="frozen account card number scam")),
        reply(content="Now inform the person.</think>I looked it up, and this matches a known scam. Please don't read them your card number."),
    ])
    agent.tools._tool_web_search = lambda query, topic="general": "Summary: a known scam."

    agent.run("I got a call saying my account is frozen.")

    assert mouth.said == ["Please don't do anything yet, let me look that up.",
                          "I looked it up, and this matches a known scam. Please don't read them your card number."]


def test_a_reply_written_after_the_person_answers_is_said_even_on_a_schedule_trigger(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="How did you sleep?")), reply(content="Glad to hear it. Have a lovely day!"),
    ], ears=["Very well, thank you."])

    agent.run("[Daily check-in.]", speaker="system")

    assert mouth.said == ["How did you sleep?", "Glad to hear it. Have a lovely day!"]


def test_closing_text_stays_private_once_kin_has_spoken_or_for_schedule_triggers(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Good morning!")), reply(content="Greeted the person."),
        reply(content="Nothing was due."),
    ])

    agent.run("Hello")
    agent.run("[Reminder check.]", speaker="system")

    assert mouth.said == ["Good morning!"]

def test_a_person_who_has_heard_nothing_gets_the_closing_text_even_after_bookkeeping(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say")),  # a malformed call with no text: an error, nothing is spoken
        reply(tool_call("set_reminder", text="water the plants", time_of_day="20:00")),
        reply(content="Got it! I've set a daily reminder for 8 p.m. to water the plants."),
    ])

    agent.run("Remind me daily at 8 in the evening to water the plants.")

    assert mouth.said == ["Got it! I've set a daily reminder for 8 p.m. to water the plants."]

@pytest.mark.parametrize("tools_called, continues", [
    (["weather"], True),                    # a silent lookup: still one thought
    (["remember", "set_reminder"], True),
    (["say"], False),                       # about to speak
    (["weather", "say"], False),
    (["look_for_face"], False),             # about to move
    (["describe_scene"], False),            # needs the head still for the photo
    ([], False),                            # finished
])
def test_the_thinking_show_runs_through_silent_steps_and_ends_before_speech_or_movement(tools_called, continues):
    assert thinking_continues(tools_called) is continues


def test_one_continuous_think_from_question_to_answer(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("weather")),
        reply(tool_call("say", text="It's 59 and clear.")),
        reply(content="Told the person the weather."),
    ])
    agent.tools._tool_weather = lambda place="": "59 degrees Fahrenheit, clear."
    stops = []
    agent._stop_thinking = lambda: stops.append(len(mouth.said))

    agent.run("What's the weather?")

    # never stopped between the lookup and the answer: the first stop comes with nothing yet said, right
    # before the `say`; the rest are the harmless end-of-task ones
    assert stops[0] == 0 and all(said == 1 for said in stops[1:])
    assert len(stops) == 3

def test_a_stalled_planner_ends_the_task_and_tells_the_person(memory):
    """Sep 22, 09:01: a planner call hung mid check-in with no timeout; the thinking animation looped and
    the person was left waiting. Now the call is bounded, and if it fails he says so and stops."""
    def broken_planner(someone_is_waiting=True):
        raise TimeoutError("Request timed out.")

    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="How did you sleep?")),
    ], ears=["Well, thank you."])
    calls = [agent._think]

    def think(someone_is_waiting=True):
        return calls.pop(0)(someone_is_waiting) if calls else broken_planner()
    agent._think = think

    summary = agent.run("[Daily check-in.]", speaker="system")

    assert summary.startswith("stopped: planner unavailable")
    assert mouth.said == ["How did you sleep?", "Sorry, I've lost my train of thought. Let's talk again in a little while."]


def test_a_stalled_planner_before_anyone_answered_stays_silent(memory):
    agent, mouth, _ = make_agent(memory, [])
    agent._think = lambda someone_is_waiting=True: (_ for _ in ()).throw(ConnectionError("no route"))

    assert agent.run("[Reminder #1 is due.]", speaker="system").startswith("stopped: planner unavailable")
    assert mouth.said == [], "nobody was in the conversation yet; a robot talking to an empty room is worse than silence"


def test_a_reply_to_a_spoken_closing_question_carries_the_conversation_on(memory):
    """Sep 23, 09:15: "...Do you need anything else?" came as closing text, was said (a closing question
    listens), the person answered, and the answer was dropped."""
    agent, mouth, _ = make_agent(memory, [
        reply(content="It's 59 and clear. Do you need anything else?"),
        reply(tool_call("ask", question="Sure, what would you like to know?")),
        reply(content="Okay, I'm here if you need me."),
    ], ears=["Yes, one more thing.", "Nothing, never mind."])

    agent.run("tell me the weather")

    assert mouth.said == ["It's 59 and clear. Do you need anything else?", "Sure, what would you like to know?",
                          "Okay, I'm here if you need me."]
    assert [text for _, who, text in memory.recent_turns() if who == "person"] == \
        ["tell me the weather", "Yes, one more thing.", "Nothing, never mind."]
