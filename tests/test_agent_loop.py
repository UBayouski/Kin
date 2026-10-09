import json
from types import SimpleNamespace

import pytest

from kin.agent import SIGNED_OFF, Agent, sign_off, speakable, thinking_continues


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


def test_a_no_to_anything_else_gets_the_same_sign_off_whatever_the_planner_writes(memory):
    """Oct 8, 11:28: "No, I'm good." was answered with "Checked in with Jennie, she's good.", read out to her.
    Oct 3, 18:00: "Reminder handled, ..." was filtered out and she heard nothing. The sign-off is Kin's own."""
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Take a light jacket.")),
        reply(tool_call("ask", question="Anything else?")),
        reply(content="Checked in with Jennie, she's good."),
    ], ears=["No, thank you."])

    agent.run("should I take a jacket?")

    assert mouth.said == ["Take a light jacket.", "Anything else?", sign_off()]
    assert [who for _, who, _ in memory.recent_turns()][-2:] == ["person", "kin"]


def test_a_quiet_no_to_anything_else_is_still_a_no(memory):
    """Oct 9, first evening on the Linux hub: "Anything else?" got a quiet "no" (three seconds of voice at a
    third of her usual level, no words found in it) and Kin ended without a word, as if nobody were there."""
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Take a light jacket.")),
        reply(tool_call("ask", question="Anything else?")),
        reply(content="Checked in with Jennie, she's good."),
    ], ears=[])
    agent.tools.ears.heard_voice = True  # what the ears report after a capture with no words in it

    agent.run("should I take a jacket?")

    assert mouth.said == ["Take a light jacket.", "Anything else?", sign_off()]
    assert [who for _, who, _ in memory.recent_turns()][-1] == "kin", "nothing is put in her mouth"


def test_a_voice_with_no_words_is_not_a_no_to_an_ordinary_question(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="How did you sleep?")),
        reply(content="No answer to the question."),
    ], ears=[])
    agent.tools.ears.heard_voice = True

    agent.run("[check-in]", speaker="system")

    assert mouth.said == ["How did you sleep?"], "an unclear answer to a real question is left to the planner"


def test_nothing_more_is_said_after_the_sign_off(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="It's 66 and clear. Do you need anything else?")),
        reply(tool_call("say", text="Alright, take care!")),  # a second goodbye of the planner's own
        reply(content="Gave the weather; she needed nothing else."),
    ], ears=["Now I'm good."])  # Whisper's "No" often comes out as "Now"

    agent.run("What's the weather?")

    assert mouth.said == ["It's 66 and clear. Do you need anything else?", sign_off()]
    tool_results = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert "You have signed off" in tool_results[0] and tool_results[1] == SIGNED_OFF


def test_a_no_to_a_question_that_is_not_an_offer_of_more_does_not_end_the_conversation(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="Did you take your pills?")),
        reply(tool_call("ask", question="Shall I remind you again in ten minutes?")),
        reply(content="Alright, I'll leave you to it."),
    ], ears=["No.", "No, I'll take them now."])

    agent.run("[Reminder #1 is due: 'pills'.]", speaker="system")

    assert mouth.said == ["Did you take your pills?", "Shall I remind you again in ten minutes?",
                          "Alright, I'll leave you to it."]


def test_a_no_with_a_request_in_it_carries_on(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="Do you need anything else?")),
        reply(tool_call("ask", question="It's 66 and clear. Anything else?")),
        reply(content="Gave the weather."),
    ], ears=["No, but what's the weather?", "No, thank you."])

    agent.run("Hello")

    assert mouth.said == ["Do you need anything else?", "It's 66 and clear. Anything else?", sign_off()]


def test_the_sign_off_wishes_the_rest_of_the_day():
    assert sign_off(9).endswith("Have a lovely day.")
    assert sign_off(15).endswith("Enjoy your afternoon.")
    assert sign_off(19).endswith("Have a lovely evening.")
    assert all(sign_off(hour).startswith("Okay, I'm here if you need me.") for hour in range(24))


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


def test_a_reply_is_answered_even_when_bookkeeping_came_after_it(memory):
    """Sep 29, 18:00: "did you drink your water?" - "Yes, I do." - Kin logged the outcome and finished
    without a word, because `reminder_outcome` counts as nobody waiting. She had just answered him."""
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("ask", question="Jennie, did you remember to drink your water today?")),
        reply(tool_call("reminder_outcome", reminder_id=1, outcome="confirmed")),
        reply(content="Lovely, well done. Do you need anything else?"),
        reply(content="Reminder handled, Jennie confirmed drinking water."),  # Oct 3: this met her "No, thank you"
    ], ears=["Yes, I do.", "No thanks."])
    memory.add_reminder("drink water", "18:00")
    agent.tools.current_reminder = 1

    agent.run("[Reminder #1 is due: 'drink water'.]", speaker="system")

    assert mouth.said == ["Jennie, did you remember to drink your water today?",
                          "Lovely, well done. Do you need anything else?", sign_off()]


def test_a_note_to_himself_after_his_own_words_is_still_not_spoken(memory):
    agent, mouth, _ = make_agent(memory, [
        reply(tool_call("say", text="Take a light jacket.")),
        reply(tool_call("remember", fact="She is going out.")),
        reply(content="Told her about the jacket and noted the outing."),
    ])

    agent.run("should I take a jacket?")

    assert mouth.said == ["Take a light jacket."], "he spoke last, so his closing note stays private"


@pytest.mark.parametrize("closing, spoken", [
    # Oct 2, after the water reminder: Kin answered her properly and then read out his own log line.
    ("Jennie drank water and said she does not need anything else.", ""),
    ("Jennie confirmed she had her water.", ""),
    ("The person declined the reminder.", ""),
    ("Jennie seemed tired this morning.", ""),
    # Oct 8, right after "No, I'm good": a subjectless log line, and her name followed by "she"
    ("Checked in with Jennie, she's good.", ""),
    ("Reminded her about the water; no reply.", ""),
    ("Set the reminder for 8 pm.", ""),
    # direct address and ordinary speech are untouched
    ("Jennie, have a lovely day.", "Jennie, have a lovely day."),
    ("Okay Jennie, I'm here if you need me.", "Okay Jennie, I'm here if you need me."),
    ("Lovely, well done.", "Lovely, well done."),
    ("Mila said she will call at six.", "Mila said she will call at six."),
    ("Jennie, your daughter says she will visit on Sunday.", "Jennie, your daughter says she will visit on Sunday."),
    ("Settled, then. I'm here if you need me.", "Settled, then. I'm here if you need me."),
])
def test_a_third_person_report_about_her_is_never_said_to_her(closing, spoken, monkeypatch):
    monkeypatch.setenv("KIN_PERSON_NAME", "Jennie")

    assert speakable(closing) == spoken


def test_the_planner_is_told_when_the_fact_list_is_complete(memory):
    """Oct 9: seven facts, all in the prompt, and he still ran `recall` after her hello, eight seconds of
    silence with the weather call. He is now told whether there is anything left to search."""
    for n in range(3):
        memory.remember(f"Fact number {n}.")
    agent, _, _ = make_agent(memory, [])
    assert "That is every fact you have saved; `recall` would find nothing more." in agent._context()

    for n in range(3, 20):
        memory.remember(f"Fact number {n}.")
    context = agent._context()
    assert "There are older facts than these; `recall` finds them." in context
    assert context.count("- #") == 15, "the list itself stays at fifteen"
