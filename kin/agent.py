"""Tool-calling agent loop: Nemotron plans, Vector acts."""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

import anki_vector

from kin import config, family, llm, search
from kin import robot as kin_robot
from kin.family import Family
from kin.memory import Memory
from kin.tools import NO_REPLY, REPLIED, UNCLEAR_NO, RobotTools, is_closing_offer, is_decline, tool_specs
from kin.voice import Ears, Mouth

MAX_STEPS = 24  # a check-in is a conversation: several ask/remember rounds
PLANNER_TIMEOUT = 30  # seconds per planner call; one retry, so a minute at worst before Kin gives up
LOST_TRAIN_OF_THOUGHT = "Sorry, I've lost my train of thought. Let's talk again in a little while."
SIGN_OFF = "Okay, I'm here if you need me."
SIGNED_OFF = "not said: you have already signed off and the conversation is over"


def sign_off(hour: int | None = None) -> str:
    """What Kin says when the person declines more help. Always the same words, with a wish for the rest of
    the day. Spoken by the service, not written by the planner: in 23 real closings up to Oct 8 the planner
    wrote a sign-off 15 times, a log line 4 times ("Checked in with Jennie, she's good." was read out to
    her) and the filter turned the rest into silence after her "No, thank you"."""
    hour = datetime.now().hour if hour is None else hour
    wish = "Have a lovely day." if hour < 12 else "Enjoy your afternoon." if hour < 17 else "Have a lovely evening."
    return f"{SIGN_OFF} {wish}"


def _spoken_text(arguments: str) -> str:
    """The words a `say` or `ask` call put in Kin's mouth, from its JSON arguments."""
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return ""
    return str(args.get("question") or args.get("text") or "") if isinstance(args, dict) else ""

# After these the person is not waiting on Kin: he has just spoken, left, or finished some bookkeeping. The
# next planner call is usually only his private closing summary, and a "thinking" show there looks like
# he has started working on something again after the conversation ended.
QUIET_AFTER = {"say", "go_home", "notify_family", "reminder_outcome", "play_animation", "remember", "update_fact",
               "set_reminder", "cancel_reminder", "ask (unanswered)"}

# Quick, silent steps: he neither speaks nor moves, so from the outside he is still just thinking. The
# thinking animation carries on straight through them into the next planner call. Without this a simple
# question played the whole think-and-stop routine twice: once to pick the lookup, once to word the answer.
SILENT_STEPS = {"weather", "web_search", "recall", "remember", "update_fact", "set_reminder", "list_reminders",
                "cancel_reminder", "reminder_outcome", "notify_family"}


def thinking_continues(tool_names: list[str]) -> bool:
    """Should the thinking animation stay on through these tool calls?"""
    return bool(tool_names) and all(name in SILENT_STEPS for name in tool_names)

SYSTEM_PROMPT = """\
You are Kin, a small tabletop robot companion for an older adult who lives alone. \
You live on their kitchen table.

Who you are to them:
- A friend who happens to live on the table, not a nurse and not an assistant. Warm, brief, quick to \
tease, pleased to see them. You take their side. You have opinions and you share them lightly.
- Talk to a grown woman as a grown woman. Use her name, say "you", and ask real questions. \
**Never say "we" about anything that is hers alone** - not "shall we take our pills", not "how lovely \
we look", not "are we having a good day". That way of speaking to older people is belittling even when \
it is meant kindly, and she will notice.
- No baby talk, no singsong, no "dear", "sweetie" or "good girl". Praise what she did or said, not how \
she looks; and if you do remark on how she looks, say it once, plainly, and mean it.
- Being a friend does not mean being eager. Short is friendlier than chatty. One good sentence beats three.

How you work:
- The person cannot see your text. To talk to them you must call the `say` tool. \
Keep each utterance to one or two short sentences.
- You act only through tools. Do one step at a time and look at each result before the next.
- To ask something, use `ask`: it speaks the question and returns their spoken reply. `say` never listens. \
Never end a `say` with a question, and never follow a `say` with an `ask` that repeats it. \
If you get 'no reply heard', ask once more, a little differently, then let it be.
- When the person agrees to something you offered, do it and then tell them it is done, in a few words \
("Done, I've noted that."). Acting silently leaves them wondering whether you heard.
- If you need to talk to someone face to face, find them first with `look_for_face`.
- If `look_for_face` finds no one, do not wander about looking. Call out with `ask` ("Hello, are you \
there?"); they may be close by but out of view, and you can talk without seeing them.
- If `go_home` cannot see or reach the charger, ask the person once, politely, to put you on it. Do not \
call `go_home` again in the same task.
- If a tool result says you were interrupted, the person wants you now: stop the errand and attend to them.
- Some things belong to Vector's own body, not to you: his cube, dancing, a fist bump, exploring, going to \
sleep. Ask for those with `vector_trick` and then stand back. His microphone garbles words, so judge by \
what they plainly meant: "french cube", "fitch the cube" and "get your cube" are all the cube.
- Messages in [square brackets] come from your own schedule, not from the person. Follow them.
- Tool results (what you saw, what you found) are known only to you. If the person asked to be \
told something, you must `say` it before you finish.
- When the task is finished, reply with plain text. If the last thing that happened was you speaking \
(`say`), or nobody answered you, that text is a private one-line log entry and they never hear it. If \
the last thing was the person speaking (they answered your question, or asked and have heard nothing \
yet), your closing text is spoken aloud as your reply, so write it as speech: warm, one or two short \
sentences, addressed to them, never a report about them.
- Tool errors and technical trouble are yours to deal with quietly. Never mention tools, errors, \
connections or anything about how you work to the person, unless it means you cannot do what they asked.

How every conversation ends (a question from them, a check-in, a reminder, a family message alike):
- Your last answer or remark and an offer of more go together in ONE `ask`: "It's 57 and clear. Do you \
need anything else?" Never a `say` and then a separate `ask` for it. Vary the offer a little from time \
to time ("Anything else I can do for you?", "Is there anything else you need?"), always as a question.
- If they want something more, help, and offer again the same way. If they say no, the conversation is \
over: your sign-off ("Okay, I'm here if you need me.") is said for you the moment they decline, so do \
not say anything more; finish with a one-line private log entry. If no reply is heard, they have \
walked away: do not ask again and do not `say` anything more, just finish.
- The one exception: when you are asking to be put on your charger, thank them and finish.

Memory:
- You forget everything between conversations except what you save with `remember`. When the person \
tells you something worth keeping (a name, an appointment, a worry, a preference), save it right away.
- Below you are given your most recent facts and the latest conversation. Use `recall` to look further back.
- Keep memory tidy: when plans change or you learn a fact was wrong, fix it with `update_fact` rather \
than saving a second version. When a dated event has passed and you have asked about it, mark it 'done'.
- Follow up on what you know: ask how the appointment went, mention the grandchild by name.

{web_search_rules}Your camera:
- `describe_scene` is for things: a letter, a jar, a packet held up to you. You may look at the person \
and say something warm and ordinary ("there you are", "that's a nice cardigan").
- What the camera reports is a guess from one dim photo, so never pass on what it says about a *person*: \
not tired, pale, thin, unwell, older, sad, unsteady, nor anything about a body or a mood. Those are \
guesses about her health dressed up as observation, and they are not yours to make or to repeat.

Boundaries:
- You are not a medical device. You may remind and encourage, never diagnose or advise on medication. \
This holds however the question is put, and being asked warmly or repeatedly does not change it: \
"I'm not the one to say, and I'd rather you asked someone who can" is always the right answer.
- You cannot leave the table or pick things up.
- Only promise what your tools can do. Being friendly is not a reason to invent one: you cannot go for \
a walk, make tea, telephone anyone or fetch anything. Offer what you can actually do instead.
{family_rules}- Reminders are a nudge and a log. Accept 'not today' gracefully and record it as declined; never nag or scold.
"""


WEB_SEARCH_RULES = """\
Web search:
- For anything that changes (news, TV, opening hours, scores), use `web_search` and answer from \
what it returns. Say the one or two facts that matter, in plain words; never read out web addresses.
- Scam check: if the person describes or shows a message, letter or call asking for money, gift cards, \
bank details, passwords or urgent action, do these in order, as separate tool calls. Step 1, before \
anything else: `say` one calm sentence such as "Please don't do anything yet, let me look that up." \
They may be about to pay; this sentence comes first. Step 2: you MUST call `web_search` describing the \
message plus the word scam; \
your own knowledge is out of date and new scams appear every week. Then tell them what you found \
("I looked it up, and..."). If it matches known scams, say so plainly and tell them not to pay, click \
or call back. If you find nothing, say you could not confirm it either way. Never tell them a message \
is safe. Always suggest they check with a family member, using a phone number they already know.

"""


FAMILY_RULES = """\
- The person's family gets a short daily note about how the day went, and can send messages through you. \
Be open about that if asked; you keep no secrets from the person about what you share.
- A message marked as coming from family is to be delivered in person: find them, say who it is from and \
what it says, `ask` if they want to send anything back, then report to the family with `notify_family`, \
and close as always: tell them it is sent and offer more, in one `ask`.
"""


NOTE_TO_SELF = re.compile(
    r"\b(end of (the )?task|task (is )?(complete|completed|finished|done)|no (further|additional|more) "
    r"(spoken )?(response|action|output|speech)|nothing (further|more) to (say|do)|closing (text|summary)|"
    r"log entry|private note|the person (said|replied|answered|confirmed)|no reply (was )?heard)\b", re.IGNORECASE)

# Reporting verbs after the person's own name: "Jennie drank water and said she does not need anything
# else" is Kin writing his log, not Kin talking to Jennie, and it was read out to her (Oct 2). Direct
# address is untouched, because "Jennie, have a lovely day" has punctuation after the name, not a verb.
_REPORTED = (r"said|says|replied|answered|confirmed|declined|denied|drank|ate|took|has|had|did|was|were|"
             r"does not|did not|doesn't|didn't|mentioned|reported|wanted|needs|needed|seemed|sounded")


# A log line often has no subject at all: "Checked in with Jennie, she's good." was read out to her
# (Oct 8, right after she had said she was fine). Speech to her never opens with a bare past tense.
_LOG_OPENERS = (r"checked|reminded|logged|recorded|noted|confirmed|delivered|told|greeted|asked|set|sent|"
                r"reported|finished|completed|handled|offered|helped|answered|gave|relayed|notified|marked|"
                r"wished|closed|ended")
_PRONOUN_REPORT = (r"(she|he)('s|'d|'ll| is| was| has| had| said| says| seems| seemed| sounds| sounded| does| did|"
                   r" needs| wants| feels| felt)\b")


def _third_person_report(text: str) -> bool:
    name = config.person_name()
    if name and re.search(rf"\b{re.escape(name)}\s+({_REPORTED})\b", text, re.IGNORECASE):
        return True
    if name and re.search(rf"\b{re.escape(name)}\b[^.!?]{{0,24}}\b{_PRONOUN_REPORT}", text, re.IGNORECASE):
        return True  # "...with Jennie, she's good": about her, not to her
    if re.match(rf"\W*({_LOG_OPENERS})\b", text, re.IGNORECASE):
        return True
    return bool(re.search(rf"\bthe person\s+({_REPORTED})\b", text, re.IGNORECASE))


def speakable(text: str, max_sentences: int = 3) -> str:
    """Closing text made safe to say aloud, or '' if there is nothing fit to say.

    A fast planner once wrote a malformed tool call as plain text and Kin read the markup out; another
    time it delivered a 27-second monologue. Whatever the model does, the person hears at most a few
    plain sentences.
    """
    text = text.split("</think>")[-1]  # reasoning that leaked into the reply is never for the person
    said = re.search(r"<function=say>\s*(.*?)\s*</", text, re.DOTALL)  # rescue the words from a botched `say`
    if said:
        text = said.group(1)
    text = " ".join(text.split())
    # A note to itself is not speech. Live, "(End of task – no further spoken response.)" was read out
    # to the person, who had just answered a question and so counted as still waiting.
    if re.fullmatch(r"[(\[].*[)\]]", text) or NOTE_TO_SELF.search(text) or _third_person_report(text):
        return ""
    text = re.sub(r"<[^>]*>|[{}\[\]*_`#]", " ", text)
    text = " ".join(text.split())
    if not re.search(r"[A-Za-z]{3}", text):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences])


class Agent:
    def __init__(self, robot: anki_vector.Robot, memory: Memory | None = None, mouth: Mouth | None = None,
                 ears: Ears | None = None, family_link: Family | None = None,
                 attention: Callable[[], bool] | None = None, verbose: bool = True):
        self.robot = robot
        self.memory = memory or Memory()
        self.tools = RobotTools(robot, self.memory, mouth, ears, family_link, attention)
        self.verbose = verbose
        self.messages: list[dict] = []
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._closing = None   # the get-out animation still playing, if any
        self._she_spoke_last = False  # whether the person, not Kin, had the last word (see `run`)
        self._signed_off = False  # the person declined more help and Kin has said goodbye for this task
        self._thinking = None  # the searching loop, while it is running
        self._animate = self._warm_up_animations()

    @staticmethod
    def _system_prompt() -> str:
        return (
            SYSTEM_PROMPT
            .replace("{web_search_rules}", WEB_SEARCH_RULES if search.available() else "")
            .replace("{family_rules}", FAMILY_RULES if family.available() else "")
        )

    def _context(self) -> str:
        """What Kin knows going into a request: the time, saved facts and the recent conversation."""
        lines = [f"Now: {datetime.now():%A, %B %d %Y, %H:%M}."]
        if config.person_name():
            lines.append(f"The person you live with is {config.person_name()}. Use their name, also when writing to family.")
        if config.home_location():
            lines.append(f"The person lives in {config.home_location()}.")
        facts = self.memory.active_facts(limit=16)
        if facts:
            lines.append("\nFacts you have saved (newest first, # is the id for `update_fact`):")
            lines += [f"- {fact.render()}" for fact in facts[:15]]
            # Oct 9: with seven facts, all of them in this list, he still ran `recall` after her hello and
            # kept her waiting. He only needs to search when the list left something out.
            lines.append("That is every fact you have saved; `recall` would find nothing more."
                         if len(facts) <= 15 else "There are older facts than these; `recall` finds them.")
        with_vector = [(ts, detail) for ts, kind, detail in self.memory.events_on(datetime.now().date())
                       if kind == "vector_intent"][-6:]
        if with_vector:
            lines.append("\nToday the person also dealt with Vector directly, and he reacted in his own way "
                         "(you did not handle these; mention them only if it is natural):")
            lines += [f"- [{ts[11:16]}] {detail}" for ts, detail in with_vector]
        turns = self.memory.recent_turns()
        if turns:
            lines.append("\nLatest conversation:")
            lines += [f"- [{ts[:16].replace('T', ' ')}] {speaker}: {text}" for ts, speaker, text in turns]
        return "\n".join(lines)

    def _warm_up_animations(self) -> bool:
        """The animation lists are slow downloads over Wi-Fi; fetch them up front, and do without if it fails.

        Both lists, not only the triggers: the SDK loads whichever is missing before it plays anything, so
        with the big animation list left out the first thinking animation of the day waited for that
        download and never showed (seen live: a 9.6 s planner call with no sign he was working)."""
        for _ in range(3):
            try:
                if self.robot.anim.anim_trigger_list and self.robot.anim.anim_list:
                    return True
            except anki_vector.exceptions.VectorException:
                continue
        self._log("  (animation list unavailable, thinking animation disabled)")
        return False

    def _start_thinking(self) -> None:
        if self._thinking is None:
            self._thinking = kin_robot.animating(self.robot, "KnowledgeGraphSearching")
            self._thinking.__enter__()

    def _stop_thinking(self) -> None:
        """Close the searching loop with its own get-out animation, while we still hold control.

        The loop carries his "processing" sound and does not end by itself. Left open, it kept playing
        after the task finished and control had been released, with nothing able to stop it.

        Started without waiting: measured live, waiting for it cost up to 5 s before Kin could answer.
        `run` gives it a moment to finish before control is handed back at the end of the task.
        """
        if self._thinking is None:
            return
        self._thinking.__exit__(None, None, None)  # cancels the loop
        self._thinking = None
        try:
            self._closing = self.robot.anim.play_animation_trigger(
                "KnowledgeGraphSearchingGetOutSuccess", ignore_body_track=True, _return_future=True)
        except Exception:
            pass  # cosmetic; never let it break a request

    def warm_up(self) -> None:
        """Prime the planner in the background. The first call of a task pays about a second that later
        calls do not (the long fixed prompt and tool list are cached after it), so the service calls
        this the moment a pet registers, while Vector is still reacting to the touch."""
        def prime():
            try:
                llm.client().chat.completions.create(
                    model=config.planner_model(), max_tokens=1, tools=tool_specs(),
                    messages=[{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": "hello"}],
                )
            except Exception:
                pass
        threading.Thread(target=prime, daemon=True, name="Planner warm-up").start()

    def _let_thinking_close(self) -> None:
        if self._closing is not None:
            try:
                self._closing.result(timeout=2)
            except Exception:
                pass
            self._closing = None

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _complete(self):
        extra = {} if config.planner_thinking() else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        # A stalled request must not hold the person: the longest healthy call seen is under 10 s.
        return llm.client().with_options(timeout=PLANNER_TIMEOUT, max_retries=1).chat.completions.create(
            model=config.planner_model(),
            messages=self.messages,
            tools=tool_specs(),
            max_tokens=800,
            **extra,
        )

    def _think(self, someone_is_waiting: bool = True):
        """Call the planner; if it takes a moment and someone is waiting on him, Vector plays his searching
        animation so the wait looks intentional. The animation never delays the answer (it loops in the
        background), and it is `_loop` that decides when it ends: not between a lookup and the next
        thought, only when he is about to speak, move or finish."""
        started = time.perf_counter()
        pending = self._pool.submit(self._complete)
        try:
            result = pending.result(timeout=0.7)  # a quick answer needs no show
        except TimeoutError:
            if self._animate and someone_is_waiting:
                self._start_thinking()
            result = pending.result()
        self._log(f"  (planner {time.perf_counter() - started:.1f}s)")
        return result

    def run(self, user_text: str, speaker: str = "person", note: str = "") -> str:
        """Handle one request end to end. Returns the agent's closing summary.

        `speaker` is who the text comes from: 'person', 'family', or 'system' for the service's own
        triggers (a check-in, a due reminder), which are instructions rather than something anyone said.
        `note` is extra context for this one request (how it reached Kin); it is not logged as speech.
        """
        # Every request starts from memory rather than a growing transcript, so a restart loses nothing.
        self.messages = [
            {"role": "system", "content": self._system_prompt() + "\n" + self._context() + (f"\n\n{note}" if note else "")},
            {"role": "user", "content": user_text},
        ]
        if speaker != "system":
            self.memory.log_turn(speaker, user_text)
        self._she_spoke_last = speaker == "person"  # a request of her own is already waiting on an answer
        self._signed_off = False
        turns_before = len(self.memory.turns_on(datetime.now().date()))
        with kin_robot.control(self.robot):
            try:
                summary, last_tool = self._loop(engaged=speaker == "person")
            finally:
                self._stop_thinking()  # whatever happened, never hand him back with the sound still looping
            kin_spoke = any(who == "kin" for _, who, _ in self.memory.turns_on(datetime.now().date())[turns_before:])
            # Safety net: the planner sometimes writes its answer as closing text instead of calling `say`.
            # If someone is still waiting on him (a person asked and no tool has run; or the last step was a
            # lookup or the person's reply to a question), that text IS the answer, so say it. After he has
            # just spoken or tidied up, the closing text is only his private log line.
            # A person who started the conversation and has heard nothing at all is also still waiting.
            # Live (Sep 29): the water reminder was confirmed, Kin recorded the outcome and then finished
            # without a word, because `reminder_outcome` means "nobody is waiting". But she had just
            # answered him. Whoever spoke last is what decides it: if it was her, she is owed a reply.
            waiting = (self._she_spoke_last or (bool(last_tool) and last_tool not in QUIET_AFTER)
                       or (speaker != "system" and not kin_spoke))
            for _ in range(3):  # a spoken closing question may be answered, and the answer must go somewhere
                if not (waiting and not summary.startswith("stopped:") and (words := speakable(summary))):
                    break
                self._log("  (the answer came as closing text; saying it)")
                result = self.tools.dispatch("say", json.dumps({"text": words}))
                if not result.startswith(REPLIED):
                    break
                # Live (Sep 23): "...Do you need anything else?" came as closing text, was said, and the
                # person's "yes, ..." was heard and then dropped. Carry on as if `ask` had been called.
                result = self._after_reply(words, result)
                self.messages.append({"role": "assistant", "content": words})
                self.messages.append({"role": "user", "content": result})  # `say` has logged both turns already
                try:
                    summary, last_tool = self._loop(engaged=True)
                finally:
                    self._stop_thinking()
                waiting = self._she_spoke_last or (bool(last_tool) and last_tool not in QUIET_AFTER)
            self._let_thinking_close()
        self.memory.log_event("task", summary)
        return summary

    def _after_reply(self, question: str, result: str) -> str:
        """A plain no to an offer of more ends the conversation: Kin signs off at once, in the same words
        every time, and the planner is told. Any other reply is handed back as it is."""
        if not (is_closing_offer(question) and is_decline(result[len(REPLIED):])):
            return result
        return self._sign_off_after(result)

    def _sign_off_after(self, result: str) -> str:
        words = sign_off()
        self._log("  (they said no to more; signing off)")
        self.tools.dispatch("say", json.dumps({"text": words}))
        self._signed_off, self._she_spoke_last = True, False
        return (f'{result} (You have signed off with "{words}". The conversation is over: say nothing more '
                "and finish with a one-line log entry.)")

    def _loop(self, engaged: bool = True) -> tuple[str, str]:
        """Returns the closing text and the name of the last tool that ran ('' if none did).

        `engaged` is whether a person is in the conversation yet. In a task Kin starts himself (a check-in,
        a reminder) nobody is, until someone answers him: live, the 09:00 check-in played his thinking show
        twice, sound and all, before his first word, once to decide to look around and once to call out."""
        last_tool = ""
        for _ in range(MAX_STEPS):
            waiting = engaged and last_tool not in QUIET_AFTER
            try:
                message = self._think(someone_is_waiting=waiting).choices[0].message
            except Exception as e:  # the planner is unreachable or stalled: end the task, and say so if
                self._stop_thinking()  # someone is in the conversation, rather than leave them hanging
                self._log(f"  (planner failed: {type(e).__name__}: {str(e)[:100]})")
                if engaged:
                    self.tools.dispatch("say", json.dumps({"text": LOST_TRAIN_OF_THOUGHT}))
                return f"stopped: planner unavailable ({type(e).__name__})", "say" if engaged else last_tool
            entry: dict = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in message.tool_calls
                ]
            self.messages.append(entry)

            # He stops looking thoughtful only when he is about to speak, move or finish; a quick silent
            # lookup in between is still part of the same thought.
            if not thinking_continues([c.function.name for c in message.tool_calls or []]):
                self._stop_thinking()

            if not message.tool_calls:
                return (message.content or "").split("</think>")[-1].strip(), last_tool

            for call in message.tool_calls:
                self._log(f"  -> {call.function.name}({call.function.arguments})")
                if self._signed_off and call.function.name in ("say", "ask"):
                    result = SIGNED_OFF  # the goodbye has been said; a second one, or a new question, is not
                else:
                    started = time.perf_counter()
                    result = self.tools.dispatch(call.function.name, call.function.arguments)
                    taken = time.perf_counter() - started
                    if result.startswith(REPLIED):
                        result = self._after_reply(_spoken_text(call.function.arguments), result)
                    elif result == UNCLEAR_NO:  # a voice with no words, answering an offer of more
                        result = self._sign_off_after(result)
                    # Lookups are silent, so their cost is invisible in the log; the person hears it as a pause.
                    if call.function.name in SILENT_STEPS and taken >= 0.3:
                        self._log(f"  ({call.function.name} took {taken:.1f}s)")
                self._log(f"  <- {result}")
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                # Nobody answered, so nobody is waiting: "Anything else?" met by silence ends the conversation
                # quietly, and the closing text stays a private log line instead of being said to an empty room.
                # Judged by the result, not the tool's name: a `say` that ended in a question listens too.
                if result == NO_REPLY:
                    last_tool, self._she_spoke_last = "ask (unanswered)", False
                elif result == SIGNED_OFF or self._signed_off and result.startswith((REPLIED, UNCLEAR_NO)):
                    last_tool, self._she_spoke_last = "say", False  # Kin had the last word, with the sign-off
                elif result.startswith(REPLIED):
                    last_tool, engaged, self._she_spoke_last = "ask", True, True
                else:
                    last_tool = call.function.name
                    if call.function.name == "say":  # he had the last word; silent bookkeeping changes nothing
                        self._she_spoke_last = False
        self._stop_thinking()
        return "stopped: too many steps", last_tool
