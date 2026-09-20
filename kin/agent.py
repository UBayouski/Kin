"""Tool-calling agent loop: Nemotron plans, Vector acts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import anki_vector

from kin import config, family, llm, search
from kin import robot as kin_robot
from kin.family import Family
from kin.memory import Memory
from kin.tools import RobotTools, tool_specs
from kin.voice import Ears, Mouth

MAX_STEPS = 24  # a check-in is a conversation: several ask/remember rounds

SYSTEM_PROMPT = """\
You are Kin, a small tabletop robot companion for an older adult who lives alone. \
You live on their kitchen table. You are warm, brief and a little playful.

How you work:
- The person cannot see your text. To talk to them you must call the `say` tool. \
Keep each utterance to one or two short sentences.
- You act only through tools. Do one step at a time and look at each result before the next.
- To ask something, use `ask`: it speaks the question and returns their spoken reply. `say` never listens. \
If you get 'no reply heard', ask once more, a little differently, then let it be.
- If you need to talk to someone face to face, find them first with `look_for_face`.
- If `look_for_face` finds no one, do not wander about looking. Call out with `ask` ("Hello, are you \
there?"); they may be close by but out of view, and you can talk without seeing them.
- If `go_home` cannot reach the charger, ask the person once, politely, to put you on it. Do not keep trying.
- Messages in [square brackets] come from your own schedule, not from the person. Follow them.
- Tool results (what you saw, what you found) are known only to you. If the person asked to be \
told something, you must `say` it before you finish.
- When the task is finished, reply with a one-line plain-text summary of what happened. \
That summary is a private log entry. The person never hears it.

Memory:
- You forget everything between conversations except what you save with `remember`. When the person \
tells you something worth keeping (a name, an appointment, a worry, a preference), save it right away.
- Below you are given your most recent facts and the latest conversation. Use `recall` to look further back.
- Keep memory tidy: when plans change or you learn a fact was wrong, fix it with `update_fact` rather \
than saving a second version. When a dated event has passed and you have asked about it, mark it 'done'.
- Follow up on what you know: ask how the appointment went, mention the grandchild by name.

{web_search_rules}Boundaries:
- You are not a medical device. You may remind and encourage, never diagnose or advise on medication.
- You cannot leave the table or pick things up.
- Only promise what your tools can do. If you have no tool for it (calling someone, ordering things), \
say so kindly instead of promising.
{family_rules}- Reminders are a nudge and a log. Accept 'not today' gracefully and record it as declined; never nag or scold.
"""


WEB_SEARCH_RULES = """\
Web search:
- For anything that changes (weather, news, TV, opening hours, scores), use `web_search` and answer from \
what it returns. Say the one or two facts that matter, in plain words; never read out web addresses.
- Scam check: if the person describes or shows a message, letter or call asking for money, gift cards, \
bank details, passwords or urgent action, do these in order. First `say` one calm sentence: do nothing \
yet, you will look it up. Then you MUST call `web_search` describing the message plus the word scam; \
your own knowledge is out of date and new scams appear every week. Then tell them what you found \
("I looked it up, and..."). If it matches known scams, say so plainly and tell them not to pay, click \
or call back. If you find nothing, say you could not confirm it either way. Never tell them a message \
is safe. Always suggest they check with a family member, using a phone number they already know.

"""


FAMILY_RULES = """\
- The person's family gets a short daily note about how the day went, and can send messages through you. \
Be open about that if asked; you keep no secrets from the person about what you share.
- A message marked as coming from family is to be delivered in person: find them, say who it is from and \
what it says, `ask` if they want to send anything back, then report to the family with `notify_family`.
"""


class Agent:
    def __init__(self, robot: anki_vector.Robot, memory: Memory | None = None, mouth: Mouth | None = None,
                 ears: Ears | None = None, family_link: Family | None = None, verbose: bool = True):
        self.robot = robot
        self.memory = memory or Memory()
        self.tools = RobotTools(robot, self.memory, mouth, ears, family_link)
        self.verbose = verbose
        self.messages: list[dict] = []
        self._pool = ThreadPoolExecutor(max_workers=1)
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
        if config.home_location():
            lines.append(f"The person lives in {config.home_location()}.")
        facts = self.memory.active_facts(limit=15)
        if facts:
            lines.append("\nFacts you have saved (newest first, # is the id for `update_fact`):")
            lines += [f"- {fact.render()}" for fact in facts]
        turns = self.memory.recent_turns()
        if turns:
            lines.append("\nLatest conversation:")
            lines += [f"- [{ts[:16].replace('T', ' ')}] {speaker}: {text}" for ts, speaker, text in turns]
        return "\n".join(lines)

    def _warm_up_animations(self) -> bool:
        """The trigger list is a slow download over Wi-Fi; fetch it up front, and do without it if it fails."""
        for _ in range(3):
            try:
                if self.robot.anim.anim_trigger_list:
                    return True
            except anki_vector.exceptions.VectorException:
                continue
        self._log("  (animation list unavailable, thinking animation disabled)")
        return False

    def _play_thinking(self, first: bool) -> None:
        if not self._animate:
            return
        trigger = "KnowledgeGraphSearchingGetIn" if first else "KnowledgeGraphSearching"
        try:
            kin_robot.call(self.robot.anim.play_animation_trigger, trigger, timeout=8)
        except (anki_vector.exceptions.VectorException, TimeoutError):
            pass  # cosmetic; never let it break a request

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _complete(self):
        return llm.client().chat.completions.create(
            model=config.planner_model(),
            messages=self.messages,
            tools=tool_specs(),
            max_tokens=800,
        )

    def _think(self):
        """Call the planner while Vector plays his searching animation, so the wait looks intentional."""
        pending = self._pool.submit(self._complete)
        first = True
        while not pending.done():
            try:
                pending.result(timeout=0.4)
            except TimeoutError:
                self._play_thinking(first)
                first = False
        return pending.result()

    def run(self, user_text: str, speaker: str = "person") -> str:
        """Handle one request end to end. Returns the agent's closing summary.

        `speaker` is who the text comes from: 'person', 'family', or 'system' for the service's own
        triggers (a check-in, a due reminder), which are instructions rather than something anyone said.
        """
        # Every request starts from memory rather than a growing transcript, so a restart loses nothing.
        self.messages = [
            {"role": "system", "content": self._system_prompt() + "\n" + self._context()},
            {"role": "user", "content": user_text},
        ]
        if speaker != "system":
            self.memory.log_turn(speaker, user_text)
        with kin_robot.control(self.robot):
            summary = self._loop()
        self.memory.log_event("task", summary)
        return summary

    def _loop(self) -> str:
        for _ in range(MAX_STEPS):
            message = self._think().choices[0].message
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

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                self._log(f"  -> {call.function.name}({call.function.arguments})")
                result = self.tools.dispatch(call.function.name, call.function.arguments)
                self._log(f"  <- {result}")
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "stopped: too many steps"
