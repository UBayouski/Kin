# Plan

Deadline: **Oct 30, 2026, 10:00 AM PT**. Started Sep 18, 2026. See [IDEATION.md](IDEATION.md) for the product.

## Architecture

```
  "Hey Vector ..."                         Telegram (family)
        |                                        ^  |
        v                                        |  v
  wire-pod (STT, custom intent) ---transcript--> agent (Python, local PC)
                                                 |   |        |
                         Vector SDK <--- tools --+   |        +--> memory (SQLite, local)
              (drive, say, face, camera, anims)      |
                                                     v
                                     Nebius Token Factory
                              Nano: chat | Super/Ultra: planner | VLM: perception
                                                     
  nightly: Nebius Serverless Job  <-- day log --  memory  --> digest --> Telegram
```

- **Deployment shape: home hub, Nebius brain.** Something must run on the home LAN, because the SDK and wire-pod talk to Vector locally; a cloud-only brain cannot reach him. The hub (the dev PC now; a Raspberry Pi or mini PC on Ethernet later) is a thin coordinator: wire-pod, robot connection, scheduler, memory. Every intelligent decision is a Nebius call (`kin/llm.py` is the only place a model is called), which is also the hackathon requirement. Running models locally on the hub is out of scope.
- **Headless service:** Kin runs as `python -m kin.service` with no one at a keyboard. Four triggers start the agent loop: the clock (check-ins, reminders), a periodic presence look, voice (wire-pod forwards the transcript), and family messages (Telegram, polled outbound so no router ports are opened). `scripts/console.py` stays as a debugging tool.
- **Packaging:** Docker Compose brings up wire-pod and Kin with one command, for the hub and for judges. On Linux wire-pod's mDNS works natively, so `scripts/escapepod_mdns.py` is only needed on Windows.
- **Agent loop:** OpenAI-compatible tool calling against Token Factory. Tools wrap SDK actions: `say`, `drive`, `turn`, `look_for_face`, `take_photo`, `describe_scene`, `play_animation`, `show_on_face`, `go_home`, plus `web_search` (Tavily), `remember` / `recall`, `notify_family`.
- **Scheduler:** triggers check-ins, reminders and the presence deadline.
- **Behavior control:** acquired per task and released afterwards so Vector keeps his personality.

## Milestones

### Week 1 (Sep 18 - Sep 25): prove the pipes
- [x] wire-pod installed on Windows ([installer](https://github.com/kercre123/WirePod/releases/latest))
- [x] Vector flashed with the DDL **"ep" firmware** via https://wpsetup.keriganc.com/ (now `2.0.1.6076ep`). Plain 2.0.x is not enough; the CCIS firmware string must end in `ep`
- [x] Authenticated against wire-pod (Sep 18). See "Robot setup notes" in the README for what went wrong
- [x] `wirepod_vector_sdk` working: drive, speak, capture a frame (`scripts/01_vector_hello.py`)
- [x] Token Factory key; models picked (Sep 18, measured on a first tool-call turn):
  - chat: `nvidia/Nemotron-3_5-Lightning` (0.3 s)
  - planner: `nvidia/nemotron-3-super-120b-a12b` (0.9 s); `nvidia/Nemotron-3-Ultra-550b-a55b` (0.8 s) for the nightly job
  - VLM: `google/gemma-3-27b-it` (1.2 s); `openbmb/MiniCPM-V-4_5` (0.4 s) is the fast alternative. No Nemotron on Token Factory accepts images
- [x] One VLM call on an image
- [x] End to end: Vector photographs, VLM describes, Vector says it (`scripts/03_vector_sees.py`, 2.1 s VLM round trip)
- [x] Language: **en-US**. Built-in TTS and wire-pod STT are enough; no external TTS needed

### Week 2 (Sep 26 - Oct 2): agent core
- [x] Tool wrapper over the SDK with safe relative moves (`kin/tools.py`; clamped moves, retry on control loss)
- [x] Tool-calling agent loop, thinking animation during calls (`kin/agent.py`, try it with `scripts/console.py`)
- [x] Headless service (`kin/service.py`, Sep 19): `python -m kin.service` runs all day. Triggers: daily check-in time, due reminders with re-nudges (10 min apart, gives up after 3 and logs an alert), petting Vector's back to start a conversation, nightly memory tidy-up. Control is borrowed per task so Vector keeps his personality in between; reconnects after a dropped link. `--checkin-now` and `--once` for testing and demos
- [x] Conversation: the `ask` tool speaks a question and returns the spoken reply, so one task can be a multi-turn dialogue
- [x] Hub audio (`kin/voice.py`): swappable mouth (`vector` | `hub` = Piper, offline) and ears (`console` | `hub` = microphone + Whisper `small.en`, offline). Tested live Sep 19 on the dev PC's USB mic and speakers: a full spoken check-in worked. A USB speakerphone is a drop-in (`KIN_AUDIO_INPUT` / `KIN_AUDIO_OUTPUT`). This replaces wire-pod voice forwarding as the main voice path
- [x] Reminder tools: `set_reminder`, `list_reminders`, `cancel_reminder`, `reminder_outcome`
- [ ] Test the reminder loop live end to end (set one by voice, let it fire, confirm, decline, ignore)
- [ ] Optional: voice in through wire-pod ("Hey Vector" / back button) for setups without a hub microphone
- [x] Local memory (`kin/memory.py`, SQLite at `data/kin.db`): facts, conversation, events; `remember` / `recall` tools; every request starts from memory, so restarts lose nothing. Reminder tables come with the reminder loop
- [x] Memory, second pass (Sep 19): facts have a status (`active` / `done` / `retired`) and an optional date; `update_fact` tool; semantic recall with `Qwen/Qwen3-Embedding-8B` on Token Factory plus keyword matches, falling back to keywords alone when offline; `reminders` and `reminder_log` tables with due logic (the tools come with the reminder loop)
- [x] Memory consolidation (`kin/consolidate.py`, `scripts/consolidate.py [--dry-run]`): Nemotron Ultra merges duplicates, applies changes from the day's conversation, closes past events with a follow-up to ask about, and lifts missed facts. Runs locally now; the nightly Serverless Job will call the same function
- [x] Vector's voice slowed to 1.2x (`KIN_SPEECH_SLOWDOWN`) for intelligibility
- [x] Timeouts on every SDK call (`kin_robot.call`) and a disconnect that cannot hang; Wi-Fi stalls used to freeze the process

### Weeks 3-4 (Oct 3 - Oct 16): product loops
- [x] Daily check-in with memory (first live run Sep 19)
- [ ] Embodied reminders with voice confirmation, re-nudge, log (built; needs the live test above)
- [ ] Presence deadline and family alert
- [x] Family side over Telegram (`kin/family.py`, `kin/digest.py`, Sep 19). Long polling from the hub, so no router ports are opened; only chats in `KIN_FAMILY_CHAT_IDS` are accepted, strangers are told their chat id and ignored. Inbound: a family message is delivered in person and the reply reported back. Outbound: `notify_family` tool (offered only when configured; Kin tells the person what he sends), an alert when a reminder gets no response after three tries, and a daily note at `KIN_DIGEST_TIME` written by Nemotron Ultra from that day's log only. Commands: `/status` (no model call), `/digest`, `/help`. Digest and wiring tested; **the Telegram connection itself is untested until a bot token exists**
- [ ] Create the bot (@BotFather), set `TELEGRAM_BOT_TOKEN` and `KIN_FAMILY_CHAT_IDS`, then test live: relay a message, `/status`, `/digest`, a missed-reminder alert
- [ ] Nightly Serverless Job on Nebius AI Cloud: move `consolidate()` and `digest.build()` (both already written and running locally) into a job, and add week-over-week trends (missed reminders, mood)
- [x] Tavily `web_search` tool (`kin/search.py`, Sep 19). Offered to the model only when `TAVILY_API_KEY` is set, so Kin never promises a search he cannot do; `KIN_LOCATION` gives him the person's town. Scam-check rules are in the system prompt. Tested with a real key Sep 19: searches return in about 3 s
- [x] Tested on the robot: "should I take a jacket?" (searched the weather for `KIN_LOCATION`, advised a light jacket) and a gift-card bail scam text (searched, found current reports, told the person not to pay and to call family). The scam rule had to be made a MUST: on the first try he answered correctly from his own knowledge without searching
- [ ] Still to test: a news question, and a scam letter held up to the camera (`describe_scene` then `web_search`). Tune: he skips the opening "do nothing yet, I will look it up" sentence
- [ ] Work the briefing (weather + one personalised item) into the morning check-in
- [ ] Optional purchase: **USB conference speakerphone for the hub** (Jabra Speak / Anker PowerConf class, $40-100). The software side is done and device-agnostic. It matters for the real deployment: a far-field mic array hears a quiet voice across a table, and one puck replaces a desk mic plus speakers. Decide after testing with a real older listener
- [ ] Have Vector animate (talking / listening poses) while the hub speaks and listens, so the voice still feels like his
- [ ] Optional: **cube as a "yes" button.** The cube reports taps over the SDK, so "tap the cube when you have taken your pills" could confirm a reminder without the wake word, and its lights could glow while a reminder is pending. Only after the core loops work, and only if a quick tap-detection test proves reliable (the cube's Bluetooth link is known to be flaky)
- [ ] Stretch: NVIDIA VLM for an all-NVIDIA stack. The Token Factory catalog offers **Nemotron-Nano-V2-12b (Vision)** and **Nemotron-3-Nano-Omni** as **Dedicated** endpoints only, not on the public API (checked Sep 19). A dedicated endpoint bills GPU time while it is up, so bring it up for filming and judging; Gemma 3 stays as the everyday fallback

### Week 5 (Oct 17 - Oct 23): polish
- [ ] Personality, latency, idle animations, paired-speaker audio
- [ ] Failure handling: off charger, cliff, lost face, network down
- [ ] Docker Compose for wire-pod + Kin; move to a dedicated hub (Pi / mini PC) if one is available
- [ ] Record footage with the real user

### Week 6 (Oct 24 - Oct 29): ship
- [ ] 3 min video (at least 1 min of hardware), uploaded public on YouTube
- [ ] README with setup steps, architecture diagram, license
- [ ] Devpost text and the Nebius / NVIDIA feedback write-up (keep `docs/FEEDBACK_NOTES.md` as we go)
- [ ] Submit by Oct 29, a day early

## Demo video outline (3 min)

1. 0:00 - 0:25: the family member states the problem.
2. 0:25 - 1:55: robot footage: morning greeting, medication reminder with Vector driving over, relayed family message.
3. 1:55 - 2:25: the digest arriving on the relative's phone; a Tavily-backed question.
4. 2:25 - 2:50: architecture slide naming Token Factory, Nemotron, Serverless Jobs.
5. 2:50 - 3:00: open source, repo link.
