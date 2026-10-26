# Kin

A companion for an older adult living alone, running on a Vector 2.0 robot with a cloud brain: NVIDIA Nemotron models on Nebius Token Factory.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com), Physical AI track.

Kin is not a medical device. Reminders are a nudge and a log.

## What Kin does

A phone notification is easy to ignore. A small robot that drives over, looks up at you and asks a question is not. Kin is for two people: an older adult living alone, and the relative who worries about them.

- **Daily check-in.** At a set time he finds the person, greets them by name, asks how they are, and follows up on what he remembers ("You mentioned a dentist visit, how did it go?"). He tells them the day's weather.
- **Reminders that come to you.** Pills, water, a walk: set by voice ("remind me to drink water at six every day"). When one is due he goes looking for the person and asks. He tries again ten minutes later if there is no answer, accepts "not today" gracefully, and logs the outcome.
- **Presence.** Through the morning he quietly looks for the person from his charger. If there has been no sign of them by a deadline he calls out once; then the family is told, and told again when the person turns up.
- **Family bridge (Telegram).** A relative's message is delivered in person, and the spoken reply goes back. The family gets a short note each evening, a weekly note on patterns (reminders missed several days running, a worry mentioned again and again), and an alert when a reminder goes unanswered or the robot itself stops responding.
- **Scam check.** Older adults are the main target of phone, text and mail fraud. Describe a suspicious message and Kin says "please don't do anything yet", searches the live web for matching scam reports, and says plainly what he found. He will say a message matches a known scam or that he could not confirm it; he never says a message is safe.
- **He checks himself.** Every night a large model audits what Kin said against his own rules: medical advice, promises he cannot keep, implausible numbers, nagging, ignoring a question.

You talk to him the way you talk to any Vector: say "Hey Vector" or press the button on his back. He acknowledges in his own voice, Kin answers in a natural voice while he moves as if the words were his, then asks if there is anything else and listens through the hub's microphone. A pet is just a pet: he purrs, and it counts as a sign you are around (`KIN_PET_TO_TALK=on` makes it start a conversation too).

## How it works

```
                    Telegram (family)
                          ^  |
                          |  v
   microphone --> +-------------------+ --> speaker          Nebius Token Factory
   (Whisper)      |   Kin hub (PC)    |     (Piper)          - Nemotron 3 Super: every live decision
                  |  service, agent,  | <------------------> - Nemotron 3 Ultra: nightly digest, trends,
   touch, faces   |  memory (SQLite)  |                        memory tidy-up, self-check
   camera    ---> +-------------------+                      - vision model: what the camera sees
                     ^             |                         - embeddings: memory search by meaning
        Vector SDK   |             v   Vector SDK
                  +-------------------+                      Tavily: live web search (scam checks, news)
                  |   Vector 2.0      |                      Open-Meteo: weather, with explicit units
                  |   via wire-pod    |
                  +-------------------+
```

**Thin robot, cloud brain.** Vector senses and moves; the hub keeps the connection, the schedule and the memory; every decision is a call to a Nemotron model on Nebius Token Factory. The agent is a tool-calling loop: the model picks one action (`say`, `ask`, `look_for_face`, `remember`, `set_reminder`, `weather`, `web_search`, `notify_family`, `go_home`...), sees the result, and picks the next. Nobody programmed "if face detection fails, call out instead"; the model sees the failure and adapts.

**Private by design.** Speech recognition and the voice run offline on the hub, so no audio leaves the home. Camera frames are analysed and never stored. Memory is a local SQLite file. The family note is written from one day's log, not the whole history, and Kin tells the person what he shares.

**Built to be left alone.** Every robot call has a deadline and can be cut short by a pet; a dropped link is noticed within 30 seconds and reconnected; family messages are saved before they are acknowledged, so a restart cannot lose one; each alert is sent exactly once. 233 tests run without a robot, a network or API keys.

## Setup

Requirements: a Vector robot paired with [wire-pod](https://github.com/kercre123/wire-pod), Python 3.10+, a Nebius Token Factory API key.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .
pip install wirepod_vector_sdk
python -m anki_vector.configure   # registers the robot with the SDK
copy .env.example .env            # then fill it in
```

## Robot setup notes

Pairing a new Vector 2.0 with wire-pod, as done for this project:

1. Install wire-pod ([Windows installer](https://github.com/kercre123/WirePod/releases/latest)) and finish its initial setup page.
2. Charge Vector, then open https://wpsetup.keriganc.com/ in Chrome with Bluetooth on. Double-press his back button, pair, enter the PIN, join 2.4 GHz Wi-Fi and let it flash the `ep` firmware. A factory-fresh bot (OS 0.9.x) needs no recovery mode. The OS string must end in `ep` afterwards.
3. Clear user data (double-press, lift the arm, turn a wheel to select, lift to confirm), pair again and click ACTIVATE. wire-pod then writes `~/.anki_vector/sdk_config.ini` by itself.

If ACTIVATE fails with "Error logging in", Vector cannot reach wire-pod at `escapepod.local`:

- Check that the PC can ping Vector (his IP is on the info screen: double-press, lift the arm once). If not, fix the network first. On an ASUS AiMesh, broadcast and multicast between bands were dropped until the router and node firmware was updated and everything rebooted.
- On Windows, wire-pod's mDNS broadcaster may bind IPv6 only. Run `python scripts/escapepod_mdns.py` to publish `escapepod.local` over IPv4 (add `--unicast <vector-ip>` if the router drops multicast).
- If the PIN prompt never appears, see [wire-pod issue #531](https://github.com/kercre123/wire-pod/issues/531).

**Power the charger from a wall adapter (5 V, 2 A or more), not a computer's USB port.** A PC port supplies 0.5 to 0.9 A and may cut power to save energy; under charging load Vector's body board can reset, which shows up as reboots while docking and error 899 (the head has lost contact with the body).

## Smoke tests

```powershell
python scripts/00_list_models.py
python scripts/01_vector_hello.py
python scripts/02_describe_image.py path\to\image.jpg
python scripts/03_vector_sees.py
```

## Run Kin

```powershell
.\scripts\hub.ps1                            # Windows hub: wire-pod + mDNS helper + the Kin service, restarted if it exits, logged to data\logs
.\scripts\hub.ps1 -Install                   # ...and run all of that at logon as a scheduled task (-Status, -Uninstall)
python -m kin.service                        # the service alone; runs all day, no keyboard needed
python -m kin.service --checkin-now --once   # do one check-in now and exit
```

The service wakes on the clock (a daily check-in, reminders with gentle re-nudges, the morning presence looks, the evening family note, a nightly reflection), when a family message arrives, and when the person speaks to Vector ("Hey Vector" or his back button). Between tasks Vector is left to be himself.

### Running in the background (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-kin.ps1          # start hidden; logs in data\logs
powershell -ExecutionPolicy Bypass -File scripts\windows\start-kin.ps1 -Stop    # stop
powershell -ExecutionPolicy Bypass -File scripts\windows\install-autostart.ps1  # start at every login
```

Stop the background service before using `scripts/console.py`, so the two do not compete for the robot.

### Family (Telegram)

1. In Telegram, talk to **@BotFather**, send `/newbot`, and follow the prompts. Put the token it gives you in `.env` as `TELEGRAM_BOT_TOKEN`.
2. Start Kin, then send your new bot any message. It replies with your chat id. Put that in `.env` as `KIN_FAMILY_CHAT_IDS` (comma-separated for several relatives) and restart Kin.
3. Now any message you send is delivered by Kin in person, and he reports the reply. `/status` shows how today is going, `/digest` sends today's note now, and the note arrives by itself at `KIN_DIGEST_TIME`. You are also told when a reminder goes unanswered.

Only listed chats are accepted. The person is told about the family link if they ask, and Kin says what he is sending before he sends it.

### "Hey Vector" and the back button

Vector keeps his own character. His built-in commands ("I love you", "give me a fist bump", "go to sleep") are still handled by him, through wire-pod. Anything wire-pod does not recognise is passed to Kin: Vector answers "Let me think." (or another short phrase, `KIN_ACK_PHRASES`) in his own voice, and Kin then does the work.

```powershell
python scripts/wirepod_link.py --link     # with Kin's service running
python scripts/wirepod_link.py --unlink   # put wire-pod back as it was
```

Kin also notices what is said to Vector himself and how he reacts. Those moments count as signs that the person is around, and they reach the family's daily note.

### Voice

By default Kin speaks with Vector's own voice and takes typed replies. For spoken conversation, install the voice extra and switch both ends to the hub (any microphone and speaker; a USB conference speakerphone is ideal):

```powershell
pip install -e ".[voice]"
# in .env:  KIN_MOUTH=hub   KIN_EARS=hub   KIN_VOICE=lessac
python scripts/voices.py          # list the voices; add `play` to hear them
```

`KIN_VOICE` takes a short name (`lessac`, `amy`, `ryan`, `joe`, `hfc_male`, `alan`) or any Piper voice name; a voice is downloaded the first time it is used.

Speech recognition (Whisper) and the voice (Piper) both run offline on the hub; no audio leaves the home.

## Talk to Kin from a console

```powershell
python scripts/console.py                       # interactive
python scripts/console.py "Find me and say hi"  # one request
```

Optional: put a [Tavily](https://tavily.com) key in `.env` as `TAVILY_API_KEY` (and your town as `KIN_LOCATION`) and Kin can search the live web: weather, news, and checking whether a suspicious message matches a known scam.

A Nemotron planner on Token Factory turns each request into robot actions, one tool call at a time (`kin/agent.py`, `kin/tools.py`). What Kin learns about the person is kept in a local SQLite file, `data/kin.db`, which never leaves the machine.

## Tests

```powershell
pip install -e ".[dev]"
python -m pytest
```

233 tests covering memory, reminders, family alerts, the durable inbox, the agent loop, listening, tool dispatch, weather, the voice and the nightly reflection. They need no robot, network or API keys.

## License

MIT
