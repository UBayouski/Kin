# Kin

A companion for an older adult living alone, running on a Vector 2.0 robot with a cloud brain: NVIDIA Nemotron models on Nebius Token Factory.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com), Physical AI track.

- [Ideation](docs/IDEATION.md)
- [Plan](docs/PLAN.md)

Kin is not a medical device. Reminders are a nudge and a log.

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

## Smoke tests

```powershell
python scripts/00_list_models.py
python scripts/01_vector_hello.py
python scripts/02_describe_image.py path\to\image.jpg
python scripts/03_vector_sees.py
```

## Run Kin

```powershell
python -m kin.service                        # runs all day, no keyboard needed
python -m kin.service --checkin-now --once   # do one check-in now and exit
```

The service wakes on the clock (a daily check-in, reminders with gentle re-nudges, a nightly memory tidy-up) and when the person pets Vector's back. Between tasks Vector is left to be himself.

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

### Voice

By default Kin speaks with Vector's own voice and takes typed replies. For spoken conversation, install the voice extra and switch both ends to the hub (any microphone and speaker; a USB conference speakerphone is ideal):

```powershell
pip install -e ".[voice]"
python -m piper.download_voices en_US-lessac-medium --download-dir data/voices
# in .env:  KIN_MOUTH=hub   KIN_EARS=hub
```

Speech recognition (Whisper) and the voice (Piper) both run offline on the hub; no audio leaves the home.

## Talk to Kin from a console

```powershell
python scripts/console.py                       # interactive
python scripts/console.py "Find me and say hi"  # one request
```

Optional: put a [Tavily](https://tavily.com) key in `.env` as `TAVILY_API_KEY` (and your town as `KIN_LOCATION`) and Kin can search the live web: weather, news, and checking whether a suspicious message matches a known scam.

A Nemotron planner on Token Factory turns each request into robot actions, one tool call at a time (`kin/agent.py`, `kin/tools.py`). What Kin learns about the person is kept in a local SQLite file, `data/kin.db`, which never leaves the machine.

## License

MIT
