# Ideation

**Kin** (as in family) — an open-source companion brain for an older adult's kitchen table, running on a Vector 2.0 robot.

Hackathon: [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com) — **Physical AI track**.

## Hackathon constraints

| Item | Requirement |
|---|---|
| Deadline | Oct 30, 2026, 10:00 AM PT |
| Platform | Must run on Nebius Token Factory or Nebius AI Cloud |
| Models | At least one NVIDIA open model (Nemotron) |
| Video | 3 min max, public YouTube, **at least 1 min of real hardware operating** |
| Repo | Public, MIT / Apache-2.0 / MPL-2.0, README with setup instructions |
| Demo URL | Not required for Physical AI |
| Judging | Four equal criteria: technological implementation, design, potential impact, quality of idea |
| Other | Judges may request physical access to hardware; written feedback on Nebius/NVIDIA tools is part of the submission |

Prizes worth designing for: overall top 3 ($20k / $10k / $6k, cross-track), Physical AI track winner (Jetson Orin Nano), **Best Use of Tavily ($3k)**, Most Valuable Feedback (10 x $100).

Half the score is "is this a complete product for a real audience", so the project is framed as a product, not a tech demo.

## Ideas considered

1. **Vector as a tool-using embodied agent.** LLM calls SDK actions as tools. Strong on tech, but it is the obvious entry for any team with a robot, and weak on impact. **Kept as the architecture, not the product.**
2. **Desk focus coach.** Pomodoro plus presence/phone detection. Easy to demo authentically, but a crowded category with modest impact.
3. **Companion for an older relative.** Strongest impact story. ElliQ validates the category commercially and is also tabletop-only, which removes the main objection to Vector. **Chosen.**
4. **"Find my stuff" patrol.** Vector's camera sits about 3 cm above the surface and he has no SLAM, so this is unreliable. Dropped.

## The product

**Users:** an older adult living alone, and the family member who worries about them.

**Why a robot and not a smart speaker:** phone reminders get ignored. A small robot that drives over, looks up at you and asks a question is hard to ignore. Every feature must depend on Vector sensing or moving.

### Core loops (the whole scope)

1. **Morning check-in.** Vector leaves the charger, looks for a face, greets the person by name and has a short conversation with memory ("How did the doctor visit go yesterday?").
2. **Embodied reminders** (medication, water, a walk). Vector drives toward the person, gets their attention and asks. Confirmation is by voice; optionally the person holds the pill organizer up to the camera and a VLM checks it. Gentle re-nudge if no response. Everything is logged.
3. **Presence awareness.** No face seen by a configured time, or no response to a check-in, sends a message to the family. Framed as "didn't see Mom this morning", never as fall detection.
4. **Family bridge (Telegram bot).**
   - Outbound: daily digest with mood, confirmed reminders, topics discussed, flags.
   - Inbound: a relative sends "Tell Grandma I'll call at 6"; Vector drives over and delivers it.
5. **Live knowledge through Tavily web search.** A language model only knows its training data; asked about today's weather or a suspicious text message it would guess. Tavily is a search API built for agents: it returns clean, sourced passages the model can read. Two uses, chosen so that search is essential rather than decorative (this is also our entry for the Best Use of Tavily prize):
   - **Scam check.** Older adults are the main target of phone, text and mail fraud, and new scams appear weekly, so only live search can catch them. The person reads out or holds up a suspicious message; Kin searches for matching scam reports and answers plainly ("This matches a known scam. Do not pay."), suggests checking with family, and can alert them. Kin may say a message matches a scam or that he could not confirm it; he never says a message is safe, because a false "safe" is the dangerous error.
   - **Grounded daily briefing and everyday questions.** Weather for the walk, local news, "what's on TV tonight", opening hours, personalised from memory (gardening tips, their team's score). Part of the morning check-in.

### Optional

- **Cube as a "yes" button.** Tapping Vector's cube confirms a reminder ("tap the cube when you have taken your pills"); its lights glow while a reminder is pending. A big tactile button is easier for an older person than a wake word, which has been unreliable in testing. Voice confirmation stays the primary path; this is added only if tap detection proves reliable.

### Out of scope

Cube tricks (rolling, lifting, fetching), free navigation, multi-room anything, medical claims of any kind.

## Nebius and NVIDIA mapping

| Component | Role |
|---|---|
| Nemotron Nano (Token Factory) | Live conversation, where latency matters |
| Nemotron Super / Ultra (Token Factory) | Tool-calling planner that decides when and how to act |
| VLM (Token Factory) | Presence, "who's here", pill-organizer check. NVIDIA VLM preferred; any Token Factory VLM is an acceptable fallback since Nemotron already satisfies the model requirement |
| Serverless Job (AI Cloud), nightly | Consolidates the day's memory, writes the family digest, finds week-over-week trends |
| Tavily | Live web search as an agent tool: scam checks, weather, news, local questions. Nemotron still does the reasoning; Tavily only supplies fresh facts |

Pitch line: **thin robot, cloud brain.** Privacy story: open models, camera frames processed and never stored, memory kept on the local machine.

## Risks

| Risk | Mitigation |
|---|---|
| Vector's speaker is quiet and robotic, mic is mediocre; bad for hearing loss | **Hub audio (built):** a natural offline voice (Piper) and offline speech recognition (Whisper) on the hub's speaker and microphone, ideally a USB conference speakerphone beside the charger. Vector's own voice stays available (`KIN_MOUTH=vector`), slowed to 1.2x |
| The wake word is unreliable (detected once in a day of trying) | Kin does not depend on it: he starts most conversations himself and listens after each question (`ask`); the person starts one by **petting his back** |
| Built-in TTS is mostly English | External TTS played through the SDK's `stream_wav_file`; wire-pod STT is multilingual |
| SDK does not expose the mic stream on production bots | Voice input via wire-pod's "Hey Vector" pipeline with a custom intent that forwards the transcript to the agent; laptop mic as fallback |
| No SLAM, drifting odometry, cliff sensors misbehave on dark surfaces | Relative moves only, charger as landmark, light bordered tabletop |
| SDK behavior control freezes Vector's personality | Release control between tasks, script idle and "thinking" animations |
| LLM/VLM latency of a few seconds | Thinking animation; stream a Nano reply while the planner works |
| Battery lasts 30-40 min off charger | Charger is home base; film in short sessions, record footage as we go |
| Pairing with wire-pod requires clearing user data and likely drops the DDL cloud link | Confirm before wiping |
| Overclaiming | Reminders are "a nudge and a log", not adherence monitoring |

## Open questions

- Is there an older relative available to film with? Real footage is far stronger than acting.
- ~~Language?~~ en-US; built-in TTS and wire-pod STT suffice.
- ~~Which models are served on Token Factory?~~ Settled Sep 18, see `docs/PLAN.md`. No NVIDIA VLM is served, so perception uses Gemma 3.
- ~~Does `wirepod_vector_sdk` install on Python 3.13?~~ Yes: 0.8.1 installs and imports (Sep 18). Not yet tested against the robot.
- Is the region eligible under the hackathon rules?
