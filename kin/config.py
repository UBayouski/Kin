import os

from dotenv import dotenv_values, find_dotenv, load_dotenv

# .env is looked for from the working directory upwards (the project root for every launch path),
# not from wherever this file happens to be installed.
load_dotenv(find_dotenv(usecwd=True))


def overridden() -> dict[str, tuple[str, str]]:
    """Settings where the process environment disagrees with .env, as {name: (env value, .env value)}.

    The environment wins, silently; the service prints these at startup so nobody is left wondering
    why Kin is not doing what .env says.
    """
    secret = ("KEY", "TOKEN")
    found = {}
    for name, in_file in dotenv_values(find_dotenv(usecwd=True)).items():
        in_env = os.environ.get(name)
        if in_env is not None and in_env.strip() != (in_file or "").strip():
            hide = any(s in name for s in secret)
            found[name] = ("***" if hide else in_env, "***" if hide else (in_file or ""))
    return found


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def nebius_api_key() -> str:
    return _required("NEBIUS_API_KEY")


def nebius_base_url() -> str:
    return os.getenv("NEBIUS_BASE_URL", "").strip() or "https://api.tokenfactory.nebius.com/v1/"


def planner_model() -> str:
    """The live model: every decision and reply in the agent loop. Speed matters most here."""
    return _required("KIN_PLANNER_MODEL")


def planner_thinking() -> bool:
    """Whether the planner may 'think' before answering (Nemotron's reasoning mode). Off is faster."""
    return _choice("KIN_PLANNER_THINKING", ("on", "off")) == "on"


def vlm_model() -> str:
    return _required("KIN_VLM_MODEL")


def nightly_model() -> str:
    """The big model for offline work: memory consolidation and the family digest."""
    return os.getenv("KIN_NIGHTLY_MODEL", "").strip() or planner_model()


def embed_model() -> str:
    return os.getenv("KIN_EMBED_MODEL", "").strip() or "Qwen/Qwen3-Embedding-8B"


def speech_slowdown() -> float:
    """Stretch factor for Vector's voice; above 1.0 is slower and easier to follow."""
    return float(os.getenv("KIN_SPEECH_SLOWDOWN", "").strip() or 1.2)


def _choice(name: str, options: tuple[str, ...]) -> str:
    value = os.getenv(name, "").strip().lower() or options[0]
    if value not in options:
        raise RuntimeError(f"{name} must be one of {options}, not {value!r}")
    return value


def mouth() -> str:
    """'vector' speaks through the robot; 'hub' through the hub's speaker with a natural voice."""
    return _choice("KIN_MOUTH", ("vector", "hub"))


def ears() -> str:
    """'console' takes typed replies; 'hub' listens on the hub's microphone."""
    return _choice("KIN_EARS", ("console", "hub"))


# Short names for the hub voices we have tried. Any other Piper voice works by its full name.
VOICES = {
    # female
    "lessac": ("en_US-lessac-medium", "female, US, clear and friendly (the original)"),
    "amy": ("en_US-amy-medium", "female, US, softer"),
    "kristin": ("en_US-kristin-medium", "female, US"),
    "hfc_female": ("en_US-hfc_female-medium", "female, US, clear and neutral"),
    "cori": ("en_GB-cori-high", "female, British, high quality; slower to synthesize"),
    "jenny": ("en_GB-jenny_dioco-medium", "female, British"),
    "alba": ("en_GB-alba-medium", "female, Scottish"),
    # male
    "ryan": ("en_US-ryan-high", "male, US, richest and most natural; about 4x slower to synthesize"),
    "ryan_fast": ("en_US-ryan-medium", "male, US, the same voice as ryan at normal speed; best pick for a small hub"),
    "joe": ("en_US-joe-medium", "male, US, warm, a little deeper"),
    "hfc_male": ("en_US-hfc_male-medium", "male, US, clear and neutral"),
    "john": ("en_US-john-medium", "male, US"),
    "bryce": ("en_US-bryce-medium", "male, US"),
    "norman": ("en_US-norman-medium", "male, US, older-sounding"),
    "kusal": ("en_US-kusal-medium", "male, US"),
    "alan": ("en_GB-alan-medium", "male, British"),
    "northern": ("en_GB-northern_english_male-medium", "male, Northern English"),
}
VOICE_DIR = "data/voices"


def voice_name() -> str:
    """KIN_VOICE as a full Piper voice name. Accepts a short name from VOICES or any Piper name."""
    value = os.getenv("KIN_VOICE", "").strip() or "lessac"
    return VOICES.get(value.lower(), (value, ""))[0]


def piper_voice() -> str:
    """Path of the voice model file."""
    return f"{VOICE_DIR}/{voice_name()}.onnx"


def end_pause() -> float:
    """Seconds of silence that mean the person has finished speaking. Lower is snappier; too low cuts off
    people who pause mid-sentence (1.2 s clipped replies in testing)."""
    return float(os.getenv("KIN_END_PAUSE", "").strip() or 1.5)


def whisper_model() -> str:
    return os.getenv("KIN_WHISPER_MODEL", "").strip() or "small.en"


def _device(name: str) -> int | str | None:
    """A sounddevice device: index, name fragment, or None for the system default."""
    value = os.getenv(name, "").strip()
    return int(value) if value.isdigit() else value or None


def audio_input_device() -> int | str | None:
    return _device("KIN_AUDIO_INPUT")


def audio_output_device() -> int | str | None:
    return _device("KIN_AUDIO_OUTPUT")


def stay_on_charger() -> bool:
    """Whether Kin keeps Vector on his charger. On by default: docking is his least reliable move, and a
    robot that leaves and cannot get back runs flat and goes dark. Off lets him drive to the person."""
    return _choice("KIN_STAY_ON_CHARGER", ("on", "off")) == "on"


def idle_hold() -> bool:
    """Whether Kin keeps a light hold on Vector between tasks (the SDK's RESERVE_CONTROL) so his idle
    behaviours do not drive him off the charger to explore. He wandered off twice on Sep 21, ran flat the
    second time and went dark. With the hold he sat on the dock for as long as it was held (Sep 22) and
    "Hey Vector" still worked; the cost is that he does less by himself. Off for demos where he should roam."""
    return _choice("KIN_IDLE_HOLD", ("on", "off")) == "on"


def pet_to_talk() -> bool:
    """Whether petting Vector's back starts a conversation. Off by default: a pet is affection, and his own
    purr is the right answer to it; a robot that replies "what can I do for you?" to a stroke is confusing,
    and his back button sits beside the touch sensor. A pet still counts as a sign the person is around."""
    return _choice("KIN_PET_TO_TALK", ("off", "on")) == "on"


# Chosen before Kin knows what was said, so each must fit a question as well as a request: nothing that
# promises an action ("Okay, will do." in answer to "should I take a jacket?" was confusing, live).
ACK_PHRASES = (
    "Let me think.", "One moment.", "Hmm, let me see.", "Let me check.", "Give me a second.",
    "Hmm, one moment.", "Let me think about that.", "Just a moment.",
)


def ack_phrases() -> tuple[str, ...]:
    """What Vector says, in his own voice, when he has heard a request and handed it to Kin.
    KIN_ACK_PHRASES overrides the list: phrases separated by '|'."""
    custom = [p.strip() for p in os.getenv("KIN_ACK_PHRASES", "").split("|") if p.strip()]
    return tuple(custom) or ACK_PHRASES


def gateway_port() -> int:
    """Local port where wire-pod hands Kin the speech it did not recognise. 0 turns the gateway off."""
    return int(os.getenv("KIN_GATEWAY_PORT", "").strip() or 8765)


def wirepod_url() -> str:
    return (os.getenv("KIN_WIREPOD_URL", "").strip() or "http://localhost:2427").rstrip("/")


def checkin_time() -> str:
    """HH:MM of the daily check-in."""
    return os.getenv("KIN_CHECKIN_TIME", "").strip() or "09:00"


def renudge_minutes() -> int:
    """How long Kin waits before asking again about a reminder that got no response."""
    return int(os.getenv("KIN_RENUDGE_MINUTES", "").strip() or 10)


def reminder_attempts() -> int:
    """Tries per reminder per day before Kin gives up and tells the family."""
    return int(os.getenv("KIN_REMINDER_ATTEMPTS", "").strip() or 3)


def presence_start() -> str:
    """HH:MM from when Kin starts quietly looking for the person each day."""
    return os.getenv("KIN_PRESENCE_START", "").strip() or "07:30"


def presence_deadline() -> str:
    """HH:MM by which the person should have been seen or heard; after it the family is told. 'off' disables."""
    return os.getenv("KIN_PRESENCE_DEADLINE", "").strip() or "10:30"


def presence_look_minutes() -> int:
    return int(os.getenv("KIN_PRESENCE_LOOK_MINUTES", "").strip() or 30)


def telegram_token() -> str:
    """Optional. Without it Kin has no family side."""
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def family_chat_ids() -> set[int]:
    """Telegram chats allowed to talk to Kin. Anyone else is ignored."""
    raw = os.getenv("KIN_FAMILY_CHAT_IDS", "")
    return {int(part) for part in raw.replace(";", ",").split(",") if part.strip().lstrip("-").isdigit()}


def digest_time() -> str:
    """HH:MM when the daily note is sent to the family."""
    return os.getenv("KIN_DIGEST_TIME", "").strip() or "20:00"


def weekly_day() -> str:
    """Weekday ('mon'..'sun') on which the family gets the weekly note, after that day's digest. 'off' disables."""
    return os.getenv("KIN_WEEKLY_DAY", "").strip().lower() or "sun"


def tavily_api_key() -> str:
    """Optional. Without it Kin has no web search tool."""
    return os.getenv("TAVILY_API_KEY", "").strip()


def person_name() -> str:
    """Optional: what Kin calls the person, e.g. 'Anna' or 'Grandma Anna'."""
    return os.getenv("KIN_PERSON_NAME", "").strip()


def units() -> str:
    """'metric' or 'imperial', for anything Kin says with a unit."""
    return _choice("KIN_UNITS", ("metric", "imperial"))


def home_location() -> str:
    """Optional town or city, so 'what's the weather' and local questions can be searched."""
    return os.getenv("KIN_LOCATION", "").strip()


def db_path() -> str:
    return os.getenv("KIN_DB_PATH", "").strip() or "data/kin.db"


def vector_serial() -> str | None:
    return os.getenv("VECTOR_SERIAL", "").strip() or None


def nightly_url() -> str:
    """Where the night's thinking runs when not on the hub: the URL of `python -m kin.nightly --serve`
    in a Nebius serverless container. Empty means the hub does it."""
    return os.getenv("KIN_NIGHTLY_URL", "").strip()


def nightly_token() -> str:
    """The bearer token that service expects, if any."""
    return os.getenv("KIN_NIGHTLY_TOKEN", "").strip()
