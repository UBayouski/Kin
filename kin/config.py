import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def nebius_api_key() -> str:
    return _required("NEBIUS_API_KEY")


def nebius_base_url() -> str:
    return os.getenv("NEBIUS_BASE_URL", "").strip() or "https://api.tokenfactory.nebius.com/v1/"


def chat_model() -> str:
    return _required("KIN_CHAT_MODEL")


def planner_model() -> str:
    return _required("KIN_PLANNER_MODEL")


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


def piper_voice() -> str:
    return os.getenv("KIN_PIPER_VOICE", "").strip() or "data/voices/en_US-lessac-medium.onnx"


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


def checkin_time() -> str:
    """HH:MM of the daily check-in."""
    return os.getenv("KIN_CHECKIN_TIME", "").strip() or "09:00"


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


def tavily_api_key() -> str:
    """Optional. Without it Kin has no web search tool."""
    return os.getenv("TAVILY_API_KEY", "").strip()


def home_location() -> str:
    """Optional town or city, so 'what's the weather' and local questions can be searched."""
    return os.getenv("KIN_LOCATION", "").strip()


def db_path() -> str:
    return os.getenv("KIN_DB_PATH", "").strip() or "data/kin.db"


def vector_serial() -> str | None:
    return os.getenv("VECTOR_SERIAL", "").strip() or None
