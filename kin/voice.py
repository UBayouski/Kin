"""How Kin speaks and listens. Both ends are swappable, so the same agent runs with any of:

mouth  'vector'   Vector's own robotic voice (default; most charming, hardest to follow)
       'hub'      a natural offline voice (Piper) through the hub's speaker, e.g. a USB speakerphone
ears   'console'  typed replies (development)
       'hub'      the hub's microphone with offline speech recognition (Whisper)

Speech stays on the hub: neither Piper nor Whisper sends audio anywhere.
"""

import queue
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from kin import config

LISTEN_RATE = 16000
_BLOCK_SECONDS = 0.1


class Mouth(Protocol):
    def say(self, text: str) -> None: ...


class Ears(Protocol):
    def listen(self, timeout: float = 12.0) -> str | None:
        """The person's next utterance as text, or None if they said nothing."""


# --- mouths ---------------------------------------------------------------------------------------------------------


class VectorMouth:
    def __init__(self, robot):
        self.robot = robot

    def say(self, text: str) -> None:
        from kin.robot import call

        call(self.robot.behavior.say_text, text, duration_scalar=config.speech_slowdown(), timeout=60)


class HubMouth:
    """Piper text-to-speech through the hub's default speaker."""

    def __init__(self):
        from piper import PiperVoice, SynthesisConfig

        model = Path(config.piper_voice())
        if not model.exists():
            raise RuntimeError(
                f"Piper voice not found at {model}. Download one with:\n"
                f"  python -m piper.download_voices {model.stem} --download-dir {model.parent}"
            )
        self._voice = PiperVoice.load(model)
        self._config = SynthesisConfig(length_scale=config.speech_slowdown())
        list(self._voice.synthesize("Hello.", syn_config=self._config))  # the first synthesis is slow; spend it here

    def say(self, text: str) -> None:
        import sounddevice as sd

        chunks = list(self._voice.synthesize(text, syn_config=self._config))
        if not chunks:
            return
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        sd.play(audio, samplerate=chunks[0].sample_rate, device=config.audio_output_device())
        sd.wait()


# --- ears -----------------------------------------------------------------------------------------------------------


class ConsoleEars:
    def listen(self, timeout: float = 12.0) -> str | None:
        try:
            return input("   (reply) > ").strip() or None
        except EOFError:
            return None


class HubEars:
    """The hub's microphone plus Whisper. Records one utterance: waits for speech, stops after a pause.

    The pause is generous (1.8 s) because older speakers pause mid-sentence; a shorter one clipped replies in testing.
    """

    def __init__(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(config.whisper_model(), device="cpu", compute_type="int8")
        self._noise_floor: float | None = None

    def _record(self, timeout: float, end_pause: float = 1.8, max_seconds: float = 25.0) -> np.ndarray | None:
        import sounddevice as sd

        blocks: queue.Queue[np.ndarray] = queue.Queue()
        stream = sd.InputStream(
            samplerate=LISTEN_RATE, channels=1, dtype="float32", device=config.audio_input_device(),
            blocksize=int(LISTEN_RATE * _BLOCK_SECONDS), callback=lambda data, *_: blocks.put(data[:, 0].copy()),
        )
        heard: list[np.ndarray] = []
        started = speech_started = last_voice = None
        with stream:
            started = time.monotonic()
            while True:
                block = blocks.get()
                now = time.monotonic()
                level = float(np.sqrt(np.mean(block**2)))
                if self._noise_floor is None:
                    self._noise_floor = level
                threshold = max(0.008, self._noise_floor * 3.0)

                if level > threshold:
                    last_voice = now
                    speech_started = speech_started or now
                elif speech_started is None:
                    # Track the room's background level only while nobody is talking.
                    self._noise_floor = 0.95 * self._noise_floor + 0.05 * level

                if speech_started is not None:
                    heard.append(block)
                    if now - last_voice > end_pause or now - speech_started > max_seconds:
                        break
                elif now - started > timeout:
                    return None
        return np.concatenate(heard)

    def listen(self, timeout: float = 12.0) -> str | None:
        audio = self._record(timeout)
        if audio is None or len(audio) < LISTEN_RATE * 0.3:
            return None
        segments, _ = self._model.transcribe(audio, language="en", vad_filter=True, beam_size=3)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text or None


# --- wiring ---------------------------------------------------------------------------------------------------------


def make_mouth(robot) -> Mouth:
    return HubMouth() if config.mouth() == "hub" else VectorMouth(robot)


def make_ears() -> Ears:
    return HubEars() if config.ears() == "hub" else ConsoleEars()
