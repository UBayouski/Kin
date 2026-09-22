"""How Kin speaks and listens. Both ends are swappable, so the same agent runs with any of:

mouth  'vector'   Vector's own robotic voice (default; most charming, hardest to follow)
       'hub'      a natural offline voice (Piper) through the hub's speaker, e.g. a USB speakerphone
ears   'console'  typed replies (development)
       'hub'      the hub's microphone with offline speech recognition (Whisper)

Speech stays on the hub: neither Piper nor Whisper sends audio anywhere.
"""

import contextlib
import queue
import time
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np

from kin import config

LISTEN_RATE = 16000
_BLOCK_SECONDS = 0.1
_LEAD_IN_BLOCKS = 4  # 0.4 s kept from just before the voice crossed the threshold


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
    """Piper text-to-speech through the hub's speaker, with Vector moving as if the words were his.

    Pass the robot and he plays his own talking animation for as long as the audio lasts; without it
    (benchmarks, tests) the voice simply plays.
    """

    def __init__(self, robot=None):
        from piper import PiperVoice, SynthesisConfig

        self._robot = robot

        model = Path(config.piper_voice())
        if not model.exists():  # a voice named in .env is fetched the first time it is used (about 60 MB)
            from piper.download_voices import download_voice

            model.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading the voice {model.stem}...", flush=True)
            download_voice(model.stem, model.parent)
        self._voice = PiperVoice.load(model)
        self._config = SynthesisConfig(length_scale=config.speech_slowdown())
        list(self._voice.synthesize("Hello.", syn_config=self._config))  # the first synthesis is slow; spend it here

    def say(self, text: str) -> None:
        import sounddevice as sd

        started = time.perf_counter()
        chunks = list(self._voice.synthesize(text, syn_config=self._config))
        if not chunks:
            return
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        print(f"  (voice ready in {time.perf_counter() - started:.1f}s, "
              f"{len(audio) / chunks[0].sample_rate:.1f}s to say)", flush=True)
        with self._talking():
            sd.play(audio, samplerate=chunks[0].sample_rate, device=config.audio_output_device())
            sd.wait()

    def _talking(self):
        if self._robot is None:
            return contextlib.nullcontext()
        from kin.robot import animating

        return animating(self._robot, "KnowledgeGraphAnswer")


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

    _stream = None  # the open microphone, between `prepare` and the end of `listen`
    _blocks: "queue.Queue[np.ndarray] | None" = None

    def __init__(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(config.whisper_model(), device="cpu", compute_type="int8")
        self._noise_floor: float | None = None

    def prepare(self) -> None:
        """Open the microphone now, before Kin asks his question, so it is already running when he stops.

        Opening it takes 0.5 to 0.9 s here (measured), and that used to happen after the question: a person
        who answered promptly lost their first words. Live, "I slept well. Thank you." arrived as its tail
        only, which Whisper padded into "Thank you very much. Thank you." """
        if self._stream is None:
            import sounddevice as sd

            blocks: queue.Queue[np.ndarray] = queue.Queue()
            self._blocks = blocks
            self._stream = sd.InputStream(
                samplerate=LISTEN_RATE, channels=1, dtype="float32", device=config.audio_input_device(),
                blocksize=int(LISTEN_RATE * _BLOCK_SECONDS), callback=lambda data, *_: blocks.put(data[:, 0].copy()),
            )
            self._stream.start()

    def _release(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = self._blocks = None

    def _utterance(self, blocks, timeout: float, max_seconds: float = 25.0) -> np.ndarray | None:
        """One utterance from successive microphone blocks: waits for speech, stops after a pause. None if
        nobody spoke within `timeout`. Time is counted in audio received, so a backlog cannot distort it."""
        end_pause = config.end_pause()
        lead_in: deque[np.ndarray] = deque(maxlen=_LEAD_IN_BLOCKS)
        heard: list[np.ndarray] = []
        now = 0.0
        speech_started = last_voice = None
        for block in blocks:
            now += len(block) / LISTEN_RATE
            level = float(np.sqrt(np.mean(block**2)))
            if self._noise_floor is None:
                self._noise_floor = level
            threshold = max(0.008, self._noise_floor * 3.0)

            if level > threshold:
                last_voice = now
                if speech_started is None:
                    speech_started = now
                    heard.extend(lead_in)  # the soft start of a word ("I sl...") sits under the threshold
            elif speech_started is None:
                # Track the room's background level only while nobody is talking.
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * level
                lead_in.append(block)

            if speech_started is not None:
                heard.append(block)
                if now - last_voice > end_pause or now - speech_started > max_seconds:
                    break
            elif now > timeout:
                return None
        return np.concatenate(heard) if heard else None

    def _record(self, timeout: float) -> np.ndarray | None:
        self.prepare()  # already open when `ask` prepared it; opened here otherwise
        source = self._blocks

        def from_the_microphone():
            while True:
                yield source.get(timeout=5)  # a microphone that stops delivering must not hang the task

        try:
            return self._utterance(from_the_microphone(), timeout)
        except queue.Empty:
            return None

    def listen(self, timeout: float = 12.0) -> str | None:
        """Keeps listening until it hears words or the time is up.

        A sound is not a reply. After a pet Vector purrs and whirs, the microphone took that for the start
        of an answer, captured a few seconds of robot noise, found no words in it and gave up while the
        person was still drawing breath: four conversations in a row heard "no reply". A capture with no
        words in it now just means "not yet".
        """
        if self._blocks is not None:  # opened by `prepare`: what it has so far is Kin's own question
            while not self._blocks.empty():
                self._blocks.get_nowait()
        try:
            deadline = time.monotonic() + timeout
            while (remaining := deadline - time.monotonic()) > 0.5:
                started = time.perf_counter()
                audio = self._record(remaining)
                if audio is None:
                    return None  # silence to the end
                recorded = time.perf_counter()
                text = ""
                if len(audio) >= LISTEN_RATE * 0.3:
                    segments, _ = self._model.transcribe(audio, language="en", vad_filter=True, beam_size=3)
                    text = " ".join(s.text.strip() for s in segments).strip()
                print(f"  (listened {recorded - started:.1f}s, of which {config.end_pause():.1f}s waiting to be sure "
                      f"they had finished; transcribed in {time.perf_counter() - recorded:.1f}s"
                      f"{'' if text else '; no words in it, still listening'})", flush=True)
                if text:
                    return text
            return None
        finally:
            self._release()


# --- wiring ---------------------------------------------------------------------------------------------------------


def make_mouth(robot) -> Mouth:
    return HubMouth(robot) if config.mouth() == "hub" else VectorMouth(robot)


def make_ears() -> Ears:
    return HubEars() if config.ears() == "hub" else ConsoleEars()
