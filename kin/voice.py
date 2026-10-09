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
_FLOOR_CAP = 0.006   # the room's background level is never taken to be more than this (speech is 0.03 to 0.1 here)
_DEAD_LEVEL = 0.0002  # quieter than any real room: the input device is giving us nothing (see `seems_deaf`)


def device_id(setting: int | str | None, kind: str) -> int | None:
    """Turn a `KIN_AUDIO_*` setting into one device index.

    A name fragment is matched here rather than left to sounddevice, which refuses a fragment that hits
    the same hardware on several host APIs ("Multiple input devices found for 'QuadCast'": MME,
    DirectSound, WASAPI, WDM-KS all list it). The same thing happens on Linux with ALSA and PipeWire
    entries. Pinning a name matters: on Oct 3 the Windows default input had become a webcam that delivers
    silence, and the 09:00 check-in called out twice and heard nothing.
    """
    if setting is None or isinstance(setting, int):
        return setting
    import sounddevice as sd

    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    wanted = setting.strip().lower()
    matches = [i for i, d in enumerate(sd.query_devices()) if d[channels] > 0 and wanted in d["name"].lower()]
    if not matches:
        raise RuntimeError(f"no {kind} device matching {setting!r}; `python scripts/voices.py devices` lists them")
    default_api = sd.default.hostapi
    on_default = [i for i in matches if sd.query_devices(i)["hostapi"] == default_api]
    return (on_default or matches)[0]


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


def with_tail(audio: np.ndarray, sample_rate: int, seconds: float = 0.3) -> np.ndarray:
    """The audio followed by a little silence. Piper ends exactly on the last sound and the output device
    closed on it, so the last letters of every sentence were clipped (Oct 8, on the hub speaker)."""
    return np.concatenate([audio, np.zeros(int(sample_rate * seconds), dtype=audio.dtype)])


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
        audio = with_tail(np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate)
        print(f"  (voice ready in {time.perf_counter() - started:.1f}s, "
              f"{len(audio) / chunks[0].sample_rate:.1f}s to say)", flush=True)
        with self._talking():
            sd.play(audio, samplerate=chunks[0].sample_rate, device=device_id(config.audio_output_device(), "output"))
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
        self._loudest = 1.0        # the loudest level of the last listen that heard nothing
        self._dead_listens = 0     # how many in a row were not merely quiet but silent (see `seems_deaf`)

    def seems_deaf(self) -> bool:
        """Is the microphone giving us nothing at all?

        A real room has a floor: 0.0003 to 0.006 here. A dead or wrong input device reads near zero, and
        on Oct 3 it did, because the Windows default input had become a webcam. Kin then called out twice,
        "heard" silence, and was one presence deadline away from telling the family nobody was home - a
        claim about a person's wellbeing drawn from his own broken ear. Three in a row, so one quiet
        listen in a soundproof room is not enough.
        """
        return self._dead_listens >= 3

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
                samplerate=LISTEN_RATE, channels=1, dtype="float32", device=device_id(config.audio_input_device(), "input"),
                blocksize=int(LISTEN_RATE * _BLOCK_SECONDS), callback=lambda data, *_: blocks.put(data[:, 0].copy()),
            )
            self._stream.start()

    def _skip_speaker_tail(self, quiet_level: float = 0.02, quiet_blocks: int = 3, max_seconds: float = 1.0) -> float:
        """Let the end of Kin's own words pass before listening for the person's.

        The speaker finishes a little after `say` returns (the output device's latency, then the room),
        and that tail was landing in every capture: the transcripts began with Kin's last words ("Are you
        there? Yes.") and the "voice" was heard from the first block. Returns the seconds skipped."""
        skipped, quiet = 0.0, 0
        while skipped < max_seconds - 1e-6 and quiet < quiet_blocks:
            try:
                block = self._blocks.get(timeout=1)
            except queue.Empty:
                break
            skipped += len(block) / LISTEN_RATE
            quiet = quiet + 1 if float(np.sqrt(np.mean(block**2))) < quiet_level else 0
        return skipped

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
        now = peak = 0.0
        speech_started = last_voice = None
        for block in blocks:
            now += len(block) / LISTEN_RATE
            level = float(np.sqrt(np.mean(block**2)))
            peak = max(peak, level)
            if self._noise_floor is None:
                self._noise_floor = min(level, _FLOOR_CAP)
            # Capped: a floor poisoned by an earlier noise put the threshold above the person's voice, and
            # a floor that sits above the voice then adapts towards it and never fires (Sep 23: a reply at
            # 0.076 went unheard against a printed threshold of 0.008, which had been far higher as she spoke).
            threshold = max(0.008, min(self._noise_floor, _FLOOR_CAP) * 3.0)

            if level > threshold:
                last_voice = now
                if speech_started is None:
                    speech_started = now
                    heard.extend(lead_in)  # the soft start of a word ("I sl...") sits under the threshold
            elif speech_started is None:
                # Track the room's background level only while nobody is talking.
                self._noise_floor = min(_FLOOR_CAP, 0.95 * self._noise_floor + 0.05 * level)
                lead_in.append(block)

            if speech_started is not None:
                heard.append(block)
                if now - last_voice > end_pause or now - speech_started > max_seconds:
                    # Live, a two-word reply took 12 s to come back: something kept the level up. These
                    # numbers say what: how long the "voice" lasted and against what threshold.
                    print(f"  (capture: voice from {speech_started:.1f}s for {last_voice - speech_started:.1f}s, "
                          f"threshold {threshold:.4f}, room {self._noise_floor:.4f}, loudest {peak:.4f})", flush=True)
                    break
            elif now > timeout:
                # Live, a "no, thank you" went unheard with nothing in the log to say why; these numbers do.
                print(f"  (heard nothing for {timeout:.0f}s: loudest {peak:.4f} against a threshold of "
                      f"{threshold:.4f}, room {self._noise_floor:.4f})", flush=True)
                self._loudest = peak
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
            self._skip_speaker_tail()
        self._noise_floor = None  # measured afresh each time; what the room was like an hour ago is no guide
        self._loudest = 1.0
        try:
            deadline = time.monotonic() + timeout
            while (remaining := deadline - time.monotonic()) > 0.5:
                started = time.perf_counter()
                audio = self._record(remaining)
                if audio is None:
                    self._dead_listens = self._dead_listens + 1 if self._loudest < _DEAD_LEVEL else 0
                    if self.seems_deaf():
                        print(f"  (the microphone is giving us nothing: loudest {self._loudest:.5f} over "
                              f"{self._dead_listens} listens; check KIN_AUDIO_INPUT)", flush=True)
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
                    self._dead_listens = 0
                    return text
            return None
        finally:
            self._release()


# --- wiring ---------------------------------------------------------------------------------------------------------


def make_mouth(robot) -> Mouth:
    return HubMouth(robot) if config.mouth() == "hub" else VectorMouth(robot)


def make_ears() -> Ears:
    return HubEars() if config.ears() == "hub" else ConsoleEars()
