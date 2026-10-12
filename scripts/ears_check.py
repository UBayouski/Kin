"""Hear what Kin hears: record a few seconds from the configured microphone, transcribe them exactly the way
the service does, and print every Whisper segment with the scores the filter judges it by.

    python scripts/ears_check.py            # 4 seconds
    python scripts/ears_check.py 8          # longer

Written on Oct 11, after a filter change silently dropped every reply for a day: the service log only said
"no words in it", and nothing showed what Whisper had actually heard.
"""
import sys
import time

import numpy as np
import sounddevice as sd

from kin import config, voice

seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() else 4.0
device = voice.device_id(config.audio_input_device(), "input")
print(f"microphone: {device} {sd.query_devices(device)['name'] if device is not None else '(default)'}")
print(f"recording {seconds:.0f} seconds at {voice.LISTEN_RATE} Hz; say a short reply...")
audio = sd.rec(int(seconds * voice.LISTEN_RATE), samplerate=voice.LISTEN_RATE, channels=1, dtype="float32", device=device)
sd.wait()
audio = audio[:, 0]
print(f"loudest {float(np.abs(audio).max()):.4f}")

from faster_whisper import WhisperModel  # noqa: E402  (slow import, after the recording)

model = WhisperModel(config.whisper_model(), device="cpu", compute_type="int8")
started = time.perf_counter()
segments, _ = model.transcribe(audio, language="en", vad_filter=True, beam_size=3, condition_on_previous_text=False)
segments = list(segments)
print(f"transcribed in {time.perf_counter() - started:.1f}s; {len(segments)} segment(s)")
for s in segments:
    print(f"  {s.text.strip()!r:40}  no-speech {s.no_speech_prob:.2f}  log-prob {s.avg_logprob:.2f}  ({s.start:.1f}s to {s.end:.1f}s)")
print(f"kin hears: {voice.words_in(segments)!r}")
