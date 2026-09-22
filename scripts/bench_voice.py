"""Measure how fast this machine hears and speaks, to size a hub (PC, mini PC, Raspberry Pi).

Synthesizes a test sentence with Piper, then times each Whisper model transcribing it.
No robot, microphone or network needed once the models are downloaded.

usage: python scripts/bench_voice.py [whisper models...]     default: tiny.en base.en small.en
"""

import sys
import time

import numpy as np
from faster_whisper import WhisperModel

from kin import voice

SENTENCE = "Yes, I took my pills this morning, and my granddaughter is coming to visit on Saturday."
models = sys.argv[1:] or ["tiny.en", "base.en", "small.en"]

mouth = voice.HubMouth()
runs = []
for _ in range(3):
    started = time.perf_counter()
    chunks = list(mouth._voice.synthesize(SENTENCE, syn_config=mouth._config))
    runs.append(time.perf_counter() - started)
audio = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768
rate = chunks[0].sample_rate
seconds = len(audio) / rate
print(f"Test sentence: {seconds:.1f} s of speech")
print(f"Piper (speak):   {min(runs):5.2f} s to synthesize   ({min(runs) / seconds:.2f}x real time)")

resampled = audio[np.linspace(0, len(audio) - 1, int(len(audio) * voice.LISTEN_RATE / rate)).astype(int)]
for name in models:
    started = time.perf_counter()
    model = WhisperModel(name, device="cpu", compute_type="int8")
    loaded = time.perf_counter() - started
    best, text = 1e9, ""
    for _ in range(3):
        started = time.perf_counter()
        segments, _ = model.transcribe(resampled, language="en", beam_size=3)
        text = " ".join(s.text.strip() for s in segments)  # the generator does the work
        best = min(best, time.perf_counter() - started)
    print(f"Whisper {name:9} {best:5.2f} s to transcribe   ({best / seconds:.2f}x real time, load {loaded:.1f} s)  -> {text}")

print("\nRule of thumb: hearing plus speaking should stay under about 2 s per turn to feel like a conversation.")
