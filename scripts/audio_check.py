"""Check the hub's speaker and microphone the way Kin uses them.

    python scripts/audio_check.py            # a tone, then a Piper sentence, with every rate decision printed
    python scripts/audio_check.py --listen   # also record three seconds and play them back

Reads KIN_AUDIO_INPUT / KIN_AUDIO_OUTPUT and KIN_VOICE from .env like the service does. Written on
Oct 9 when the first Linux hub played Piper at the wrong rate (a cartoon voice) through a Jabra that
takes only 16 and 48 kHz.
"""
import sys
import time

import numpy as np
import sounddevice as sd

from kin import config, voice

out = voice.device_id(config.audio_output_device(), "output")
print(f"output device: {out} {sd.query_devices(out)['name'] if out is not None else '(default)'}")
for rate in (16000, 22050, 44100, 48000):
    try:
        sd.check_output_settings(device=out, samplerate=rate)
        print(f"  {rate} Hz accepted")
    except Exception as e:
        print(f"  {rate} Hz rejected: {str(e)[:60]}")

rate = voice.playback_rate(sd, out, 22050)
print(f"tone at {rate} Hz for one second (440 Hz: a steady, middling pitch)")
t = np.arange(rate) / rate
sd.play((np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16), samplerate=rate, device=out)
sd.wait()

from piper import PiperVoice  # noqa: E402  (slow import; after the tone so the tone is heard first)

piper = PiperVoice.load(config.piper_voice())
chunks = list(piper.synthesize("The quick brown fox jumps over the lazy dog."))
audio = np.concatenate([c.audio_int16_array for c in chunks])
src = chunks[0].sample_rate
rate = voice.playback_rate(sd, out, src)
played = voice.resample(audio, src, rate)
print(f"piper: {src} Hz, {len(audio)} samples, chunk shape {chunks[0].audio_int16_array.shape}, dtype {audio.dtype}")
print(f"playing at {rate} Hz, {len(played)} samples, {len(played) / rate:.1f}s")
started = time.perf_counter()
sd.play(played, samplerate=rate, device=out)
sd.wait()
print(f"took {time.perf_counter() - started:.1f}s (should match the length above)")

if "--listen" in sys.argv:
    inp = voice.device_id(config.audio_input_device(), "input")
    print(f"input device: {inp} {sd.query_devices(inp)['name'] if inp is not None else '(default)'}")
    print("recording 3 seconds at 16 kHz, say something...")
    started = time.perf_counter()
    rec = sd.rec(int(3 * voice.LISTEN_RATE), samplerate=voice.LISTEN_RATE, channels=1, dtype="float32", device=inp)
    sd.wait()
    print(f"took {time.perf_counter() - started:.1f}s (should be about 3.0; much less means the device captures at a higher rate than it admits)")
    level = float(np.abs(rec).max())
    print(f"loudest {level:.4f} (a reply at arm's length is about 0.05 or more; under 0.01 is a quiet room)")
    rate = voice.playback_rate(sd, out, voice.LISTEN_RATE)
    sd.play(voice.resample((rec[:, 0] * 32767).astype(np.int16), voice.LISTEN_RATE, rate), samplerate=rate, device=out)
    sd.wait()
