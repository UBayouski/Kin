"""Choose Kin's hub voice.

    python scripts/voices.py                 list the voices and show which one .env selects
    python scripts/voices.py play            hear every voice, one after another
    python scripts/voices.py play joe ryan   hear just these (downloads a voice if it is missing)
    python scripts/voices.py play all        hear every distinct English voice Piper offers (about 30), numbered
    python scripts/voices.py catalog         every English voice Piper offers (38), beyond the short list
    python scripts/voices.py catalog de      the same for another language, e.g. de, fr, es, uk, pl

Then set the one you like in .env, for example  KIN_VOICE=joe , and restart Kin.
Any other Piper voice works too, by its full name: https://github.com/OHF-Voice/piper1-gpl
"""

import sys
import time
from pathlib import Path

from kin import config

SAMPLE = "Good morning {person}. It's 66 degrees and mostly clear. Did you take your morning pills?"


def full_name(name: str) -> str:
    return config.VOICES.get(name.lower(), (name, ""))[0]


def listing() -> None:
    current = config.voice_name()
    print(f"KIN_VOICE currently selects: {current}\n")
    for short, (name, description) in config.VOICES.items():
        installed = "installed" if Path(config.VOICE_DIR, f"{name}.onnx").exists() else "not downloaded"
        marker = "->" if name == current else "  "
        print(f"{marker} {short:11} {name:36} {installed:15} {description}")
    print("\nHear them:  python scripts/voices.py play     Choose one:  KIN_VOICE=<name> in .env")


def play(names: list[str]) -> None:
    import numpy as np
    import sounddevice as sd
    from piper import PiperVoice, SynthesisConfig
    from piper.download_voices import download_voice

    from kin.voice import device_id, with_tail

    synthesis = SynthesisConfig(length_scale=config.speech_slowdown())
    line = SAMPLE.format(person=config.person_name() or "there")
    numbered = names == ["all"]
    if numbered:
        names = distinct_voices()
        line = "Good morning. Did you take your pills?"  # thirty voices: keep each one short
    for number, short in enumerate(names, 1):
        name = full_name(short)
        model = Path(config.VOICE_DIR, f"{name}.onnx")
        if not model.exists():
            print(f"downloading {name}...", flush=True)
            model.parent.mkdir(parents=True, exist_ok=True)
            download_voice(name, model.parent)
        voice = PiperVoice.load(model)
        list(voice.synthesize("Hello.", syn_config=synthesis))  # warm up, so the timing below is fair
        started = time.perf_counter()
        spoken_name = name.split("-")[1].replace("_", " ") if numbered else short
        intro = f"Number {number}. {spoken_name}." if numbered else f"This is {spoken_name}."
        chunks = list(voice.synthesize(f"{intro} {line}", syn_config=synthesis))
        took = time.perf_counter() - started
        audio = with_tail(np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate)
        print(f"{number:>2}  {short:11} {name:36} synthesized in {took:.2f} s", flush=True)
        sd.play(audio, samplerate=chunks[0].sample_rate, device=device_id(config.audio_output_device(), "output"))
        sd.wait()
        time.sleep(0.6)


def piper_voices(language: str = "en") -> list[str]:
    import contextlib
    import io

    from piper.download_voices import list_voices

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):  # Piper prints the list rather than returning it
        list_voices()
    return sorted(n for n in captured.getvalue().split() if n.lower().startswith(language.lower()))


def distinct_voices(language: str = "en") -> list[str]:
    """One entry per voice: its medium model if there is one, otherwise high, otherwise low."""
    best: dict[str, str] = {}
    rank = {"medium": 0, "high": 1, "low": 2}
    for name in piper_voices(language):
        voice, _, quality = name.rpartition("-")
        if voice not in best or rank.get(quality, 3) < rank.get(best[voice].rpartition("-")[2], 3):
            best[voice] = name
    return sorted(best.values())


def catalog(language: str = "en") -> None:
    short = {name: alias for alias, (name, _) in config.VOICES.items()}
    names = piper_voices(language)
    print(f"{len(names)} Piper voices starting with '{language}'. Quality: low < medium < high (high is slower).\n")
    for name in names:
        print(f"  {name:40} {'short name: ' + short[name] if name in short else ''}")
    print("\nUse any of them by full name:  KIN_VOICE=<name>   or hear one:  python scripts/voices.py play <name>")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "play":
        play(sys.argv[2:] or list(config.VOICES))
    elif command == "catalog":
        catalog(sys.argv[2] if len(sys.argv) > 2 else "en")
    else:
        listing()
