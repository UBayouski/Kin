"""Make "Hey Vector" and his back button reach Kin.

wire-pod keeps handling Vector's built-in commands ("I love you", "fist bump", "go to sleep"). Anything it
does not recognise is handed to Kin's local gateway: Vector says a short "On it!" in his own voice, then
Kin answers. The previous wire-pod settings are saved and can be put back.

    python scripts/wirepod_link.py            show the current wire-pod fallback settings
    python scripts/wirepod_link.py --link     send unrecognised speech to Kin (Kin's service must be running)
    python scripts/wirepod_link.py --unlink   restore what was there before
"""

import sys

from kin import config, wirepod


def show(settings: dict) -> None:
    for key in ("enable", "provider", "endpoint", "model", "intentgraph", "save_chat", "commands_enable"):
        print(f"  {key:16} {settings.get(key)!r}")


if "--link" in sys.argv:
    print(f"wire-pod at {config.wirepod_url()} now sends unrecognised speech to Kin:")
    show(wirepod.link())
    print(f"(previous settings saved in {wirepod.BACKUP})")
    taken = wirepod.take_over()
    print(f"and Kin answers these in place of Vector's built-in commands: {', '.join(taken['utterances'])}")
elif "--unlink" in sys.argv:
    print("wire-pod fallback restored to:")
    show(wirepod.unlink())
    print("Kin's custom intent removed" if wirepod.give_back() else "(Kin's custom intent was not installed)")
else:
    print(f"wire-pod at {config.wirepod_url()}, fallback for unrecognised speech:")
    show(wirepod.current())
    print("linked to Kin" if str(config.gateway_port()) in str(wirepod.current().get("endpoint")) else "not linked to Kin")
    taken = any(i.get("name") == wirepod.TAKEOVER for i in wirepod.custom_intents())
    print("weather questions go to Kin" if taken else "weather questions get Vector's built-in answer (run --link)")
