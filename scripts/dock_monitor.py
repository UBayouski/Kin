"""Watch Vector on his charger and log what his body reports.

The first unit lost its body board while charging: error 899, reboot loops, or a face that kept moving
while touch, motors and the battery reading were dead. This polls the body every few seconds without
taking behaviour control, so his own personality runs as it would under Kin's idle hold, and writes one
line per change plus a heartbeat every minute. A dropped connection is logged with how long it lasted.

    python scripts/dock_monitor.py                  watch until Ctrl+C
    python scripts/dock_monitor.py --minutes 30     watch for half an hour, then print a summary
    python scripts/dock_monitor.py --every 5        poll every 5 s (default 10)

Log: data/logs/dock-monitor.log (appended).
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from kin import robot as kin_robot
from kin.robot import call

parser = argparse.ArgumentParser()
parser.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
parser.add_argument("--every", type=float, default=10, help="seconds between polls")
args = parser.parse_args()

log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "dock-monitor.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
logging.getLogger("anki_vector").setLevel(logging.ERROR)
log = logging.getLogger("dock")

WATCHED = ("is_on_charger", "is_charging", "is_picked_up", "is_being_held", "is_cliff_detected",
           "is_button_pressed", "is_in_calm_power_mode", "is_robot_moving", "is_animating")


def snapshot(robot) -> dict:
    b = call(robot.get_battery_state, timeout=10)
    s = robot.status
    out = {"on_charger": b.is_on_charger_platform, "charging": b.is_charging, "level": b.battery_level,
           "volts": round(b.battery_volts, 2)}
    out.update({k.removeprefix("is_"): bool(getattr(s, k)) for k in WATCHED})
    return out


def describe(snap: dict) -> str:
    flags = [k for k, v in snap.items() if v is True and k not in ("on_charger", "charging")]
    return (f"on_charger={snap['on_charger']} charging={snap['charging']} level={snap['level']} "
            f"volts={snap['volts']:.2f}" + (f" [{', '.join(flags)}]" if flags else ""))


started = datetime.now()
stop_at = time.monotonic() + args.minutes * 60 if args.minutes else None
drops: list[tuple[str, float]] = []
polls = 0
last: dict | None = None
last_heartbeat = 0.0
log.info("dock monitor started (every %.0f s%s), log in %s", args.every,
         f", for {args.minutes:g} min" if args.minutes else "", log_path)

try:
    while stop_at is None or time.monotonic() < stop_at:
        lost_at = None
        try:
            with kin_robot.connect(take_control=False) as robot:
                log.info("connected")
                while stop_at is None or time.monotonic() < stop_at:
                    snap = snapshot(robot)
                    polls += 1
                    now = time.monotonic()
                    if last is None:
                        log.info("state: %s", describe(snap))
                    else:
                        changed = {k: v for k, v in snap.items() if v != last.get(k) and k != "volts"}
                        if abs(snap["volts"] - last["volts"]) >= 0.05:
                            changed["volts"] = snap["volts"]
                        if changed:
                            log.info("change: %s  ->  %s", " ".join(f"{k}={v}" for k, v in changed.items()),
                                     describe(snap))
                    if now - last_heartbeat >= 60:
                        log.info("ok: %s", describe(snap))
                        last_heartbeat = now
                    last = snap
                    time.sleep(args.every)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # the SDK raises CancelledError and friends when the link goes
            lost_at = time.monotonic()
            log.warning("lost him: %s: %s", type(e).__name__, str(e).splitlines()[0] if str(e) else "")
            last = None
            while stop_at is None or time.monotonic() < stop_at:
                time.sleep(5)
                try:
                    with kin_robot.connect(take_control=False) as robot:
                        call(robot.get_battery_state, timeout=10)
                    break
                except BaseException as e2:
                    log.warning("still gone (%.0f s): %s", time.monotonic() - lost_at, type(e2).__name__)
            gap = time.monotonic() - lost_at
            drops.append((datetime.now().strftime("%H:%M:%S"), gap))
            log.warning("back after %.0f s", gap)
except KeyboardInterrupt:
    pass

minutes = (datetime.now() - started).total_seconds() / 60
log.info("summary: watched %.0f min, %d polls, %d drop(s)%s", minutes, polls, len(drops),
         "".join(f"; {at} for {gap:.0f} s" for at, gap in drops))
