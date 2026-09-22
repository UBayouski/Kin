"""Can Vector see his charger right now? Put him 20-30 cm in front of it, facing it, and run this.

Tells apart "the light is too poor / the charger is out of view" from "the check does not work".
Safe to run while the service is running; he does not move.

usage: python scripts/charger_check.py [seconds]     default 20
"""

import sys
import time

from anki_vector.util import degrees

from kin import robot as kin_robot
from kin.robot import call

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20

with kin_robot.connect() as robot:
    call(robot.behavior.set_head_angle, degrees(0), timeout=10)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        charger = robot.world.charger
        if charger is not None:
            pose = getattr(charger, "pose", None)  # the SDK can report the charger before it has placed it
            where = f" at x={pose.position.x:.0f} y={pose.position.y:.0f} mm from where he started" if pose else ""
            print(f"sees the charger{where} (first seen after {seconds - (deadline - time.monotonic()):.0f} s)", flush=True)
            break
        print("not seen yet...", flush=True)
        time.sleep(1.0)
    else:
        print("never saw it. Try brighter, even light, and make sure the charger's back wall faces him.")
