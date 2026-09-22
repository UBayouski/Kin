"""Live view of Vector's touch sensor, to learn what kind of touch registers (try it on and off the charger).

Prints a bar 4 times a second: '#' while the firmware says "being touched", '.' otherwise, the raw sensor
value, and PET! when Kin's own detector would fire. Safe to run while the service is running.

usage: python scripts/touch_meter.py [seconds]     default 60
"""

import sys
import time

from kin import robot as kin_robot

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60

with kin_robot.connect(take_control=False) as robot:
    watcher = kin_robot.TouchWatcher(robot)
    watcher.COOLDOWN_SECONDS = 3  # so repeated tries show up; the service waits 20 s
    print("Stroke the strip on his back. Ctrl+C to stop.", flush=True)
    history = ""
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            reading = robot.touch.last_sensor_reading
            touched = bool(reading and reading.is_being_touched)
            history = (history + ("#" if touched else "."))[-40:]
            fired = "  PET!" if watcher.petted() else ""
            charger = "on charger" if robot.status.is_on_charger else "off charger"
            print(f"{history:40}  raw {reading.raw_touch_value if reading else 0:5}  {charger}{fired}", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    watcher.stop()
