"""The video intro, in one take: Vector drives forward, stops, and introduces himself in his own voice;
after a pause for the camera cut, Kin introduces himself through the hub speaker in his usual voice.

Run on the hub with the service stopped (it holds the robot between tasks):

    systemctl --user stop kin
    .venv/bin/python scripts/intro.py --distance 200 --pause 4
    systemctl --user start kin

Options: --vector-line / --kin-line to change the words, --turn degrees (negative is right), --distance in mm (after leaving the charger,
if he starts on it), --pause seconds between the two lines, --loud to turn his built-in voice up first,
--skip-vector / --skip-kin to film one half again. He is left where he stopped; put him back by hand.
"""

import argparse
import time

from anki_vector.audio import RobotVolumeLevel
from anki_vector.util import degrees, distance_mm, speed_mmps

from kin import robot as kin_robot
from kin.robot import call, with_retry

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--vector-line", default="Hello. I am Vector.")
parser.add_argument("--kin-line", default="And I am Kin. I am his voice, his ears and his memory. I do my thinking on Nebius.")
parser.add_argument("--distance", type=int, default=200, help="mm to drive forward (default 200)")
parser.add_argument("--speed", type=int, default=60, help="mm per second (default 60)")
parser.add_argument("--turn", type=int, default=0, help="degrees to turn after the drive; negative is right (default 0)")
parser.add_argument("--pause", type=float, default=4.0, help="seconds between the two lines (default 4)")
parser.add_argument("--wave-lift", action="store_true", help="raise and lower the lift after his line")
parser.add_argument("--loud", action="store_true", help="set Vector's own volume to high before he speaks")
parser.add_argument("--skip-vector", action="store_true")
parser.add_argument("--skip-kin", action="store_true")
args = parser.parse_args()

with kin_robot.connect() as robot:
    mouth = None
    if not args.skip_kin:
        from kin.voice import HubMouth

        mouth = HubMouth(robot)  # loading the voice takes a few seconds; spend them before the camera rolls
    if args.loud:
        call(robot.audio.set_master_volume, RobotVolumeLevel.HIGH, timeout=10)
    if not args.skip_vector:
        if robot.status.is_on_charger:
            print("Leaving the charger...", flush=True)
            with_retry(lambda: call(robot.behavior.drive_off_charger, timeout=30))
        print(f"Driving {args.distance} mm...", flush=True)
        with_retry(lambda: call(robot.behavior.set_head_angle, degrees(0), timeout=10))
        with_retry(lambda: call(robot.behavior.drive_straight, distance_mm(args.distance), speed_mmps(args.speed), timeout=30))
        time.sleep(0.8)  # a visible stop before he looks up
        if args.turn:
            with_retry(lambda: call(robot.behavior.turn_in_place, degrees(args.turn), timeout=20))
            time.sleep(0.5)
        with_retry(lambda: call(robot.behavior.set_head_angle, degrees(30), timeout=10))
        time.sleep(0.5)
        print(f"Vector: {args.vector_line}", flush=True)
        with_retry(lambda: call(robot.behavior.say_text, args.vector_line, timeout=20))
        if args.wave_lift:
            time.sleep(0.5)
            with_retry(lambda: call(robot.behavior.set_lift_height, 1.0, timeout=10))
            time.sleep(0.8)
            with_retry(lambda: call(robot.behavior.set_lift_height, 0.0, timeout=10))
    if mouth is not None:
        if not args.skip_vector:
            print(f"Pausing {args.pause:.0f} s for the cut...", flush=True)
            time.sleep(args.pause)
        print(f"Kin: {args.kin_line}", flush=True)
        mouth.say(args.kin_line)
    print("Done. He stays where he is; put him back on the charger by hand.", flush=True)
