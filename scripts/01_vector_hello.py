"""Robot smoke test: leave the charger, speak, save a camera frame, go home.

Each move is retried once control comes back: right off the charger the SDK can lose
behaviour control mid-action (a control_lost_event), which cancels the call."""

from pathlib import Path

from anki_vector.util import degrees, distance_mm, speed_mmps

from kin import robot as kin_robot
from kin.robot import call, with_retry

out = Path(__file__).resolve().parent.parent / "captures"
out.mkdir(exist_ok=True)

with kin_robot.connect() as robot:
    with_retry(lambda: call(robot.behavior.drive_off_charger))
    with_retry(lambda: call(robot.behavior.say_text, "Hello. I am online."))
    with_retry(lambda: call(robot.behavior.drive_straight, distance_mm(50), speed_mmps(50)))
    with_retry(lambda: call(robot.behavior.turn_in_place, degrees(45)))
    with_retry(lambda: call(robot.behavior.set_head_angle, degrees(25)))

    frame = kin_robot.take_photo(robot)
    path = out / "hello.jpg"
    frame.save(path)
    print(f"Saved {path} ({frame.width}x{frame.height})")

    with_retry(lambda: call(robot.behavior.drive_on_charger, timeout=120))
