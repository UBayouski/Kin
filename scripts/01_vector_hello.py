"""Robot smoke test: leave the charger, speak, save a camera frame, go home."""

from pathlib import Path

from anki_vector.util import degrees, distance_mm, speed_mmps

from kin import robot as kin_robot
from kin.robot import call

out = Path(__file__).resolve().parent.parent / "captures"
out.mkdir(exist_ok=True)

with kin_robot.connect() as robot:
    call(robot.behavior.drive_off_charger)
    call(robot.behavior.say_text, "Hello. I am online.")
    call(robot.behavior.drive_straight, distance_mm(50), speed_mmps(50))
    call(robot.behavior.turn_in_place, degrees(45))
    call(robot.behavior.set_head_angle, degrees(25))

    frame = kin_robot.take_photo(robot)
    path = out / "hello.jpg"
    frame.save(path)
    print(f"Saved {path} ({frame.width}x{frame.height})")

    call(robot.behavior.drive_on_charger, timeout=120)
