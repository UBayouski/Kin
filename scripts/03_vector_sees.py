"""End to end: Vector takes a photo, the VLM describes it, Vector says it."""

import time

from anki_vector.util import degrees

from kin import llm
from kin import robot as kin_robot
from kin.robot import call

PROMPT = (
    "You are the eyes of a small tabletop robot. In one short spoken sentence, "
    "say what is in front of you. Mention a person if you see one."
)

with kin_robot.connect() as robot:
    call(robot.behavior.set_head_angle, degrees(25), timeout=10)
    frame = kin_robot.take_photo(robot)

    started = time.perf_counter()
    description = llm.describe_image(frame, PROMPT, max_tokens=60)
    print(f"[{time.perf_counter() - started:.1f}s] {description}")

    call(robot.behavior.say_text, description, timeout=45)
