"""Type requests to Kin and watch Vector carry them out. For development; the product runs as `python -m kin.service`.

usage: python scripts/console.py ["one-shot request"]

Kin speaks and listens according to KIN_MOUTH and KIN_EARS, so with KIN_EARS=hub his questions are
answered out loud even though the request itself is typed.
"""

import sys

from kin import robot as kin_robot
from kin import voice
from kin.agent import Agent

with kin_robot.connect() as robot:
    agent = Agent(robot, mouth=voice.make_mouth(robot), ears=voice.make_ears())
    if len(sys.argv) > 1:
        print(agent.run(" ".join(sys.argv[1:])))
    else:
        print("Kin console. Empty line or Ctrl+C to quit.")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            print("kin>", agent.run(text))
