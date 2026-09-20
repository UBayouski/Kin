"""Thin helpers over the wire-pod Vector SDK (imports as anki_vector)."""

import asyncio
import concurrent.futures
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import anki_vector
from PIL import Image

from kin import config

T = TypeVar("T")

# What the SDK raises when a reflex (cliff, pickup, leaving the charger) takes
# behavior control away mid-action. asyncio.CancelledError is a BaseException.
_CONTROL_LOST = (
    asyncio.CancelledError,
    concurrent.futures.CancelledError,
    anki_vector.exceptions.VectorControlException,
)


def _disconnect_quietly(robot: anki_vector.Robot) -> None:
    try:
        robot.disconnect()
    except Exception:  # a link that already dropped raises from the SDK's event stream; nothing left to clean up
        pass


@contextmanager
def connect(take_control: bool = True) -> Iterator[anki_vector.Robot]:
    """`with connect() as robot:`. Unlike the SDK's own context manager, a stalled disconnect cannot hang us.

    With `take_control=False` Vector keeps his own personality; wrap each task in `control(robot)` instead.
    """
    level = anki_vector.connection.ControlPriorityLevel.DEFAULT_PRIORITY if take_control else None
    robot = anki_vector.Robot(
        serial=config.vector_serial(), enable_face_detection=True, behavior_control_level=level
    )
    robot.connect()
    try:
        yield robot
    finally:
        closer = threading.Thread(target=_disconnect_quietly, args=(robot,), daemon=True)
        closer.start()
        closer.join(timeout=10)


@contextmanager
def control(robot: anki_vector.Robot) -> Iterator[None]:
    """Hold behavior control for one task, then give Vector back to himself so he stays alive between tasks."""
    if robot.conn.requires_behavior_control:  # connected with control already; nothing to borrow
        yield
        return
    robot.conn.request_control(timeout=15)
    try:
        yield
    finally:
        try:
            robot.conn.release_control(timeout=10)
        except Exception:  # a dropped link has released it for us
            pass


def is_touched(robot: anki_vector.Robot) -> bool:
    reading = robot.touch.last_sensor_reading
    return bool(reading and reading.is_being_touched)


def call(method: Callable[..., Any], *args: Any, timeout: float = 30, **kwargs: Any) -> Any:
    """Run an SDK call with a deadline.

    The SDK's blocking calls wait forever, and a Wi-Fi stall then freezes the whole process.
    Every SDK method accepts `_return_future`, which gives us something we can time out on.
    """
    return method(*args, _return_future=True, **kwargs).result(timeout=timeout)


def take_photo(robot: anki_vector.Robot) -> Image.Image:
    return call(robot.camera.capture_single_image, timeout=15).raw_image


def with_retry(action: Callable[[], T], attempts: int = 3) -> T:
    """Run an SDK action, retrying if behavior control was lost under it."""
    for attempt in range(attempts):
        try:
            return action()
        except _CONTROL_LOST:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")
