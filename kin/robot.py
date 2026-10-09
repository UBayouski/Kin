"""Thin helpers over the wire-pod Vector SDK (imports as anki_vector)."""

import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import anki_vector
from anki_vector.connection import ControlPriorityLevel
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
    hold_requested = getattr(robot.conn, "behavior_control_level", None) == ControlPriorityLevel.RESERVE_CONTROL
    if robot.conn.requires_behavior_control and not hold_requested:  # connected with control already; nothing to borrow
        yield
        return
    if hold_requested:
        _answered(robot, lambda: robot.conn.release_control(timeout=10))  # a request on top of the hold gets no grant
    _answered(robot, lambda: robot.conn.request_control(timeout=15), timeout=15)
    try:
        yield
    finally:
        try:  # safety net: a loop still running at the end of a task would run on under the idle hold
            if config.idle_hold() and robot.status.is_animating:
                end_animation(robot)
                time.sleep(0.5)
        except Exception:
            pass
        try:
            robot.conn.release_control(timeout=10)  # releasing task control gets no answer from him (measured)
        except Exception:
            pass  # a dropped link has released it for us
        time.sleep(0.3)  # let the release go out before the next request; sent together they merge into one
        hold_idle(robot)


def hold_idle(robot: anki_vector.Robot) -> bool:
    """Between tasks, keep the light hold that stops Vector's idle behaviours (see config.idle_hold).
    Returns whether the robot answered the request. A dropped link releases every hold, and a request
    made while the link is still shaky can go unanswered: live, he wandered off the dock in the gap."""
    if not config.idle_hold() or busy_being_himself():
        return False
    held = _answered(robot, lambda: robot.conn.request_control(
        behavior_control_level=ControlPriorityLevel.RESERVE_CONTROL, timeout=10))
    robot.kin_idle_held = held  # the SDK records the level as soon as it is asked for, answered or not
    return held


_his_own_time_until = 0.0  # while this is in the future, Vector is left entirely to himself


def let_him_be(seconds: float) -> None:
    """Hand Vector back for a while so one of his own behaviours can run to the end. Kin takes no control
    and starts no task meanwhile: a fetch takes longer than a minute, and control cut it short (Sep 23)."""
    global _his_own_time_until
    _his_own_time_until = max(_his_own_time_until, time.monotonic() + seconds)


def busy_being_himself() -> bool:
    return time.monotonic() < _his_own_time_until


def hand_back(robot: anki_vector.Robot) -> None:
    """Release whatever control Kin holds, right now, mid-task if need be."""
    try:
        robot.conn.release_control(timeout=10)
    except Exception:
        pass
    robot.kin_idle_held = False


def drop_hold(robot: anki_vector.Robot) -> None:
    """Give Vector back to himself for a while (he has been spoken to and may want to react)."""
    if getattr(robot.conn, "behavior_control_level", None) == ControlPriorityLevel.RESERVE_CONTROL:
        _answered(robot, lambda: robot.conn.release_control(timeout=10))
    robot.kin_idle_held = False


def idle_held(robot: anki_vector.Robot) -> bool:
    """Whether the robot answered the last hold request. Not the SDK's level, which is set on asking:
    a request that went unanswered on a stale link looked held for half an hour (Sep 23)."""
    return bool(getattr(robot, "kin_idle_held", False))


def _answered(robot: anki_vector.Robot, send: Callable[[], Any], timeout: float = 10.0) -> bool:
    """Send a control request or release and wait for the robot's own answer to it.

    The SDK's request_control / release_control return before the answer arrives (measured: up to 0.8 s
    early), and two sent back to back merge into one, after which the robot never sends a fresh grant
    and every action waits "until behavior control is granted" forever. The answers all pass through
    the event manager's `update`, so they are counted there. Never raises: a link that has dropped has
    released everything anyway, and a request that gets no answer shows up as the task's first action."""
    events = getattr(robot.conn, "_control_events", None)
    if events is not None and not hasattr(events, "kin_answers"):
        events.kin_answers = 0
        original = events.update

        def counted(enabled: bool) -> None:
            original(enabled)
            events.kin_answers += 1
        events.update = counted
    before = events.kin_answers if events is not None else 0
    try:
        send()
    except Exception:
        return False
    if events is None:  # a stand-in robot in tests
        return True
    deadline = time.monotonic() + timeout
    while events.kin_answers == before and time.monotonic() < deadline:
        time.sleep(0.05)
    return events.kin_answers != before


class TouchWatcher:
    """Notices a deliberate pet on Vector's back.

    The firmware's `is_being_touched` flag flickers on and off even during a steady stroke, so one
    sample means little. This samples it 20 times a second on a background thread and reports a pet
    when enough of the last 1.5 seconds was 'touched'.

    A hand always comes off again, so a second pet is only reported after the sensor has been released.
    Without that rule, anything resting against his back reports a pet every cooldown for as long as it
    is there: in his new enclosure that was one every 20 s for an hour (Sep 28), which filled memory with
    presence, made the house look permanently occupied, and interrupted what he was doing.
    """

    SAMPLE_SECONDS = 0.05
    WINDOW_SAMPLES = 30      # 1.5 s
    TOUCHED_SAMPLES = 8      # at least 0.4 s of it touched
    RELEASED_SAMPLES = 10    # 0.5 s untouched before another pet can be reported
    COOLDOWN_SECONDS = 20

    def __init__(self, robot: anki_vector.Robot):
        self._robot = robot
        self._pet = threading.Event()
        self._stop = threading.Event()
        self._resting_since: float | None = None  # when the sensor last started reading touched without a break
        self._window: deque[bool] = deque(maxlen=self.WINDOW_SAMPLES)
        self._released = True
        self._quiet_until = 0.0
        threading.Thread(target=self._watch, daemon=True, name="Touch watcher").start()

    def resting_on_something(self, seconds: float = 120.0) -> bool:
        """Has his back been touched without a break for this long? Then it is furniture, not a person."""
        return self._resting_since is not None and time.monotonic() - self._resting_since >= seconds

    def step(self, touched: bool, now: float) -> bool:
        """One sample. Returns whether this one completes a pet. Pure, so the rules can be tested."""
        self._window.append(touched)
        recent = list(self._window)[-self.RELEASED_SAMPLES:]
        if len(recent) >= self.RELEASED_SAMPLES and not any(recent):
            self._released = True
            self._resting_since = None
        elif all(self._window) and len(self._window) == self._window.maxlen and self._resting_since is None:
            self._resting_since = now
        if self._released and now >= self._quiet_until and sum(self._window) >= self.TOUCHED_SAMPLES:
            self._released = False
            self._window.clear()
            self._quiet_until = now + self.COOLDOWN_SECONDS
            return True
        return False

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                reading = self._robot.touch.last_sensor_reading
                touched = bool(reading and reading.is_being_touched)
            except Exception:
                touched = False
            if self.step(touched, time.monotonic()):
                self._pet.set()
            time.sleep(self.SAMPLE_SECONDS)

    def petted(self) -> bool:
        """True once per pet."""
        if self._pet.is_set():
            self._pet.clear()
            return True
        return False

    def discard(self) -> None:
        """Forget a pet that happened while Kin was busy, so a finished task is not followed by a stale one."""
        self._pet.clear()

    def stop(self) -> None:
        self._stop.set()


class FaceWatcher:
    """Notices any face Vector happens to see while being himself. Costs nothing: no control, no movement."""

    def __init__(self, robot: anki_vector.Robot):
        self._robot = robot
        self._seen = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True, name="Face watcher").start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                if any(True for _ in self._robot.world.visible_faces):
                    self._seen.set()
            except Exception:
                pass
            time.sleep(1.0)

    def seen(self) -> bool:
        """True once per sighting since the last call."""
        if self._seen.is_set():
            self._seen.clear()
            return True
        return False

    def stop(self) -> None:
        self._stop.set()


def body_silent(level: int, volts: float, touch_values: list[int | None]) -> bool:
    """Has Vector's head lost contact with his body board? (His error 899, which does not always show.)

    Seen live on Sep 21: the face, Wi-Fi and SDK link carried on for hours while the back button, touch
    sensor, motors and battery gauge were dead, and a battery read still "succeeded", so nothing noticed.
    The head then reports a "full" battery at 3.60 to 3.61 V (a real full charge is about 4.1 V) and a
    touch value that never changes (a live sensor jitters; it sat at exactly 5491 for 20 readings)."""
    impossible_battery = level == 3 and volts < 3.7
    touch_frozen = len(set(touch_values)) <= 1
    return impossible_battery and touch_frozen


def gauge_read(battery, previous_volts: float, status_on_charger: bool | None) -> bool:
    """Is this battery report worth acting on?

    Not when the gauge has not been read: 0.00 V right after a boot (Sep 22), or 3.60 to 3.61 V pinned
    to the hundredth across readings, which is the placeholder the head reports when it has no fresh
    figures from the body (Sep 21, and Sep 22 evening: 3.60 V "ok", off the charger for an hour by this
    report while the robot's own status said on it; it set off the rescue and the stranded alert).
    A real reading drifts by hundredths. Nor when the two charger-state sources disagree."""
    volts = battery.battery_volts
    if volts < 1.0:
        return False
    if abs(volts - 3.605) <= 0.01 and abs(volts - previous_volts) < 0.005:
        return False
    # Sep 24: pinned at exactly 3.635 V, "full", not charging, on the dock: a placeholder that had drifted
    # outside the band above. A real gauge moves by millivolts; "full" below 3.7 V is never real.
    if battery.battery_level == 3 and volts < 3.7 and abs(volts - previous_volts) < 0.0005:
        return False
    if status_on_charger is not None and bool(status_on_charger) != bool(battery.is_on_charger_platform):
        return False
    return True


def touch_samples(robot: anki_vector.Robot, count: int = 20, every: float = 0.05) -> list[int | None]:
    values: list[int | None] = []
    for _ in range(count):
        try:
            reading = robot.touch.last_sensor_reading
            values.append(reading.raw_touch_value if reading else None)
        except Exception:
            values.append(None)
        time.sleep(every)
    return values


def quiet_look(robot: anki_vector.Robot, seconds: float = 6.0) -> bool:
    """Raise the head and look for a face for a few seconds, silently and without driving.

    Meant to be run from the charger, which should face where the person usually sits. Leaving the
    charger for every look would mean a docking manoeuvre each time, and docking is the robot's least
    reliable move.
    """
    with control(robot):
        call(robot.behavior.set_head_angle, anki_vector.util.degrees(35), timeout=10)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if any(True for _ in robot.world.visible_faces):
                return True
            time.sleep(0.25)
    return False


class Interrupted(Exception):
    """A robot action was cut short because the person asked for Kin's attention."""


def call(method: Callable[..., Any], *args: Any, timeout: float = 30,
         interrupt: Callable[[], bool] | None = None, **kwargs: Any) -> Any:
    """Run an SDK call with a deadline, and optionally let the person cut it short.

    The SDK's blocking calls wait forever, and a Wi-Fi stall then freezes the whole process.
    Every SDK method accepts `_return_future`, which gives us something we can time out on.
    On a timeout or an interrupt the action is cancelled on the robot too; otherwise he would carry on
    (say, trying to dock) after we had stopped waiting for him.

    `interrupt` is polled a few times a second; when it returns True the action stops and Interrupted is raised.
    """
    future = method(*args, _return_future=True, **kwargs)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return future.result(timeout=0.2)
        except TimeoutError:
            if interrupt is not None and interrupt():
                future.cancel()
                raise Interrupted() from None
            if time.monotonic() >= deadline:
                future.cancel()
                raise


# How each looped animation is ended. Cancelling its future stops nothing on the robot (measured: still
# animating 2 s later); only another animation does. His own idle behaviours used to end the last loop
# of a task once control was released, and under the idle hold nothing does: he sat "listening" for
# minutes. So every loop is closed with its own get-out, or a neutral one where his set has none.
GET_OUT = {
    "VC_ListeningLoop": "VC_ListeningGetOut",
    "KnowledgeGraphAnswer": "KnowledgeGraphGetOut",
    "KnowledgeGraphSearching": "KnowledgeGraphSearchingGetOutSuccess",
}


def end_animation(robot: anki_vector.Robot, trigger: str = "KnowledgeGraphGetOut") -> None:
    """Play the short get-out that stops whatever loop is running. Not waited for; never raises."""
    try:
        robot.anim.play_animation_trigger(GET_OUT.get(trigger, trigger), ignore_body_track=True, _return_future=True)
    except Exception:
        pass


@contextmanager
def animating(robot: anki_vector.Robot, trigger: str) -> Iterator[None]:
    """Loop one of Vector's own animations for the length of a block: talking while the hub speaks for him,
    his listening look while the hub listens. His wheels stay still. Purely cosmetic, so it never raises."""
    future = None
    try:
        future = robot.anim.play_animation_trigger(trigger, loop_count=40, ignore_body_track=True, _return_future=True)
    except Exception:
        pass
    try:
        yield
    finally:
        if future is not None:
            future.cancel()
            end_animation(robot, trigger)


def charger_in_view(robot: anki_vector.Robot, interrupt: Callable[[], bool] | None = None) -> bool:
    """Does Vector know where his charger is? If not, turn once on the spot looking for it.

    Docking with no charger in sight is what used to eat two minutes before failing. This answers in
    about fifteen seconds, so Kin can ask for help instead.
    """
    if robot.world.charger is not None:
        return True
    call(robot.behavior.set_head_angle, anki_vector.util.degrees(0), timeout=10)  # the charger's marker sits low
    for _ in range(6):
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            if robot.world.charger is not None:
                return True
            time.sleep(0.2)
        call(robot.behavior.turn_in_place, anki_vector.util.degrees(60), timeout=15, interrupt=interrupt)
    return robot.world.charger is not None


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
