from types import SimpleNamespace

import numpy as np

from kin.voice import LISTEN_RATE, HubEars


def make_ears(captures, transcripts):
    """HubEars without the microphone or the Whisper model: `captures` is what each recording returns,
    `transcripts` what the model makes of each one that is long enough to transcribe."""
    ears = HubEars.__new__(HubEars)
    ears._noise_floor = None
    recordings, heard = list(captures), list(transcripts)
    ears._record = lambda timeout: recordings.pop(0) if recordings else None
    ears._model = SimpleNamespace(
        transcribe=lambda audio, **_: ([SimpleNamespace(text=heard.pop(0))], None)
    )
    return ears


def seconds_of_sound(seconds: float) -> np.ndarray:
    return np.zeros(int(LISTEN_RATE * seconds), dtype=np.float32)


def test_robot_noise_with_no_words_is_not_taken_for_a_reply():
    ears = make_ears(
        captures=[seconds_of_sound(3), seconds_of_sound(4)],  # first Vector purring after a pet, then the person
        transcripts=["", " What's the weather today? "],
    )

    assert ears.listen(timeout=12) == "What's the weather today?"


def test_a_click_too_short_to_be_speech_is_skipped_without_transcribing():
    ears = make_ears(captures=[seconds_of_sound(0.1), seconds_of_sound(2)], transcripts=["Yes, I did."])

    assert ears.listen(timeout=12) == "Yes, I did."


def test_silence_to_the_end_is_no_reply():
    assert make_ears(captures=[None], transcripts=[]).listen(timeout=12) is None


def test_noise_all_the_way_to_the_deadline_is_no_reply():
    ears = make_ears(captures=[seconds_of_sound(2)] * 50, transcripts=[""] * 50)

    assert ears.listen(timeout=0.6) is None


def blocks_of(*levels):
    """0.1 s microphone blocks at the given loudness levels."""
    return [np.full(int(LISTEN_RATE * 0.1), level, dtype=np.float32) for level in levels]


def test_the_soft_start_of_a_reply_is_kept(monkeypatch):
    """Live, "I slept well. Thank you." was heard as "Thank you very much. Thank you.": the opening words
    were lost. Part of the fix: what came just before the voice crossed the threshold belongs to the reply."""
    monkeypatch.setenv("KIN_END_PAUSE", "0.5")
    ears = make_ears([], [])
    quiet, soft, loud = 0.001, 0.004, 0.2  # `soft` is speech, but under the loudness threshold

    audio = ears._utterance(iter(blocks_of(*[quiet] * 10, soft, soft, loud, loud, loud, *[quiet] * 10)), timeout=12)

    assert audio is not None
    levels = audio[:: int(LISTEN_RATE * 0.1)]
    assert list(levels[:4]) == [np.float32(quiet)] * 2 + [np.float32(soft)] * 2, "0.4 s of lead-in, soft onset included"
    assert np.isclose(levels[4], loud)


def test_nobody_speaking_within_the_timeout_is_none():
    ears = make_ears([], [])

    assert ears._utterance(iter(blocks_of(*[0.001] * 50)), timeout=2) is None


def test_ask_opens_the_microphone_before_kin_speaks(memory):
    """Opening it takes 0.5 to 0.9 s; done after the question, a prompt answer lost its first words."""
    from kin.tools import RobotTools

    order = []
    ears = SimpleNamespace(prepare=lambda: order.append("mic open"), listen=lambda: order.append("listen") or "fine")
    mouth = SimpleNamespace(say=lambda text: order.append("speak"))
    kit = RobotTools(SimpleNamespace(anim=SimpleNamespace(play_animation_trigger=lambda *a, **k: None)),
                     memory, mouth=mouth, ears=ears)

    assert kit.dispatch("ask", '{"question": "How did you sleep?"}') == "the person replied: fine"
    assert order == ["mic open", "speak", "listen"]
