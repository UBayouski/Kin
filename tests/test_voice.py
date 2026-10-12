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
    ears = SimpleNamespace(prepare=lambda: order.append("mic open"), listen=lambda timeout=12.0: order.append("listen") or "fine")
    mouth = SimpleNamespace(say=lambda text: order.append("speak"))
    kit = RobotTools(SimpleNamespace(anim=SimpleNamespace(play_animation_trigger=lambda *a, **k: None)),
                     memory, mouth=mouth, ears=ears)

    assert kit.dispatch("ask", '{"question": "How did you sleep?"}') == "the person replied: fine"
    assert order == ["mic open", "speak", "listen"]


def test_a_poisoned_noise_floor_cannot_hide_the_persons_voice(monkeypatch):
    """Sep 23: "No, thank you" at 0.076 went unheard; the floor carried over from earlier noise, sat above
    her voice, and then adapted towards it."""
    monkeypatch.setenv("KIN_END_PAUSE", "0.5")
    ears = make_ears([], [])
    ears._noise_floor = 0.03  # poisoned

    audio = ears._utterance(iter(blocks_of(*[0.0003] * 5, *[0.07] * 8, *[0.0003] * 10)), timeout=12)

    assert audio is not None, "0.07 is speech in any room"
    assert ears._noise_floor <= 0.006


def test_the_floor_is_measured_afresh_on_each_listen():
    ears = make_ears(captures=[None], transcripts=[])
    ears._noise_floor = 0.03

    ears.listen(timeout=1)

    assert ears._noise_floor is None


def test_the_tail_of_kins_own_words_is_let_pass_before_listening():
    """Every capture began with Kin's last words ("Are you there? Yes."): the speaker finishes a little
    after `say` returns. Blocks are skipped until the room has been quiet for 0.3 s."""
    import queue

    ears = make_ears([], [])
    ears._blocks = queue.Queue()
    for block in blocks_of(0.1, 0.08, 0.05, 0.001, 0.001, 0.001, 0.07):  # tail, quiet, then the person
        ears._blocks.put(block)

    skipped = ears._skip_speaker_tail()

    assert abs(skipped - 0.6) < 0.01, "three loud blocks and three quiet ones"
    assert ears._blocks.qsize() == 1, "the person's first block is left for the capture"


def test_skipping_the_tail_gives_up_after_a_second_of_noise():
    import queue

    ears = make_ears([], [])
    ears._blocks = queue.Queue()
    for block in blocks_of(*[0.1] * 15):
        ears._blocks.put(block)

    assert abs(ears._skip_speaker_tail() - 1.0) < 0.01


def test_a_dead_microphone_is_told_from_a_quiet_room():
    """Oct 3: the Windows default input had become a webcam delivering silence. Kin called out twice,
    "heard" nothing, and was one presence deadline away from telling the family nobody was home."""
    def silent_at(level):
        ears = make_ears([], [])
        ears._dead_listens = 0

        def record(timeout):          # what `_utterance` does when nobody speaks
            ears._loudest = level
            return None
        ears._record = record
        return ears

    webcam = silent_at(0.00003)       # what the wrong device gave
    for _ in range(4):
        webcam.listen(timeout=1)
    assert webcam.seems_deaf()

    room = silent_at(0.0027)          # a real room with nobody talking
    for _ in range(4):
        room.listen(timeout=1)
    assert not room.seems_deaf(), "an empty quiet room is not a broken microphone"


def test_hearing_words_clears_the_deaf_count():
    ears = make_ears(captures=[seconds_of_sound(2)], transcripts=["I'm here."])
    ears._dead_listens = 2

    assert ears.listen(timeout=5) == "I'm here."
    assert ears._dead_listens == 0 and not ears.seems_deaf()


def test_hub_speech_ends_with_a_little_silence():
    """Oct 8: the last letters of every sentence were clipped; the output device closed on the final sample."""
    from kin.voice import with_tail

    audio = np.full(1000, 1234, dtype=np.int16)
    padded = with_tail(audio, sample_rate=22050, seconds=0.3)

    assert padded.dtype == np.int16
    assert len(padded) == 1000 + int(22050 * 0.3)
    assert padded[:1000].tolist() == audio.tolist() and not padded[1000:].any()


def test_speech_is_resampled_to_a_rate_the_speaker_takes():
    """Oct 9, first run on the Linux hub: the Jabra takes 16 or 48 kHz through ALSA and refused Piper's
    22050 Hz, so every sentence failed. Windows had been resampling silently."""
    import numpy as np

    from kin import voice

    class Speaker:
        def query_devices(self, device, kind):
            return {"default_samplerate": 48000.0}

        def check_output_settings(self, device, samplerate):
            if samplerate not in (16000, 48000):
                raise ValueError("Invalid sample rate")

    assert voice.playback_rate(Speaker(), 6, 22050) == 48000

    class LyingSpeaker(Speaker):
        """The real Jabra: reports 16 kHz as its default and accepts it, then plays it at 48 kHz."""
        def query_devices(self, device, kind):
            return {"default_samplerate": 16000.0}

    assert voice.playback_rate(LyingSpeaker(), 6, 22050) == 48000, "a higher accepted rate beats the device's default"

    class AnySpeaker(Speaker):
        def check_output_settings(self, device, samplerate):
            pass

    assert voice.playback_rate(AnySpeaker(), 6, 22050) == 22050, "a device that takes the voice's rate gets it unchanged"

    tone = np.full(22050, 1000, dtype=np.int16)  # one second of a constant signal
    out = voice.resample(tone, 22050, 48000)
    assert len(out) == 48000 and out.dtype == np.int16
    assert int(out.min()) == 1000 and int(out.max()) == 1000, "resampling must not change the signal's level"
    assert voice.resample(tone, 22050, 22050) is tone


def test_the_ears_remember_a_voice_they_could_not_make_out():
    ears = make_ears(captures=[seconds_of_sound(3)] * 50, transcripts=[""] * 50)
    assert ears.listen(timeout=2) is None
    assert ears.heard_voice is True, "someone spoke; `ask` treats that as a quiet no to an offer of more"

    silent = make_ears(captures=[None], transcripts=[])
    silent._loudest = 0.01
    assert silent.listen(timeout=2) is None
    assert silent.heard_voice is False

    answered = make_ears(captures=[seconds_of_sound(2)], transcripts=["Yes."])
    assert answered.listen(timeout=2) == "Yes."
    assert answered.heard_voice is False


def test_whispers_guesses_on_noise_are_not_taken_for_words():
    """Oct 9, a quiet evening: near-silence became "I want you to know that I am a human being" (twice) and
    Kin thanked her for sharing; 11 s of room noise decoded, slowly, into "yeah yeah yeah aye aye aye"."""
    from kin.voice import words_in

    seg = lambda text, no_speech=0.0, logprob=-0.3: SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob)

    assert words_in([seg(" Yes, I did. ")]) == "Yes, I did."
    assert words_in([seg("Thank you.")]) == "Thank you.", "a real short reply stays"
    assert words_in([seg("I want you to know that I am a human being.", no_speech=0.1)]) == ""
    assert words_in([seg("See ya Yeah yeah yeah yeah Aye aye aye aye aye gite Prize Aye aye aye")]) == ""
    assert words_in([seg("Call my daughter please."), seg(" Call my daughter please.")]) == ""
    assert words_in([seg("Not today, I think.", no_speech=0.9)]) == "Not today, I think.", "a high no-speech score alone is not enough"
    assert words_in([seg("Yes, I did.", logprob=-1.4)]) == "Yes, I did.", "nor a poor decode alone: Oct 10 to 11, an 'or' here dropped every real reply for a day"
    assert words_in([seg("mumble mumble", no_speech=0.9, logprob=-1.4)]) == "", "both together is Whisper's own rule for silence"
    assert words_in([seg("Fine,", logprob=-0.2), seg("thanks.", no_speech=0.95, logprob=-1.3)]) == "Fine,", "segments are judged one by one"
    assert words_in([SimpleNamespace(text="Yes.")]) == "Yes.", "a segment without the scores (the tests' fake) passes through"
