from datetime import datetime

import numpy as np
import pytest

from kin.memory import Memory


def test_recall_by_keyword_when_embeddings_are_unavailable(memory):
    memory.remember("The person's granddaughter Mila is visiting on 2026-09-20.", due="2026-09-20")
    memory.remember("The person loves growing tomatoes on the balcony.")

    hits = memory.recall("when is Mila coming?")

    assert [f.text for f in hits] == ["The person's granddaughter Mila is visiting on 2026-09-20."]
    assert hits[0].due == "2026-09-20"


def test_recall_with_nothing_related_returns_recent_facts(memory):
    memory.remember("First fact.")
    memory.remember("Second fact.")

    assert [f.text for f in memory.recall("quantum physics")] == ["Second fact.", "First fact."]
    assert [f.text for f in memory.recall("")] == ["Second fact.", "First fact."]


def test_recall_by_meaning_uses_the_margin_around_the_best_hit(tmp_path):
    vectors = {
        "family": [1.0, 0.0, 0.0], "Mila visits.": [0.9, 0.1, 0.0],
        "Tomatoes on the balcony.": [0.0, 1.0, 0.0], "Book club on Thursdays.": [0.1, 0.0, 1.0],
    }
    memory = Memory(tmp_path / "m.db", embed=lambda text, is_query: vectors[text])
    for text in list(vectors)[1:]:
        memory.remember(text)

    assert [f.text for f in memory.recall("family")] == ["Mila visits."]


def test_update_fact_changes_text_status_and_date(memory):
    fact_id = memory.remember("Dentist on Tuesday.", due="2026-09-22")

    assert memory.update_fact(fact_id, text="Dentist moved to Wednesday.", due="2026-09-23")
    assert memory.active_facts()[0].render().endswith("Dentist moved to Wednesday. (date: 2026-09-23)")

    assert memory.update_fact(fact_id, status="done")
    assert memory.active_facts() == []
    assert memory.recall("dentist") == []  # closed facts are not recalled


def test_update_fact_rejects_unknown_ids_and_statuses(memory):
    assert memory.update_fact(99, status="done") is False
    with pytest.raises(ValueError):
        memory.update_fact(1, status="forgotten")


def test_a_failed_embedding_does_not_lose_the_fact(memory):
    fact_id = memory.remember("Still saved while offline.")

    assert memory.active_facts()[0].id == fact_id
    assert memory.db.execute("SELECT embedding FROM facts").fetchone()[0] is None


def test_embeddings_are_stored_as_float32(tmp_path):
    memory = Memory(tmp_path / "m.db", embed=lambda text, is_query: [0.5, 0.25])
    memory.remember("A fact.")

    blob = memory.db.execute("SELECT embedding FROM facts").fetchone()[0]
    assert np.frombuffer(blob, dtype=np.float32).tolist() == [0.5, 0.25]


def test_last_sign_of_person_counts_speech_sightings_and_settled_reminders(memory):
    today = datetime.now().date()
    assert memory.last_sign_of_person(today) is None

    memory.log_turn("kin", "Hello, are you there?")  # Kin talking is not a sign of the person
    assert memory.last_sign_of_person(today) is None

    memory.log_turn("person", "Yes, I'm here.")
    spoke = memory.last_sign_of_person(today)
    assert spoke is not None

    memory.log_event("presence", "face seen")
    assert memory.last_sign_of_person(today) > spoke


def test_conversation_is_returned_oldest_first(memory):
    for text in ("one", "two", "three"):
        memory.log_turn("person", text)

    assert [text for _, _, text in memory.recent_turns(limit=2)] == ["two", "three"]
