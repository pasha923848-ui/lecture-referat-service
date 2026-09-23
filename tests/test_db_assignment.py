import pytest

from app import db
from app.assignment import AssignmentRules, LectureQuestions


@pytest.fixture(autouse=True)
def clean_state():
    with db._lock:
        db._conn.execute("DELETE FROM assignment_rules")
        db._conn.execute("DELETE FROM assignment_lectures")
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM ingested_files")
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield


def test_get_assignment_rules_defaults_when_none_saved():
    rules = db.get_assignment_rules()
    assert rules.min_pages == 5
    assert rules.min_questions == 3


def test_save_and_get_assignment_rules_roundtrip():
    db.save_assignment(AssignmentRules(min_pages=7, font_size=14, min_questions=2), [])
    rules = db.get_assignment_rules()
    assert rules.min_pages == 7
    assert rules.font_size == 14
    assert rules.min_questions == 2


def test_save_and_get_lecture_questions_roundtrip():
    lectures = [
        LectureQuestions(lecture_number=1, topic="Основы теории информации", questions=["Вопрос А?", "Вопрос Б?"]),
        LectureQuestions(lecture_number=2, topic="Сетевые технологии", questions=["Вопрос В?"]),
    ]
    db.save_assignment(AssignmentRules(), lectures)

    lecture1 = db.get_lecture_questions(1)
    assert lecture1.topic == "Основы теории информации"
    assert lecture1.questions == ["Вопрос А?", "Вопрос Б?"]

    assert db.get_lecture_questions(99) is None

    all_lectures = db.list_lecture_questions()
    assert [l.lecture_number for l in all_lectures] == [1, 2]


def test_material_lecture_number_defaults_to_none():
    material = db.create_material("m1", "Занятие 1", "/tmp/m1")
    assert material.lecture_number is None
    fetched = db.get_material("m1")
    assert fetched.lecture_number is None


def test_update_material_sets_lecture_number_and_source():
    db.create_material("m2", "Занятие 2", "/tmp/m2")
    db.update_material("m2", lecture_number=4, lecture_number_source="auto")

    material = db.get_material("m2")
    assert material.lecture_number == 4
    assert material.lecture_number_source == "auto"

    db.update_material("m2", lecture_number=5, lecture_number_source="manual")
    material = db.get_material("m2")
    assert material.lecture_number == 5
    assert material.lecture_number_source == "manual"


def test_update_material_tracks_stage_and_progress_while_processing():
    db.create_material("m3", "Занятие 3", "/tmp/m3")
    db.update_material("m3", status="processing", stage="Извлечение звука из видео…", progress=0)

    material = db.get_material("m3")
    assert material.stage == "Извлечение звука из видео…"
    assert material.progress == 0

    db.update_material("m3", status="processing", stage="Распознавание речи: 42%", progress=42)
    material = db.get_material("m3")
    assert material.stage == "Распознавание речи: 42%"
    assert material.progress == 42


def test_update_material_clears_stage_and_progress_once_finished():
    db.create_material("m4", "Занятие 4", "/tmp/m4")
    db.update_material("m4", status="processing", stage="Распознавание речи: 90%", progress=90)

    db.update_material("m4", status="done")
    material = db.get_material("m4")
    assert material.status == "done"
    assert material.stage is None
    assert material.progress is None


def test_settings_roundtrip_and_delete():
    assert db.get_setting("gigachat_api_key") is None

    db.set_setting("gigachat_api_key", "abc123")
    assert db.get_setting("gigachat_api_key") == "abc123"

    db.set_setting("gigachat_api_key", "replaced")
    assert db.get_setting("gigachat_api_key") == "replaced"

    db.delete_setting("gigachat_api_key")
    assert db.get_setting("gigachat_api_key") is None


def test_save_and_get_lecture_questions_roundtrip_includes_notes():
    lectures = [
        LectureQuestions(
            lecture_number=1,
            topic="Тема",
            questions=["Вопрос?"],
            notes="Использовать материал лекции 2 тоже.",
        ),
    ]
    db.save_assignment(AssignmentRules(), lectures)

    lecture = db.get_lecture_questions(1)
    assert lecture.notes == "Использовать материал лекции 2 тоже."


def test_delete_material_removes_row_and_ingested_files_tracking():
    db.create_material("m5", "Занятие 5", "/tmp/m5")
    db.mark_ingested("local", "remote-123", "m5")
    assert db.is_ingested("local", "remote-123")

    db.delete_material("m5")

    assert db.get_material("m5") is None
    assert not db.is_ingested("local", "remote-123")
