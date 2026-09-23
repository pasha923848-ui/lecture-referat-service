import pytest

from app import config, db
from app.assignment import AssignmentRules, LectureQuestions
from app.reference.writer import (
    ReferenceError,
    combined_reference_filename,
    generate_combined_reference,
)

LONG_ANSWER = "Существует несколько важных аспектов данной темы. " * 60


def _fake_long_generate(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    return LONG_ANSWER


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM assignment_rules")
        db._conn.execute("DELETE FROM assignment_lectures")
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDENT_NAME", "Иванов И.И.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "2395")
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def _setup_two_lectures():
    db.create_material("m4", "Занятие 4 - Сети", "/tmp/does-not-matter")
    db.update_material("m4", lecture_number=4, lecture_number_source="auto")
    db.create_material("m5", "Занятие 5 - Маршрутизация", "/tmp/does-not-matter")
    db.update_material("m5", lecture_number=5, lecture_number_source="auto")

    lectures = [
        LectureQuestions(
            lecture_number=4,
            topic="Коммутация",
            questions=[f"Вопрос номер {i} про коммутацию?" for i in range(1, 6)],
        ),
        LectureQuestions(
            lecture_number=5,
            topic="Маршрутизация",
            questions=[f"Вопрос номер {i} про маршрутизацию?" for i in range(1, 6)],
        ),
    ]
    db.save_assignment(AssignmentRules(), lectures)


def test_combined_reference_filename_joins_lecture_numbers(monkeypatch):
    monkeypatch.setattr(config, "STUDENT_NAME", "Петров П.П.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "1234")
    assert combined_reference_filename([4, 5, 6]) == "ЛК4-5-6_Петров_1234.pdf"


def test_generate_combined_reference_requires_at_least_two_materials():
    db.create_material("m4", "Занятие 4", "/tmp/x")
    db.update_material("m4", lecture_number=4, lecture_number_source="auto")
    with pytest.raises(ReferenceError, match="минимум 2"):
        generate_combined_reference(["m4"], generate_fn=_fake_long_generate)


def test_generate_combined_reference_rejects_duplicate_material_ids():
    _setup_two_lectures()
    with pytest.raises(ReferenceError, match="несколько раз"):
        generate_combined_reference(["m4", "m4"], generate_fn=_fake_long_generate)


def test_generate_combined_reference_merges_materials_sharing_a_lecture_number():
    # A teacher sometimes uploads one lecture as several video parts (e.g.
    # "Лекция 1_1", "Лекция 1_2") — the ingestion pipeline gives each its
    # own material, but they all auto-detect to the same lecture_number.
    # Those should merge into ONE section for that lecture (using both
    # materials' transcripts together), not be rejected as a conflict.
    db.create_material("m4a", "Занятие 4, часть 1", "/tmp/m4a")
    db.update_material("m4a", lecture_number=4, lecture_number_source="auto")
    db.create_material("m4b", "Занятие 4, часть 2", "/tmp/m4b")
    db.update_material("m4b", lecture_number=4, lecture_number_source="manual")

    lectures = [
        LectureQuestions(
            lecture_number=4,
            topic="Коммутация",
            questions=[f"Вопрос номер {i} про коммутацию?" for i in range(1, 4)],
        ),
    ]
    db.save_assignment(AssignmentRules(), lectures)

    with pytest.raises(ReferenceError, match="минимум 2"):
        # Sanity check the *count* guard is about materials, not distinct
        # lecture numbers — a single one-material "group" still isn't enough.
        generate_combined_reference(["m4a"], generate_fn=_fake_long_generate)

    output_path, report = generate_combined_reference(
        ["m4a", "m4b"], question_count=2, generate_fn=_fake_long_generate
    )

    assert output_path.exists()
    # Both videos of one lecture make an ordinary single-lecture реферат:
    # each question once as a heading and once in СОДЕРЖАНИЕ — not twice
    # over as it would be if each material produced its own group.
    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(output_path)).pages)
    assert text.count("Вопрос номер 1 про коммутацию") == 2
    assert "к лекциям" not in text


def test_generate_combined_reference_fails_without_lecture_number():
    db.create_material("m4", "Занятие 4", "/tmp/x")
    db.update_material("m4", lecture_number=4, lecture_number_source="auto")
    db.create_material("m5", "Видео без номера", "/tmp/x")
    with pytest.raises(ReferenceError, match="номер лекции"):
        generate_combined_reference(["m4", "m5"], generate_fn=_fake_long_generate)


def test_generate_combined_reference_fails_without_assignment_data():
    db.create_material("m4", "Занятие 4", "/tmp/x")
    db.update_material("m4", lecture_number=4, lecture_number_source="auto")
    db.create_material("m5", "Занятие 5", "/tmp/x")
    db.update_material("m5", lecture_number=5, lecture_number_source="auto")
    with pytest.raises(ReferenceError, match="вопросов"):
        generate_combined_reference(["m4", "m5"], generate_fn=_fake_long_generate)


def test_generate_combined_reference_produces_pdf_covering_both_lectures():
    _setup_two_lectures()

    output_path, report = generate_combined_reference(["m5", "m4"], question_count=2, generate_fn=_fake_long_generate)

    # Sorted into ascending lecture-number order regardless of input order.
    assert output_path.name == "ЛК4-5_Иванов_2395.pdf"
    assert output_path.exists()

    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(output_path)).pages)
    assert "Лекция 4" in text
    assert "Лекция 5" in text
    assert "Вопрос номер 1 про коммутацию" in text
    assert "Вопрос номер 1 про маршрутизацию" in text
    assert report.details["questions_matched"] == 4


def test_generate_combined_reference_without_question_count_auto_selects_per_lecture():
    # No explicit question_count -> each lecture's questions are picked via
    # select_covered_questions instead of a fixed count. With no real
    # transcript and a generate_fn that never returns parseable numbers,
    # coverage detection finds nothing and falls back to topping up to
    # AssignmentRules' default min_questions (3) per lecture.
    _setup_two_lectures()

    output_path, report = generate_combined_reference(["m4", "m5"], generate_fn=_fake_long_generate)

    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(output_path)).pages)
    assert "Вопрос номер 1 про коммутацию" in text
    assert "Вопрос номер 3 про коммутацию" in text
    assert "Вопрос номер 1 про маршрутизацию" in text
    assert "Вопрос номер 3 про маршрутизацию" in text
