import pytest

from app import config, db
from app.assignment import AssignmentRules, LectureQuestions
from app.reference.writer import generate_reference_documents

LONG_ANSWER = "Существует несколько важных аспектов данной темы. " * 60


def _fake_generate(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
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


def _setup_lecture_without_split(tmp_path, num_questions=5):
    folder = tmp_path / "m1"
    folder.mkdir()
    db.create_material("m1", "Занятие 4", str(folder))
    db.update_material("m1", lecture_number=4, lecture_number_source="auto")
    questions = [f"Вопрос номер {i}?" for i in range(1, num_questions + 1)]
    db.save_assignment(AssignmentRules(), [LectureQuestions(lecture_number=4, topic="Тема", questions=questions)])
    return questions


def _setup_lecture_with_split(tmp_path):
    folder = tmp_path / "m1"
    folder.mkdir()
    db.create_material("m1", "Занятие 1", str(folder))
    db.update_material("m1", lecture_number=1, lecture_number_source="auto")
    questions = [
        "Вопрос про энтропию номер один?",
        "Вопрос про энтропию номер два?",
        "Вопрос про TCP протокол номер один?",
        "Вопрос про TCP протокол номер два?",
    ]
    db.save_assignment(
        AssignmentRules(min_questions=1),
        [
            LectureQuestions(
                lecture_number=1,
                topic="Основы теории информации",
                additional_topic="Протоколы стека TCP",
                questions=questions,
            )
        ],
    )
    return questions


def test_no_split_when_lecture_has_no_additional_topic(tmp_path):
    _setup_lecture_without_split(tmp_path)
    results = generate_reference_documents("m1", generate_fn=_fake_generate)
    assert len(results) == 1


def test_explicit_question_count_never_splits_even_with_additional_topic(tmp_path):
    _setup_lecture_with_split(tmp_path)
    results = generate_reference_documents("m1", question_count=2, generate_fn=_fake_generate)
    assert len(results) == 1


def test_splits_into_two_documents_when_additional_topic_present(tmp_path):
    _setup_lecture_with_split(tmp_path)

    def fake_generate(system_prompt, user_prompt, max_tokens):
        if "относится к основной теме" in user_prompt or "букву" in user_prompt:
            return "О,О,Д,Д"
        return LONG_ANSWER

    results = generate_reference_documents("m1", generate_fn=fake_generate)

    assert len(results) == 2
    paths = {p.name for p, _ in results}
    assert any("доп" in name for name in paths)
    assert any("доп" not in name for name in paths)


def test_main_document_is_kept_when_the_additional_part_has_no_material(tmp_path):
    from app.reference import writer

    _setup_lecture_with_split(tmp_path)

    def fake_generate(system_prompt, user_prompt, max_tokens):
        if "букву" in user_prompt:
            return "О,О,Д,Д"
        if "TCP" in user_prompt.split("Вопрос преподавателя:")[-1]:
            return writer.NOT_COVERED_PHRASE
        return LONG_ANSWER

    results = generate_reference_documents("m1", generate_fn=fake_generate)

    assert len(results) == 1
    path, report = results[0]
    assert path.exists() and "доп" not in path.name
    assert any(issue.startswith("Дополнительный реферат не составлен") for issue in report.issues)


def test_split_documents_have_distinct_content(tmp_path):
    _setup_lecture_with_split(tmp_path)

    def fake_generate(system_prompt, user_prompt, max_tokens):
        if "букву" in user_prompt:
            return "О,О,Д,Д"
        return LONG_ANSWER

    results = generate_reference_documents("m1", generate_fn=fake_generate)

    from pypdf import PdfReader

    texts = ["\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages) for path, _ in results]
    main_text, additional_text = texts[0], texts[1]

    assert "энтропию" in main_text
    assert "TCP протокол" not in main_text
    assert "TCP протокол" in additional_text
    assert "энтропию" not in additional_text
