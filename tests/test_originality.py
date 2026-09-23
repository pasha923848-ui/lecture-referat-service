import pytest

from app import config, db
from app.assignment import AssignmentRules, LectureQuestions
from app.reference import writer

FILLER = "Своими словами: модель делит сетевое взаимодействие на уровни с отдельными задачами. " * 12
LECTURE = (
    "Модель открытых систем состоит из семи уровней и каждый уровень выполняет свою строго "
    "определённую функцию при передаче данных между компьютерами в сети. "
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    with db._lock:
        for table in ("materials", "assignment_rules", "assignment_lectures", "settings"):
            db._conn.execute(f"DELETE FROM {table}")
        db._conn.commit()
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDENT_NAME", "Иванов И.И.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "2395")
    yield


def test_copied_fragment_finds_long_verbatim_runs_only():
    assert writer.copied_fragment("Как сказал лектор, " + LECTURE, LECTURE)
    assert writer.copied_fragment("Модель включает семь уровней, у каждого своя функция.", LECTURE) == ""


def test_answer_copying_the_lecture_is_paraphrased():
    context = LECTURE * 3
    calls = []

    def fake(system_prompt, user_prompt, max_tokens):
        calls.append(user_prompt)
        if "дословно повторён" in user_prompt:
            return FILLER * 3
        return LECTURE * 8

    text = writer.write_answer("Что такое модель OSI?", context, "Дисциплина", target_words=200, generate_fn=fake)

    assert any("дословно повторён" in c for c in calls)
    assert writer.copied_fragment(text, context) == ""
    assert "Своими словами" in text


def test_every_reference_gets_one_random_style_directive(tmp_path):
    db.create_material("m1", "Занятие 4", str(tmp_path))
    db.update_material("m1", lecture_number=4, lecture_number_source="manual")
    questions = [f"Вопрос номер {i}?" for i in range(1, 4)]
    db.save_assignment(AssignmentRules(), [LectureQuestions(lecture_number=4, topic="Сети", questions=questions)])

    prompts = []

    def fake(system_prompt, user_prompt, max_tokens):
        prompts.append(system_prompt)
        return "Существует несколько важных аспектов данной темы. " * 60

    writer.generate_reference("m1", question_count=3, generate_fn=fake)

    used = {style for style in writer.STYLE_VARIANTS if any(style in p for p in prompts)}
    assert len(used) == 1
    assert all("СТИЛЬ ИЗЛОЖЕНИЯ" in p for p in prompts)


def test_style_choice_varies_between_runs():
    assert {writer.pick_style() for _ in range(200)} == set(writer.STYLE_VARIANTS)
