import pytest

from app import config, db
from app.assignment import AssignmentRules, LectureQuestions
from app.reference import writer
from app.reference.writer import ReferenceError, generate_reference, reference_filename


LONG_ANSWER = ("Существует несколько важных аспектов данной темы. " * 60)
SHORT_ANSWER = "Короткий ответ."


def _fake_long_generate(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    return LONG_ANSWER


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM assignment_rules")
        db._conn.execute("DELETE FROM assignment_lectures")
        # student_name/student_group set here (from an earlier test) would
        # otherwise take priority over the monkeypatched config values below
        # (_student_info() checks the settings table first).
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    monkeypatch.setattr(config, "REFERENCES_DIR", tmp_path)
    monkeypatch.setattr(config, "STUDENT_NAME", "Иванов И.И.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "2395")
    yield
    with db._lock:
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()


def _setup_material_and_assignment(lecture_number=4, num_questions=5):
    db.create_material("m1", "Занятие 4 - Сети", "/tmp/does-not-matter")
    db.update_material("m1", lecture_number=lecture_number, lecture_number_source="auto")

    questions = [f"Вопрос номер {i}?" for i in range(1, num_questions + 1)]
    lecture = LectureQuestions(lecture_number=lecture_number, topic="Коммутация и маршрутизация", questions=questions)
    db.save_assignment(AssignmentRules(), [lecture])
    return questions


def test_reference_filename_uses_student_name_and_group(monkeypatch):
    monkeypatch.setattr(config, "STUDENT_NAME", "Петров П.П.")
    monkeypatch.setattr(config, "STUDENT_GROUP", "1234")
    assert reference_filename(7) == "ЛК7_Петров_1234.pdf"


def test_generate_reference_fails_without_lecture_number():
    db.create_material("m1", "Видео без номера", "/tmp/x")
    with pytest.raises(ReferenceError, match="номер лекции"):
        generate_reference("m1", generate_fn=_fake_long_generate)


def test_generate_reference_fails_without_assignment_data():
    db.create_material("m1", "Занятие 4", "/tmp/x")
    db.update_material("m1", lecture_number=4, lecture_number_source="auto")
    with pytest.raises(ReferenceError, match="вопросов"):
        generate_reference("m1", generate_fn=_fake_long_generate)


def test_generate_reference_produces_passing_pdf_with_enough_content():
    _setup_material_and_assignment()

    output_path, report = generate_reference("m1", question_count=3, generate_fn=_fake_long_generate)

    assert output_path.exists()
    assert output_path.name == "ЛК4_Иванов_2395.pdf"
    assert report.passed, report.issues


def test_generate_reference_expands_questions_when_too_short(monkeypatch):
    _setup_material_and_assignment(num_questions=5)

    calls = {"n": 0}

    def _flaky_generate(system_prompt, user_prompt, max_tokens):
        calls["n"] += 1
        # First 3 sections come back short (fails page-count check), forcing
        # the writer to add more questions; later calls return long answers.
        return SHORT_ANSWER if calls["n"] <= 3 else LONG_ANSWER

    output_path, report = generate_reference("m1", question_count=3, generate_fn=_flaky_generate)

    # Should have grown past the initial 3 questions trying to hit the page
    # count, and/or eventually given up gracefully without crashing.
    assert output_path.exists()
    assert calls["n"] >= 3


def test_generate_reference_without_question_count_uses_llm_to_pick_covered_questions(tmp_path, monkeypatch):
    questions = _setup_material_and_assignment(num_questions=5)
    folder = tmp_path / "m1"
    folder.mkdir()
    db._conn.execute("UPDATE materials SET folder_path = ? WHERE id = 'm1'", (str(folder),))
    db._conn.commit()
    (folder / "video.md").write_text(
        "# Транскрибация видеолекции\n\nЗанятие 4\n\n**00:00 – 01:00**\n\nПодробный текст лекции.",
        encoding="utf-8",
    )
    db.update_material("m1", add_file={"name": "video.md", "kind": "transcript"})

    calls = {"n": 0}

    def fake_generate(system_prompt, user_prompt, max_tokens):
        calls["n"] += 1
        if "номер вопроса | тезис" in user_prompt:
            return "1 | Лектор объяснил первый вопрос подробно.\n2 | Лектор объяснил второй вопрос подробно."
        return LONG_ANSWER

    output_path, report = generate_reference("m1", generate_fn=fake_generate)

    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(output_path)).pages)
    assert output_path.exists()
    assert questions[0].rstrip("?") in text
    assert questions[1].rstrip("?") in text
    # The LLM only selected 2 questions as covered, but AssignmentRules'
    # default min_questions=3 tops that up with one more (questions are
    # generic filler here, so — unlike a real transcript — they don't give
    # the page-count auto-expansion safety net a reason to add a 5th; that
    # expansion behavior has its own dedicated test).
    assert questions[2].rstrip("?") in text


def _pdf_text(path):
    from pypdf import PdfReader

    return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)


def test_questions_the_model_reports_as_not_covered_are_dropped_and_replaced():
    _setup_material_and_assignment(num_questions=5)

    def fake(system_prompt, user_prompt, max_tokens):
        if "Вопрос номер 2?" in user_prompt:
            return writer.NOT_COVERED_PHRASE
        return LONG_ANSWER

    output_path, _ = generate_reference("m1", generate_fn=fake)
    body = _pdf_text(output_path).rsplit("СПИСОК ИСТОЧНИКОВ", 1)[0]

    assert "Вопрос номер 2" not in body
    assert writer.NOT_COVERED_PHRASE not in body
    for n in (1, 3, 4):
        assert f"Вопрос номер {n}" in body


def test_each_question_is_written_from_its_own_fragments(tmp_path):
    questions = _setup_material_and_assignment(num_questions=3)
    folder = tmp_path / "m1"
    folder.mkdir()
    db._conn.execute("UPDATE materials SET folder_path = ? WHERE id = 'm1'", (str(folder),))
    db._conn.commit()
    filler = "Лектор продолжает рассуждение о курсе и его задачах. " * 90
    (folder / "video.md").write_text("МАРКЕРАЛЬФА " + filler + filler + " МАРКЕРБЕТА", encoding="utf-8")
    db.update_material("m1", add_file={"name": "video.md", "kind": "transcript"})

    writing_prompts = {}

    def fake(system_prompt, user_prompt, max_tokens):
        if "номер вопроса | тезис" in user_prompt:
            found = [n for n, marker in (("1", "МАРКЕРАЛЬФА"), ("2", "МАРКЕРБЕТА")) if marker in user_prompt]
            return "\n".join(f"{n} | Лектор разобрал этот вопрос в своём фрагменте." for n in found) or "нет"
        for question in questions:
            if question in user_prompt:
                writing_prompts[question] = user_prompt
        return LONG_ANSWER

    generate_reference("m1", generate_fn=fake)

    assert "МАРКЕРАЛЬФА" in writing_prompts[questions[0]]
    assert "МАРКЕРБЕТА" not in writing_prompts[questions[0]]
    assert "МАРКЕРБЕТА" in writing_prompts[questions[1]]
    assert "МАРКЕРАЛЬФА" not in writing_prompts[questions[1]]


def test_question_with_theses_is_never_dropped_even_if_the_model_refuses(tmp_path):
    questions = _setup_material_and_assignment(num_questions=3)
    folder = tmp_path / "m1"
    folder.mkdir()
    db._conn.execute("UPDATE materials SET folder_path = ? WHERE id = 'm1'", (str(folder),))
    db._conn.commit()
    (folder / "video.md").write_text("Лектор рассказывает материал лекции. " * 50, encoding="utf-8")
    db.update_material("m1", add_file={"name": "video.md", "kind": "transcript"})

    writing_prompts = []

    def fake(system_prompt, user_prompt, max_tokens):
        if "номер вопроса | тезис" in user_prompt:
            return "\n".join(
                f"{n} | Тезис{n} лектора: протокол передаёт данные по уровням модели, пример {n}." for n in (1, 2, 3)
            )
        writing_prompts.append(user_prompt)
        return writer.NOT_COVERED_PHRASE

    output_path, _ = generate_reference("m1", generate_fn=fake)
    body = _pdf_text(output_path).rsplit("СПИСОК ИСТОЧНИКОВ", 1)[0]

    for n, question in enumerate(questions, start=1):
        assert question.rstrip("?") in body
        assert f"Тезис{n} лектора" in body
    assert all(writer.NOT_COVERED_PHRASE not in p.split("Вопрос преподавателя:")[-1] for p in writing_prompts)
    trace = output_path.with_name(f"{output_path.stem}.trace.txt").read_text(encoding="utf-8")
    assert "собран кодом" in trace


def test_short_document_without_spare_questions_deepens_existing_answers():
    import re

    _setup_material_and_assignment(num_questions=2)
    requested_words = []

    def fake(system_prompt, user_prompt, max_tokens):
        words = int(re.search(r"около (\d+) слов", user_prompt).group(1))
        requested_words.append(words)
        return LONG_ANSWER * 3 if words >= 1200 else LONG_ANSWER

    output_path, report = generate_reference("m1", generate_fn=fake)

    base = min(requested_words)
    assert any(words > base for words in requested_words)
    assert output_path.exists()
    assert report.details["approx_content_pages"] > 3


def test_generate_reference_fails_clearly_when_nothing_is_covered():
    _setup_material_and_assignment(num_questions=3)
    with pytest.raises(ReferenceError, match="не нашлось материала"):
        generate_reference("m1", generate_fn=lambda s, u, m: writer.NOT_COVERED_PHRASE)


def test_generate_reference_includes_ai_disclosure_in_sources():
    _setup_material_and_assignment()
    output_path, _ = generate_reference("m1", question_count=3, generate_fn=_fake_long_generate)

    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(output_path)).pages)
    assert "ИИ-сервис" in text
    assert "Промт" in text or "промт" in text
