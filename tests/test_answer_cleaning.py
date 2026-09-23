import pytest

from app import db
from app.reference import writer
from app.reference.writer import NOT_COVERED_PHRASE, clean_llm_output, judge_answer, write_answer

FILLER = "Лектор подробно объяснил, как устроена модель и зачем нужны её уровни. " * 12


@pytest.fixture(autouse=True)
def clean_state():
    with db._lock:
        db._conn.execute("DELETE FROM materials")
        db._conn.execute("DELETE FROM settings")
        db._conn.commit()
    yield


def test_exact_not_covered_phrase_is_not_covered():
    assert judge_answer(NOT_COVERED_PHRASE) == ("", False)


def test_refusal_in_own_words_followed_by_general_knowledge_is_not_covered():
    # Opening of a real generated section (реальный сгенерированный реферат, вопрос 1.1).
    raw = (
        "Тема отличия энтропийного оценивания информации от алфавитного не была подробно раскрыта в "
        "представленной лекции. Однако, исходя из контекста предложенного материала, можно предположить, "
        "что речь идет о двух подходах.\n\n" + FILLER
    )
    assert judge_answer(raw, strict=True)[1] is False

    text, covered = judge_answer(raw)
    assert covered
    assert "можно предположить" not in text and "не была подробно раскрыта" not in text


@pytest.mark.parametrize(
    "opening",
    [
        "В тексте лекции нет информации о протоколе UDP, однако можно отметить следующее.",
        "В данной лекции нет сведений о протоколе UDP.",
        "Лекция не содержит информации о протоколе UDP. Исходя из общих знаний, UDP работает без соединения.",
        "Лектор не затрагивал эту тему.",
    ],
)
def test_refusal_openings_in_own_words_reject_reserve_answers(opening):
    assert judge_answer(opening + " " + FILLER, strict=True)[1] is False


def test_grounded_answers_that_mention_the_source_or_impossibility_are_kept():
    text, covered = judge_answer("В представленной лекции подробно рассматривается модель OSI. " + FILLER)
    assert covered and "представленной лекции" not in text

    text, covered = judge_answer(
        "Протокол ARP определяет MAC-адрес по IP-адресу. Без него невозможно выполнить доставку кадра в "
        "локальной сети. " + FILLER
    )
    assert covered and "невозможно выполнить доставку" in text


def test_covered_question_keeps_grounded_text_around_a_hedging_sentence():
    raw = "Лектор не затрагивал недостатки модели подробно. " + FILLER
    text, covered = judge_answer(raw)
    assert covered and "не затрагивал" not in text
    assert judge_answer(raw, strict=True)[1] is False

    middle = FILLER + "Однако в лекции не рассматривались детали протокола. " + FILLER
    text, covered = judge_answer(middle, strict=True)
    assert covered and "не рассматривались" not in text


def test_networking_words_like_fragment_are_not_mistaken_for_refusals():
    raw = "На транспортном уровне данные делятся на фрагменты, которые не содержат служебных заголовков. " + FILLER
    text, covered = judge_answer(raw, strict=True)
    assert covered and "фрагменты" in text


def test_meta_sentence_inside_grounded_answer_is_removed():
    raw = (
        FILLER
        + "\n\nВажно отметить, что хотя в лекции непосредственно не обсуждались детали этих подходов, они важны. "
        + FILLER
    )
    text, covered = judge_answer(raw)
    assert covered
    assert "не обсуждались" not in text
    assert text.count("Лектор подробно") == 24


def test_heading_that_repeats_the_question_is_dropped():
    raw = "Достоинства и недостатки использования модели OSI\n\n" + FILLER
    result = clean_llm_output(raw, "Достоинства и недостатки использования модели OSI (модели открытых систем)?")
    assert result.startswith("Лектор подробно")


def test_generic_heading_is_dropped_and_real_heading_kept_as_sentence():
    result = clean_llm_output("## Введение\nТекст.\n#### Уровни модели\nЕщё текст.")
    assert "#" not in result
    assert "Введение" not in result
    assert "Уровни модели." in result


def test_list_items_are_folded_into_one_paragraph():
    raw = "Модель включает уровни:\n- физический\n- канальный\n- сетевой"
    assert clean_llm_output(raw) == "Модель включает уровни: физический; канальный; сетевой"


def test_underscore_emphasis_removed_but_identifiers_kept():
    assert clean_llm_output("Флаг __SYN__ и поле tcp_flags.") == "Флаг SYN и поле tcp_flags."


def test_markdown_table_rows_become_text():
    result = clean_llm_output("| Протокол | Надёжность |\n|---|---|\n| TCP | есть |")
    assert "|" not in result
    assert "TCP — есть" in result


def test_write_answer_returns_none_for_not_covered():
    fake = lambda s, u, m: NOT_COVERED_PHRASE
    assert write_answer("Вопрос?", "контекст", "Дисциплина", generate_fn=fake) is None


def test_write_answer_sends_question_and_context_and_retries_once_when_short():
    calls = []

    def fake(system_prompt, user_prompt, max_tokens):
        calls.append((system_prompt, user_prompt))
        if len(calls) == 1:
            return "Короткий, но содержательный ответ по теме лекции. " * 8
        return FILLER * 4

    text = write_answer("Что такое OSI?", "ФРАГМЕНТЛЕКЦИИ", "Дисциплина", target_words=500, generate_fn=fake, notes="УКАЗАНИЕ")

    assert len(calls) == 2
    assert "ФРАГМЕНТЛЕКЦИИ" in calls[0][1] and "Что такое OSI?" in calls[0][1]
    assert NOT_COVERED_PHRASE in calls[0][1]
    assert "УКАЗАНИЕ" in calls[0][0] and "ПРАВИЛА ФОРМАТИРОВАНИЯ" in calls[0][0]
    assert "Предыдущий ответ получился слишком коротким" in calls[1][1]
    assert text.count("Лектор подробно") == 48


def _make_material(tmp_path, material_id, title, files):
    folder = tmp_path / material_id
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    db.create_material(material_id, title, str(folder))
    for name in files:
        db.update_material(material_id, add_file={"name": name, "kind": "transcript"})


def test_pool_transcripts_orders_parts_naturally_and_drops_duplicates(tmp_path):
    _make_material(
        tmp_path, "c", "2026-09-14_Лекция_1_4_TCP",
        {"video.md": "# Транскрибация видеолекции\n\n2026-09-14_Лекция_1_4_TCP\n\n**00:00 – 01:30**\n\nЧетвёртая часть."},
    )
    _make_material(tmp_path, "a", "2026-09-14_Лекция_1_1_Теория", {"video.md": "Первая часть.", "video_2.md": "Первая   часть."})
    _make_material(tmp_path, "b", "2026-09-14_Лекция_00_Общая", {"video.txt": "Вводная часть."})

    parts = writer.pool_transcripts([db.get_material(m) for m in ("c", "a", "b")])

    assert parts == ["Вводная часть.", "Первая часть.", "Четвёртая часть."]


def _analysis():
    return writer.analyze_lecture(
        ["текст лекции"], ["Вопрос?"], lambda s, u, m: "1 | Лектор объяснил суть вопроса.", lambda *args: None
    )


def test_disclosure_describes_full_cycle_with_api_route(monkeypatch):
    monkeypatch.setattr(db, "get_setting", lambda key: "key" if key == "gigachat_api_key" else None)
    text = writer.build_disclosure([_analysis()], engine_calls=7, split=True, notes=["ОСОБОЕ УКАЗАНИЕ"])

    for marker in [
        "ИИ-сервис", "GigaChat-Pro", "Этап 1", "ffmpeg", "Whisper", "локально", "через GigaChat API",
        "Разделение на основной и дополнительный", "Системный промт", "Пользовательский промт",
        "Промт извлечения тезисов", "knowledge_extractions", "собирается программно из тезисов",
        "ошибочно сообщил, что материала нет",
        "выполнено запросов к модели: 7", "chat/completions", "api/v2/oauth",
        "Промт разделения", "одну букву", "Предыдущий ответ получился слишком коротким",
        "Номер наиболее подходящей лекции", "«ОСОБОЕ УКАЗАНИЕ»", "задан в программе",
        "СТИЛЬ ИЗЛОЖЕНИЯ", "дословно повторённый фрагмент",
    ]:
        assert marker in text, marker
    assert "\\n" not in text
    assert all("\n" not in paragraph for paragraph in text.split("\n\n"))


def test_disclosure_for_local_engine_mentions_no_cloud_api(monkeypatch):
    monkeypatch.setattr(db, "get_setting", lambda key: None)
    text = writer.build_disclosure([_analysis()], engine_calls=3)

    assert "llama-cpp-python" in text
    assert "GigaChat API" not in text
    assert "Номер наиболее подходящей лекции" not in text
    assert "УКАЗАНИЯ ПРЕПОДАВАТЕЛЯ: …" not in text
    assert "Разделение на основной и дополнительный" not in text
