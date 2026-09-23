from app.reference import retrieval
from app.reference.retrieval import (
    build_context,
    keyword_score,
    map_questions_to_chunks,
    parse_facts_response,
    split_into_chunks,
)

SENTENCES = [f"Предложение номер {i} рассказывает о протоколе и уровне {i % 7}." for i in range(400)]
TEXT = " ".join(SENTENCES[:200]) + "\n\n" + " ".join(SENTENCES[200:])


def test_short_text_is_a_single_chunk():
    assert split_into_chunks("Короткий текст.", 1000, 100) == ["Короткий текст."]


def test_chunks_respect_size_and_keep_every_sentence():
    chunks = split_into_chunks(TEXT, 2000, 300)
    assert len(chunks) > 5
    assert all(len(c) <= 2000 for c in chunks)
    joined = "\n".join(chunks)
    assert all(sentence in joined for sentence in SENTENCES)


def test_consecutive_chunks_overlap_without_cutting_words():
    chunks = split_into_chunks(TEXT, 2000, 300)
    words = set(TEXT.split())
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split()[0] in words and previous.split()[-1] in words
        assert current[:40] in previous


def test_overlap_exists_even_when_paragraphs_are_longer_than_the_overlap():
    paragraphs = [
        " ".join(f"Абзац {j}, фраза {i} о передаче данных по сети." for i in range(60)) for j in range(12)
    ]
    text = "\n\n".join(paragraphs)
    assert min(len(p) for p in paragraphs) > 800

    chunks = split_into_chunks(text, 5000, 800)

    assert len(chunks) > 3
    assert all(len(c) <= 5000 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert len(current) > 300 and current[:300] in previous


def test_parse_facts_response_variants():
    response = (
        "1 | Энтропия — мера неопределённости источника сообщений.\n"
        "- 2: Модель OSI содержит семь уровней взаимодействия.\n"
        "Вопрос 2) Стек TCP/IP охватывает четыре уровня модели.\n"
        "3. Слишком\n"
        "9 | Тезис к несуществующему вопросу, длинный текст.\n"
        "просто текст без номера, который надо пропустить"
    )
    assert parse_facts_response(response, 3) == {
        0: ["Энтропия — мера неопределённости источника сообщений."],
        1: ["Модель OSI содержит семь уровней взаимодействия.", "Стек TCP/IP охватывает четыре уровня модели."],
    }
    assert parse_facts_response("нет", 3) == {}


def test_keyword_score_tolerates_inflection():
    assert keyword_score("мы обсудили протокол стека tcp и его состояние", "Опишите состояния протоколов стека TCP") >= 0.75
    assert keyword_score("про погоду", "Опишите состояния протоколов стека TCP") == 0


def test_map_questions_collects_theses_per_question():
    chunks = ["про энтропию", "про модель osi", "про энтропию снова"]

    def fake(system_prompt, user_prompt, max_tokens):
        if "энтропию снова" in user_prompt:
            return "1 | Энтропия максимальна при равных вероятностях."
        if "энтропию" in user_prompt:
            return "1 | Энтропия — мера неопределённости сообщения.\n2 | Модель описывает передачу по уровням."
        return "2 | Модель OSI состоит из семи уровней."

    coverage = map_questions_to_chunks(chunks, ["Энтропия?", "Модель OSI?"], llm_select=fake)
    assert coverage.covered == {0: [0, 2], 1: [0, 1]}
    assert coverage.mentioned == {}
    assert coverage.facts_for(0) == [
        "Энтропия — мера неопределённости сообщения.",
        "Энтропия максимальна при равных вероятностях.",
    ]
    assert coverage.covered_questions() == [0, 1]
    assert coverage.llm_chunks == 3


def test_failed_chunk_falls_back_to_keywords_only_for_that_chunk():
    chunks = ["лектор рассказал про энтропию информации", "что-то ещё"]
    calls = {"n": 0}

    def flaky(system_prompt, user_prompt, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("timeout")
        return "нет"

    coverage = map_questions_to_chunks(chunks, ["Что такое энтропия информации?"], llm_select=flaky)
    assert coverage.covered == {}
    assert coverage.mentioned == {0: [0]}
    assert coverage.llm_chunks == 1


def test_knowledge_base_cache_is_used_instead_of_the_model():
    from app import db

    cache = retrieval.ExtractionCache("test-namespace", db.get_knowledge, db.set_knowledge)
    chunks = ["уникальный фрагмент про энтропию для проверки кэша базы знаний"]
    questions = ["Что такое энтропия?"]
    calls = []

    def fake(system_prompt, user_prompt, max_tokens):
        calls.append(user_prompt)
        return "1 | Энтропия — мера неопределённости источника."

    first = map_questions_to_chunks(chunks, questions, llm_select=fake, cache=cache)
    second = map_questions_to_chunks(chunks, questions, llm_select=fake, cache=cache)

    assert len(calls) == 1
    assert first.llm_chunks == 1 and second.cached_chunks == 1 and second.llm_chunks == 0
    assert second.facts_for(0) == first.facts_for(0) == ["Энтропия — мера неопределённости источника."]


def test_build_context_keeps_lecture_order_and_budget():
    chunks = ["А" * 100, "про tcp протокол " + "Б" * 90, "В" * 100, "tcp протокол tcp " + "Г" * 90]
    context = build_context(chunks, [3, 1], "протокол TCP", max_chars=1000)
    assert context.index("Б") < context.index("Г")

    limited = build_context(chunks, [0, 1, 3], "протокол TCP", max_chars=220)
    assert "А" not in limited
    assert len(limited) <= 220 + len("\n\n[…]\n\n")


def test_best_keyword_chunks_skips_unrelated():
    chunks = ["погода", "модель osi семь уровней"]
    assert retrieval.best_keyword_chunks(chunks, "Уровни модели OSI", 2) == [1]
