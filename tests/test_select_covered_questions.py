from app.reference.writer import engine_limits, select_covered_questions

QUESTIONS = [
    "Что такое энтропия информации?",
    "Достоинства и недостатки модели OSI?",
    "Опишите протоколы стека TCP и их состояния?",
    "Что такое DarkNet и Deep Web?",
]


def _facts(*numbers):
    return "\n".join(f"{n} | Лектор подробно объяснил этот вопрос на примере." for n in numbers) or "нет"


def test_questions_with_theses_are_selected():
    fake_llm = lambda s, u, m: _facts(1, 3)
    result = select_covered_questions("любой транскрипт", QUESTIONS, min_questions=1, llm_select=fake_llm)
    assert result == [QUESTIONS[0], QUESTIONS[2]]


def test_tops_up_with_best_keyword_match_first():
    fake_llm = lambda s, u, m: _facts(1)
    transcript = "Сначала поговорили про DarkNet и Deep Web."
    result = select_covered_questions(transcript, QUESTIONS, min_questions=2, llm_select=fake_llm)
    assert result == [QUESTIONS[0], QUESTIONS[3]]


def test_llm_error_falls_back_to_keywords_for_that_chunk():
    def failing_llm(s, u, m):
        raise RuntimeError("network down")

    transcript = "Сегодня говорили про DarkNet и Deep Web как теневые сегменты интернета."
    result = select_covered_questions(transcript, QUESTIONS, min_questions=1, llm_select=failing_llm)
    assert QUESTIONS[3] in result


def test_genuine_llm_zero_coverage_is_not_overridden_by_keyword_match():
    fake_llm = lambda s, u, m: "нет"
    transcript = "Обсудили модель OSI, её достоинства и недостатки подробно."
    assert select_covered_questions(transcript, QUESTIONS, min_questions=0, llm_select=fake_llm) == []


def test_never_returns_fewer_than_min_questions_even_with_empty_transcript():
    result = select_covered_questions("", QUESTIONS, min_questions=3, llm_select=None)
    assert result == QUESTIONS[:3]


def test_long_pooled_transcript_is_processed_chunk_by_chunk():
    padding = "Общие рассуждения о курсе. " * 2000
    transcript = "Сначала про энтропию. " + padding + "А в конце про протоколы TCP."
    assert len(transcript) > engine_limits().chunk_chars

    calls = []

    def fake_llm(system_prompt, user_prompt, max_tokens):
        calls.append(user_prompt)
        if "про энтропию" in user_prompt:
            return _facts(1)
        if "про протоколы TCP" in user_prompt:
            return _facts(3)
        return "нет"

    result = select_covered_questions(transcript, QUESTIONS, min_questions=1, llm_select=fake_llm)

    assert len(calls) > 2
    assert result == [QUESTIONS[0], QUESTIONS[2]]
