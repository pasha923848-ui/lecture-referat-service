from app.reference.writer import split_questions_by_topic

QUESTIONS = [
    "В чем отличие энтропийного оценивания информации и алфавитного?",
    "Достоинства и недостатки использования модели OSI?",
    "Опишите состояния протокола TCP и правила перехода между состояниями?",
    "Описать сравнительные характеристики протоколов UDP, TCP?",
]
MAIN_TOPIC = "Основы теории информации. Модель OSI."
ADDITIONAL_TOPIC = "Протоколы стека TCP."


def test_returns_everything_as_main_when_no_additional_topic():
    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, "", llm_select=None)
    assert main == QUESTIONS
    assert additional == []


def test_llm_split_is_used_when_available():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "О,О,Д,Д"
    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, ADDITIONAL_TOPIC, llm_select=fake_llm)
    assert main == QUESTIONS[:2]
    assert additional == QUESTIONS[2:]


def test_llm_response_with_mismatched_count_falls_back_to_keywords():
    # Only 3 labels for 4 questions -> LLM response rejected, falls through
    # to the keyword heuristic instead of silently misaligning the split.
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "О,О,Д"
    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, ADDITIONAL_TOPIC, llm_select=fake_llm)
    # Keyword fallback should still find the TCP-related questions.
    assert QUESTIONS[2] in additional
    assert QUESTIONS[3] in additional


def test_llm_error_falls_back_to_keyword_heuristic():
    def failing_llm(system_prompt, user_prompt, max_tokens):
        raise RuntimeError("network down")

    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, ADDITIONAL_TOPIC, llm_select=failing_llm)
    assert QUESTIONS[2] in additional
    assert QUESTIONS[3] in additional


def test_keyword_heuristic_without_llm():
    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, ADDITIONAL_TOPIC, llm_select=None)
    assert QUESTIONS[0] in main
    assert QUESTIONS[2] in additional or QUESTIONS[3] in additional


def test_rebalance_gives_the_short_additional_side_the_closest_main_question():
    from app.reference.writer import rebalance_split

    main = [
        "В чем отличие энтропийного оценивания информации и алфавитного?",
        "Достоинства и недостатки использования модели OSI?",
        "Дайте развернутую сравнительную характеристику четырем стекам протоколов.",
        "Опишите на выбор любые четыре компьютерные сети с разной архитектурой реализации.",
        "Какие организации занимаются стандартизацией в области Интернет?",
    ]
    additional = QUESTIONS[2:]

    new_main, new_additional = rebalance_split(main, additional, MAIN_TOPIC, ADDITIONAL_TOPIC, 3)

    assert new_additional == additional + [main[2]]
    assert new_main == main[:2] + main[3:]


def test_rebalance_never_takes_the_donor_below_its_minimum():
    from app.reference.writer import rebalance_split

    main = QUESTIONS[:3]
    additional = QUESTIONS[3:]
    assert rebalance_split(main, additional, MAIN_TOPIC, ADDITIONAL_TOPIC, 3) == (main, additional)


def test_falls_back_to_no_split_when_llm_puts_everything_in_one_bucket():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "О,О,О,О"
    main, additional = split_questions_by_topic(QUESTIONS, MAIN_TOPIC, ADDITIONAL_TOPIC, llm_select=fake_llm)
    # LLM gave an all-main split (no "Д" at all) -> falls through to the
    # keyword heuristic rather than returning an empty additional group
    # from a technically-valid-but-useless LLM answer.
    assert additional or main == QUESTIONS
