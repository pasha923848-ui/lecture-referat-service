from app.assignment import parse_assignment, parse_lecture_questions, parse_rules

# A trimmed-down excerpt mirroring the real structure this parser is
# written against (headers, numbered questions with either "1)" or "1."
# style, "====" section dividers, multi-line topic descriptions, and a
# trailing title-page template that must not bleed into the last question).
SAMPLE_TEXT = """
1. Требования к отчету
Написать реферат объемом не менее 5 страниц 12 п. TimesRoman,(5 стр. чистого
текста, + (титульник, содержание, список источников)).

Условия и причины непринятия Реферата
1) Если реферат выполнен с нарушением требований к оформлению или не формат pdf.
2) Если в Реферате рассмотрено менее 3-х вопросов.

Требования к выбору вопросов
В реферате рассмотреть не менее 3-х вопросов (можно все) из предлагаемого списка.

2. ВОПРОСЫ ДЛЯ РЕФЕРАТОВ
===================== ========================================
Вопросы для реферата – Лекция 1, Темы: Основы теории информации.
Модель OSI.
1) В чем отличие энтропийного оценивания информации и алфавитного?
2) Достоинства и недостатки использования модели OSI?
============================ ===========================
Вопросы для реферата – Лекция 2. Темы: Сетевые технологии LAN
1. Чем Hub отличается от Switch?
2. Какие виды VLAN существуют?
3. Технологии STP: возможности, преимущества, недостатки.
==================== ===================================
Вопросы для реферата – Лекция 3. VPN, IP6, DNS
1. Структура IP адресации.
2. Что такое DarkNet?

МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
РЕФЕРАТ
к лекции № __« Название лекции»
"""


def test_parse_rules_extracts_page_count_font_and_min_questions():
    rules = parse_rules(SAMPLE_TEXT)
    assert rules.min_pages == 5
    assert rules.font_size == 12
    assert rules.min_questions == 3


def test_parse_lecture_questions_splits_all_lecture_blocks():
    lectures = parse_lecture_questions(SAMPLE_TEXT)
    assert [l.lecture_number for l in lectures] == [1, 2, 3]


def test_parse_lecture_questions_handles_both_numbering_styles():
    lectures = parse_lecture_questions(SAMPLE_TEXT)
    lecture1 = next(l for l in lectures if l.lecture_number == 1)
    lecture2 = next(l for l in lectures if l.lecture_number == 2)

    assert lecture1.questions == [
        "В чем отличие энтропийного оценивания информации и алфавитного?",
        "Достоинства и недостатки использования модели OSI?",
    ]
    assert lecture2.questions == [
        "Чем Hub отличается от Switch?",
        "Какие виды VLAN существуют?",
        "Технологии STP: возможности, преимущества, недостатки.",
    ]


def test_parse_lecture_questions_joins_multiline_topic():
    lectures = parse_lecture_questions(SAMPLE_TEXT)
    lecture1 = next(l for l in lectures if l.lecture_number == 1)
    assert "Основы теории информации" in lecture1.topic
    assert "Модель OSI" in lecture1.topic


def test_last_lecture_last_question_excludes_title_page_boilerplate():
    lectures = parse_lecture_questions(SAMPLE_TEXT)
    lecture3 = next(l for l in lectures if l.lecture_number == 3)
    assert lecture3.questions[-1] == "Что такое DarkNet?"
    assert "МИНИСТЕРСТВО" not in lecture3.questions[-1]


def test_parse_assignment_returns_both_rules_and_lectures():
    rules, lectures = parse_assignment(SAMPLE_TEXT)
    assert rules.min_pages == 5
    assert len(lectures) == 3


def test_question_immediately_after_number_with_no_space_is_not_merged():
    # A hand-typed list sometimes has "11.Text" with no space after the
    # punctuation instead of "11. Text" — the boundary regex must recognize
    # that as a new question too, or it gets silently swallowed into the
    # end of the PREVIOUS question's text (found against a real assignment
    # document where lecture 4 lost 15 of its 25 questions this way).
    text = """
Вопросы для реферата – Лекция 9. Тема теста
1. Первый вопрос с пробелом?
2. Второй вопрос с пробелом?
3.Третий вопрос без пробела?
4.Четвёртый вопрос без пробела?
"""
    lectures = parse_lecture_questions(text)
    lecture9 = next(l for l in lectures if l.lecture_number == 9)
    assert lecture9.questions == [
        "Первый вопрос с пробелом?",
        "Второй вопрос с пробелом?",
        "Третий вопрос без пробела?",
        "Четвёртый вопрос без пробела?",
    ]


def test_lecture_notes_default_empty_when_document_has_no_preface():
    # This document's real-world structure (see the module docstring) never
    # leaves room for a preface between a lecture's topic and its first
    # numbered question — the topic capture is lazy up to that exact
    # boundary, so anything a teacher writes there ends up folded into
    # `topic` itself, not `notes`. `notes` exists for documents that do
    # have a clearer separation; asserting it's empty here pins down that
    # this sample format doesn't trigger it, rather than silently no-op'ing.
    lectures = parse_lecture_questions(SAMPLE_TEXT)
    assert all(l.notes == "" for l in lectures)
