from app.assignment import LectureQuestions
from app.lecture_detect import detect_from_llm, detect_from_title, detect_from_transcript, detect_lecture_number

LECTURES = [
    LectureQuestions(lecture_number=1, topic="Основы теории информации. Модель OSI.", questions=[]),
    LectureQuestions(lecture_number=5, topic="Виртуализация и OS Linux", questions=[]),
    LectureQuestions(lecture_number=8, topic="VPN, IP6, DNS", questions=[]),
]


def test_detect_from_title_various_formats():
    assert detect_from_title("Лекция 4 - Коммутация") == 4
    assert detect_from_title("ЛК7_Иванов_2395") == 7
    assert detect_from_title("Занятие 2 - Физика") == 2
    assert detect_from_title("Lecture 3") == 3


def test_detect_from_title_returns_none_without_a_number():
    assert detect_from_title("2026-09-13_видео") is None


def test_detect_from_transcript_picks_best_keyword_overlap():
    transcript = "Сегодня поговорим про гипервизор, виртуализацию и контейнеры в OS Linux"
    assert detect_from_transcript(transcript, LECTURES) == 5


def test_detect_from_transcript_returns_none_without_overlap():
    transcript = "Просто случайный текст ни о чём"
    assert detect_from_transcript(transcript, LECTURES) is None


def test_detect_lecture_number_prefers_title_over_transcript():
    # Title says lecture 8, but transcript content matches lecture 5's topic —
    # the explicit title number should win.
    transcript = "Сегодня поговорим про виртуализацию и OS Linux"
    assert detect_lecture_number("Лекция 8 - VPN", transcript, LECTURES) == 8


def test_detect_lecture_number_falls_back_to_transcript():
    transcript = "Обсудим DNS сервис, VPN туннелирование и IP6 адресацию"
    assert detect_lecture_number("2026-09-13_видео", transcript, LECTURES) == 8


def test_detect_from_llm_parses_valid_lecture_number():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "5"
    assert detect_from_llm("транскрипт про виртуализацию", LECTURES, "", fake_llm) == 5


def test_detect_from_llm_ignores_number_outside_valid_set():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "42"
    assert detect_from_llm("транскрипт", LECTURES, "", fake_llm) is None


def test_detect_from_llm_handles_zero_as_no_match():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "0"
    assert detect_from_llm("транскрипт", LECTURES, "", fake_llm) is None


def test_detect_from_llm_handles_non_numeric_response():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "не могу определить"
    assert detect_from_llm("транскрипт", LECTURES, "", fake_llm) is None


def test_detect_from_llm_swallows_llm_errors():
    def failing_llm(system_prompt, user_prompt, max_tokens):
        raise RuntimeError("network down")

    assert detect_from_llm("транскрипт", LECTURES, "", failing_llm) is None


def test_detect_from_llm_returns_none_without_transcript_or_lectures():
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "5"
    assert detect_from_llm("", LECTURES, "", fake_llm) is None
    assert detect_from_llm("транскрипт", [], "", fake_llm) is None


def test_detect_from_llm_includes_teacher_notes_in_prompt():
    captured = {}

    def fake_llm(system_prompt, user_prompt, max_tokens):
        captured["user_prompt"] = user_prompt
        return "1"

    detect_from_llm("транскрипт", LECTURES, "Дистанционный формат, определи сам", fake_llm)
    assert "Дистанционный формат, определи сам" in captured["user_prompt"]


def test_detect_lecture_number_uses_llm_before_keyword_fallback():
    # Transcript content would keyword-match lecture 5 ("виртуализацию"),
    # but the LLM classifier's answer (8) should win over that heuristic
    # when no title number is present.
    transcript = "Сегодня поговорим про виртуализацию и OS Linux"
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "8"
    result = detect_lecture_number("2026-09-13_видео", transcript, LECTURES, llm_classify=fake_llm)
    assert result == 8


def test_detect_lecture_number_falls_back_to_keywords_when_llm_finds_nothing():
    transcript = "Обсудим DNS сервис, VPN туннелирование и IP6 адресацию"
    fake_llm = lambda system_prompt, user_prompt, max_tokens: "0"
    result = detect_lecture_number("2026-09-13_видео", transcript, LECTURES, llm_classify=fake_llm)
    assert result == 8  # from the keyword heuristic, since the LLM found no match
