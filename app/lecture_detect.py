"""Figures out which lecture number (1, 2, 3, ...) a lesson folder is about,
so the right question list from the parsed assignment can be used for it.

Two strategies, tried in order:
1. A number literally in the folder/file name — "Лекция 4", "ЛК4",
   "Занятие 4", "Lecture 4" — the strong, reliable signal when a teacher
   names things consistently.
2. Falling back to keyword overlap between the lecture's topic description
   (parsed from the assignment doc) and the video's transcript — useful
   when the folder is just named after a date or has no lecture number at
   all.

Either way the result is a best guess: the web UI lets a person confirm or
override it (see db.update_material(..., lecture_number_source="manual")).
"""
import re
from typing import Callable, Optional

from app.assignment import LectureQuestions

# (system_prompt, user_prompt, max_tokens) -> raw text response, same shape
# as app.reference.writer.GenerateFn — lets detect_lecture_number take
# whichever LLM call the caller already has (GigaChat, local model, or a
# test stub) without this module needing to know which.
LlmClassifyFn = Callable[[str, str, int], str]

TRANSCRIPT_EXCERPT_CHARS_FOR_CLASSIFY = 3000

_TITLE_NUMBER_RE = re.compile(
    r"(?:Лекци[яию]|Занятие|ЛК|Lecture)\s*[№#]?\s*(\d{1,2})", re.IGNORECASE
)

_STOPWORDS = {
    "тема", "темы", "лекция", "лекции", "занятие", "дополнительная", "введение",
}


def detect_from_title(title: str) -> Optional[int]:
    if match := _TITLE_NUMBER_RE.search(title):
        return int(match.group(1))
    return None


def _topic_keywords(topic: str) -> set[str]:
    # >=3 chars (not >=4) so short but distinctive acronyms like "VPN",
    # "DNS", "IP6" survive — topics like "VPN, IP6, DNS" would otherwise
    # contribute zero keywords and never match anything.
    words = re.findall(r"[а-яёa-z0-9]{3,}", topic.lower())
    return {w for w in words if w not in _STOPWORDS}


def detect_from_transcript(transcript_text: str, lectures: list[LectureQuestions]) -> Optional[int]:
    if not transcript_text or not lectures:
        return None

    transcript_lower = transcript_text.lower()
    best_lecture: Optional[int] = None
    best_score = 0

    for lecture in lectures:
        keywords = _topic_keywords(lecture.topic)
        score = sum(1 for word in keywords if word in transcript_lower)
        if score > best_score:
            best_score = score
            best_lecture = lecture.lecture_number

    return best_lecture if best_score > 0 else None


DETECT_SYSTEM_PROMPT = (
    "Ты помогаешь определить, к какой лекции из списка относится транскрипт видеозаписи. "
    "Ответь СТРОГО одним числом — номером наиболее подходящей лекции из списка, без пояснений. "
    "Если ни одна лекция явно не подходит, ответь 0."
)
DETECT_USER_PROMPT_TEMPLATE = (
    "{notes_block}Список лекций:\n{catalog}\n\n"
    "Транскрипт видео (начало):\n{transcript}\n\n"
    "Номер наиболее подходящей лекции:"
)


def detect_from_llm(
    transcript_text: str,
    lectures: list[LectureQuestions],
    teacher_notes: str,
    llm_classify: LlmClassifyFn,
) -> Optional[int]:
    """Ask an LLM to pick which lecture this transcript belongs to — a much
    better signal than keyword overlap when a teacher's general
    instructions explain the mapping in prose (e.g. "лекция переведена в
    дистанционный формат, посмотрите видео и определите, к какому из двух
    заданий — основному или дополнительному — оно относится") rather than
    the video being cleanly named "Лекция N"."""
    if not transcript_text or not lectures:
        return None

    valid_numbers = {lecture.lecture_number for lecture in lectures}
    catalog = "\n".join(
        f"{lecture.lecture_number}. {lecture.topic}" + (f" ({lecture.notes})" if lecture.notes else "")
        for lecture in lectures
    )
    notes_block = f"Указания преподавателя: {teacher_notes}\n\n" if teacher_notes else ""
    user_prompt = DETECT_USER_PROMPT_TEMPLATE.format(
        notes_block=notes_block,
        catalog=catalog,
        transcript=transcript_text[:TRANSCRIPT_EXCERPT_CHARS_FOR_CLASSIFY],
    )

    try:
        response = llm_classify(DETECT_SYSTEM_PROMPT, user_prompt, 20)
    except Exception:
        return None

    match = re.search(r"\d+", response)
    if not match:
        return None
    number = int(match.group())
    return number if number in valid_numbers else None


def detect_lecture_number(
    title: str,
    transcript_text: str,
    lectures: list[LectureQuestions],
    teacher_notes: str = "",
    llm_classify: Optional[LlmClassifyFn] = None,
) -> Optional[int]:
    if number := detect_from_title(title):
        return number
    if llm_classify and (number := detect_from_llm(transcript_text, lectures, teacher_notes, llm_classify)):
        return number
    return detect_from_transcript(transcript_text, lectures)
