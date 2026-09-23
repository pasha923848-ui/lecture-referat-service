"""Parses a teacher's "Задание на реферат" document into structured data:
the formatting rules (page count, font, filename pattern, ...) and, per
lecture number, the list of candidate questions a reference on that lecture
must address at least a few of.

Written against the real structure of one such document (numbered sections
"Вопросы для реферата – Лекция N. Темы: ..." followed by numbered
questions, separated by "====" banners) but tries to stay a bit forgiving
about exact punctuation/whitespace, since these are hand-typed Word/PDF
documents that vary from teacher to teacher.
"""
import re
from dataclasses import dataclass, field


@dataclass
class AssignmentRules:
    min_pages: int = 5
    font_name: str = "Times New Roman"
    font_size: int = 12
    min_questions: int = 3
    file_format: str = "pdf"
    filename_pattern: str = r"^ЛК\d+_[А-Яа-яЁёA-Za-z]+_\d+\.pdf$"


@dataclass
class LectureQuestions:
    lecture_number: int
    topic: str
    questions: list[str] = field(default_factory=list)
    # Free-text instructions a teacher sometimes writes between the lecture
    # header and the first numbered question — e.g. "по этой лекции реферат
    # не пишем, используйте материал лекции 3" or "объединить с лекцией 5".
    # Passed along to the LLM as extra guidance (see
    # app.reference.writer.generate_section) instead of being discarded.
    notes: str = ""
    # Set when the topic line itself names a second, "дополнительная"
    # sub-topic for this same lecture number (a real document had: "Темы:
    # Основы теории информации. Модель OSI. Лекция 1 (дополнительная). Тема:
    # Протоколы стека TCP.") — questions covering two unrelated subjects
    # under one lecture number are meant to become TWO separate рефераты,
    # not one mixed document (see app.reference.writer.split_questions_by_topic).
    additional_topic: str = ""


_LECTURE_HEADER_RE = re.compile(
    r"Вопросы\s+для\s+реферата\s*[–—-]\s*Лекция\s*(\d+)[.,]?\s*(.*?)(?=\n\s*\d{1,2}[.)]\s*\S|\n\s*={5,})",
    re.IGNORECASE | re.DOTALL,
)

# The boundary lookahead uses \s* (zero or more), matching the same
# optional-whitespace-after-the-punctuation the capture group itself
# allows — a hand-typed list often has some items as "11. Text" and others
# as "11.Text" with no space, and the two patterns must agree or a
# no-space item silently gets swallowed into the PREVIOUS question's text
# instead of starting a new one (the lookahead would keep failing to find
# a boundary and the lazy .+? would just keep consuming).
_QUESTION_ITEM_RE = re.compile(
    r"\n\s*\d{1,2}[.)]\s*(.+?)(?=\n\s*\d{1,2}[.)]\s*\S|\Z)",
    re.DOTALL,
)

_SECTION_DIVIDER_RE = re.compile(r"\n\s*={5,}")

# Matches a "дополнительная" sub-topic named inline within the same topic
# line as the lecture's main topic, e.g. "...Модель OSI. Лекция 1
# (дополнительная). Тема: Протоколы стека TCP." — group(1) is everything
# from "Тема:" onward (the additional sub-topic's own description).
_ADDITIONAL_TOPIC_RE = re.compile(
    r"Лекция\s*\d+\s*\(дополнительная\)\.?\s*Тема:?\s*(.+)", re.IGNORECASE | re.DOTALL
)

# Russian university title pages conventionally open with this boilerplate;
# cutting the search text there keeps it from bleeding into the last
# lecture's last question when there's no other clear boundary.
_TITLE_PAGE_BOILERPLATE_RE = re.compile(
    r"МИНИСТЕРСТВО\s+НАУКИ|ФЕДЕРАЛЬНОЕ\s+ГОСУДАРСТВЕННОЕ", re.IGNORECASE
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_lecture_questions(full_text: str) -> list[LectureQuestions]:
    """Split the document into per-lecture blocks and pull out that
    lecture's topic line and numbered questions."""
    if boilerplate := _TITLE_PAGE_BOILERPLATE_RE.search(full_text):
        full_text = full_text[: boilerplate.start()]

    headers = list(_LECTURE_HEADER_RE.finditer(full_text))
    results: list[LectureQuestions] = []

    for i, match in enumerate(headers):
        lecture_number = int(match.group(1))
        topic = _clean(match.group(2))

        additional_topic = ""
        if additional_match := _ADDITIONAL_TOPIC_RE.search(topic):
            additional_topic = _clean(additional_match.group(1))
            topic = _clean(topic[: additional_match.start()])

        block_start = match.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        block_text = full_text[block_start:block_end]

        # Cut off the "====" banner that separates lecture sections so it
        # doesn't get swallowed into the last question's text.
        if divider := _SECTION_DIVIDER_RE.search(block_text):
            block_text = block_text[: divider.start()]

        question_matches = list(_QUESTION_ITEM_RE.finditer(block_text))
        questions = [q for q in (_clean(m.group(1)) for m in question_matches) if q]

        # Anything between the header and the first numbered question is
        # free-text preface/instructions rather than a question.
        notes_end = question_matches[0].start() if question_matches else len(block_text)
        notes = _clean(block_text[:notes_end])

        results.append(
            LectureQuestions(
                lecture_number=lecture_number,
                topic=topic,
                questions=questions,
                notes=notes,
                additional_topic=additional_topic,
            )
        )

    return results


# Rules are mostly stable across versions of this kind of document, so we
# only bother extracting the couple of numbers that plausibly vary
# (page count, font size) — everything else falls back to sane defaults.
_MIN_PAGES_RE = re.compile(r"не\s+менее\s+(\d+)\s+страниц", re.IGNORECASE)
_FONT_SIZE_RE = re.compile(r"(\d{1,2})\s*п\.?\s*(?:Times\s*Roman|TimesRoman)", re.IGNORECASE)
_MIN_QUESTIONS_RE = re.compile(r"не\s+менее\s+(\d)[\s-]*х?\s+вопрос", re.IGNORECASE)


def parse_rules(full_text: str) -> AssignmentRules:
    rules = AssignmentRules()

    if match := _MIN_PAGES_RE.search(full_text):
        rules.min_pages = int(match.group(1))
    if match := _FONT_SIZE_RE.search(full_text):
        rules.font_size = int(match.group(1))
    if match := _MIN_QUESTIONS_RE.search(full_text):
        rules.min_questions = int(match.group(1))

    return rules


def parse_assignment(full_text: str) -> tuple[AssignmentRules, list[LectureQuestions]]:
    return parse_rules(full_text), parse_lecture_questions(full_text)
