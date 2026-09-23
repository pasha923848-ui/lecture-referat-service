"""Rule-based verification of a реферат PDF against a teacher's assignment
requirements — works on any PDF (our own generated one, or a student's own
draft), no LLM involved. Necessarily heuristic for page-count and
question-coverage (a PDF doesn't self-report which of its pages are the
title/содержание/список источников, or which listed questions it answers),
but conservative and explained via `details` so a human can sanity-check it.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.assignment import AssignmentRules, LectureQuestions

# Reportlab's built-in "Times-Roman" has no Cyrillic; a compliant PDF for a
# Russian реферат realistically uses Liberation Serif (Times New Roman's
# metric-compatible, Cyrillic-capable free substitute) instead — both count.
_ACCEPTABLE_FONT_SUBSTRINGS = ("times", "liberationserif", "liberation serif", "ptserif", "pt serif")


@dataclass
class CheckReport:
    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def _extract_fonts(reader) -> set[str]:
    fonts: set[str] = set()
    for page in reader.pages:
        try:
            font_dict = page["/Resources"].get("/Font", {})
        except Exception:
            continue
        for font_ref in font_dict.values():
            base_font = str(font_ref.get_object().get("/BaseFont", ""))
            fonts.add(base_font.lower())
    return fonts


def _count_matched_questions(text_lower: str, questions: list[str]) -> int:
    matched = 0
    for question in questions:
        keywords = re.findall(r"[а-яёa-z0-9]{4,}", question.lower())
        if not keywords:
            continue
        hits = sum(1 for word in keywords if word in text_lower)
        if hits >= max(2, len(keywords) // 2):
            matched += 1
    return matched


_SOURCES_HEADING_RE = re.compile(r"список\s+(?:использованных\s+)?(?:источник|литератур)")


def _content_pages(page_texts: list[str]) -> Optional[float]:
    """Pages of text between СОДЕРЖАНИЕ and СПИСОК ИСТОЧНИКОВ, a partly
    filled page counted by its share of a typical full page. The contents
    and the sources can each span several pages, so "total minus three"
    overstates the text. None when the structure can't be located."""
    lowered = [t.lower() for t in page_texts]
    toc_index = next((i for i, t in enumerate(lowered) if "содержание" in t), None)
    sources_index = next(
        (i for i in range(len(lowered) - 1, -1, -1) if _SOURCES_HEADING_RE.search(lowered[i])), None
    )
    if toc_index is None or sources_index is None or sources_index <= toc_index:
        return None

    lengths = [len(re.sub(r"\s+", "", t)) for t in page_texts[toc_index + 1 : sources_index]]
    heading = _SOURCES_HEADING_RE.search(lowered[sources_index])
    lengths.append(len(re.sub(r"\s+", "", page_texts[sources_index][: heading.start()])))
    lengths = [n for n in lengths if n > 20]
    if not lengths:
        return 0.0
    full_page = sorted(lengths)[len(lengths) // 2]
    return round(sum(min(1.0, n / full_page) for n in lengths), 1)


def check_reference(
    pdf_path: Path,
    filename: str,
    rules: AssignmentRules,
    lecture: Optional[LectureQuestions],
) -> CheckReport:
    issues: list[str] = []
    details: dict = {}

    if pdf_path.suffix.lower() != f".{rules.file_format}":
        issues.append(f"Файл должен быть в формате {rules.file_format}, а не {pdf_path.suffix.lstrip('.')}")

    if not re.match(rules.filename_pattern, filename, re.IGNORECASE):
        issues.append(f"Имя файла «{filename}» не соответствует требуемому шаблону")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        issues.append(f"Не удалось открыть файл как PDF: {exc}")
        return CheckReport(passed=False, issues=issues, details=details)

    total_pages = len(reader.pages)
    details["total_pages"] = total_pages

    page_texts = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(page_texts)
    text_lower = full_text.lower()

    has_toc = "содержание" in text_lower
    has_sources = bool(re.search(r"список\s+источник", text_lower))
    details["has_toc"] = has_toc
    details["has_sources"] = has_sources
    if not has_toc:
        issues.append("Не найден раздел «Содержание»")
    if not has_sources:
        issues.append("Не найден раздел «Список источников»")

    # Illustrations are not supposed to count either, but can't be detected
    # from text alone.
    content_pages = _content_pages(page_texts)
    if content_pages is None:
        structural_pages = 1 + int(has_toc) + int(has_sources)
        content_pages = max(total_pages - structural_pages, 0)
    details["approx_content_pages"] = content_pages
    if content_pages < rules.min_pages:
        issues.append(
            f"Похоже, страниц чистого текста меньше {rules.min_pages} (примерно {content_pages})"
        )

    fonts_found = _extract_fonts(reader)
    details["fonts_found"] = sorted(fonts_found)
    if not any(any(sub in font for sub in _ACCEPTABLE_FONT_SUBSTRINGS) for font in fonts_found):
        issues.append(
            f"Не обнаружен шрифт {rules.font_name} (или его аналог) — найдено: "
            + (", ".join(sorted(fonts_found)) or "ничего")
        )

    if lecture is not None and lecture.questions:
        matched = _count_matched_questions(text_lower, lecture.questions)
        details["questions_matched"] = matched
        if matched < rules.min_questions:
            issues.append(
                f"Похоже, раскрыто менее {rules.min_questions} вопросов из списка "
                f"Лекции {lecture.lecture_number} (найдено совпадений: {matched})"
            )
    else:
        details["questions_matched"] = None

    return CheckReport(passed=not issues, issues=issues, details=details)
