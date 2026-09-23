from pathlib import Path

import pytest

from app.assignment import AssignmentRules, LectureQuestions
from app.reference.checker import check_reference
from app.reference.pdf_writer import ReferenceContent, TitlePageInfo, build_reference_pdf

TITLE = TitlePageInfo(
    university_header="МИНИСТЕРСТВО НАУКИ",
    department="КАФЕДРА № 00",
    teacher_position="доцент, к.т.н.",
    teacher_name="Сидоров С.С.",
    lecture_number=4,
    lecture_title="Коммутация, Маршрутизация, СКС",
    discipline="Введение в ИТ",
    student_group="2395",
    student_name="Иванов И.И.",
    city_year="Санкт-Петербург, 2025",
)

LECTURE = LectureQuestions(
    lecture_number=4,
    topic="Коммутация",
    questions=[
        "Какие методы коммутации бывают?",
        "Какие алгоритмы маршрутизации используются в компьютерных сетях?",
        "Что такое СКС? И для чего их используют?",
    ],
)

_LONG_PARAGRAPH = (
    "Существует несколько основных методов коммутации данных в компьютерных "
    "сетях, которые применяются в зависимости от типа передаваемого трафика. "
) * 150


def _build(tmp_path: Path, sections, filename="ЛК4_Иванов_2395.pdf") -> Path:
    content = ReferenceContent(title_page=TITLE, sections=sections, sources=["Видеолекция №4, 2025."])
    output = tmp_path / filename
    build_reference_pdf(output, content)
    return output


def test_well_formed_reference_passes(tmp_path):
    sections = [(q, _LONG_PARAGRAPH) for q in LECTURE.questions]
    pdf_path = _build(tmp_path, sections)

    report = check_reference(pdf_path, pdf_path.name, AssignmentRules(), LECTURE)

    assert report.passed, report.issues
    assert report.details["has_toc"]
    assert report.details["has_sources"]
    assert report.details["questions_matched"] == 3


def test_too_short_reference_fails_page_count(tmp_path):
    sections = [(q, "Короткий ответ.") for q in LECTURE.questions]
    pdf_path = _build(tmp_path, sections)

    report = check_reference(pdf_path, pdf_path.name, AssignmentRules(), LECTURE)

    assert not report.passed
    assert any("страниц" in issue for issue in report.issues)


def test_too_few_questions_fails(tmp_path):
    sections = [(LECTURE.questions[0], _LONG_PARAGRAPH)]
    pdf_path = _build(tmp_path, sections)

    report = check_reference(pdf_path, pdf_path.name, AssignmentRules(), LECTURE)

    assert not report.passed
    assert any("вопрос" in issue for issue in report.issues)


def test_wrong_filename_pattern_fails(tmp_path):
    sections = [(q, _LONG_PARAGRAPH) for q in LECTURE.questions]
    pdf_path = _build(tmp_path, sections, filename="my_essay.pdf")

    report = check_reference(pdf_path, pdf_path.name, AssignmentRules(), LECTURE)

    assert not report.passed
    assert any("шаблону" in issue for issue in report.issues)


def test_non_pdf_extension_fails(tmp_path):
    sections = [(q, _LONG_PARAGRAPH) for q in LECTURE.questions]
    pdf_path = _build(tmp_path, sections, filename="ЛК4_Иванов_2395.pdf")
    docx_path = pdf_path.with_suffix(".docx")
    docx_path.write_bytes(pdf_path.read_bytes())

    report = check_reference(docx_path, "ЛК4_Иванов_2395.docx", AssignmentRules(), LECTURE)

    assert not report.passed
    assert any("формате" in issue for issue in report.issues)


def _page_texts(pdf_path):
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]


def test_pages_numbered_from_page_two_and_toc_has_leaders_and_sources(tmp_path):
    import re

    sections = [(q, _LONG_PARAGRAPH) for q in LECTURE.questions]
    texts = _page_texts(_build(tmp_path, sections))

    assert not re.search(r"(^|\n)\s*1\s*($|\n)", texts[0])
    assert "Санкт-Петербург" in texts[0] and "2025" in texts[0]
    for number, text in enumerate(texts[1:], start=2):
        assert re.search(rf"(^|\n)\s*{number}\s*($|\n)", text), number
    toc = texts[1]
    assert "СОДЕРЖАНИЕ" in toc and "СПИСОК ИСТОЧНИКОВ" in toc
    assert re.search(r"(\.\s*){10,}\d", toc)


def test_special_characters_in_plain_text_do_not_break_layout(tmp_path):
    sections = [("Что значит <5 мс & TCP>?", "Задержка <5 мс & потери > 1%. " * 200)]
    content = ReferenceContent(
        title_page=TITLE, sections=sections, sources=["Источник <1> & ко\n\nВторой абзац источника."]
    )
    pdf_path = tmp_path / "ЛК4_Иванов_2395.pdf"
    build_reference_pdf(pdf_path, content)
    text = "\n".join(_page_texts(pdf_path))

    assert "<5 мс & TCP>" in text
    assert "1. Источник <1> & ко" in text
    assert "Второй абзац источника." in text


def test_content_page_estimate_ignores_multi_page_sources(tmp_path):
    sections = [(q, _LONG_PARAGRAPH) for q in LECTURE.questions]
    reports = []
    for name, sources in (
        ("short", ["Видеолекция №4, 2025."]),
        ("long", ["Видеолекция №4, 2025.", "Описание технологии подготовки текста. " * 900]),
    ):
        folder = tmp_path / name
        folder.mkdir()
        pdf_path = folder / "ЛК4_Иванов_2395.pdf"
        build_reference_pdf(pdf_path, ReferenceContent(title_page=TITLE, sections=sections, sources=sources))
        reports.append(check_reference(pdf_path, pdf_path.name, AssignmentRules(), LECTURE))

    short, long = reports
    assert long.details["total_pages"] >= short.details["total_pages"] + 2
    assert abs(long.details["approx_content_pages"] - short.details["approx_content_pages"]) <= 0.2


def test_unreadable_file_fails_gracefully(tmp_path):
    fake_pdf = tmp_path / "ЛК4_Иванов_2395.pdf"
    fake_pdf.write_bytes(b"not actually a pdf")

    report = check_reference(fake_pdf, fake_pdf.name, AssignmentRules(), LECTURE)

    assert not report.passed
    assert any("открыть" in issue for issue in report.issues)
