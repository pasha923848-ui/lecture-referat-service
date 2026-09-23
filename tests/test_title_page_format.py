"""Regression test against a real accepted реферат's title page structure
(образец принятого реферата) — ОЦЕНКА РЕФЕРАТА/РУКОВОДИТЕЛЬ instead of
ЗАЩИЩЕН С ОЦЕНКОЙ/ПРЕПОДАВАТЕЛЬ, a bare descriptive title instead of
"к лекции №N", and a РЕФЕРАТ ВЫПОЛНИЛ heading before the student line."""
from pypdf import PdfReader

from app.reference.pdf_writer import ReferenceContent, TitlePageInfo, build_reference_pdf

TITLE = TitlePageInfo(
    university_header="Университет",
    department="КАФЕДРА 24",
    teacher_position="доцент, к.т.н.",
    teacher_name="Сидоров С.С.",
    lecture_number=1,
    lecture_title="Основы теории информации и сетевая архитектура. Часть 1",
    discipline="Введение в информационные технологии",
    student_group="2395",
    student_name="Петрова А.А.",
    city_year="Санкт-Петербург, 2026",
)


def _build(tmp_path):
    content = ReferenceContent(
        title_page=TITLE,
        sections=[("1. Вопрос?", "Ответ на вопрос номер один. " * 20)],
        sources=["Источник 1"],
    )
    pdf_path = tmp_path / "test.pdf"
    build_reference_pdf(pdf_path, content)
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    return text


def test_title_page_uses_otsenka_referata_not_zashchishchen(tmp_path):
    text = _build(tmp_path)
    assert "ОЦЕНКА РЕФЕРАТА" in text
    assert "ЗАЩИЩЕН С ОЦЕНКОЙ" not in text


def test_title_page_uses_rukovoditel_not_prepodavatel(tmp_path):
    text = _build(tmp_path)
    assert "РУКОВОДИТЕЛЬ" in text
    assert "ПРЕПОДАВАТЕЛЬ" not in text


def test_title_page_shows_descriptive_title_not_lecture_number_phrasing(tmp_path):
    text = _build(tmp_path)
    assert "Основы теории информации и сетевая архитектура. Часть 1" in text
    assert "к лекции №" not in text


def test_title_page_has_referat_vypolnil_heading(tmp_path):
    text = _build(tmp_path)
    assert "РЕФЕРАТ ВЫПОЛНИЛ" in text


def test_title_page_includes_student_group_and_name(tmp_path):
    text = _build(tmp_path)
    assert "2395" in text
    assert "Петрова А.А." in text


def _y_positions(pdf_path, needle):
    positions = []

    def visitor(text, cm, tm, font_dict, font_size):
        if needle in text:
            positions.append(tm[4] * cm[1] + tm[5] * cm[3] + cm[5])

    PdfReader(str(pdf_path)).pages[0].extract_text(visitor_text=visitor)
    return positions


def test_student_block_sits_low_on_the_title_page_above_city(tmp_path):
    from dataclasses import replace

    for name, info in (
        ("short", TITLE),
        ("long", replace(TITLE, lecture_label="к лекциям № 1, 2 и 3<br/>«" + "Очень длинная тема лекции; " * 8 + "»")),
    ):
        pdf_path = tmp_path / f"{name}.pdf"
        build_reference_pdf(
            pdf_path, ReferenceContent(title_page=info, sections=[("1 Вопрос?", "Ответ. " * 50)], sources=["Источник"])
        )
        student = _y_positions(pdf_path, "РЕФЕРАТ ВЫПОЛНИЛ")
        city = _y_positions(pdf_path, "Санкт-Петербург")
        assert student, name
        assert city and city[0] < student[0] < 842 * 0.33, (name, student, city)


def test_title_page_includes_signature_captions(tmp_path):
    text = _build(tmp_path)
    assert "должность, уч. степень, звание" in text
    assert "инициалы, фамилия" in text
