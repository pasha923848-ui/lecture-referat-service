"""Lays out a реферат as a PDF by ГОСТ 7.32-2017: title page (no page
number), СОДЕРЖАНИЕ with dot leaders and page numbers, numbered section
headings, justified 1.5-spaced body with a 1.25 cm paragraph indent,
СПИСОК ИСТОЧНИКОВ, page numbers bottom-centre from page 2.

Section, source and title-field texts are plain text and are escaped here;
`university_header` and `lecture_label` are reportlab mini-markup (<br/>,
&nbsp;) by contract.
"""
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# reportlab's built-in "Times-Roman" has no Cyrillic glyphs. Liberation
# Serif is metrically identical to Times New Roman and does, so it is
# registered under the name "TimesRoman" the assignment literally asks for.
DEFAULT_FONT_NAME = "TimesRoman"
_FONT_DIR = os.environ.get("REFERENCE_FONT_DIR", "/usr/share/fonts/truetype/liberation")
_FONT_FILES = {
    DEFAULT_FONT_NAME: "LiberationSerif-Regular.ttf",
    f"{DEFAULT_FONT_NAME}-Bold": "LiberationSerif-Bold.ttf",
    f"{DEFAULT_FONT_NAME}-Italic": "LiberationSerif-Italic.ttf",
    f"{DEFAULT_FONT_NAME}-BoldItalic": "LiberationSerif-BoldItalic.ttf",
}
_registered = False


def _ensure_font_registered() -> None:
    global _registered
    if _registered:
        return
    for font_name, filename in _FONT_FILES.items():
        path = Path(_FONT_DIR) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Cyrillic-capable serif font not found at {path}. Install it "
                "(Debian/Ubuntu: apt-get install fonts-liberation) or point "
                "REFERENCE_FONT_DIR at a directory with Liberation Serif."
            )
        pdfmetrics.registerFont(TTFont(font_name, str(path)))
    pdfmetrics.registerFontFamily(
        DEFAULT_FONT_NAME,
        normal=DEFAULT_FONT_NAME,
        bold=f"{DEFAULT_FONT_NAME}-Bold",
        italic=f"{DEFAULT_FONT_NAME}-Italic",
        boldItalic=f"{DEFAULT_FONT_NAME}-BoldItalic",
    )
    _registered = True


@dataclass
class TitlePageInfo:
    university_header: str
    department: str
    teacher_position: str
    teacher_name: str
    lecture_number: int
    lecture_title: str
    discipline: str
    student_group: str
    student_name: str
    city_year: str
    # Reportlab markup, set for a реферат covering several lectures; when
    # empty the (plain-text) lecture_title is shown instead.
    lecture_label: str = ""


@dataclass
class ReferenceContent:
    title_page: TitlePageInfo
    # (heading, body) or (heading, body, level): level 0 = section,
    # level 1 = subsection under a level-0 lecture group heading.
    sections: list[tuple] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


_TOC_LEVELS = {"SectionHeading": 0, "SubsectionHeading": 1, "StructuralHeading": 0}

LEFT_MARGIN = 3 * cm
RIGHT_MARGIN = 1.5 * cm
TOP_MARGIN = 2 * cm
BOTTOM_MARGIN = 2 * cm


class _ReferenceDocTemplate(BaseDocTemplate):
    """Feeds headings into СОДЕРЖАНИЕ as they are laid out, so the table of
    contents gets real page numbers (multiBuild runs the passes)."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            level = _TOC_LEVELS.get(getattr(flowable.style, "name", ""))
            if level is not None:
                self.notify("TOCEntry", (level, escape(flowable.getPlainText()), self.page))


_BOLD_VARIANTS = {
    DEFAULT_FONT_NAME: f"{DEFAULT_FONT_NAME}-Bold",
    "Times-Roman": "Times-Bold",
    "Helvetica": "Helvetica-Bold",
    "Courier": "Courier-Bold",
}


def _styles(font_name: str, font_size: int) -> dict[str, ParagraphStyle]:
    bold = _BOLD_VARIANTS.get(font_name, font_name)
    leading = font_size * 1.5
    return {
        "header": ParagraphStyle("TitleHeader", fontName=font_name, fontSize=10, leading=12.5, alignment=TA_CENTER),
        "title": ParagraphStyle("TitleText", fontName=font_name, fontSize=font_size, leading=font_size * 1.25, alignment=TA_CENTER),
        "title_left": ParagraphStyle("TitleLeft", fontName=font_name, fontSize=font_size, leading=font_size * 1.4, alignment=TA_LEFT),
        "title_cell": ParagraphStyle("TitleCell", fontName=font_name, fontSize=font_size, leading=font_size * 1.25, alignment=TA_CENTER),
        "caption": ParagraphStyle("TitleCaption", fontName=font_name, fontSize=9, leading=11, alignment=TA_CENTER),
        "referat": ParagraphStyle("TitleReferat", fontName=bold, fontSize=16, leading=20, alignment=TA_CENTER),
        "topic": ParagraphStyle("TitleTopic", fontName=bold, fontSize=14, leading=18, alignment=TA_CENTER),
        "toc_heading": ParagraphStyle("TocHeading", fontName=bold, fontSize=font_size, leading=leading, alignment=TA_CENTER, spaceAfter=12),
        "structural_heading": ParagraphStyle(
            "StructuralHeading", fontName=bold, fontSize=font_size, leading=leading, alignment=TA_CENTER, spaceAfter=12
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading", fontName=bold, fontSize=font_size, leading=leading, firstLineIndent=1.25 * cm,
            spaceBefore=12, spaceAfter=6, keepWithNext=1,
        ),
        "subsection_heading": ParagraphStyle(
            "SubsectionHeading", fontName=bold, fontSize=font_size, leading=leading, firstLineIndent=1.25 * cm,
            spaceBefore=10, spaceAfter=6, keepWithNext=1,
        ),
        "body": ParagraphStyle(
            "Body", fontName=font_name, fontSize=font_size, leading=leading, alignment=TA_JUSTIFY, firstLineIndent=1.25 * cm
        ),
        "source": ParagraphStyle(
            "Source", fontName=font_name, fontSize=font_size, leading=leading, alignment=TA_JUSTIFY, firstLineIndent=1.25 * cm
        ),
        "toc0": ParagraphStyle(
            "TOCLevel0", fontName=font_name, fontSize=font_size, leading=leading, leftIndent=0.6 * cm, firstLineIndent=-0.6 * cm
        ),
        "toc1": ParagraphStyle(
            "TOCLevel1", fontName=font_name, fontSize=font_size, leading=leading, leftIndent=1.6 * cm, firstLineIndent=-1.0 * cm
        ),
    }


def _signature_table(cells: list[str], captions: list[str], styles: dict, underline_middle: bool = True) -> Table:
    widths = [6.3 * cm, 4.4 * cm, 5.8 * cm]
    table = Table(
        [
            [Paragraph(cells[0], styles["title_left"]), Paragraph(cells[1], styles["title_cell"]), Paragraph(cells[2], styles["title_cell"])],
            [Paragraph(c, styles["caption"]) for c in captions],
        ],
        colWidths=widths,
    )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("LINEBELOW", (2, 0), (2, 0), 0.6, colors.black),
    ]
    if underline_middle:
        commands.append(("LINEBELOW", (1, 0), (1, 0), 0.6, colors.black))
    table.setStyle(TableStyle(commands))
    return table


class _PinToBottom(Flowable):
    """Takes the rest of the frame and draws its flowables at the bottom,
    `reserve` above the frame edge — keeps the student block low on the
    title page, just above the city and year, however long the lines above
    it are."""

    def __init__(self, flowables: list, reserve: float):
        super().__init__()
        self._flowables = flowables
        self._reserve = reserve
        self._sizes: list = []

    def wrap(self, avail_width, avail_height):
        self._sizes = [f.wrap(avail_width, avail_height) for f in self._flowables]
        content = sum(h for _, h in self._sizes)
        self.width = avail_width
        self.height = max(content + self._reserve, avail_height - 1)
        return self.width, self.height

    def draw(self):
        y = self._reserve + sum(h for _, h in self._sizes)
        for flowable, (_, h) in zip(self._flowables, self._sizes):
            y -= h
            flowable.drawOn(self.canv, 0, y)


def _header_markup(header: str) -> str:
    lines = header.split("<br/>")
    return "<br/>".join(f"<b>{line}</b>" if "«" in line else line for line in lines)


def _title_page_flowables(info: TitlePageInfo, styles: dict) -> list:
    today = date.today().strftime("%d.%m.%Y")
    title_markup = info.lecture_label or escape(info.lecture_title)
    return [
        Paragraph(_header_markup(info.university_header), styles["header"]),
        Spacer(1, 0.5 * cm),
        Paragraph(escape(info.department), styles["title"]),
        Spacer(1, 2.0 * cm),
        Paragraph("ОЦЕНКА РЕФЕРАТА", styles["title_left"]),
        Paragraph("РУКОВОДИТЕЛЬ", styles["title_left"]),
        Spacer(1, 0.3 * cm),
        _signature_table(
            [escape(info.teacher_position), "", escape(info.teacher_name)],
            ["должность, уч. степень, звание", "подпись, дата", "инициалы, фамилия"],
            styles,
        ),
        Spacer(1, 2.3 * cm),
        Paragraph("РЕФЕРАТ", styles["referat"]),
        Spacer(1, 0.5 * cm),
        Paragraph(title_markup, styles["topic"]),
        Spacer(1, 0.5 * cm),
        Paragraph(f"по дисциплине: {escape(info.discipline)}", styles["title"]),
        _PinToBottom(
            [
                Paragraph("РЕФЕРАТ ВЫПОЛНИЛ", styles["title_left"]),
                Spacer(1, 0.3 * cm),
                _signature_table(
                    [f"СТУДЕНТ гр. №&nbsp;{escape(info.student_group)}", today, escape(info.student_name)],
                    ["", "подпись, дата", "инициалы, фамилия"],
                    styles,
                ),
            ],
            reserve=2.4 * cm,
        ),
        PageBreak(),
    ]


def build_reference_pdf(
    output_path: Path,
    content: ReferenceContent,
    font_name: str = DEFAULT_FONT_NAME,
    font_size: int = 12,
) -> None:
    if font_name == DEFAULT_FONT_NAME:
        _ensure_font_registered()
    styles = _styles(font_name, font_size)
    info = content.title_page

    story: list = _title_page_flowables(info, styles)

    story.append(Paragraph("СОДЕРЖАНИЕ", styles["toc_heading"]))
    toc = TableOfContents()
    toc.levelStyles = [styles["toc0"], styles["toc1"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())

    for section in content.sections:
        heading, body = section[0], section[1]
        level = section[2] if len(section) > 2 else 0
        story.append(Paragraph(escape(heading), styles["subsection_heading" if level else "section_heading"]))
        for paragraph_text in body.split("\n\n"):
            if paragraph_text.strip():
                story.append(Paragraph(escape(paragraph_text.strip()), styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("СПИСОК ИСТОЧНИКОВ", styles["structural_heading"]))
    for number, source in enumerate(content.sources, start=1):
        paragraphs = [p.strip() for p in source.split("\n\n") if p.strip()]
        for i, paragraph_text in enumerate(paragraphs):
            text = f"{number}. {paragraph_text}" if i == 0 else paragraph_text
            story.append(Paragraph(escape(text), styles["source"]))

    city, _, year = info.city_year.rpartition(",")
    city_lines = [city.strip(), year.strip()] if city else [info.city_year.strip()]

    def draw_page_furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, font_size)
        center = doc.leftMargin + doc.width / 2
        if doc.page == 1:
            y = doc.bottomMargin + 0.4 * cm + font_size * 1.3 * (len(city_lines) - 1)
            for line in city_lines:
                canvas.drawCentredString(center, y, line)
                y -= font_size * 1.3
        else:
            canvas.drawCentredString(center, 1.0 * cm, str(doc.page))
        canvas.restoreState()

    doc = _ReferenceDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=escape(info.lecture_title),
        author=info.student_name,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=draw_page_furniture)])
    doc.multiBuild(story)
