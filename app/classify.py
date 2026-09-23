"""Guess what a file from the teacher actually is, from its name and (for
documents) a bit of its text content, so the pipeline can sort it into the
right slot instead of dumping everything in one place.
"""
from pathlib import Path

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"}

# Keywords are matched case-insensitively against the filename and, for
# documents, the first page/paragraphs of extracted text. Assignment
# keywords are checked first since a conspect can casually mention
# "требования" while an assignment file is very unlikely to be a lecture.
ASSIGNMENT_KEYWORDS = (
    "техническое задание", "тз", "требовани", "оформлен",
    "как писать", "как оформлять", "методичк", "задание на реферат",
    "критери",
)
CONSPECT_KEYWORDS = ("конспект", "лекция", "материал", "презентац", "слайд")

KIND_VIDEO = "video"
KIND_ASSIGNMENT = "assignment"
KIND_CONSPECT = "conspect"
KIND_DOCUMENT = "document"  # recognized as a document, but kind unclear
KIND_OTHER = "other"


def _matches_any(haystack: str, keywords: tuple[str, ...]) -> bool:
    haystack = haystack.lower()
    return any(keyword in haystack for keyword in keywords)


def classify_file(filename: str, extracted_text: str = "") -> str:
    """Classify a single ingested file into one of the KIND_* buckets.

    `extracted_text` is optional plain text pulled from the first part of a
    document (see app.text_extract) — used only as a tie-breaker when the
    filename alone isn't conclusive.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in VIDEO_EXTENSIONS:
        return KIND_VIDEO

    if suffix not in DOCUMENT_EXTENSIONS:
        return KIND_OTHER

    signal = f"{filename}\n{extracted_text[:2000]}"
    if _matches_any(signal, ASSIGNMENT_KEYWORDS):
        return KIND_ASSIGNMENT
    if _matches_any(signal, CONSPECT_KEYWORDS):
        return KIND_CONSPECT

    return KIND_DOCUMENT
