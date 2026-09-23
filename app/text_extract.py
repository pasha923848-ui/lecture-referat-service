"""Best-effort plain-text extraction from documents, used for classification
and to give the web UI a readable preview without opening the original file.
Never raises — a file we can't parse just gets an empty string back.
"""
from pathlib import Path

# Enough for classification/preview without paying to OCR/parse a huge file.
_PREVIEW_PAGE_LIMIT = 5


def extract_text(path: Path, max_pages: int | None = _PREVIEW_PAGE_LIMIT) -> str:
    """Extract text from a document. `max_pages` caps how many PDF pages are
    read (only PDFs can be partial like this); pass None to read the whole
    file — needed for things like fully parsing a multi-page assignment
    document (see app/assignment.py), where truncating would silently drop
    later lectures' questions.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path, max_pages)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix in (".txt", ".rtf"):
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def _extract_pdf(path: Path, max_pages: int | None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    return "\n".join(page.extract_text() or "" for page in pages)


def _extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)
