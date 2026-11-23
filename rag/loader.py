"""Stage 1 - PDF loading.

Input : one or more PDF files (paths or raw bytes)
Output: list[Page] - one record per page with the cleaned text and its source
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str      # file name, used for citations
    page: int        # 1-based page number
    text: str


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)                # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)             # keep paragraph breaks, drop blank runs
    return text.strip()


def load_pdf(data: bytes | str | Path, source: str | None = None) -> list[Page]:
    if isinstance(data, (str, Path)):
        path = Path(data)
        source = source or path.name
        reader = PdfReader(str(path))
    else:
        reader = PdfReader(io.BytesIO(data))
        source = source or "uploaded.pdf"

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"{source} is password protected") from exc

    pages = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = _clean(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(source=source, page=i, text=text))
    return pages


def load_pdfs(items: list[tuple[str, bytes | str | Path]]) -> list[Page]:
    """items: [(display_name, path_or_bytes), ...]"""
    pages: list[Page] = []
    for name, data in items:
        pages.extend(load_pdf(data, source=name))
    return pages
