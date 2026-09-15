"""Safe, in-memory text extraction for Review-2 evidence inputs."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

MAX_EVIDENCE_BYTES = 10 * 1024 * 1024
SUPPORTED_EVIDENCE_SUFFIXES = {".txt", ".pdf"}


class DocumentParseError(ValueError):
    """Raised when an uploaded evidence document cannot be safely parsed."""


@dataclass(frozen=True)
class ParsedDocument:
    """Text and non-sensitive extraction metadata retained in memory."""

    text: str
    suffix: str
    character_count: int
    page_count: int | None = None


def _clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_evidence_file(
    filename: str,
    data: bytes,
    max_bytes: int = MAX_EVIDENCE_BYTES,
) -> ParsedDocument:
    """Validate and extract UTF-8 TXT or digitally readable PDF evidence."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EVIDENCE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_EVIDENCE_SUFFIXES))
        raise DocumentParseError(f"Unsupported evidence type. Allowed types: {allowed}")
    if not data:
        raise DocumentParseError("The uploaded evidence file is empty.")
    if len(data) > max_bytes:
        raise DocumentParseError(
            f"The file exceeds the {max_bytes / (1024 * 1024):.0f} MB in-memory limit."
        )

    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("TXT evidence must use UTF-8 encoding.") from exc
        cleaned = _clean_text(text)
        if not cleaned:
            raise DocumentParseError("The TXT evidence contains no readable text.")
        return ParsedDocument(cleaned, suffix, len(cleaned))

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParseError(
            "PDF extraction requires pypdf. Install the project requirements and retry."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise DocumentParseError("Password-protected PDFs are not supported.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"The PDF could not be parsed: {exc}") from exc

    cleaned = _clean_text("\n".join(pages))
    if not cleaned:
        raise DocumentParseError(
            "No digital text was found. This may be a scanned/image-only PDF; OCR is not enabled."
        )
    return ParsedDocument(cleaned, suffix, len(cleaned), len(pages))


__all__ = [
    "DocumentParseError",
    "MAX_EVIDENCE_BYTES",
    "ParsedDocument",
    "SUPPORTED_EVIDENCE_SUFFIXES",
    "parse_evidence_file",
]
