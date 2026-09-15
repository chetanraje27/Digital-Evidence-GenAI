from __future__ import annotations

import unittest

from src.document_parser import DocumentParseError, parse_evidence_file


def build_text_pdf(text: str) -> bytes:
    """Build a tiny standards-compliant PDF for extraction testing."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


class DocumentParserTests(unittest.TestCase):
    def test_utf8_text(self) -> None:
        parsed = parse_evidence_file("note.txt", b"First line.\n\nSecond line.")
        self.assertEqual(parsed.text, "First line.\nSecond line.")

    def test_digitally_extractable_pdf(self) -> None:
        parsed = parse_evidence_file("evidence.pdf", build_text_pdf("Device EX-17 collected at 09:15."))
        self.assertIn("Device EX-17", parsed.text)
        self.assertEqual(parsed.page_count, 1)

    def test_rejects_empty_or_unsupported_files(self) -> None:
        with self.assertRaises(DocumentParseError):
            parse_evidence_file("empty.txt", b"")
        with self.assertRaises(DocumentParseError):
            parse_evidence_file("evidence.docx", b"data")


if __name__ == "__main__":
    unittest.main()
