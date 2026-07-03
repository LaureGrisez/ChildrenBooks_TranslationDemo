from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from src.utils.pdf_compressor import _rendered_pages_match, compress_pdf


class PdfCompressorTests(unittest.TestCase):
    def _make_pdf(self, path: Path, color: tuple[float, float, float]) -> None:
        document = fitz.open()
        page = document.new_page(width=200, height=200)
        page.draw_rect(page.rect, fill=color)
        page.insert_text((20, 100), "Searchable text")
        document.save(path)
        document.close()

    def test_in_place_compression_preserves_render_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            self._make_pdf(path, (0.3, 0.7, 0.4))
            reference = Path(directory) / "reference.pdf"
            reference.write_bytes(path.read_bytes())

            compress_pdf(path, path)

            self.assertTrue(_rendered_pages_match(reference, path))
            with fitz.open(path) as document:
                self.assertIn("Searchable text", document[0].get_text())

    def test_visual_validation_rejects_material_page_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pdf"
            second = Path(directory) / "second.pdf"
            self._make_pdf(first, (1, 1, 1))
            self._make_pdf(second, (0, 0, 0))

            self.assertFalse(_rendered_pages_match(first, second))


if __name__ == "__main__":
    unittest.main()
