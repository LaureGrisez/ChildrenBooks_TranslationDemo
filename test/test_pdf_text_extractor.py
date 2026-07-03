"""Tests for deterministic PDF source cleanup."""

from __future__ import annotations

import unittest

from src.utils.pdf_text_extractor import deterministic_repair


class PdfTextExtractorTests(unittest.TestCase):
    def test_removes_print_production_metadata_without_touching_story(self) -> None:
        raw = (
            "Barbapapa est né dans un jardin.\n"
            "Barbapapa-BARBAPAPA_Layout 1  13/09/10  14:59  Page6\n\n"
            "François arrosait ses fleurs."
        )
        self.assertEqual(
            "Barbapapa est né dans un jardin.\n\nFrançois arrosait ses fleurs.\n",
            deterministic_repair(raw),
        )


if __name__ == "__main__":
    unittest.main()
