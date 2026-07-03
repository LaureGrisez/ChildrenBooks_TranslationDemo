"""Source-title extraction and dedicated translation prompting."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from .glossary import CharacterGlossary


def extract_source_title(pdf_path: Path | None, page_number: int, override: str = "") -> str:
    """Extract the title from the largest text on a 1-based PDF title page."""

    if override.strip():
        return " ".join(override.split())
    if pdf_path is None:
        return ""
    with fitz.open(pdf_path) as document:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"TITLE_PAGE_NUMBER={page_number} is outside the {document.page_count}-page PDF."
            )
        spans = [
            span
            for block in document.load_page(page_number - 1).get_text("dict").get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if str(span.get("text", "")).strip()
        ]
    if not spans:
        raise ValueError(f"No title text found on PDF page {page_number}.")
    largest_size = max(float(span.get("size", 0)) for span in spans)
    title_spans = [
        span for span in spans if float(span.get("size", 0)) >= largest_size * 0.9
    ]
    title_spans.sort(key=lambda span: (span["bbox"][1], span["bbox"][0]))
    return re.sub(r"\s+", " ", " ".join(span["text"] for span in title_spans)).strip()


def title_translation_prompt(
    *, source_title: str, source_language: str, target_language: str,
    glossary: CharacterGlossary,
) -> str:
    """Build a focused prompt that translates only the publication title."""

    return f"""
Translate this children's-book title from {source_language} into {target_language}.
Preserve the franchise and character names using the required mapping below.
Make the title concise, idiomatic, appealing to children, and faithful to its meaning.
Return only the translated title as one plain-text line, without quotes or commentary.

Required character names:
{glossary.format_name_guidance(target_language)}

Source title:
{source_title}
"""
