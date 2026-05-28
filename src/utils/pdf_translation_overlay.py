"""Replace per-page PDF text with translated paragraphs using PyMuPDF.

This script assumes:
- the PDF body text lives on pages `skip_first` through `page_count - skip_last - 1`
- the translation file contains one paragraph per translated page
- each target page has one logical text area made of one or more text lines

Typical usage:
    python src/utils/pdf_translation_overlay.py \
        flag_ship__l_arbre_de_barbapapa_INT.pdf \
        translation/l_arbre_de_barbapapa_INT_fi.txt \
        -o output_fi.pdf

For scripts outside Latin-1, pass a Unicode font file:
    python src/utils/pdf_translation_overlay.py input.pdf translation_hi.txt \
        -o output_hi.pdf --font-file /path/to/NotoSansDevanagari-Regular.ttf
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_SKIP_FIRST = 5
DEFAULT_SKIP_LAST = 4
DEFAULT_PADDING = 2.0
MIN_FONT_SIZE = 6.0
DEFAULT_UNICODE_FONT_CANDIDATES = (
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/LucidaGrande.ttc"),
    Path("/System/Library/Fonts/HelveticaNeue.ttc"),
)


def split_paragraphs(text: str) -> list[str]:
    """Split a text file into non-empty paragraphs separated by blank lines."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return [part.strip() for part in normalized.split("\n\n") if part.strip()]


def load_translations(path: Path) -> list[str]:
    """Load translated paragraphs from a plain-text file."""

    return split_paragraphs(path.read_text(encoding="utf-8"))


def parse_page_numbers(spec: str | None) -> list[int]:
    """Parse a comma-separated list of 1-based page numbers."""

    if not spec:
        return []

    page_numbers: list[int] = []
    for part in spec.split(","):
        value = part.strip()
        if not value:
            continue
        page_number = int(value)
        if page_number <= 0:
            raise ValueError("Page numbers must be 1 or greater.")
        page_numbers.append(page_number)
    return page_numbers


def build_glossary_mapping(csv_path: Path, target_language: str) -> dict[str, str]:
    """Return source-to-target character name mappings for one language."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
        fieldnames = [header.strip() for header in reader.fieldnames or []]

    mapping: dict[str, str] = {}
    if "French" not in fieldnames:
        raise ValueError(f"Glossary CSV must contain a French column: {csv_path}")

    target_header = next(
        (header for header in fieldnames if header.lower() == target_language.lower()),
        None,
    )
    if target_header is None:
        raise ValueError(f"Unsupported glossary target language: {target_language}")

    for row in rows:
        source_name = row["French"].strip()
        target_name = row[target_header].strip()
        if source_name and target_name:
            mapping[source_name] = target_name
    return mapping


def replace_glossary_names_in_text(text: str, mapping: dict[str, str]) -> str | None:
    """Replace glossary names inside a text fragment while preserving spacing."""

    updated_text = text
    replaced = False

    for source_name in sorted(mapping, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(source_name)}(?!\w)")
        updated_text, count = pattern.subn(mapping[source_name], updated_text)
        if count:
            replaced = True

    return updated_text if replaced else None


def choose_default_unicode_font() -> Path | None:
    """Return a broadly Unicode-capable system font when available."""

    for candidate in DEFAULT_UNICODE_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def collect_page_lines(page: fitz.Page) -> tuple[list[fitz.Rect], fitz.Rect, float]:
    """Return line rectangles, their union box, and a median source font size."""

    text_dict = page.get_text("dict")
    line_rects: list[fitz.Rect] = []
    font_sizes: list[float] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue
            line_rects.append(fitz.Rect(line["bbox"]))
            font_sizes.extend(
                span["size"] for span in spans if span.get("text", "").strip()
            )

    if not line_rects:
        raise ValueError("No text lines found on page.")

    union_rect = fitz.Rect(line_rects[0])
    for rect in line_rects[1:]:
        union_rect.include_rect(rect)

    median_font_size = statistics.median(font_sizes) if font_sizes else 12.0
    return line_rects, union_rect, median_font_size


def add_redactions(page: fitz.Page, rects: list[fitz.Rect], padding: float) -> None:
    """Cover existing text lines without touching the rest of the page."""

    for rect in rects:
        padded = fitz.Rect(rect)
        padded.x0 -= padding
        padded.y0 -= padding
        padded.x1 += padding
        padded.y1 += padding
        page.add_redact_annot(padded, fill=(1, 1, 1))

    page.apply_redactions()


def register_font(
    page: fitz.Page,
    font_file: Path | None,
) -> str:
    """Register a custom font when provided, otherwise use a Unicode fallback."""

    resolved_font = font_file or choose_default_unicode_font()
    if resolved_font is None:
        return "helv"

    font_name = "translation_font"
    page.insert_font(fontname=font_name, fontfile=str(resolved_font))
    return font_name


def insert_translation(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    font_name: str,
    font_size: float,
    align: int = fitz.TEXT_ALIGN_LEFT,
) -> float:
    """Insert text and return PyMuPDF's textbox status value."""

    return page.insert_textbox(
        rect,
        text,
        fontname=font_name,
        fontsize=font_size,
        color=(0, 0, 0),
        align=align,
    )


def replace_page_text(
    page: fitz.Page,
    translation: str,
    font_file: Path | None,
    padding: float,
) -> None:
    """Redact source text on a page and write the translated paragraph."""

    line_rects, union_rect, source_font_size = collect_page_lines(page)
    add_redactions(page, line_rects, padding=padding)

    insert_rect = fitz.Rect(union_rect)
    insert_rect.x0 -= padding
    insert_rect.y0 -= padding
    insert_rect.x1 += padding
    insert_rect.y1 += padding

    font_name = register_font(page, font_file)

    font_size = float(source_font_size)
    status = insert_translation(page, insert_rect, translation, font_name, font_size)
    while status < 0 and font_size > MIN_FONT_SIZE:
        font_size -= 0.5
        status = insert_translation(page, insert_rect, translation, font_name, font_size)

    if status < 0:
        raise ValueError(
            "Translation does not fit on the page text area even at the minimum font size."
        )


def replace_names_on_page(
    page: fitz.Page,
    mapping: dict[str, str],
    font_file: Path | None,
    padding: float,
) -> None:
    """Replace glossary names found on one page while preserving other text."""

    replacements: list[tuple[fitz.Rect, str, float]] = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            replacement_text = replace_glossary_names_in_text(line_text, mapping)
            if not replacement_text:
                continue

            span_sizes = [span["size"] for span in spans if span.get("text", "").strip()]
            font_size = statistics.median(span_sizes) if span_sizes else 12.0
            replacements.append((fitz.Rect(line["bbox"]), replacement_text, font_size))

    if not replacements:
        return

    rects = [rect for rect, _, _ in replacements]
    add_redactions(page, rects, padding=padding)
    font_name = register_font(page, font_file)

    for rect, replacement_text, font_size in replacements:
        insert_rect = fitz.Rect(rect)
        insert_rect.x0 -= padding
        insert_rect.y0 -= padding
        insert_rect.x1 += padding
        insert_rect.y1 += padding

        status = insert_translation(
            page,
            insert_rect,
            replacement_text,
            font_name,
            float(font_size),
        )
        trial_font_size = float(font_size)
        while status < 0 and trial_font_size > MIN_FONT_SIZE:
            trial_font_size -= 0.5
            status = insert_translation(
                page,
                insert_rect,
                replacement_text,
                font_name,
                trial_font_size,
            )

        if status < 0:
            raise ValueError(
                f"Glossary replacement '{replacement_text}' does not fit on page {page.number + 1}."
            )


def replace_pdf_text(
    pdf_path: Path,
    translation_path: Path,
    output_path: Path,
    skip_first: int = DEFAULT_SKIP_FIRST,
    skip_last: int = DEFAULT_SKIP_LAST,
    font_file: Path | None = None,
    padding: float = DEFAULT_PADDING,
    glossary_csv: Path | None = None,
    glossary_language: str | None = None,
    glossary_pages: list[int] | None = None,
) -> None:
    """Produce a new PDF with translated text inserted over body pages."""

    translations = load_translations(translation_path)

    with fitz.open(pdf_path) as document:
        start_page = min(skip_first, document.page_count)
        end_page = max(start_page, document.page_count - skip_last)
        target_pages: list[int] = []

        for page_index in range(start_page, end_page):
            page = document.load_page(page_index)
            if page.get_text("text").strip():
                target_pages.append(page_index)

        target_page_count = len(target_pages)

        if len(translations) != target_page_count:
            raise ValueError(
                f"Expected {target_page_count} translated paragraphs for non-empty pages "
                f"between pages {start_page + 1}-{end_page}, found {len(translations)}."
            )

        for offset, page_index in enumerate(target_pages):
            page = document.load_page(page_index)
            replace_page_text(
                page,
                translations[offset],
                font_file=font_file,
                padding=padding,
            )

        if glossary_csv and glossary_language and glossary_pages:
            mapping = build_glossary_mapping(glossary_csv, glossary_language)
            for page_number in glossary_pages:
                if page_number > document.page_count:
                    raise ValueError(
                        f"Glossary page {page_number} exceeds document length {document.page_count}."
                    )
                page = document.load_page(page_number - 1)
                replace_names_on_page(
                    page,
                    mapping=mapping,
                    font_file=font_file,
                    padding=padding,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(output_path, garbage=4, deflate=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Replace body-page PDF text with translated paragraphs."
    )
    parser.add_argument("pdf", type=Path, help="Path to the original PDF.")
    parser.add_argument(
        "translation",
        type=Path,
        help="Path to the translated text file with one paragraph per page.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the translated output PDF.",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=DEFAULT_SKIP_FIRST,
        help=f"Pages to skip at the start. Default: {DEFAULT_SKIP_FIRST}.",
    )
    parser.add_argument(
        "--skip-last",
        type=int,
        default=DEFAULT_SKIP_LAST,
        help=f"Pages to skip at the end. Default: {DEFAULT_SKIP_LAST}.",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        help="Optional TTF/OTF font to embed for non-Latin translations.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=f"Extra padding in points around detected text lines. Default: {DEFAULT_PADDING}.",
    )
    parser.add_argument(
        "--glossary-csv",
        type=Path,
        help="Optional glossary CSV used for exact character-name replacements.",
    )
    parser.add_argument(
        "--glossary-language",
        help="Target language column to use with --glossary-csv, for example Finnish.",
    )
    parser.add_argument(
        "--glossary-pages",
        help="Comma-separated 1-based page numbers for glossary-only name replacement, for example 2,3.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PDF text replacement script."""

    args = parse_args()
    replace_pdf_text(
        pdf_path=args.pdf,
        translation_path=args.translation,
        output_path=args.output,
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        font_file=args.font_file,
        padding=args.padding,
        glossary_csv=args.glossary_csv,
        glossary_language=args.glossary_language,
        glossary_pages=parse_page_numbers(args.glossary_pages),
    )


if __name__ == "__main__":
    main()
