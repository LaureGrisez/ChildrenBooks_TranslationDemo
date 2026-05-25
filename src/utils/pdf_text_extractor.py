"""Extract plain text from a PDF while skipping front and back matter.

Usage:
    python src/utils/pdf_text_extractor.py book.pdf
    python src/utils/pdf_text_extractor.py book.pdf -o extracted.txt

The default behavior skips the first 5 pages and the last 4 pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_SKIP_FIRST = 5
DEFAULT_SKIP_LAST = 4


def extract_pdf_text(
    pdf_path: Path,
    skip_first: int = DEFAULT_SKIP_FIRST,
    skip_last: int = DEFAULT_SKIP_LAST,
) -> str:
    """Return plain text from a PDF, excluding the first and last pages."""

    if skip_first < 0 or skip_last < 0:
        raise ValueError("skip_first and skip_last must be zero or greater.")

    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        start_page = min(skip_first, page_count)
        end_page = max(start_page, page_count - skip_last)

        pages_text: list[str] = []
        for page_index in range(start_page, end_page):
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                pages_text.append(text)

    return "\n\n".join(pages_text)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the extraction script."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract plain text from a PDF, skipping the first 5 pages and "
            "the last 4 pages by default."
        )
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path for the extracted text file. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=DEFAULT_SKIP_FIRST,
        help=f"Number of pages to skip from the start. Default: {DEFAULT_SKIP_FIRST}.",
    )
    parser.add_argument(
        "--skip-last",
        type=int,
        default=DEFAULT_SKIP_LAST,
        help=f"Number of pages to skip from the end. Default: {DEFAULT_SKIP_LAST}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run PDF text extraction from the command line."""

    args = parse_args()
    extracted_text = extract_pdf_text(
        args.pdf,
        skip_first=args.skip_first,
        skip_last=args.skip_last,
    )

    if args.output:
        args.output.write_text(extracted_text, encoding="utf-8")
    else:
        print(extracted_text)


if __name__ == "__main__":
    main()
