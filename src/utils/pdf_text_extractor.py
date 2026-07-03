"""Extract plain text from a PDF while skipping front and back matter.

Usage:
    python src/utils/pdf_text_extractor.py book.pdf
    python src/utils/pdf_text_extractor.py book.pdf -o extracted.txt
    python src/utils/pdf_text_extractor.py book.pdf -o repaired.txt --repair
    python src/utils/pdf_text_extractor.py book.pdf -o repaired.txt --llm

The default behavior skips the first 5 pages and the last 4 pages.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_SKIP_FIRST = 5
DEFAULT_SKIP_LAST = 4
DEFAULT_LLM_MODEL = "gpt-4o-mini"


LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

PRODUCTION_METADATA_RE = re.compile(
    r".*(?:_Layout\s+\d+|XP-[A-Z0-9_-]+).*\bPage\s*\d+\s*$",
    flags=re.IGNORECASE,
)


def is_production_metadata_line(text: str) -> bool:
    """Identify print-production footer/header lines, not story content."""

    return bool(PRODUCTION_METADATA_RE.fullmatch(" ".join(text.split())))


def remove_production_metadata(text: str) -> str:
    """Drop layout filename/date/page markers emitted by publishing software."""

    return "\n".join(
        line for line in text.splitlines() if not is_production_metadata_line(line)
    )


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


def normalize_ligatures(text: str) -> str:
    """Replace PDF ligature characters with normal letters."""

    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def mend_ligature_word_splits(text: str) -> str:
    """Join words split immediately after a ligature replacement."""

    return re.sub(
        r"\b([A-Za-zÀ-ÖØ-öø-ÿ]*(?:fi|fl|ffi|ffl))\s+([a-zà-öø-ÿ]{1,20})\b",
        r"\1\2",
        text,
    )


def normalize_whitespace(text: str) -> str:
    """Clean spacing while preserving paragraph breaks."""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def deterministic_repair(text: str) -> str:
    """Apply safe, rule-based repairs for French PDF-extracted text."""

    text = remove_production_metadata(text)
    text = normalize_ligatures(text)
    text = mend_ligature_word_splits(text)
    return normalize_whitespace(text)


def llm_repair(text: str, model: str = DEFAULT_LLM_MODEL) -> str:
    """Ask a small LLM to lightly repair extraction artifacts in French text."""

    from dotenv import load_dotenv

    load_dotenv()

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu repares du texte francais extrait d'un PDF. "
                    "Corrige uniquement les artefacts d'extraction: mots coupes, "
                    "ligatures mal lues, espaces incorrects et sauts de ligne "
                    "manifestement errones. Ne reformule pas, ne traduis pas, "
                    "n'ajoute rien et conserve les paragraphes autant que possible."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    repaired = response.choices[0].message.content or ""
    return normalize_whitespace(repaired)


def repair_text(text: str, use_llm: bool = False, model: str = DEFAULT_LLM_MODEL) -> str:
    """Repair extracted text using rules first, then optionally an LLM."""

    repaired = deterministic_repair(text)
    if use_llm:
        repaired = llm_repair(repaired, model=model)
    return repaired


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
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Apply deterministic French PDF text repairs after extraction.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also use a small LLM to repair extraction artifacts.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI model to use with --llm. Default: {DEFAULT_LLM_MODEL}.",
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
    output_text = (
        repair_text(extracted_text, use_llm=args.llm, model=args.model)
        if args.repair or args.llm
        else extracted_text
    )

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text, end="" if output_text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
