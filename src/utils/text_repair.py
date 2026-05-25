"""Repair text extracted from French PDFs.

Usage:
    python src/utils/text_repair.py input.txt -o repaired.txt
    python src/utils/text_repair.py input.txt -o repaired.txt --llm

The default repair is deterministic and handles common PDF extraction issues,
especially ligatures such as "ﬁ" and "ﬂ" that sometimes become split words.
The optional LLM mode can be useful for messier OCR/text extraction, but the
deterministic mode is enough for simple cases like "difﬁ cile" -> "difficile".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


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


def normalize_ligatures(text: str) -> str:
    """Replace PDF ligature characters with normal letters."""

    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def mend_ligature_word_splits(text: str) -> str:
    """Join words split immediately after a ligature replacement.

    Examples:
        "fi n" -> "fin"
        "diffi cile" -> "difficile"
        "souffl e" -> "souffle"
    """

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
            {
                "role": "user",
                "content": text,
            },
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
    """Parse command-line arguments for the repair script."""

    parser = argparse.ArgumentParser(
        description="Repair French text extracted from a PDF."
    )
    parser.add_argument("input", type=Path, help="Path to the extracted text file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output path. Prints repaired text to stdout if omitted.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also use a small LLM after deterministic cleanup.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI model to use with --llm. Default: {DEFAULT_LLM_MODEL}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run text repair from the command line."""

    args = parse_args()
    input_text = args.input.read_text(encoding="utf-8")
    repaired_text = repair_text(input_text, use_llm=args.llm, model=args.model)

    if args.output:
        args.output.write_text(repaired_text, encoding="utf-8")
    else:
        print(repaired_text, end="")


if __name__ == "__main__":
    main()
