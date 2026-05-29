"""Generate a Markdown comparison report for any two text files.

Usage:
    python src/utils/text_comparison_report.py left.txt right.txt -o report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.translation.reporting import build_pairwise_comparison_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pairwise text comparison."""

    parser = argparse.ArgumentParser(
        description="Generate a Markdown report comparing any two text files."
    )
    parser.add_argument("left", type=Path, help="Path to the left/base text file.")
    parser.add_argument("right", type=Path, help="Path to the right/target text file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the Markdown report to write.",
    )
    parser.add_argument(
        "--left-label",
        help="Optional label for the left text. Defaults to the filename stem.",
    )
    parser.add_argument(
        "--right-label",
        help="Optional label for the right text. Defaults to the filename stem.",
    )
    parser.add_argument(
        "--title",
        help="Optional report title. Defaults to 'Text Comparison Report'.",
    )
    return parser.parse_args()


def main() -> None:
    """Read two text files and write a Markdown comparison report."""

    args = parse_args()
    left_text = args.left.read_text(encoding="utf-8")
    right_text = args.right.read_text(encoding="utf-8")
    report = build_pairwise_comparison_report(
        left_label=args.left_label or args.left.stem,
        right_label=args.right_label or args.right.stem,
        left_text=left_text,
        right_text=right_text,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
