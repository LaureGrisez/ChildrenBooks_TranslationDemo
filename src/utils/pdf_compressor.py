"""Compress PDFs while preserving searchable text and vector graphics."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


DEFAULT_DPI = 144
DEFAULT_DPI_THRESHOLD = 180
DEFAULT_QUALITY = 75
VALIDATION_DPI = 36
MAX_MEAN_CHANNEL_DIFFERENCE = 8.0
MAX_SEVERE_PIXEL_RATIO = 0.02
MAX_VALIDATION_PAGES = 10


def _rendered_pages_match(input_path: Path, candidate_path: Path) -> bool:
    """Return whether low-resolution renders remain visually equivalent."""

    matrix = fitz.Matrix(VALIDATION_DPI / 72, VALIDATION_DPI / 72)
    with fitz.open(input_path) as original, fitz.open(candidate_path) as candidate:
        if original.page_count != candidate.page_count:
            return False
        if original.page_count <= MAX_VALIDATION_PAGES:
            page_numbers = range(original.page_count)
        else:
            page_numbers = sorted(
                {
                    round(index * (original.page_count - 1) / (MAX_VALIDATION_PAGES - 1))
                    for index in range(MAX_VALIDATION_PAGES)
                }
                | {min(5, original.page_count - 1)}
            )
        for page_number in page_numbers:
            before = original[page_number].get_pixmap(matrix=matrix, alpha=False)
            after = candidate[page_number].get_pixmap(matrix=matrix, alpha=False)
            if (before.width, before.height, before.n) != (
                after.width,
                after.height,
                after.n,
            ):
                return False
            differences = [
                abs(left - right)
                for left, right in zip(before.samples, after.samples)
            ]
            if not differences:
                continue
            mean_difference = sum(differences) / len(differences)
            severe_ratio = sum(value > 50 for value in differences) / len(differences)
            if (
                mean_difference > MAX_MEAN_CHANNEL_DIFFERENCE
                or severe_ratio > MAX_SEVERE_PIXEL_RATIO
            ):
                return False
    return True


def _save_pdf(input_path: Path, output_path: Path, *, rewrite_images: bool,
              dpi: int, dpi_threshold: int, quality: int) -> None:
    """Write one compression candidate."""

    with fitz.open(input_path) as document:
        if rewrite_images:
            document.rewrite_images(
                dpi_threshold=dpi_threshold,
                dpi_target=dpi,
                quality=quality,
            )
        document.save(
            output_path,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=1,
        )


def compress_pdf(
    input_path: Path,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    dpi_threshold: int = DEFAULT_DPI_THRESHOLD,
    quality: int = DEFAULT_QUALITY,
    recompress_images: bool = True,
) -> tuple[int, int]:
    """Compress a PDF and return its original and compressed sizes in bytes.

    ``input_path`` and ``output_path`` may be the same. In that case the new
    file is written beside the original and atomically replaces it only after
    compression succeeds.
    """

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDF does not exist: {input_path}")
    if dpi <= 0 or dpi_threshold <= 0:
        raise ValueError("DPI values must be positive integers.")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = input_path.stat().st_size
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".pdf",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        _save_pdf(
            input_path,
            temporary_path,
            rewrite_images=recompress_images,
            dpi=dpi,
            dpi_threshold=dpi_threshold,
            quality=quality,
        )
        if not _rendered_pages_match(input_path, temporary_path):
            if not recompress_images:
                raise RuntimeError(
                    "Safe compression changed the rendered PDF; input was not replaced."
                )
            print(
                "Image recompression changed the rendered PDF; using safe "
                "stream-only compression instead.",
                file=sys.stderr,
            )
            temporary_path.unlink()
            _save_pdf(
                input_path,
                temporary_path,
                rewrite_images=False,
                dpi=dpi,
                dpi_threshold=dpi_threshold,
                quality=quality,
            )
            if not _rendered_pages_match(input_path, temporary_path):
                raise RuntimeError(
                    "Safe compression also changed the rendered PDF; input was not replaced."
                )

        os.replace(temporary_path, output_path)
        temporary_path = None
        return original_size, output_path.stat().st_size
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress PDF images and streams with PyMuPDF."
    )
    parser.add_argument("input", type=Path, help="PDF to compress.")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--dpi-threshold", type=int, default=DEFAULT_DPI_THRESHOLD,
        help="Only downsample images above this effective DPI.",
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY,
        help="JPEG image quality from 1 to 100.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Skip lossy image rewriting and only compress PDF streams.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_size, compressed_size = compress_pdf(
        args.input,
        args.output,
        dpi=args.dpi,
        dpi_threshold=args.dpi_threshold,
        quality=args.quality,
        recompress_images=not args.safe,
    )
    saving = 100 * (original_size - compressed_size) / original_size if original_size else 0
    print(
        f"Saved compressed PDF: {args.output} "
        f"({original_size / 1_048_576:.2f} MiB -> "
        f"{compressed_size / 1_048_576:.2f} MiB, {saving:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()
