"""Segmentation helpers for spread-aligned multimodal translation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpreadSegment:
    """One spread-aligned translation unit covering one or two text pages."""

    index: int
    spread_pages: tuple[int, ...]
    page_numbers: tuple[int, ...]
    page_texts: tuple[str, ...]
    previous_source_text: str

    @property
    def source_text(self) -> str:
        """Return the current spread source text in page order."""

        return "\n\n".join(self.page_texts)


def split_source_paragraphs(text: str) -> list[str]:
    """Split cleaned source text into non-empty paragraphs."""

    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def spread_pages_for_body_page(page_number: int) -> tuple[int, ...]:
    """Return the two-page spread containing a 1-based page number."""

    if page_number % 2 == 0:
        return (page_number, page_number + 1)
    return (max(1, page_number - 1), page_number)


def build_spread_segments(
    source_text: str,
    page_numbers: list[int],
    context_window: int = 0,
) -> list[SpreadSegment]:
    """Group page-aligned source paragraphs into spread-aligned segments."""

    paragraphs = split_source_paragraphs(source_text)
    if len(paragraphs) != len(page_numbers):
        raise ValueError(
            "The number of source paragraphs must match the number of non-empty body "
            f"pages for multimodal mode. Found {len(paragraphs)} paragraphs and "
            f"{len(page_numbers)} page targets."
        )

    grouped: list[tuple[tuple[int, ...], list[int], list[str]]] = []
    for page_number, paragraph in zip(page_numbers, paragraphs):
        spread_pages = spread_pages_for_body_page(page_number)
        if grouped and grouped[-1][0] == spread_pages:
            grouped[-1][1].append(page_number)
            grouped[-1][2].append(paragraph)
        else:
            grouped.append((spread_pages, [page_number], [paragraph]))

    segments: list[SpreadSegment] = []
    spread_source_texts = ["\n\n".join(page_texts) for _, _, page_texts in grouped]
    for index, (spread_pages, grouped_page_numbers, page_texts) in enumerate(grouped):
        previous_start = max(0, index - context_window)
        previous_source_text = "\n\n".join(spread_source_texts[previous_start:index])
        segments.append(
            SpreadSegment(
                index=index,
                spread_pages=spread_pages,
                page_numbers=tuple(grouped_page_numbers),
                page_texts=tuple(page_texts),
                previous_source_text=previous_source_text,
            )
        )
    return segments
