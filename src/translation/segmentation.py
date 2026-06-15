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
    body_page_numbers: list[int],
    text_page_numbers: list[int],
    context_window: int = 0,
) -> list[SpreadSegment]:
    """Group source paragraphs into spread-aligned segments across all body pages."""

    paragraphs = split_source_paragraphs(source_text)
    if len(paragraphs) != len(text_page_numbers):
        raise ValueError(
            "The number of source paragraphs must match the number of text-bearing "
            f"body pages for multimodal mode. Found {len(paragraphs)} paragraphs and "
            f"{len(text_page_numbers)} text-bearing page targets."
        )

    text_page_set = set(text_page_numbers)
    paragraph_index = 0
    grouped: list[tuple[tuple[int, ...], list[int], list[str]]] = []
    for page_number in body_page_numbers:
        spread_pages = spread_pages_for_body_page(page_number)
        if grouped and grouped[-1][0] == spread_pages:
            current = grouped[-1]
        else:
            grouped.append((spread_pages, [], []))
            current = grouped[-1]

        if page_number not in text_page_set:
            continue

        if paragraph_index >= len(paragraphs):
            raise ValueError(
                "Ran out of source paragraphs while assigning text-bearing pages in "
                "multimodal mode."
            )
        current[1].append(page_number)
        current[2].append(paragraphs[paragraph_index])
        paragraph_index += 1

    if paragraph_index != len(paragraphs):
        raise ValueError(
            "Not all source paragraphs were assigned to multimodal spread segments. "
            f"Assigned {paragraph_index} of {len(paragraphs)} paragraphs."
        )

    segments: list[SpreadSegment] = []
    populated_grouped = [
        (spread_pages, grouped_page_numbers, page_texts)
        for spread_pages, grouped_page_numbers, page_texts in grouped
        if page_texts
    ]
    spread_source_texts = [
        "\n\n".join(page_texts) for _, _, page_texts in populated_grouped
    ]
    for index, (spread_pages, grouped_page_numbers, page_texts) in enumerate(
        populated_grouped
    ):
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
