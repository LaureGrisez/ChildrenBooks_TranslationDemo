"""Exact paragraph alignment for panel evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .segmentation import split_source_paragraphs


def align_candidate_paragraphs(
    source_text: str,
    candidates: list[dict[str, Any]],
    *,
    source_context_window: int = 1,
) -> list[dict[str, Any]]:
    """Align successful candidates to source paragraphs or fail explicitly."""

    source_paragraphs = split_source_paragraphs(source_text)
    successful = [candidate for candidate in candidates if candidate.get("status") == "ok"]
    if len(successful) < 2:
        raise ValueError("Panel evaluation requires at least two successful candidates.")
    duplicate_names = [
        name
        for name, count in Counter(candidate["name"] for candidate in successful).items()
        if count > 1
    ]
    if duplicate_names:
        raise ValueError(
            "Panel evaluation requires unique candidate names; duplicates: "
            + ", ".join(sorted(duplicate_names))
        )

    candidate_paragraphs = {}
    for candidate in successful:
        paragraphs = split_source_paragraphs(candidate.get("text", ""))
        if len(paragraphs) != len(source_paragraphs):
            raise ValueError(
                f"Candidate '{candidate['name']}' has {len(paragraphs)} paragraphs; "
                f"expected {len(source_paragraphs)}."
            )
        candidate_paragraphs[candidate["name"]] = paragraphs

    aligned = []
    for index, source_paragraph in enumerate(source_paragraphs):
        previous_start = max(0, index - source_context_window)
        next_end = min(len(source_paragraphs), index + source_context_window + 1)
        aligned.append(
            {
                "paragraph_id": f"p{index + 1:04d}",
                "index": index,
                "source": source_paragraph,
                "previous_source": "\n\n".join(
                    source_paragraphs[previous_start:index]
                ),
                "next_source": "\n\n".join(source_paragraphs[index + 1 : next_end]),
                "options": {
                    candidate["name"]: candidate_paragraphs[candidate["name"]][index]
                    for candidate in successful
                },
            }
        )
    return aligned
