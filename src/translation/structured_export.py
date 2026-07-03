"""Validated, page-addressable JSON export for digital book consumers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import TranslationWorkflowConfig
from .glossary import CharacterGlossary
from .multimodal import collect_body_pages, collect_non_empty_body_pages
from .segmentation import split_source_paragraphs, spread_pages_for_body_page


def character_name_map(glossary: CharacterGlossary, language: str) -> dict[str, str]:
    source = glossary.normalize_language(glossary.source_language)
    target = glossary.normalize_language(language)
    return {
        row[source].strip(): row[target].strip()
        for row in glossary.rows
        if row[source].strip() and row[target].strip()
    }


def build_structured_book(
    *, source_text: str, translated_text: str, source_title: str,
    translated_title: str, language: str, glossary: CharacterGlossary,
    config: TranslationWorkflowConfig,
) -> dict[str, Any]:
    """Build the canonical JSON payload and reject unsafe paragraph drift."""

    if config.source_pdf_path is None:
        raise ValueError("Structured JSON export requires SOURCE_PDF_PATH.")
    source_paragraphs = split_source_paragraphs(source_text)
    translated_paragraphs = split_source_paragraphs(translated_text)
    body_pages = collect_body_pages(
        config.source_pdf_path, config.pdf_skip_first, config.pdf_skip_last
    )
    text_pages = collect_non_empty_body_pages(
        config.source_pdf_path, config.pdf_skip_first, config.pdf_skip_last
    )
    if len(source_paragraphs) != len(text_pages):
        raise ValueError(
            "Cannot export JSON: source paragraph count does not match text-bearing "
            f"PDF pages ({len(source_paragraphs)} != {len(text_pages)})."
        )
    if len(translated_paragraphs) != len(source_paragraphs):
        raise ValueError(
            "Cannot export JSON: final translation paragraph count drifted from the "
            f"source ({len(translated_paragraphs)} != {len(source_paragraphs)})."
        )

    page_content = {
        page_number: {
            "original": source_paragraphs[index],
            "translated": translated_paragraphs[index],
        }
        for index, page_number in enumerate(text_pages)
    }
    spread_pairs: list[tuple[int, ...]] = []
    for page_number in body_pages:
        pair = spread_pages_for_body_page(page_number)
        if pair not in spread_pairs:
            spread_pairs.append(pair)

    translations = []
    for spread_index, pages in enumerate(spread_pairs, start=1):
        left_page, right_page = pages
        translations.append(
            {
                "spread": spread_index,
                "pages_index": [left_page, right_page],
                "left": {
                    "page_index": left_page,
                    **page_content.get(left_page, {"original": "", "translated": ""}),
                },
                "right": {
                    "page_index": right_page,
                    **page_content.get(right_page, {"original": "", "translated": ""}),
                },
            }
        )

    return {
        "language": config.language_code(language),
        "metadata": {
            "schema_version": 1,
            "run_id": config.run_id,
            "experiment_name": config.experiment_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_language": config.source_language,
            "target_language": language,
            "source_text": str(config.source_text_path),
            "source_pdf": str(config.source_pdf_path),
            "workflow_mode": config.workflow_mode,
            "evaluation_mode": config.evaluation_mode,
            "translation_profile": config.translation_profile,
            "aggregation_model": config.default_aggregation_model,
            "title_translation_model": config.title_translation_model_ref(),
            "page_indexing": "1-based physical PDF pages",
        },
        "title": {"original": source_title, "translated": translated_title},
        "characters_names": character_name_map(glossary, language),
        "translation": translations,
    }
