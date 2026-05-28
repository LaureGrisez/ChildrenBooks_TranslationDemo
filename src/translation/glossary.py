"""Load and format Barbapapa character names across languages."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CharacterGlossary:
    """Character-name lookup table keyed by language."""

    source_language: str
    rows: list[dict[str, str]]
    supported_languages: list[str]

    def default_target_languages(self) -> list[str]:
        """Return all supported targets except the source language."""

        return [
            language
            for language in self.supported_languages
            if language.lower() != self.source_language.lower()
        ]

    def has_language(self, language: str) -> bool:
        """Return True if the language exists in the CSV."""

        return any(
            candidate.lower() == language.lower()
            for candidate in self.supported_languages
        )

    def normalize_language(self, language: str) -> str:
        """Return the canonical header name for a language."""

        for candidate in self.supported_languages:
            if candidate.lower() == language.lower():
                return candidate
        raise KeyError(f"Unsupported language: {language}")

    def format_name_guidance(self, target_language: str) -> str:
        """Build explicit character-name guidance for prompts and logs."""

        normalized_target = self.normalize_language(target_language)
        normalized_source = self.normalize_language(self.source_language)
        lines = []
        for row in self.rows:
            source_name = row[normalized_source].strip()
            target_name = row[normalized_target].strip()
            if not source_name or not target_name:
                continue
            lines.append(f"- {source_name} -> {target_name}")
        return "\n".join(lines)


def load_character_glossary(
    csv_path: Path, source_language: str = "French"
) -> CharacterGlossary:
    """Load the language table from CSV."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]

    if not reader.fieldnames:
        raise ValueError(f"No headers found in glossary CSV: {csv_path}")

    supported_languages = [header.strip() for header in reader.fieldnames]
    if source_language not in supported_languages:
        raise ValueError(
            f"Source language '{source_language}' not found in glossary CSV headers."
        )

    return CharacterGlossary(
        source_language=source_language,
        rows=rows,
        supported_languages=supported_languages,
    )
