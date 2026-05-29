"""Configuration helpers for the translation workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_TEXT_PATH = REPO_ROOT / "l_arbre_de_barbapapa_INT.repaired.txt"
DEFAULT_SOURCE_PDF_PATH = REPO_ROOT / "flag_ship__l_arbre_de_barbapapa_INT.pdf"
DEFAULT_CHARACTER_NAMES_CSV = REPO_ROOT / "Noms barbapapas - Sheet1.csv"
DEFAULT_TRANSLATION_OUTPUT_DIR = REPO_ROOT / "translation"
DEFAULT_TRANSLATION_CACHE_DIR = REPO_ROOT / ".translation_cache"
DEFAULT_GOOGLE_CREDENTIALS = (
    REPO_ROOT / "credentials" / "children-book-translation-4e958984d7f8.json"
)
DEFAULT_LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "portuguese": "pt",
    "german": "de",
    "spanish": "es",
    "swedish": "sv",
    "italian": "it",
    "finnish": "fi",
    "hindi": "hi",
    "tamil": "ta",
}


def _split_csv_env(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_run_id() -> str:
    """Create a readable per-run identifier for artifact storage."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    """Normalize model and candidate names for safe filenames."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip(".-_")
    return slug or "unknown"


@dataclass(slots=True)
class TranslationWorkflowConfig:
    """Runtime configuration for the Burr translation workflow."""

    source_language: str = "French"
    prompt_language: str = "English"
    source_text_path: Path = DEFAULT_SOURCE_TEXT_PATH
    source_pdf_path: Path | None = DEFAULT_SOURCE_PDF_PATH
    character_names_csv: Path = DEFAULT_CHARACTER_NAMES_CSV
    translation_output_dir: Path = DEFAULT_TRANSLATION_OUTPUT_DIR
    translation_cache_dir: Path = DEFAULT_TRANSLATION_CACHE_DIR
    workflow_mode: str = field(
        default_factory=lambda: os.getenv("WORKFLOW_MODE", "text").strip().lower()
    )
    source_context_window: int = field(
        default_factory=lambda: max(0, int(os.getenv("SOURCE_CONTEXT_WINDOW", "0")))
    )
    target_history_window: int = field(
        default_factory=lambda: max(0, int(os.getenv("TARGET_HISTORY_WINDOW", "1")))
    )
    pdf_skip_first: int = field(
        default_factory=lambda: max(0, int(os.getenv("PDF_SKIP_FIRST", "5")))
    )
    pdf_skip_last: int = field(
        default_factory=lambda: max(0, int(os.getenv("PDF_SKIP_LAST", "4")))
    )
    multimodal_image_dpi: int = field(
        default_factory=lambda: max(72, int(os.getenv("MULTIMODAL_IMAGE_DPI", "110")))
    )
    project_name: str = field(
        default_factory=lambda: os.getenv(
            "BURR_PROJECT", "children-book-translation-advanced"
        )
    )
    burr_storage_dir: str = field(
        default_factory=lambda: os.path.expanduser(
            os.getenv("BURR_STORAGE_DIR", "~/.burr")
        )
    )
    external_translator: str = field(
        default_factory=lambda: os.getenv("EXTERNAL_TRANSLATOR", "google").lower()
    )
    max_parallel_candidates: int = field(
        default_factory=lambda: min(
            5,
            max(2, int(os.getenv("MAX_PARALLEL_CANDIDATES", "3"))),
        )
    )
    candidate_names: list[str] = field(default_factory=list)
    openai_base_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
    )
    openai_adversarial_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_ADVERSARIAL_MODEL", "gpt-5.5")
    )
    anthropic_sonnet_model: str = field(
        default_factory=lambda: os.getenv(
            "ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-6"
        )
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3")
    )
    openai_critic_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_CRITIC_MODEL", os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
        )
    )
    openai_final_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_FINAL_MODEL", os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
        )
    )
    language_code_overrides: dict[str, str] = field(default_factory=dict)
    target_languages: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=_build_run_id)
    enable_cache: bool = field(
        default_factory=lambda: os.getenv("TRANSLATION_CACHE", "1").strip() != "0"
    )
    openai_retry_attempts: int = field(
        default_factory=lambda: max(1, int(os.getenv("OPENAI_RETRY_ATTEMPTS", "4")))
    )
    openai_retry_base_delay_seconds: float = field(
        default_factory=lambda: max(
            0.5, float(os.getenv("OPENAI_RETRY_BASE_DELAY_SECONDS", "2.0"))
        )
    )

    @classmethod
    def from_env(cls) -> "TranslationWorkflowConfig":
        config = cls()
        config.language_code_overrides = {
            pair.split(":", 1)[0].strip().lower(): pair.split(":", 1)[1].strip()
            for pair in os.getenv("LANGUAGE_CODES", "").split(",")
            if ":" in pair
        }
        raw_targets = os.getenv("TARGET_LANGUAGES", "").strip()
        if raw_targets:
            config.target_languages = _split_csv_env(raw_targets)
        raw_candidate_names = os.getenv("CANDIDATE_NAMES", "").strip()
        if raw_candidate_names:
            config.candidate_names = _split_csv_env(raw_candidate_names)
        if DEFAULT_GOOGLE_CREDENTIALS.exists():
            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS", str(DEFAULT_GOOGLE_CREDENTIALS)
            )
        return config

    def load_source_text(self) -> str:
        """Load the French source text from disk."""

        return self.source_text_path.read_text(encoding="utf-8").strip()

    def is_multimodal_mode(self) -> bool:
        """Return whether the workflow should use spread images in generation."""

        return self.workflow_mode == "multimodal"

    def book_name(self) -> str:
        """Return a stable book slug derived from the source filename."""

        name = self.source_text_path.stem
        name = re.sub(r"\.(repaired|from_extractor_repaired)$", "", name)
        return name

    def latest_translation_path(self, language: str) -> Path:
        """Return the stable top-level output path for one language."""

        language_indicator = self.language_code(language)
        return self.translation_output_dir / (
            f"{self.book_name()}_{language_indicator}.txt"
        )

    def language_run_dir(self, language: str) -> Path:
        """Return the per-language, per-run artifact directory."""

        return self.translation_output_dir / self.language_code(language) / self.run_id

    def versioned_translation_path(self, language: str) -> Path:
        """Return the per-run final translation path."""

        model_slug = _slugify(self.openai_final_model)
        return self.language_run_dir(language) / (
            f"{self.book_name()}_{model_slug}.txt"
        )

    def candidate_output_path(
        self, language: str, candidate_name: str, candidate_model: str
    ) -> Path:
        """Return the per-run candidate output path."""

        candidate_slug = _slugify(candidate_name)
        model_slug = _slugify(candidate_model)
        return self.language_run_dir(language) / "candidates" / (
            f"{self.book_name()}_{candidate_slug}_{model_slug}.txt"
        )

    def report_output_path(self, language: str) -> Path:
        """Return the per-run Markdown report path."""

        model_slug = _slugify(self.openai_final_model)
        return self.language_run_dir(language) / (
            f"report_{self.book_name()}_{model_slug}.md"
        )

    def language_code(self, language: str) -> str:
        """Return the configured filesystem/API code for a target language."""

        key = language.strip().lower()
        return self.language_code_overrides.get(key, DEFAULT_LANGUAGE_CODES.get(key, key))
