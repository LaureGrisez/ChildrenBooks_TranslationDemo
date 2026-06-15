"""Configuration helpers for the translation workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


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
    evaluation_mode: str = field(
        default_factory=lambda: os.getenv("EVALUATION_MODE", "single").strip().lower()
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
    multimodal_save_debug_images: bool = field(
        default_factory=lambda: os.getenv("MULTIMODAL_SAVE_DEBUG_IMAGES", "0").strip()
        != "0"
    )
    multimodal_jpeg_quality: int = field(
        default_factory=lambda: min(
            95, max(20, int(os.getenv("MULTIMODAL_JPEG_QUALITY", "65")))
        )
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
    candidate_temperature: float | None = field(
        default_factory=lambda: (
            float(os.getenv("CANDIDATE_TEMPERATURE"))
            if os.getenv("CANDIDATE_TEMPERATURE", "").strip()
            else None
        )
    )
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
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    default_critic_model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_CRITIC_MODEL",
            "openai:"
            + os.getenv("OPENAI_CRITIC_MODEL", os.getenv("OPENAI_BASE_MODEL", "gpt-4o")),
        ).strip()
    )
    default_aggregation_model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_AGGREGATION_MODEL",
            "openai:"
            + os.getenv("OPENAI_FINAL_MODEL", os.getenv("OPENAI_BASE_MODEL", "gpt-4o")),
        ).strip()
    )
    default_critic_summarizer_model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_CRITIC_SUMMARIZER_MODEL",
            os.getenv(
                "DEFAULT_CRITIC_MODEL",
                "openai:"
                + os.getenv(
                    "OPENAI_CRITIC_MODEL", os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
                ),
            ),
        ).strip()
    )
    panel_judges: list[str] = field(default_factory=list)
    panel_judge_temperature: float = field(
        default_factory=lambda: float(os.getenv("PANEL_JUDGE_TEMPERATURE", "0.1"))
    )
    panel_max_parallel_judges: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("PANEL_MAX_PARALLEL_JUDGES", "3"))
        )
    )
    panel_source_context_window: int = field(
        default_factory=lambda: max(
            0, int(os.getenv("PANEL_SOURCE_CONTEXT_WINDOW", "1"))
        )
    )
    panel_target_history_window: int = field(
        default_factory=lambda: max(
            0, int(os.getenv("PANEL_TARGET_HISTORY_WINDOW", "1"))
        )
    )
    panel_pairwise_weight: float = field(
        default_factory=lambda: float(os.getenv("PANEL_PAIRWISE_WEIGHT", "0.45"))
    )
    panel_ranking_weight: float = field(
        default_factory=lambda: float(os.getenv("PANEL_RANKING_WEIGHT", "0.35"))
    )
    panel_score_weight: float = field(
        default_factory=lambda: float(os.getenv("PANEL_SCORE_WEIGHT", "0.20"))
    )
    panel_critical_error_confirmations: int = field(
        default_factory=lambda: max(
            1, int(os.getenv("PANEL_CRITICAL_ERROR_CONFIRMATIONS", "2"))
        )
    )
    panel_random_seed: str = field(
        default_factory=lambda: os.getenv("PANEL_RANDOM_SEED", "").strip()
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
        load_dotenv()
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
        raw_panel_judges = os.getenv("PANEL_JUDGES", "").strip()
        if raw_panel_judges:
            config.panel_judges = _split_csv_env(raw_panel_judges)
        config.validate()
        config.apply_environment_defaults()
        return config

    def apply_environment_defaults(self) -> None:
        """Expose repository-local credentials when no explicit override exists."""

        if DEFAULT_GOOGLE_CREDENTIALS.exists():
            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS", str(DEFAULT_GOOGLE_CREDENTIALS)
            )

    def validate(self) -> None:
        """Reject invalid workflow combinations before model calls begin."""

        if self.workflow_mode not in {"text", "multimodal"}:
            raise ValueError("WORKFLOW_MODE must be either 'text' or 'multimodal'.")
        if self.evaluation_mode not in {"single", "panel"}:
            raise ValueError(
                "EVALUATION_MODE must be 'single' or 'panel'. "
                "adaptive_panel is not implemented yet."
            )
        weights = (
            self.panel_pairwise_weight,
            self.panel_ranking_weight,
            self.panel_score_weight,
        )
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Panel aggregation weights must be non-negative and sum to 1.")
        invalid_judges = []
        normalized_judges = []
        for judge in self.panel_judges:
            if ":" not in judge:
                invalid_judges.append(judge)
                continue
            try:
                provider, model = self.parse_model_ref(judge)
            except ValueError:
                invalid_judges.append(judge)
                continue
            normalized_judges.append(f"{provider}:{model}")
        if invalid_judges:
            raise ValueError(f"Invalid PANEL_JUDGES entries: {', '.join(invalid_judges)}")
        for variable, model_ref in (
            ("DEFAULT_CRITIC_MODEL", self.default_critic_model),
            ("DEFAULT_AGGREGATION_MODEL", self.default_aggregation_model),
            ("DEFAULT_CRITIC_SUMMARIZER_MODEL", self.default_critic_summarizer_model),
        ):
            try:
                self.parse_model_ref(model_ref)
            except ValueError as exc:
                raise ValueError(f"Invalid {variable}: {exc}") from exc
        if self.evaluation_mode == "panel" and self.panel_judges:
            if len(normalized_judges) < 2 or len(set(normalized_judges)) < 2:
                raise ValueError(
                    "Panel mode requires at least two distinct judge models."
                )
            if len(normalized_judges) != len(set(normalized_judges)):
                raise ValueError("Panel judge models must be unique.")
        if self.candidate_temperature is not None and not 0 <= self.candidate_temperature <= 2:
            raise ValueError("CANDIDATE_TEMPERATURE must be between 0 and 2.")
        if not 0 <= self.panel_judge_temperature <= 2:
            raise ValueError("PANEL_JUDGE_TEMPERATURE must be between 0 and 2.")

    def load_source_text(self) -> str:
        """Load the French source text from disk."""

        return self.source_text_path.read_text(encoding="utf-8").strip()

    def is_multimodal_mode(self) -> bool:
        """Return whether the workflow should use spread images in generation."""

        return self.workflow_mode == "multimodal"

    def is_panel_mode(self) -> bool:
        """Return whether candidates should use panel evaluation."""

        return self.evaluation_mode == "panel"

    def panel_judge_specs(self) -> list[tuple[str, str]]:
        """Return configured panel judges as provider/model pairs."""

        return [self.parse_model_ref(item) for item in self.panel_judges]

    @staticmethod
    def parse_model_ref(model_ref: str) -> tuple[str, str]:
        """Parse a provider-qualified model, defaulting bare names to OpenAI."""

        provider, separator, model = model_ref.partition(":")
        if not separator:
            provider, model = "openai", provider
        if provider not in {"openai", "anthropic", "gemini"} or not model.strip():
            raise ValueError(
                "model references must use 'provider:model' with provider "
                "openai, anthropic, or gemini"
            )
        return provider, model.strip()

    def critic_model_spec(self) -> tuple[str, str]:
        """Return the provider and model used for primary criticism and audits."""

        return self.parse_model_ref(self.default_critic_model)

    def aggregation_model_spec(self) -> tuple[str, str]:
        """Return the provider and model used to synthesize frozen judgments."""

        return self.parse_model_ref(self.default_aggregation_model)

    def critic_summarizer_model_spec(self) -> tuple[str, str]:
        """Return the provider and model used to summarize critic findings."""

        return self.parse_model_ref(self.default_critic_summarizer_model)

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

        model_slug = _slugify(self.default_aggregation_model)
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

        model_slug = _slugify(self.default_aggregation_model)
        return self.language_run_dir(language) / (
            f"report_{self.book_name()}_{model_slug}.md"
        )

    def language_code(self, language: str) -> str:
        """Return the configured filesystem/API code for a target language."""

        key = language.strip().lower()
        return self.language_code_overrides.get(key, DEFAULT_LANGUAGE_CODES.get(key, key))

    def multimodal_debug_dir(self) -> Path:
        """Return the per-run debug directory for rendered spread images."""

        return self.translation_output_dir / "_multimodal_debug" / self.run_id

    def panel_artifact_dir(self, language: str) -> Path:
        """Return the panel artifact directory for one language."""

        return self.language_run_dir(language) / "panel"
