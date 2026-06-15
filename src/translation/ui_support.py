"""Framework-independent helpers for the Streamlit translation UI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .config import TranslationWorkflowConfig
from .reporting import build_pairwise_comparison_report


GENERATION_MODE_OPTIONS = {
    "Text only": "text",
    "Multimodal": "multimodal",
}

JUDGING_MODE_OPTIONS = {
    "Single judge": "single",
    "Panel judges": "panel",
}

CANDIDATE_OPTIONS = {
    "Google Translate": "google_translation",
    "OpenAI base": "gpt4o",
    "OpenAI adversarial": "gpt5_5",
    "Anthropic Sonnet": "claude_sonnet_4_6",
    "Google Gemini": "gemini_3",
}


def missing_candidate_credentials(candidate_names: list[str]) -> list[str]:
    """Return human-readable credential requirements missing for candidates."""

    missing = []
    if "claude_sonnet_4_6" in candidate_names and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("Anthropic Sonnet requires ANTHROPIC_API_KEY.")
    if "gemini_3" in candidate_names and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        missing.append("Google Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY.")
    return missing


def missing_judge_credentials(panel_judges: list[str]) -> list[str]:
    """Return human-readable credential requirements missing for judges."""

    missing = []
    providers = {judge.partition(":")[0] for judge in panel_judges}
    if "anthropic" in providers and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("An Anthropic judge requires ANTHROPIC_API_KEY.")
    if "gemini" in providers and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        missing.append("A Google Gemini judge requires GEMINI_API_KEY or GOOGLE_API_KEY.")
    return missing


def workflow_mermaid(generation_mode_label: str, judging_mode_label: str) -> str:
    """Return the workflow graph for one generation/evaluation combination."""

    if judging_mode_label == "Panel judges":
        generation_steps = (
            """A[Upload source text + PDF] --> B[Align text to spreads]
    B --> C[Render textless spread images]
    C --> D[Generate multimodal candidates]
    D --> E[Align candidate paragraphs]"""
            if generation_mode_label == "Multimodal"
            else """A[Upload source text] --> D[Generate text candidates]
    D --> E[Align candidate paragraphs]"""
        )
        return """flowchart LR
    %s
    E --> F[Blind options per judge]
    F --> G[Independent judge panel]
    G --> H[Deterministic aggregation]
    H --> I[Sequential paragraph synthesis]
    I --> J[Whole-book audit]
    J --> K[Targeted repairs]
    K --> L[Download final translation]
    classDef llm fill:#fde7d7,stroke:#c2410c,stroke-width:2px,color:#431407
    class D,G,I,J,K llm""" % generation_steps
    if generation_mode_label == "Multimodal":
        return """flowchart LR
    A[Upload source text + PDF] --> B[Align text to spreads]
    B --> C[Render textless spread images]
    C --> D[Generate multimodal candidates]
    D --> E[Single critic]
    E --> F[Critic summary]
    F --> G[Final synthesis]
    G --> H[Download final translation]
    classDef llm fill:#fde7d7,stroke:#c2410c,stroke-width:2px,color:#431407
    class D,E,F,G llm"""
    return """flowchart LR
    A[Upload source text] --> B[Generate text candidates]
    B --> C[Single critic]
    C --> D[Critic summary]
    D --> E[Final synthesis]
    E --> F[Download final translation]
    classDef llm fill:#fde7d7,stroke:#c2410c,stroke-width:2px,color:#431407
    class B,C,D,E llm"""


def workflow_node_descriptions(
    generation_mode_label: str, judging_mode_label: str
) -> dict[str, str]:
    """Return descriptions for one generation/evaluation combination."""

    if judging_mode_label == "Panel judges":
        descriptions = {
            "Align candidate paragraphs": "Checks that every successful candidate preserves the exact source paragraph structure.",
            "Blind options per judge": "Removes candidate identities and independently shuffles option order for each judge.",
            "Independent judge panel": "Calls independent judge models to rank and score every blinded paragraph option.",
            "Deterministic aggregation": "Combines pairwise results, rankings, normalized scores, and confirmed critical errors in deterministic code.",
            "Sequential paragraph synthesis": "Calls the final model sequentially to synthesize one approved paragraph at a time.",
            "Whole-book audit": "Calls an audit model to identify book-level consistency problems without rewriting the whole book.",
            "Targeted repairs": "Calls the final model only for paragraphs explicitly flagged by the audit.",
            "Download final translation": "Exports the assembled and repaired final translation as a text file.",
        }
        if generation_mode_label == "Multimodal":
            return {
                "Upload source text + PDF": "Loads cleaned page-aligned source text and the illustrated source PDF.",
                "Align text to spreads": "Maps source paragraphs to text-bearing pages and groups them into double-page spreads.",
                "Render textless spread images": "Creates textless spread images so candidate models can use visible actions and scene details.",
                "Generate multimodal candidates": "Calls selected candidate models using source text, spread images, glossary, and recent translation context.",
                **descriptions,
            }
        return {
            "Upload source text": "Loads the cleaned source text whose paragraphs define the evaluation units.",
            "Generate text candidates": "Calls selected candidate models or machine translation to produce alternative full-book translations.",
            **descriptions,
        }
    if generation_mode_label == "Multimodal":
        return {
            "Upload source text + PDF": "Loads cleaned page-aligned source text and the illustrated source PDF.",
            "Align text to spreads": "Maps source paragraphs to text-bearing pages and groups them into double-page spreads.",
            "Render textless spread images": "Creates textless spread images so models can use visible actions and scene details.",
            "Generate multimodal candidates": "Calls selected models for each spread using source text, image, glossary, and recent translation context.",
            "Single critic": "Calls one critic model to compare the completed candidate translations.",
            "Critic summary": "Calls the critic model to condense its review into actionable final-synthesis guidance.",
            "Final synthesis": "Calls the final model once per source page paragraph, preserving PDF overlay alignment in code.",
            "Download final translation": "Exports the page-aligned final translation as a text file.",
        }
    return {
        "Upload source text": "Loads the cleaned source text to translate.",
        "Generate text candidates": "Calls the selected translation models or machine-translation provider to produce alternative full-book translations.",
        "Single critic": "Calls one critic model to compare candidate quality, tradeoffs, and paragraph-level choices.",
        "Critic summary": "Calls the critic model to condense its review into actionable final-synthesis guidance.",
        "Final synthesis": "Calls the final model to synthesize one coherent full-book translation.",
        "Download final translation": "Exports the final translation as a text file.",
    }


def safe_upload_name(name: str, fallback: str) -> str:
    """Normalize an uploaded filename before writing it under the repo."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip(".-_")
    return normalized or fallback


def build_ui_config(
    *,
    generation_mode_label: str,
    judging_mode_label: str,
    source_text_path: Path,
    source_pdf_path: Path | None,
    target_language: str,
    candidate_names: list[str],
    candidate_temperature: float,
    openai_base_model: str,
    openai_adversarial_model: str,
    output_dir: Path,
    cache_dir: Path,
    burr_storage_dir: str,
    anthropic_sonnet_model: str = "claude-sonnet-4-6",
    gemini_model: str = "gemini-2.5-flash",
    panel_judges: list[str] | None = None,
    panel_judge_temperature: float = 0.1,
) -> TranslationWorkflowConfig:
    """Construct a validated workflow configuration from UI selections."""

    workflow_mode = GENERATION_MODE_OPTIONS[generation_mode_label]
    evaluation_mode = JUDGING_MODE_OPTIONS[judging_mode_label]
    config = TranslationWorkflowConfig(
        source_text_path=source_text_path,
        source_pdf_path=source_pdf_path,
        translation_output_dir=output_dir,
        translation_cache_dir=cache_dir,
        burr_storage_dir=burr_storage_dir,
        target_languages=[target_language],
        workflow_mode=workflow_mode,
        evaluation_mode=evaluation_mode,
        max_parallel_candidates=len(candidate_names),
        candidate_names=candidate_names,
        candidate_temperature=candidate_temperature,
        openai_base_model=openai_base_model,
        openai_adversarial_model=openai_adversarial_model,
        anthropic_sonnet_model=anthropic_sonnet_model,
        gemini_model=gemini_model,
        panel_judge_temperature=panel_judge_temperature,
    )
    if evaluation_mode == "panel" and panel_judges is not None:
        config.panel_judges = panel_judges
    config.validate()
    return config


def collect_versions(state: dict[str, Any], language: str) -> dict[str, str]:
    """Collect candidate and final texts for comparison and download."""

    versions = {
        f"Candidate: {candidate['name']}": candidate.get("text", "")
        for candidate in state.get("candidate_translations", {}).get(language, [])
        if candidate.get("text")
    }
    final = state.get("final_translations", {}).get(language)
    final_paragraphs = state.get("final_paragraphs", {}).get(language, {})
    if final_paragraphs:
        ordered = sorted(final_paragraphs.items())
        versions["Panel synthesis before audit repairs"] = "\n\n".join(
            text for _, text in ordered
        )
    if final:
        versions["Final translation"] = final
    return versions


def comparison_report(
    versions: dict[str, str], left_label: str, right_label: str
) -> str:
    """Build the existing side-by-side comparison report for UI rendering."""

    return build_pairwise_comparison_report(
        left_label=left_label,
        right_label=right_label,
        left_text=versions[left_label],
        right_text=versions[right_label],
        title="Translation Version Comparison",
    )


def panel_score_rows(state: dict[str, Any], language: str) -> list[dict[str, Any]]:
    """Flatten per-judge panel criterion scores for tabular display."""

    rows = []
    results = state.get("panel_judge_results", {}).get(language, {})
    for paragraph_id, judges in results.items():
        for judge_id, judgment in judges.items():
            if judgment.get("status") != "ok":
                rows.append(
                    {
                        "paragraph": paragraph_id,
                        "judge": judge_id,
                        "provider": judgment.get("provider", ""),
                        "candidate": "",
                        "score": "",
                        "rank": "",
                        "status": judgment.get("error", "error"),
                    }
                )
                continue
            result = judgment["result"]
            ranks = {
                candidate: index + 1
                for index, candidate in enumerate(result["overall_ranking"])
            }
            for candidate, scores in result["option_scores"].items():
                numeric = [
                    value
                    for key, value in scores.items()
                    if key not in {"critical_errors", "remarks"}
                    and isinstance(value, (int, float))
                ]
                rows.append(
                    {
                        "paragraph": paragraph_id,
                        "judge": judge_id,
                        "provider": judgment.get("provider", ""),
                        "candidate": candidate,
                        "score": round(sum(numeric) / len(numeric), 2) if numeric else "",
                        "rank": ranks.get(candidate, ""),
                        "status": "ok",
                    }
                )
    return rows


def panel_aggregate_rows(state: dict[str, Any], language: str) -> list[dict[str, Any]]:
    """Flatten aggregate paragraph rankings for tabular display."""

    rows = []
    aggregates = state.get("panel_aggregates", {}).get(language, {})
    for paragraph_id, aggregate in aggregates.items():
        for rank, candidate in enumerate(aggregate["ranking"], start=1):
            rows.append(
                {
                    "paragraph": paragraph_id,
                    "rank": rank,
                    "candidate": candidate,
                    "aggregate_score": round(aggregate["total_scores"][candidate], 4),
                    "confirmed_critical_errors": json.dumps(
                        aggregate["confirmed_critical_errors"][candidate],
                        ensure_ascii=False,
                    ),
                }
            )
    return rows
