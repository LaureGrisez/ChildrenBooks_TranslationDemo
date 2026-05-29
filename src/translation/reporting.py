"""Artifact persistence and comparison reporting for translation runs."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import TranslationWorkflowConfig


GREEN = "#dcfce7"
RED = "#fee2e2"
NEUTRAL = "#f8fafc"


@dataclass(slots=True)
class ArtifactBundle:
    """Paths written for one workflow execution."""

    latest_final_paths: dict[str, Path] = field(default_factory=dict)
    versioned_final_paths: dict[str, Path] = field(default_factory=dict)
    candidate_paths: dict[str, dict[str, Path]] = field(default_factory=dict)
    report_paths: dict[str, Path] = field(default_factory=dict)


def persist_run_artifacts(
    state: dict[str, Any], config: TranslationWorkflowConfig
) -> ArtifactBundle:
    """Write any available candidates, finals, and reports to disk."""

    bundle = ArtifactBundle()
    config.translation_output_dir.mkdir(parents=True, exist_ok=True)

    candidate_translations = state.get("candidate_translations", {})
    final_translations = state.get("final_translations", {})
    critic_reviews = state.get("critic_reviews", {})
    critic_reasoning = state.get("critic_reasoning", {})
    critic_winners = state.get("critic_winners", {})

    for language, candidates in candidate_translations.items():
        language_candidates = {}
        for candidate in candidates:
            candidate_path = config.candidate_output_path(
                language,
                candidate["name"],
                candidate.get("model", "unknown"),
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            _write_candidate_artifact(candidate_path, candidate)
            language_candidates[candidate["name"]] = candidate_path
        bundle.candidate_paths[language] = language_candidates

    for language, final_text in final_translations.items():
        latest_path = config.latest_translation_path(language)
        versioned_path = config.versioned_translation_path(language)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        versioned_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_final = final_text.strip() + "\n"
        latest_path.write_text(normalized_final, encoding="utf-8")
        versioned_path.write_text(normalized_final, encoding="utf-8")
        bundle.latest_final_paths[language] = latest_path
        bundle.versioned_final_paths[language] = versioned_path

        report_path = config.report_output_path(language)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_translation_report(
                language=language,
                candidates=candidate_translations.get(language, []),
                final_text=final_text,
                critic_review_raw=critic_reviews.get(language, ""),
                critic_reasoning=critic_reasoning.get(language, ""),
                critic_winner=critic_winners.get(language, "unknown"),
                config=config,
            ),
            encoding="utf-8",
        )
        bundle.report_paths[language] = report_path

    return bundle


def _write_candidate_artifact(candidate_path: Path, candidate: dict[str, Any]) -> None:
    """Write one candidate file without clobbering valid content with blanks."""

    candidate_text = candidate.get("text") or candidate.get("error") or ""
    normalized = candidate_text.strip()

    if not normalized and candidate_path.exists():
        existing = candidate_path.read_text(encoding="utf-8").strip()
        if existing:
            return

    candidate_path.write_text(normalized + "\n", encoding="utf-8")


def build_translation_report(
    *,
    language: str,
    candidates: list[dict[str, Any]],
    final_text: str,
    critic_review_raw: str,
    critic_reasoning: str,
    critic_winner: str,
    config: TranslationWorkflowConfig,
) -> str:
    """Create a Markdown report comparing the winner to the final text."""

    critique = _parse_review_payload(critic_review_raw)
    best_candidate = next(
        (candidate for candidate in candidates if candidate["name"] == critic_winner),
        None,
    )
    if best_candidate is None and candidates:
        best_candidate = candidates[0]

    lines = [
        f"# Translation Report: {config.book_name()} / {language}",
        "",
        f"- Run ID: `{config.run_id}`",
        f"- Final model: `{config.openai_final_model}`",
        f"- Critic winner: `{critic_winner}`",
        "",
        "## ROUGE-L Similarity To Final",
        "",
        "| Candidate | Status | ROUGE-L F1 | Notes |",
        "| --- | --- | ---: | --- |",
    ]

    for candidate in candidates:
        score = rouge_l_f1(candidate.get("text", ""), final_text)
        notes = "(best candidate)" if candidate["name"] == critic_winner else ""
        lines.append(
            f"| `{candidate['name']}` | {candidate['status']} | {score:.4f} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Critic Remarks On Best Candidate",
            "",
        ]
    )
    lines.extend(_render_best_candidate_remarks(critique, critic_winner, critic_reasoning))

    lines.extend(
        [
            "",
            "## Best Candidate Vs Final",
            "",
            _render_side_by_side_diff(best_candidate, final_text, critic_winner),
            "",
            "Legend: green highlights mark unchanged spans; red highlights mark edited or replaced spans.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pairwise_comparison_report(
    *,
    left_label: str,
    right_label: str,
    left_text: str,
    right_text: str,
    title: str | None = None,
) -> str:
    """Create a Markdown report comparing any two text files."""

    rouge_score = rouge_l_f1(left_text, right_text)
    left_candidate = {"name": left_label, "text": left_text}

    lines = [
        f"# {title or 'Text Comparison Report'}",
        "",
        f"- Left text: `{left_label}`",
        f"- Right text: `{right_label}`",
        "",
        "## ROUGE-L Similarity",
        "",
        f"- ROUGE-L F1: `{rouge_score:.4f}`",
        "",
        "## Side-By-Side Comparison",
        "",
        _render_side_by_side_diff(
            left_candidate,
            right_text,
            right_label,
            left_heading=left_label,
            right_heading=right_label,
        ),
        "",
        "Legend: green highlights mark unchanged spans; red highlights mark edited or replaced spans.",
        "",
    ]
    return "\n".join(lines)


def rouge_l_f1(left: str, right: str) -> float:
    """Compute a simple token-level ROUGE-L F1 score."""

    left_tokens = _tokenize_for_scoring(left)
    right_tokens = _tokenize_for_scoring(right)
    if not left_tokens or not right_tokens:
        return 0.0

    lcs = _lcs_length(left_tokens, right_tokens)
    precision = lcs / len(left_tokens)
    recall = lcs / len(right_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Compute LCS length with a compact dynamic-programming table."""

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _tokenize_for_scoring(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _split_paragraphs(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return re.split(r"\n\s*\n", stripped)


def _parse_review_payload(raw_review: str) -> dict[str, Any]:
    if not raw_review.strip():
        return {}
    try:
        return json.loads(raw_review)
    except json.JSONDecodeError:
        return {}


def _render_best_candidate_remarks(
    critique: dict[str, Any], critic_winner: str, critic_reasoning: str
) -> list[str]:
    lines = []
    assessment = next(
        (
            item
            for item in critique.get("candidate_assessment", [])
            if item.get("candidate") == critic_winner
        ),
        {},
    )

    if critic_reasoning.strip():
        lines.append("### Decision Reasoning")
        lines.append("")
        lines.append(critic_reasoning.strip())
        lines.append("")

    strengths = assessment.get("strengths", [])
    weaknesses = assessment.get("weaknesses", [])
    if strengths:
        lines.append("### Strengths")
        lines.append("")
        lines.extend(f"- {item}" for item in strengths)
        lines.append("")
    if weaknesses:
        lines.append("### Weaknesses")
        lines.append("")
        lines.extend(f"- {item}" for item in weaknesses)
        lines.append("")

    winner_notes = [
        item.get("notes", "").strip()
        for item in critique.get("paragraph_analysis", [])
        if item.get("best_candidate") == critic_winner and item.get("notes", "").strip()
    ]
    if winner_notes:
        lines.append("### Paragraph Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in winner_notes)
        lines.append("")

    if not lines:
        lines.extend(["No structured critic remarks were available for the winning candidate.", ""])
    return lines


def _render_side_by_side_diff(
    best_candidate: dict[str, Any] | None,
    final_text: str,
    critic_winner: str,
    left_heading: str = "Best Candidate",
    right_heading: str = "Final Translation",
) -> str:
    if best_candidate is None:
        return "No winning candidate text was available for diffing."

    left_paragraphs = _split_paragraphs(best_candidate.get("text", ""))
    right_paragraphs = _split_paragraphs(final_text)
    paragraph_count = max(len(left_paragraphs), len(right_paragraphs), 1)

    rows = [
        "<table>",
        (
            "  <tr><th>Paragraph</th>"
            f"<th>{html.escape(left_heading)}</th>"
            f"<th>{html.escape(right_heading)}</th></tr>"
        ),
    ]

    for index in range(paragraph_count):
        left = left_paragraphs[index] if index < len(left_paragraphs) else ""
        right = right_paragraphs[index] if index < len(right_paragraphs) else ""
        normalized_left = _normalize_for_display_diff(left)
        normalized_right = _normalize_for_display_diff(right)
        ratio = SequenceMatcher(None, normalized_left, normalized_right).ratio()
        similarity = _similarity_label(ratio)
        rows.append(
            "  <tr>"
            f"<td><strong>{index + 1}</strong><br/>{similarity}</td>"
            f"<td>{_diff_cell(normalized_left, normalized_right, left_side=True)}</td>"
            f"<td>{_diff_cell(normalized_left, normalized_right, left_side=False)}</td>"
            "</tr>"
        )

    rows.extend(["</table>", "", f"Compared against critic winner: `{critic_winner}`."])
    return "\n".join(rows)


def _diff_cell(left: str, right: str, *, left_side: bool) -> str:
    left_tokens = _display_tokens(left)
    right_tokens = _display_tokens(right)
    matcher = SequenceMatcher(None, left_tokens, right_tokens)
    fragments = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        current_tokens = left_tokens[i1:i2] if left_side else right_tokens[j1:j2]
        if not current_tokens:
            continue
        if tag == "equal":
            color = GREEN
        elif tag in {"replace", "delete", "insert"}:
            color = RED
        else:
            color = NEUTRAL
        fragments.append(_wrap_tokens(current_tokens, color))

    return "".join(fragments) or "&nbsp;"


def _display_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)


def _normalize_for_display_diff(text: str) -> str:
    """Remove whitespace-only noise so red highlights stay meaningful."""

    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _wrap_tokens(tokens: list[str], color: str) -> str:
    content = "".join(html.escape(token) for token in tokens)
    return (
        f"<span style=\"background-color:{color}; padding:0 1px;\">{content}</span>"
    )


def _similarity_label(ratio: float) -> str:
    if ratio >= 0.9:
        return f"<span style=\"color:#166534;\">High similarity ({ratio:.2%})</span>"
    if ratio >= 0.65:
        return f"<span style=\"color:#92400e;\">Medium similarity ({ratio:.2%})</span>"
    return f"<span style=\"color:#991b1b;\">Low similarity ({ratio:.2%})</span>"
