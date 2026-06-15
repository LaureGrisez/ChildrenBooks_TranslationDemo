"""Structured contracts used by panel evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CRITERIA = (
    "faithfulness",
    "naturalness",
    "child_friendliness",
    "read_aloud",
    "continuity",
    "glossary_compliance",
    "structure",
)


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """One independent panel judge."""

    judge_id: str
    provider: str
    model: str


def parse_judge_specs(raw_specs: list[tuple[str, str]]) -> list[JudgeSpec]:
    """Create stable judge IDs from configured provider/model pairs."""

    return [
        JudgeSpec(judge_id=f"judge_{index + 1}", provider=provider, model=model)
        for index, (provider, model) in enumerate(raw_specs)
    ]


def validate_judge_result(payload: dict[str, Any], option_ids: list[str]) -> dict[str, Any]:
    """Validate the minimum structured contract required for aggregation."""

    ranking = payload.get("overall_ranking")
    if not isinstance(ranking, list) or set(ranking) != set(option_ids):
        raise ValueError("overall_ranking must contain every option exactly once.")
    if len(ranking) != len(set(ranking)):
        raise ValueError("overall_ranking contains duplicate options.")

    scores = payload.get("option_scores")
    if not isinstance(scores, dict) or set(scores) != set(option_ids):
        raise ValueError("option_scores must contain every option.")
    for option_id, option_scores in scores.items():
        if not isinstance(option_scores, dict):
            raise ValueError(f"Scores for {option_id} must be an object.")
        for criterion in CRITERIA:
            score = option_scores.get(criterion)
            if not isinstance(score, (int, float)) or not 0 <= score <= 10:
                raise ValueError(f"{option_id}.{criterion} must be between 0 and 10.")
        if not isinstance(option_scores.get("critical_errors", []), list):
            raise ValueError(f"{option_id}.critical_errors must be a list.")
        if not isinstance(option_scores.get("remarks", []), list):
            raise ValueError(f"{option_id}.remarks must be a list.")

    confidence = payload.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    payload["confidence"] = float(confidence)
    payload.setdefault("comparisons", [])
    payload.setdefault("recommended_phrases", [])
    return payload
