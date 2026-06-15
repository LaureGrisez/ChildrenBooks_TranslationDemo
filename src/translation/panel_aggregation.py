"""Deterministic aggregation for panel judgments."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import pvariance
from typing import Any

from .panel_models import CRITERIA


def _min_max(values: dict[str, float]) -> dict[str, float]:
    lowest = min(values.values())
    highest = max(values.values())
    if highest == lowest:
        return {key: 0.5 for key in values}
    return {key: (value - lowest) / (highest - lowest) for key, value in values.items()}


def _pairwise_points(ranking: list[str]) -> dict[str, float]:
    points = {option: 0.0 for option in ranking}
    possible = max(1, len(ranking) - 1)
    positions = {option: index for index, option in enumerate(ranking)}
    for left, right in combinations(ranking, 2):
        winner = left if positions[left] < positions[right] else right
        points[winner] += 1 / possible
    return points


def aggregate_judgments(
    judge_results: list[dict[str, Any]],
    *,
    pairwise_weight: float,
    ranking_weight: float,
    score_weight: float,
    critical_error_confirmations: int,
) -> dict[str, Any]:
    """Combine restored judge results into one inspectable paragraph decision."""

    if not judge_results:
        raise ValueError("At least one valid judge result is required.")
    options = list(judge_results[0]["overall_ranking"])
    totals = {option: 0.0 for option in options}
    pairwise_totals = {option: 0.0 for option in options}
    ranking_totals = {option: 0.0 for option in options}
    score_totals = {option: 0.0 for option in options}
    first_places = Counter()
    critical_errors: dict[str, list[str]] = defaultdict(list)
    remarks: dict[str, list[str]] = defaultdict(list)
    phrases: list[dict[str, Any]] = []
    normalized_by_judge = []

    for result in judge_results:
        confidence = float(result.get("confidence", 1.0))
        ranking = result["overall_ranking"]
        first_places[ranking[0]] += 1

        pairwise = _pairwise_points(ranking)
        borda = {
            option: (len(options) - 1 - index) / max(1, len(options) - 1)
            for index, option in enumerate(ranking)
        }
        criterion_normalized = {}
        for criterion in CRITERIA:
            criterion_normalized[criterion] = _min_max(
                {
                    option: float(result["option_scores"][option][criterion])
                    for option in options
                }
            )
        option_normalized = {
            option: sum(criterion_normalized[criterion][option] for criterion in CRITERIA)
            / len(CRITERIA)
            for option in options
        }
        normalized_by_judge.append(option_normalized)

        for option in options:
            pairwise_totals[option] += pairwise[option] * confidence
            ranking_totals[option] += borda[option] * confidence
            score_totals[option] += option_normalized[option] * confidence
            critical_errors[option].extend(
                str(error).strip()
                for error in result["option_scores"][option].get("critical_errors", [])
                if str(error).strip()
            )
            remarks[option].extend(
                str(remark).strip()
                for remark in result["option_scores"][option].get("remarks", [])
                if str(remark).strip()
            )
        phrases.extend(result.get("recommended_phrases", []))

    judge_count = len(judge_results)
    confirmed_errors = {}
    for option in options:
        counts = Counter(error.casefold() for error in critical_errors[option])
        confirmed_errors[option] = [
            error for error, count in counts.items() if count >= critical_error_confirmations
        ]
        pairwise_totals[option] /= judge_count
        ranking_totals[option] /= judge_count
        score_totals[option] /= judge_count
        totals[option] = (
            pairwise_weight * pairwise_totals[option]
            + ranking_weight * ranking_totals[option]
            + score_weight * score_totals[option]
        )
        if confirmed_errors[option]:
            totals[option] -= 1.0

    ranking = sorted(
        options,
        key=lambda option: (
            bool(confirmed_errors[option]),
            -totals[option],
            option,
        ),
    )
    score_variance = {
        option: pvariance(values[option] for values in normalized_by_judge)
        for option in options
    }
    consensus_remarks = {
        option: [
            remark
            for remark, count in Counter(item.casefold() for item in remarks[option]).items()
            if count >= 2
        ]
        for option in options
    }
    selected = ranking[: min(2, len(ranking))]
    if len(ranking) > 2:
        selected.append(ranking[2])

    return {
        "ranking": ranking,
        "total_scores": totals,
        "signals": {
            "pairwise": pairwise_totals,
            "ranking": ranking_totals,
            "normalized_scores": score_totals,
        },
        "confirmed_critical_errors": confirmed_errors,
        "remarks": dict(remarks),
        "consensus_remarks": consensus_remarks,
        "recommended_phrases": phrases,
        "selected_options": selected,
        "disagreement": {
            "different_first_place_choices": len(first_places),
            "first_place_votes": dict(first_places),
            "normalized_score_variance": score_variance,
        },
    }
