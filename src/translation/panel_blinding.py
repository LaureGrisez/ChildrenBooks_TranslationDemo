"""Candidate blinding helpers for independent panel judges."""

from __future__ import annotations

import hashlib
import random
import string
from typing import Any


def blind_options(
    options: dict[str, str], *, seed: str, paragraph_id: str, judge_id: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Return shuffled neutral options and a private option-to-candidate mapping."""

    digest = hashlib.sha256(f"{seed}:{paragraph_id}:{judge_id}".encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    candidates = list(options)
    rng.shuffle(candidates)
    labels = [f"option_{letter}" for letter in string.ascii_lowercase]
    mapping = {labels[index]: candidate for index, candidate in enumerate(candidates)}
    blinded = {option_id: options[candidate] for option_id, candidate in mapping.items()}
    return blinded, mapping


def restore_judge_result(
    result: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    """Restore blinded option IDs to internal candidate names."""

    def restore(value: Any) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {mapping.get(key, key): restore(item) for key, item in value.items()}
        return value

    return restore(result)
