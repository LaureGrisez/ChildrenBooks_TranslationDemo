"""Persistence helpers for inspectable panel artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TranslationWorkflowConfig


def persist_panel_artifact(
    config: TranslationWorkflowConfig,
    language: str,
    name: str,
    payload: Any,
) -> Path:
    """Write one versioned JSON panel artifact."""

    path = config.panel_artifact_dir(language) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
