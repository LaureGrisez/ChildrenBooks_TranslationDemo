"""Persistent response cache for workflow recovery and reruns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """Small file-based cache keyed by provider/model/prompt payload."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        *,
        provider: str,
        model: str,
        temperature: float | None,
        prompt: str,
    ) -> Path:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.root_dir / provider / model / f"{digest}.json"

    def get(
        self,
        *,
        provider: str,
        model: str,
        temperature: float | None,
        prompt: str,
    ) -> str | None:
        """Return a cached response if present."""

        path = self._cache_path(
            provider=provider,
            model=model,
            temperature=temperature,
            prompt=prompt,
        )
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["response"])

    def set(
        self,
        *,
        provider: str,
        model: str,
        temperature: float | None,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist one response to disk."""

        path = self._cache_path(
            provider=provider,
            model=model,
            temperature=temperature,
            prompt=prompt,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "response": response,
            "metadata": metadata or {},
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
