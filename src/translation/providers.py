"""Shared model-provider requests for panel evaluation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

from .cache import ResponseCache
from .config import TranslationWorkflowConfig


_openai: OpenAI | None = None


def _openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {408, 429} or exc.code >= 500:
            raise urllib.error.URLError(
                f"Provider HTTP {exc.code}: {detail}"
            ) from exc
        raise ValueError(f"Provider HTTP {exc.code}: {detail}") from exc


def _image_content(image_data_url: str) -> tuple[str, str]:
    """Return an image media type and base64 payload from a data URL."""

    header, separator, data = image_data_url.partition(",")
    if not separator or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("Expected a base64 image data URL.")
    media_type = header[5:].split(";", 1)[0]
    return media_type, data


def _ask_provider(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    image_data_url: str | None = None,
) -> str:
    if provider == "openai":
        content: str | list[dict[str, Any]] = prompt
        if image_data_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}},
            ]
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if not model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
            request["temperature"] = temperature
        response = _openai_client().chat.completions.create(**request)
        return response.choices[0].message.content or ""

    if provider == "anthropic":
        api_key = os.environ["ANTHROPIC_API_KEY"]
        content: str | list[dict[str, Any]] = prompt
        if image_data_url:
            media_type, data = _image_content(image_data_url)
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        payload = {
            "model": model,
            "max_tokens": 8192,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        return "".join(
            item.get("text", "") for item in response.get("content", []) if item.get("type") == "text"
        )

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        encoded_model = urllib.parse.quote(model, safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:"
            f"generateContent?key={urllib.parse.quote(api_key, safe='')}"
        )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image_data_url:
            media_type, data = _image_content(image_data_url)
            parts.insert(
                0,
                {
                    "inline_data": {
                        "mime_type": media_type,
                        "data": data,
                    }
                },
            )
        response = _post_json(
            url,
            {
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": temperature},
            },
            {},
        )
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")
        return "".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
        )

    raise ValueError(f"Unsupported model provider: {provider}")


def ask_model_with_recovery(
    *,
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    image_data_url: str | None = None,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> str:
    """Call one provider with the workflow's shared cache and retry settings."""

    cache_prompt = prompt
    if image_data_url:
        cache_prompt = json.dumps(
            {"prompt": prompt, "image_data_url": image_data_url},
            ensure_ascii=False,
            sort_keys=True,
        )
    if config.enable_cache:
        cached = cache.get(
            provider=provider,
            model=model,
            temperature=temperature,
            prompt=cache_prompt,
        )
        if cached is not None:
            return cached

    retryable = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        urllib.error.URLError,
        TimeoutError,
    )
    last_error: Exception | None = None
    for attempt in range(1, config.openai_retry_attempts + 1):
        try:
            response = _ask_provider(
                provider,
                model,
                temperature,
                prompt,
                image_data_url=image_data_url,
            )
            if config.enable_cache:
                cache.set(
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    prompt=cache_prompt,
                    response=response,
                    metadata={"label": label},
                )
            return response
        except retryable as exc:
            last_error = exc
            if attempt >= config.openai_retry_attempts:
                break
            time.sleep(config.openai_retry_base_delay_seconds * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Model request failed unexpectedly for {label}.")
