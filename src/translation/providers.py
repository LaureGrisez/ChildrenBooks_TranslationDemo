"""Shared LiteLLM model-provider requests for translation workflows."""

from __future__ import annotations

import json
import os
import time
from typing import Any

# Avoid an import-time network request for LiteLLM's optional pricing metadata.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from .cache import ResponseCache
from .config import TranslationWorkflowConfig


def litellm_model_name(provider: str, model: str) -> str:
    """Return the provider-qualified model name expected by LiteLLM."""

    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError("Provider and model must both be non-empty.")
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def _message_content(
    prompt: str,
    image_data_urls: list[str],
) -> str | list[dict[str, Any]]:
    """Build provider-neutral OpenAI-format content for LiteLLM."""

    if not image_data_urls:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "low"},
        }
        for image_url in image_data_urls
    )
    return content


def _supports_custom_temperature(provider: str, model: str) -> bool:
    """Return whether this model is known to accept a temperature value."""

    if provider.strip().lower() != "openai":
        return True
    normalized = model.strip().lower()
    return not normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _ask_provider(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    image_data_url: str | None = None,
    image_data_urls: list[str] | None = None,
    return_metrics: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Call any configured provider through LiteLLM's unified interface."""

    images = image_data_urls or ([image_data_url] if image_data_url else [])
    request: dict[str, Any] = {
        "model": litellm_model_name(provider, model),
        "messages": [
            {
                "role": "user",
                "content": _message_content(prompt, images),
            }
        ],
        "timeout": 180,
    }
    if _supports_custom_temperature(provider, model):
        request["temperature"] = temperature

    response = litellm.completion(**request)
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is not None and not isinstance(usage, dict):
        usage = dict(usage)
    try:
        estimated_cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        estimated_cost = None
    text = response.choices[0].message.content or ""
    metrics = {
        "usage": usage or {},
        "estimated_cost_usd": estimated_cost,
    }
    return (text, metrics) if return_metrics else text


def ask_model_with_recovery(
    *,
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    image_data_url: str | None = None,
    image_data_urls: list[str] | None = None,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> str:
    """Call one LiteLLM provider with the workflow's cache and retry settings."""

    if image_data_url and image_data_urls:
        raise ValueError("Use either image_data_url or image_data_urls, not both.")

    cache_prompt = prompt
    if image_data_url or image_data_urls:
        cache_prompt = json.dumps(
            {
                "prompt": prompt,
                "image_data_urls": image_data_urls
                or ([image_data_url] if image_data_url else []),
            },
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
            config.model_call_metrics.append({
                "label": label, "provider": provider, "model": model,
                "temperature": temperature, "cache_hit": True,
                "latency_seconds": 0.0, "usage": {}, "estimated_cost_usd": 0.0,
            })
            return cached

    retryable = (
        litellm.APIConnectionError,
        litellm.BadGatewayError,
        litellm.InternalServerError,
        litellm.RateLimitError,
        litellm.ServiceUnavailableError,
        litellm.Timeout,
        TimeoutError,
    )
    last_error: Exception | None = None
    for attempt in range(1, config.openai_retry_attempts + 1):
        try:
            started = time.monotonic()
            result = _ask_provider(
                provider,
                model,
                temperature,
                prompt,
                image_data_url=image_data_url,
                image_data_urls=image_data_urls,
                return_metrics=True,
            )
            if isinstance(result, tuple):
                response, metrics = result
            else:  # Compatibility with patched/custom adapters returning plain text.
                response, metrics = result, {"usage": {}, "estimated_cost_usd": None}
            config.model_call_metrics.append({
                "label": label, "provider": provider, "model": model,
                "temperature": temperature, "cache_hit": False,
                "latency_seconds": round(time.monotonic() - started, 3),
                **metrics,
            })
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
