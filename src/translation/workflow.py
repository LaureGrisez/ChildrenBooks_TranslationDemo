"""Burr workflow for multilingual Barbapapa translation."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from burr.core import ApplicationBuilder, State, action
from burr.tracking import LocalTrackingClient

from .cache import ResponseCache
from .config import TranslationWorkflowConfig
from .glossary import CharacterGlossary, load_character_glossary
from .multimodal import collect_non_empty_body_pages, render_spread_images
from .prompts import (
    critic_prompt,
    final_prompt,
    segmented_translation_prompt,
    summary_prompt,
    translation_prompt,
)
from .reporting import ArtifactBundle, persist_run_artifacts
from .segmentation import SpreadSegment, build_spread_segments


load_dotenv()

console = Console()
client = OpenAI()

@dataclass(slots=True)
class CandidateSpec:
    name: str
    provider: str
    model: str
    temperature: float
    stance: str


@dataclass(slots=True)
class TranslationCandidate:
    language: str
    name: str
    provider: str
    model: str
    temperature: float | None
    stance: str
    text: str
    status: str
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentImageInput:
    """Per-segment image payload for multimodal translation."""

    spread_pages: tuple[int, ...]
    data_url: str


def log_event(state: State, step: str, message: str, **details: Any) -> State:
    """Append one structured event to the workflow decision log."""

    event = {
        "ts": time.strftime("%H:%M:%S"),
        "step": step,
        "message": message,
        "details": details,
    }
    return state.update(decision_log=[*state["decision_log"], event])


def live_log(message: str) -> None:
    """Write a lightweight timestamped progress log."""

    console.print(f"[dim]{time.strftime('%H:%M:%S')} {message}[/dim]")


def preview_text(text: str, limit: int = 220) -> str:
    """Compact text for table/log previews."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse critic JSON, tolerating fenced code blocks."""

    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {
        "overall_winner": "unknown",
        "ranking": [],
        "decision_reasoning": raw_text,
        "paragraph_analysis": [],
        "candidate_assessment": [],
        "revision_instructions": raw_text,
        "concise_summary": raw_text,
    }


def normalize_critic_references(
    critique: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Map critic labels like 'Candidate 3' back to actual candidate names."""

    index_to_name = {
        index + 1: candidate["name"] for index, candidate in enumerate(candidates)
    }
    known_names = {candidate["name"] for candidate in candidates}

    def resolve(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped in known_names:
            return stripped

        lowered = stripped.lower()
        match = re.fullmatch(r"candidate[\s_-]*(\d+)", lowered)
        if match:
            resolved = index_to_name.get(int(match.group(1)))
            if resolved:
                return resolved
        return value

    normalized = dict(critique)
    normalized["overall_winner"] = resolve(critique.get("overall_winner", "unknown"))
    normalized["ranking"] = [
        resolve(item) for item in critique.get("ranking", []) if isinstance(item, str)
    ]

    normalized["paragraph_analysis"] = [
        {
            **item,
            "best_candidate": resolve(item.get("best_candidate")),
        }
        for item in critique.get("paragraph_analysis", [])
        if isinstance(item, dict)
    ]

    normalized["candidate_assessment"] = [
        {
            **item,
            "candidate": resolve(item.get("candidate")),
        }
        for item in critique.get("candidate_assessment", [])
        if isinstance(item, dict)
    ]
    return normalized


def build_candidate_specs(
    config: TranslationWorkflowConfig,
) -> list[CandidateSpec]:
    """Create the configured adversarial set of translation strategies."""

    if config.max_parallel_candidates > 3:
        raise ValueError(
            "MAX_PARALLEL_CANDIDATES > 3 is deprecated and not supported yet. "
            "Use at most 3 active candidates for now."
        )

    registry = {
        "google_translation": CandidateSpec(
            name="google_translation",
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            stance="Literal baseline, useful for checking factual coverage and names.",
        ),
        "google_literal": CandidateSpec(
            name="google_translation",
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            stance="Literal baseline, useful for checking factual coverage and names.",
        ),
        "gpt4o": CandidateSpec(
            name="gpt4o",
            provider="openai",
            model=config.openai_base_model,
            temperature=0.3,
            stance="Faithful, gentle, and clear, with smooth read-aloud rhythm.",
        ),
        "gpt4o_grounded": CandidateSpec(
            name="gpt4o",
            provider="openai",
            model=config.openai_base_model,
            temperature=0.3,
            stance="Faithful, gentle, and clear, with smooth read-aloud rhythm.",
        ),
        "gpt5_5": CandidateSpec(
            name="gpt5_5",
            provider="openai",
            model=config.openai_adversarial_model,
            temperature=0.8,
            stance="Slightly more playful and lively while preserving every scene.",
        ),
        "gpt55_playful": CandidateSpec(
            name="gpt5_5",
            provider="openai",
            model=config.openai_adversarial_model,
            temperature=0.8,
            stance="Slightly more playful and lively while preserving every scene.",
        ),
        "claude_sonnet_4_6": CandidateSpec(
            name="claude_sonnet_4_6",
            provider="anthropic",
            model=config.anthropic_sonnet_model,
            temperature=0.4,
            stance="Balanced literary clarity and child-friendly fluency.",
        ),
        "gemini_3": CandidateSpec(
            name="gemini_3",
            provider="gemini",
            model=config.gemini_model,
            temperature=0.4,
            stance="Visually grounded and natural child-friendly narration.",
        ),
    }

    default_order = [
        "google_translation",
        "gpt4o",
        "gpt5_5",
        "claude_sonnet_4_6",
        "gemini_3",
    ]
    requested_names = config.candidate_names or default_order

    specs: list[CandidateSpec] = []
    unknown_names = []
    for name in requested_names:
        key = name.strip()
        spec = registry.get(key)
        if spec is None:
            unknown_names.append(key)
            continue
        specs.append(spec)

    if unknown_names:
        raise ValueError(
            "Unsupported candidate names: "
            f"{', '.join(unknown_names)}. "
            f"Supported names: {', '.join(sorted(registry))}"
        )

    return specs[: config.max_parallel_candidates]


def supports_custom_temperature(model: str) -> bool:
    """Return whether the model accepts a temperature parameter."""

    normalized = model.lower()
    return not (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def ask_openai(model: str, temperature: float, prompt: str) -> str:
    """Send one user prompt to OpenAI."""

    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if supports_custom_temperature(model):
        request["temperature"] = temperature

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def ask_openai_multimodal(
    model: str,
    temperature: float,
    prompt: str,
    image_data_url: str,
) -> str:
    """Send one text+image prompt to OpenAI."""

    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "low"},
                    },
                ],
            }
        ],
    }
    if supports_custom_temperature(model):
        request["temperature"] = temperature

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def ask_openai_with_recovery(
    model: str,
    temperature: float,
    prompt: str,
    *,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> str:
    """Call OpenAI with retries and a persistent disk cache."""

    if config.enable_cache:
        cached = cache.get(
            provider="openai",
            model=model,
            temperature=temperature,
            prompt=prompt,
        )
        if cached is not None:
            live_log(f"Cache hit for {label} ({model}).")
            return cached

    last_error: Exception | None = None
    for attempt in range(1, config.openai_retry_attempts + 1):
        try:
            response = ask_openai(model, temperature, prompt)
            if config.enable_cache:
                cache.set(
                    provider="openai",
                    model=model,
                    temperature=temperature,
                    prompt=prompt,
                    response=response,
                    metadata={"label": label},
                )
            return response
        except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
            last_error = exc
            if attempt >= config.openai_retry_attempts:
                break
            delay = config.openai_retry_base_delay_seconds * attempt
            live_log(
                f"{label} transient OpenAI error on attempt {attempt}/"
                f"{config.openai_retry_attempts}: {type(exc).__name__}. "
                f"Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"OpenAI request failed unexpectedly for {label}.")


def ask_openai_multimodal_with_recovery(
    model: str,
    temperature: float,
    prompt: str,
    *,
    image_data_url: str,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> str:
    """Call OpenAI multimodal with retries and a persistent disk cache."""

    cache_prompt = json.dumps(
        {"prompt": prompt, "image_data_url": image_data_url},
        ensure_ascii=False,
        sort_keys=True,
    )
    if config.enable_cache:
        cached = cache.get(
            provider="openai_multimodal",
            model=model,
            temperature=temperature,
            prompt=cache_prompt,
        )
        if cached is not None:
            live_log(f"Cache hit for {label} ({model}).")
            return cached

    last_error: Exception | None = None
    for attempt in range(1, config.openai_retry_attempts + 1):
        try:
            response = ask_openai_multimodal(model, temperature, prompt, image_data_url)
            if config.enable_cache:
                cache.set(
                    provider="openai_multimodal",
                    model=model,
                    temperature=temperature,
                    prompt=cache_prompt,
                    response=response,
                    metadata={"label": label},
                )
            return response
        except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
            last_error = exc
            if attempt >= config.openai_retry_attempts:
                break
            delay = config.openai_retry_base_delay_seconds * attempt
            live_log(
                f"{label} transient OpenAI error on attempt {attempt}/"
                f"{config.openai_retry_attempts}: {type(exc).__name__}. "
                f"Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"OpenAI multimodal request failed unexpectedly for {label}.")


def external_language_code(
    language: str, config: TranslationWorkflowConfig
) -> str:
    """Map a display language to an API code."""

    return config.language_code(language)


def save_final_translations(
    state: State, config: TranslationWorkflowConfig
) -> ArtifactBundle:
    """Write final translations, candidates, and reports outside Burr traces."""

    return persist_run_artifacts(state, config)


def persist_partial_artifacts(state: State, config: TranslationWorkflowConfig) -> None:
    """Persist whatever artifacts are available after a completed workflow step."""

    persist_run_artifacts(state, config)


def ask_external_translator(
    text: str, language: str, config: TranslationWorkflowConfig
) -> str:
    """Use DeepL or Google Translate as an external baseline."""

    language_code = external_language_code(language, config)

    if config.external_translator == "deepl":
        import deepl

        import os

        translator = deepl.Translator(os.environ["DEEPL_AUTH_KEY"])
        result = translator.translate_text(text, target_lang=language_code.upper())
        return result.text

    if config.external_translator == "google":
        from google.cloud import translate_v2 as translate

        translator = translate.Client()
        result = translator.translate(text, target_language=language_code)
        return result["translatedText"]

    raise ValueError("EXTERNAL_TRANSLATOR must be either 'google' or 'deepl'.")


def ask_external_translator_with_cache(
    text: str,
    language: str,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> str:
    """Call the external MT provider with cache-backed recovery."""

    prompt = json.dumps(
        {
            "text": text,
            "language": language,
            "provider": config.external_translator,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if config.enable_cache:
        cached = cache.get(
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            prompt=prompt,
        )
        if cached is not None:
            live_log(f"Cache hit for {label} ({config.external_translator}).")
            return cached

    response = ask_external_translator(text, language, config)
    if config.enable_cache:
        cache.set(
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            prompt=prompt,
            response=response,
            metadata={"label": label, "language": language},
        )
    return response


def run_candidate(
    spec: CandidateSpec,
    text: str,
    source_language: str,
    language: str,
    glossary: CharacterGlossary,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    segments: list[SpreadSegment] | None = None,
    segment_images: dict[int, SegmentImageInput] | None = None,
) -> TranslationCandidate:
    """Run one candidate translation and capture outcome metadata."""

    started = time.monotonic()
    live_log(f"Starting candidate {language} / {spec.name}.")
    try:
        if config.is_multimodal_mode() and segments:
            translated_parts = []
            for segment in segments:
                history_window = config.target_history_window
                previous_translated_segments = (
                    translated_parts[-history_window:] if history_window > 0 else []
                )
                prompt = segmented_translation_prompt(
                    segment=segment,
                    source_language=source_language,
                    target_language=language,
                    stance=spec.stance,
                    glossary=glossary,
                    total_segments=len(segments),
                    previous_translated_segments=previous_translated_segments,
                    spread_pages=(
                        segment_images[segment.index].spread_pages
                        if segment_images and segment.index in segment_images
                        else None
                    ),
                )
                label = (
                    f"{language} candidate {spec.name} segment {segment.index + 1}/"
                    f"{len(segments)}"
                )
                if spec.provider == "openai" and segment_images and segment.index in segment_images:
                    translated_part = ask_openai_multimodal_with_recovery(
                        spec.model,
                        spec.temperature,
                        prompt,
                        image_data_url=segment_images[segment.index].data_url,
                        config=config,
                        cache=cache,
                        label=label,
                    )
                elif spec.provider == "openai":
                    translated_part = ask_openai_with_recovery(
                        spec.model,
                        spec.temperature,
                        prompt,
                        config=config,
                        cache=cache,
                        label=label,
                    )
                elif spec.provider in {"anthropic", "gemini"}:
                    raise NotImplementedError(
                        f"Candidate provider '{spec.provider}' is reserved in the "
                        "candidate registry but is not implemented yet."
                    )
                else:
                    translated_part = ask_external_translator_with_cache(
                        segment.source_text,
                        language,
                        config,
                        cache,
                        label,
                    )
                translated_parts.append(translated_part.strip())
            translated = "\n\n".join(translated_parts)
        elif spec.provider == "openai":
            prompt = translation_prompt(
                text=text,
                source_language=source_language,
                target_language=language,
                stance=spec.stance,
                glossary=glossary,
            )
            translated = ask_openai_with_recovery(
                spec.model,
                spec.temperature,
                prompt,
                config=config,
                cache=cache,
                label=f"{language} candidate {spec.name}",
            )
        elif spec.provider in {"anthropic", "gemini"}:
            raise NotImplementedError(
                f"Candidate provider '{spec.provider}' is reserved in the candidate "
                "registry but is not implemented yet."
            )
        else:
            translated = ask_external_translator_with_cache(
                text,
                language,
                config,
                cache,
                f"{language} candidate {spec.name}",
            )
        status = "ok"
        error = None
    except Exception as exc:
        translated = ""
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        live_log(f"Candidate {language} / {spec.name} failed: {error}")

    candidate = TranslationCandidate(
        language=language,
        name=spec.name,
        provider=spec.provider,
        model=spec.model,
        temperature=spec.temperature,
        stance=spec.stance,
        text=translated,
        status=status,
        latency_seconds=round(time.monotonic() - started, 2),
        error=error,
    )
    live_log(
        f"Candidate {language} / {spec.name} finished with {candidate.status} "
        f"in {candidate.latency_seconds}s."
    )
    return candidate


def _resolve_target_languages(
    config: TranslationWorkflowConfig, glossary: CharacterGlossary
) -> list[str]:
    """Resolve target languages from config or the CSV headers."""

    languages = config.target_languages or glossary.default_target_languages()
    normalized = [glossary.normalize_language(language) for language in languages]
    seen = set()
    unique_languages = []
    for language in normalized:
        if language not in seen:
            seen.add(language)
            unique_languages.append(language)
    return unique_languages


def build_application(config: TranslationWorkflowConfig | None = None):
    """Create the Burr app using src-owned configuration and prompts."""

    runtime = config or TranslationWorkflowConfig.from_env()
    glossary = load_character_glossary(
        runtime.character_names_csv, source_language=runtime.source_language
    )
    target_languages = _resolve_target_languages(runtime, glossary)
    source_text = runtime.load_source_text()
    candidate_specs = build_candidate_specs(runtime)
    page_numbers = []
    segments: list[SpreadSegment] | None = None
    segment_images: dict[int, SegmentImageInput] | None = None
    if runtime.is_multimodal_mode():
        page_numbers = collect_non_empty_body_pages(
            runtime.source_pdf_path,
            runtime.pdf_skip_first,
            runtime.pdf_skip_last,
        )
        segments = build_spread_segments(
            source_text=source_text,
            page_numbers=page_numbers,
            context_window=runtime.source_context_window,
        )
        rendered_spreads = render_spread_images(runtime, segments)
        segment_images = {
            spread.segment_index: SegmentImageInput(
                spread_pages=spread.spread_pages,
                data_url=spread.as_data_url(),
            )
            for spread in rendered_spreads
        }
    cache = ResponseCache(runtime.translation_cache_dir)
    tracker = LocalTrackingClient(
        project=runtime.project_name,
        storage_dir=runtime.burr_storage_dir,
    )

    @action(
        reads=["text", "target_languages", "decision_log"],
        writes=["candidate_translations", "decision_log"],
    )
    def generate_candidates(state: State) -> State:
        all_candidates = {}
        updated = log_event(
            state,
            "generate_candidates",
            "Starting adversarial candidate generation.",
            languages=state["target_languages"],
            max_parallel_candidates=runtime.max_parallel_candidates,
            external_translator=runtime.external_translator,
            workflow_mode=runtime.workflow_mode,
        )

        for language in state["target_languages"]:
            live_log(f"Generating candidates for {language}.")
            candidates: list[TranslationCandidate] = []

            with ThreadPoolExecutor(max_workers=len(candidate_specs)) as executor:
                futures = [
                    executor.submit(
                        run_candidate,
                        spec,
                        state["text"],
                        runtime.source_language,
                        language,
                        glossary,
                        runtime,
                        cache,
                        segments,
                        segment_images,
                    )
                    for spec in candidate_specs
                ]
                for future in as_completed(futures):
                    candidates.append(future.result())

            candidates.sort(key=lambda candidate: candidate.name)
            all_candidates[language] = [asdict(candidate) for candidate in candidates]
            updated = log_event(
                updated,
                "generate_candidates",
                f"Generated {len(candidates)} candidates for {language}.",
                statuses={
                    candidate["name"]: candidate["status"]
                    for candidate in all_candidates[language]
                },
                previews={
                    candidate["name"]: preview_text(candidate["text"])
                    for candidate in all_candidates[language]
                },
            )

        next_state = updated.update(candidate_translations=all_candidates)
        persist_partial_artifacts(next_state, runtime)
        return next_state

    @action(
        reads=["text", "candidate_translations", "decision_log"],
        writes=["critic_reviews", "critic_reasoning", "critic_winners", "decision_log"],
    )
    def critique_candidates(state: State) -> State:
        reviews = {}
        reasoning = {}
        winners = {}
        updated = log_event(
            state,
            "critique_candidates",
            "Starting paragraph-level critic comparison.",
        )

        for language, candidates in state["candidate_translations"].items():
            live_log(f"Running critic for {language}.")
            prompt = critic_prompt(
                text=state["text"],
                source_language=runtime.source_language,
                target_language=language,
                candidates=candidates,
                glossary=glossary,
            )
            critique_raw = ask_openai_with_recovery(
                runtime.openai_critic_model,
                0.2,
                prompt,
                config=runtime,
                cache=cache,
                label=f"{language} critic",
            )
            critique = parse_json_response(critique_raw)
            critique = normalize_critic_references(critique, candidates)
            reviews[language] = json.dumps(critique, ensure_ascii=False, indent=2)
            reasoning[language] = str(critique.get("decision_reasoning", ""))
            winners[language] = str(critique.get("overall_winner", "unknown"))
            updated = log_event(
                updated,
                "critique_candidates",
                f"Critic completed review for {language}.",
                model=runtime.openai_critic_model,
                winner=winners[language],
                ranking=critique.get("ranking", []),
                reasoning_preview=preview_text(reasoning[language]),
            )
            live_log(
                f"Critic for {language} chose {winners[language]}: "
                f"{preview_text(reasoning[language], 120)}"
            )

        next_state = updated.update(
            critic_reviews=reviews,
            critic_reasoning=reasoning,
            critic_winners=winners,
        )
        persist_partial_artifacts(next_state, runtime)
        return next_state

    @action(
        reads=["critic_reviews", "decision_log"],
        writes=["critic_summaries", "decision_log"],
    )
    def summarize_critic(state: State) -> State:
        summaries = {}
        updated = log_event(
            state,
            "summarize_critic",
            "Condensing critic reviews into final-generation guidance.",
        )

        for language, critique in state["critic_reviews"].items():
            live_log(f"Summarizing critic guidance for {language}.")
            summaries[language] = ask_openai_with_recovery(
                runtime.openai_critic_model,
                0.1,
                summary_prompt(language, critique),
                config=runtime,
                cache=cache,
                label=f"{language} critic summary",
            )
            updated = log_event(
                updated,
                "summarize_critic",
                f"Critic summary completed for {language}.",
            )

        next_state = updated.update(critic_summaries=summaries)
        persist_partial_artifacts(next_state, runtime)
        return next_state

    @action(
        reads=["text", "candidate_translations", "critic_summaries", "decision_log"],
        writes=["final_translations", "decision_log"],
    )
    def generate_final_text(state: State) -> State:
        finals = {}
        updated = log_event(
            state,
            "generate_final_text",
            "Starting final translation generation.",
            model=runtime.openai_final_model,
        )

        for language, candidates in state["candidate_translations"].items():
            live_log(f"Generating final translation for {language}.")
            prompt = final_prompt(
                text=state["text"],
                source_language=runtime.source_language,
                target_language=language,
                candidates=candidates,
                critique_summary=state["critic_summaries"][language],
                glossary=glossary,
            )
            finals[language] = ask_openai_with_recovery(
                runtime.openai_final_model,
                0.25,
                prompt,
                config=runtime,
                cache=cache,
                label=f"{language} final translation",
            )
            updated = log_event(
                updated,
                "generate_final_text",
                f"Final translation completed for {language}.",
                winner_used_as_reference=state.get("critic_winners", {}).get(
                    language, "unknown"
                ),
                final_preview=preview_text(finals[language]),
            )
            live_log(
                f"Final translation for {language}: "
                f"{preview_text(finals[language], 120)}"
            )

        next_state = updated.update(final_translations=finals)
        persist_partial_artifacts(next_state, runtime)
        return next_state

    app = (
        ApplicationBuilder()
        .with_state(
            text=source_text,
            source_language=runtime.source_language,
            target_languages=target_languages,
            supported_languages=glossary.supported_languages,
            workflow_mode=runtime.workflow_mode,
            character_name_guidance={
                language: glossary.format_name_guidance(language)
                for language in target_languages
            },
            decision_log=[],
            critic_reasoning={},
            critic_winners={},
        )
        .with_actions(
            generate_candidates,
            critique_candidates,
            summarize_critic,
            generate_final_text,
        )
        .with_transitions(
            ("generate_candidates", "critique_candidates"),
            ("critique_candidates", "summarize_critic"),
            ("summarize_critic", "generate_final_text"),
        )
        .with_entrypoint("generate_candidates")
        .with_tracker(tracker)
        .build()
    )
    return app, runtime


def print_decision_log(events: list[dict[str, Any]]) -> None:
    """Render the decision log with Rich."""

    table = Table(title="Burr Decision Log", show_lines=True)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Step", style="magenta")
    table.add_column("Decision")
    table.add_column("Details")

    for event in events:
        table.add_row(
            event["ts"],
            event["step"],
            event["message"],
            str(event.get("details", {})),
        )

    console.print(table)


def print_results(
    state: State,
    config: TranslationWorkflowConfig,
    saved_paths: ArtifactBundle | None = None,
) -> None:
    """Render source text, candidate outputs, and final translations."""

    console.print(
        Panel(
            state["text"].strip(),
            title=f"Source {config.source_language}",
            border_style="blue",
        )
    )

    for language in state["target_languages"]:
        console.print(f"\n[bold]{language} candidates[/bold]")
        candidate_table = Table(show_lines=True)
        candidate_table.add_column("Candidate")
        candidate_table.add_column("Provider")
        candidate_table.add_column("Status")
        candidate_table.add_column("Latency")
        candidate_table.add_column("Error")
        for candidate in state["candidate_translations"][language]:
            candidate_table.add_row(
                candidate["name"],
                f"{candidate['provider']} / {candidate['model']}",
                candidate["status"],
                f"{candidate['latency_seconds']}s",
                candidate["error"] or "",
            )
        console.print(candidate_table)
        for candidate in state["candidate_translations"][language]:
            console.print(
                Panel(
                    candidate["text"] or (candidate["error"] or "No output"),
                    title=f"{language} Candidate: {candidate['name']}",
                    border_style="cyan",
                )
            )

        console.print(
            Panel(
                state["critic_reviews"][language],
                title=(
                    f"{language} Critic Review "
                    f"(winner: {state.get('critic_winners', {}).get(language, 'unknown')})"
                ),
                border_style="magenta",
            )
        )
        console.print(
            Panel(
                state.get("critic_reasoning", {}).get(language, ""),
                title=f"{language} Critic Reasoning",
                border_style="red",
            )
        )
        console.print(
            Panel(
                state["critic_summaries"][language],
                title=f"{language} Critic Summary",
                border_style="yellow",
            )
        )
        console.print(
            Panel(
                state["final_translations"][language],
                title=f"{language} Final Translation",
                border_style="green",
            )
        )

    print_decision_log(state["decision_log"])
    if saved_paths:
        for language, output_path in saved_paths.latest_final_paths.items():
            console.print(
                f"[dim]{language} latest translation saved to {output_path}.[/dim]"
            )
        for language, output_path in saved_paths.versioned_final_paths.items():
            console.print(
                f"[dim]{language} run artifact saved to {output_path}.[/dim]"
            )
        for language, report_path in saved_paths.report_paths.items():
            console.print(
                f"[dim]{language} report saved to {report_path}.[/dim]"
            )
    console.print(
        f"\n[dim]Burr tracker wrote local traces under "
        f"{config.burr_storage_dir}/{config.project_name}. "
        "Run `burr` from the project root to inspect the UI.[/dim]"
    )
