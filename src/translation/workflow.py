"""Burr workflow for multilingual Barbapapa translation."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from burr.core import ApplicationBuilder, State, action
from burr.tracking import LocalTrackingClient

from .cache import ResponseCache
from .alignment import align_candidate_paragraphs as align_paragraphs
from .config import TranslationWorkflowConfig
from .glossary import CharacterGlossary, load_character_glossary
from .multimodal import collect_body_pages, collect_non_empty_body_pages, render_spread_images
from .panel_aggregation import aggregate_judgments
from .panel_blinding import blind_options, restore_judge_result
from .panel_models import parse_judge_specs, validate_judge_result
from .panel_prompts import audit_prompt, judge_prompt, repair_prompt, synthesis_prompt
from .panel_reporting import persist_panel_artifact
from .prompts import (
    aligned_final_paragraph_prompt,
    critic_prompt,
    final_prompt,
    segmented_translation_prompt,
    summary_prompt,
    translation_prompt,
)
from .providers import ask_model_with_recovery
from .reporting import ArtifactBundle, persist_run_artifacts
from .segmentation import SpreadSegment, build_spread_segments, split_source_paragraphs


load_dotenv()

console = Console()
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


def normalize_single_paragraph(text: str) -> str:
    """Collapse accidental model paragraph breaks inside one page translation."""

    return " ".join(part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip())


def normalize_segment_translation(text: str, expected_page_count: int) -> str:
    """Normalize a spread response while preserving real page boundaries."""

    if expected_page_count < 1:
        raise ValueError("A multimodal segment must contain at least one text page.")
    if expected_page_count == 1:
        return normalize_single_paragraph(text)

    page_blocks = split_source_paragraphs(text)
    if len(page_blocks) != expected_page_count:
        raise ValueError(
            "Multimodal segment returned "
            f"{len(page_blocks)} page blocks; expected {expected_page_count}. "
            "The model may have added or removed a blank-line page separator."
        )
    return "\n\n".join(normalize_single_paragraph(block) for block in page_blocks)


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


def parse_strict_json_response(raw_text: str) -> dict[str, Any]:
    """Parse a structured model response or raise a useful error."""

    candidate = raw_text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Model response did not contain a JSON object.")
    payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


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


def anonymize_candidate_references(
    value: Any, candidates: list[dict[str, Any]]
) -> Any:
    """Replace internal candidate names with stable neutral labels for prompts."""

    replacements = {
        candidate["name"]: f"Candidate {index + 1}"
        for index, candidate in enumerate(candidates)
    }

    if isinstance(value, str):
        anonymized = value
        for candidate_name, neutral_label in replacements.items():
            anonymized = anonymized.replace(candidate_name, neutral_label)
        return anonymized
    if isinstance(value, list):
        return [anonymize_candidate_references(item, candidates) for item in value]
    if isinstance(value, dict):
        return {
            replacements.get(key, key): anonymize_candidate_references(item, candidates)
            for key, item in value.items()
        }
    return value


def build_candidate_specs(
    config: TranslationWorkflowConfig,
) -> list[CandidateSpec]:
    """Create the configured adversarial set of translation strategies."""

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

    selected = specs[: config.max_parallel_candidates]
    duplicate_names = [
        name
        for name in {spec.name for spec in selected}
        if sum(spec.name == name for spec in selected) > 1
    ]
    if duplicate_names:
        raise ValueError(
            "Candidate selections must resolve to unique candidate names; duplicates: "
            + ", ".join(sorted(duplicate_names))
        )
    if config.candidate_temperature is not None:
        selected = [
            CandidateSpec(
                name=spec.name,
                provider=spec.provider,
                model=spec.model,
                temperature=config.candidate_temperature,
                stance=spec.stance,
            )
            for spec in selected
        ]
    return selected


def candidate_judge_refs(candidate_specs: list[CandidateSpec]) -> list[str]:
    """Return unique LLM judge references derived from selected candidates."""

    refs = []
    for spec in candidate_specs:
        ref = f"{spec.provider}:{spec.model}"
        if spec.name != "google_translation" and ref not in refs:
            refs.append(ref)
    return refs


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


def ask_external_translator_batch(
    texts: list[str], language: str, config: TranslationWorkflowConfig
) -> list[str]:
    """Translate structured strings in one request while preserving boundaries."""

    language_code = external_language_code(language, config)

    if config.external_translator == "deepl":
        import deepl

        import os

        translator = deepl.Translator(os.environ["DEEPL_AUTH_KEY"])
        results = translator.translate_text(texts, target_lang=language_code.upper())
        return [result.text for result in results]

    if config.external_translator == "google":
        from google.cloud import translate_v2 as translate

        translator = translate.Client()
        results = translator.translate(texts, target_language=language_code)
        return [result["translatedText"] for result in results]

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


def ask_external_translator_batch_with_cache(
    texts: list[str],
    language: str,
    config: TranslationWorkflowConfig,
    cache: ResponseCache,
    label: str,
) -> list[str]:
    """Call external MT once for structured inputs with cache-backed recovery."""

    prompt = json.dumps(
        {
            "texts": texts,
            "language": language,
            "provider": config.external_translator,
            "structured_batch": True,
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
            responses = json.loads(cached)
            if isinstance(responses, list) and len(responses) == len(texts):
                return [str(response) for response in responses]

    responses = ask_external_translator_batch(texts, language, config)
    if len(responses) != len(texts):
        raise ValueError(
            f"{config.external_translator} batch translation returned "
            f"{len(responses)} items for {len(texts)} inputs."
        )
    if config.enable_cache:
        cache.set(
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            prompt=prompt,
            response=json.dumps(responses, ensure_ascii=False),
            metadata={"label": label, "language": language, "structured_batch": True},
        )
    return responses


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
    identity = f"{spec.name} ({spec.provider}/{spec.model})"
    live_log(f"Starting candidate {language} / {identity}.")
    try:
        if config.is_multimodal_mode() and segments:
            if spec.provider not in {"openai", "anthropic", "gemini"}:
                page_texts = [
                    page_text for segment in segments for page_text in segment.page_texts
                ]
                try:
                    translated_pages = ask_external_translator_batch_with_cache(
                        page_texts,
                        language,
                        config,
                        cache,
                        f"{language} candidate {spec.name} page-aligned batch",
                    )
                except (TypeError, ValueError) as exc:
                    live_log(
                        f"Structured batch unavailable for {identity}; "
                        f"falling back to page requests: {exc}"
                    )
                    translated_pages = [
                        ask_external_translator_with_cache(
                            page_text,
                            language,
                            config,
                            cache,
                            (
                                f"{language} candidate {spec.name} page "
                                f"{index + 1}/{len(page_texts)}"
                            ),
                        )
                        for index, page_text in enumerate(page_texts)
                    ]
                translated = "\n\n".join(
                    normalize_single_paragraph(page) for page in translated_pages
                )
            else:
                translated_pages = []
                for segment in segments:
                    for page_index, page_text in enumerate(segment.page_texts):
                        history_window = config.target_history_window
                        previous_translated_segments = (
                            translated_pages[-history_window:] if history_window > 0 else []
                        )
                        previous_source_parts = [
                            segment.previous_source_text,
                            *segment.page_texts[:page_index],
                        ]
                        page_segment = SpreadSegment(
                            index=segment.index,
                            spread_pages=segment.spread_pages,
                            page_numbers=(segment.page_numbers[page_index],),
                            page_texts=(page_text,),
                            previous_source_text="\n\n".join(
                                part for part in previous_source_parts if part.strip()
                            ),
                        )
                        prompt = segmented_translation_prompt(
                            segment=page_segment,
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
                            f"{language} candidate {spec.name} segment "
                            f"{segment.index + 1}/{len(segments)} page "
                            f"{page_index + 1}/{len(segment.page_texts)}"
                        )
                        translated_part = ask_model_with_recovery(
                            provider=spec.provider,
                            model=spec.model,
                            temperature=spec.temperature,
                            prompt=prompt,
                            image_data_url=(
                                segment_images[segment.index].data_url
                                if segment_images
                                and segment.index in segment_images
                                else None
                            ),
                            config=config,
                            cache=cache,
                            label=label,
                        )
                        translated_pages.append(
                            normalize_single_paragraph(translated_part)
                        )
                translated = "\n\n".join(translated_pages)
        elif spec.name != "google_translation":
            prompt = translation_prompt(
                text=text,
                source_language=source_language,
                target_language=language,
                stance=spec.stance,
                glossary=glossary,
            )
            translated = ask_model_with_recovery(
                provider=spec.provider,
                model=spec.model,
                temperature=spec.temperature,
                prompt=prompt,
                config=config,
                cache=cache,
                label=f"{language} candidate {spec.name}",
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
        live_log(f"Candidate {language} / {identity} failed: {error}")

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
        f"Candidate {language} / {identity} finished with {candidate.status} "
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
    """Create the configured Burr translation application.

    Two independent mode settings determine the graph's behavior:

    - ``workflow_mode`` controls candidate generation. ``text`` translates the
      complete source in one call per candidate; ``multimodal`` first aligns
      source paragraphs to PDF spreads and translates each spread with its image.
    - ``evaluation_mode`` controls evaluation and final synthesis. ``single``
      uses one critic, one critic summary, and one final model. ``panel`` uses
      blinded paragraph-level judges, deterministic aggregation, sequential
      synthesis, and a final audit/repair pass.

    The nested functions below are Burr actions. Each action reads selected keys
    from the shared state and returns a new state containing its declared writes.
    Only the actions belonging to the selected evaluation mode are registered in
    the final graph.
    """

    # Resolve and validate all runtime dependencies before constructing actions.
    # Values captured here (runtime, glossary, segments, cache, etc.) are shared
    # by the nested Burr actions without being stored repeatedly in Burr state.
    runtime = config or TranslationWorkflowConfig.from_env()
    runtime.validate()
    runtime.apply_environment_defaults()
    critic_provider, critic_model = runtime.critic_model_spec()
    aggregation_provider, aggregation_model = runtime.aggregation_model_spec()
    summarizer_provider, summarizer_model = runtime.critic_summarizer_model_spec()
    glossary = load_character_glossary(
        runtime.character_names_csv, source_language=runtime.source_language
    )
    target_languages = _resolve_target_languages(runtime, glossary)
    source_text = runtime.load_source_text()
    candidate_specs = build_candidate_specs(runtime)
    if runtime.is_panel_mode() and not runtime.panel_judges:
        runtime.panel_judges = candidate_judge_refs(candidate_specs)
        runtime.validate()
    judge_specs = parse_judge_specs(runtime.panel_judge_specs())
    body_page_numbers = []
    text_page_numbers = []
    segments: list[SpreadSegment] | None = None
    segment_images: dict[int, SegmentImageInput] | None = None
    if runtime.is_multimodal_mode():
        # Multimodal candidate generation needs a stable mapping from cleaned
        # source paragraphs to text-bearing PDF pages and their spread images.
        # Text mode leaves segments and images as None.
        body_page_numbers = collect_body_pages(
            runtime.source_pdf_path,
            runtime.pdf_skip_first,
            runtime.pdf_skip_last,
        )
        text_page_numbers = collect_non_empty_body_pages(
            runtime.source_pdf_path,
            runtime.pdf_skip_first,
            runtime.pdf_skip_last,
        )
        segments = build_spread_segments(
            source_text=source_text,
            body_page_numbers=body_page_numbers,
            text_page_numbers=text_page_numbers,
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

    # Shared first action for every mode combination.
    @action(
        reads=["text", "target_languages", "decision_log"],
        writes=["candidate_translations", "decision_log"],
    )
    def generate_candidates(state: State) -> State:
        """Generate all configured translation candidates for every language.

        Runs in every workflow. Candidate jobs run concurrently per language.
        In text mode, ``run_candidate`` translates the complete source once.
        In multimodal mode, it translates the prebuilt spread segments in order,
        using spread images and previous translated-segment context.

        Candidate failures are captured inside their candidate records instead
        of stopping the workflow. Partial candidate artifacts are persisted so a
        later critic/final-stage failure does not discard successful generation.
        """

        all_candidates = {}
        updated = log_event(
            state,
            "generate_candidates",
            "Starting adversarial candidate generation.",
            languages=state["target_languages"],
            max_parallel_candidates=runtime.max_parallel_candidates,
            external_translator=runtime.external_translator,
            workflow_mode=runtime.workflow_mode,
            candidate_identities={
                spec.name: {
                    "provider": spec.provider,
                    "model": spec.model,
                    "temperature": spec.temperature,
                    "stance": spec.stance,
                }
                for spec in candidate_specs
            },
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

    # The following six actions form the panel-evaluation branch. They are only
    # registered when EVALUATION_MODE=panel.
    @action(
        reads=["text", "candidate_translations", "decision_log"],
        writes=["aligned_candidates", "decision_log"],
    )
    def align_candidate_paragraphs(state: State) -> State:
        """Build exact paragraph-level comparison units for panel evaluation.

        Panel mode only. Each successful candidate must contain the same number
        of paragraphs as the source. A mismatch stops panel evaluation because
        shifted paragraphs would make every later score and judgment unreliable.
        Nearby source paragraphs are attached according to the configured panel
        context window.
        """

        aligned = {}
        updated = log_event(
            state,
            "align_candidate_paragraphs",
            "Validating exact paragraph alignment for panel evaluation.",
        )
        for language, candidates in state["candidate_translations"].items():
            aligned[language] = align_paragraphs(
                state["text"],
                candidates,
                source_context_window=runtime.panel_source_context_window,
            )
            persist_panel_artifact(runtime, language, "alignment", aligned[language])
            updated = log_event(
                updated,
                "align_candidate_paragraphs",
                f"Aligned {len(aligned[language])} paragraphs for {language}.",
            )
        return updated.update(aligned_candidates=aligned)

    @action(
        reads=["aligned_candidates", "decision_log"],
        writes=[
            "panel_judge_requests",
            "panel_judge_results",
            "panel_private_mappings",
            "decision_log",
        ],
    )
    def run_panel_judges(state: State) -> State:
        """Run independent blinded judges for every aligned paragraph.

        Panel mode only. For each paragraph and judge, candidate names are hidden
        behind independently shuffled option IDs. Judges run concurrently and
        return validated structured JSON. Invalid structured output receives one
        schema-correction attempt; persistent failures are recorded explicitly.

        Raw prompts, responses, and private option mappings are persisted for
        traceability, while restored candidate names are used by later actions.
        """

        all_requests = {}
        all_results = {}
        all_mappings = {}
        updated = log_event(
            state,
            "run_panel_judges",
            "Starting independent blinded panel judgments.",
            judges=[asdict(spec) for spec in judge_specs],
        )

        for language, blocks in state["aligned_candidates"].items():
            language_requests = {}
            language_results = {}
            language_mappings = {}
            for block in blocks:
                paragraph_id = block["paragraph_id"]
                paragraph_results = {}
                paragraph_requests = {}
                paragraph_mappings = {}
                futures = {}
                with ThreadPoolExecutor(
                    max_workers=min(runtime.panel_max_parallel_judges, len(judge_specs))
                ) as executor:
                    for judge in judge_specs:
                        seed = runtime.panel_random_seed or runtime.run_id
                        blinded, mapping = blind_options(
                            block["options"],
                            seed=seed,
                            paragraph_id=paragraph_id,
                            judge_id=judge.judge_id,
                        )
                        paragraph_mappings[judge.judge_id] = mapping
                        prompt = judge_prompt(
                            block=block,
                            target_language=language,
                            glossary=glossary,
                            blinded_options=blinded,
                        )
                        paragraph_requests[judge.judge_id] = {
                            "provider": judge.provider,
                            "model": judge.model,
                            "prompt": prompt,
                        }
                        future = executor.submit(
                            ask_model_with_recovery,
                            provider=judge.provider,
                            model=judge.model,
                            temperature=runtime.panel_judge_temperature,
                            prompt=prompt,
                            config=runtime,
                            cache=cache,
                            label=f"{language} {paragraph_id} {judge.judge_id}",
                        )
                        futures[future] = (judge, list(blinded))

                    for future in as_completed(futures):
                        judge, option_ids = futures[future]
                        raw = ""
                        try:
                            raw = future.result()
                            parsed = validate_judge_result(
                                parse_strict_json_response(raw), option_ids
                            )
                            paragraph_results[judge.judge_id] = {
                                "status": "ok",
                                "provider": judge.provider,
                                "model": judge.model,
                                "raw_response": raw,
                                "result": restore_judge_result(
                                    parsed, paragraph_mappings[judge.judge_id]
                                ),
                            }
                        except Exception as exc:
                            if raw:
                                try:
                                    correction_prompt = (
                                        paragraph_requests[judge.judge_id]["prompt"]
                                        + "\n\nYour previous response failed validation with: "
                                        + str(exc)
                                        + "\nReturn corrected valid JSON only."
                                    )
                                    corrected_raw = ask_model_with_recovery(
                                        provider=judge.provider,
                                        model=judge.model,
                                        temperature=0.0,
                                        prompt=correction_prompt,
                                        config=runtime,
                                        cache=cache,
                                        label=(
                                            f"{language} {paragraph_id} {judge.judge_id} "
                                            "schema correction"
                                        ),
                                    )
                                    corrected = validate_judge_result(
                                        parse_strict_json_response(corrected_raw), option_ids
                                    )
                                    paragraph_results[judge.judge_id] = {
                                        "status": "ok",
                                        "provider": judge.provider,
                                        "model": judge.model,
                                        "raw_response": corrected_raw,
                                        "result": restore_judge_result(
                                            corrected,
                                            paragraph_mappings[judge.judge_id],
                                        ),
                                        "schema_corrected": True,
                                    }
                                    continue
                                except Exception as correction_exc:
                                    exc = correction_exc
                            paragraph_results[judge.judge_id] = {
                                "status": "error",
                                "provider": judge.provider,
                                "model": judge.model,
                                "raw_response": raw,
                                "error": f"{type(exc).__name__}: {exc}",
                            }

                language_results[paragraph_id] = paragraph_results
                language_requests[paragraph_id] = paragraph_requests
                language_mappings[paragraph_id] = paragraph_mappings

            all_requests[language] = language_requests
            all_results[language] = language_results
            all_mappings[language] = language_mappings
            persist_panel_artifact(runtime, language, "judge_requests", language_requests)
            persist_panel_artifact(runtime, language, "judge_results", language_results)
            persist_panel_artifact(runtime, language, "private_mappings", language_mappings)
            updated = log_event(
                updated,
                "run_panel_judges",
                f"Panel judgments completed for {language}.",
            )

        return updated.update(
            panel_judge_requests=all_requests,
            panel_judge_results=all_results,
            panel_private_mappings=all_mappings,
        )

    @action(
        reads=["panel_judge_results", "decision_log"],
        writes=["panel_aggregates", "decision_log"],
    )
    def aggregate_panel_judgments(state: State) -> State:
        """Combine valid panel judgments using deterministic scoring code.

        Panel mode only. No model call occurs here. The action combines pairwise
        results, ranking positions, normalized criterion scores, and confirmed
        critical errors. At least two valid judges are required per paragraph.
        The resulting ranking and selected synthesis options are inspectable in
        the persisted aggregate artifact.
        """

        all_aggregates = {}
        updated = log_event(
            state,
            "aggregate_panel_judgments",
            "Aggregating panel judgments deterministically.",
        )
        for language, paragraph_results in state["panel_judge_results"].items():
            language_aggregates = {}
            for paragraph_id, judge_results in paragraph_results.items():
                valid_results = [
                    result["result"]
                    for result in judge_results.values()
                    if result["status"] == "ok"
                ]
                if len(valid_results) < 2:
                    raise ValueError(
                        f"Panel evaluation requires at least two valid judgments for "
                        f"{language} {paragraph_id}; received {len(valid_results)}."
                    )
                language_aggregates[paragraph_id] = aggregate_judgments(
                    valid_results,
                    pairwise_weight=runtime.panel_pairwise_weight,
                    ranking_weight=runtime.panel_ranking_weight,
                    score_weight=runtime.panel_score_weight,
                    critical_error_confirmations=runtime.panel_critical_error_confirmations,
                )
            all_aggregates[language] = language_aggregates
            persist_panel_artifact(runtime, language, "aggregates", language_aggregates)
            updated = log_event(
                updated,
                "aggregate_panel_judgments",
                f"Panel aggregation completed for {language}.",
            )
        return updated.update(panel_aggregates=all_aggregates)

    @action(
        reads=["aligned_candidates", "panel_aggregates", "decision_log"],
        writes=["final_paragraphs", "decision_log"],
    )
    def generate_final_paragraphs(state: State) -> State:
        """Synthesize the panel-guided final translation paragraph by paragraph.

        Panel mode only. Paragraphs are generated sequentially so completed
        target-language paragraphs can provide continuity context to later ones.
        Each call receives only the panel-selected candidate options and frozen
        aggregate guidance for the current paragraph. Completed paragraphs are
        persisted incrementally for recovery and later auditing.
        """

        all_final_paragraphs = {}
        updated = log_event(
            state,
            "generate_final_paragraphs",
            "Starting sequential panel-guided paragraph synthesis.",
        )
        for language, blocks in state["aligned_candidates"].items():
            final_paragraphs = {}
            history: list[str] = []
            for block in blocks:
                paragraph_id = block["paragraph_id"]
                aggregate = state["panel_aggregates"][language][paragraph_id]
                selected = {
                    candidate: block["options"][candidate]
                    for candidate in aggregate["selected_options"]
                }
                history_window = runtime.panel_target_history_window
                previous_final = "\n\n".join(
                    history[-history_window:] if history_window > 0 else []
                )
                final_paragraph = ask_model_with_recovery(
                    provider=aggregation_provider,
                    model=aggregation_model,
                    temperature=0.25,
                    prompt=synthesis_prompt(
                        block=block,
                        target_language=language,
                        glossary=glossary,
                        selected_options=selected,
                        aggregate=aggregate,
                        previous_final=previous_final,
                    ),
                    config=runtime,
                    cache=cache,
                    label=f"{language} {paragraph_id} panel synthesis",
                ).strip()
                final_paragraphs[paragraph_id] = final_paragraph
                history.append(final_paragraph)
                persist_panel_artifact(
                    runtime, language, "final_paragraphs", final_paragraphs
                )
            all_final_paragraphs[language] = final_paragraphs
            updated = log_event(
                updated,
                "generate_final_paragraphs",
                f"Sequential synthesis completed for {language}.",
            )
        return updated.update(final_paragraphs=all_final_paragraphs)

    @action(
        reads=["text", "aligned_candidates", "final_paragraphs", "decision_log"],
        writes=["book_audits", "decision_log"],
    )
    def audit_book_consistency(state: State) -> State:
        """Audit the assembled panel translation for book-level consistency.

        Panel mode only. This model call sees the complete source and assembled
        final text, but it must return paragraph-scoped findings rather than
        rewriting the book. Findings with unknown paragraph IDs or empty repair
        instructions are discarded before the targeted repair stage.
        """

        audits = {}
        updated = log_event(
            state,
            "audit_book_consistency",
            "Starting whole-book consistency audit.",
        )
        for language, blocks in state["aligned_candidates"].items():
            final_text = "\n\n".join(
                state["final_paragraphs"][language][block["paragraph_id"]]
                for block in blocks
            )
            raw = ask_model_with_recovery(
                provider=critic_provider,
                model=critic_model,
                temperature=0.1,
                prompt=audit_prompt(
                    source_text=state["text"],
                    final_text=final_text,
                    target_language=language,
                    glossary=glossary,
                ),
                config=runtime,
                cache=cache,
                label=f"{language} book consistency audit",
            )
            audit = parse_strict_json_response(raw)
            valid_ids = {block["paragraph_id"] for block in blocks}
            findings = [
                finding
                for finding in audit.get("findings", [])
                if isinstance(finding, dict)
                and finding.get("paragraph_id") in valid_ids
                and str(finding.get("instruction", "")).strip()
            ]
            audits[language] = {"findings": findings}
            persist_panel_artifact(runtime, language, "audit", audits[language])
            updated = log_event(
                updated,
                "audit_book_consistency",
                f"Audit completed for {language} with {len(findings)} findings.",
            )
        return updated.update(book_audits=audits)

    @action(
        reads=["aligned_candidates", "final_paragraphs", "book_audits", "decision_log"],
        writes=["repair_results", "final_translations", "decision_log"],
    )
    def repair_flagged_paragraphs(state: State) -> State:
        """Repair only audit-flagged paragraphs and assemble the panel final.

        Panel mode only. Unflagged paragraphs remain byte-for-byte unchanged.
        Each repair receives neighboring final paragraphs for continuity. The
        action then joins paragraphs in source order, writes ``final_translations``,
        and persists the normal final artifacts used by the CLI and Streamlit UI.
        """

        repairs = {}
        finals = {}
        updated = log_event(
            state,
            "repair_flagged_paragraphs",
            "Repairing only audit-flagged paragraphs and assembling final text.",
        )
        for language, blocks in state["aligned_candidates"].items():
            paragraphs = dict(state["final_paragraphs"][language])
            findings_by_id: dict[str, list[dict[str, Any]]] = {}
            for finding in state["book_audits"][language]["findings"]:
                findings_by_id.setdefault(finding["paragraph_id"], []).append(finding)
            language_repairs = {}
            for index, block in enumerate(blocks):
                paragraph_id = block["paragraph_id"]
                findings = findings_by_id.get(paragraph_id, [])
                if not findings:
                    continue
                previous_id = blocks[index - 1]["paragraph_id"] if index > 0 else None
                next_id = (
                    blocks[index + 1]["paragraph_id"] if index + 1 < len(blocks) else None
                )
                repaired = ask_model_with_recovery(
                    provider=aggregation_provider,
                    model=aggregation_model,
                    temperature=0.15,
                    prompt=repair_prompt(
                        block=block,
                        current_final=paragraphs[paragraph_id],
                        previous_final=paragraphs.get(previous_id, "") if previous_id else "",
                        next_final=paragraphs.get(next_id, "") if next_id else "",
                        findings=findings,
                        target_language=language,
                        glossary=glossary,
                    ),
                    config=runtime,
                    cache=cache,
                    label=f"{language} {paragraph_id} audit repair",
                ).strip()
                language_repairs[paragraph_id] = {
                    "before": paragraphs[paragraph_id],
                    "after": repaired,
                    "findings": findings,
                }
                paragraphs[paragraph_id] = repaired
            repairs[language] = language_repairs
            finals[language] = "\n\n".join(
                paragraphs[block["paragraph_id"]] for block in blocks
            )
            persist_panel_artifact(runtime, language, "repairs", language_repairs)
            updated = log_event(
                updated,
                "repair_flagged_paragraphs",
                f"Final panel translation assembled for {language}.",
                repaired_paragraphs=len(language_repairs),
            )
        next_state = updated.update(repair_results=repairs, final_translations=finals)
        persist_partial_artifacts(next_state, runtime)
        return next_state

    # The following three actions form the single-critic evaluation branch. They
    # are registered for EVALUATION_MODE=single after either text or multimodal
    # candidate generation.
    @action(
        reads=["text", "candidate_translations", "decision_log"],
        writes=["critic_reviews", "critic_reasoning", "critic_winners", "decision_log"],
    )
    def critique_candidates(state: State) -> State:
        """Compare complete candidate translations with one critic model.

        Single evaluation mode only. The critic receives the source and all
        candidate books, then returns a structured overall ranking plus
        paragraph-level remarks. Candidate references such as ``Candidate 2``
        are normalized back to internal candidate names before being stored.

        The action is the same after text and multimodal candidate generation;
        multimodal images are not supplied to this critic.
        """

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
            critique_raw = ask_model_with_recovery(
                provider=critic_provider,
                model=critic_model,
                temperature=0.2,
                prompt=prompt,
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
                provider=critic_provider,
                model=critic_model,
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
        reads=["critic_reviews", "candidate_translations", "decision_log"],
        writes=["critic_summaries", "decision_log"],
    )
    def summarize_critic(state: State) -> State:
        """Condense the single critic review into final-synthesis guidance.

        Single evaluation mode only. This is a second critic-model call that
        reduces the detailed structured review to decisions and concrete revision
        instructions. The summary is used by ``generate_final_text``.
        """

        summaries = {}
        updated = log_event(
            state,
            "summarize_critic",
            "Condensing critic reviews into final-generation guidance.",
        )

        for language, critique in state["critic_reviews"].items():
            live_log(f"Summarizing critic guidance for {language}.")
            candidates = state["candidate_translations"][language]
            anonymized_critique = anonymize_candidate_references(
                json.loads(critique), candidates
            )
            summaries[language] = ask_model_with_recovery(
                provider=summarizer_provider,
                model=summarizer_model,
                temperature=0.1,
                prompt=summary_prompt(
                    language,
                    json.dumps(anonymized_critique, ensure_ascii=False, indent=2),
                ),
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
        """Generate the final translation for the single-critic branch.

        Single evaluation mode only, with behavior depending on workflow mode:

        - Text mode makes one whole-book final-synthesis call per language.
        - Multimodal mode makes one final call per source/page paragraph, then
          joins the results in code. This preserves the paragraph count required
          by the PDF overlay while still using the complete candidate books and
          critic summary as editorial references.

        Final translations and reports are persisted after generation.
        """

        finals = {}
        updated = log_event(
            state,
            "generate_final_text",
            "Starting final translation generation.",
            provider=aggregation_provider,
            model=aggregation_model,
        )

        for language, candidates in state["candidate_translations"].items():
            live_log(f"Generating final translation for {language}.")
            if runtime.is_multimodal_mode():
                source_paragraphs = split_source_paragraphs(state["text"])
                final_paragraphs = []
                for index, source_paragraph in enumerate(source_paragraphs):
                    previous_final = (
                        final_paragraphs[-1] if final_paragraphs else ""
                    )
                    prompt = aligned_final_paragraph_prompt(
                        source_paragraph=source_paragraph,
                        previous_source=(
                            source_paragraphs[index - 1] if index > 0 else ""
                        ),
                        next_source=(
                            source_paragraphs[index + 1]
                            if index + 1 < len(source_paragraphs)
                            else ""
                        ),
                        previous_final=previous_final,
                        source_language=runtime.source_language,
                        target_language=language,
                        candidates=candidates,
                        critique_summary=state["critic_summaries"][language],
                        glossary=glossary,
                        paragraph_number=index + 1,
                        paragraph_count=len(source_paragraphs),
                    )
                    generated = ask_model_with_recovery(
                        provider=aggregation_provider,
                        model=aggregation_model,
                        temperature=0.25,
                        prompt=prompt,
                        config=runtime,
                        cache=cache,
                        label=(
                            f"{language} final paragraph {index + 1}/"
                            f"{len(source_paragraphs)}"
                        ),
                    )
                    normalized = normalize_single_paragraph(generated)
                    if not normalized:
                        raise ValueError(
                            f"Final synthesis returned an empty paragraph for "
                            f"{language} paragraph {index + 1}."
                        )
                    final_paragraphs.append(normalized)
                finals[language] = "\n\n".join(final_paragraphs)
            else:
                prompt = final_prompt(
                    text=state["text"],
                    source_language=runtime.source_language,
                    target_language=language,
                    candidates=candidates,
                    critique_summary=state["critic_summaries"][language],
                    glossary=glossary,
                )
                finals[language] = ask_model_with_recovery(
                    provider=aggregation_provider,
                    model=aggregation_model,
                    temperature=0.25,
                    prompt=prompt,
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

    # Seed only compact, serializable values into Burr state. Large reusable
    # runtime objects such as the glossary, cache, and rendered images remain in
    # the action closures above.
    builder = (
        ApplicationBuilder()
        .with_state(
            text=source_text,
            source_language=runtime.source_language,
            target_languages=target_languages,
            supported_languages=glossary.supported_languages,
            workflow_mode=runtime.workflow_mode,
            evaluation_mode=runtime.evaluation_mode,
            character_name_guidance={
                language: glossary.format_name_guidance(language)
                for language in target_languages
            },
            decision_log=[],
            critic_reasoning={},
            critic_winners={},
        )
    )
    if runtime.is_panel_mode():
        # Panel evaluation replaces the single critic/summary/final actions.
        # Candidate generation remains shared and can itself be text or multimodal.
        builder = builder.with_actions(
            generate_candidates,
            align_candidate_paragraphs,
            run_panel_judges,
            aggregate_panel_judgments,
            generate_final_paragraphs,
            audit_book_consistency,
            repair_flagged_paragraphs,
        ).with_transitions(
            ("generate_candidates", "align_candidate_paragraphs"),
            ("align_candidate_paragraphs", "run_panel_judges"),
            ("run_panel_judges", "aggregate_panel_judgments"),
            ("aggregate_panel_judgments", "generate_final_paragraphs"),
            ("generate_final_paragraphs", "audit_book_consistency"),
            ("audit_book_consistency", "repair_flagged_paragraphs"),
        )
    else:
        # Single evaluation preserves the original critic -> summary -> final
        # sequence after either text or multimodal candidate generation.
        builder = builder.with_actions(
            generate_candidates,
            critique_candidates,
            summarize_critic,
            generate_final_text,
        ).with_transitions(
            ("generate_candidates", "critique_candidates"),
            ("critique_candidates", "summarize_critic"),
            ("summarize_critic", "generate_final_text"),
        )
    app = (
        builder.with_entrypoint("generate_candidates")
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

        if config.is_panel_mode():
            console.print(
                Panel(
                    json.dumps(
                        state.get("book_audits", {}).get(language, {}),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    title=f"{language} Panel Audit",
                    border_style="magenta",
                )
            )
        else:
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
