"""Burr workflow for multilingual Barbapapa translation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from burr.core import ApplicationBuilder, State, action
from burr.tracking import LocalTrackingClient

from .config import TranslationWorkflowConfig
from .glossary import CharacterGlossary, load_character_glossary
from .prompts import critic_prompt, final_prompt, summary_prompt, translation_prompt
from .reporting import ArtifactBundle, persist_run_artifacts


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


def build_candidate_specs(
    config: TranslationWorkflowConfig,
) -> list[CandidateSpec]:
    """Create a small adversarial set of translation strategies."""

    specs = [
        CandidateSpec(
            name="gpt4o_grounded",
            provider="openai",
            model=config.openai_base_model,
            temperature=0.3,
            stance="Faithful, gentle, and clear, with smooth read-aloud rhythm.",
        ),
        CandidateSpec(
            name="gpt55_playful",
            provider="openai",
            model=config.openai_adversarial_model,
            temperature=0.8,
            stance="Slightly more playful and lively while preserving every scene.",
        ),
        CandidateSpec(
            name=f"{config.external_translator}_literal",
            provider=config.external_translator,
            model=config.external_translator,
            temperature=0.0,
            stance="Literal baseline, useful for checking factual coverage and names.",
        ),
        CandidateSpec(
            name="gpt4o_simple",
            provider="openai",
            model=config.openai_base_model,
            temperature=0.1,
            stance="Extra simple vocabulary for early readers and oral clarity.",
        ),
        CandidateSpec(
            name="gpt55_oral",
            provider="openai",
            model=config.openai_adversarial_model,
            temperature=0.5,
            stance="Natural spoken storytelling rhythm without drifting from the source.",
        ),
    ]
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


def run_candidate(
    spec: CandidateSpec,
    text: str,
    source_language: str,
    language: str,
    glossary: CharacterGlossary,
    config: TranslationWorkflowConfig,
) -> TranslationCandidate:
    """Run one candidate translation and capture outcome metadata."""

    started = time.monotonic()
    live_log(f"Starting candidate {language} / {spec.name}.")
    try:
        if spec.provider == "openai":
            translated = ask_openai(
                spec.model,
                spec.temperature,
                translation_prompt(
                    text=text,
                    source_language=source_language,
                    target_language=language,
                    stance=spec.stance,
                    glossary=glossary,
                ),
            )
        else:
            translated = ask_external_translator(text, language, config)
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

        return updated.update(candidate_translations=all_candidates)

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
            critique_raw = ask_openai(
                runtime.openai_critic_model,
                0.2,
                critic_prompt(
                    text=state["text"],
                    source_language=runtime.source_language,
                    target_language=language,
                    candidates=candidates,
                    glossary=glossary,
                ),
            )
            critique = parse_json_response(critique_raw)
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

        return updated.update(
            critic_reviews=reviews,
            critic_reasoning=reasoning,
            critic_winners=winners,
        )

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
            summaries[language] = ask_openai(
                runtime.openai_critic_model,
                0.1,
                summary_prompt(language, critique),
            )
            updated = log_event(
                updated,
                "summarize_critic",
                f"Critic summary completed for {language}.",
            )

        return updated.update(critic_summaries=summaries)

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
            finals[language] = ask_openai(
                runtime.openai_final_model,
                0.25,
                final_prompt(
                    text=state["text"],
                    source_language=runtime.source_language,
                    target_language=language,
                    candidates=candidates,
                    critique_summary=state["critic_summaries"][language],
                    glossary=glossary,
                ),
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

        return updated.update(final_translations=finals)

    app = (
        ApplicationBuilder()
        .with_state(
            text=source_text,
            source_language=runtime.source_language,
            target_languages=target_languages,
            supported_languages=glossary.supported_languages,
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
