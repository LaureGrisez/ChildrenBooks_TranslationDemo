# pip install -U langgraph langchain-openai langsmith python-dotenv rich deepl google-cloud-translate
#
# Advanced LangGraph experiment for children's-book translation:
# 1. For each target language, create 2-5 adversarial translation candidates.
# 2. Compare candidates paragraph-by-paragraph with a critic model.
# 3. Summarize the critic's decision.
# 4. Produce a final translation using the best evidence from all candidates.
# 5. Print graph stream updates and optionally send traces to LangSmith.
#
# To enable LangSmith tracing:
# export LANGSMITH_TRACING=true
# export LANGSMITH_API_KEY=your_langsmith_key

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
if os.getenv("LANGSMITH_PROJECT"):
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
if os.getenv("LANGSMITH_TRACING"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ["LANGSMITH_TRACING"])

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import get_current_run_tree, traceable
from langsmith.run_helpers import set_run_metadata, set_tracing_parent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOOGLE_CREDENTIALS = (
    REPO_ROOT / "credentials" / "children-book-translation-4e958984d7f8.json"
)
if DEFAULT_GOOGLE_CREDENTIALS.exists():
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", str(DEFAULT_GOOGLE_CREDENTIALS)
    )

console = Console()

TARGET_LANGUAGES = [
    language.strip()
    for language in os.getenv("TARGET_LANGUAGES", "Hindi,Tamil").split(",")
    if language.strip()
]
LANGUAGE_CODE_OVERRIDES = {
    pair.split(":", 1)[0].strip().lower(): pair.split(":", 1)[1].strip()
    for pair in os.getenv("LANGUAGE_CODES", "").split(",")
    if ":" in pair
}
DEFAULT_LANGUAGE_CODES = {
    "hindi": "hi",
    "tamil": "ta",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
}

EXTERNAL_TRANSLATOR = os.getenv("EXTERNAL_TRANSLATOR", "google").lower()
MAX_PARALLEL_CANDIDATES = min(
    5,
    max(2, int(os.getenv("MAX_PARALLEL_CANDIDATES", "3"))),
)

OPENAI_BASE_MODEL = os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
OPENAI_ADVERSARIAL_MODEL = os.getenv("OPENAI_ADVERSARIAL_MODEL", "gpt-5.5")
OPENAI_CRITIC_MODEL = os.getenv("OPENAI_CRITIC_MODEL", OPENAI_BASE_MODEL)
OPENAI_FINAL_MODEL = os.getenv("OPENAI_FINAL_MODEL", OPENAI_BASE_MODEL)
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT", "children-book-translation-advanced"
)

input_text = """
Milo the little mouse found a shiny red button under the old oak tree.
"Maybe it belongs to the moon!" he whispered.

His sister Mina tied it to a blue ribbon and held it up to the sky.
The button winked in the sunlight, but the moon did not answer.
"""


class TranslationState(TypedDict, total=False):
    text: str
    target_languages: list[str]
    decision_log: list[dict[str, Any]]
    candidate_translations: dict[str, list[dict[str, Any]]]
    critic_reviews: dict[str, str]
    critic_reasoning: dict[str, str]
    critic_winners: dict[str, str]
    critic_summaries: dict[str, str]
    final_translations: dict[str, str]


@dataclass
class CandidateSpec:
    name: str
    provider: str
    model: str
    temperature: float
    stance: str


@dataclass
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


def supports_custom_temperature(model_name: str) -> bool:
    normalized = model_name.lower()
    return not (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def model(model_name: str, temperature: float) -> ChatOpenAI:
    if supports_custom_temperature(model_name):
        return ChatOpenAI(model=model_name, temperature=temperature)
    return ChatOpenAI(model=model_name)


def add_event(
    state: TranslationState, step: str, message: str, **details: Any
) -> list[dict[str, Any]]:
    return [
        *state.get("decision_log", []),
        {
            "ts": time.strftime("%H:%M:%S"),
            "step": step,
            "message": message,
            "details": details,
        },
    ]


def live_log(message: str) -> None:
    console.print(f"[dim]{time.strftime('%H:%M:%S')} {message}[/dim]")


def preview_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def serialize_for_trace(value: Any) -> dict[str, Any]:
    if isinstance(value, TranslationCandidate):
        data = asdict(value)
        data["text_preview"] = preview_text(data.get("text", ""))
        return data
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"text", "critique", "summary", "translation"} and isinstance(
                item, str
            ):
                result[key] = item
                result[f"{key}_preview"] = preview_text(item)
            else:
                result[key] = item
        return result
    return {"value": value}


def parse_json_response(raw_text: str) -> dict[str, Any]:
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


def build_candidate_specs() -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            name="gpt4o_grounded",
            provider="openai",
            model=OPENAI_BASE_MODEL,
            temperature=0.3,
            stance="faithful, gentle, read-aloud translation",
        ),
        CandidateSpec(
            name="gpt55_playful",
            provider="openai",
            model=OPENAI_ADVERSARIAL_MODEL,
            temperature=0.8,
            stance="more playful wording while preserving the story",
        ),
        CandidateSpec(
            name=f"{EXTERNAL_TRANSLATOR}_literal",
            provider=EXTERNAL_TRANSLATOR,
            model=EXTERNAL_TRANSLATOR,
            temperature=0.0,
            stance="external MT baseline, useful as a literal counterpoint",
        ),
        CandidateSpec(
            name="gpt4o_simple",
            provider="openai",
            model=OPENAI_BASE_MODEL,
            temperature=0.1,
            stance="extra simple vocabulary for early readers",
        ),
        CandidateSpec(
            name="gpt55_oral",
            provider="openai",
            model=OPENAI_ADVERSARIAL_MODEL,
            temperature=0.5,
            stance="natural oral storytelling rhythm",
        ),
    ]
    return specs[:MAX_PARALLEL_CANDIDATES]


def translation_prompt(text: str, language: str, stance: str) -> str:
    return f"""
Translate this children's-book passage into natural {language}.

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech
- Do not add explanations outside the translation

Translator stance:
{stance}

Source text:
{text}
"""


def critic_prompt(text: str, language: str, candidates: list[dict[str, Any]]) -> str:
    candidate_blocks = "\n\n".join(
        f"Candidate {idx + 1}: {candidate['name']} "
        f"({candidate['provider']} / {candidate['model']} / T={candidate['temperature']})\n"
        f"Status: {candidate['status']}\n"
        f"{candidate['text']}"
        for idx, candidate in enumerate(candidates)
    )
    return f"""
You are a senior editor for translated picture books.

Compare the candidate translations into {language}. Evaluate each paragraph
for:
- faithfulness to the English source
- natural child-friendly phrasing
- read-aloud rhythm
- cultural fit without over-localizing
- handling of quotes, imagery, and paragraph breaks

Return a compact but specific critique with:
Return valid JSON only with this shape:
{{
  "overall_winner": "candidate_name",
  "ranking": ["candidate_a", "candidate_b", "candidate_c"],
  "decision_reasoning": "Explain clearly why the winning candidate wins and what tradeoffs drove the decision.",
  "paragraph_analysis": [
    {{
      "paragraph_number": 1,
      "best_candidate": "candidate_name",
      "notes": "Comparison notes for this paragraph."
    }}
  ],
  "candidate_assessment": [
    {{
      "candidate": "candidate_name",
      "strengths": ["..."],
      "weaknesses": ["..."]
    }}
  ],
  "revision_instructions": [
    "Concrete instruction 1",
    "Concrete instruction 2"
  ],
  "concise_summary": "A short editorial summary for the final translator."
}}

Source:
{text}

Candidates:
{candidate_blocks}
"""


def summary_prompt(language: str, critique: str) -> str:
    return f"""
Summarize this translation critique for the final {language} translator.
Focus on decisions, tradeoffs, and concrete instructions.

Critique:
{critique}
"""


def final_prompt(
    text: str,
    language: str,
    candidates: list[dict[str, Any]],
    critique_summary: str,
) -> str:
    candidate_blocks = "\n\n".join(
        f"{candidate['name']}:\n{candidate['text']}" for candidate in candidates
    )
    return f"""
Create the final {language} children's-book translation.

Use the critic summary as editorial guidance. You may borrow the best phrases
from any candidate, but the final output must read as one coherent translation.
Preserve paragraph breaks and quoted speech. Return only the final translation
as plain text. Do not use Markdown formatting.

Source:
{text}

Candidate translations:
{candidate_blocks}

Critic summary:
{critique_summary}
"""


@traceable(
    run_type="llm",
    name="openai_translation_or_editor_call",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=lambda outputs: {"text": outputs},
)
def ask_openai(model_name: str, temperature: float, prompt: str) -> str:
    set_run_metadata(model=model_name, requested_temperature=temperature)
    response = model(model_name, temperature).invoke(prompt)
    return str(response.content)


@traceable(
    run_type="tool",
    name="external_translation_call",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=lambda outputs: {"text": outputs},
)
def ask_external_translator(text: str, language: str) -> str:
    language_code = external_language_code(language)
    set_run_metadata(provider=EXTERNAL_TRANSLATOR, language=language, code=language_code)

    if EXTERNAL_TRANSLATOR == "deepl":
        import deepl

        auth_key = os.environ["DEEPL_AUTH_KEY"]
        translator = deepl.Translator(auth_key)
        result = translator.translate_text(text, target_lang=language_code.upper())
        return result.text

    if EXTERNAL_TRANSLATOR == "google":
        from google.cloud import translate_v2 as translate

        translator = translate.Client()
        result = translator.translate(text, target_language=language_code)
        return result["translatedText"]

    raise ValueError("EXTERNAL_TRANSLATOR must be either 'google' or 'deepl'.")


def external_language_code(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_CODE_OVERRIDES.get(key, DEFAULT_LANGUAGE_CODES.get(key, key))


@traceable(
    run_type="tool",
    name="candidate_translation_attempt",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
def run_candidate(spec: CandidateSpec, text: str, language: str) -> TranslationCandidate:
    started = time.monotonic()
    set_run_metadata(
        language=language,
        candidate_name=spec.name,
        provider=spec.provider,
        model=spec.model,
        requested_temperature=spec.temperature,
        stance=spec.stance,
    )
    live_log(f"Starting candidate {language} / {spec.name}.")
    try:
        if spec.provider == "openai":
            translated = ask_openai(
                spec.model,
                spec.temperature,
                translation_prompt(text, language, spec.stance),
            )
        else:
            translated = ask_external_translator(text, language)
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


@traceable(
    run_type="chain",
    name="translate_language_candidates",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=lambda outputs: {"candidates": outputs},
)
def translate_language_candidates(text: str, language: str) -> list[dict[str, Any]]:
    specs = build_candidate_specs()
    candidates: list[TranslationCandidate] = []
    parent_run = get_current_run_tree()

    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = []
        for spec in specs:
            if parent_run is None:
                futures.append(executor.submit(run_candidate, spec, text, language))
                continue

            def task(candidate_spec: CandidateSpec = spec) -> TranslationCandidate:
                with set_tracing_parent(parent_run):
                    return run_candidate(candidate_spec, text, language)

            futures.append(executor.submit(task))
        for future in as_completed(futures):
            candidates.append(future.result())

    candidates.sort(key=lambda candidate: candidate.name)
    return [asdict(candidate) for candidate in candidates]


@traceable(
    run_type="chain",
    name="langgraph_generate_candidates",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
def generate_candidates(state: TranslationState) -> dict[str, Any]:
    all_candidates = {}
    set_run_metadata(
        languages=state["target_languages"],
        external_translator=EXTERNAL_TRANSLATOR,
        max_parallel_candidates=MAX_PARALLEL_CANDIDATES,
    )
    events = add_event(
        state,
        "generate_candidates",
        "Starting adversarial candidate generation.",
        languages=state["target_languages"],
        max_parallel_candidates=MAX_PARALLEL_CANDIDATES,
        external_translator=EXTERNAL_TRANSLATOR,
    )

    for language in state["target_languages"]:
        live_log(f"Generating candidates for {language}.")
        candidates = translate_language_candidates(state["text"], language)
        all_candidates[language] = candidates
        events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "step": "generate_candidates",
                "message": f"Generated {len(candidates)} candidates for {language}.",
                "details": {
                    "statuses": {
                        candidate["name"]: candidate["status"]
                        for candidate in candidates
                    },
                    "previews": {
                        candidate["name"]: preview_text(candidate["text"])
                        for candidate in candidates
                    },
                },
            }
        )

    return {"candidate_translations": all_candidates, "decision_log": events}


@traceable(
    run_type="chain",
    name="langgraph_critique_candidates",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
def critique_candidates(state: TranslationState) -> dict[str, Any]:
    reviews = {}
    reasoning = {}
    winners = {}
    events = add_event(
        state,
        "critique_candidates",
        "Starting paragraph-level critic comparison.",
    )

    for language, candidates in state["candidate_translations"].items():
        live_log(f"Running critic for {language}.")
        critique_raw = ask_openai(
            OPENAI_CRITIC_MODEL,
            0.2,
            critic_prompt(state["text"], language, candidates),
        )
        critique = parse_json_response(critique_raw)
        reviews[language] = json.dumps(critique, ensure_ascii=False, indent=2)
        reasoning[language] = str(critique.get("decision_reasoning", ""))
        winners[language] = str(critique.get("overall_winner", "unknown"))
        events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "step": "critique_candidates",
                "message": f"Critic completed review for {language}.",
                "details": {
                    "model": OPENAI_CRITIC_MODEL,
                    "winner": winners[language],
                    "ranking": critique.get("ranking", []),
                    "reasoning_preview": preview_text(reasoning[language]),
                },
            }
        )
        live_log(
            f"Critic for {language} chose {winners[language]}: "
            f"{preview_text(reasoning[language], 120)}"
        )

    return {
        "critic_reviews": reviews,
        "critic_reasoning": reasoning,
        "critic_winners": winners,
        "decision_log": events,
    }


@traceable(
    run_type="chain",
    name="langgraph_summarize_critic",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
def summarize_critic(state: TranslationState) -> dict[str, Any]:
    summaries = {}
    events = add_event(
        state,
        "summarize_critic",
        "Condensing critic reviews into final-generation guidance.",
    )

    for language, critique in state["critic_reviews"].items():
        live_log(f"Summarizing critic guidance for {language}.")
        summaries[language] = ask_openai(
            OPENAI_CRITIC_MODEL,
            0.1,
            summary_prompt(language, critique),
        )
        events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "step": "summarize_critic",
                "message": f"Critic summary completed for {language}.",
                "details": {},
            }
        )

    return {"critic_summaries": summaries, "decision_log": events}


@traceable(
    run_type="chain",
    name="langgraph_generate_final_text",
    project_name=LANGSMITH_PROJECT,
    process_inputs=serialize_for_trace,
    process_outputs=serialize_for_trace,
)
def generate_final_text(state: TranslationState) -> dict[str, Any]:
    finals = {}
    events = add_event(
        state,
        "generate_final_text",
        "Starting final translation generation.",
        model=OPENAI_FINAL_MODEL,
    )

    for language, candidates in state["candidate_translations"].items():
        live_log(f"Generating final translation for {language}.")
        finals[language] = ask_openai(
            OPENAI_FINAL_MODEL,
            0.25,
            final_prompt(
                state["text"],
                language,
                candidates,
                state["critic_summaries"][language],
            ),
        )
        events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "step": "generate_final_text",
                "message": f"Final translation completed for {language}.",
                "details": {
                    "winner_used_as_reference": state.get("critic_winners", {}).get(
                        language, "unknown"
                    ),
                    "final_preview": preview_text(finals[language]),
                },
            }
        )
        live_log(f"Final translation for {language}: {preview_text(finals[language], 120)}")

    return {"final_translations": finals, "decision_log": events}


def build_graph():
    graph_builder = StateGraph(TranslationState)
    graph_builder.add_node("generate_candidates", generate_candidates)
    graph_builder.add_node("critique_candidates", critique_candidates)
    graph_builder.add_node("summarize_critic", summarize_critic)
    graph_builder.add_node("generate_final_text", generate_final_text)

    graph_builder.add_edge(START, "generate_candidates")
    graph_builder.add_edge("generate_candidates", "critique_candidates")
    graph_builder.add_edge("critique_candidates", "summarize_critic")
    graph_builder.add_edge("summarize_critic", "generate_final_text")
    graph_builder.add_edge("generate_final_text", END)

    return graph_builder.compile()


def print_stream_update(update: dict[str, Any]) -> None:
    node_name = next(iter(update.keys()))
    node_update = update[node_name]
    summary = {
        key: (
            list(value.keys())
            if isinstance(value, dict)
            else f"{len(value)} events"
            if key == "decision_log"
            else list(value.keys())
            if key in {"critic_reasoning", "critic_winners"}
            else type(value).__name__
        )
        for key, value in node_update.items()
    }
    console.print(f"[dim]LangGraph node completed:[/dim] {node_name} {summary}")


def print_decision_log(events: list[dict[str, Any]]) -> None:
    table = Table(title="LangGraph Decision Log", show_lines=True)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Node", style="magenta")
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


def print_results(result: TranslationState) -> None:
    console.print(Panel(input_text.strip(), title="Source English", border_style="blue"))

    for language in TARGET_LANGUAGES:
        console.print(f"\n[bold]{language} candidates[/bold]")
        candidate_table = Table(show_lines=True)
        candidate_table.add_column("Candidate")
        candidate_table.add_column("Provider")
        candidate_table.add_column("Status")
        candidate_table.add_column("Latency")
        candidate_table.add_column("Error")
        for candidate in result["candidate_translations"][language]:
            candidate_table.add_row(
                candidate["name"],
                f"{candidate['provider']} / {candidate['model']}",
                candidate["status"],
                f"{candidate['latency_seconds']}s",
                candidate["error"] or "",
            )
        console.print(candidate_table)
        for candidate in result["candidate_translations"][language]:
            console.print(
                Panel(
                    candidate["text"] or (candidate["error"] or "No output"),
                    title=f"{language} Candidate: {candidate['name']}",
                    border_style="cyan",
                )
            )

        console.print(
            Panel(
                result["critic_reviews"][language],
                title=(
                    f"{language} Critic Review "
                    f"(winner: {result.get('critic_winners', {}).get(language, 'unknown')})"
                ),
                border_style="magenta",
            )
        )
        console.print(
            Panel(
                result.get("critic_reasoning", {}).get(language, ""),
                title=f"{language} Critic Reasoning",
                border_style="red",
            )
        )

        console.print(
            Panel(
                result["critic_summaries"][language],
                title=f"{language} Critic Summary",
                border_style="yellow",
            )
        )
        console.print(
            Panel(
                result["final_translations"][language],
                title=f"{language} Final Translation",
                border_style="green",
            )
        )

    print_decision_log(result["decision_log"])

    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        console.print(
            f"\n[dim]LangSmith tracing is enabled. Project: {LANGSMITH_PROJECT}.[/dim]"
        )
    else:
        console.print(
            "\n[dim]Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to inspect"
            " LangGraph traces in LangSmith.[/dim]"
        )


if __name__ == "__main__":
    graph = build_graph()
    initial_state: TranslationState = {
        "text": input_text,
        "target_languages": TARGET_LANGUAGES,
        "decision_log": [],
    }
    config = {
        "run_name": "children_book_translation_advanced",
        "tags": ["children-book", "translation", "adversarial-candidates"],
        "metadata": {
            "project": LANGSMITH_PROJECT,
            "languages": TARGET_LANGUAGES,
            "external_translator": EXTERNAL_TRANSLATOR,
            "max_parallel_candidates": MAX_PARALLEL_CANDIDATES,
        },
    }

    result: TranslationState = initial_state
    for update in graph.stream(initial_state, config=config):
        print_stream_update(update)
        node_update = next(iter(update.values()))
        result = {**result, **node_update}

    print_results(result)
