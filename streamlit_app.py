"""Streamlit test UI for the translation workflows."""

from __future__ import annotations

import html
import json
import re
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from src.translation.config import (
    DEFAULT_CHARACTER_NAMES_CSV,
    DEFAULT_TRANSLATION_CACHE_DIR,
    DEFAULT_TRANSLATION_OUTPUT_DIR,
)
from src.translation.glossary import load_character_glossary
from src.translation.ui_support import (
    CANDIDATE_OPTIONS,
    GENERATION_MODE_OPTIONS,
    JUDGING_MODE_OPTIONS,
    build_ui_config,
    collect_versions,
    comparison_report,
    missing_candidate_credentials,
    missing_judge_credentials,
    panel_aggregate_rows,
    panel_score_rows,
    safe_upload_name,
    workflow_mermaid,
    workflow_node_descriptions,
)
from src.translation.workflow import build_application, save_final_translations


st.set_page_config(page_title="Children's Book Translation", layout="wide")


def render_mermaid(graph: str, descriptions: dict[str, str]) -> None:
    """Render Mermaid with an expandable workflow description list."""

    graph_json = json.dumps(graph)
    block_names_by_id = dict(
        re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\[([^\]]+)\]", graph)
    )
    description_items = "".join(
        f"<li><strong>"
        f"{html.escape(block_names_by_id.get(block_name, block_name))}"
        f"</strong>: "
        f"{html.escape(description)}</li>"
        for block_name, description in descriptions.items()
    )
    components.html(
        f"""
        <style>
          body {{
            margin: 0;
            font-family: sans-serif;
          }}
          #legend {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            color: #475569;
            font-size: 13px;
          }}
          #llm-swatch {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            background: #fde7d7;
            border: 2px solid #c2410c;
          }}
          details {{
            margin-top: 8px;
            color: #334155;
            font-size: 13px;
          }}
          details summary {{
            cursor: pointer;
            font-weight: 600;
          }}
          details ul {{
            margin: 8px 0 0;
            padding-left: 24px;
          }}
          details li {{
            margin-bottom: 5px;
          }}
        </style>
        <div id="legend">
          <span id="llm-swatch"></span> LLM/model call
        </div>
        <div id="workflow-graph"></div>
        <details>
          <summary>Workflow block descriptions</summary>
          <ul>{description_items}</ul>
        </details>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
          const graph = {graph_json};
          mermaid.initialize({{ startOnLoad: false, theme: "neutral" }});
          const {{ svg }} = await mermaid.render("workflow-svg", graph);
          document.getElementById("workflow-graph").innerHTML = svg;
        </script>
        """,
        height=390,
        scrolling=True,
    )


def persist_upload(uploaded_file, fallback: str) -> Path:
    """Persist an uploaded file for the duration of a workflow run."""

    upload_root = Path(tempfile.gettempdir()) / "children_book_translation_ui"
    upload_root.mkdir(parents=True, exist_ok=True)
    path = upload_root / f"{uuid4().hex}_{safe_upload_name(uploaded_file.name, fallback)}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def candidate_controls(count: int) -> tuple[list[str], str, str, str, str]:
    """Render candidate selections and return backend names and model overrides."""

    labels = list(CANDIDATE_OPTIONS)
    defaults = labels[:count]
    selected = []
    openai_base_model = "gpt-4o"
    openai_adversarial_model = "gpt-5.5"
    anthropic_sonnet_model = "claude-sonnet-4-6"
    gemini_model = "gemini-2.5-flash"

    for index in range(count):
        left, right = st.columns([1, 1])
        label = left.selectbox(
            f"Candidate {index + 1}",
            labels,
            index=labels.index(defaults[index]),
            key=f"candidate_type_{index}",
        )
        selected.append(CANDIDATE_OPTIONS[label])
        if label == "OpenAI base":
            openai_base_model = right.text_input(
                "Model",
                value=openai_base_model,
                key=f"candidate_model_{index}",
            )
        elif label == "OpenAI adversarial":
            openai_adversarial_model = right.text_input(
                "Model",
                value=openai_adversarial_model,
                key=f"candidate_model_{index}",
            )
        elif label == "Anthropic Sonnet":
            anthropic_sonnet_model = right.text_input(
                "Model",
                value=anthropic_sonnet_model,
                key=f"candidate_model_{index}",
            )
        elif label == "Google Gemini":
            gemini_model = right.text_input(
                "Model",
                value=gemini_model,
                key=f"candidate_model_{index}",
            )
        else:
            right.text_input(
                "Model",
                value="google",
                disabled=True,
                key=f"candidate_model_{index}",
            )
    return (
        selected,
        openai_base_model,
        openai_adversarial_model,
        anthropic_sonnet_model,
        gemini_model,
    )


def judge_controls(
    candidate_names: list[str],
    openai_base_model: str,
    openai_adversarial_model: str,
    anthropic_sonnet_model: str,
    gemini_model: str,
) -> tuple[list[str], float]:
    """Render candidate-style judge rows, defaulting to selected LLM candidates."""

    judge_options = {
        label: candidate_name
        for label, candidate_name in CANDIDATE_OPTIONS.items()
        if candidate_name != "google_translation"
    }
    labels_by_candidate = {
        candidate_name: label for label, candidate_name in judge_options.items()
    }
    default_labels = list(
        dict.fromkeys(
            labels_by_candidate[name]
            for name in candidate_names
            if name in labels_by_candidate
        )
    )
    labels = list(judge_options)
    ordered_defaults = [*default_labels, *[label for label in labels if label not in default_labels]]
    default_count = max(2, len(default_labels))
    default_key = "|".join(default_labels)
    judge_count = st.number_input(
        "Number of judges",
        min_value=2,
        max_value=len(judge_options),
        value=default_count,
        step=1,
        key=f"judge_count_{default_key}",
        help=(
            "Selected LLM candidates are judges by default. Increase or decrease "
            "this value to add or remove independently configurable judges."
        ),
    )
    judge_temperature = st.slider(
        "Temperature applied to all judges",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="Applied to every panel judgment call.",
    )
    model_defaults = {
        "OpenAI base": openai_base_model,
        "OpenAI adversarial": openai_adversarial_model,
        "Anthropic Sonnet": anthropic_sonnet_model,
        "Google Gemini": gemini_model,
    }
    providers = {
        "OpenAI base": "openai",
        "OpenAI adversarial": "openai",
        "Anthropic Sonnet": "anthropic",
        "Google Gemini": "gemini",
    }
    judges = []
    for index in range(int(judge_count)):
        default_label = ordered_defaults[index]
        left, right = st.columns([1, 1])
        label = left.selectbox(
            f"Judge {index + 1}",
            labels,
            index=labels.index(default_label),
            key=f"judge_type_{default_key}_{index}",
        )
        model = right.text_input(
            "Model",
            value=model_defaults[label],
            key=f"judge_model_{default_key}_{index}",
        ).strip()
        judges.append(f"{providers[label]}:{model}")
    return judges, judge_temperature


def show_run_outputs(state: dict, language: str) -> None:
    """Display generated texts and panel diagnostics."""

    if not st.toggle("Show generated outputs and scores", value=True):
        return

    candidates = state.get("candidate_translations", {}).get(language, [])
    candidate_tab, evaluation_tab, audit_tab = st.tabs(
        ["Generated versions", "Evaluation scores", "Audit and repairs"]
    )
    with candidate_tab:
        for candidate in candidates:
            with st.expander(
                f"{candidate['name']} | {candidate['provider']} / {candidate['model']} "
                f"| {candidate['status']}"
            ):
                st.text(candidate.get("text") or candidate.get("error") or "No output")
        with st.expander("Final translation", expanded=True):
            st.text(state["final_translations"][language])

    with evaluation_tab:
        if state.get("evaluation_mode") == "panel":
            st.subheader("Per-judge paragraph scores")
            st.dataframe(panel_score_rows(state, language), use_container_width=True)
            st.subheader("Aggregated paragraph rankings")
            st.dataframe(panel_aggregate_rows(state, language), use_container_width=True)
            with st.expander("Raw panel aggregates"):
                st.json(state.get("panel_aggregates", {}).get(language, {}))
        else:
            st.subheader("Single critic")
            st.json(
                json.loads(state.get("critic_reviews", {}).get(language, "{}"))
            )

    with audit_tab:
        if state.get("evaluation_mode") == "panel":
            st.subheader("Whole-book audit")
            st.json(state.get("book_audits", {}).get(language, {}))
            st.subheader("Targeted repairs")
            st.json(state.get("repair_results", {}).get(language, {}))
        else:
            st.info("Whole-book audit and targeted repairs are available in panel mode.")


def show_comparator(state: dict, language: str) -> None:
    """Render selectable version comparison using the existing report code."""

    versions = collect_versions(state, language)
    if len(versions) < 2:
        return
    st.header("Version comparator")
    left_column, right_column = st.columns(2)
    labels = list(versions)
    left = left_column.selectbox("Left version", labels, index=0)
    right = right_column.selectbox(
        "Right version", labels, index=len(labels) - 1
    )
    report = comparison_report(versions, left, right)
    st.markdown(report, unsafe_allow_html=True)


st.title("Children's Book Translation")
st.caption(
    "Run and inspect text-only, multimodal, and panel-judging translation workflows."
)

glossary = load_character_glossary(DEFAULT_CHARACTER_NAMES_CSV)
languages = glossary.default_target_languages()

with st.sidebar:
    st.header("Workflow configuration")
    generation_mode_label = st.selectbox("Mode", list(GENERATION_MODE_OPTIONS))
    judging_mode_label = st.selectbox("Judging mode", list(JUDGING_MODE_OPTIONS))
    target_language = st.selectbox("Target language", languages)
    candidate_count = st.number_input(
        "Number of candidates",
        min_value=2,
        max_value=len(CANDIDATE_OPTIONS),
        value=3,
        step=1,
        help="Select between two and five unique candidate strategies.",
    )
    candidate_temperature = st.slider(
        "Temperature applied to all candidates",
        min_value=0.0,
        max_value=2.0,
        value=0.4,
        step=0.1,
        help="Applied to model-based candidates. Google Translate ignores temperature.",
    )
    (
        candidate_names,
        openai_base_model,
        openai_adversarial_model,
        anthropic_sonnet_model,
        gemini_model,
    ) = candidate_controls(int(candidate_count))
    panel_judges = []
    panel_judge_temperature = 0.1
    if judging_mode_label == "Panel judges":
        panel_judges, panel_judge_temperature = judge_controls(
            candidate_names,
            openai_base_model,
            openai_adversarial_model,
            anthropic_sonnet_model,
            gemini_model,
        )
        if generation_mode_label == "Multimodal":
            st.caption(
                "Candidates use the PDF spread images; panel judges evaluate the "
                "resulting aligned candidate text."
            )
    st.divider()
    source_text_upload = st.file_uploader(
        "Cleaned source text",
        type=["txt"],
        help="Paragraph breaks must match the intended translated paragraph structure.",
    )
    source_pdf_upload = None
    if generation_mode_label == "Multimodal":
        source_pdf_upload = st.file_uploader("Source PDF", type=["pdf"])
        st.caption(
            "Multimodal mode requires both the cleaned paragraph-aligned text and PDF."
        )
    run_clicked = st.button("Run translation", type="primary", use_container_width=True)

st.header("Selected workflow")
render_mermaid(
    workflow_mermaid(generation_mode_label, judging_mode_label),
    workflow_node_descriptions(generation_mode_label, judging_mode_label),
)

with st.expander("Run configuration", expanded=False):
    st.json(
        {
            "mode": generation_mode_label,
            "judging_mode": judging_mode_label,
            "target_language": target_language,
            "candidate_names": candidate_names,
            "candidate_temperature": candidate_temperature,
            "openai_base_model": openai_base_model,
            "openai_adversarial_model": openai_adversarial_model,
            "anthropic_sonnet_model": anthropic_sonnet_model,
            "gemini_model": gemini_model,
            "panel_judges": panel_judges,
            "panel_judge_temperature": panel_judge_temperature,
        }
    )

if run_clicked:
    errors = []
    if source_text_upload is None:
        errors.append("Upload a cleaned source text file.")
    if generation_mode_label == "Multimodal" and source_pdf_upload is None:
        errors.append("Upload the source PDF for multimodal mode.")
    if len(set(candidate_names)) != len(candidate_names):
        errors.append("Candidate selections must be unique.")
    if judging_mode_label == "Panel judges" and len(set(panel_judges)) < 2:
        errors.append("Panel judging requires at least two distinct judge models.")
    if judging_mode_label == "Panel judges" and len(set(panel_judges)) != len(panel_judges):
        errors.append("Judge selections must be unique.")
    if judging_mode_label == "Panel judges" and any(judge.endswith(":") for judge in panel_judges):
        errors.append("Every judge must have a model name.")
    errors.extend(missing_candidate_credentials(candidate_names))
    errors.extend(missing_judge_credentials(panel_judges))
    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            source_text_path = persist_upload(source_text_upload, "source.txt")
            source_pdf_path = (
                persist_upload(source_pdf_upload, "source.pdf")
                if source_pdf_upload is not None
                else None
            )
            config = build_ui_config(
                generation_mode_label=generation_mode_label,
                judging_mode_label=judging_mode_label,
                source_text_path=source_text_path,
                source_pdf_path=source_pdf_path,
                target_language=target_language,
                candidate_names=candidate_names,
                candidate_temperature=candidate_temperature,
                openai_base_model=openai_base_model,
                openai_adversarial_model=openai_adversarial_model,
                anthropic_sonnet_model=anthropic_sonnet_model,
                gemini_model=gemini_model,
                panel_judges=panel_judges,
                panel_judge_temperature=panel_judge_temperature,
                output_dir=DEFAULT_TRANSLATION_OUTPUT_DIR,
                cache_dir=DEFAULT_TRANSLATION_CACHE_DIR,
                burr_storage_dir="/private/tmp/.burr-streamlit",
            )
            with st.spinner("Running translation workflow..."):
                app, config = build_application(config)
                terminal_action = (
                    "repair_flagged_paragraphs"
                    if config.is_panel_mode()
                    else "generate_final_text"
                )
                _, _, final_state = app.run(halt_after=[terminal_action])
                bundle = save_final_translations(final_state, config)
            st.session_state["translation_run"] = {
                "state": dict(final_state),
                "config": config,
                "bundle": bundle,
                "language": target_language,
            }
            st.success(f"Translation run {config.run_id} completed.")
        except Exception as exc:
            st.exception(exc)

run = st.session_state.get("translation_run")
if run:
    state = run["state"]
    language = run["language"]
    final_text = state["final_translations"][language]
    st.header("Download")
    st.download_button(
        "Download final translation",
        data=final_text,
        file_name=run["config"].versioned_translation_path(language).name,
        mime="text/plain",
    )
    show_run_outputs(state, language)
    show_comparator(state, language)
