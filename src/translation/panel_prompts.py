"""Prompt builders for panel judging, synthesis, and audit."""

from __future__ import annotations

import json
from typing import Any

from .glossary import CharacterGlossary
from .panel_models import CRITERIA


def _anonymize_panel_guidance(
    selected_options: dict[str, str], aggregate: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Hide restored candidate identities before panel-guided synthesis."""

    candidate_names = list(aggregate["ranking"])
    labels = {
        candidate: f"Option {index + 1}"
        for index, candidate in enumerate(candidate_names)
    }

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return labels.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {labels.get(key, key): replace(item) for key, item in value.items()}
        return value

    anonymous_options = {
        labels[candidate]: text for candidate, text in selected_options.items()
    }
    guidance = replace(
        {
            "ranking": aggregate["ranking"],
            "confirmed_critical_errors": aggregate["confirmed_critical_errors"],
            "consensus_remarks": aggregate["consensus_remarks"],
            "recommended_phrases": aggregate["recommended_phrases"],
        }
    )
    return anonymous_options, guidance


def judge_prompt(
    *,
    block: dict[str, Any],
    target_language: str,
    glossary: CharacterGlossary,
    blinded_options: dict[str, str],
    visual_context: str = "",
    image_attached: bool = False,
) -> str:
    """Build one blinded comparative paragraph-evaluation prompt."""

    score_shape = ",\n        ".join(f'"{criterion}": 0' for criterion in CRITERIA)
    visual_block = (
        f"\nLocked visual evidence for this paragraph:\n{visual_context}\n"
        if visual_context
        else ("\nA source-spread image is attached. Use it only for visual grounding.\n" if image_attached else "")
    )
    return f"""
You are an independent senior judge of children's-book translations.
Compare only the current paragraph options. Identify concrete evidence before
scoring. Candidate identities and providers are intentionally hidden.

Target language: {target_language}
Required character names:
{glossary.format_name_guidance(target_language)}

Previous source paragraph:
{block["previous_source"] or "(none)"}

Current source paragraph:
{block["source"]}

Next source paragraph:
{block["next_source"] or "(none)"}
{visual_block}

Blinded options:
{json.dumps(blinded_options, ensure_ascii=False, indent=2)}

Return valid JSON only:
{{
  "comparisons": [
    {{"criterion": "faithfulness", "preferred_options": ["option_a"], "evidence": "..."}}
  ],
  "option_scores": {{
    "option_a": {{
        {score_shape},
        "critical_errors": [],
        "remarks": ["..."]
    }}
  }},
  "overall_ranking": ["option_a", "option_b"],
  "recommended_phrases": [
    {{"option": "option_a", "phrase": "...", "reason": "..."}}
  ],
  "confidence": 0.8
}}

Score every option from 0 to 10 for every criterion. Include every option
exactly once in overall_ranking and option_scores.
"""


def synthesis_prompt(
    *,
    block: dict[str, Any],
    target_language: str,
    glossary: CharacterGlossary,
    selected_options: dict[str, str],
    aggregate: dict[str, Any],
    previous_final: str,
    visual_context: str = "",
    image_attached: bool = False,
) -> str:
    """Build one sequential final-paragraph synthesis prompt."""

    anonymous_options, guidance = _anonymize_panel_guidance(
        selected_options, aggregate
    )
    visual_block = (
        f"\nLocked visual evidence:\n{visual_context}\n"
        if visual_context
        else ("\nA source-spread image is attached for visual grounding.\n" if image_attached else "")
    )
    return f"""
Create the final {target_language} translation of the current source paragraph.
Return only that one translated paragraph, without Markdown or commentary.

Requirements:
- Warm, natural language for children aged 5-8
- Faithful meaning and child-friendly read-aloud rhythm
- Preserve quotes and all actions
- Follow required character names
- Candidate identities are hidden. Treat option labels only as neutral references.

Required character names:
{glossary.format_name_guidance(target_language)}

Previous final target paragraph:
{previous_final or "(none)"}

Previous source paragraph:
{block["previous_source"] or "(none)"}

Current source paragraph:
{block["source"]}

Next source paragraph:
{block["next_source"] or "(none)"}
{visual_block}

Selected candidate options:
{json.dumps(anonymous_options, ensure_ascii=False, indent=2)}

Frozen panel guidance:
{json.dumps(guidance, ensure_ascii=False, indent=2)}
"""


def audit_prompt(
    *, source_text: str, final_text: str, target_language: str, glossary: CharacterGlossary,
    visual_context: str = "", images_attached: bool = False,
) -> str:
    """Build a paragraph-scoped whole-book consistency audit prompt."""

    visual_block = (
        f"\nLocked visual evidence by page:\n{visual_context}\n"
        if visual_context
        else ("\nSource-spread images are attached for visual consistency checking.\n" if images_attached else "")
    )
    return f"""
Audit this complete {target_language} children's-book translation for consistency.
Do not rewrite the book. Flag only paragraphs that require repair.

Check character names, terminology, narrative voice, tense, repeated phrases,
transitions, read-aloud rhythm, quotes, paragraph structure, and source meaning.

Required character names:
{glossary.format_name_guidance(target_language)}

Source:
{source_text}

Final translation:
{final_text}
{visual_block}

Return valid JSON only:
{{
  "findings": [
    {{
      "paragraph_id": "p0001",
      "severity": "critical|major|minor",
      "instruction": "Specific repair instruction"
    }}
  ]
}}
"""


def repair_prompt(
    *,
    block: dict[str, Any],
    current_final: str,
    previous_final: str,
    next_final: str,
    findings: list[dict[str, Any]],
    target_language: str,
    glossary: CharacterGlossary,
) -> str:
    """Build a targeted repair prompt for one flagged paragraph."""

    return f"""
Repair only the current {target_language} paragraph using the audit findings.
Return only the repaired paragraph.

Required character names:
{glossary.format_name_guidance(target_language)}

Source paragraph:
{block["source"]}

Previous final paragraph:
{previous_final or "(none)"}

Current final paragraph:
{current_final}

Next final paragraph:
{next_final or "(none)"}

Audit findings:
{json.dumps(findings, ensure_ascii=False, indent=2)}
"""
