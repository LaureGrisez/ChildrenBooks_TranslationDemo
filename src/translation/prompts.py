"""Prompt builders for the translation workflow."""

from __future__ import annotations

from typing import Any

from .glossary import CharacterGlossary
from .segmentation import SpreadSegment


def translation_prompt(
    text: str,
    source_language: str,
    target_language: str,
    stance: str,
    glossary: CharacterGlossary,
) -> str:
    """Prompt for one translation candidate."""

    return f"""
You are translating a Barbapapa children's book from {source_language} into {target_language}.

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech
- Keep at best the original line breaks
- Stay faithful to the source scene and tone
- Do not add explanations, notes, or Markdown

Language guidance:
- Use natural modern {target_language}
- Avoid overly formal wording
- Keep repetitions and rhythms that work well in read-aloud storytelling

Translator stance:
- {stance}

Character and side-character names:
{glossary.format_name_guidance(target_language)}

Source text in {source_language}:
{text}
"""


def critic_prompt(
    text: str,
    source_language: str,
    target_language: str,
    candidates: list[dict[str, Any]],
    glossary: CharacterGlossary,
) -> str:
    """Prompt for comparing candidate translations."""

    candidate_blocks = "\n\n".join(
        f"Candidate {idx + 1}: {candidate['name']} "
        f"({candidate['provider']} / {candidate['model']} / T={candidate['temperature']})\n"
        f"Status: {candidate['status']}\n"
        f"{candidate['text']}"
        for idx, candidate in enumerate(candidates)
    )
    return f"""
You are a senior editor for translated picture books.

The source language is {source_language}. The target language is {target_language}.
Use this name guidance when checking consistency:
{glossary.format_name_guidance(target_language)}

Compare the candidate translations paragraph by paragraph. Evaluate:
- faithfulness to the {source_language} source
- natural child-friendly phrasing
- read-aloud rhythm
- consistency of character names
- cultural fit without over-localizing
- handling of quotes, imagery, and paragraph breaks

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

Source text in {source_language}:
{text}

Candidates:
{candidate_blocks}
"""


def segmented_translation_prompt(
    segment: SpreadSegment,
    source_language: str,
    target_language: str,
    stance: str,
    glossary: CharacterGlossary,
    total_segments: int,
    previous_translated_segments: list[str] | None = None,
    spread_pages: tuple[int, ...] | None = None,
) -> str:
    """Prompt for one spread-aligned translation segment."""

    spread_label = ""
    if spread_pages:
        spread_label = (
            f"\nCurrent spread pages: {', '.join(str(page) for page in spread_pages)}"
        )

    previous_source_context = segment.previous_source_text.strip()
    previous_translated_context = "\n\n".join(previous_translated_segments or []).strip()

    return f"""
You are translating one double-page spread segment of a Barbapapa children's book from {source_language} into {target_language}.{spread_label}

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech
- Keep at best the original line breaks
- Stay faithful to the source scene and tone
- Do not add explanations, notes, or Markdown

Language guidance:
- Use natural modern {target_language}
- Avoid overly formal wording
- Keep repetitions and rhythms that work well in read-aloud storytelling

Translator stance:
- {stance}

Character and side-character names:
{glossary.format_name_guidance(target_language)}

Book context:
- Segment {segment.index + 1} of {total_segments}
- Previously translated segments in {target_language}:
{previous_translated_context or "(none)"}

- Previous source context in {source_language}:
{previous_source_context or "(none)"}

Current source pages in {source_language}:
{segment.source_text}

Return only the translation of the current source pages in {target_language}.
If the current spread contains text from two body pages, keep the two translated page blocks in order and separate them with one blank line.
"""


def summary_prompt(target_language: str, critique: str) -> str:
    """Prompt for condensing the critic review."""

    return f"""
Summarize this translation critique for the final {target_language} translator.
Focus on decisions, tradeoffs, and concrete revision instructions.

Critique:
{critique}
"""


def final_prompt(
    text: str,
    source_language: str,
    target_language: str,
    candidates: list[dict[str, Any]],
    critique_summary: str,
    glossary: CharacterGlossary,
) -> str:
    """Prompt for composing the final translation."""

    candidate_blocks = "\n\n".join(
        f"{candidate['name']}:\n{candidate['text']}" for candidate in candidates
    )
    return f"""
Create the final {target_language} translation for a Barbapapa children's book.

Requirements:
- Translate from {source_language} into natural {target_language}
- Use the critic summary as editorial guidance
- Keep the character and side-character names consistent with this list:
{glossary.format_name_guidance(target_language)}
- Preserve paragraph breaks and quoted speech
- Keep at best the original line breaks
- Keep the tone warm, clear, and easy to read aloud
- Return only the final translation as plain text

Source text in {source_language}:
{text}

Candidate translations:
{candidate_blocks}

Critic summary:
{critique_summary}
"""
