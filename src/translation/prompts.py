"""Prompt builders for the translation workflow."""

from __future__ import annotations

from typing import Any

from .glossary import CharacterGlossary
from .segmentation import SpreadSegment


def neutral_candidate_label(index: int) -> str:
    """Return a stable model-facing label that hides candidate identity."""

    return f"Candidate {index + 1}"


def neutral_candidate_blocks(
    candidates: list[dict[str, Any]], *, successful_only: bool = False
) -> str:
    """Format candidate texts without exposing names, providers, or models."""

    return "\n\n".join(
        f"{neutral_candidate_label(index)}\n"
        f"Status: {candidate.get('status', 'unknown')}\n"
        f"{candidate.get('text', '')}"
        for index, candidate in enumerate(candidates)
        if not successful_only or candidate.get("status") == "ok"
    )


def translation_prompt(
    text: str,
    source_language: str,
    target_language: str,
    stance: str,
    glossary: CharacterGlossary,
    profile: str = "normal",
    visual_context: str = "",
) -> str:
    """Prompt for one translation candidate."""
    creative_guidance = (
        "- You may reshape syntax and idioms for livelier storytelling, while preserving "
        "every fact, action, character, and narrative intention.\n"
        "- Prefer vivid, playful, memorable read-aloud phrasing over literal structure."
        if profile == "creative"
        else "- Prefer faithful, polished phrasing; alter the source only when natural target-language usage requires it."
    )
    visual_block = f"\nLocked visual context (do not invent beyond it):\n{visual_context}\n" if visual_context else ""
    return f"""
You are translating a Barbapapa children's book from {source_language} into {target_language}.

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech.
- Keep at best the original line breaks
- Stay faithful to the source scene and tone
- Do not add explanations, notes, or Markdown

Language guidance:
- Use natural modern {target_language}
- Avoid overly formal wording
- Keep repetitions and rhythms that work well in read-aloud storytelling
{creative_guidance}

Translator stance:
- {stance}

Character and side-character names:
{glossary.format_name_guidance(target_language)}

Source text in {source_language}:
{text}
{visual_block}
"""


def critic_prompt(
    text: str,
    source_language: str,
    target_language: str,
    candidates: list[dict[str, Any]],
    glossary: CharacterGlossary,
) -> str:
    """Prompt for comparing candidate translations."""
    candidate_blocks = neutral_candidate_blocks(candidates)
    return f"""
You are a senior editor for translated picture books.

The source language is {source_language}. The target language is {target_language}.
Candidate identities, providers, model names, and generation settings are hidden
to avoid bias. Refer to candidates only by the neutral labels shown below.
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
  "overall_winner": "Candidate 1",
  "ranking": ["Candidate 1", "Candidate 2", "Candidate 3"],
  "decision_reasoning": "Explain clearly why the winning candidate wins and what tradeoffs drove the decision.",
  "paragraph_analysis": [
    {{
      "paragraph_number": 1,
      "best_candidate": "Candidate 1",
      "notes": "Comparison notes for this paragraph."
    }}
  ],
  "candidate_assessment": [
    {{
      "candidate": "Candidate 1",
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
    profile: str = "normal",
    visual_context: str = "",
) -> str:
    """Prompt for one spread-aligned translation segment."""

    spread_label = ""
    if spread_pages:
        spread_label = (
            f"\nCurrent spread pages: {', '.join(str(page) for page in spread_pages)}"
        )

    previous_source_context = segment.previous_source_text.strip()
    previous_translated_context = "\n\n".join(previous_translated_segments or []).strip()

    creative_guidance = (
        "- You may reshape syntax and idioms for livelier storytelling, while preserving every visible and stated action.\n"
        "- Prefer vivid, playful, memorable read-aloud phrasing."
        if profile == "creative"
        else "- Prefer faithful, polished phrasing and conservative adaptation."
    )
    visual_block = f"\nLocked visual context:\n{visual_context}\n" if visual_context else ""
    return f"""
You are translating one double-page spread segment of a Barbapapa children's book from {source_language} into {target_language}.{spread_label}

Audience and style:
- Children aged 5-8
- Warm, simple, playful
- Easy to read aloud
- Preserve paragraph breaks and quoted speech
- Keep at best the original line breaks
- Do not create blank lines inside one page's translated text
- Stay faithful to the source scene and tone
- Do not add explanations, notes, or Markdown

Language guidance:
- Use natural modern {target_language}
- Avoid overly formal wording
- Keep repetitions and rhythms that work well in read-aloud storytelling
{creative_guidance}

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
{visual_block}

Return only the translation of the current source pages in {target_language}.
If the current spread contains text from two body pages, keep the two translated page blocks in order and separate them with one blank line.
The number of returned page blocks must exactly match the number of current source page blocks.
"""


def summary_prompt(target_language: str, critique: str) -> str:
    """Prompt for condensing the critic review."""

    return f"""
Summarize this translation critique for the final {target_language} translator.
Focus on decisions, tradeoffs, and concrete revision instructions.
Candidate identities are intentionally hidden. Refer to candidates only by the
neutral labels present in the critique. Do not guess or mention providers,
model names, or generation settings.

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
    candidate_blocks = neutral_candidate_blocks(candidates)
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
- Treat candidate labels as neutral references; do not infer quality from order
- Return only the final translation as plain text

Source text in {source_language}:
{text}

Candidate translations:
{candidate_blocks}

Critic summary:
{critique_summary}
"""


def aligned_final_paragraph_prompt(
    *,
    source_paragraph: str,
    previous_source: str,
    next_source: str,
    previous_final: str,
    source_language: str,
    target_language: str,
    candidates: list[dict[str, Any]],
    critique_summary: str,
    glossary: CharacterGlossary,
    paragraph_number: int,
    paragraph_count: int,
) -> str:
    """Prompt for one page-aligned final paragraph in multimodal single mode."""

    candidate_blocks = neutral_candidate_blocks(candidates, successful_only=True)
    return f"""
Create only paragraph {paragraph_number} of {paragraph_count} for the final
{target_language} translation of this children's book.

The current source paragraph corresponds to exactly one text-bearing PDF page.
Return exactly one paragraph. Do not include blank lines, paragraph labels,
Markdown, commentary, previous paragraphs, or following paragraphs.

Requirements:
- Translate only the current {source_language} source paragraph
- Use the candidate books and critic summary as editorial references
- Preserve every action, quote, speaker, and required character name
- Keep the tone warm, natural, and easy to read aloud
- Maintain continuity with the previous final paragraph
- The candidate books were generated with spread images. Preserve visible
  actions and scene details supported by the candidates and source, but do not
  invent details merely because they appear in only one candidate.
- Treat candidate labels as neutral references; do not infer quality from order

Required character names:
{glossary.format_name_guidance(target_language)}

Previous source context:
{previous_source or "(none)"}

Current source paragraph:
{source_paragraph}

Next source context:
{next_source or "(none)"}

Previous final target paragraph:
{previous_final or "(none)"}

Candidate translation books:
{candidate_blocks}

Critic summary:
{critique_summary}
"""
