"""Standalone French text adaptation from double-page spreads to single pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import fitz  # PyMuPDF
from PIL import Image

from .cache import ResponseCache
from .config import TranslationWorkflowConfig
from .multimodal import (
    collect_body_pages,
    collect_non_empty_body_pages,
    render_spread_image_bytes,
)
from .providers import ask_model_with_recovery
from .segmentation import split_source_paragraphs, spread_pages_for_body_page
from src.utils.pdf_translation_overlay import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    add_redactions,
    collect_page_lines,
    insert_translation,
    register_font,
)
from src.utils.pdf_compressor import compress_pdf


DEFAULT_BLANK_TEXT_MARGIN_RATIO = 0.08
DEFAULT_BLANK_TEXT_HEIGHT_RATIO = 0.24
STORY_CONTEXT_SPREADS_BEFORE = 1
STORY_CONTEXT_SPREADS_AFTER = 2
STORY_PLAN_PAGES_BEFORE = 1
STORY_PLAN_PAGES_AFTER = 2
STORY_PLANNER_CONTEXT_SPREADS = 1
DEFAULT_STORY_PLANNER_SPREADS_PER_CALL = 5
DEFAULT_IMAGE_SUMMARY_MODEL = "openai:gpt-4o"
DEFAULT_IMAGE_SUMMARY_TEMPERATURE = 0.2
IMAGE_SUMMARY_SCHEMA_VERSION = 2
IMAGE_SUMMARY_PROMPT_VERSION = "barbapapa-spread-and-visible-pages-v2"
PREPROCESSED_SOURCE_MARKERS = (".single_page", "_single_page", ".single-page")


@dataclass(frozen=True, slots=True)
class PageSource:
    """One physical body page and its original French text, if any."""

    page_number: int
    source_text: str


@dataclass(frozen=True, slots=True)
class SpreadSource:
    """One physical spread presented to the adaptation model."""

    index: int
    pages: tuple[PageSource, ...]

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(page.page_number for page in self.pages)


def build_spread_sources(
    source_text: str,
    body_page_numbers: list[int],
    text_page_numbers: list[int],
) -> list[SpreadSource]:
    """Map cleaned source paragraphs to every physical page and group by spread."""

    paragraphs = split_source_paragraphs(source_text)
    if len(paragraphs) != len(text_page_numbers):
        raise ValueError(
            "The number of source paragraphs must match the number of text-bearing "
            f"body pages. Found {len(paragraphs)} paragraphs and "
            f"{len(text_page_numbers)} text-bearing pages."
        )

    text_by_page = dict(zip(text_page_numbers, paragraphs, strict=True))
    grouped: list[list[PageSource]] = []
    previous_spread: tuple[int, ...] | None = None
    for page_number in body_page_numbers:
        spread_pages = spread_pages_for_body_page(page_number)
        if spread_pages != previous_spread:
            grouped.append([])
            previous_spread = spread_pages
        grouped[-1].append(
            PageSource(
                page_number=page_number,
                source_text=text_by_page.get(page_number, ""),
            )
        )
    return [
        SpreadSource(index=index, pages=tuple(pages))
        for index, pages in enumerate(grouped)
    ]


def _page_source_blocks(spread: SpreadSource) -> str:
    return "\n\n".join(
        f"PAGE {page.page_number}\n{page.source_text or '(aucun texte original)'}"
        for page in spread.pages
    )


def editorial_plan_prompt(
    *,
    spread: SpreadSource,
    previous_source: str,
    next_source: str,
    previous_adapted: str,
) -> str:
    """Ask for a conservative page-by-page adaptation plan."""

    page_numbers = ", ".join(str(page) for page in spread.page_numbers)
    return f"""
Tu es éditrice ou éditeur de livres illustrés pour enfants. Tu adaptes la mise
en page française d'un album conçu en doubles pages vers une lecture numérique
page par page. L'image jointe montre la double page actuelle, sans son texte.

Fais uniquement un PLAN éditorial pour les pages physiques {page_numbers}.
Observe séparément chaque page de l'image, de gauche à droite.

Principes:
- Préserver l'histoire, la voix, l'humour, les faits et l'ordre des actions.
- Modifier le moins possible le texte original lorsqu'il fonctionne déjà.
- Chaque page doit pouvoir être lue seule avec son image sans devenir confuse.
- Respecter strictement la chronologie de lecture page par page: la page de
  gauche est lue avant la page de droite, et la double page suivante vient
  seulement après.
- Identifier pour chaque page son rôle narratif: amorce, action, réaction,
  conclusion ou transition. Une conclusion doit rester après l'action qu'elle
  conclut; une solution doit rester après le problème qu'elle résout.
- Répartir ou déplacer le texte de la double page si l'action décrite se trouve
  surtout sur l'autre page.
- Sur une page initialement muette, ajouter seulement une phrase courte si son
  image ou l'enchaînement serait autrement incompréhensible ou abrupt.
- Pour une action révélée sur la page suivante, préférer une amorce ou un petit
  suspense à une description qui gâche la surprise.
- Si une page montre une tentative, un obstacle ou une action en cours, ne place
  pas avant elle une phrase qui commente déjà son échec, sa réussite ou sa
  solution, sauf si cette phrase appartient vraiment à la page précédente.
- Ne jamais inventer un événement, un personnage, un dialogue ou un détail
  invisible et non soutenu par le contexte.
- Ne pas traduire: tout reste en français.

Contexte source précédent:
{previous_source or "(aucun)"}

Texte déjà adapté sur la double page précédente:
{previous_adapted or "(aucun)"}

Pages actuelles et texte original:
{_page_source_blocks(spread)}

Contexte source suivant:
{next_source or "(aucun)"}

Retourne uniquement un objet JSON valide:
{{
  "spread_assessment": "Pourquoi une adaptation est ou non nécessaire",
  "chronology_check": "Ordre narratif page par page, en indiquant ce qui doit rester avant ou après quoi",
  "pages": [
    {{
      "page_number": {spread.page_numbers[0]},
      "intervention": "keep|split|move|add|rewrite",
      "narrative_role": "setup|action|reaction|conclusion|transition",
      "visual_action": "Action visible utile à l'édition",
      "writing_instruction": "Instruction concise pour le texte final"
    }}
  ],
  "continuity_note": "Point à préserver pour la double page suivante"
}}
La liste pages doit contenir exactement une entrée pour chaque page physique:
{page_numbers}.
""".strip()


def rewrite_prompt(
    *,
    spread: SpreadSource,
    plan: dict[str, Any],
    previous_adapted: str,
) -> str:
    """Ask for final French text after the editorial planning pass."""

    page_numbers = ", ".join(str(page) for page in spread.page_numbers)
    return f"""
Tu es autrice ou auteur jeunesse francophone. À partir de l'image jointe, du
texte original et du plan éditorial, écris le texte final d'une version
numérique lue une page à la fois.

Contraintes absolues:
- Produire exactement un texte non vide pour chacune des pages {page_numbers}.
- Écrire uniquement en français naturel, chaleureux et facile à lire à voix haute.
- Conserver autant que possible les formulations, dialogues et rythme originaux.
- Faire des ajouts brefs. Une page muette n'a souvent besoin que d'une amorce,
  d'une transition, d'une réaction ou d'un petit effet de suspense.
- Respecter l'ordre des actions dans la lecture page par page. Avant d'écrire,
  vérifie mentalement que chaque page prépare, montre ou conclut l'action au bon
  moment.
- Ne place jamais une conclusion, une explication ou une solution avant la page
  qui montre l'action, le problème ou la tentative correspondante.
- Si le plan éditorial inverse une cause et sa conséquence, corrige cet ordre
  dans le texte final en gardant la chronologie de l'image et du récit.
- Ne pas répéter le même texte ou la même information sur les deux pages.
- Ne pas annoncer une révélation avant la page où elle devient visible.
- Ne rien inventer qui ne soit soutenu par l'image, le texte ou le contexte.
- Ne pas inclure de numéro de page, commentaire ou Markdown dans le champ text.

Texte déjà adapté sur la double page précédente:
{previous_adapted or "(aucun)"}

Texte original des pages actuelles:
{_page_source_blocks(spread)}

Plan éditorial:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Retourne uniquement un objet JSON valide:
{{
  "pages": [
    {{
      "page_number": {spread.page_numbers[0]},
      "text": "Texte final de cette page",
      "intervention": "keep|split|move|add|rewrite",
      "narrative_role": "setup|action|reaction|conclusion|transition",
      "change_note": "Justification très courte"
    }}
  ]
}}
La liste pages doit contenir exactement une entrée pour chaque page physique,
dans cet ordre: {page_numbers}.
""".strip()


def story_prompt(
    *,
    source_text: str,
    page_numbers: list[int],
    story_plan: dict[str, Any] | None = None,
) -> str:
    """Prompt for the original one-call whole-story adaptation behavior."""

    page_list = ", ".join(str(page_number) for page_number in page_numbers)
    if story_plan is None:
        plan_section = "(aucun plan verrouillé fourni)"
        plan_constraint = "Déduire prudemment la progression à partir du texte et des images."
    else:
        plan_section = json.dumps(story_plan, ensure_ascii=False, indent=2)
        plan_constraint = (
            "Suivre le PLAN PAGE PAR PAGE comme une contrainte verrouillée: ne pas "
            "réinterpréter la progression, les interdits ou les éléments à préserver."
        )
    return f"""
Tu es éditrice ou éditeur narratif pour albums illustrés jeunesse. Nous voulons
adapter une histoire française conçue pour une double-page papier vers un livre
numérique où chaque page est lue seule.

Tu reçois:
- le texte complet de l'histoire;
- les images sans texte des pages physiques, jointes dans cet ordre:
  {page_list}.

Objectif:
Créer une version page par page où chaque page a son propre texte, tout en
gardant la progression globale de l'histoire.

Méthode attendue:
- Pense d'abord à l'arc complet: situation, problème, tentatives, solution,
  conséquences, retour au calme.
- Découpe ensuite l'histoire en petits moments narratifs correspondant aux
  images, dans l'ordre strict des pages.
- Chaque page doit être compréhensible avec son image et donner envie de tourner
  la page.
- Les pages sans texte original peuvent recevoir une phrase courte si elles ont
  besoin d'une amorce, d'une réaction, d'une transition ou d'un suspense.
- Une page qui montre la conséquence ou la fin d'une action doit venir après la
  page qui montre l'action. Ne mets jamais une conclusion avant sa cause.
- Ne répète pas la même information sur deux pages voisines.
- Ne transforme pas l'histoire en résumé: écris un vrai texte d'album, simple,
  oral, chaleureux et enfantin.
- Ne change pas les personnages, les faits importants, ni l'ordre des actions.
- N'invente pas un événement majeur absent du texte complet et des images.
- Tout doit rester en français.

Longueur:
- En général 1 à 2 phrases courtes par page.
- Utilise moins si l'image porte déjà très bien le moment.
- Utilise un peu plus seulement si la page serait confuse autrement.

Texte complet de l'histoire:
{source_text}

PLAN PAGE PAR PAGE

{plan_section}

Contrainte de plan:
- {plan_constraint}

Retourne uniquement un objet JSON valide:
{{
  "story_strategy": "Résumé bref de la logique d'adaptation globale",
  "chronology_check": "Vérification de l'ordre narratif sur les pages demandées",
  "pages": [
    {{
      "page_number": {page_numbers[0] if page_numbers else 0},
      "text": "Texte final de cette page",
      "narrative_role": "setup|action|reaction|conclusion|transition",
      "visual_grounding": "Ce qui, dans l'image, justifie ce texte",
      "source_story_beats": ["Moment(s) du texte complet utilisés"],
      "change_note": "Pourquoi ce choix fonctionne pour la lecture page par page"
    }}
  ]
}}
La liste pages doit contenir exactement une entrée pour chaque page physique,
dans cet ordre: {page_list}.
""".strip()


def _page_labeled_source_blocks(spreads: list[SpreadSource]) -> str:
    """Return source text with explicit spread and physical-page boundaries."""

    blocks: list[str] = []
    for spread in spreads:
        page_list = ", ".join(str(page) for page in spread.page_numbers)
        page_blocks = "\n\n".join(
            f"PAGE {page.page_number}\n"
            f"{page.source_text or '(aucun texte original imprimé sur cette page)'}"
            for page in spread.pages
        )
        blocks.append(f"DOUBLE PAGE {page_list}\n{page_blocks}")
    return "\n\n".join(blocks)


def story_planner_prompt(
    *,
    source_text: str,
    spreads: list[SpreadSource],
    target_page_numbers: list[int] | None = None,
    image_summaries: dict[int, list[str]] | None = None,
    spread_image_summaries: dict[int, str] | None = None,
    raw_images_attached: bool = True,
) -> str:
    """Prompt for a locked page-by-page planning artifact."""

    image_page_numbers = [
        page.page_number
        for spread in spreads
        for page in spread.pages
    ]
    page_numbers = target_page_numbers or image_page_numbers
    image_page_list = ", ".join(
        str(page_number) for page_number in image_page_numbers
    )
    page_list = ", ".join(str(page_number) for page_number in page_numbers)
    target_pages = set(page_numbers)
    context_page_numbers = [
        page_number
        for page_number in image_page_numbers
        if page_number not in target_pages
    ]
    context_instruction = (
        "Les pages de contexte "
        + ", ".join(str(page_number) for page_number in context_page_numbers)
        + " sont fournies uniquement pour comprendre les transitions. Ne crée "
        "aucune entrée de plan pour ces pages."
        if context_page_numbers
        else "Toutes les pages fournies correspondent aux pages à planifier."
    )
    image_attachment_instruction = (
        f"Les images sont jointes individuellement dans cet ordre: {image_page_list}."
        if raw_images_attached
        else (
            "Aucune image brute n'est jointe à cet appel. Les résumés visuels "
            "ci-dessous constituent le contexte visuel validé."
        )
    )
    if image_summaries is None:
        visual_evidence = "(aucun résumé visuel séparé fourni)"
        visual_instruction = (
            "Déduis visible_on_page prudemment à partir des images jointes."
        )
    else:
        visual_evidence = json.dumps(
            {
                str(page_number): image_summaries[page_number]
                for page_number in image_page_numbers
                if page_number in image_summaries
            },
            ensure_ascii=False,
            indent=2,
        )
        visual_instruction = (
            "Les RÉSUMÉS VISUELS VERROUILLÉS ont été produits séparément par "
            "GPT-4o. Pour chaque page cible, copie exactement la liste fournie "
            "dans visible_on_page; ne la complète pas et ne la réinterprète pas."
        )
    spread_visual_evidence = (
        json.dumps(
            [
                {
                    "spread_index": spread.index,
                    "page_numbers": list(spread.page_numbers),
                    "main_action": spread_image_summaries[spread.index],
                }
                for spread in spreads
                if spread.index in spread_image_summaries
            ],
            ensure_ascii=False,
            indent=2,
        )
        if spread_image_summaries is not None
        else "(aucun résumé global de double page fourni)"
    )
    return f"""
Tu es éditrice ou éditeur narratif pour albums illustrés jeunesse.

Nous préparons l'adaptation d'un album français conçu en doubles pages vers une
lecture numérique page par page.

Cette étape est uniquement un PLAN. N'écris pas le texte final des pages.

Objectif:
Créer une carte narrative page par page qui verrouille:
- ce qui peut être raconté sur chaque page;
- ce qui ne doit pas encore être révélé;
- quelles formulations originales doivent être conservées si possible;
- quels éléments du texte source doivent être déplacés ailleurs parce qu'ils ne
  sont pas visibles ou pas encore compréhensibles sur cette page;
- comment la page doit préparer la suivante.

Principes:
- Respecter strictement l'ordre page par page.
- Une conséquence ne doit jamais apparaître avant sa cause.
- Une solution ne doit jamais être annoncée avant la page où elle devient visible.
- Ne confonds jamais le texte source avec l'image: si un détail est écrit dans
  le texte source mais n'est pas visible sur l'image de la page, ne le mets pas
  dans "visible_on_page".
- Si le texte source d'une page décrit plutôt l'image suivante ou une page plus
  tardive, place ce détail dans "content_to_move_later" et interdis-le sur la
  page actuelle.
- Les pages sans texte original peuvent recevoir une courte amorce, réaction,
  transition ou phrase de suspense seulement si nécessaire.
- Ne pas inventer d'événement majeur.
- Préserver autant que possible les formulations originales.
- Rester sobre: ce plan doit guider une deuxième étape d'écriture, pas raconter
  l'histoire à sa place.

Texte original complet:
{source_text}

Pages physiques et textes sources disponibles:
{_page_labeled_source_blocks(spreads)}

{image_attachment_instruction}
{context_instruction}

RÉSUMÉS VISUELS VERROUILLÉS, indexés par page physique:
{visual_evidence}
{visual_instruction}

ACTIONS PRINCIPALES VERROUILLÉES, par double page:
{spread_visual_evidence}

Crée le plan uniquement pour les pages cibles suivantes: {page_list}.

Retourne uniquement un objet JSON valide:
{{
  "story_arc": {{
    "global_strategy": "Logique générale d'adaptation en 2 ou 3 phrases",
    "major_turns": ["Grandes étapes de l'histoire, dans l'ordre"],
    "tone_rules": ["Règles de ton utiles pour l'écriture finale"]
  }},
  "pages": [
    {{
      "page_number": {page_numbers[0] if page_numbers else 0},
      "visible_on_page": ["Éléments réellement visibles sur l'image de cette page"],
      "source_text_on_page": ["Éléments présents dans le texte source assigné à cette page"],
      "content_to_move_later": ["Éléments du texte source à déplacer car pas encore visibles ou pas encore lisibles ici"],
      "allowed_content": ["Ce que le texte final peut dire ici"],
      "forbidden_content": ["Ce que le texte final ne doit pas encore dire ici"],
      "source_to_preserve": ["Mots, fragments ou dialogues originaux à garder si possible"],
      "adaptation_instruction": "Instruction courte pour l'écriture finale de cette page",
      "handoff_state_after_page": "État narratif à préserver après cette page"
    }}
  ]
}}
La liste pages doit contenir exactement une entrée pour chaque page cible, dans
cet ordre: {page_list}.
""".strip()


def image_summary_prompt(spread: SpreadSource) -> str:
    """Prompt GPT-4o for concise, page-specific visual evidence."""

    page_list = ", ".join(str(page) for page in spread.page_numbers)
    example_pages = ",\n".join(
        "    {\n"
        f'      "page_number": {page_number},\n'
        '      "visible_on_page": ["Une phrase décrivant une action ou une scène réellement visible."]\n'
        "    }"
        for page_number in spread.page_numbers
    )
    return f"""
You are creating visual evidence for a children's-book preprocessing workflow.

Context about the book:
- This is a Barbapapa picture book for children aged 5-8.
- The image is one double-page spread from the book.
- The goal is to understand the visible action so a translator can better interpret each physical page.

Printed source text associated with this spread:
{_page_source_blocks(spread)}

Instructions:
- Describe only the action and scene visible in the image.
- Use the printed source text to understand the story context and ambiguous actions, but do not claim that a textual detail is visible unless the image supports it.
- First summarize the main action or narrative beat across the entire spread in 2 to 3 sentences.
- Treat physical pages {page_list} separately, from left to right.
- Then describe the main action visible on each page in 1 to 3 concise sentences, with one sentence per visible_on_page list item.
- Mention the main characters only if they are visually recognizable.
- Use the character context to identify a character, but do not infer an action that is not visible.
- If something is unclear, say so briefly instead of inventing details.
- Do not mention page layout, OCR, prompt design, source text, or translation strategy.

Book and character context:
- The Barbapapas are shape-shifting beings who can transform into many forms to solve tricky situations.
- Together they are a family.
- Barbapapa is the father: a large light-pink character.
- Barbamama is the mother: a black character with a curvier silhouette and red flowers around her head.
- They have seven children. The boys resemble their father more, and the girls resemble their mother more.
- Barbotine is orange, wears glasses, and loves reading.
- Barbidur is red and athletic.
- Barbabelle is purple and fashion-conscious.
- Barbidou is yellow and loves animals.
- Barbibul is blue, curious, and clever.
- Barbouille is black, hairy, and paints.
- Barbalala is green and loves singing.
- Claudine and François are regular humans who join the family on some adventures.
- Lolita is their dog.

Return only this valid JSON shape, with exactly one entry for each physical page:
{{
  "spread_summary": "Two or three sentences describing the main action across the complete spread.",
  "pages": [
{example_pages}
  ]
}}
""".strip()


def story_chunk_prompt(
    *,
    source_before: str,
    current_source: str,
    source_after: str,
    previous_final_pages: str,
    page_numbers: list[int],
    story_plan: dict[str, Any] | None = None,
) -> str:
    """Prompt for one story-aware chunk adaptation over page images."""

    page_list = ", ".join(str(page_number) for page_number in page_numbers)
    if story_plan is None:
        plan_section = "(aucun plan verrouillé fourni)"
        locked_plan_instruction = (
            "Déduire prudemment la progression locale à partir du texte et des images."
        )
    else:
        plan_section = json.dumps(story_plan, ensure_ascii=False, indent=2)
        locked_plan_instruction = (
            "Suivre le PLAN PAGE PAR PAGE comme une contrainte verrouillée: ne pas "
            "réinterpréter la progression, les interdits ou les éléments à préserver."
        )
    return f"""
Tu es éditrice ou éditeur narratif pour albums illustrés jeunesse. Tu adaptes
une histoire française conçue en doubles pages vers un livre numérique où
chaque page est lue seule.

Tu travailles sur un extrait du livre. Le texte de contexte disponible est
divisé ci-dessous afin de préserver la continuité locale. Lorsque le plan est
fourni, son arc global sert de référence pour la progression de l'histoire.

TEXTE DE RÉFÉRENCE

Voici le contexte original disponible. Les séparateurs délimitent la portion à
adapter. Les pages physiques sont indiquées explicitement pour éviter de
déplacer par erreur un événement sur la mauvaise image. Le texte avant et le
texte après servent uniquement de contexte local.

{source_before or "(début du livre)"}

--- DÉBUT DE LA PORTION À ADAPTER ---

{current_source or "(aucun texte original dans cette portion)"}

--- FIN DE LA PORTION À ADAPTER ---

{source_after or "(fin du livre)"}

CONTINUITÉ DE LA VERSION ADAPTÉE

Voici le texte final déjà validé des pages précédentes. Ne le réécris pas.
Utilise-le uniquement pour assurer la continuité du ton, des personnages et de
l'action:

{previous_final_pages or "(aucune page précédente)"}

PLAN PAGE PAR PAGE

Voici le plan éditorial verrouillé pour cette portion. Il ne contient pas le
texte final. La clé "pages" contient les pages à écrire. Les éventuelles clés
"context_pages_before" et "context_pages_after" servent uniquement à comprendre
les passages de relais: ne les réécris pas et ne déplace pas leur contenu dans
la portion actuelle.

{plan_section}

IMAGES

Les images sans texte jointes correspondent individuellement, dans cet ordre,
aux pages physiques: {page_list}.

OBJECTIF

Adapte uniquement la portion délimitée en un vrai texte d'album où chaque page
possède son propre moment narratif. Pense d'abord à la progression de l'extrait: situation
au début, actions visibles dans l'ordre, réactions, conséquences, puis
transition vers la suite.

CONTRAINTES

- Produire exactement une entrée pour chaque page demandée.
- {locked_plan_instruction}
- Respecter strictement l'ordre chronologique des images et de l'histoire.
- Conserver les personnages, les faits et les relations de cause à effet.
- Ne jamais placer une réaction avant l'action qui la provoque.
- Ne jamais placer une conclusion avant l'action qu'elle conclut.
- Ne jamais anticiper un événement situé après la fin de la portion à adapter.
- Si le texte source assigné à une page décrit un détail qui n'est pas visible
  sur cette page, ne prétends pas qu'il est visible: écris une transition courte
  ou déplace ce détail vers la page où l'image le soutient.
- Chaque page doit avoir un rôle clair: amorce, action, réaction, transition ou
  conclusion.
- Chaque page doit être compréhensible avec son image.
- Ne pas répéter la même information sur deux pages voisines.
- Ajouter seulement ce qui est nécessaire pour rendre la lecture fluide.
- Ne pas inventer de nouvel événement.
- Conserver autant que possible les formulations et dialogues originaux.
- Garder un français simple, chaleureux et agréable à lire à voix haute.
- Écrire généralement une ou deux phrases courtes par page.
- Les pages citées dans CONTINUITÉ sont LOCKED: ne pas les réécrire.
- Produire uniquement les nouvelles pages demandées.

Retourne uniquement un objet JSON valide:
{{
  "chunk_summary": "Progression narrative de cet extrait",
  "continuity_check": "Lien avec les pages précédentes",
  "chronology_check": "Vérification de l'ordre action/réaction/conséquence",
  "transition_to_next_chunk": "État narratif à préserver pour le prochain extrait",
  "pages": [
    {{
      "page_number": {page_numbers[0] if page_numbers else 0},
      "text": "Texte final de cette page",
      "narrative_role": "setup|action|reaction|conclusion|transition",
      "source_anchor": "Passage ou événement original utilisé",
      "visual_grounding": "Élément visible qui soutient ce texte"
    }}
  ]
}}
La liste pages doit contenir exactement une entrée pour chaque page physique,
dans cet ordre: {page_list}.
""".strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a model JSON object, tolerating a surrounding Markdown fence."""

    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected the model response to be a JSON object.")
    return payload


def validate_page_result(
    payload: dict[str, Any], expected_page_numbers: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Validate and normalize the final page-level output contract."""

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("The model response must contain a pages list.")
    actual_numbers = [page.get("page_number") for page in pages if isinstance(page, dict)]
    if actual_numbers != list(expected_page_numbers):
        raise ValueError(
            "The model response page numbers must exactly match "
            f"{list(expected_page_numbers)}; received {actual_numbers}."
        )
    for page in pages:
        text = page.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Page {page.get('page_number')} has no final text.")
        page["text"] = text.strip()
    return pages


def validate_plan(
    payload: dict[str, Any], expected_page_numbers: tuple[int, ...]
) -> dict[str, Any]:
    """Reject a planning response that does not cover the exact physical pages."""

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("The editorial plan must contain a pages list.")
    actual_numbers = [page.get("page_number") for page in pages if isinstance(page, dict)]
    if actual_numbers != list(expected_page_numbers):
        raise ValueError(
            "The editorial plan page numbers must exactly match "
            f"{list(expected_page_numbers)}; received {actual_numbers}."
        )
    return payload


def validate_story_plan(
    payload: dict[str, Any], expected_page_numbers: tuple[int, ...]
) -> dict[str, Any]:
    """Validate and lightly normalize the story-level page plan."""

    story_arc = payload.get("story_arc")
    if not isinstance(story_arc, dict):
        raise ValueError("The story plan must contain a story_arc object.")

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("The story plan must contain a pages list.")
    actual_numbers = [page.get("page_number") for page in pages if isinstance(page, dict)]
    if actual_numbers != list(expected_page_numbers):
        raise ValueError(
            "The story plan page numbers must exactly match "
            f"{list(expected_page_numbers)}; received {actual_numbers}."
        )

    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Every story plan page must be an object.")
        for key in (
            "adaptation_instruction",
            "handoff_state_after_page",
        ):
            value = page.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Story plan page {page.get('page_number')} is missing {key}."
                )
            page[key] = value.strip()
        for key in (
            "visible_on_page",
            "source_text_on_page",
            "content_to_move_later",
            "allowed_content",
            "forbidden_content",
            "source_to_preserve",
        ):
            value = page.get(key)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(
                    f"Story plan page {page.get('page_number')} must contain "
                    f"a string list for {key}."
                )
            page[key] = [item.strip() for item in value if item.strip()]
    return payload


def validate_image_summaries(
    payload: dict[str, Any], expected_page_numbers: tuple[int, ...]
) -> dict[int, list[str]]:
    """Validate reusable visual evidence and return it indexed by page."""

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Image summaries must contain a pages list.")
    actual_numbers = [page.get("page_number") for page in pages if isinstance(page, dict)]
    if actual_numbers != list(expected_page_numbers):
        raise ValueError(
            "Image summary page numbers must exactly match "
            f"{list(expected_page_numbers)}; received {actual_numbers}."
        )

    summaries: dict[int, list[str]] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Every image summary page must be an object.")
        visible = page.get("visible_on_page")
        if not isinstance(visible, list) or not all(
            isinstance(item, str) for item in visible
        ):
            raise ValueError(
                f"Image summary page {page.get('page_number')} must contain a "
                "visible_on_page string list."
            )
        normalized = [item.strip() for item in visible if item.strip()]
        if not normalized or len(normalized) > 3:
            raise ValueError(
                f"Image summary page {page.get('page_number')} must contain "
                "between one and three observations."
            )
        summaries[int(page["page_number"])] = normalized
    return summaries


def validate_spread_summary(payload: dict[str, Any]) -> str:
    """Validate the complete-spread action summary returned by the vision model."""

    summary = payload.get("spread_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Image summaries must contain a non-empty spread_summary.")
    return summary.strip()


def spread_summaries_from_artifact(
    payload: dict[str, Any], spreads: list[SpreadSource]
) -> dict[int, str]:
    """Load and validate complete-spread summaries from a reusable artifact."""

    records = payload.get("spreads")
    if not isinstance(records, list):
        raise ValueError("Image summaries must contain a spreads list.")
    expected = [(spread.index, list(spread.page_numbers)) for spread in spreads]
    actual = [
        (record.get("spread_index"), record.get("page_numbers"))
        for record in records
        if isinstance(record, dict)
    ]
    if actual != expected:
        raise ValueError(
            f"Image summary spreads must exactly match {expected}; received {actual}."
        )
    summaries: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every image summary spread must be an object.")
        summaries[int(record["spread_index"])] = validate_spread_summary(
            {"spread_summary": record.get("spread_summary")}
        )
    return summaries


def inject_image_summaries(
    story_plan: dict[str, Any], image_summaries: dict[int, list[str]]
) -> dict[str, Any]:
    """Make reusable image evidence authoritative in a planner response."""

    pages = story_plan.get("pages")
    if not isinstance(pages, list):
        return story_plan
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, int) and page_number in image_summaries:
            page["visible_on_page"] = list(image_summaries[page_number])
    return story_plan


def adapted_pages_from_text(
    adapted_text: str,
    body_page_numbers: list[int],
) -> list[dict[str, Any]]:
    """Map an existing single-page text file to consecutive physical body pages."""

    paragraphs = split_source_paragraphs(adapted_text)
    if len(paragraphs) > len(body_page_numbers):
        raise ValueError(
            f"Found {len(paragraphs)} adapted paragraphs but only "
            f"{len(body_page_numbers)} body pages are available."
        )
    return [
        {
            "page_number": page_number,
            "text": text,
            "intervention": "unknown",
            "change_note": "Loaded from an existing preprocessing text file.",
        }
        for page_number, text in zip(body_page_numbers, paragraphs, strict=False)
    ]


def _context_text(spread: SpreadSource) -> str:
    return "\n\n".join(page.source_text for page in spread.pages if page.source_text)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_progress(
    *,
    output_path: Path,
    report_path: Path,
    adapted_pages: list[dict[str, Any]],
    spread_artifacts: list[dict[str, Any]],
    source_path: Path,
    pdf_path: Path,
    provider: str,
    model: str,
    temperature: float,
) -> None:
    """Persist completed spreads so a long experimental run stays inspectable."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n".join(page["text"] for page in adapted_pages) + "\n",
        encoding="utf-8",
    )
    _save_json(
        report_path,
        {
            "source_text": str(source_path),
            "source_pdf": str(pdf_path),
            "model": f"{provider}:{model}",
            "temperature": temperature,
            "pages": adapted_pages,
            "spreads": spread_artifacts,
        },
    )


def _save_story_progress(
    *,
    output_path: Path,
    report_path: Path,
    adapted_pages: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    source_path: Path,
    pdf_path: Path,
    provider: str,
    model: str,
    temperature: float,
    spreads_per_chunk: int | None,
    story_plan: dict[str, Any] | None = None,
    image_summary_path: Path | None = None,
) -> None:
    """Persist completed story chunks so a partial run remains inspectable."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n\n".join(page["text"] for page in adapted_pages) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "mode": "story",
        "source_text": str(source_path),
        "source_pdf": str(pdf_path),
        "model": f"{provider}:{model}",
        "temperature": temperature,
        "spreads_per_chunk": spreads_per_chunk,
        "pages": adapted_pages,
        "chunks": chunk_results,
    }
    if story_plan is not None:
        report["story_plan"] = story_plan
    if image_summary_path is not None:
        report["image_summaries"] = str(image_summary_path)
    if spreads_per_chunk is None and len(chunk_results) == 1:
        report["page_numbers"] = chunk_results[0]["page_numbers"]
        report["story_result"] = chunk_results[0]["result"]
    _save_json(report_path, report)


def _data_url_from_image(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _save_page_image(
    *,
    pdf_path: Path,
    page_number: int,
    dpi: int,
    image_dir: Path | None,
    jpeg_quality: int | None = None,
) -> str:
    image_bytes = render_spread_image_bytes(
        pdf_path=pdf_path,
        spread_pages=(page_number,),
        dpi=dpi,
    )
    suffix = ".png"
    mime_type = "image/png"
    if jpeg_quality is not None:
        output = io.BytesIO()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(
                output,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
        image_bytes = output.getvalue()
        suffix = ".jpg"
        mime_type = "image/jpeg"
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"page_{page_number:02d}{suffix}").write_bytes(image_bytes)
    return _data_url_from_image(image_bytes, mime_type)


def _spread_image_payload(
    *,
    pdf_path: Path,
    spread: SpreadSource,
    dpi: int,
    jpeg_quality: int,
    image_dir: Path | None,
) -> tuple[str, str]:
    """Render a spread once and return its JPEG data URL and content digest."""

    png_bytes = render_spread_image_bytes(
        pdf_path=pdf_path,
        spread_pages=spread.page_numbers,
        dpi=dpi,
    )
    output = io.BytesIO()
    with Image.open(io.BytesIO(png_bytes)) as image:
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )
    image_bytes = output.getvalue()
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        page_slug = "-".join(str(page) for page in spread.page_numbers)
        (image_dir / f"spread_{spread.index + 1:02d}_pages_{page_slug}.jpg").write_bytes(
            image_bytes
        )
    return (
        _data_url_from_image(image_bytes, "image/jpeg"),
        hashlib.sha256(image_bytes).hexdigest(),
    )


def _blank_page_text_rect(page: fitz.Page) -> fitz.Rect:
    """Return a conservative bottom caption area for pages with no source text."""

    bounds = page.rect
    margin_x = bounds.width * DEFAULT_BLANK_TEXT_MARGIN_RATIO
    height = bounds.height * DEFAULT_BLANK_TEXT_HEIGHT_RATIO
    margin_bottom = bounds.height * 0.06
    return fitz.Rect(
        bounds.x0 + margin_x,
        bounds.y1 - margin_bottom - height,
        bounds.x1 - margin_x,
        bounds.y1 - margin_bottom,
    )


def _fallback_page_text_rect(
    page: fitz.Page,
    preferred_rect: fitz.Rect | None = None,
) -> fitz.Rect:
    """Return a larger caption panel when the preferred text area is too small."""

    bounds = page.rect
    margin_x = bounds.width * DEFAULT_BLANK_TEXT_MARGIN_RATIO
    margin_y = bounds.height * 0.06
    height = bounds.height * 0.40
    use_top = preferred_rect is not None and preferred_rect.y1 < bounds.y0 + bounds.height / 2
    if use_top:
        y0 = bounds.y0 + margin_y
        y1 = y0 + height
    else:
        y1 = bounds.y1 - margin_y
        y0 = y1 - height
    return fitz.Rect(bounds.x0 + margin_x, y0, bounds.x1 - margin_x, y1)


def _insert_text_fitting(
    *,
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    font_file: Path | None,
    initial_font_size: float,
    align: int = fitz.TEXT_ALIGN_LEFT,
) -> None:
    """Insert text, reducing the font size until it fits the target rectangle."""

    font_name = register_font(page, font_file)
    font_size = float(initial_font_size)
    status = insert_translation(
        page,
        rect,
        text,
        font_name,
        font_size,
        align=align,
    )
    while status < 0 and font_size > MIN_FONT_SIZE:
        font_size -= 0.5
        status = insert_translation(
            page,
            rect,
            text,
            font_name,
            font_size,
            align=align,
        )

    if status < 0:
        raise ValueError(
            f"Text does not fit on page {page.number + 1} even at the minimum font size."
        )


def _insert_text_with_fallback(
    *,
    page: fitz.Page,
    primary_rect: fitz.Rect,
    text: str,
    font_file: Path | None,
    initial_font_size: float,
) -> None:
    """Use a larger caption panel if text cannot fit its preferred rectangle."""

    try:
        _insert_text_fitting(
            page=page,
            rect=primary_rect,
            text=text,
            font_file=font_file,
            initial_font_size=initial_font_size,
        )
        return
    except ValueError:
        fallback_rect = _fallback_page_text_rect(page, primary_rect)

    page.draw_rect(
        fallback_rect,
        color=None,
        fill=(1, 1, 1),
        overlay=True,
    )
    _insert_text_fitting(
        page=page,
        rect=fallback_rect,
        text=text,
        font_file=font_file,
        initial_font_size=min(float(initial_font_size), 12.0),
    )


def render_preprocessed_pdf(
    *,
    pdf_path: Path,
    output_path: Path,
    pages: list[dict[str, Any]],
    font_file: Path | None = None,
    padding: float = DEFAULT_PADDING,
    selected_pages_only: bool = True,
) -> None:
    """Render adapted single-page French text back onto source PDF pages."""

    if not pages:
        raise ValueError("No adapted pages are available to render.")

    page_numbers = [int(page["page_number"]) for page in pages]
    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("Adapted pages must not contain duplicate page numbers.")

    with fitz.open(pdf_path) as source_document:
        for page_number in page_numbers:
            if not 1 <= page_number <= source_document.page_count:
                raise ValueError(
                    f"Adapted page {page_number} exceeds document length "
                    f"{source_document.page_count}."
                )

        if selected_pages_only:
            document = fitz.open()
            page_lookup: dict[int, int] = {}
            for output_index, page_number in enumerate(page_numbers):
                document.insert_pdf(
                    source_document,
                    from_page=page_number - 1,
                    to_page=page_number - 1,
                )
                page_lookup[page_number] = output_index
        else:
            document = source_document
            page_lookup = {page_number: page_number - 1 for page_number in page_numbers}

        for adapted_page in pages:
            page_number = int(adapted_page["page_number"])
            page = document.load_page(page_lookup[page_number])
            text = str(adapted_page["text"]).strip()
            try:
                line_rects, union_rect, source_font_size = collect_page_lines(page)
            except ValueError:
                insert_rect = _blank_page_text_rect(page)
                page.draw_rect(
                    insert_rect,
                    color=None,
                    fill=(1, 1, 1),
                    overlay=True,
                )
                _insert_text_with_fallback(
                    page=page,
                    primary_rect=insert_rect,
                    text=text,
                    font_file=font_file,
                    initial_font_size=12.0,
                )
                continue

            add_redactions(page, line_rects, padding=padding)
            insert_rect = fitz.Rect(union_rect)
            insert_rect.x0 -= padding
            insert_rect.y0 -= padding
            insert_rect.x1 += padding
            insert_rect.y1 += padding
            _insert_text_with_fallback(
                page=page,
                primary_rect=insert_rect,
                text=text,
                font_file=font_file,
                initial_font_size=source_font_size,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.save(output_path, garbage=4, deflate=True)
        if selected_pages_only:
            document.close()

    compress_pdf(output_path, output_path, recompress_images=False)


def _story_chunks(
    all_spreads: list[SpreadSource],
    *,
    spreads_per_chunk: int | None,
    max_spreads: int | None,
) -> list[list[SpreadSource]]:
    """Split selected story spreads into sequential model-call chunks."""

    selected_spreads = (
        all_spreads if max_spreads is None else all_spreads[:max_spreads]
    )
    if not selected_spreads:
        return []
    if spreads_per_chunk is None:
        return [selected_spreads]
    if spreads_per_chunk <= 0:
        raise ValueError("story_spreads_per_chunk must be greater than zero.")
    return [
        selected_spreads[start : start + spreads_per_chunk]
        for start in range(0, len(selected_spreads), spreads_per_chunk)
    ]


def _source_section(spreads: list[SpreadSource]) -> str:
    """Return original text from a sequence of spreads in story order."""

    return "\n\n".join(
        page.source_text
        for spread in spreads
        for page in spread.pages
        if page.source_text
    )


def _previous_final_context(adapted_pages: list[dict[str, Any]]) -> str:
    """Format the last two validated page texts for chunk continuity."""

    return "\n\n".join(
        f"PAGE {page['page_number']}\n{page['text']}"
        for page in adapted_pages[-2:]
    )


def _story_plan_for_pages(
    story_plan: dict[str, Any] | None,
    page_numbers: list[int],
    *,
    pages_before: int = STORY_PLAN_PAGES_BEFORE,
    pages_after: int = STORY_PLAN_PAGES_AFTER,
) -> dict[str, Any] | None:
    """Return current plan entries plus context-only entries at chunk boundaries."""

    if story_plan is None:
        return None
    wanted = set(page_numbers)
    all_pages = story_plan.get("pages", [])
    selected_indexes = [
        index
        for index, page in enumerate(all_pages)
        if page.get("page_number") in wanted
    ]
    if not selected_indexes:
        return {
            "story_arc": story_plan.get("story_arc", {}),
            "pages": [],
            "context_pages_before": [],
            "context_pages_after": [],
        }

    first_index = min(selected_indexes)
    last_index = max(selected_indexes)
    return {
        "story_arc": story_plan.get("story_arc", {}),
        "pages": [
            page
            for page in all_pages
            if page.get("page_number") in wanted
        ],
        "context_pages_before": all_pages[
            max(0, first_index - pages_before) : first_index
        ],
        "context_pages_after": all_pages[
            last_index + 1 : last_index + 1 + pages_after
        ],
    }


def _story_source_context(
    all_spreads: list[SpreadSource],
    *,
    chunk_start: int,
    chunk_end: int,
    bounded: bool,
) -> tuple[list[SpreadSource], list[SpreadSource]]:
    """Select source context around a chunk, bounding it when a plan is available."""

    if not bounded:
        return all_spreads[:chunk_start], all_spreads[chunk_end:]
    return (
        all_spreads[max(0, chunk_start - STORY_CONTEXT_SPREADS_BEFORE) : chunk_start],
        all_spreads[chunk_end : chunk_end + STORY_CONTEXT_SPREADS_AFTER],
    )


def validate_source_provenance(
    source_path: Path,
    pdf_path: Path,
    *,
    allow_preprocessed_source: bool,
) -> None:
    """Guard against accidentally adapting generated single-page artifacts again."""

    suspicious = [
        path
        for path in (source_path, pdf_path)
        if any(marker in path.stem.lower() for marker in PREPROCESSED_SOURCE_MARKERS)
    ]
    if suspicious and not allow_preprocessed_source:
        paths = ", ".join(str(path) for path in suspicious)
        raise ValueError(
            "The preprocessing input appears to be an already generated single-page "
            f"artifact: {paths}. Use the original double-page source, or pass "
            "--allow-preprocessed-source to proceed intentionally."
        )


def _story_planner_windows(
    spreads: list[SpreadSource],
    *,
    spreads_per_call: int,
    context_spreads: int = STORY_PLANNER_CONTEXT_SPREADS,
) -> list[tuple[list[SpreadSource], list[SpreadSource]]]:
    """Return target windows paired with overlapping visual/source context."""

    if spreads_per_call <= 0:
        raise ValueError("story_planner_spreads_per_call must be greater than zero.")
    windows: list[tuple[list[SpreadSource], list[SpreadSource]]] = []
    for start in range(0, len(spreads), spreads_per_call):
        end = min(len(spreads), start + spreads_per_call)
        target_spreads = spreads[start:end]
        context_start = max(0, start - context_spreads)
        context_end = min(len(spreads), end + context_spreads)
        windows.append((target_spreads, spreads[context_start:context_end]))
    return windows


def generate_image_summaries(
    *,
    args: argparse.Namespace,
    config: TranslationWorkflowConfig,
    pdf_path: Path,
    spreads: list[SpreadSource],
    cache: ResponseCache,
    artifact_dir: Path,
) -> tuple[dict[int, list[str]], dict[int, str], dict[str, Any]]:
    """Run the reusable GPT-4o image-summary stage over complete spreads."""

    provider, model = config.parse_model_ref(args.image_summary_model)
    expected_page_numbers = tuple(
        page.page_number for spread in spreads for page in spread.pages
    )
    image_dir = artifact_dir / "image_summary_spreads" if args.save_images else None
    pages: list[dict[str, Any]] = []
    spread_records: list[dict[str, Any]] = []
    spread_summaries: dict[int, str] = {}

    for spread in spreads:
        image_data_url, image_sha256 = _spread_image_payload(
            pdf_path=pdf_path,
            spread=spread,
            dpi=args.dpi,
            jpeg_quality=config.multimodal_jpeg_quality,
            image_dir=image_dir,
        )
        prompt = image_summary_prompt(spread)
        raw = ask_model_with_recovery(
            provider=provider,
            model=model,
            temperature=args.image_summary_temperature,
            prompt=prompt,
            image_data_url=image_data_url,
            config=config,
            cache=cache,
            label=f"image summary spread {spread.index + 1}",
        )
        response_payload = parse_json_object(raw)
        page_summaries = validate_image_summaries(
            response_payload, spread.page_numbers
        )
        spread_summary = validate_spread_summary(response_payload)
        spread_summaries[spread.index] = spread_summary
        pages.extend(
            {
                "page_number": page_number,
                "visible_on_page": page_summaries[page_number],
            }
            for page_number in spread.page_numbers
        )
        spread_records.append(
            {
                "spread_index": spread.index,
                "page_numbers": list(spread.page_numbers),
                "image_sha256": image_sha256,
                "spread_summary": spread_summary,
            }
        )
        print(
            f"Image summary {spread.index + 1}/{len(spreads)}: "
            f"pages {spread.page_numbers}"
        )

    artifact = {
        "schema_version": IMAGE_SUMMARY_SCHEMA_VERSION,
        "prompt_version": IMAGE_SUMMARY_PROMPT_VERSION,
        "source_pdf": str(pdf_path),
        "model": f"{provider}:{model}",
        "temperature": args.image_summary_temperature,
        "render_dpi": args.dpi,
        "jpeg_quality": config.multimodal_jpeg_quality,
        "spreads": spread_records,
        "pages": pages,
    }
    return (
        validate_image_summaries(artifact, expected_page_numbers),
        spread_summaries,
        artifact,
    )


def ensure_workflow_image_summaries(
    config: TranslationWorkflowConfig, output_path: Path
) -> Path:
    """Generate the reusable visual-evidence artifact required by text workflows."""

    if output_path.is_file():
        return output_path
    if config.source_pdf_path is None:
        raise ValueError("Automatic image summaries require SOURCE_PDF_PATH.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = config.load_source_text()
    body_pages = collect_body_pages(
        config.source_pdf_path, config.pdf_skip_first, config.pdf_skip_last
    )
    text_pages = collect_non_empty_body_pages(
        config.source_pdf_path, config.pdf_skip_first, config.pdf_skip_last
    )
    spreads = build_spread_sources(source_text, body_pages, text_pages)
    args = SimpleNamespace(
        image_summary_model=config.image_summary_model,
        image_summary_temperature=config.image_summary_temperature,
        dpi=config.multimodal_image_dpi,
        save_images=config.multimodal_save_debug_images,
    )
    _, _, artifact = generate_image_summaries(
        args=args,
        config=config,
        pdf_path=config.source_pdf_path,
        spreads=spreads,
        cache=ResponseCache(config.translation_cache_dir),
        artifact_dir=output_path.parent,
    )
    _save_json(output_path, artifact)
    print(f"Saved automatically generated image summaries: {output_path}")
    return output_path


def generate_story_plan(
    *,
    args: argparse.Namespace,
    config: TranslationWorkflowConfig,
    source_text: str,
    pdf_path: Path,
    spreads: list[SpreadSource],
    provider: str,
    model: str,
    cache: ResponseCache,
    artifact_dir: Path,
    image_summaries: dict[int, list[str]] | None = None,
    spread_image_summaries: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Generate and merge bounded page-plan windows for the selected story."""

    expected_page_numbers = [
        page.page_number
        for spread in spreads
        for page in spread.pages
    ]
    image_dir = artifact_dir / "planner_pages" if args.save_images else None
    windows = _story_planner_windows(
        spreads,
        spreads_per_call=args.story_planner_spreads_per_call,
    )
    merged_pages: list[dict[str, Any]] = []
    story_arc: dict[str, Any] | None = None
    window_dir = artifact_dir / "planner_windows"

    for window_index, (target_spreads, context_spreads) in enumerate(windows, start=1):
        target_page_numbers = [
            page.page_number
            for spread in target_spreads
            for page in spread.pages
        ]
        image_page_numbers = [
            page.page_number
            for spread in context_spreads
            for page in spread.pages
        ]
        image_data_urls = (
            [
                _save_page_image(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    dpi=args.dpi,
                    image_dir=image_dir,
                    jpeg_quality=config.multimodal_jpeg_quality,
                )
                for page_number in image_page_numbers
            ]
            if image_summaries is None
            else []
        )
        raw = ask_model_with_recovery(
            provider=provider,
            model=model,
            temperature=args.temperature,
            prompt=story_planner_prompt(
                source_text=source_text,
                spreads=context_spreads,
                target_page_numbers=target_page_numbers,
                image_summaries=image_summaries,
                spread_image_summaries=spread_image_summaries,
                raw_images_attached=image_summaries is None,
            ),
            image_data_urls=image_data_urls,
            config=config,
            cache=cache,
            label=f"story preprocessing planner window {window_index}",
        )
        if image_summaries is not None:
            parsed_plan = inject_image_summaries(
                parse_json_object(raw), image_summaries
            )
        else:
            parsed_plan = parse_json_object(raw)
        window_plan = validate_story_plan(
            parsed_plan,
            tuple(target_page_numbers),
        )
        if story_arc is None:
            story_arc = window_plan["story_arc"]
        merged_pages.extend(window_plan["pages"])
        _save_json(
            window_dir / f"window_{window_index:02d}.json",
            {
                "target_page_numbers": target_page_numbers,
                "context_page_numbers": image_page_numbers,
                "plan": window_plan,
            },
        )
        print(
            f"Planned story window {window_index}/{len(windows)}: "
            f"pages {tuple(target_page_numbers)}"
        )

    return validate_story_plan(
        {"story_arc": story_arc or {}, "pages": merged_pages},
        tuple(expected_page_numbers),
    )


def run_story_preprocessing(
    *,
    args: argparse.Namespace,
    config: TranslationWorkflowConfig,
    source_path: Path,
    pdf_path: Path,
    all_spreads: list[SpreadSource],
    provider: str,
    model: str,
    cache: ResponseCache,
    output_path: Path,
    report_path: Path,
    artifact_dir: Path,
    story_plan: dict[str, Any] | None = None,
    image_summary_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run sequential story-aware calls over bounded groups of spread images."""

    chunks = _story_chunks(
        all_spreads,
        spreads_per_chunk=args.story_spreads_per_chunk,
        max_spreads=args.max_spreads,
    )
    if not chunks:
        raise ValueError("No body pages were selected for story preprocessing.")

    adapted_pages: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []

    spread_offset = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_end = spread_offset + len(chunk)
        page_numbers = [
            page.page_number
            for spread in chunk
            for page in spread.pages
        ]
        if args.story_spreads_per_chunk is None:
            image_dir = artifact_dir / "pages" if args.save_images else None
            image_data_urls = [
                _save_page_image(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    dpi=args.dpi,
                    image_dir=image_dir,
                )
                for page_number in page_numbers
            ]
            prompt = story_prompt(
                source_text=_page_labeled_source_blocks(all_spreads),
                page_numbers=page_numbers,
                story_plan=_story_plan_for_pages(story_plan, page_numbers),
            )
            current_source = _page_labeled_source_blocks(chunk)
            image_format = "single_pages"
        else:
            image_dir = artifact_dir / "pages" if args.save_images else None
            image_data_urls = [
                _save_page_image(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    dpi=args.dpi,
                    image_dir=image_dir,
                )
                for page_number in page_numbers
            ]
            current_source = _page_labeled_source_blocks(chunk)
            context_before, context_after = _story_source_context(
                all_spreads,
                chunk_start=spread_offset,
                chunk_end=chunk_end,
                bounded=story_plan is not None,
            )
            prompt = story_chunk_prompt(
                source_before=_page_labeled_source_blocks(context_before),
                current_source=current_source,
                source_after=_page_labeled_source_blocks(context_after),
                previous_final_pages=_previous_final_context(adapted_pages),
                page_numbers=page_numbers,
                story_plan=_story_plan_for_pages(story_plan, page_numbers),
            )
            image_format = "single_pages"

        raw = ask_model_with_recovery(
            provider=provider,
            model=model,
            temperature=args.temperature,
            prompt=prompt,
            image_data_urls=image_data_urls,
            config=config,
            cache=cache,
            label=f"story preprocessing chunk {chunk_index + 1}",
        )
        result = parse_json_object(raw)
        pages = validate_page_result(result, tuple(page_numbers))
        adapted_pages.extend(pages)
        chunk_results.append(
            {
                "chunk_index": chunk_index + 1,
                "spread_indexes": [spread.index for spread in chunk],
                "page_numbers": page_numbers,
                "image_format": image_format,
                "source_before": (
                    _page_labeled_source_blocks(context_before)
                    if args.story_spreads_per_chunk is not None
                    else ""
                ),
                "current_source": current_source,
                "source_after": (
                    _page_labeled_source_blocks(context_after)
                    if args.story_spreads_per_chunk is not None
                    else ""
                ),
                "previous_final_pages": _previous_final_context(
                    adapted_pages[:-len(pages)]
                ),
                "result": result,
            }
        )
        _save_story_progress(
            output_path=output_path,
            report_path=report_path,
            adapted_pages=adapted_pages,
            chunk_results=chunk_results,
            source_path=source_path,
            pdf_path=pdf_path,
            provider=provider,
            model=model,
            temperature=args.temperature,
            spreads_per_chunk=args.story_spreads_per_chunk,
            story_plan=story_plan,
            image_summary_path=image_summary_path,
        )
        print(
            f"Adapted story chunk {chunk_index + 1}/{len(chunks)}: "
            f"pages {tuple(page_numbers)}"
        )
        spread_offset = chunk_end

    return adapted_pages


def run_preprocessing(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run the standalone two-pass preprocessing experiment."""

    config = TranslationWorkflowConfig.from_env()
    source_path = args.source or config.source_text_path
    pdf_path = args.pdf or config.source_pdf_path
    if pdf_path is None:
        raise ValueError("A source PDF is required.")

    source_text = source_path.read_text(encoding="utf-8").strip()
    body_pages = collect_body_pages(pdf_path, args.skip_first, args.skip_last)

    if args.render_from_text is not None:
        adapted_pages = adapted_pages_from_text(
            args.render_from_text.read_text(encoding="utf-8"),
            body_pages,
        )
        pdf_output_path = args.pdf_output or args.render_from_text.with_suffix(".pdf")
        render_preprocessed_pdf(
            pdf_path=pdf_path,
            output_path=pdf_output_path,
            pages=adapted_pages,
            font_file=args.font_file,
            padding=args.padding,
            selected_pages_only=not args.render_full_pdf,
        )
        report_path = pdf_output_path.with_suffix(".render_report.json")
        _save_json(
            report_path,
            {
                "source_pdf": str(pdf_path),
                "adapted_text": str(args.render_from_text),
                "rendered_pdf": str(pdf_output_path),
                "selected_pages_only": not args.render_full_pdf,
                "pages": adapted_pages,
            },
        )
        print(f"Saved rendered PDF: {pdf_output_path}")
        return args.render_from_text, report_path

    validate_source_provenance(
        source_path,
        pdf_path,
        allow_preprocessed_source=args.allow_preprocessed_source,
    )

    text_pages = collect_non_empty_body_pages(pdf_path, args.skip_first, args.skip_last)
    all_spreads = build_spread_sources(source_text, body_pages, text_pages)
    spreads = all_spreads
    if args.max_spreads is not None:
        spreads = all_spreads[: args.max_spreads]

    provider, model = config.parse_model_ref(args.model)
    if args.no_cache:
        config.enable_cache = False
    cache = ResponseCache(config.translation_cache_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.artifacts_dir or (
        config.translation_output_dir / "_preprocessing" / run_id
    )
    output_path = args.output or artifact_dir / f"{source_path.stem}.single_page.txt"
    report_path = artifact_dir / "preprocessing_report.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "story":
        story_plan: dict[str, Any] | None = None
        image_summary_path: Path | None = None
        if args.story_planner in {"on", "only"}:
            planner_spreads = (
                all_spreads
                if args.max_spreads is None
                else all_spreads[: args.max_spreads]
            )
            expected_summary_pages = tuple(
                page.page_number
                for spread in planner_spreads
                for page in spread.pages
            )
            if args.image_summaries is not None:
                image_summary_artifact = json.loads(
                    args.image_summaries.read_text(encoding="utf-8")
                )
                if not isinstance(image_summary_artifact, dict):
                    raise ValueError("The image summaries artifact must be a JSON object.")
                if image_summary_artifact.get("schema_version") != IMAGE_SUMMARY_SCHEMA_VERSION:
                    raise ValueError(
                        "Unsupported image summaries schema_version; expected "
                        f"{IMAGE_SUMMARY_SCHEMA_VERSION}."
                    )
                image_summaries = validate_image_summaries(
                    image_summary_artifact, expected_summary_pages
                )
                spread_image_summaries = spread_summaries_from_artifact(
                    image_summary_artifact, planner_spreads
                )
                image_summary_path = args.image_summaries
                print(f"Loaded image summaries: {image_summary_path}")
            else:
                (
                    image_summaries,
                    spread_image_summaries,
                    image_summary_artifact,
                ) = generate_image_summaries(
                    args=args,
                    config=config,
                    pdf_path=pdf_path,
                    spreads=planner_spreads,
                    cache=cache,
                    artifact_dir=artifact_dir,
                )
                image_summary_path = args.image_summary_output or (
                    artifact_dir / f"{source_path.stem}.image_summaries.json"
                )
                _save_json(image_summary_path, image_summary_artifact)
                print(f"Saved image summaries: {image_summary_path}")
            story_plan = generate_story_plan(
                args=args,
                config=config,
                source_text=source_text,
                pdf_path=pdf_path,
                spreads=planner_spreads,
                provider=provider,
                model=model,
                cache=cache,
                artifact_dir=artifact_dir,
                image_summaries=image_summaries,
                spread_image_summaries=spread_image_summaries,
            )
            planner_path = args.planner_output or (
                artifact_dir / f"{source_path.stem}.story_plan.json"
            )
            _save_json(planner_path, story_plan)
            print(f"Saved story planner output: {planner_path}")
            if args.story_planner == "only":
                _save_json(
                    report_path,
                    {
                        "mode": "story_planner",
                        "source_text": str(source_path),
                        "source_pdf": str(pdf_path),
                        "model": f"{provider}:{model}",
                        "temperature": args.temperature,
                        "image_summary_model": image_summary_artifact.get("model"),
                        "image_summaries": str(image_summary_path),
                        "planner_output": str(planner_path),
                        "story_plan": story_plan,
                    },
                )
                return planner_path, report_path

        adapted_pages = run_story_preprocessing(
            args=args,
            config=config,
            source_path=source_path,
            pdf_path=pdf_path,
            all_spreads=all_spreads,
            provider=provider,
            model=model,
            cache=cache,
            output_path=output_path,
            report_path=report_path,
            artifact_dir=artifact_dir,
            story_plan=story_plan,
            image_summary_path=image_summary_path,
        )
        if args.render_pdf:
            pdf_output_path = args.pdf_output or output_path.with_suffix(".pdf")
            render_preprocessed_pdf(
                pdf_path=pdf_path,
                output_path=pdf_output_path,
                pages=adapted_pages,
                font_file=args.font_file,
                padding=args.padding,
                selected_pages_only=not args.render_full_pdf,
            )
            print(f"Saved rendered PDF: {pdf_output_path}")
        return output_path, report_path

    adapted_pages: list[dict[str, Any]] = []
    spread_artifacts: list[dict[str, Any]] = []
    for index, spread in enumerate(spreads):
        image_bytes = render_spread_image_bytes(
            pdf_path=pdf_path,
            spread_pages=spread.page_numbers,
            dpi=args.dpi,
        )
        if args.save_images:
            image_dir = artifact_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            page_slug = "-".join(str(page) for page in spread.page_numbers)
            (image_dir / f"spread_{index + 1:02d}_{page_slug}.png").write_bytes(
                image_bytes
            )

        previous_source = _context_text(spreads[index - 1]) if index else ""
        next_source = (
            _context_text(all_spreads[index + 1])
            if index + 1 < len(all_spreads)
            else ""
        )
        previous_adapted = "\n\n".join(page["text"] for page in adapted_pages[-2:])
        image_data_url = (
            "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        )

        plan_raw = ask_model_with_recovery(
            provider=provider,
            model=model,
            temperature=args.temperature,
            prompt=editorial_plan_prompt(
                spread=spread,
                previous_source=previous_source,
                next_source=next_source,
                previous_adapted=previous_adapted,
            ),
            image_data_url=image_data_url,
            config=config,
            cache=cache,
            label=f"preprocessing plan spread {index + 1}",
        )
        plan = validate_plan(parse_json_object(plan_raw), spread.page_numbers)

        result_raw = ask_model_with_recovery(
            provider=provider,
            model=model,
            temperature=args.temperature,
            prompt=rewrite_prompt(
                spread=spread,
                plan=plan,
                previous_adapted=previous_adapted,
            ),
            image_data_url=image_data_url,
            config=config,
            cache=cache,
            label=f"preprocessing rewrite spread {index + 1}",
        )
        result = parse_json_object(result_raw)
        pages = validate_page_result(result, spread.page_numbers)
        adapted_pages.extend(pages)
        spread_artifacts.append(
            {
                "spread_index": index + 1,
                "source": asdict(spread),
                "plan": plan,
                "result": result,
            }
        )
        _save_progress(
            output_path=output_path,
            report_path=report_path,
            adapted_pages=adapted_pages,
            spread_artifacts=spread_artifacts,
            source_path=source_path,
            pdf_path=pdf_path,
            provider=provider,
            model=model,
            temperature=args.temperature,
        )
        print(f"Adapted spread {index + 1}/{len(spreads)}: pages {spread.page_numbers}")

    if not spreads:
        _save_progress(
            output_path=output_path,
            report_path=report_path,
            adapted_pages=adapted_pages,
            spread_artifacts=spread_artifacts,
            source_path=source_path,
            pdf_path=pdf_path,
            provider=provider,
            model=model,
            temperature=args.temperature,
        )
    if args.render_pdf:
        pdf_output_path = args.pdf_output or output_path.with_suffix(".pdf")
        render_preprocessed_pdf(
            pdf_path=pdf_path,
            output_path=pdf_output_path,
            pages=adapted_pages,
            font_file=args.font_file,
            padding=args.padding,
            selected_pages_only=not args.render_full_pdf,
        )
        print(f"Saved rendered PDF: {pdf_output_path}")
    return output_path, report_path


def parse_args() -> argparse.Namespace:
    """Parse standalone preprocessing CLI options."""

    defaults = TranslationWorkflowConfig.from_env()
    parser = argparse.ArgumentParser(
        description="Adapt double-page French book text for single-page digital reading."
    )
    parser.add_argument(
        "--mode",
        choices=("spread", "story"),
        default="spread",
        help=(
            "Preprocessing strategy. 'spread' adapts one spread at a time; "
            "'story' uses page-labelled text with either one image call or "
            "parameterized, optionally planned chunks."
        ),
    )
    parser.add_argument("--source", type=Path, help="Cleaned French source text.")
    parser.add_argument("--pdf", type=Path, help="Source illustrated PDF.")
    parser.add_argument(
        "--allow-preprocessed-source",
        action="store_true",
        help=(
            "Allow source files whose names indicate an existing single-page "
            "preprocessing artifact. By default these inputs are rejected."
        ),
    )
    parser.add_argument(
        "--model",
        default=f"openai:{defaults.openai_adversarial_model}",
        help="Provider-qualified model, for example openai:gpt-5.5.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--skip-first", type=int, default=defaults.pdf_skip_first)
    parser.add_argument("--skip-last", type=int, default=defaults.pdf_skip_last)
    parser.add_argument("--dpi", type=int, default=defaults.multimodal_image_dpi)
    parser.add_argument("--max-spreads", type=int, help="Process only the first N spreads.")
    parser.add_argument(
        "--story-spreads-per-chunk",
        type=int,
        help=(
            "Double-page spreads per story-mode model call. When omitted, "
            "story mode keeps its original single-call behavior."
        ),
    )
    parser.add_argument(
        "--story-planner",
        choices=("off", "on", "only"),
        default="off",
        help=(
            "Optional story-mode page planner. 'on' generates a locked page plan "
            "before rewriting chunks; 'only' writes the planner output and stops."
        ),
    )
    parser.add_argument(
        "--story-planner-spreads-per-call",
        type=int,
        default=DEFAULT_STORY_PLANNER_SPREADS_PER_CALL,
        help=(
            "Target double-page spreads per planner call. Calls include one "
            "neighboring spread on each side as context and are merged into a "
            f"complete plan. Default: {DEFAULT_STORY_PLANNER_SPREADS_PER_CALL}."
        ),
    )
    parser.add_argument(
        "--image-summary-model",
        default=DEFAULT_IMAGE_SUMMARY_MODEL,
        help=(
            "Provider-qualified vision model used to create locked "
            f"visible_on_page evidence. Default: {DEFAULT_IMAGE_SUMMARY_MODEL}."
        ),
    )
    parser.add_argument(
        "--image-summary-temperature",
        type=float,
        default=DEFAULT_IMAGE_SUMMARY_TEMPERATURE,
        help=(
            "Sampling temperature for image summaries. Default: "
            f"{DEFAULT_IMAGE_SUMMARY_TEMPERATURE}."
        ),
    )
    parser.add_argument(
        "--image-summaries",
        type=Path,
        help=(
            "Load an existing versioned image_summaries JSON artifact instead "
            "of calling the vision model."
        ),
    )
    parser.add_argument(
        "--image-summary-output",
        type=Path,
        help="Optional path for the reusable image_summaries JSON artifact.",
    )
    parser.add_argument(
        "--planner-output",
        type=Path,
        help="Optional path for the standalone story planner JSON output.",
    )
    parser.add_argument("--output", type=Path, help="Final page-aligned French text path.")
    parser.add_argument("--artifacts-dir", type=Path, help="Plans and report output directory.")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached model responses.")
    parser.add_argument(
        "--render-from-text",
        type=Path,
        help=(
            "Render an existing single-page preprocessing text file without "
            "calling a model. Paragraphs are mapped from the first body page."
        ),
    )
    parser.add_argument(
        "--render-pdf",
        action="store_true",
        help="Render completed adapted pages into a PDF preview.",
    )
    parser.add_argument(
        "--render-full-pdf",
        action="store_true",
        help="Keep all source PDF pages in the rendered preview.",
    )
    parser.add_argument("--pdf-output", type=Path, help="Rendered PDF output path.")
    parser.add_argument(
        "--font-file",
        type=Path,
        help="Optional TTF/OTF font to embed for the rendered PDF.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=f"Extra padding around detected source text boxes. Default: {DEFAULT_PADDING}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing CLI."""

    args = parse_args()
    output_path, report_path = run_preprocessing(args)
    print(f"Saved adapted text: {output_path}")
    print(f"Saved preprocessing report: {report_path}")


if __name__ == "__main__":
    main()
