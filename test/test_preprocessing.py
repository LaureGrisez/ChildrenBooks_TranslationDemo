"""Tests for standalone single-page preprocessing helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from src.translation.preprocessing import (
    PageSource,
    SpreadSource,
    adapted_pages_from_text,
    build_spread_sources,
    editorial_plan_prompt,
    parse_json_object,
    render_preprocessed_pdf,
    rewrite_prompt,
    story_chunk_prompt,
    story_planner_prompt,
    story_prompt,
    validate_story_plan,
    _previous_final_context,
    _source_section,
    _story_chunks,
    _story_plan_for_pages,
    _story_source_context,
    validate_plan,
    validate_page_result,
    validate_source_provenance,
)


class PreprocessingTests(unittest.TestCase):
    def test_maps_text_and_blank_pages_into_spreads(self) -> None:
        spreads = build_spread_sources(
            "Premier texte.\n\nDeuxième texte.",
            body_page_numbers=[6, 7, 8, 9],
            text_page_numbers=[7, 8],
        )

        self.assertEqual([(6, 7), (8, 9)], [spread.page_numbers for spread in spreads])
        self.assertEqual("", spreads[0].pages[0].source_text)
        self.assertEqual("Premier texte.", spreads[0].pages[1].source_text)
        self.assertEqual("Deuxième texte.", spreads[1].pages[0].source_text)

    def test_parses_fenced_json(self) -> None:
        self.assertEqual({"pages": []}, parse_json_object("```json\n{\"pages\": []}\n```"))

    def test_validates_exact_page_order_and_non_empty_text(self) -> None:
        pages = validate_page_result(
            {
                "pages": [
                    {"page_number": 6, "text": " À gauche. "},
                    {"page_number": 7, "text": "À droite."},
                ]
            },
            (6, 7),
        )
        self.assertEqual("À gauche.", pages[0]["text"])

        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_page_result(
                {"pages": [{"page_number": 7, "text": "Wrong order."}]},
                (6, 7),
            )

        with self.assertRaisesRegex(ValueError, "no final text"):
            validate_page_result(
                {
                    "pages": [
                        {"page_number": 6, "text": ""},
                        {"page_number": 7, "text": "Text."},
                    ]
                },
                (6, 7),
            )

    def test_validates_plan_page_order(self) -> None:
        plan = {"pages": [{"page_number": 6}, {"page_number": 7}]}
        self.assertIs(plan, validate_plan(plan, (6, 7)))
        with self.assertRaisesRegex(ValueError, "plan page numbers"):
            validate_plan({"pages": [{"page_number": 7}]}, (6, 7))

    def test_maps_existing_text_to_body_pages(self) -> None:
        pages = adapted_pages_from_text("Page six.\n\nPage seven.", [6, 7, 8])
        self.assertEqual(
            [(6, "Page six."), (7, "Page seven.")],
            [(page["page_number"], page["text"]) for page in pages],
        )

        with self.assertRaisesRegex(ValueError, "adapted paragraphs"):
            adapted_pages_from_text("One.\n\nTwo.\n\nThree.", [6, 7])

    def test_render_pdf_keeps_only_adapted_pages_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_pdf = Path(tmpdir) / "source.pdf"
            output_pdf = Path(tmpdir) / "preview.pdf"
            document = fitz.open()
            for index in range(5):
                page = document.new_page(width=200, height=200)
                page.insert_text((20, 180), f"Original page {index + 1}")
            document.save(source_pdf)
            document.close()

            render_preprocessed_pdf(
                pdf_path=source_pdf,
                output_path=output_pdf,
                pages=[
                    {"page_number": 2, "text": "Adapted two."},
                    {"page_number": 3, "text": "Adapted three."},
                ],
            )

            with fitz.open(output_pdf) as rendered:
                self.assertEqual(2, rendered.page_count)

    def test_render_pdf_uses_larger_panel_when_source_box_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_pdf = Path(tmpdir) / "source.pdf"
            output_pdf = Path(tmpdir) / "preview.pdf"
            document = fitz.open()
            page = document.new_page(width=400, height=600)
            page.insert_text((20, 30), "Bref.", fontsize=8)
            document.save(source_pdf)
            document.close()

            render_preprocessed_pdf(
                pdf_path=source_pdf,
                output_path=output_pdf,
                pages=[
                    {
                        "page_number": 1,
                        "text": (
                            "Ce texte adapté est volontairement bien plus long que "
                            "la minuscule zone de texte originale afin de vérifier "
                            "le panneau de secours."
                        ),
                    }
                ],
            )

            with fitz.open(output_pdf) as rendered:
                self.assertEqual(1, rendered.page_count)
                self.assertIn("Ce texte adapté", rendered[0].get_text())

    def test_prompts_require_chronological_page_order(self) -> None:
        spread = SpreadSource(
            index=0,
            pages=(
                PageSource(page_number=8, source_text="Une tentative."),
                PageSource(page_number=9, source_text="Une conclusion."),
            ),
        )
        plan_prompt = editorial_plan_prompt(
            spread=spread,
            previous_source="",
            next_source="",
            previous_adapted="",
        )
        final_prompt = rewrite_prompt(
            spread=spread,
            plan={"chronology_check": "La tentative doit précéder la conclusion."},
            previous_adapted="",
        )

        self.assertIn("chronologie", plan_prompt)
        self.assertIn("chronology_check", plan_prompt)
        self.assertIn("narrative_role", plan_prompt)
        self.assertIn("cause et sa conséquence", final_prompt)
        self.assertIn("narrative_role", final_prompt)

    def test_story_prompt_preserves_original_whole_story_behavior(self) -> None:
        prompt = story_prompt(
            source_text="Une histoire complète.",
            page_numbers=[6, 7, 8, 9],
        )
        self.assertIn("texte complet de l'histoire", prompt)
        self.assertIn("Une histoire complète.", prompt)
        self.assertIn("story_strategy", prompt)
        self.assertIn("6, 7, 8, 9", prompt)

    def test_story_chunk_prompt_partitions_story_and_page_images(self) -> None:
        prompt = story_chunk_prompt(
            source_before="Avant.",
            current_source="À adapter.",
            source_after="Après.",
            previous_final_pages="PAGE 4\nTexte validé.",
            page_numbers=[6, 7, 8, 9],
        )
        self.assertIn("TEXTE DE RÉFÉRENCE", prompt)
        self.assertIn("Avant.", prompt)
        self.assertIn("DÉBUT DE LA PORTION À ADAPTER", prompt)
        self.assertIn("À adapter.", prompt)
        self.assertIn("FIN DE LA PORTION À ADAPTER", prompt)
        self.assertIn("Après.", prompt)
        self.assertIn("PAGE 4", prompt)
        self.assertNotIn("APERÇU DE LA SUITE", prompt)
        self.assertIn("6, 7, 8, 9", prompt)
        self.assertIn("correspondent individuellement", prompt)
        self.assertIn("LOCKED", prompt)
        self.assertIn("chunk_summary", prompt)
        self.assertIn("visual_grounding", prompt)
        self.assertIn("une entrée pour chaque page physique", prompt)

    def test_story_planner_prompt_uses_lean_locked_schema(self) -> None:
        spreads = [
            SpreadSource(
                index=0,
                pages=(
                    PageSource(page_number=6, source_text="Début."),
                    PageSource(page_number=7, source_text=""),
                ),
            )
        ]
        prompt = story_planner_prompt(source_text="Début.", spreads=spreads)

        self.assertIn("uniquement un PLAN", prompt)
        self.assertIn("DOUBLE PAGE 6, 7", prompt)
        self.assertIn("PAGE 7", prompt)
        self.assertIn("aucun texte original", prompt)
        self.assertIn("visible_on_page", prompt)
        self.assertIn("source_text_on_page", prompt)
        self.assertIn("content_to_move_later", prompt)
        self.assertIn("allowed_content", prompt)
        self.assertIn("forbidden_content", prompt)
        self.assertIn("source_to_preserve", prompt)
        self.assertIn("handoff_state_after_page", prompt)
        self.assertNotIn("visual_moment", prompt)
        self.assertNotIn("story_beat", prompt)
        self.assertNotIn("text_budget", prompt)

    def test_validates_story_plan_and_injects_page_subset(self) -> None:
        plan = validate_story_plan(
            {
                "story_arc": {"global_strategy": "Simple."},
                "pages": [
                    {
                        "page_number": 6,
                        "visible_on_page": ["Image six."],
                        "source_text_on_page": ["Source six."],
                        "content_to_move_later": [],
                        "allowed_content": ["Dire six."],
                        "forbidden_content": ["Ne pas dire sept."],
                        "source_to_preserve": ["Six"],
                        "adaptation_instruction": "Garder court.",
                        "handoff_state_after_page": "Six fini.",
                    },
                    {
                        "page_number": 7,
                        "visible_on_page": ["Image sept."],
                        "source_text_on_page": ["Source sept."],
                        "content_to_move_later": ["Dire six ailleurs."],
                        "allowed_content": ["Dire sept."],
                        "forbidden_content": [],
                        "source_to_preserve": [],
                        "adaptation_instruction": "Continuer.",
                        "handoff_state_after_page": "Sept fini.",
                    },
                ],
            },
            (6, 7),
        )

        subset = _story_plan_for_pages(plan, [7])
        self.assertEqual([7], [page["page_number"] for page in subset["pages"]])
        self.assertEqual(
            [6],
            [page["page_number"] for page in subset["context_pages_before"]],
        )
        self.assertEqual([], subset["context_pages_after"])
        prompt = story_chunk_prompt(
            source_before="",
            current_source="Sept.",
            source_after="",
            previous_final_pages="PAGE 6\nSix.",
            page_numbers=[7],
            story_plan=subset,
        )
        self.assertIn("PLAN PAGE PAR PAGE", prompt)
        self.assertIn("contrainte verrouillée", prompt)
        self.assertIn("Dire sept.", prompt)
        self.assertIn("Dire six.", prompt)
        self.assertIn("context_pages_before", prompt)
        self.assertIn("ne les réécris pas", prompt)

        with self.assertRaisesRegex(ValueError, "page numbers"):
            validate_story_plan({"story_arc": {}, "pages": []}, (6,))

    def test_story_mode_chunks_selected_spreads(self) -> None:
        spreads = [
            SpreadSource(
                index=index,
                pages=(
                    PageSource(page_number=6 + index * 2, source_text=f"Beat {index}."),
                    PageSource(page_number=7 + index * 2, source_text=""),
                ),
            )
            for index in range(7)
        ]
        chunks = _story_chunks(
            spreads,
            spreads_per_chunk=5,
            max_spreads=None,
        )
        self.assertEqual([5, 2], [len(chunk) for chunk in chunks])

        original_behavior = _story_chunks(
            spreads,
            spreads_per_chunk=None,
            max_spreads=None,
        )
        self.assertEqual([7], [len(chunk) for chunk in original_behavior])

        limited_chunks = _story_chunks(
            spreads,
            spreads_per_chunk=5,
            max_spreads=6,
        )
        self.assertEqual([5, 1], [len(chunk) for chunk in limited_chunks])
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _story_chunks(spreads, spreads_per_chunk=0, max_spreads=None)

    def test_story_context_uses_original_sections_and_two_final_pages(self) -> None:
        spreads = [
            SpreadSource(
                index=0,
                pages=(PageSource(page_number=6, source_text="Premier beat."),),
            ),
            SpreadSource(
                index=1,
                pages=(PageSource(page_number=7, source_text="Deuxième beat."),),
            ),
        ]
        self.assertEqual(
            "Premier beat.\n\nDeuxième beat.",
            _source_section(spreads),
        )
        self.assertEqual(
            "PAGE 7\nSept.\n\nPAGE 8\nHuit.",
            _previous_final_context(
                [
                    {"page_number": 6, "text": "Six."},
                    {"page_number": 7, "text": "Sept."},
                    {"page_number": 8, "text": "Huit."},
                ]
            ),
        )

    def test_planned_story_context_is_bounded_around_chunk(self) -> None:
        spreads = [
            SpreadSource(
                index=index,
                pages=(PageSource(page_number=6 + index, source_text=str(index)),),
            )
            for index in range(7)
        ]

        before, after = _story_source_context(
            spreads,
            chunk_start=3,
            chunk_end=5,
            bounded=True,
        )
        self.assertEqual([2], [spread.index for spread in before])
        self.assertEqual([5, 6], [spread.index for spread in after])

        full_before, full_after = _story_source_context(
            spreads,
            chunk_start=3,
            chunk_end=5,
            bounded=False,
        )
        self.assertEqual([0, 1, 2], [spread.index for spread in full_before])
        self.assertEqual([5, 6], [spread.index for spread in full_after])

    def test_rejects_generated_single_page_inputs_without_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "already generated single-page"):
            validate_source_provenance(
                Path("book.repaired.single_page.txt"),
                Path("book.repaired.single_page.pdf"),
                allow_preprocessed_source=False,
            )

        validate_source_provenance(
            Path("book.repaired.single_page.txt"),
            Path("book.repaired.single_page.pdf"),
            allow_preprocessed_source=True,
        )
        validate_source_provenance(
            Path("book.txt"),
            Path("book.pdf"),
            allow_preprocessed_source=False,
        )


if __name__ == "__main__":
    unittest.main()
