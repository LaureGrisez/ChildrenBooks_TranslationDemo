"""Tests for page-addressable JSON export and title extraction."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import fitz

from src.translation.config import TranslationWorkflowConfig
from src.translation.glossary import CharacterGlossary
from src.translation.structured_export import build_structured_book
from src.translation.title import extract_source_title, title_translation_prompt
from src.translation.workflow import TranslationCandidate, build_application
from src.translation.preprocessing import ensure_workflow_image_summaries


class StructuredExportTests(unittest.TestCase):
    def make_pdf(self, path: Path) -> None:
        document = fitz.open()
        for page_number in range(1, 7):
            page = document.new_page()
            if page_number == 1:
                page.insert_text((50, 80), "Auteur", fontsize=10)
                page.insert_text((50, 150), "LE GRAND TITRE", fontsize=32)
            elif page_number == 2:
                page.insert_text((50, 80), "Premier texte.")
            elif page_number == 4:
                page.insert_text((50, 80), "Deuxieme texte.")
        document.save(path)
        document.close()

    def glossary(self) -> CharacterGlossary:
        return CharacterGlossary(
            source_language="French",
            rows=[{"French": "Barbapapa", "English": "Barbapapa"}],
            supported_languages=["French", "English"],
        )

    def test_extracts_largest_title_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "book.pdf"
            self.make_pdf(pdf_path)
            self.assertEqual("LE GRAND TITRE", extract_source_title(pdf_path, 1))

    def test_builds_spreads_with_physical_page_indexes_and_blank_sides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "book.pdf"
            source_path = root / "source.txt"
            self.make_pdf(pdf_path)
            source_path.write_text("Premier texte.\n\nDeuxieme texte.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path,
                source_pdf_path=pdf_path,
                pdf_skip_first=1,
                pdf_skip_last=1,
                target_languages=["English"],
            )
            payload = build_structured_book(
                source_text=source_path.read_text(encoding="utf-8"),
                translated_text="First text.\n\nSecond text.",
                source_title="LE GRAND TITRE",
                translated_title="THE GREAT TITLE",
                language="English",
                glossary=self.glossary(),
                config=config,
            )

        self.assertEqual("en", payload["language"])
        self.assertEqual([2, 3], payload["translation"][0]["pages_index"])
        self.assertEqual("First text.", payload["translation"][0]["left"]["translated"])
        self.assertEqual("", payload["translation"][0]["right"]["translated"])
        self.assertEqual([4, 5], payload["translation"][1]["pages_index"])
        self.assertEqual(4, payload["translation"][1]["left"]["page_index"])
        self.assertEqual(
            {"original": "LE GRAND TITRE", "translated": "THE GREAT TITLE"},
            payload["title"],
        )

    def test_rejects_translated_paragraph_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "book.pdf"
            self.make_pdf(pdf_path)
            config = TranslationWorkflowConfig(
                source_pdf_path=pdf_path, pdf_skip_first=1, pdf_skip_last=1
            )
            with self.assertRaisesRegex(ValueError, "paragraph count drifted"):
                build_structured_book(
                    source_text="One.\n\nTwo.", translated_text="Merged.",
                    source_title="Title", translated_title="Title", language="English",
                    glossary=self.glossary(), config=config,
                )

    def test_title_prompt_uses_character_mapping(self) -> None:
        prompt = title_translation_prompt(
            source_title="L'arbre de Barbapapa", source_language="French",
            target_language="English", glossary=self.glossary(),
        )
        self.assertIn("Barbapapa -> Barbapapa", prompt)
        self.assertIn("Return only the translated title", prompt)

    def test_workflow_translates_title_and_builds_structured_state(self) -> None:
        def fake_candidate(spec, text, source_language, language, glossary, config,
                           cache, segments, segment_images, image_summaries):
            return TranslationCandidate(
                language=language, name=spec.name, provider=spec.provider,
                model=spec.model, temperature=spec.temperature, stance=spec.stance,
                text=f"{spec.name} first.\n\n{spec.name} second.", status="ok",
                latency_seconds=0.0,
            )

        def fake_model(**kwargs):
            prompt = kwargs["prompt"]
            if "Translate this children's-book title" in prompt:
                return "THE GREAT TITLE"
            if "senior editor" in prompt:
                return json.dumps({
                    "overall_winner": "Candidate 1", "ranking": ["Candidate 1"],
                    "decision_reasoning": "Good.", "paragraph_analysis": [],
                    "candidate_assessment": [], "revision_instructions": [],
                    "concise_summary": "Keep it.",
                })
            if "Summarize this translation critique" in prompt:
                return "Keep it."
            if "Create the final English translation" in prompt:
                return "First text.\n\nSecond text."
            raise AssertionError(prompt[:100])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "book.pdf"
            source_path = root / "source.txt"
            self.make_pdf(pdf_path)
            source_path.write_text("Premier texte.\n\nDeuxieme texte.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path, source_pdf_path=pdf_path,
                pdf_skip_first=1, pdf_skip_last=1, title_page_number=1,
                workflow_mode="text", evaluation_mode="single",
                target_languages=["English"], candidate_names=["gpt4o", "gpt5_5"],
                max_parallel_candidates=2, translation_output_dir=root / "translation",
                translation_cache_dir=root / "cache", burr_storage_dir=str(root / "burr"),
            )
            with patch("src.translation.workflow.run_candidate", side_effect=fake_candidate), patch(
                "src.translation.workflow.ask_model_with_recovery", side_effect=fake_model
            ):
                app, _ = build_application(config)
                _, _, state = app.run(halt_after=["generate_final_text"])

        payload = state["structured_translations"]["English"]
        self.assertEqual("THE GREAT TITLE", payload["title"]["translated"])
        self.assertEqual([2, 3], payload["translation"][0]["pages_index"])

    def test_missing_image_summary_artifact_is_created_at_configured_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "book.pdf"
            source_path = root / "source.txt"
            output_path = root / "client" / "preprocessing" / "summaries.json"
            self.make_pdf(pdf_path)
            source_path.write_text("Premier texte.\n\nDeuxieme texte.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path, source_pdf_path=pdf_path,
                pdf_skip_first=1, pdf_skip_last=1,
                translation_cache_dir=root / "cache",
            )
            artifact = {
                "schema_version": 2,
                "pages": [
                    {"page_number": 2, "visible_on_page": ["Scene one."]},
                    {"page_number": 4, "visible_on_page": ["Scene two."]},
                ],
                "spreads": [],
            }
            with patch(
                "src.translation.preprocessing.generate_image_summaries",
                return_value=({2: ["Scene one."], 4: ["Scene two."]}, {}, artifact),
            ):
                ensure_workflow_image_summaries(config, output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(2, json.loads(output_path.read_text())["schema_version"])


if __name__ == "__main__":
    unittest.main()
