"""Unit tests for deterministic panel-mode behavior."""

from __future__ import annotations

import unittest
import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.translation.alignment import align_candidate_paragraphs
from src.translation.config import TranslationWorkflowConfig
from src.translation.panel_aggregation import aggregate_judgments
from src.translation.panel_blinding import blind_options, restore_judge_result
from src.translation.panel_models import CRITERIA, validate_judge_result
from src.translation.panel_prompts import synthesis_prompt
from src.translation.prompts import (
    aligned_final_paragraph_prompt,
    critic_prompt,
    final_prompt,
)
from src.translation.ui_support import (
    build_ui_config,
    collect_versions,
    missing_candidate_credentials,
    missing_judge_credentials,
    panel_aggregate_rows,
    workflow_mermaid,
    workflow_node_descriptions,
)


def judge_result(ranking: list[str], scores: dict[str, int]) -> dict:
    return {
        "overall_ranking": ranking,
        "option_scores": {
            option: {
                **{criterion: score for criterion in CRITERIA},
                "critical_errors": [],
                "remarks": [],
            }
            for option, score in scores.items()
        },
        "confidence": 1.0,
        "comparisons": [],
        "recommended_phrases": [],
    }


class AlignmentTests(unittest.TestCase):
    def test_aligns_successful_candidates_exactly(self) -> None:
        aligned = align_candidate_paragraphs(
            "Source one.\n\nSource two.",
            [
                {"name": "a", "status": "ok", "text": "A one.\n\nA two."},
                {"name": "b", "status": "ok", "text": "B one.\n\nB two."},
            ],
        )
        self.assertEqual(["p0001", "p0002"], [item["paragraph_id"] for item in aligned])
        self.assertEqual("B two.", aligned[1]["options"]["b"])

    def test_rejects_paragraph_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2"):
            align_candidate_paragraphs(
                "One.\n\nTwo.",
                [
                    {"name": "a", "status": "ok", "text": "Only one."},
                    {"name": "b", "status": "ok", "text": "B one.\n\nB two."},
                ],
            )

    def test_rejects_duplicate_candidate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique candidate names"):
            align_candidate_paragraphs(
                "One.",
                [
                    {"name": "same", "status": "ok", "text": "A."},
                    {"name": "same", "status": "ok", "text": "B."},
                ],
            )


class BlindingTests(unittest.TestCase):
    def test_blinding_is_deterministic_and_restorable(self) -> None:
        options = {"candidate_a": "A", "candidate_b": "B", "candidate_c": "C"}
        blinded, mapping = blind_options(
            options, seed="seed", paragraph_id="p0001", judge_id="judge_1"
        )
        repeated, repeated_mapping = blind_options(
            options, seed="seed", paragraph_id="p0001", judge_id="judge_1"
        )
        self.assertEqual(blinded, repeated)
        self.assertEqual(mapping, repeated_mapping)
        restored = restore_judge_result(
            {"overall_ranking": list(blinded), "option_scores": {key: {} for key in blinded}},
            mapping,
        )
        self.assertEqual(set(options), set(restored["overall_ranking"]))
        self.assertEqual(set(options), set(restored["option_scores"]))


class AggregationTests(unittest.TestCase):
    def test_majority_winner_ranks_first(self) -> None:
        results = [
            judge_result(["a", "b", "c"], {"a": 9, "b": 7, "c": 5}),
            judge_result(["a", "c", "b"], {"a": 8, "b": 5, "c": 7}),
            judge_result(["b", "a", "c"], {"a": 8, "b": 9, "c": 4}),
        ]
        aggregate = aggregate_judgments(
            results,
            pairwise_weight=0.45,
            ranking_weight=0.35,
            score_weight=0.20,
            critical_error_confirmations=2,
        )
        self.assertEqual("a", aggregate["ranking"][0])
        self.assertEqual(3, len(aggregate["selected_options"]))

    def test_confirmed_critical_error_vetoes_option(self) -> None:
        results = [
            judge_result(["a", "b"], {"a": 10, "b": 6}),
            judge_result(["a", "b"], {"a": 10, "b": 6}),
        ]
        for result in results:
            result["option_scores"]["a"]["critical_errors"] = ["Omitted action"]
        aggregate = aggregate_judgments(
            results,
            pairwise_weight=0.45,
            ranking_weight=0.35,
            score_weight=0.20,
            critical_error_confirmations=2,
        )
        self.assertEqual("b", aggregate["ranking"][0])


class ContractAndConfigTests(unittest.TestCase):
    def test_rejects_incomplete_ranking(self) -> None:
        payload = judge_result(["a"], {"a": 8, "b": 7})
        with self.assertRaisesRegex(ValueError, "overall_ranking"):
            validate_judge_result(payload, ["a", "b"])

    def test_panel_requires_two_distinct_judge_models(self) -> None:
        config = TranslationWorkflowConfig(
            evaluation_mode="panel",
            panel_judges=["openai:a"],
        )
        with self.assertRaisesRegex(ValueError, "two distinct judge models"):
            config.validate()

    def test_panel_accepts_distinct_judges_from_same_provider(self) -> None:
        TranslationWorkflowConfig(
            evaluation_mode="panel",
            panel_judges=["openai:a", "openai:b"],
        ).validate()

    def test_panel_normalizes_judge_model_references(self) -> None:
        config = TranslationWorkflowConfig(
            evaluation_mode="panel",
            panel_judges=["openai: a", "anthropic:b"],
        )
        config.validate()
        self.assertEqual(
            [("openai", "a"), ("anthropic", "b")],
            config.panel_judge_specs(),
        )

    def test_accepts_additional_litellm_provider_references(self) -> None:
        self.assertEqual(
            ("zai", "glm-4.5"),
            TranslationWorkflowConfig.parse_model_ref("ZAI:glm-4.5"),
        )

    def test_ui_config_maps_panel_mode_and_language(self) -> None:
        config = build_ui_config(
            generation_mode_label="Text only",
            judging_mode_label="Panel judges",
            source_text_path=Path("/tmp/source.txt"),
            source_pdf_path=None,
            target_language="Finnish",
            candidate_names=["google_translation", "gpt4o"],
            candidate_temperature=0.6,
            openai_base_model="gpt-4o-mini",
            openai_adversarial_model="gpt-5.5",
            output_dir=Path("/tmp/output"),
            cache_dir=Path("/tmp/cache"),
            burr_storage_dir="/tmp/burr",
        )
        self.assertEqual("panel", config.evaluation_mode)
        self.assertEqual(["Finnish"], config.target_languages)
        self.assertEqual(0.6, config.candidate_temperature)
        self.assertEqual("gpt-4o-mini", config.openai_base_model)
        self.assertEqual([], config.panel_judges)

    def test_ui_config_preserves_explicit_panel_judges(self) -> None:
        config = build_ui_config(
            generation_mode_label="Multimodal",
            judging_mode_label="Panel judges",
            source_text_path=Path("/tmp/source.txt"),
            source_pdf_path=None,
            target_language="Finnish",
            candidate_names=["gpt4o", "gpt5_5"],
            candidate_temperature=0.6,
            openai_base_model="gpt-4o-mini",
            openai_adversarial_model="gpt-5.5",
            panel_judges=["openai:gpt-4o-mini", "anthropic:claude-test"],
            panel_judge_temperature=0.3,
            output_dir=Path("/tmp/output"),
            cache_dir=Path("/tmp/cache"),
            burr_storage_dir="/tmp/burr",
        )
        self.assertEqual(
            ["openai:gpt-4o-mini", "anthropic:claude-test"],
            config.panel_judges,
        )
        self.assertEqual(0.3, config.panel_judge_temperature)
        self.assertEqual("multimodal", config.workflow_mode)

    def test_ui_helpers_expose_panel_versions_and_scores(self) -> None:
        state = {
            "candidate_translations": {
                "Finnish": [{"name": "a", "text": "Candidate.", "status": "ok"}]
            },
            "final_paragraphs": {"Finnish": {"p0001": "Before audit."}},
            "final_translations": {"Finnish": "Final."},
            "panel_aggregates": {
                "Finnish": {
                    "p0001": {
                        "ranking": ["a"],
                        "total_scores": {"a": 0.8},
                        "confirmed_critical_errors": {"a": []},
                    }
                }
            },
        }
        versions = collect_versions(state, "Finnish")
        self.assertIn("Panel synthesis before audit repairs", versions)
        self.assertEqual("a", panel_aggregate_rows(state, "Finnish")[0]["candidate"])
        self.assertIn(
            "Independent judge panel",
            workflow_mermaid("Text only", "Panel judges"),
        )

    def test_mermaid_graphs_style_llm_nodes_and_describe_every_node(self) -> None:
        for generation_mode in ("Text only", "Multimodal"):
            for judging_mode in ("Single judge", "Panel judges"):
                graph = workflow_mermaid(generation_mode, judging_mode)
                descriptions = workflow_node_descriptions(
                    generation_mode, judging_mode
                )
                self.assertIn("classDef llm", graph)
                node_names = set(re.findall(r"\b[A-Z]\[([^\]]+)\]", graph))
                self.assertEqual(node_names, set(descriptions))

    def test_candidate_temperature_override_applies_to_all_specs(self) -> None:
        from src.translation.workflow import build_candidate_specs

        config = TranslationWorkflowConfig(
            candidate_names=["google_translation", "gpt4o", "gpt5_5"],
            candidate_temperature=0.7,
        )
        specs = build_candidate_specs(config)
        self.assertEqual([0.7, 0.7, 0.7], [spec.temperature for spec in specs])

    def test_per_candidate_temperatures_override_selected_specs(self) -> None:
        from src.translation.workflow import build_candidate_specs

        config = TranslationWorkflowConfig(
            candidate_names=["gpt5_5", "claude_sonnet_4_6", "gemini_3"],
            candidate_temperatures={"gpt5_5": 1.1, "gemini_3": 0.8},
        )
        specs = build_candidate_specs(config)
        self.assertEqual([1.1, 0.4, 0.8], [spec.temperature for spec in specs])

    def test_validates_experiment_profile_and_image_parameters(self) -> None:
        TranslationWorkflowConfig(
            translation_profile="creative",
            image_context_mode="summary",
            image_summaries_path=Path("summaries.json"),
            evaluation_image_stages={"judges", "synthesis", "audit"},
        ).validate()
        with self.assertRaisesRegex(ValueError, "IMAGE_SUMMARIES_PATH"):
            TranslationWorkflowConfig(image_context_mode="summary").validate()
        with self.assertRaisesRegex(ValueError, "TRANSLATION_PROFILE"):
            TranslationWorkflowConfig(translation_profile="wild").validate()

    def test_runtime_directories_are_created_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TranslationWorkflowConfig(
                translation_output_dir=root / "client" / "outputs",
                translation_cache_dir=root / "client" / "cache",
                burr_storage_dir=str(root / "client" / "burr"),
                image_summaries_path=root / "client" / "preprocessing" / "summary.json",
            )
            config.apply_environment_defaults()
            self.assertTrue(config.translation_output_dir.is_dir())
            self.assertTrue(config.translation_cache_dir.is_dir())
            self.assertTrue(Path(config.burr_storage_dir).is_dir())
            self.assertTrue(config.image_summaries_path.parent.is_dir())

    def test_original_source_guard_rejects_preprocessed_inputs(self) -> None:
        config = TranslationWorkflowConfig(
            source_text_path=Path("book.single_page.txt"),
            source_pdf_path=Path("book.single_page.pdf"),
            require_original_source=True,
        )
        with self.assertRaisesRegex(ValueError, "rejects preprocessed source"):
            config.validate()

    def test_all_five_builtin_candidates_can_be_selected(self) -> None:
        from src.translation.workflow import build_candidate_specs, candidate_judge_refs

        config = TranslationWorkflowConfig(
            max_parallel_candidates=5,
            gemini_model="gemini-2.5-flash",
            candidate_names=[
                "google_translation",
                "gpt4o",
                "gpt5_5",
                "claude_sonnet_4_6",
                "gemini_3",
            ],
        )
        specs = build_candidate_specs(config)
        self.assertEqual(
            ["google", "openai", "openai", "anthropic", "gemini"],
            [spec.provider for spec in specs],
        )
        self.assertEqual(
            [
                "openai:gpt-4o",
                "openai:gpt-5.5",
                "anthropic:claude-sonnet-4-6",
                "gemini:gemini-2.5-flash",
            ],
            candidate_judge_refs(specs),
        )

    def test_candidate_judge_refs_exclude_google_and_deduplicate_models(self) -> None:
        from src.translation.workflow import build_candidate_specs, candidate_judge_refs

        config = TranslationWorkflowConfig(
            candidate_names=["google_translation", "gpt4o", "gpt5_5"],
            openai_base_model="same-model",
            openai_adversarial_model="same-model",
        )
        self.assertEqual(
            ["openai:same-model"],
            candidate_judge_refs(build_candidate_specs(config)),
        )

    def test_build_application_derives_panel_judges_from_candidates(self) -> None:
        from src.translation.workflow import build_application

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.txt"
            source_path.write_text("Source paragraph.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path,
                source_pdf_path=None,
                workflow_mode="text",
                target_languages=["Finnish"],
                evaluation_mode="panel",
                candidate_names=["google_translation", "gpt4o", "gpt5_5"],
                panel_judges=[],
                burr_storage_dir=str(Path(temp_dir) / "burr"),
            )
            _, resolved = build_application(config)

        self.assertEqual(
            ["openai:gpt-4o", "openai:gpt-5.5"],
            resolved.panel_judges,
        )

    def test_candidate_credential_requirements_are_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            missing = missing_candidate_credentials(
                ["claude_sonnet_4_6", "gemini_3"]
            )
        self.assertEqual(2, len(missing))
        self.assertIn("ANTHROPIC_API_KEY", missing[0])
        self.assertIn("GEMINI_API_KEY", missing[1])

    def test_judge_credential_requirements_are_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            missing = missing_judge_credentials(
                ["openai:gpt-test", "anthropic:claude-test", "gemini:gemini-test"]
            )
        self.assertEqual(2, len(missing))

    def test_provider_qualified_default_models_are_parsed(self) -> None:
        config = TranslationWorkflowConfig(
            default_critic_model="anthropic:claude-test",
            default_aggregation_model="openai:gpt-test",
            default_critic_summarizer_model="gemini:gemini-test",
        )
        self.assertEqual(("anthropic", "claude-test"), config.critic_model_spec())
        self.assertEqual(("openai", "gpt-test"), config.aggregation_model_spec())
        self.assertEqual(
            ("gemini", "gemini-test"), config.critic_summarizer_model_spec()
        )

    def test_source_paths_can_be_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SOURCE_TEXT_PATH": "/tmp/source.txt",
                "SOURCE_PDF_PATH": "/tmp/source.pdf",
            },
        ):
            config = TranslationWorkflowConfig()
        self.assertEqual(Path("/tmp/source.txt"), config.source_text_path)
        self.assertEqual(Path("/tmp/source.pdf"), config.source_pdf_path)

    def test_bare_default_model_remains_an_openai_model(self) -> None:
        self.assertEqual(
            ("openai", "legacy-model"),
            TranslationWorkflowConfig.parse_model_ref("legacy-model"),
        )

    def test_direct_config_workflow_exposes_default_google_credentials(self) -> None:
        from src.translation.config import DEFAULT_GOOGLE_CREDENTIALS
        from src.translation.workflow import build_application

        previous = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        try:
            config = TranslationWorkflowConfig(
                target_languages=["Finnish"],
                burr_storage_dir="/private/tmp/.burr-google-credentials-test",
            )
            build_application(config)
            self.assertEqual(
                str(DEFAULT_GOOGLE_CREDENTIALS),
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            )
        finally:
            if previous is None:
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = previous

    def test_single_paragraph_normalization_removes_internal_blank_lines(self) -> None:
        from src.translation.workflow import (
            normalize_segment_translation,
            normalize_single_paragraph,
        )

        self.assertEqual(
            "First sentence. Second sentence.",
            normalize_single_paragraph("First sentence.\n\nSecond sentence."),
        )
        self.assertEqual(
            "First sentence. Second sentence.",
            normalize_segment_translation(
                "First sentence.\n\nSecond sentence.",
                expected_page_count=1,
            ),
        )

    def test_segment_normalization_preserves_two_page_boundary(self) -> None:
        from src.translation.workflow import normalize_segment_translation

        self.assertEqual(
            "First page.\n\nSecond page.",
            normalize_segment_translation(
                "First page.\n\nSecond page.",
                expected_page_count=2,
            ),
        )
        with self.assertRaisesRegex(ValueError, "3 page blocks; expected 2"):
            normalize_segment_translation(
                "First page.\n\nAccidental break.\n\nSecond page.",
                expected_page_count=2,
            )

    def test_model_facing_prompts_hide_candidate_identities(self) -> None:
        from src.translation.glossary import load_character_glossary

        glossary = load_character_glossary(
            TranslationWorkflowConfig().character_names_csv
        )
        candidates = [
            {
                "name": "secret_candidate_alpha",
                "provider": "secret_provider_alpha",
                "model": "secret_model_alpha",
                "temperature": 0.4,
                "status": "ok",
                "text": "Alpha translation.",
            },
            {
                "name": "secret_candidate_beta",
                "provider": "secret_provider_beta",
                "model": "secret_model_beta",
                "temperature": 0.7,
                "status": "ok",
                "text": "Beta translation.",
            },
        ]
        prompts = [
            critic_prompt("Source.", "French", "English", candidates, glossary),
            final_prompt(
                "Source.",
                "French",
                "English",
                candidates,
                "Prefer Candidate 1.",
                glossary,
            ),
            aligned_final_paragraph_prompt(
                source_paragraph="Source.",
                previous_source="",
                next_source="",
                previous_final="",
                source_language="French",
                target_language="English",
                candidates=candidates,
                critique_summary="Prefer Candidate 1.",
                glossary=glossary,
                paragraph_number=1,
                paragraph_count=1,
            ),
        ]
        forbidden = {
            candidate[field]
            for candidate in candidates
            for field in ("name", "provider", "model")
        }
        for prompt in prompts:
            self.assertTrue(forbidden.isdisjoint(prompt))
            self.assertIn("Candidate 1", prompt)

    def test_panel_synthesis_prompt_hides_restored_candidate_names(self) -> None:
        from src.translation.glossary import load_character_glossary

        glossary = load_character_glossary(
            TranslationWorkflowConfig().character_names_csv
        )
        prompt = synthesis_prompt(
            block={
                "source": "Source.",
                "previous_source": "",
                "next_source": "",
            },
            target_language="English",
            glossary=glossary,
            selected_options={
                "secret_candidate_alpha": "Alpha.",
                "secret_candidate_beta": "Beta.",
            },
            aggregate={
                "ranking": ["secret_candidate_alpha", "secret_candidate_beta"],
                "confirmed_critical_errors": {
                    "secret_candidate_alpha": [],
                    "secret_candidate_beta": [],
                },
                "consensus_remarks": {
                    "secret_candidate_alpha": ["Strong rhythm."],
                    "secret_candidate_beta": [],
                },
                "recommended_phrases": [
                    {"option": "secret_candidate_alpha", "phrase": "Alpha."}
                ],
            },
            previous_final="",
        )
        self.assertNotIn("secret_candidate_alpha", prompt)
        self.assertNotIn("secret_candidate_beta", prompt)
        self.assertIn("Option 1", prompt)

    def test_anonymizes_candidate_names_in_structured_and_prose_critique(self) -> None:
        from src.translation.workflow import anonymize_candidate_references

        candidates = [
            {"name": "secret_candidate_alpha"},
            {"name": "secret_candidate_beta"},
        ]
        anonymized = anonymize_candidate_references(
            {
                "winner": "secret_candidate_alpha",
                "reason": "secret_candidate_alpha beats secret_candidate_beta.",
            },
            candidates,
        )
        self.assertEqual("Candidate 1", anonymized["winner"])
        self.assertEqual(
            "Candidate 1 beats Candidate 2.",
            anonymized["reason"],
        )


class AnthropicCandidateTests(unittest.TestCase):
    def test_text_candidate_uses_shared_anthropic_provider(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.workflow import CandidateSpec, run_candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                translation_cache_dir=Path(temp_dir) / "cache",
            )
            glossary = load_character_glossary(config.character_names_csv)
            with patch(
                "src.translation.workflow.ask_model_with_recovery",
                return_value="Claude translation.",
            ) as ask_model:
                candidate = run_candidate(
                    CandidateSpec(
                        name="claude",
                        provider="anthropic",
                        model="claude-test",
                        temperature=0.4,
                        stance="Natural.",
                    ),
                    "Source.",
                    "French",
                    "English",
                    glossary,
                    config,
                    ResponseCache(config.translation_cache_dir),
                )

        self.assertEqual("ok", candidate.status)
        self.assertEqual("Claude translation.", candidate.text)
        self.assertEqual("anthropic", ask_model.call_args.kwargs["provider"])
        self.assertEqual("claude-test", ask_model.call_args.kwargs["model"])

    def test_multimodal_anthropic_candidate_receives_spread_image(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.segmentation import SpreadSegment
        from src.translation.workflow import (
            CandidateSpec,
            SegmentImageInput,
            run_candidate,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                workflow_mode="multimodal",
                translation_cache_dir=Path(temp_dir) / "cache",
            )
            glossary = load_character_glossary(config.character_names_csv)
            segment = SpreadSegment(
                index=0,
                spread_pages=(6, 7),
                page_numbers=(6,),
                page_texts=("Source.",),
                previous_source_text="",
            )
            image_data_url = "data:image/jpeg;base64,YWJj"
            with patch(
                "src.translation.workflow.ask_model_with_recovery",
                return_value="Claude visual translation.",
            ) as ask_model:
                candidate = run_candidate(
                    CandidateSpec(
                        name="claude",
                        provider="anthropic",
                        model="claude-test",
                        temperature=0.4,
                        stance="Natural.",
                    ),
                    "Source.",
                    "French",
                    "English",
                    glossary,
                    config,
                    ResponseCache(config.translation_cache_dir),
                    [segment],
                    {
                        0: SegmentImageInput(
                            spread_pages=(6, 7),
                            data_url=image_data_url,
                        )
                    },
                )

        self.assertEqual("ok", candidate.status)
        self.assertEqual(
            image_data_url, ask_model.call_args.kwargs["image_data_url"]
        )

    def test_multimodal_gemini_candidate_receives_spread_image(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.segmentation import SpreadSegment
        from src.translation.workflow import (
            CandidateSpec,
            SegmentImageInput,
            run_candidate,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                workflow_mode="multimodal",
                translation_cache_dir=Path(temp_dir) / "cache",
            )
            glossary = load_character_glossary(config.character_names_csv)
            segment = SpreadSegment(
                index=0,
                spread_pages=(6, 7),
                page_numbers=(6,),
                page_texts=("Source.",),
                previous_source_text="",
            )
            image_data_url = "data:image/jpeg;base64,YWJj"
            with patch(
                "src.translation.workflow.ask_model_with_recovery",
                return_value="Gemini visual translation.",
            ) as ask_model:
                candidate = run_candidate(
                    CandidateSpec(
                        name="gemini",
                        provider="gemini",
                        model="gemini-test",
                        temperature=0.4,
                        stance="Natural.",
                    ),
                    "Source.",
                    "French",
                    "English",
                    glossary,
                    config,
                    ResponseCache(config.translation_cache_dir),
                    [segment],
                    {
                        0: SegmentImageInput(
                            spread_pages=(6, 7),
                            data_url=image_data_url,
                        )
                    },
                )

        self.assertEqual("ok", candidate.status)
        self.assertEqual(
            image_data_url, ask_model.call_args.kwargs["image_data_url"]
        )

    def test_multimodal_model_translates_each_page_with_spread_image(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.segmentation import SpreadSegment
        from src.translation.workflow import (
            CandidateSpec,
            SegmentImageInput,
            run_candidate,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                workflow_mode="multimodal",
                translation_cache_dir=Path(temp_dir) / "cache",
            )
            glossary = load_character_glossary(config.character_names_csv)
            segment = SpreadSegment(
                index=0,
                spread_pages=(10, 11),
                page_numbers=(10, 11),
                page_texts=("First source page.", "Second source page."),
                previous_source_text="",
            )
            image_data_url = "data:image/jpeg;base64,YWJj"
            with patch(
                "src.translation.workflow.ask_model_with_recovery",
                side_effect=[
                    "First translated page.\n\nAccidental internal break.",
                    "Second translated page.",
                ],
            ) as ask_model:
                candidate = run_candidate(
                    CandidateSpec(
                        name="gemini",
                        provider="gemini",
                        model="gemini-test",
                        temperature=0.4,
                        stance="Natural.",
                    ),
                    segment.source_text,
                    "French",
                    "English",
                    glossary,
                    config,
                    ResponseCache(config.translation_cache_dir),
                    [segment],
                    {
                        0: SegmentImageInput(
                            spread_pages=(10, 11),
                            data_url=image_data_url,
                        )
                    },
                )

        self.assertEqual("ok", candidate.status)
        self.assertEqual(
            "First translated page. Accidental internal break.\n\n"
            "Second translated page.",
            candidate.text,
        )
        self.assertEqual(2, ask_model.call_count)
        self.assertEqual(
            image_data_url,
            ask_model.call_args_list[0].kwargs["image_data_url"],
        )
        self.assertEqual(
            image_data_url,
            ask_model.call_args_list[1].kwargs["image_data_url"],
        )

    def test_multimodal_google_batches_page_paragraphs_with_boundaries(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.segmentation import SpreadSegment
        from src.translation.workflow import CandidateSpec, run_candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                workflow_mode="multimodal",
                translation_cache_dir=Path(temp_dir) / "cache",
                enable_cache=False,
            )
            glossary = load_character_glossary(config.character_names_csv)
            segment = SpreadSegment(
                index=0,
                spread_pages=(10, 11),
                page_numbers=(10, 11),
                page_texts=("First page.", "Second page."),
                previous_source_text="",
            )
            with patch(
                "src.translation.workflow.ask_external_translator_batch_with_cache",
                return_value=["First translated.", "Second translated."],
            ) as translator:
                candidate = run_candidate(
                    CandidateSpec(
                        name="google_translation",
                        provider="google",
                        model="google",
                        temperature=0.0,
                        stance="Literal.",
                    ),
                    segment.source_text,
                    "French",
                    "English",
                    glossary,
                    config,
                    ResponseCache(config.translation_cache_dir),
                    [segment],
                    None,
                )

        self.assertEqual("ok", candidate.status)
        self.assertEqual(
            "First translated.\n\nSecond translated.",
            candidate.text,
        )
        self.assertEqual(1, translator.call_count)
        self.assertEqual(
            ["First page.", "Second page."],
            translator.call_args.args[0],
        )

    def test_multimodal_google_falls_back_to_page_requests(self) -> None:
        from src.translation.cache import ResponseCache
        from src.translation.glossary import load_character_glossary
        from src.translation.segmentation import SpreadSegment
        from src.translation.workflow import CandidateSpec, run_candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TranslationWorkflowConfig(
                workflow_mode="multimodal",
                translation_cache_dir=Path(temp_dir) / "cache",
                enable_cache=False,
            )
            glossary = load_character_glossary(config.character_names_csv)
            segment = SpreadSegment(
                index=0,
                spread_pages=(10, 11),
                page_numbers=(10, 11),
                page_texts=("First page.", "Second page."),
                previous_source_text="",
            )
            with patch(
                "src.translation.workflow.ask_external_translator_batch_with_cache",
                side_effect=ValueError("Batch unavailable."),
            ):
                with patch(
                    "src.translation.workflow.ask_external_translator_with_cache",
                    side_effect=["First translated.", "Second translated."],
                ) as translator:
                    candidate = run_candidate(
                        CandidateSpec(
                            name="google_translation",
                            provider="google",
                            model="google",
                            temperature=0.0,
                            stance="Literal.",
                        ),
                        segment.source_text,
                        "French",
                        "English",
                        glossary,
                        config,
                        ResponseCache(config.translation_cache_dir),
                        [segment],
                        None,
                    )

        self.assertEqual("ok", candidate.status)
        self.assertEqual("First translated.\n\nSecond translated.", candidate.text)
        self.assertEqual(2, translator.call_count)


class PanelWorkflowTests(unittest.TestCase):
    def test_fake_provider_single_workflow_is_preserved(self) -> None:
        from src.translation.workflow import TranslationCandidate, build_application

        def fake_candidate(spec, text, source_language, language, glossary, config, cache, segments, segment_images, image_summaries):
            return TranslationCandidate(
                language=language,
                name=spec.name,
                provider=spec.provider,
                model=spec.model,
                temperature=spec.temperature,
                stance=spec.stance,
                text=f"{spec.name} translation.",
                status="ok",
                latency_seconds=0.0,
            )

        def fake_model(**kwargs):
            label = kwargs["label"]
            if label.endswith(" critic"):
                return json.dumps(
                    {
                        "overall_winner": "google_translation",
                        "ranking": ["google_translation", "gpt4o", "gpt5_5"],
                        "decision_reasoning": "Coverage.",
                        "paragraph_analysis": [],
                        "candidate_assessment": [],
                        "revision_instructions": [],
                        "concise_summary": "Keep coverage.",
                    }
                )
            if label.endswith("critic summary"):
                return "Keep coverage."
            if label.endswith("final translation"):
                return "Preserved single final."
            raise AssertionError(f"Unexpected fake model label: {label}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.txt"
            source_path.write_text("Source.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path,
                source_pdf_path=None,
                workflow_mode="text",
                evaluation_mode="single",
                translation_output_dir=root / "translation",
                translation_cache_dir=root / "cache",
                burr_storage_dir=str(root / "burr"),
                target_languages=["Finnish"],
            )
            with patch("src.translation.workflow.run_candidate", side_effect=fake_candidate):
                with patch(
                    "src.translation.workflow.ask_model_with_recovery",
                    side_effect=fake_model,
                ):
                    app, _ = build_application(config)
                    _, _, state = app.run(halt_after=["generate_final_text"])

            self.assertEqual(
                "Preserved single final.", state["final_translations"]["Finnish"]
            )

    def test_fake_provider_panel_workflow_reaches_final_translation(self) -> None:
        from src.translation.workflow import (
            TranslationCandidate,
            build_application,
        )

        def fake_candidate(spec, text, source_language, language, glossary, config, cache, segments, segment_images, image_summaries):
            return TranslationCandidate(
                language=language,
                name=spec.name,
                provider=spec.provider,
                model=spec.model,
                temperature=spec.temperature,
                stance=spec.stance,
                text=f"{spec.name} first.\n\n{spec.name} second.",
                status="ok",
                latency_seconds=0.0,
            )

        def fake_model(**kwargs):
            prompt = kwargs["prompt"]
            if "independent senior judge" in prompt:
                options_payload = prompt.split("Blinded options:\n", 1)[1].split(
                    "\n\nReturn valid JSON only:", 1
                )[0]
                options = list(json.loads(options_payload))
                payload = judge_result(
                    options,
                    {option: 10 - index for index, option in enumerate(options)},
                )
                return json.dumps(payload)
            if "Audit this complete" in prompt:
                return '{"findings": []}'
            if "Current source paragraph:\nFirst source." in prompt:
                return "Final first."
            if "Current source paragraph:\nSecond source." in prompt:
                return "Final second."
            raise AssertionError(f"Unexpected fake model prompt: {prompt[:80]}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.txt"
            source_path.write_text("First source.\n\nSecond source.", encoding="utf-8")
            config = TranslationWorkflowConfig(
                source_text_path=source_path,
                source_pdf_path=None,
                workflow_mode="text",
                translation_output_dir=root / "translation",
                translation_cache_dir=root / "cache",
                burr_storage_dir=str(root / "burr"),
                target_languages=["Finnish"],
                evaluation_mode="panel",
                panel_judges=["openai:a", "anthropic:b", "gemini:c"],
            )
            with patch("src.translation.workflow.run_candidate", side_effect=fake_candidate):
                with patch(
                    "src.translation.workflow.ask_model_with_recovery",
                    side_effect=fake_model,
                ):
                    app, _ = build_application(config)
                    _, _, state = app.run(halt_after=["repair_flagged_paragraphs"])

            self.assertEqual(
                "Final first.\n\nFinal second.",
                state["final_translations"]["Finnish"],
            )
            self.assertTrue(
                (config.panel_artifact_dir("Finnish") / "aggregates.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
