"""Unit tests for the unified LiteLLM provider adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.translation.cache import ResponseCache
from src.translation.config import TranslationWorkflowConfig
from src.translation.providers import (
    _ask_provider,
    ask_model_with_recovery,
    litellm_model_name,
)


def completion_response(text: str = "Translated.") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class LiteLLMProviderTests(unittest.TestCase):
    def test_qualifies_models_for_each_provider(self) -> None:
        cases = {
            ("openai", "gpt-4o"): "openai/gpt-4o",
            ("anthropic", "claude-sonnet-4-6"): "anthropic/claude-sonnet-4-6",
            ("gemini", "gemini-2.5-flash"): "gemini/gemini-2.5-flash",
            ("zai", "glm-4.5"): "zai/glm-4.5",
            ("huggingface", "org/model"): "huggingface/org/model",
            ("minimax", "MiniMax-M2.1"): "minimax/MiniMax-M2.1",
        }
        for arguments, expected in cases.items():
            with self.subTest(provider=arguments[0]):
                self.assertEqual(expected, litellm_model_name(*arguments))

    def test_does_not_duplicate_existing_provider_prefix(self) -> None:
        self.assertEqual(
            "anthropic/claude-test",
            litellm_model_name("anthropic", "anthropic/claude-test"),
        )

    def test_all_providers_use_the_same_completion_call(self) -> None:
        with patch(
            "src.translation.providers.litellm.completion",
            return_value=completion_response(),
        ) as completion:
            for provider, model in (
                ("openai", "gpt-4o"),
                ("anthropic", "claude-test"),
                ("gemini", "gemini-test"),
            ):
                with self.subTest(provider=provider):
                    self.assertEqual(
                        "Translated.",
                        _ask_provider(provider, model, 0.2, "Translate this."),
                    )

        self.assertEqual(3, completion.call_count)
        self.assertEqual(
            ["openai/gpt-4o", "anthropic/claude-test", "gemini/gemini-test"],
            [call.kwargs["model"] for call in completion.call_args_list],
        )

    def test_multimodal_content_uses_provider_neutral_image_format(self) -> None:
        image_url = "data:image/jpeg;base64,YWJj"
        with patch(
            "src.translation.providers.litellm.completion",
            return_value=completion_response("Visual translation."),
        ) as completion:
            result = _ask_provider(
                "anthropic",
                "claude-test",
                0.2,
                "Translate this spread.",
                image_data_url=image_url,
            )

        self.assertEqual("Visual translation.", result)
        content = completion.call_args.kwargs["messages"][0]["content"]
        self.assertEqual({"type": "text", "text": "Translate this spread."}, content[0])
        self.assertEqual(image_url, content[1]["image_url"]["url"])

    def test_reasoning_openai_models_omit_temperature(self) -> None:
        with patch(
            "src.translation.providers.litellm.completion",
            return_value=completion_response(),
        ) as completion:
            _ask_provider("openai", "gpt-5.5", 0.7, "Translate this.")

        self.assertNotIn("temperature", completion.call_args.kwargs)

    def test_shared_recovery_retries_transient_provider_failures(self) -> None:
        config = TranslationWorkflowConfig(
            enable_cache=False,
            openai_retry_attempts=2,
            openai_retry_base_delay_seconds=0,
        )
        with patch(
            "src.translation.providers._ask_provider",
            side_effect=[TimeoutError("temporary"), "Recovered."],
        ) as ask_provider:
            result = ask_model_with_recovery(
                provider="minimax",
                model="MiniMax-M2.1",
                temperature=0.2,
                prompt="Translate this.",
                config=config,
                cache=ResponseCache(config.translation_cache_dir),
                label="test",
            )

        self.assertEqual("Recovered.", result)
        self.assertEqual(2, ask_provider.call_count)


if __name__ == "__main__":
    unittest.main()
