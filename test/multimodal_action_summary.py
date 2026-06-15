"""Run a small multimodal evaluation for one double-page image.

This compares one or more OpenAI models on the same image-understanding prompt
and writes a Markdown report for easy side-by-side review.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.translation.cache import ResponseCache
from src.translation.config import TranslationWorkflowConfig
from src.translation.multimodal import render_spread_image_bytes

DEFAULT_SYSTEM_PROMPT_PATH = (
    REPO_ROOT / "test" / "prompts" / "multimodal_action_summary_system_prompt.txt"
)
DEFAULT_USER_PROMPT = (
    "Summarize the visible action in this double-page spread."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multimodal image understanding on one double-page spread "
            "using one or more OpenAI models."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--image",
        type=Path,
        help="Path to an existing spread image (PNG/JPEG/WebP).",
    )
    source_group.add_argument(
        "--spread-pages",
        help="Comma-separated PDF page numbers to render, for example 10,11.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help=(
            "Source PDF to render when using --spread-pages. "
            "Defaults to the configured source PDF."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help=(
            "Rendering DPI when using --spread-pages. "
            "Defaults to MULTIMODAL_IMAGE_DPI or the config default."
        ),
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high"),
        default="low",
        help="OpenAI image detail setting. Default: low.",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated OpenAI models to test. "
            "Defaults to OPENAI_BASE_MODEL and OPENAI_ADVERSARIAL_MODEL."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_USER_PROMPT,
        help="Short user instruction sent alongside the image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for models that support it. Default: 0.2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown report output path. Defaults under translation/_multimodal_debug/evals/.",
    )
    parser.add_argument(
        "--save-input-image",
        action="store_true",
        help="Copy the exact evaluated input image next to the report for reproducibility.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the local disk cache for this evaluation.",
    )
    return parser.parse_args()


def supports_custom_temperature(model: str) -> bool:
    """Return whether the model accepts a custom temperature."""

    normalized = model.lower()
    return not (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def parse_models(raw_models: str | None, config: TranslationWorkflowConfig) -> list[str]:
    """Resolve the model list for the evaluation."""

    if raw_models:
        models = [item.strip() for item in raw_models.split(",") if item.strip()]
    else:
        models = [config.openai_base_model, config.openai_adversarial_model]

    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def parse_spread_pages(raw_pages: str) -> tuple[int, ...]:
    """Parse a comma-separated spread page list."""

    pages = tuple(int(part.strip()) for part in raw_pages.split(",") if part.strip())
    if not pages:
        raise ValueError("--spread-pages must contain at least one page number.")
    return pages


def detect_mime_type(path: Path) -> str:
    """Return the MIME type for an image path."""

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        return "image/png"
    return mime_type


def build_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Encode image bytes as a data URL."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_image_input(
    args: argparse.Namespace,
    config: TranslationWorkflowConfig,
) -> tuple[bytes, str, str]:
    """Load or render the evaluated image and return bytes, MIME type, and a label."""

    if args.image is not None:
        image_path = args.image
        image_bytes = image_path.read_bytes()
        mime_type = detect_mime_type(image_path)
        label = str(image_path)
        return image_bytes, mime_type, label

    pdf_path = args.pdf or config.source_pdf_path
    if pdf_path is None:
        raise ValueError("No source PDF is configured for --spread-pages.")
    spread_pages = parse_spread_pages(args.spread_pages)
    dpi = args.dpi or config.multimodal_image_dpi
    image_bytes = render_spread_image_bytes(
        pdf_path=pdf_path,
        spread_pages=spread_pages,
        dpi=dpi,
    )
    label = f"{pdf_path} pages {','.join(str(page) for page in spread_pages)} @ {dpi} dpi"
    return image_bytes, "image/png", label


def default_output_path(config: TranslationWorkflowConfig) -> Path:
    """Build a default report path."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        config.translation_output_dir
        / "_multimodal_debug"
        / "evals"
        / f"{timestamp}_action_summary.md"
    )


def ask_openai_multimodal(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    image_detail: str,
    temperature: float,
) -> str:
    """Send one multimodal prompt to OpenAI."""

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": image_detail},
                    },
                ],
            },
        ],
    }
    if supports_custom_temperature(model):
        request["temperature"] = temperature

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def cached_or_live_response(
    client: OpenAI,
    cache: ResponseCache | None,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    image_detail: str,
    temperature: float,
) -> str:
    """Use the local disk cache when enabled, otherwise hit the API directly."""

    cache_prompt = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_data_url": image_data_url,
            "image_detail": image_detail,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    if cache is not None:
        cached = cache.get(
            provider="openai_multimodal_eval",
            model=model,
            temperature=temperature,
            prompt=cache_prompt,
        )
        if cached is not None:
            return cached

    response = ask_openai_multimodal(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_data_url=image_data_url,
        image_detail=image_detail,
        temperature=temperature,
    )

    if cache is not None:
        cache.set(
            provider="openai_multimodal_eval",
            model=model,
            temperature=temperature,
            prompt=cache_prompt,
            response=response,
            metadata={"image_detail": image_detail},
        )
    return response


def build_report(
    *,
    image_source_label: str,
    models: list[str],
    system_prompt_path: Path,
    system_prompt: str,
    user_prompt: str,
    image_detail: str,
    dpi: int | None,
    results: list[tuple[str, str]],
) -> str:
    """Build the Markdown report."""

    lines = [
        "# Multimodal Action Summary Evaluation",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Image source: `{image_source_label}`",
        f"- Models: `{', '.join(models)}`",
        f"- OpenAI image detail: `{image_detail}`",
        f"- Render DPI: `{dpi if dpi is not None else 'n/a (pre-rendered image)'}`",
        f"- System prompt file: `{system_prompt_path}`",
        "",
        "## User Prompt",
        "",
        user_prompt,
        "",
        "## System Prompt",
        "",
        "```text",
        system_prompt,
        "```",
        "",
    ]

    for model, response in results:
        lines.extend(
            [
                f"## {model}",
                "",
                response.strip() or "(empty response)",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Run the multimodal comparison."""

    load_dotenv()
    args = parse_args()
    config = TranslationWorkflowConfig.from_env()
    models = parse_models(args.models, config)
    output_path = args.output or default_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    image_bytes, mime_type, image_source_label = load_image_input(args, config)
    image_data_url = build_data_url(image_bytes, mime_type)
    cache = None if args.no_cache else ResponseCache(config.translation_cache_dir)
    client = OpenAI()

    results: list[tuple[str, str]] = []
    for model in models:
        response = cached_or_live_response(
            client,
            cache,
            model=model,
            system_prompt=system_prompt,
            user_prompt=args.user_prompt,
            image_data_url=image_data_url,
            image_detail=args.detail,
            temperature=args.temperature,
        )
        results.append((model, response))

    report = build_report(
        image_source_label=image_source_label,
        models=models,
        system_prompt_path=args.system_prompt_file,
        system_prompt=system_prompt,
        user_prompt=args.user_prompt,
        image_detail=args.detail,
        dpi=args.dpi if args.spread_pages else None,
        results=results,
    )
    output_path.write_text(report, encoding="utf-8")

    if args.save_input_image:
        image_suffix = ".png" if mime_type == "image/png" else ".jpg"
        image_copy_path = output_path.with_suffix(image_suffix)
        image_copy_path.write_bytes(image_bytes)

    print(output_path)


if __name__ == "__main__":
    main()
