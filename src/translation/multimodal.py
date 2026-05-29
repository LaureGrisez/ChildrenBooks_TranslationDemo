"""PDF-derived spread rendering for multimodal translation mode."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .config import TranslationWorkflowConfig
from .segmentation import SpreadSegment


@dataclass(frozen=True, slots=True)
class SpreadImage:
    """One rendered spread image with metadata for prompts and caching."""

    segment_index: int
    spread_pages: tuple[int, ...]
    image_bytes: bytes
    mime_type: str = "image/png"

    def as_data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def collect_non_empty_body_pages(
    pdf_path: Path,
    skip_first: int,
    skip_last: int,
) -> list[int]:
    """Return 1-based page numbers of non-empty body pages."""

    with fitz.open(pdf_path) as document:
        start_page = min(skip_first, document.page_count)
        end_page = max(start_page, document.page_count - skip_last)
        page_numbers = []
        for page_index in range(start_page, end_page):
            if document.load_page(page_index).get_text("text").strip():
                page_numbers.append(page_index + 1)
    return page_numbers


def _redact_page_text(page: fitz.Page) -> None:
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue
            rect = fitz.Rect(line["bbox"])
            page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()


def _compose_spread_pixmap(
    document: fitz.Document,
    spread_pages: tuple[int, ...],
    dpi: int,
) -> bytes:
    pages = []
    for page_number in spread_pages:
        page_index = page_number - 1
        if not (0 <= page_index < document.page_count):
            continue
        temp_doc = fitz.open()
        temp_doc.insert_pdf(document, from_page=page_index, to_page=page_index)
        page = temp_doc.load_page(0)
        _redact_page_text(page)
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pages.append(pixmap)
        temp_doc.close()

    if not pages:
        raise ValueError(f"Unable to render spread for pages {spread_pages}.")

    total_width = sum(pix.width for pix in pages)
    max_height = max(pix.height for pix in pages)
    canvas = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, total_width, max_height), False)
    canvas.clear_with(255)

    offset_x = 0
    for pixmap in pages:
        canvas.copy(pixmap, fitz.IRect(offset_x, 0, offset_x + pixmap.width, pixmap.height))
        offset_x += pixmap.width

    return canvas.tobytes("png")


def render_spread_images(
    config: TranslationWorkflowConfig,
    segments: list[SpreadSegment],
) -> list[SpreadImage]:
    """Render one textless spread image per spread segment."""

    if config.source_pdf_path is None:
        raise ValueError("Multimodal mode requires a source PDF path.")

    spreads: list[SpreadImage] = []
    with fitz.open(config.source_pdf_path) as document:
        for segment in segments:
            image_bytes = _compose_spread_pixmap(
                document=document,
                spread_pages=segment.spread_pages,
                dpi=config.multimodal_image_dpi,
            )
            spreads.append(
                SpreadImage(
                    segment_index=segment.index,
                    spread_pages=segment.spread_pages,
                    image_bytes=image_bytes,
                )
            )
    return spreads
