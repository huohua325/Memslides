from __future__ import annotations

from pathlib import Path
from typing import Literal

from memslides.utils.webview import (
    PlaywrightConverter,
    convert_html_to_pptx_with_retry,
)


class ExportPipeline:
    """HTML slide export backend for PPTX/PDF/preview artifacts."""

    async def to_pptx(
        self,
        html_inputs: Path | list[Path] | list[str],
        output_pptx: Path,
        aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
    ) -> Path:
        return await convert_html_to_pptx_with_retry(
            html_inputs,
            output_pptx,
            aspect_ratio,
        )

    async def to_pdf(
        self,
        html_files: list[Path] | list[str],
        output_pdf: Path,
        aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
    ) -> Path:
        async with PlaywrightConverter() as converter:
            await converter.convert_to_pdf(html_files, output_pdf, aspect_ratio)
        return output_pdf
