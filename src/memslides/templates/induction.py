from __future__ import annotations

from pathlib import Path

from memslides.memory.extract.template_analyzer import TemplateAnalyzer, TemplateAnalysis


async def induct_template(
    template_file: Path | str,
    *,
    output_dir: Path | str | None = None,
    workspace: Path | str | None = None,
    language_model=None,
    vision_model=None,
) -> TemplateAnalysis:
    analyzer = TemplateAnalyzer(
        workspace=workspace,
        language_model=language_model,
        vision_model=vision_model,
    )
    return await analyzer.analyze(template_file, output_dir=output_dir)


__all__ = ["induct_template", "TemplateAnalyzer", "TemplateAnalysis"]
