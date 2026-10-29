from __future__ import annotations

from pathlib import Path

from memslides.tools import deck_runtime


def test_design_plan_accepts_suffix_numbered_and_cjk_heading_variants() -> None:
    content = """# 1. Design Plan（开源复现实验）

## 1. Design Goal (audience and task)
Readable conference-style slides for paper reproduction.

## Theme Keywords（风格关键词）
Research, precise, structured.

## Color Palette (semantic roles)
Background #ffffff, text #111827, accent #0d9488.

## Typography - projector readable
Title 36px, body 22px, caption 15px.

## Spacing & Grid（10页执行蓝图）
Use 64px page margins and two-column content grids.

## Page Archetypes (cover/content/ending)
Cover, method, result, revision, ending pages.

## Component Rules - tables / charts / diagrams
Tables use three-line style; diagrams use structured visual assets.

## Do / Don't (conversion-safe)
Do keep native text; don't show debug repair notes.
"""

    validation = deck_runtime._validate_structured_design_plan_markdown(content)

    assert validation["valid"] is True
    assert validation["missing_sections"] == []
    assert validation["empty_sections"] == []
    assert validation["title_present"] is True


def test_design_plan_still_rejects_missing_or_empty_required_sections() -> None:
    missing = """# Design Plan

## Design Goal
Goal text.

## Theme Keywords
Precise.
"""
    empty = """# Design Plan

## Design Goal
Goal text.

## Theme Keywords
Precise.

## Color Palette

## Typography
Readable.

## Spacing & Grid
Margins.

## Page Archetypes
Cover/content.

## Component Rules
Tables/charts.

## Do / Don't
No debug text.
"""

    missing_validation = deck_runtime._validate_structured_design_plan_markdown(missing)
    empty_validation = deck_runtime._validate_structured_design_plan_markdown(empty)

    assert missing_validation["valid"] is False
    assert "Color Palette" in missing_validation["missing_sections"]
    assert empty_validation["valid"] is False
    assert "Color Palette" in empty_validation["empty_sections"]


def test_visible_safe_zone_note_is_warning_but_debug_repair_text_is_error(tmp_path: Path) -> None:
    safe_zone_html = "<html><body><p>保持底部安全区，底部 >= 48px</p></body></html>"
    debug_html = "<html><body><p>Layout repaired deterministically for export canvas.</p></body></html>"

    safe_diagnostics, _ = deck_runtime._collect_visual_quality_diagnostics(
        tmp_path / "slide_01.html",
        safe_zone_html,
    )
    debug_diagnostics, _ = deck_runtime._collect_visual_quality_diagnostics(
        tmp_path / "slide_02.html",
        debug_html,
    )

    safe = {item["code"]: item["severity"] for item in safe_diagnostics}
    debug = {item["code"]: item["severity"] for item in debug_diagnostics}
    assert safe["visible_safe_zone_note"] == "warning"
    assert debug["visible_layout_repair_note"] == "error"
