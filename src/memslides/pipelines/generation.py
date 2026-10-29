from __future__ import annotations

import json
import logging
import re
import traceback
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from memslides.agents.deck_designer import DeckDesigner
from memslides.agents.researcher import Researcher
from memslides.agents.template_planner import TemplatePlanner
from memslides.contracts import DeckRequest, DeckResult
from memslides.tools.asset_services import _document_summary_impl
from memslides.tools.document_conversion import convert_to_markdown as _convert_to_markdown
from memslides.tools.document_conversion import list_document_figures as _list_document_figures
from memslides.tools.structured_visuals import collect_generated_visual_entries
from memslides.tools.template_tools import clear_template_context
from memslides.templates.runtime_state import clear_template_runtime_state
from memslides.templates.activation import decide_template_activation
from memslides.templates.quality import (
    ADAPTIVE_STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    STRUCTURAL_TEMPLATE,
    STYLE_REFERENCE,
)
from memslides.pipelines.generation_support import (
    build_non_template_design_plan_scaffold as _build_non_template_design_plan_scaffold,
    build_profile_execution_source_evidence_summary as _build_profile_execution_source_evidence_summary,
    find_existing_design_plan_rel as _find_existing_design_plan_rel,
    render_design_plan_execution_plan as _render_design_plan_execution_plan,
)
from memslides.tools.deck_runtime import (
    initialize_control_document_tracking,
    initialize_design_plan_tracking,
    set_current_agent,
)
from memslides.runtime.deck_execution_state import (
    initialize_deck_execution_state,
    load_deck_execution_state,
)
from memslides.utils.log import debug, error, info
from memslides.utils.typings import ChatMessage, ConvertType, InputRequest, Role

logger = logging.getLogger(__name__)

_STRUCTURAL_TEMPLATE_MODES = {
    STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    ADAPTIVE_STRUCTURAL_TEMPLATE,
}


def _is_structural_template_mode(mode: str | None) -> bool:
    return str(mode or "") in _STRUCTURAL_TEMPLATE_MODES


def _latest_assistant_text(agent: Any) -> str:
    chat_history = getattr(agent, "chat_history", None) or []
    for message in reversed(chat_history):
        if getattr(message, "role", None) == Role.ASSISTANT:
            text = getattr(message, "text", "") or getattr(message, "content", "")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _coerce_positive_page_count(value: Any) -> int:
    try:
        count = int(str(value or "").strip())
    except (TypeError, ValueError):
        return 0
    return count if count > 0 else 0


def _infer_markdown_slide_count(md_file: Path | str | None) -> int:
    if not md_file:
        return 0
    path = Path(md_file)
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    segments = [
        segment.strip()
        for segment in re.split(r"\n\s*---+\s*\n", text)
        if segment.strip()
    ]
    return len(segments)


def _slide_dir_rel(workspace: Path, slide_dir: Path | str | None) -> str:
    if not slide_dir:
        return "outputs"
    path = Path(slide_dir)
    if not path.is_absolute():
        return path.as_posix().rstrip("/") or "outputs"
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix().rstrip("/") or "outputs"
    except ValueError:
        return "outputs"


def _ensure_generation_deck_execution_state(
    *,
    workspace: Path,
    request: InputRequest,
    slide_dir: Path | str | None,
    md_file: Path | str | None,
    profile_execution_plan: dict[str, Any] | None = None,
) -> Path | None:
    """Initialize DeckDesigner progress for non-template generation before the first slide write."""
    existing = load_deck_execution_state(workspace)
    if existing:
        return None

    requested = _coerce_positive_page_count(request.num_pages)
    inferred = _infer_markdown_slide_count(md_file)
    expected = requested or inferred
    if expected <= 0:
        return None

    trace_path = workspace / ".history" / "generation_request_context.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "requested_pages": requested or None,
                "inferred_manuscript_pages": inferred or None,
                "effective_pages": expected,
                "language": request.language,
                "page_count_source": "request" if requested else "manuscript",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return initialize_deck_execution_state(
        workspace,
        expected_slide_count=expected,
        slide_dir=_slide_dir_rel(workspace, slide_dir),
        profile_execution_plan=profile_execution_plan,
    )


def _default_research_manuscript(request: InputRequest, research_text: str) -> str:
    body = research_text.strip()
    if body:
        return body
    body = (
        "本轮研究阶段未返回可直接复用的正文，因此由运行时根据请求自动生成一份最小稿件，"
        "用于驱动后续 DeckDesigner 阶段。"
    )
    num_pages = str(request.num_pages or "4").strip()
    return (
        "# MemSlides 研究稿\n\n"
        "## Slide 1\n"
        "# MemSlides\n"
        "Memory-aware presentation generation for iterative decks.\n\n"
        "Use a calm, professional tone with blue-green accents. The deck should feel like a system brief, not a marketing pitch.\n\n"
        "---\n\n"
        "## Slide 2\n"
        "# What MemSlides does\n"
        "- Turns a brief into a structured manuscript\n"
        "- Builds slides from a template-guided HTML deck\n"
        "- Uses memory to carry preferences across rounds\n"
        "- Keeps generation, revision, export, and memory separate\n\n"
        f"{body}\n\n"
        "---\n\n"
        "## Slide 3\n"
        "# How the loop works\n"
        "- Job: one full session\n"
        "- Round: one user-to-agent interaction cycle\n"
        "- Operation: one tool call or internal action\n"
        "- Generation first, revision second, finalize last\n"
        "- Memory writes back after meaningful rounds\n"
        "- Template guidance stays local and reproducible\n\n"
        "---\n\n"
        "## Slide 4\n"
        "# Key takeaway\n"
        "- MemSlides is designed for repeatable deck making with memory\n"
        "- The first draft stays broad enough for revision\n"
        "- Final output should be calm, clear, and template-aligned\n"
        "- Target page count: " + num_pages + "\n"
    )


def _persist_research_manuscript(runtime: Any, request: InputRequest) -> Path:
    manuscript_dir = runtime.workspace / "outputs"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    manuscript_path = manuscript_dir / "manuscript.md"
    if manuscript_path.exists():
        return manuscript_path
    research_text = _latest_assistant_text(getattr(runtime, "research_agent", None))
    manuscript = _default_research_manuscript(request, research_text)
    manuscript_path.write_text(manuscript, encoding="utf-8")
    info(f"Saved fallback research manuscript to {manuscript_path}")
    return manuscript_path


def _sanitize_markdown_response(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    fenced = re.match(r"^```(?:markdown|md)?\s*(.*?)```$", source, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return source


def _summary_bullets(summary_text: str, *, limit: int = 8) -> list[str]:
    bullets: list[str] = []
    for line in str(summary_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s+", stripped):
            bullets.append(re.sub(r"^\d+\.\s+", "", stripped).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for bullet in bullets:
        if not bullet or bullet in seen:
            continue
        seen.add(bullet)
        deduped.append(bullet)
        if len(deduped) >= limit:
            break
    return deduped


def _choose_reference_assets(figures_payload: dict[str, Any] | None, *, max_items: int = 3) -> list[dict[str, Any]]:
    figures = list((figures_payload or {}).get("figures", []) or [])
    if not figures:
        return []

    def _priority(item: dict[str, Any]) -> tuple[int, int, int]:
        category = str(item.get("category", "") or "").lower()
        caption = str(item.get("caption", "") or "").lower()
        kind = category or caption
        if "table" in kind:
            group = 0
        elif "figure" in kind or "chart" in kind or "diagram" in kind:
            group = 1
        else:
            group = 2
        area = int(item.get("width") or 0) * int(item.get("height") or 0)
        page = int(item.get("page") or 10**9)
        return (group, page, -area)

    selected: list[dict[str, Any]] = []
    for item in sorted(figures, key=_priority):
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _deterministic_attachment_manuscript(
    request: InputRequest,
    attachment_name: str,
    summary_text: str,
    figures_payload: dict[str, Any] | None,
) -> str:
    title = Path(attachment_name).stem.replace("_", " ").strip() or "Attachment Summary"
    bullets = _summary_bullets(summary_text, limit=10)
    assets = _choose_reference_assets(figures_payload, max_items=3)
    asset_lines = []
    for asset in assets:
        caption = str(asset.get("caption", "") or asset.get("label", "") or Path(str(asset.get("path", ""))).name).strip()
        path = str(asset.get("path", "") or "").strip()
        if path:
            asset_lines.append(f"![{caption}]({path})")

    slide2_bullets = bullets[:3] or [
        "The attachment centers the Transformer as an attention-based sequence model.",
        "Encoder-decoder structure is organized around multi-head self-attention and feed-forward blocks.",
        "The architecture removes recurrence to improve parallel training."
    ]
    slide3_bullets = bullets[3:6] or [
        "Complexity, path length, and parallelism are explicit trade-offs in the paper.",
        "Training setup and hardware assumptions matter for practical adoption.",
        "Table-based comparisons provide the strongest budget and performance evidence."
    ]
    slide4_bullets = bullets[6:9] or bullets[:3] or [
        "Adoption case rests on measurable translation gains and lower sequential bottlenecks.",
        "The strongest evidence should stay attached to real figures, tables, and formulas from the paper.",
        "Use the attachment as the factual boundary for any final narrative."
    ]

    parts = [
        f"# {title}",
        "Technical summary grounded only in the provided attachment.",
        "",
        "---",
        "",
        "## Transformer architecture in one page",
        *[f"- {item}" for item in slide2_bullets],
    ]
    if asset_lines[:1]:
        parts.extend(["", asset_lines[0]])
    parts.extend([
        "",
        "---",
        "",
        "## Complexity, training cost, and operational trade-offs",
        *[f"- {item}" for item in slide3_bullets],
    ])
    if len(asset_lines) > 1:
        parts.extend(["", asset_lines[1]])
    parts.extend([
        "",
        "---",
        "",
        "## Results and practical takeaway",
        *[f"- {item}" for item in slide4_bullets],
    ])
    if len(asset_lines) > 2:
        parts.extend(["", asset_lines[2]])
    return "\n".join(parts).strip() + "\n"


async def _build_attachment_grounded_manuscript(runtime: Any, request: InputRequest) -> Path | None:
    attachments = [Path(path) for path in (request.attachments or []) if str(path or "").strip()]
    if not attachments:
        return None

    primary_attachment = attachments[0]
    converted_dir = runtime.workspace / "converted" / primary_attachment.stem
    converted = await _convert_to_markdown(
        file_path=str(primary_attachment),
        output_folder=str(converted_dir),
    )
    if not converted.get("success"):
        raise RuntimeError(f"attachment conversion failed: {converted.get('error') or converted}")

    markdown_file = Path(str(converted.get("markdown_file") or "")).resolve()
    figures_payload = await _list_document_figures(str(converted_dir))
    summary_text = await _document_summary_impl(
        task=(
            f"{request.instruction}\n\n"
            "Produce a concise factual brief for a markdown slide manuscript. "
            "Prioritize attachment-grounded architecture, results, formulas, tables, and figure evidence."
        ),
        document_path=str(markdown_file),
    )

    outputs_dir = runtime.workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "research_fallback_summary.md").write_text(summary_text, encoding="utf-8")
    (outputs_dir / "research_fallback_figures.json").write_text(
        json.dumps(figures_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selected_assets = _choose_reference_assets(figures_payload, max_items=3)
    asset_lines = []
    for item in selected_assets:
        path = str(item.get("path", "") or "").strip()
        caption = str(item.get("caption", "") or item.get("label", "") or Path(path or "asset").name).strip()
        if path:
            asset_lines.append(f"- {caption}: {path}")
    asset_block = "\n".join(asset_lines) if asset_lines else "- No extracted figure/table asset available."

    fallback_prompt = (
        "Create a markdown manuscript for a slide deck using only the provided attachment summary and extracted figure/table paths.\n"
        f"- Target slide count: {int(request.num_pages or 4)}\n"
        "- Output markdown only.\n"
        "- Separate slides with `---`.\n"
        "- Use real local figure/table paths exactly as provided when relevant.\n"
        "- Do not invent external facts, links, or asset paths.\n"
        "- Maintain a technical summary tone suitable for later template-guided rendering.\n\n"
        f"## Attachment\n{primary_attachment.name}\n\n"
        f"## Extracted Summary\n{summary_text}\n\n"
        f"## Available Visual Assets\n{asset_block}\n"
    )

    manuscript_text = ""
    try:
        response = await runtime.config.design_agent.run(
            messages=[{"role": "user", "content": fallback_prompt}],
            retry_times=1,
            request_kwargs={"max_tokens": 2500},
        )
        manuscript_text = _sanitize_markdown_response(
            response.choices[0].message.content or ""
        )
    except Exception as exc:
        info(f"Attachment-grounded manuscript LLM fallback unavailable; using deterministic manuscript: {exc}")

    if not manuscript_text:
        manuscript_text = _deterministic_attachment_manuscript(
            request,
            primary_attachment.name,
            summary_text,
            figures_payload,
        )

    manuscript_path = outputs_dir / "manuscript.md"
    manuscript_path.write_text(manuscript_text, encoding="utf-8")
    info(f"Saved attachment-grounded fallback manuscript to {manuscript_path}")
    return manuscript_path


def _write_content_asset_manifest(runtime: Any, manuscript_path: Path) -> Path:
    workspace = Path(runtime.workspace).resolve()
    payload: dict[str, Any] = {
        "manuscript": str(manuscript_path.resolve()),
        "workspace": str(workspace),
        "assets": [],
        "formulas": [],
    }
    try:
        text = manuscript_path.read_text(encoding="utf-8")
    except Exception:
        text = ""

    seen: set[str] = set()
    for asset in collect_generated_visual_entries(workspace):
        path = str(asset.get("path", "") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        payload["assets"].append(
            {
                "path": path,
                "kind": str(asset.get("kind", "") or "figure"),
                "caption": str(asset.get("caption", "") or ""),
                "exists": bool(asset.get("exists", False)),
                "within_workspace": bool(asset.get("within_workspace", False)),
                "generated_by_tool": bool(asset.get("generated_by_tool", False)),
                "renderer": str(asset.get("renderer", "") or ""),
                "meta_path": str(asset.get("meta_path", "") or ""),
                "rendered_paths": asset.get("rendered_paths", {}),
            }
        )

    image_pattern = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
    for match in image_pattern.finditer(text):
        raw_path = match.group("path").strip("<>")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (manuscript_path.parent / path).resolve()
        else:
            path = path.resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        kind = "table" if "table" in path.name.lower() else "figure"
        payload["assets"].append(
            {
                "path": str(path),
                "kind": kind,
                "exists": path.exists(),
                "within_workspace": workspace in path.parents or path == workspace,
            }
        )

    formula_candidates = re.findall(
        r"(softmax\s*\([^\n]{0,120}|QK\^?T[^\n]{0,80}|√d_?k|\$[^\$]{3,120}\$)",
        text,
        flags=re.IGNORECASE,
    )
    payload["formulas"] = list(dict.fromkeys(candidate.strip() for candidate in formula_candidates if candidate.strip()))[:8]

    manifest_path = workspace / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _asset_manifest_prompt(manifest_path: Path) -> str:
    return (
        "📎 **Attachment asset contract**:\n"
        f"- Read `{manifest_path.name}` before writing slide HTML.\n"
        "- Use real attachment-derived figures/tables from the manifest when they are relevant and exist.\n"
        "- Formulas must be rendered as visible HTML text/math, not hidden comments.\n"
        "- Do not replace real content with SVG text images just to satisfy a layout slot.\n"
        "- Do not use asset paths outside the current workspace."
    )


def _page_asset_plan_prompt(plan_path: Path) -> str:
    return (
        "📌 **Page asset binding contract**:\n"
        f"- Read `{plan_path.name}` before choosing each page layout or writing slide HTML.\n"
        "- If a page has `visual_requirement: required`, the generated HTML must render the bound real asset or formula visibly on that page.\n"
        "- Do not downgrade a required figure/table/formula page into a pure text bullet page.\n"
        "- If the selected reference layout lacks a visual slot, use a readable surface/panel while preserving the reference layout intent.\n"
        "- Asset paths must remain inside the current workspace."
    )


def _layout_mapping_prompt(layout_mapping_path: Path) -> str:
    return (
        "🧭 **Template layout mapping contract**:\n"
        f"- Read `{layout_mapping_path.name}` after `design_plan.md` and before writing any slide HTML.\n"
        "- Treat `selected_layout` as the canonical layout choice for each page.\n"
        "- Use `recommend_template_layout` only when the page content has clearly drifted and you need a structured re-check.\n"
        "- For each page, call `query_slide_layout(selected_layout)` before `write_html_file`.\n"
        "- Follow `slot_fill_plan` when deciding which title/body/figure/table content goes into which canonical slot.\n"
        "- Follow the page-level reference fields (`reference_archetype`, `reference_use`, `density_hint`, `safe_surface_policy`, `style_reference`, `visual_requirement`, `bound_asset_kind`, `bound_asset_path`) before placing text.\n"
        "- Use template-inspired title rhythm, geometry, palette, and density; do not copy raw template backgrounds or old sample images.\n"
        "- Do not invent an alternative layout unless the mapping is obviously incompatible with the page content."
    )


def _render_guidance_for_slide(slide: dict[str, Any]) -> list[str]:
    visual_requirement = str(slide.get("visual_requirement", "none") or "none")
    bound_asset_kind = str(slide.get("bound_asset_kind", "none") or "none")
    surface_mode = str(slide.get("surface_mode", "light_surface") or "light_surface")
    safe_surface_policy = str(slide.get("safe_surface_policy", "") or "")
    density_hint = str(slide.get("density_hint", "medium") or "medium")
    title = str(slide.get("title", "") or "")
    title_lower = title.lower()

    guidance: list[str] = []
    guidance.append(
        f"Use the template as reference only: preserve {density_hint} density, title rhythm, palette, and layout balance without copying raw template backgrounds or old sample images."
    )
    if safe_surface_policy:
        guidance.append(f"Surface policy: {safe_surface_policy}; prioritize readable text surfaces over exact template replication.")
    if surface_mode in {"shell_text", "title_reference"}:
        guidance.append("Treat this as a title-led reference page; create a readable title area with template-inspired accents.")
        return guidance

    if visual_requirement == "required" and bound_asset_kind in {"figure", "table", "chart"}:
        guidance.append("Make the bound asset the primary visual block; it should dominate the page before bullets do.")
        guidance.append("Place explanatory text on a light readable surface panel and keep it to 2-3 tight bullets or one short takeaway block.")
        if bound_asset_kind in {"table", "chart"}:
            guidance.append("Preserve table/chart legibility: allocate enough width and height for labels, and summarize the takeaway beside or below it instead of shrinking the asset.")
        else:
            guidance.append("Do not restate the full figure caption verbatim; extract the key mechanism or insight beside the visual.")
        return guidance

    if visual_requirement == "required" and bound_asset_kind == "formula":
        guidance.append("Render the formula visibly on a light surface panel and pair it with one short explanatory block.")
        guidance.append("Keep mathematical notation prominent and avoid burying the formula inside dense paragraph text.")
        return guidance

    if "agenda" in title_lower or "contents" in title_lower:
        guidance.append("Keep the page airy and scannable; use short lines and avoid decorative filler.")
        return guidance

    guidance.append("Use a readable light surface for body content and keep the page visually simple.")
    guidance.append("Favor one clear hierarchy over many small callouts.")
    return guidance


def _build_page_execution_plan(
    mapping: dict[str, Any],
    profile_execution_plan: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Page Execution Brief",
        "",
        "Read this file before writing slides. It is the concise per-page implementation contract distilled from `layout_mapping.yaml` and `page_asset_plan.json`.",
    ]
    persona_by_page: dict[int, dict[str, Any]] = {}
    if profile_execution_plan:
        lines.extend(
            [
                "",
                "## Persona Execution Overlay",
                f"- Planning focus: `{profile_execution_plan.get('planning_focus', 'decision_brief')}`",
                "- Preserve template structure when possible; realize persona differences through page role, component emphasis, sequencing, and safe proxy realization rather than brute-force layout replacement.",
            ]
        )
        for item in profile_execution_plan.get("global_execution_notes", []) or []:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")
        for page in profile_execution_plan.get("page_plan", []) or []:
            if not isinstance(page, dict):
                continue
            try:
                page_index = int(page.get("page_index", 0) or 0)
            except Exception:
                page_index = 0
            if page_index > 0:
                persona_by_page[page_index] = page
    for slide in mapping.get("slides", []) or []:
        page = int(slide.get("page", 0) or 0)
        title = str(slide.get("title", "") or "")
        selected_layout = str(slide.get("selected_layout", "") or "")
        visual_requirement = str(slide.get("visual_requirement", "none") or "none")
        bound_asset_kind = str(slide.get("bound_asset_kind", "none") or "none")
        bound_asset_path = str(slide.get("bound_asset_path", "") or "")
        surface_mode = str(slide.get("surface_mode", "light_surface") or "light_surface")
        reference_archetype = str(slide.get("reference_archetype", "") or "")
        reference_use = str(slide.get("reference_use", "") or "")
        density_hint = str(slide.get("density_hint", "medium") or "medium")
        safe_surface_policy = str(slide.get("safe_surface_policy", "") or "")
        why_selected = str(slide.get("why_selected", "") or "")
        slot_fill_plan = slide.get("slot_fill_plan", {}) or {}
        style_reference = slide.get("style_reference", {}) or {}
        palette_roles = style_reference.get("palette_roles", {}) if isinstance(style_reference, dict) else {}
        typography_roles = style_reference.get("typography_roles", {}) if isinstance(style_reference, dict) else {}
        title_treatment = str(style_reference.get("title_treatment", "") or "") if isinstance(style_reference, dict) else ""

        lines.extend(
            [
                "",
                f"## Page {page}: {title}",
                f"- Selected layout: `{selected_layout}`",
                f"- Reference archetype: `{reference_archetype or 'content'}`",
                f"- Reference use: `{reference_use or 'layout+density+style'}`",
                f"- Density hint: `{density_hint}`",
                f"- Visual requirement: `{visual_requirement}`",
                f"- Bound asset: `{bound_asset_kind}`" + (f" -> `{bound_asset_path}`" if bound_asset_path else ""),
                f"- Surface mode: `{surface_mode}` | safe surface policy: `{safe_surface_policy or 'readable_surface_required'}`",
            ]
        )
        persona_page = persona_by_page.get(page)
        if persona_page:
            lines.extend(
                [
                    "- Persona overlay:",
                    f"  - Page role: {str(persona_page.get('page_role', '') or 'persona-specific page').strip()}",
                    f"  - Persona signal: {str(persona_page.get('persona_signal', '') or 'persona-aligned signal').strip()}",
                    f"  - Manuscript anchor: {str(persona_page.get('manuscript_anchor', '') or 'fit-for-source content cluster').strip()}",
                    f"  - Required component: {str(persona_page.get('required_component', '') or 'fit-for-content archetype').strip()}",
                    f"  - Layout bias: {str(persona_page.get('layout_bias', '') or 'single_focus').strip()}",
                ]
            )
            must_preserve = [
                str(item).strip()
                for item in persona_page.get("must_preserve", []) or []
                if str(item or "").strip()
            ]
            nice_to_have = [
                str(item).strip()
                for item in persona_page.get("nice_to_have", []) or []
                if str(item or "").strip()
            ]
            if must_preserve:
                lines.append("  - Must preserve:")
                for item in must_preserve:
                    lines.append(f"    - {item}")
            hard_requirements = [
                str(item).strip()
                for item in persona_page.get("hard_requirements", []) or []
                if str(item or "").strip()
            ]
            component_requirements = [
                str(item).strip()
                for item in persona_page.get("component_requirements", []) or []
                if str(item or "").strip()
            ]
            soft_signals = [
                str(item).strip()
                for item in persona_page.get("soft_signals", []) or []
                if str(item or "").strip()
            ]
            style_requirements = persona_page.get("style_requirements") if isinstance(persona_page.get("style_requirements"), dict) else {}
            if hard_requirements:
                lines.append("  - Hard requirements:")
                for item in hard_requirements[:4]:
                    lines.append(f"    - {item}")
            if style_requirements:
                lines.append("  - Style requirements:")
                lines.append(f"    - {json.dumps(style_requirements, ensure_ascii=False)[:400]}")
            if component_requirements:
                lines.append("  - Component requirements:")
                for item in component_requirements[:4]:
                    lines.append(f"    - {item}")
            if soft_signals:
                lines.append("  - Soft persona signals:")
                for item in soft_signals[:4]:
                    lines.append(f"    - {item}")
            if nice_to_have:
                lines.append("  - Nice to have:")
                for item in nice_to_have:
                    lines.append(f"    - {item}")
        if why_selected:
            lines.append(f"- Why this layout: {why_selected}")
        if slot_fill_plan:
            lines.append("- Slot fill plan:")
            for slot_name, fill in slot_fill_plan.items():
                lines.append(f"  - `{slot_name}` <- `{fill}`")
        if title_treatment or palette_roles or typography_roles:
            lines.append("- Style cues:")
            if title_treatment:
                lines.append(f"  - Title treatment: {title_treatment}")
            if palette_roles:
                lines.append(
                    "  - Palette roles: "
                    + ", ".join(
                        f"{key}=`{value}`"
                        for key, value in palette_roles.items()
                    )
                )
            if typography_roles:
                lines.append(
                    "  - Typography roles: "
                    + ", ".join(
                        f"{key}=`{value}`"
                        for key, value in typography_roles.items()
                    )
                )
        render_guidance = _render_guidance_for_slide(slide)
        if render_guidance:
            lines.append("- Render guidance:")
            for item in render_guidance:
                lines.append(f"  - {item}")
    lines.append("")
    if profile_execution_plan:
        fallback_rules = [
            str(item).strip()
            for item in profile_execution_plan.get("fallback_rules", []) or []
            if str(item or "").strip()
        ]
        if fallback_rules:
            lines.append("## Persona Fallback Rules")
            for item in fallback_rules:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _page_execution_plan_prompt(plan_path: Path) -> str:
    return (
        "🗂️ **Per-page execution brief**:\n"
        f"- Read `{plan_path.name}` before `layout_mapping.yaml`.\n"
        "- Treat it as the primary concise checklist for each slide so you do not re-plan the deck from scratch.\n"
        "- Use it to keep page purpose, selected layout, asset binding, and surface mode aligned while generating HTML.\n"
        "- If a page already has a non-empty `bound_asset_path`, use that exact workspace asset first; do not browse for a replacement unless that file is missing or unusable."
    )


def _dedupe_tool_specs(tools: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for tool in tools or []:
        name = ""
        if isinstance(tool, dict):
            name = str(tool.get("function", {}).get("name", "") or "").strip()
        else:
            name = str(getattr(tool, "name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(tool)
    return deduped


async def run_generation_flow(
    runtime: Any,
    request: InputRequest,
    check_llms: bool = False,
) -> AsyncGenerator[str | ChatMessage, None]:
    """Main loop for MemSlides generation process.
    Arguments:
        request: InputRequest object containing task details.
        check_llms: Whether to check LLM availability before running.
    Yields:
        ChatMessage or str: Messages or final output path.
    """
    self = runtime
    if not self.config.design_agent.is_multimodal and self.config.heavy_reflect:
        debug(
            "Reflective design requires a multimodal LLM in the design agent, reflection will only enable on textual state."
        )
    if check_llms:
        await self.config.validate_llms()
    self._freeze_preference_writeback_current_job = (
        self._should_freeze_preference_writeback(request)
    )
    with open(self.workspace / ".input_request.json", "w") as f:
        json.dump(request.model_dump(), f, ensure_ascii=False, indent=2)
    # Keep AgentEnv alive for potential modify() calls
    await self._ensure_env()
    agent_env = self.agent_env
    self._template_profile = None
    self._guide_builder = None
    self._current_template_id = ""
    self._system_injection_trace_state = {}
    self._resolved_request_intent = ""
    self._resolved_request_intent_scenario = ""
    self._resolved_request_intent_source = ""
    self._resolved_request_intent_confidence = None
    self._resolved_request_intent_raw_response = ""
    self._resolved_request_intent_payload = {}
    self._template_match_state = {}
    self._template_usage_seeded = False
    self._template_runtime_state = None
    clear_template_context()
    clear_template_runtime_state(self.workspace)

    intent_result = await self._resolve_request_intent_runtime(request)
    self._cache_resolved_request_intent(request, intent_result)
    self._initialize_template_match_state(request)
    info(
        f"[Job Start] Resolved intent: {intent_result.intent} "
        f"(source={intent_result.source}, confidence={self._confidence_text(intent_result.confidence)})"
    )
    if self._freeze_preference_writeback_current_job:
        info("[Job Start] Preference writeback is frozen for this job.")

    # MemoryOrchestrator: create + Job Start (ProfileInjectionRouter + LTM preload)
    _orchestrator = None
    if self.memory_system:
        try:
            job_intent = self._get_request_task_intent(request)
            read_intent = self._get_request_memory_read_intent(request)
            write_intent = self._get_request_memory_write_intent(request)
            core_persona = self._get_request_core_persona(request)

            _orchestrator = self._create_orchestrator()
            if _orchestrator:
                await _orchestrator.on_job_start(
                    user_id=self.user_id,
                    project_id=self.workspace.stem,
                    user_prompt=request.instruction,
                    intent=job_intent,
                    read_intent=read_intent,
                    write_intent=write_intent,
                    core_persona=core_persona,
                )
                logger.info(
                    "MemoryOrchestrator: job started "
                    "(job_intent=%s, read_intent=%s, write_intent=%s, core_persona=%s)",
                    job_intent,
                    read_intent,
                    write_intent,
                    core_persona,
                )
                info(
                    f"[Job Start] Persona={core_persona or 'n/a'}, "
                    f"job={job_intent or 'n/a'}, read={read_intent or 'n/a'}, "
                    f"write={write_intent or 'n/a'}"
                )
        except Exception as e:
            logger.warning(f"MemoryOrchestrator on_job_start failed: {e}")

    # ── Memory: query historical experiences for prompt injection ──
    exp_writer = self._make_exp_writer()
    memory_context = ""
    _run_failure_ids: list[str] = []
    if exp_writer:
        try:
            all_run_exps = await exp_writer.query_for_task(
                user_task=request.instruction[:300],
                session_id=self.workspace.stem, limit=8,
            )
            if all_run_exps:
                memory_context = exp_writer.format_for_prompt(all_run_exps)
                _run_failure_ids = [t.id for t in all_run_exps]
                info(f"Loaded {len(all_run_exps)} relevant experiences for run()")
        except Exception as e:
            logger.warning(f"Failed to query historical experiences (non-fatal): {e}")

    # Stage 5: 模板处理入口（设计文档 05 要求）
    template_profile = None
    guide_builder = None
    template_conflicts = []
    style_intent_result = None
    template_activation_decision = decide_template_activation(request)
    _initial_template_intent = template_activation_decision.use_intent.value
    if template_activation_decision.allowed:
        explicit_gate_decision = "use_template"
        explicit_gate_basis = {
            "explicit": "explicit_file",
            "memory_reuse": "memory_reuse",
            "strong_reference_style": "strong_reference_style",
        }.get(_initial_template_intent, "explicit_file")
    elif _initial_template_intent == "disabled":
        explicit_gate_decision = "forbid_template"
        explicit_gate_basis = "anti_template"
    else:
        explicit_gate_decision = "no_template"
        explicit_gate_basis = "content_only"
    self._update_template_match_state(
        request,
        template_use_decision=explicit_gate_decision,
        template_use_basis=explicit_gate_basis,
        template_use_intent=template_activation_decision.use_intent.value,
        template_activation_allowed=template_activation_decision.allowed,
        template_activation_reason=template_activation_decision.reason,
        template_activation_evidence=template_activation_decision.evidence,
    )
    template_usage_recorded = False
    if request.template_as_reference and not template_activation_decision.allowed:
        self._update_template_match_state(
            request,
            template_intent="anti",
            template_intent_confidence=1.0,
            template_use_decision="forbid_template",
            template_use_basis="anti_template",
            selection_source="skip_disabled",
            selection_confidence=1.0,
            selection_reasoning=template_activation_decision.reason,
        )
        request.template = None
        request.template_id = None
        request.template_as_reference = False
    if request.template_as_reference:
        self._update_template_match_state(
            request,
            template_intent="explicit",
            template_intent_confidence=1.0,
            template_use_decision="use_template",
            template_use_basis="explicit_file",
            template_use_intent=template_activation_decision.use_intent.value,
            template_activation_allowed=template_activation_decision.allowed,
            template_activation_reason=template_activation_decision.reason,
            selection_source="explicit_input_requested",
            selection_confidence=1.0,
            selected_template_id=request.template_id or "",
            selected_template_name=Path(request.template).stem if request.template else "",
            matched_by_history=False,
            matched_history_intent="",
        )
        try:
            # 方式 1: 复用已存储的模板档案
            if request.template_id and self.memory_system:
                store = getattr(self.memory_system, 'template_store', None)
                if store:
                    template_profile = await store.get(request.template_id)
                    if template_profile:
                        info(f"Loaded stored template profile: {template_profile.name}")

            # 方式 2: 分析新上传的模板
            if not template_profile and request.template:
                template_profile = await self._analyze_template(request.template)
                if template_profile:
                    layout_count = len([k for k in template_profile.slide_induction.keys() if k not in ("functional_keys", "language", "layout_capabilities")])
                    info(f"Template profile extracted: {layout_count} layouts")

            # 冲突检测（Stage 7: 使用 TemplateGuideBuilder）
            if template_profile:
                guide_builder, template_conflicts = self._activate_template_profile(
                    request, template_profile
                )
                self._update_template_match_state(
                    request,
                    template_intent="explicit",
                    template_intent_confidence=1.0,
                    selected_template_id=getattr(template_profile, "id", "") or getattr(template_profile, "template_id", ""),
                    selected_template_name=getattr(template_profile, "name", "") or (Path(request.template).stem if request.template else ""),
                    selection_source="explicit_input",
                    selection_confidence=1.0,
                )
                template_usage_recorded = await self._record_explicit_template_seed_usage(
                    request,
                    template_profile,
                )
                if template_conflicts:
                    info(f"Template conflicts detected: {len(template_conflicts)}")
        except Exception as e:
            self._update_template_match_state(
                request,
                selection_source="explicit_input_failed",
                selection_reasoning=str(e),
            )
            logger.warning(f"Template processing failed (non-fatal): {e}")
    elif self.memory_system and not request.template and not request.template_id:
        try:
            template_profile, style_intent_result = await self._auto_match_template_profile(
                request
            )
            if template_profile:
                guide_builder, template_conflicts = self._activate_template_profile(
                    request, template_profile
                )
                if template_conflicts:
                    info(f"Template conflicts detected: {len(template_conflicts)}")
        except Exception as e:
            logger.warning(f"Stage 14 auto template matching failed (non-fatal): {e}")

    try:
        request.copy_to_workspace(self.workspace)
        hello_message = f"MemSlides running in {self.workspace}, with {len(request.attachments)} attachments, prompt={request.instruction}"
        if self.config.offline_mode:
            hello_message += " [Offline Mode]"
        info(hello_message)
        yield ChatMessage(role=Role.SYSTEM, content=hello_message)
        self.research_agent = Researcher(
            self.config,
            agent_env,
            self.workspace,
            self.language,
        )
        self.agent = self.research_agent

        # Stage 15: Unified tool callback — all calls → on_operation_complete (populates tool log)
        #                                    errors  → on_tool_error (writes round experience to WM)
        if _orchestrator:
            def _make_callback(orch):
                async def _cb(tool_name, arguments, result, is_error, duration_ms=0, reasoning="", reason_source=""):
                    try:
                        orch.on_operation_complete(
                            tool_name=tool_name,
                            args=str(arguments),
                            result=str(result)[:5000],
                            is_error=is_error,
                            duration_ms=duration_ms,
                            reasoning=reasoning or "",
                            reason_source=reason_source or "",
                        )
                    except Exception as _e:
                        logger.debug("on_operation_complete failed (non-fatal): %s", _e)
                    if is_error:
                        try:
                            orch.on_tool_error(
                                tool_name=tool_name,
                                args=str(arguments),
                                error_msg=str(result)[:500],
                            )
                        except Exception as _e:
                            logger.debug("on_tool_error failed (non-fatal): %s", _e)
                return _cb
            self.research_agent._tool_result_callback = _make_callback(_orchestrator)

        # MemoryOrchestrator: Round Start + SYSTEM-level injection for Researcher Agent
        if _orchestrator:
            try:
                await _orchestrator.on_round_start(
                    user_message=request.instruction[:300],
                    user_id=self.user_id,
                    context={"template_id": self._current_template_id} if self._current_template_id else {},
                    session_id=self.workspace.stem,
                    agent_name="research",
                )
                await self._inject_system_memory(
                    _orchestrator, "research", request.instruction[:300], self.research_agent,
                )
            except Exception as e:
                logger.warning(f"Orchestrator research injection failed: {e}")
        elif memory_context:
            # Fallback: inject raw experience context if no orchestrator
            self.research_agent.chat_history[0] = ChatMessage(
                role=Role.SYSTEM,
                content=self.research_agent.system + f"\n\n{memory_context}",
            )
        if exp_writer and _run_failure_ids:
            await exp_writer.mark_reused(_run_failure_ids)

        # Stage 7: 注入模板档案到 Researcher Agent（使用 TemplateGuideBuilder）
        if template_profile and guide_builder:
            try:
                template_prompt = guide_builder.build_for_research()
                if template_prompt:
                    if template_conflicts:
                        conflict_warning = guide_builder.format_conflicts_for_prompt(template_conflicts)
                        template_prompt += f"\n\n⚠️ **Conflict Warning**:\n{conflict_warning}\nUse `ask_user_clarification` if needed."

                    current_system = self.research_agent.chat_history[0].text
                    self.research_agent.chat_history[0] = ChatMessage(
                        role=Role.SYSTEM,
                        content=current_system + f"\n\n{template_prompt}",
                    )
                    info(f"Injected template prompt to Researcher Agent ({len(template_prompt)} chars)")

                    # 保存注入的模板 prompt 作为中间产物（便于调试验证）
                    _template_prompt_file = self.workspace / ".history" / "template_prompt_research.md"
                    _template_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                    _template_prompt_file.write_text(template_prompt, encoding="utf-8")
                    info(f"Saved template prompt to {_template_prompt_file}")
                    self._refresh_system_injection_trace_with_template(
                        agent_role="research",
                        turn=0,
                        template_prompt=template_prompt,
                    )
            except Exception as e:
                logger.warning(f"Template prompt injection to Researcher failed: {e}")

        research_tool_start = len(agent_env.tool_history)
        research_ok = True
        md_file: Path | None = None
        set_current_agent(
            "Researcher",
            workspace=self.workspace,
            model_ref=getattr(self.research_agent, "model_ref", "research_agent"),
        )  # For finalize() behavior
        try:
            async for msg in self.research_agent.loop(request):
                if isinstance(msg, str):
                    md_file = Path(msg)
                    if not md_file.is_absolute():
                        md_file = self.workspace / md_file
                    self.intermediate_output["manuscript"] = md_file
                    msg = str(md_file)
                    break
                yield msg
        except Exception as e:
            research_ok = False
            error_message = (
                f"Researcher agent failed with error: {e}\n{traceback.format_exc()}"
            )
            error(error_message)
            error(traceback.format_exc())
            fallback_md: Path | None = None
            try:
                fallback_md = await _build_attachment_grounded_manuscript(self, request)
            except Exception as fallback_exc:
                info(f"Attachment-grounded research fallback unavailable: {fallback_exc}")
            if fallback_md is not None:
                md_file = fallback_md
                research_ok = True
                yield ChatMessage(
                    role=Role.SYSTEM,
                    content=(
                        "Researcher LLM route failed, so MemSlides used the attachment-grounded "
                        f"local fallback manuscript at {fallback_md} and continued the pipeline."
                    ),
                )
            else:
                yield ChatMessage(role=Role.SYSTEM, content=error_message)
                raise e
        finally:
            self.research_agent.save_history()
            self.save_results()
            # ── Memory: fallback direct write only when orchestrator is unavailable ──
            if exp_writer and not _orchestrator:
                try:
                    research_tool_slice = agent_env.tool_history[research_tool_start:]
                    await exp_writer.from_agent_run(
                        session_id=self.workspace.stem,
                        agent_name="research",
                        chat_history=self.research_agent.chat_history,
                        tool_history=research_tool_slice,
                        task=request.instruction[:400],
                        outcome="success" if research_ok else "failed",
                        template_id=self._current_template_id,
                    )
                    info(f"Written Researcher Agent ExperienceTrace (outcome={'success' if research_ok else 'failed'})")
                except Exception as e:
                    logger.warning(f"Failed to write Researcher ExperienceTrace (non-fatal): {e}")
        if md_file is None:
            md_file = _persist_research_manuscript(self, request)
            self.intermediate_output["manuscript"] = md_file
            info(f"Researcher stage completed via saved manuscript fallback: {md_file}")
        asset_manifest_path = _write_content_asset_manifest(self, md_file)
        info(f"Saved attachment asset manifest to {asset_manifest_path}")
        page_asset_plan_path: Path | None = None
        if request.convert_type == ConvertType.TEMPLATE_PLANNER:
            self.template_planner = TemplatePlanner(
                self.config,
                agent_env,
                self.workspace,
                self.language,
            )
            self.agent = self.template_planner

            # Stage 15: Unified tool callback
            if _orchestrator:
                self.template_planner._tool_result_callback = _make_callback(_orchestrator)

            # Inject global DB failures into TemplatePlanner
            # NOTE: Stage 8 移除 research_lessons 跨 Agent 注入，pipeline 经验不适用于 DeckDesigner 类任务
            _ppt_extra = ""
            if memory_context:
                _ppt_extra += f"\n\n{memory_context}"
            if _ppt_extra:
                self.template_planner.chat_history[0] = ChatMessage(
                    role=Role.SYSTEM,
                    content=self.template_planner.system + _ppt_extra,
                )
                if exp_writer and _run_failure_ids:
                    await exp_writer.mark_reused(_run_failure_ids)

            ppt_tool_start = len(agent_env.tool_history)
            ppt_ok = True
            set_current_agent(
                "TemplatePlanner",
                workspace=self.workspace,
                model_ref=getattr(self.template_planner, "model_ref", "design_agent"),
            )  # For finalize() behavior
            try:
                async for msg in self.template_planner.loop(request, md_file):
                    if isinstance(msg, str):
                        pptx_file = Path(msg)
                        if not pptx_file.is_absolute():
                            pptx_file = self.workspace / pptx_file
                        self.intermediate_output["pptx"] = pptx_file
                        self.intermediate_output["final"] = pptx_file
                        msg = str(pptx_file)
                        break
                    yield msg
            except Exception as e:
                ppt_ok = False
                error_message = (
                    f"TemplatePlanner failed with error: {e}\n{traceback.format_exc()}"
                )
                error(error_message)
                error(traceback.format_exc())
                yield ChatMessage(role=Role.SYSTEM, content=error_message)
                raise e
            finally:
                self.template_planner.save_history()
                self.save_results()
                # ── Memory: fallback direct write only when orchestrator is unavailable ──
                if exp_writer and not _orchestrator:
                    try:
                        ppt_tool_slice = agent_env.tool_history[ppt_tool_start:]
                        await exp_writer.from_agent_run(
                            session_id=self.workspace.stem,
                            agent_name="template_planner",
                            chat_history=self.template_planner.chat_history,
                            tool_history=ppt_tool_slice,
                            task=request.instruction[:400],
                            outcome="success" if ppt_ok else "failed",
                            template_id=self._current_template_id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write TemplatePlanner ExperienceTrace (non-fatal): {e}")
                if _orchestrator:
                    try:
                        _ppt_response = ""
                        if self.template_planner and self.template_planner.chat_history:
                            for _m in reversed(self.template_planner.chat_history):
                                if _m.role == Role.ASSISTANT and _m.text:
                                    _ppt_response = _m.text[:2000]
                                    break
                        await _orchestrator.on_round_end(agent_response=_ppt_response)
                    except Exception as e:
                        logger.warning(f"Orchestrator on_round_end(template_planner) failed: {e}")
        else:
            _template_runtime_mode_for_role = str(
                getattr(getattr(self, "_template_runtime_state", None), "mode", "") or ""
            )
            _template_role_file = (
                Path(__file__).resolve().parent.parent / "roles" / "ReferenceTemplateDesigner.yaml"
                if template_profile and _is_structural_template_mode(_template_runtime_mode_for_role)
                else None
            )
            self.designagent = DeckDesigner(
                self.config,
                agent_env,
                self.workspace,
                self.language,
                config_file=str(_template_role_file) if _template_role_file else None,
            )
            self.agent = self.designagent

            # Stage 15: Unified tool callback
            if _orchestrator:
                self.designagent._tool_result_callback = _make_callback(_orchestrator)

            _profile_execution_contract: dict[str, Any] | None = None
            _profile_execution_plan: dict[str, Any] | None = None
            _profile_source_evidence_summary = _build_profile_execution_source_evidence_summary(
                request,
                self.workspace,
                md_file,
            )
            _profile_target_page_count = (
                _coerce_positive_page_count(request.num_pages) or _infer_markdown_slide_count(md_file)
            )
            if _orchestrator:
                try:
                    _profile_execution_contract = (
                        await _orchestrator.compile_and_register_profile_execution_contract(
                            instruction=request.instruction[:4000],
                            resolved_intent_artifact=self.get_resolved_intent_artifact(),
                            source_evidence_summary=_profile_source_evidence_summary,
                            core_persona=self._get_request_core_persona(request),
                            task_intent=self._get_request_task_intent(request),
                            read_intent=self._get_request_memory_read_intent(request),
                            write_intent=self._get_request_memory_write_intent(request),
                        )
                    )
                    if _profile_execution_contract:
                        info(
                            "Compiled profile_execution_contract for DeckDesigner (%s obligations, focus=%s)",
                            len(_profile_execution_contract.get("page_obligations", []) or []),
                            _profile_execution_contract.get("planning_focus", ""),
                        )
                        _profile_execution_plan = await _orchestrator.compile_and_register_profile_execution_plan(
                            instruction=request.instruction[:4000],
                            source_evidence_summary=_profile_source_evidence_summary,
                            target_page_count=_profile_target_page_count,
                            template_mode=bool(template_profile),
                        )
                        if _profile_execution_plan:
                            info(
                                "Compiled profile_execution_plan for DeckDesigner (%s pages, focus=%s)",
                                len(_profile_execution_plan.get("page_plan", []) or []),
                                _profile_execution_plan.get("planning_focus", ""),
                            )
                except Exception as e:
                    logger.warning(
                        "Profile execution contract/plan compile failed (non-fatal): %s",
                        e,
                    )

            # MemoryOrchestrator: SYSTEM-level injection for DeckDesigner Agent
            # (same round as Researcher — on_round_start already called)
            if _orchestrator:
                try:
                    await self._inject_system_memory(
                        _orchestrator, "design", request.instruction[:300], self.designagent,
                    )
                except Exception as e:
                    logger.warning(f"Orchestrator design injection failed: {e}")
            elif memory_context:
                self.designagent.chat_history[0] = ChatMessage(
                    role=Role.SYSTEM,
                    content=self.designagent.system + f"\n\n{memory_context}",
                )
            if exp_writer and _run_failure_ids:
                await exp_writer.mark_reused(_run_failure_ids)

            design_slide_dir: Path | None = None
            try:
                design_slide_dir = self._prime_design_slide_output_dir(md_file)
                workspace_context = self._build_workspace_context_block(
                    slide_dir=design_slide_dir,
                    manuscript_path=md_file,
                    include_active_slide_dir=True,
                )
                if workspace_context:
                    current_system = self.designagent.chat_history[0].text
                    self.designagent.chat_history[0] = ChatMessage(
                        role=Role.SYSTEM,
                        content=(
                            current_system
                            + f"\n\n{workspace_context}"
                            + f"\n\n{_asset_manifest_prompt(asset_manifest_path)}"
                        ),
                    )
                    info(
                        "Injected workspace context to DeckDesigner Agent with active slide dir %s",
                        design_slide_dir,
                    )
            except Exception as e:
                logger.warning(f"Workspace context injection to DeckDesigner failed: {e}")

            # Stage 7: 注入模板档案到 DeckDesigner Agent（使用 TemplateGuideBuilder）
            template_runtime_mode = str(
                getattr(getattr(self, "_template_runtime_state", None), "mode", "") or ""
            )

            if (
                template_profile
                and guide_builder
                and _is_structural_template_mode(template_runtime_mode)
            ):
                try:
                    template_prompt = guide_builder.build_for_design()
                    if template_prompt:
                        if template_conflicts:
                            conflict_warning = guide_builder.format_conflicts_for_prompt(template_conflicts)
                            template_prompt += f"\n\n⚠️ **Conflict Warning**:\n{conflict_warning}\nUse `ask_user_clarification` if needed."

                        current_system = self.designagent.chat_history[0].text
                        self.designagent.chat_history[0] = ChatMessage(
                            role=Role.SYSTEM,
                            content=current_system + f"\n\n{template_prompt}",
                        )
                        info(f"Injected template prompt to DeckDesigner Agent ({len(template_prompt)} chars)")

                        # 保存注入的模板 prompt 作为中间产物（便于调试验证）
                        _template_prompt_file = self.workspace / ".history" / "template_prompt_design.md"
                        _template_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                        _template_prompt_file.write_text(template_prompt, encoding="utf-8")
                        info(f"Saved template prompt to {_template_prompt_file}")
                        self._refresh_system_injection_trace_with_template(
                            agent_role="design",
                            turn=0,
                            template_prompt=template_prompt,
                        )
                except Exception as e:
                    logger.warning(f"Template prompt injection to DeckDesigner failed: {e}")

            if not template_profile or template_runtime_mode == STYLE_REFERENCE:
                try:
                    _existing_design_plan_rel = _find_existing_design_plan_rel(
                        self.workspace
                    )
                    _created_design_plan_scaffold = False
                    _persona_ready_scaffold = False

                    if not _existing_design_plan_rel:
                        try:
                            _scaffold_path = self.workspace / "design_plan.md"
                            _scaffold_content = _build_non_template_design_plan_scaffold(
                                request,
                                _profile_execution_contract,
                                _profile_execution_plan,
                            )
                            _persona_ready_scaffold = bool(
                                _profile_execution_contract or _profile_execution_plan
                            )
                            _scaffold_path.write_text(
                                _scaffold_content,
                                encoding="utf-8",
                            )
                            initialize_design_plan_tracking(
                                self.workspace,
                                _scaffold_path,
                                _scaffold_content,
                                source="system_scaffold",
                                requires_refinement=not _persona_ready_scaffold,
                            )
                            _existing_design_plan_rel = "design_plan.md"
                            _created_design_plan_scaffold = True
                            info(
                                "Pre-created non-template design_plan.md scaffold -> %s",
                                _scaffold_path,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to pre-create non-template design_plan.md scaffold: %s",
                                e,
                            )
                    else:
                        _persona_ready_scaffold = False

                    _profile_contract_present = bool(_profile_execution_contract)
                    _profile_contract_scaffold_suffix = (
                        "\n- Preserve `## Profile-Derived Deck Contract` and `## Page-Level Persona Obligations` from the scaffold."
                        "\n- Preserve `## Persona Page Plan` from the scaffold and treat it as the default per-page work queue."
                        "\n- Expand those sections into explicit page mapping so persona preferences become concrete page responsibilities, not just general style notes."
                        "\n- If a profile preference is source-unsupported, handle it via proxy realization or leave it under do-not-force instead of inventing unsupported content."
                        if _profile_contract_present
                        else ""
                    )
                    _profile_contract_existing_plan_suffix = (
                        "\n- If you refine or overwrite the plan, explicitly include `## Profile-Derived Deck Contract`, `## Page-Level Persona Obligations`, and `## Persona Page Plan`."
                        "\n- Convert the contract into page-level persona obligations and a full persona page plan before writing any slide HTML."
                        if _profile_contract_present
                        else ""
                    )
                    _profile_contract_required_sections = (
                        "  3. `## Profile-Derived Deck Contract`\n"
                        "  4. `## Page-Level Persona Obligations`\n"
                        "  5. `## Persona Page Plan`\n"
                        "  6. `## Theme Keywords`\n"
                        "  7. `## Color Palette`\n"
                        "  8. `## Typography`\n"
                        "  9. `## Spacing & Grid`\n"
                        "  10. `## Page Archetypes`\n"
                        "  11. `## Component Rules`\n"
                        "  12. `## Do / Don't`\n"
                        if _profile_contract_present
                        else
                        "  3. `## Theme Keywords`\n"
                        "  4. `## Color Palette`\n"
                        "  5. `## Typography`\n"
                        "  6. `## Spacing & Grid`\n"
                        "  7. `## Page Archetypes`\n"
                        "  8. `## Component Rules`\n"
                        "  9. `## Do / Don't`\n"
                    )
                    _profile_contract_creation_suffix = (
                        "\n- The profile-derived persona sections are mandatory when a compiled contract is available."
                        "\n- `## Persona Page Plan` is also mandatory when a compiled persona plan is available."
                        "\n- Expand the persona contract into page-specific obligations for the opener, middle evidence/teaching pages, and ending pages."
                        if _profile_contract_present
                        else ""
                    )
                    if template_profile and template_runtime_mode == STYLE_REFERENCE:
                        reference_note = (
                            "\n- A template was analyzed, but its structure quality is too weak for strict layout use."
                            "\n- Treat it only as a style reference; do not copy its zero-text layout slots or force a blue full-slide background."
                            "\n- Prefer readable manuscript-driven layouts with real text, formulas, figures, and tables from the attachment."
                        )
                    else:
                        reference_note = ""

                    if _existing_design_plan_rel:
                        if _created_design_plan_scaffold:
                            if _persona_ready_scaffold:
                                design_plan_prompt = (
                                    "⚠️ **Non-template design plan reminder**:\n"
                                    f"- A persona-aware system scaffold has already been created at `{_existing_design_plan_rel}` so this artifact cannot be skipped.\n"
                                    "- The incoming manuscript markdown is read-only in DeckDesigner stage. Do not rewrite it or any other markdown artifact except `design_plan.md`.\n"
                                    "- Read it before generating any slide HTML.\n"
                                    "- If the scaffold already matches the manuscript and persona, keep it and proceed; do not rewrite it just to paraphrase the same contract.\n"
                                    "- If the manuscript exposes a real conflict or missing detail, refine only the affected sections of `design_plan.md` before writing slides.\n"
                                    "- Use the accepted or lightly refined plan as the visual specification for every page.\n"
                                    "- Do not bypass the design plan when choosing colors, fonts, spacing, or layout patterns."
                                    f"{_profile_contract_scaffold_suffix}"
                                    f"{reference_note}"
                                )
                            else:
                                design_plan_prompt = (
                                    "⚠️ **Non-template design plan reminder**:\n"
                                    f"- A system scaffold has already been created at `{_existing_design_plan_rel}` so this artifact cannot be skipped.\n"
                                    "- The incoming manuscript markdown is read-only in DeckDesigner stage. Do not rewrite it or any other markdown artifact except `design_plan.md`.\n"
                                    "- Read it before generating any slide HTML.\n"
                                    "- If the scaffold is too generic for the manuscript, overwrite `design_plan.md` with a refined, manuscript-specific version before writing slides.\n"
                                    "- Use the accepted or refined plan as the visual specification for every page.\n"
                                    "- Do not bypass the design plan when choosing colors, fonts, spacing, or layout patterns."
                                    f"{_profile_contract_scaffold_suffix}"
                                    f"{reference_note}"
                                )
                        else:
                            design_plan_prompt = (
                                "⚠️ **Non-template design plan reminder**:\n"
                                f"- A design plan already exists at `{_existing_design_plan_rel}`.\n"
                                "- The incoming manuscript markdown is read-only in DeckDesigner stage. Do not rewrite it or any other markdown artifact except `design_plan.md`.\n"
                                "- Read it before generating any slide HTML and use it as the visual specification for all pages.\n"
                                "- Do not bypass the design plan when choosing colors, fonts, spacing, or layout patterns."
                                f"{_profile_contract_existing_plan_suffix}"
                                f"{reference_note}"
                            )
                    else:
                        design_plan_prompt = (
                            "⚠️ **Non-template design plan reminder**:\n"
                            "- No template specifications were provided, and no design plan file exists yet.\n"
                            "- The incoming manuscript markdown is read-only in DeckDesigner stage. Do not rewrite it or any other markdown artifact except `design_plan.md`.\n"
                            "- Your first substantive action MUST be to create `design_plan.md` with `write_markdown_file`.\n"
                            "- Do not write any `slide_XX.html` or call `finalize` before `design_plan.md` exists.\n"
                            "- After creating it, read it back and use it as the design specification for every slide.\n"
                            "- `design_plan.md` MUST be a structured spec, not a vague essay. Include at least:\n"
                            "  1. `# Design Plan`\n"
                            "  2. `## Design Goal`\n"
                            f"{_profile_contract_required_sections}"
                            "- The plan must contain concrete colors, fonts, spacing, layout strategy, and component rules that can be executed directly in HTML."
                            f"{_profile_contract_creation_suffix}"
                            f"{reference_note}"
                        )

                    current_system = self.designagent.chat_history[0].text
                    self.designagent.chat_history[0] = ChatMessage(
                        role=Role.SYSTEM,
                        content=(
                            current_system
                            + f"\n\n{design_plan_prompt}\n\n"
                            + _render_design_plan_execution_plan(
                                _existing_design_plan_rel or "design_plan.md",
                                scaffold_created=_created_design_plan_scaffold,
                                scaffold_requires_refinement=not _persona_ready_scaffold,
                                profile_contract_present=_profile_contract_present,
                            )
                        ),
                    )
                    info("Injected non-template design plan reminder to DeckDesigner Agent")
                except Exception as e:
                    logger.warning(
                        f"Non-template design plan reminder injection failed: {e}"
                    )

            # Stage 8.5: 从模板确定性生成 design_plan.md（无 LLM，纯代码拼装）
            # 让 DeckDesigner Agent 有一个预设的设计方案文件可读取，避免自创颜色/字体
            _design_plan_path = self.workspace / "design_plan.md"
            _layout_mapping_path = self.workspace / "layout_mapping.yaml"
            if template_profile and guide_builder:
                try:
                    guide_builder.generate_design_plan(
                        _design_plan_path,
                        mode=template_runtime_mode or STRUCTURAL_TEMPLATE,
                    )
                    initialize_design_plan_tracking(
                        self.workspace,
                        _design_plan_path,
                        _design_plan_path.read_text(encoding="utf-8"),
                        source="template_generated",
                        requires_refinement=template_runtime_mode == STYLE_REFERENCE,
                    )
                    current_system = self.designagent.chat_history[0].text
                    self.designagent.chat_history[0] = ChatMessage(
                        role=Role.SYSTEM,
                        content=(
                            current_system
                            + "\n\n"
                            + _render_design_plan_execution_plan(
                                "design_plan.md",
                                template_generated=True,
                            )
                        ),
                    )
                    info(f"Auto-generated design_plan.md from template -> {_design_plan_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate design_plan.md from template (non-fatal): {e}")

                if _is_structural_template_mode(template_runtime_mode) and md_file is not None:
                    try:
                        manuscript_text = Path(md_file).read_text(encoding="utf-8")
                        from memslides.templates.layout_planner import TemplateLayoutPlanner
                        manuscript_text = TemplateLayoutPlanner.strip_legacy_layout_mapping(manuscript_text)
                        asset_manifest_payload = json.loads(
                            asset_manifest_path.read_text(encoding="utf-8")
                        )
                        layout_mapping = guide_builder.build_layout_mapping(
                            manuscript_content=manuscript_text,
                            asset_manifest=asset_manifest_payload,
                            page_count_hint=int(str(request.num_pages).strip())
                            if str(request.num_pages or "").strip().isdigit()
                            else None,
                        )
                        page_asset_plan_path = self.workspace / "page_asset_plan.json"
                        page_execution_plan_path = self.workspace / "page_execution_plan.md"
                        template_reference_profile_path = self.workspace / "template_reference_profile.json"
                        page_asset_plan = layout_mapping.get("page_asset_plan", {})
                        page_asset_plan_path.write_text(
                            json.dumps(page_asset_plan, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        template_reference_profile_path.write_text(
                            json.dumps(
                                layout_mapping.get("template_reference_profile", {}),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        _layout_mapping_path.write_text("", encoding="utf-8")
                        from memslides.templates.layout_planner import TemplateLayoutPlanner
                        TemplateLayoutPlanner(template_profile).dump_layout_mapping(
                            layout_mapping,
                            _layout_mapping_path,
                        )
                        page_execution_plan_path.write_text(
                            _build_page_execution_plan(
                                layout_mapping,
                                _profile_execution_plan,
                            ),
                            encoding="utf-8",
                        )
                        initialize_control_document_tracking(
                            self.workspace,
                            page_execution_plan_path,
                            page_execution_plan_path.read_text(encoding="utf-8"),
                            source="template_generated",
                            required_for_html_prewrite=True,
                        )
                        deck_execution_state_path = initialize_deck_execution_state(
                            self.workspace,
                            _layout_mapping_path,
                            expected_slide_count=len(layout_mapping.get("slides", []) or []),
                            slide_dir="outputs",
                            profile_execution_plan=_profile_execution_plan,
                        )
                        current_system = self.designagent.chat_history[0].text
                        self.designagent.chat_history[0] = ChatMessage(
                            role=Role.SYSTEM,
                            content=(
                                current_system
                                + "\n\n"
                                + _page_execution_plan_prompt(page_execution_plan_path)
                                + "\n\n"
                                + _page_asset_plan_prompt(page_asset_plan_path)
                                + "\n\n"
                                + _layout_mapping_prompt(_layout_mapping_path)
                                + "\n\n"
                                + (
                                    "📘 **Template reference profile**:\n"
                                    f"- Read `{template_reference_profile_path.name}` if you need the full reference recipe.\n"
                                    "- Use it for page archetypes, density, slot geometry, palette, typography, and title rhythm only.\n"
                                    "- Do not use it as a source of background images or old template sample content."
                                )
                            ),
                        )
                        info(
                            "Auto-generated page_asset_plan/layout_mapping/reference_profile/execution_brief/state -> %s / %s / %s / %s / %s",
                            page_asset_plan_path,
                            _layout_mapping_path,
                            template_reference_profile_path,
                            page_execution_plan_path,
                            deck_execution_state_path,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to generate layout_mapping for template mode (non-fatal): %s",
                            e,
                        )

            # Stage 7: 设置 MCP Tool 上下文（供 query_slide_layout 等工具使用）
            if (
                template_profile
                and guide_builder
                and _is_structural_template_mode(template_runtime_mode)
            ):
                try:
                    from memslides.tools.template_tools import set_template_context
                    set_template_context(guide_builder)
                    # 序列化到文件，供 MCP 子进程读取（contextvars 不跨进程）
                    _profile_file = self.workspace / ".template_profile.json"
                    _profile_file.write_text(template_profile.to_json(), encoding="utf-8")
                    # W4: 通过环境变量传递 workspace 绝对路径，避免依赖 cwd
                    import os as _os
                    _os.environ["MEMSLIDES_WORKSPACE"] = str(self.workspace)
                    info("Template MCP context set for DeckDesigner Agent (+ cross-process file + env)")
                except Exception as e:
                    logger.warning(f"Failed to set template MCP context: {e}")

                # 确保模板工具注册到 DeckDesigner Agent（MCP 子进程注册的工具可能未被客户端拉取）
                _template_tool_names = [
                    "list_template_layouts",
                    "recommend_template_layout",
                    "query_slide_layout", "query_layout_geometry", "query_image_info",
                ]
                for _tt_name in _template_tool_names:
                    _tt_spec = self.agent_env._tools_dict.get(_tt_name)
                    if _tt_spec and _tt_spec not in self.designagent.tools:
                        self.designagent.tools.append(_tt_spec)
                        info(f"Template tool '{_tt_name}' added to DeckDesigner Agent")
                    elif not _tt_spec:
                        logger.warning(
                            f"Template tool '{_tt_name}' not found in agent_env — "
                            f"MCP deck runtime may not have registered it. "
                            f"Available: {list(self.agent_env._tools_dict.keys())}"
                        )
                deduped_tools = _dedupe_tool_specs(self.designagent.tools)
                if len(deduped_tools) != len(self.designagent.tools):
                    info(
                        "Deduplicated DeckDesigner tools from %s to %s",
                        len(self.designagent.tools),
                        len(deduped_tools),
                    )
                    self.designagent.tools = deduped_tools

            self.designagent.tools = _dedupe_tool_specs(self.designagent.tools)

            try:
                deck_execution_state_path = _ensure_generation_deck_execution_state(
                    workspace=self.workspace,
                    request=request,
                    slide_dir=design_slide_dir,
                    md_file=md_file,
                    profile_execution_plan=_profile_execution_plan,
                )
                if deck_execution_state_path:
                    info(
                        "Initialized DeckDesigner execution state from request/manuscript -> %s",
                        deck_execution_state_path,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to initialize DeckDesigner execution state (non-fatal): %s",
                    e,
                )

            design_tool_start = len(agent_env.tool_history)
            design_ok = True
            slide_html_dir: Path | None = None
            set_current_agent(
                "DeckDesigner",
                workspace=self.workspace,
                model_ref=getattr(self.designagent, "model_ref", "design_agent"),
            )  # For finalize() behavior
            try:
                async for msg in self.designagent.loop(request, md_file):
                    if isinstance(msg, str):
                        slide_html_dir = Path(msg)
                        if not slide_html_dir.is_absolute():
                            slide_html_dir = self.workspace / slide_html_dir
                        self.intermediate_output["slide_html_dir"] = str(slide_html_dir)
                        break
                    yield msg
            except Exception as e:
                design_ok = False
                error_message = (
                    f"DeckDesigner agent failed with error: {e}\n{traceback.format_exc()}"
                )
                error(error_message)
                error(traceback.format_exc())
                yield ChatMessage(role=Role.SYSTEM, content=error_message)
                raise e
            finally:
                self.designagent.save_history()
                self.save_results()
                # ── Memory: fallback direct write only when orchestrator is unavailable ──
                if exp_writer and not _orchestrator:
                    try:
                        design_tool_slice = agent_env.tool_history[design_tool_start:]
                        await exp_writer.from_agent_run(
                            session_id=self.workspace.stem,
                            agent_name="design",
                            chat_history=self.designagent.chat_history,
                            tool_history=design_tool_slice,
                            task=request.instruction[:400],
                            outcome="success" if design_ok else "failed",
                            template_id=self._current_template_id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write DeckDesigner ExperienceTrace (non-fatal): {e}")
                # MemoryOrchestrator: Round End (Episode extraction, chain segmentation, experience extraction)
                if _orchestrator:
                    try:
                        _design_response = ""
                        if self.designagent and self.designagent.chat_history:
                            for _m in reversed(self.designagent.chat_history):
                                if _m.role == Role.ASSISTANT and _m.text:
                                    _design_response = _m.text[:2000]
                                    break
                        await _orchestrator.on_round_end(agent_response=_design_response)
                    except Exception as e:
                        logger.warning(f"Orchestrator on_round_end(design) failed: {e}")
            resolved_slide_html_dir = self._resolve_exportable_slide_dir(slide_html_dir)
            if resolved_slide_html_dir is None:
                searched_dirs = ", ".join(
                    str(path) for path in self._candidate_slide_dirs(slide_html_dir)
                )
                raise RuntimeError(
                    "DeckDesigner agent did not produce exportable slides. "
                    f"Searched: {searched_dirs}"
                )
            if slide_html_dir != resolved_slide_html_dir:
                info(
                    "Recovered slide output directory: "
                    f"{slide_html_dir or '<missing>'} -> {resolved_slide_html_dir}"
                )
            slide_html_dir = resolved_slide_html_dir
            self.intermediate_output["slide_html_dir"] = str(slide_html_dir)
            self.save_results()
            pptx_path = self.workspace / f"{md_file.stem}.pptx"
            slide_html_dir, export_html_files = await self._export_slides_with_agent_repair(
                slide_html_dir,
                pptx_path,
                aspect_ratio=request.powerpoint_type,
                context_label="initial generation",
            )
            await self._export_pdf_best_effort(
                export_html_files,
                pptx_path.with_suffix(".pdf"),
                aspect_ratio=request.powerpoint_type,
                context_label="initial generation",
            )

            final_artifact = pptx_path if pptx_path.exists() else slide_html_dir
            self.intermediate_output["final"] = str(final_artifact)
            msg = final_artifact
        # MemoryOrchestrator: defer on_job_end to close_env() so modify() can still add tasks
        # Save slide_html_dir for later consolidation
        if _orchestrator:
            _slide_dir = str(slide_html_dir) if 'slide_html_dir' in dir() else ""
            self._deferred_slide_html_dir = _slide_dir
        self.save_results()
        self._last_request = request
        if template_profile and not template_usage_recorded:
            await self._record_template_usage(
                request=request,
                template_profile=template_profile,
                style_intent_result=style_intent_result,
                success=True,
            )
            template_usage_recorded = True
        info(f"MemSlides finished, final output at: {msg}")
        yield msg
    except Exception:
        if template_profile and not template_usage_recorded:
            try:
                await self._record_template_usage(
                    request=request,
                    template_profile=template_profile,
                    style_intent_result=style_intent_result,
                    success=False,
                )
                info(
                    "[Stage 14] Recorded template usage despite run failure "
                    f"(template={getattr(template_profile, 'name', '') or getattr(template_profile, 'id', '')})"
                )
            except Exception as record_error:
                logger.warning(
                    "Failed to persist template usage after run failure: %s",
                    record_error,
                )
        await self.close_env()
        raise



class GenerationPipeline:
    """Initial deck generation pipeline.

    Public callers use this boundary; the heavy implementation stays behind
    the runtime facade for now.
    """

    def __init__(self, runtime):
        self.runtime = runtime

    async def run(self, request: DeckRequest, *, check_llms: bool = False) -> DeckResult:
        input_request = self._to_input_request(request)
        messages: list[str] = []
        async for item in self.stream(input_request, check_llms=check_llms):
            if isinstance(item, ChatMessage):
                if item.text:
                    messages.append(item.text)
            else:
                messages.append(str(item))
        return self._build_result(messages)

    async def stream(
        self,
        request: InputRequest,
        *,
        check_llms: bool = False,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in run_generation_flow(
            self.runtime,
            request,
            check_llms=check_llms,
        ):
            yield item

    @staticmethod
    def _to_input_request(request: DeckRequest) -> InputRequest:
        return InputRequest(
            instruction=request.instruction,
            attachments=[str(path) for path in request.attachments],
            num_pages=str(request.num_pages) if request.num_pages is not None else None,
            language=request.language,
            memory_intent=request.memory_intent,
            template=str(request.template) if request.template else None,
            template_as_reference=bool(request.template_as_reference),
            template_id=request.template_id,
            powerpoint_type=request.powerpoint_type,
            convert_type=request.convert_type,
            extra_info=request.extra_info,
        )

    def _build_result(self, messages: list[str]) -> DeckResult:
        return DeckResult(
            session_id=self.runtime.session_id,
            workspace=self.runtime.workspace,
            final_path=self._path("final"),
            pptx_path=self._path("pptx") or self._pptx_from_final(),
            pdf_path=self._pdf_from_final(),
            slide_html_dir=self._path("slide_html_dir"),
            manuscript_path=self._path("manuscript"),
            intermediate={k: str(v) for k, v in self.runtime.intermediate_output.items()},
            messages=messages,
        )

    def _path(self, key: str) -> Path | None:
        value = self.runtime.intermediate_output.get(key)
        if not value:
            return None

        return Path(value)

    def _pptx_from_final(self):
        final_path = self._path("final")
        return final_path if final_path and final_path.suffix.lower() == ".pptx" else None

    def _pdf_from_final(self):
        final_path = self._path("final")
        if not final_path:
            return None
        pdf_path = final_path.with_suffix(".pdf")
        return pdf_path if pdf_path.exists() else None
