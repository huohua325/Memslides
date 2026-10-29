"""Implementation details."""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from filelock import FileLock

from memslides.tools.deck_runtime import mcp
from memslides.templates.runtime_state import (
    TemplateRuntimeState,
    load_template_runtime_state,
)
from memslides.templates.quality import (
    ADAPTIVE_STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    STRUCTURAL_TEMPLATE,
)
from memslides.templates.semantic_access import (
    canonical_layout_names,
    resolve_canonical_layout,
)
from memslides.templates.layout_planner import PageBrief, TemplateLayoutPlanner
from memslides.runtime.deck_execution_state import (
    deck_progress_summary,
    record_layout_query,
    record_shell_query,
)

if TYPE_CHECKING:
    from memslides.memory.inject.template_guide_builder import TemplateGuideBuilder

# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

_STRUCTURAL_TEMPLATE_MODES = {
    STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    ADAPTIVE_STRUCTURAL_TEMPLATE,
}

_template_context: contextvars.ContextVar[Optional["TemplateGuideBuilder"]] = contextvars.ContextVar(
    "_template_context", default=None
)


_queried_layouts: contextvars.ContextVar[set[str]] = contextvars.ContextVar(
    "_queried_layouts", default=None
)

_QUERIED_LAYOUTS_FILE = ".queried_layouts.json"


def _queried_layouts_file_path() -> Path:
    """Implementation details."""
    ws = Path(os.environ.get("MEMSLIDES_WORKSPACE", "."))
    return ws / _QUERIED_LAYOUTS_FILE


def _load_queried_layouts_from_file() -> set[str]:
    """Implementation details."""
    try:
        path = _queried_layouts_file_path()
        if path.exists():
            lock = FileLock(str(path) + ".lock", timeout=5)
            with lock:
                data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
    except Exception:
        pass
    return set()


def _save_queried_layouts_to_file(layouts: set[str]) -> None:
    """Implementation details."""
    try:
        path = _queried_layouts_file_path()
        lock = FileLock(str(path) + ".lock", timeout=5)
        with lock:
            path.write_text(json.dumps(sorted(layouts), ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def set_template_context(builder: "TemplateGuideBuilder") -> contextvars.Token:
    """Implementation details."""
    return _template_context.set(builder)


def get_template_context() -> Optional["TemplateGuideBuilder"]:
    """Implementation details."""
    return _template_context.get()


def clear_template_context() -> None:
    """Implementation details."""
    _template_context.set(None)
    _queried_layouts.set(None)

    try:
        path = _queried_layouts_file_path()
        if path.exists():
            path.unlink()
    except Exception:
        pass


def mark_layout_queried(layout_name: str) -> None:
    """Implementation details."""
    name = layout_name.lower().strip()

    layouts = _queried_layouts.get()
    if layouts is None:
        layouts = _load_queried_layouts_from_file()
        _queried_layouts.set(layouts)
    layouts.add(name)

    _save_queried_layouts_to_file(layouts)


def is_layout_queried(layout_name: str) -> bool:
    """Implementation details."""
    name = layout_name.lower().strip()
    layouts = _queried_layouts.get()
    if layouts is None:

        layouts = _load_queried_layouts_from_file()
        _queried_layouts.set(layouts)
    return name in layouts


def get_queried_layouts() -> set[str]:
    """Implementation details."""
    layouts = _queried_layouts.get()
    if layouts is None:
        layouts = _load_queried_layouts_from_file()
        _queried_layouts.set(layouts)
    return layouts


# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

def _ensure_context() -> "TemplateGuideBuilder | None":
    """Implementation details."""
    ws = Path(os.environ.get("MEMSLIDES_WORKSPACE", ".")).resolve()
    runtime_state = load_template_runtime_state(ws)
    if not runtime_state or not runtime_state.active:
        return None
    if runtime_state.mode not in _STRUCTURAL_TEMPLATE_MODES:
        return None

    builder = get_template_context()
    if builder:
        return builder

    try:
        from memslides.memory.core.template_models import TemplateProfile
        from memslides.memory.inject.template_guide_builder import TemplateGuideBuilder
        skill_path = Path(runtime_state.profile_path) if runtime_state.profile_path else ws / ".template_profile.json"
        if not skill_path.is_absolute():
            skill_path = ws / skill_path
        if not skill_path.exists():

            skill_path_alt = ws / ".template_skill.json"
            if not skill_path_alt.exists():
                return None
            skill_path = skill_path_alt
        lock = FileLock(str(skill_path) + ".lock", timeout=5)
        with lock:
            skill = TemplateProfile.from_json(skill_path.read_text(encoding="utf-8"))
        builder = TemplateGuideBuilder(skill, workspace=ws)
        set_template_context(builder)
        return builder
    except Exception:
        return None


def _resolve_layout_alias(builder: "TemplateGuideBuilder", layout_name: str) -> str:
    """Map human-friendly or historical layout aliases to canonical template layouts."""
    direct = resolve_canonical_layout(builder.skill, layout_name)
    if direct is not None:
        return direct.name
    normalized = " ".join(layout_name.strip().lower().replace(":", " ").replace("-", " ").split())
    alias_map = {
        "table of contents": "Blank:light:0t:0i",
        "toc": "Blank:light:0t:0i",
        "contents": "Blank:light:0t:0i",
        "introduction": "Blank:light:0t:1i",
        "methodology": "Blank:light:0t:1i",
        "results": "Blank:light:0t:1i",
        "discussion": "Blank:light:0t:1i",
        "conclusion": "Blank:light:0t:1i",
        "references": "Blank:light:0t:0i",
        "opening": "opening",
        "ending": "ending",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if "three column" in normalized or "3 column" in normalized or "three-column" in normalized:
        return "Blank:light:0t:3i"
    if "two column" in normalized or "2 column" in normalized or "two-column" in normalized:
        return "Blank:light:0t:2i"
    if "image" in normalized:
        return "Blank:light:0t:1i"
    return layout_name


def _template_runtime_state() -> TemplateRuntimeState | None:
    ws = Path(os.environ.get("MEMSLIDES_WORKSPACE", ".")).resolve()
    return load_template_runtime_state(ws)


def _canonical_layout_names(builder: "TemplateGuideBuilder") -> list[str]:
    state = _template_runtime_state()
    if state and state.canonical_layout_names:
        return state.canonical_layout_names
    return canonical_layout_names(builder.skill)


def _layout_list_markdown(layouts: list[str]) -> str:
    if not layouts:
        return "(The active template has no available layouts.)"
    return "\n".join(f"- `{name}`" for name in layouts)


def _resolve_canonical_layout(
    builder: "TemplateGuideBuilder",
    layout_name: str,
) -> tuple[str | None, list[str]]:
    layouts = _canonical_layout_names(builder)
    resolved = _resolve_layout_alias(builder, layout_name)
    layout = resolve_canonical_layout(builder.skill, resolved)
    if layout is None:
        return None, layouts
    return layout.name, layouts


# ═══════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════

@mcp.tool()
def list_template_layouts() -> str:
    """Implementation details."""
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Template layout tools require an active template-driven runtime."
    layouts = _canonical_layout_names(builder)
    return "Available canonical template layouts:\n" + _layout_list_markdown(layouts)


def _query_slide_layout_impl(layout_name: str) -> str:
    """Implementation details."""
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Cannot query template layouts outside template-driven mode."
    canonical_layout, available_layouts = _resolve_canonical_layout(builder, layout_name)
    if not canonical_layout:
        return (
            f"Error: TEMPLATE_LAYOUT_NOT_FOUND. `{layout_name}` is not a canonical "
            "layout in the active template. Call `list_template_layouts()` and use "
            "one of these canonical names:\n"
            + _layout_list_markdown(available_layouts)
        )
    query_state = record_layout_query(canonical_layout)
    mark_layout_queried(canonical_layout)
    if query_state.get("repeated"):
        progress = query_state.get("progress", {}) or deck_progress_summary()
        next_page = progress.get("next_page", {}) or {}
        if progress.get("next_action") == "write_html" and next_page:
            next_step = (
                f"Next step: write `{next_page.get('file')}` using cached layout "
                f"`{canonical_layout}`. Do not query this same layout again before writing HTML."
            )
        elif progress.get("next_action") == "inspect_or_fix" and next_page:
            next_step = f"Next step: inspect or fix `{next_page.get('file')}`."
        else:
            next_step = "Next step: continue with the next missing slide or finalize if all slides pass inspection."
        return (
            "Error: REPEATED_LAYOUT_QUERY\n"
            f"`{canonical_layout}` has already been queried "
            f"{query_state.get('count')} times without a new slide write.\n"
            "Reuse the cached layout contract already in the conversation and move forward.\n"
            f"{next_step}\n"
            "Use the cached reference layout recipe and write the next slide HTML."
        )
    detail = builder.get_layout_detail(canonical_layout)
    return detail


@mcp.tool()
def query_slide_layout(layout_name: str) -> str:
    return _query_slide_layout_impl(layout_name)


def _pct(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _geometry_style(geometry: dict[str, Any] | None) -> str:
    geo = geometry or {}
    left = _pct(geo.get("left_pct"))
    top = _pct(geo.get("top_pct"))
    width = _pct(geo.get("width_pct"), 100.0)
    height = _pct(geo.get("height_pct"), 100.0)
    return (
        f"left:{left:.2f}%; top:{top:.2f}%; "
        f"width:{width:.2f}%; height:{height:.2f}%;"
    )


def _asset_src(item: dict[str, Any]) -> str:
    raw = str(item.get("asset_name", "") or "").strip()
    if not raw:
        return ""
    if raw.startswith("template_shell/assets/"):
        return raw
    return f"template_shell/assets/{Path(raw).name}"


def _asset_html_src(item: dict[str, Any], *, slide_dir: str = "outputs") -> str:
    workspace_src = _asset_src(item)
    if not workspace_src:
        return ""
    try:
        return os.path.relpath(workspace_src, start=slide_dir).replace(os.sep, "/")
    except Exception:
        return f"../{workspace_src}"


def _unique_assets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        src = _asset_src(item)
        geom = json.dumps(item.get("geometry", {}) or {}, sort_keys=True)
        key = (src, geom)
        if not src or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _protected_regions_table(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return "- none"
    lines = ["| role | left/top/width/height % |", "|---|---|"]
    for region in regions[:8]:
        role = str(region.get("role", "protected") or "protected")
        lines.append(f"| {role} | `{_geometry_style(region)}` |")
    return "\n".join(lines)


def _query_template_shell_impl(layout_name: str) -> str:
    """Return executable template shell replay instructions for a canonical layout.

    Use this after `query_slide_layout` when the page may replay the template
    shell/background. The returned asset paths are workspace-relative and safe
    to use directly in slide HTML.
    """
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Cannot query the template shell outside template-driven mode."
    canonical_layout, available_layouts = _resolve_canonical_layout(builder, layout_name)
    if not canonical_layout:
        return (
            f"Error: TEMPLATE_LAYOUT_NOT_FOUND. `{layout_name}` is not a canonical "
            "layout in the active template. Available layouts:\n"
            + _layout_list_markdown(available_layouts)
        )

    from memslides.templates.shell import resolve_template_shell

    shell = resolve_template_shell(builder.skill, canonical_layout)
    record_shell_query(canonical_layout)
    if shell is None:
        return (
            f"## Template Shell: `{canonical_layout}`\n\n"
            "- shell_mode: `no_shell`\n"
            "- shell_use recommendation: `semantic_only` or `skip`\n"
            "- Reason: no replayable shell assets were extracted for this canonical layout.\n"
            f"<!-- template-shell-use: skip reason: no replayable shell for {canonical_layout} -->"
        )

    background_assets = _unique_assets(shell.background_assets)
    decorative_layers = _unique_assets(shell.decorative_layers)
    primary_bg = _asset_src(background_assets[0]) if background_assets else ""

    lines = [
        f"## Template Shell: `{canonical_layout}`",
        "",
        f"- shell_id: `{shell.shell_id}`",
        f"- shell_mode: `{shell.shell_mode}`",
        f"- surface_policy: `{shell.surface_policy}`",
        f"- reuse_confidence: `{shell.reuse_confidence:.2f}`",
        "",
        "### Workspace Assets",
    ]
    if background_assets:
        lines.append("Background assets:")
        for item in background_assets[:6]:
            lines.append(
                f"- workspace_path=`{_asset_src(item)}` html_src=`{_asset_html_src(item)}` "
                f"geometry=`{_geometry_style(item.get('geometry'))}`"
            )
    else:
        lines.append("- No dedicated background asset.")
    if decorative_layers:
        lines.append("Decorative layers:")
        for item in decorative_layers[:10]:
            role = str(item.get("role", item.get("type", "decorative")) or "decorative")
            lines.append(
                f"- workspace_path=`{_asset_src(item)}` html_src=`{_asset_html_src(item)}` "
                f"role={role} geometry=`{_geometry_style(item.get('geometry'))}`"
            )
    else:
        lines.append("- No decorative image layers.")

    lines.extend(
        [
            "",
            "### Protected Regions",
            _protected_regions_table(shell.protected_regions),
            "",
            "### Safe Content Regions",
            _protected_regions_table(shell.safe_content_regions),
            "",
            "### HTML/CSS Recipe",
            "Use the `html_src` values below in `outputs/slide_XX.html`; keep `workspace_path` only for audit/debugging.",
            "Use one of these choices and write an audit comment near the top of the HTML:",
            f"`<!-- template-shell-use: replay reason: shell {shell.shell_id} supports this page -->`",
            f"`<!-- template-shell-use: semantic_only reason: shell {shell.shell_id} would reduce readability -->`",
            f"`<!-- template-shell-use: skip reason: shell {shell.shell_id} conflicts with bound asset -->`",
            "",
        ]
    )
    snippet_lines = [
        "<div class=\"template-shell\" aria-hidden=\"true\">",
    ]
    if primary_bg:
        primary_bg_html = _asset_html_src(background_assets[0])
        snippet_lines.append(
            f"  <img class=\"template-shell-bg\" src=\"{primary_bg_html}\" alt=\"\" />"
        )
    for idx, item in enumerate(decorative_layers[:8], 1):
        src = _asset_html_src(item)
        if src:
            snippet_lines.append(
                f"  <img class=\"template-shell-deco deco-{idx}\" src=\"{src}\" alt=\"\" />"
            )
    snippet_lines.append("</div>")
    css_lines = [
        ".slide { position:relative; width:1280px; height:720px; overflow:hidden; }",
        ".template-shell { position:absolute; inset:0; z-index:0; pointer-events:none; }",
        ".template-shell-bg { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }",
        ".content-layer { position:absolute; inset:0; z-index:2; }",
        ".surface { background:#fff; color:#1f2937; border-radius:18px; }",
    ]
    for idx, item in enumerate(decorative_layers[:8], 1):
        css_lines.append(
            f".deco-{idx} {{ position:absolute; {_geometry_style(item.get('geometry'))} object-fit:contain; }}"
        )
    lines.extend(
        [
            "```html",
            "\n".join(snippet_lines),
            "```",
            "```css",
            "\n".join(css_lines),
            "```",
            "",
            "### Use / Skip Rules",
            "- Choose `replay` when shell assets do not contain old sample text and do not block the bound figure/table/formula.",
            "- Choose `semantic_only` when a readable content surface is more important than exact shell replay.",
            "- Choose `skip` only when the shell asset is missing, stale, or would hide required content; include the reason in the audit comment.",
            "- If you accidentally use `template_shell/assets/...`, `write_html_file` will normalize it, but the preferred source in HTML is `../template_shell/assets/...`.",
            "- Always put dense body text, formulas, and tables on a readable `.surface` unless it is a cover/title-only page.",
        ]
    )
    return "\n".join(lines)


@mcp.tool()
def query_template_shell(layout_name: str) -> str:
    return _query_template_shell_impl(layout_name)


@mcp.tool()
def recommend_template_layout(page_brief: str) -> str:
    """Recommend the best canonical layout for a page brief.

    Args:
        page_brief: JSON string or plain text describing page purpose, title,
            body shape, and whether the page contains figures/tables/formulas.
    """
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Cannot recommend template layouts outside template-driven mode."

    payload: dict[str, Any]
    try:
        parsed = json.loads(page_brief)
        payload = parsed if isinstance(parsed, dict) else {}
    except Exception:
        payload = {"title": page_brief, "body": page_brief}

    brief = PageBrief(
        page_index=int(payload.get("page_index", 1) or 1),
        title=str(payload.get("title", "") or ""),
        body=str(payload.get("body", "") or ""),
        raw_markdown=str(payload.get("raw_markdown", "") or payload.get("body", "") or ""),
        page_purpose=str(payload.get("page_purpose", "content") or "content"),
        content_shape=str(payload.get("content_shape", "title_body") or "title_body"),
        asset_kinds=[str(item) for item in (payload.get("asset_kinds") or []) if str(item).strip()],
        formula_count=int(payload.get("formula_count", 0) or 0),
        table_count=int(payload.get("table_count", 0) or 0),
        image_count=int(payload.get("image_count", 0) or 0),
        bound_asset_kind=str(payload.get("bound_asset_kind", "none") or "none"),
        bound_asset_path=str(payload.get("bound_asset_path", "") or ""),
        visual_requirement=str(payload.get("visual_requirement", "none") or "none"),
        formula_snippet=str(payload.get("formula_snippet", "") or ""),
    )
    planner = TemplateLayoutPlanner(builder.skill)
    rec = planner.recommend_layout(brief)
    return json.dumps(
        {
            "selected_layout": rec.selected_layout,
            "alternatives": rec.alternatives,
            "fit_score": rec.fit_score,
            "slot_assignment_preview": rec.slot_assignment_preview,
            "rejection_reasons": rec.rejection_reasons,
            "why_selected": rec.why_selected,
            "visual_requirement": brief.visual_requirement,
            "bound_asset_kind": brief.bound_asset_kind,
            "bound_asset_path": brief.bound_asset_path,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def query_layout_geometry(layout_name: str) -> str:
    """Implementation details."""
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Cannot query template geometry outside template-driven mode."
    canonical_layout, available_layouts = _resolve_canonical_layout(builder, layout_name)
    if not canonical_layout:
        return (
            f"Error: TEMPLATE_LAYOUT_NOT_FOUND. `{layout_name}` is not a canonical "
            "layout in the active template. Available layouts:\n"
            + _layout_list_markdown(available_layouts)
        )


    sg_path = getattr(builder.skill, "shape_geometry_path", "") or ""
    if sg_path:
        result = _geometry_from_file(sg_path, canonical_layout, builder)
        if result:
            return result


    return builder.get_layout_geometry(canonical_layout)


def _geometry_from_file(
    sg_path: str, layout_name: str, builder: "TemplateGuideBuilder"
) -> str | None:
    """Implementation details."""
    try:
        p = Path(sg_path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))

        layout = resolve_canonical_layout(builder.skill, layout_name)
        if layout is None:
            return None

        example_slides = list(layout.sample_slide_indices or [])
        slide_idx = str(example_slides[0] - 1) if example_slides else "0"
        slide_shapes = data.get(slide_idx)
        if not slide_shapes:
            return None

        editable_slots = list(layout.slots or [])
        if not editable_slots:
            return None

        sections = [f"## Exact geometry: {layout.name}", ""]
        sections.append("| # | Slot | Role | Type | Left % | Top % | Width % | Height % | Capacity hint |")
        sections.append("|---|------|------|------|-----|-----|-----|-----|----------|")
        for i, slot in enumerate(editable_slots, 1):
            geo = slot.geometry or {}
            hint = slot.capacity_hint or {}
            char_hint = hint.get("suggested_max_chars") or hint.get("sample_char_count") or "—"
            sections.append(
                f"| {i} | {slot.name or '?'} | {slot.role or 'slot'} | {slot.type or 'text'} "
                f"| {geo.get('left_pct', 0):.1f} | {geo.get('top_pct', 0):.1f} "
                f"| {geo.get('width_pct', 0):.1f} | {geo.get('height_pct', 0):.1f} "
                f"| {char_hint} |"
            )
        return "\n".join(sections)
    except Exception:
        return None


@mcp.tool()
def query_image_info(image_name: str) -> str:
    """Implementation details."""
    builder = _ensure_context()
    if not builder:
        return "Error: TEMPLATE_RUNTIME_INACTIVE. Cannot query template images outside template-driven mode."

    image_stats = builder.skill.image_stats or {}


    stats = image_stats.get(image_name)


    if not stats:
        for key, val in image_stats.items():
            if image_name.lower() in key.lower():
                image_name = key
                stats = val
                break

    if not stats:
        available = list(image_stats.keys())[:5]
        hint = "\n".join(f"- `{k}`" for k in available) if available else "No image data."
        return f"Error: IMAGE_NOT_FOUND. Image `{image_name}` does not exist.\n\nAvailable images:\n{hint}"

    appear_times = stats.get("appear_times", 1)
    relative_area = stats.get("relative_area", 0)
    size = stats.get("size", [0, 0])


    is_decorative = (appear_times >= 3 and relative_area < 5) or (appear_times >= 2 and relative_area > 20)
    img_type = "decorative (do not replace)" if is_decorative else "content (replaceable)"

    lines = [
        f"## Image information: {image_name}",
        "",
        f"- **Dimensions**: {size[0]} x {size[1]}",
        f"- **Occurrences**: {appear_times}",
        f"- **Relative area**: {relative_area:.1f}%",
        f"- **Type**: {img_type}",
    ]

    if stats.get("slide_numbers"):
        lines.append(f"- **Slide numbers**: {stats['slide_numbers']}")


    template_dir = getattr(builder.skill, "template_dir", "") or ""
    if template_dir:
        img_path = Path(template_dir) / "images" / image_name
        if img_path.exists():
            lines.append(f"- **File path**: `{img_path}`")
        else:
            lines.append(f"- **File path**: file does not exist ({img_path})")

    return "\n".join(lines)
