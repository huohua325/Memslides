from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock


DECK_EXECUTION_STATE_FILE = "deck_execution_state.json"


def _workspace(workspace: Path | str | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).resolve()
    env_workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
    if env_workspace:
        return Path(env_workspace).resolve()
    return Path.cwd().resolve()


def state_path(workspace: Path | str | None = None) -> Path:
    return _workspace(workspace) / DECK_EXECUTION_STATE_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state(workspace: Path | str | None = None) -> dict[str, Any]:
    ws = _workspace(workspace)
    return {
        "version": 1,
        "workspace": str(ws),
        "created_at": _now(),
        "updated_at": _now(),
        "expected_slide_count": 0,
        "slide_dir": "outputs",
        "mode": "non_template",
        "html_write_counter": 0,
        "slides": {},
        "layout_queries": {},
        "shell_queries": {},
    }


def _load_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_state(path.parent)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else _default_state(path.parent)
    except Exception:
        return _default_state(path.parent)


def _save_unlocked(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_deck_execution_state(workspace: Path | str | None = None) -> dict[str, Any] | None:
    path = state_path(workspace)
    if not path.exists():
        return None
    lock = FileLock(str(path) + ".lock", timeout=5)
    with lock:
        return _load_unlocked(path)


def _slide_path_for_page(slide_dir: str, page: int) -> str:
    return f"{slide_dir.rstrip('/')}/slide_{page:02d}.html"


def _coerce_page(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _coerce_string(value: Any) -> str:
    return str(value or "").strip()


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _profile_page_map(profile_execution_plan: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(profile_execution_plan, dict):
        return result
    for item in profile_execution_plan.get("page_plan", []) or []:
        if not isinstance(item, dict):
            continue
        page = _coerce_page(item.get("page_index"))
        if page <= 0:
            continue
        result[page] = item
    return result


def _component_layout_requirement(required_component: str, layout_bias: str) -> str:
    component = _coerce_string(required_component)
    bias = _coerce_string(layout_bias)
    if component and bias:
        return f"Realize `{component}` with a `{bias}` page bias."
    if component:
        return f"Realize required component: `{component}`."
    if bias:
        return f"Keep page bias anchored in `{bias}`."
    return ""


def _asset_requirements_for_slide(slide: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    visual_requirement = _coerce_string(slide.get("visual_requirement", "none")).lower()
    bound_asset_kind = _coerce_string(slide.get("bound_asset_kind", "none"))
    bound_asset_path = _coerce_string(slide.get("bound_asset_path", ""))
    if visual_requirement == "required":
        if bound_asset_path:
            requirements.append(
                f"Render the bound {bound_asset_kind or 'visual'} asset from `{bound_asset_path}`."
            )
        else:
            requirements.append(
                "Keep this page as a real visual page; do not degrade it into a pure bullet slide."
            )
    elif bound_asset_path:
        requirements.append(f"Prefer the bound asset `{bound_asset_path}` if it remains useful.")
    return requirements


def _layout_requirements_for_slide(slide: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    selected_layout = _coerce_string(slide.get("selected_layout", ""))
    reference_archetype = _coerce_string(slide.get("reference_archetype", ""))
    density_hint = _coerce_string(slide.get("density_hint", ""))
    safe_surface_policy = _coerce_string(slide.get("safe_surface_policy", ""))
    component_hint = _component_layout_requirement(
        _coerce_string(slide.get("required_component", "")),
        _coerce_string(slide.get("layout_bias", "")),
    )
    if selected_layout:
        requirements.append(f"Use canonical layout `{selected_layout}`.")
    if reference_archetype:
        requirements.append(f"Keep reference archetype `{reference_archetype}`.")
    if density_hint and density_hint != "medium":
        requirements.append(f"Match `{density_hint}` information density.")
    if safe_surface_policy:
        requirements.append(f"Respect safe surface policy `{safe_surface_policy}`.")
    if component_hint:
        requirements.append(component_hint)
    if not selected_layout:
        requirements.append("Follow `design_plan.md` for page composition, spacing, and hierarchy.")
    return requirements


def _build_slide_entry(
    *,
    page: int,
    slide_dir: str,
    html_exists: bool,
    slide_seed: dict[str, Any] | None = None,
    profile_page: dict[str, Any] | None = None,
    mode: str = "non_template",
) -> dict[str, Any]:
    seed = slide_seed or {}
    persona = profile_page or {}
    title = _coerce_string(seed.get("title", "")) or _coerce_string(persona.get("page_role", ""))
    entry = {
        "page": page,
        "title": title,
        "file": _slide_path_for_page(slide_dir, page),
        "selected_layout": _coerce_string(seed.get("selected_layout", "")),
        "reference_archetype": _coerce_string(seed.get("reference_archetype", "")),
        "density_hint": _coerce_string(seed.get("density_hint", "medium")) or "medium",
        "safe_surface_policy": _coerce_string(seed.get("safe_surface_policy", "")),
        "visual_requirement": _coerce_string(seed.get("visual_requirement", "none")) or "none",
        "bound_asset_kind": _coerce_string(seed.get("bound_asset_kind", "none")) or "none",
        "bound_asset_path": _coerce_string(seed.get("bound_asset_path", "")),
        "layout_queried": False,
        "shell_queried": False,
        "html_written": html_exists,
        "inspected": False,
        "inspect_passed": False,
        "accepted": False,
        "status": "html_written" if html_exists else "planned",
        "contract_mode": mode,
        "page_role": _coerce_string(persona.get("page_role", "")),
        "persona_signal": _coerce_string(persona.get("persona_signal", "")),
        "manuscript_anchor": _coerce_string(persona.get("manuscript_anchor", "")),
        "required_component": _coerce_string(persona.get("required_component", "")),
        "layout_bias": _coerce_string(persona.get("layout_bias", "")) or "single_focus",
        "hard_requirements": _coerce_string_list(persona.get("hard_requirements", [])),
        "style_requirements": persona.get("style_requirements", {}) if isinstance(persona.get("style_requirements"), dict) else {},
        "component_requirements": _coerce_string_list(persona.get("component_requirements", [])),
        "soft_signals": _coerce_string_list(persona.get("soft_signals", [])),
        "source_boundary_notes": _coerce_string_list(persona.get("source_boundary_notes", [])),
        "must_preserve": _coerce_string_list(persona.get("must_preserve", [])),
        "nice_to_have": _coerce_string_list(persona.get("nice_to_have", [])),
        "layout_requirements": [],
        "asset_requirements": [],
        "persona_status": "pending",
        "persona_retry_count": 0,
        "persona_repair_focus": "",
        "persona_unresolved": False,
    }
    entry["layout_requirements"] = _layout_requirements_for_slide(entry)
    entry["asset_requirements"] = _asset_requirements_for_slide(entry)
    return entry


def initialize_deck_execution_state(
    workspace: Path | str,
    layout_mapping_path: Path | str | None = None,
    *,
    expected_slide_count: int | None = None,
    slide_dir: str = "outputs",
    profile_execution_plan: dict[str, Any] | None = None,
) -> Path:
    """Create a concise page work queue for template-driven DeckDesigner runs."""

    ws = _workspace(workspace)
    path = state_path(ws)
    mapping: dict[str, Any] = {}
    if layout_mapping_path:
        try:
            mapping = yaml.safe_load(Path(layout_mapping_path).read_text(encoding="utf-8")) or {}
            if not isinstance(mapping, dict):
                mapping = {}
        except Exception:
            mapping = {}

    slides_payload = mapping.get("slides", []) if isinstance(mapping, dict) else []
    profile_by_page = _profile_page_map(profile_execution_plan)
    mode = "template" if layout_mapping_path else "non_template"
    slides: dict[str, dict[str, Any]] = {}
    for idx, slide in enumerate(slides_payload or [], 1):
        if not isinstance(slide, dict):
            continue
        page = _coerce_page(slide.get("page")) or idx
        rel_path = _slide_path_for_page(slide_dir, page)
        html_path = ws / rel_path
        slides[str(page)] = _build_slide_entry(
            page=page,
            slide_dir=slide_dir,
            html_exists=html_path.exists(),
            slide_seed=slide,
            profile_page=profile_by_page.get(page),
            mode=mode,
        )

    if expected_slide_count is None:
        expected_slide_count = len(slides)
    if expected_slide_count and not slides:
        for page in range(1, int(expected_slide_count) + 1):
            rel_path = _slide_path_for_page(slide_dir, page)
            html_path = ws / rel_path
            slides[str(page)] = _build_slide_entry(
                page=page,
                slide_dir=slide_dir,
                html_exists=html_path.exists(),
                slide_seed={},
                profile_page=profile_by_page.get(page),
                mode=mode,
            )

    state = _default_state(ws)
    state["expected_slide_count"] = int(expected_slide_count or len(slides) or 0)
    state["slide_dir"] = slide_dir
    state["mode"] = mode
    state["slides"] = slides
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=5)
    with lock:
        _save_unlocked(path, state)
    return path


def _mutate_state(
    workspace: Path | str | None,
    mutator,
) -> dict[str, Any]:
    path = state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=5)
    with lock:
        state = _load_unlocked(path)
        mutator(state)
        _save_unlocked(path, state)
        return state


def record_layout_query(
    layout_name: str,
    workspace: Path | str | None = None,
    *,
    repeat_threshold: int = 2,
) -> dict[str, Any]:
    """Record a layout query and report whether it is repeated without progress."""

    name = str(layout_name or "").strip()
    if not name:
        return {"repeated": False, "count": 0, "repeated_without_write": 0}

    result: dict[str, Any] = {}

    def _update(state: dict[str, Any]) -> None:
        current_write_counter = int(state.get("html_write_counter", 0) or 0)
        queries = state.setdefault("layout_queries", {})
        entry = queries.setdefault(
            name,
            {
                "count": 0,
                "repeated_without_write": 0,
                "html_write_counter_at_last_query": -1,
            },
        )
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        last_counter = entry.get("html_write_counter_at_last_query", -1)
        try:
            last_counter_int = int(last_counter)
        except Exception:
            last_counter_int = -1
        if last_counter_int < 0:
            entry["repeated_without_write"] = 0
            entry["html_write_counter_at_last_query"] = current_write_counter
        elif last_counter_int == current_write_counter:
            entry["repeated_without_write"] = int(entry.get("repeated_without_write", 0) or 0) + 1
        else:
            entry["repeated_without_write"] = 0
            entry["html_write_counter_at_last_query"] = current_write_counter

        for slide in (state.get("slides", {}) or {}).values():
            if str(slide.get("selected_layout", "") or "") == name:
                slide["layout_queried"] = True

        result.update(
            {
                "layout_name": name,
                "count": entry["count"],
                "repeated_without_write": entry["repeated_without_write"],
                "repeated": entry["repeated_without_write"] > repeat_threshold,
                "html_write_counter": current_write_counter,
            }
        )

    _mutate_state(workspace, _update)
    result["progress"] = deck_progress_summary(workspace)
    return result


def record_shell_query(layout_name: str, workspace: Path | str | None = None) -> None:
    name = str(layout_name or "").strip()
    if not name:
        return

    def _update(state: dict[str, Any]) -> None:
        queries = state.setdefault("shell_queries", {})
        queries[name] = int(queries.get(name, 0) or 0) + 1
        for slide in (state.get("slides", {}) or {}).values():
            if str(slide.get("selected_layout", "") or "") == name:
                slide["shell_queried"] = True

    _mutate_state(workspace, _update)


_SLIDE_RE = re.compile(r"slide_(\d+)\.html$", re.IGNORECASE)


def _page_from_path(path: Path) -> int | None:
    match = _SLIDE_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def record_html_written(path: Path | str, workspace: Path | str | None = None) -> None:
    html_path = Path(path)
    page = _page_from_path(html_path)
    if page is None:
        return

    def _update(state: dict[str, Any]) -> None:
        state["html_write_counter"] = int(state.get("html_write_counter", 0) or 0) + 1
        slides = state.setdefault("slides", {})
        slide = slides.setdefault(
            str(page),
            {
                "page": page,
                "file": _slide_path_for_page(str(state.get("slide_dir", "outputs") or "outputs"), page),
                "selected_layout": "",
            },
        )
        slide["html_written"] = True
        slide["inspected"] = False
        slide["inspect_passed"] = False
        slide["accepted"] = False
        slide["status"] = "html_written"
        if not state.get("expected_slide_count"):
            state["expected_slide_count"] = max(page, len(slides))

    _mutate_state(workspace, _update)


def record_slide_inspected(
    path: Path | str,
    *,
    success: bool,
    workspace: Path | str | None = None,
) -> None:
    html_path = Path(path)
    page = _page_from_path(html_path)
    if page is None:
        return

    def _update(state: dict[str, Any]) -> None:
        slides = state.setdefault("slides", {})
        slide = slides.setdefault(
            str(page),
            {
                "page": page,
                "file": _slide_path_for_page(str(state.get("slide_dir", "outputs") or "outputs"), page),
                "selected_layout": "",
            },
        )
        slide["html_written"] = True
        slide["inspected"] = True
        slide["inspect_passed"] = bool(success)
        slide["accepted"] = bool(success)
        slide["status"] = "accepted" if success else "inspect_failed"

    _mutate_state(workspace, _update)


def record_persona_verdict(
    path: Path | str,
    *,
    verdict: str,
    repair_focus: str = "",
    workspace: Path | str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    html_path = Path(path)
    page = _page_from_path(html_path)
    if page is None:
        return

    normalized_verdict = _coerce_string(verdict).lower() or "pass"
    detail_payload = details if isinstance(details, dict) else {}

    def _update(state: dict[str, Any]) -> None:
        slides = state.setdefault("slides", {})
        slide = slides.setdefault(
            str(page),
            {
                "page": page,
                "file": _slide_path_for_page(str(state.get("slide_dir", "outputs") or "outputs"), page),
                "selected_layout": "",
            },
        )
        slide["persona_status"] = normalized_verdict
        slide["persona_last_details"] = detail_payload
        slide["persona_repair_focus"] = _coerce_string(repair_focus)
        inspect_passed = bool(slide.get("inspect_passed"))
        if normalized_verdict == "retry_required":
            slide["persona_retry_count"] = int(slide.get("persona_retry_count", 0) or 0) + 1
            slide["persona_unresolved"] = False
            slide["accepted"] = False
            slide["status"] = "persona_retry_required"
        elif normalized_verdict == "unresolved_release":
            slide["persona_unresolved"] = True
            slide["accepted"] = inspect_passed or bool(slide.get("accepted"))
            slide["status"] = "accepted_with_persona_gap"
        else:
            slide["persona_unresolved"] = False
            slide["accepted"] = inspect_passed or bool(slide.get("accepted"))
            slide["status"] = "accepted" if slide.get("accepted") else "persona_pending"
            slide["persona_repair_focus"] = _coerce_string(repair_focus)

    _mutate_state(workspace, _update)


def record_persona_repair_backup(
    path: Path | str,
    *,
    backup_path: str,
    backup_hash: str,
    workspace: Path | str | None = None,
) -> None:
    html_path = Path(path)
    page = _page_from_path(html_path)
    if page is None:
        return

    def _update(state: dict[str, Any]) -> None:
        slides = state.setdefault("slides", {})
        slide = slides.setdefault(
            str(page),
            {
                "page": page,
                "file": _slide_path_for_page(str(state.get("slide_dir", "outputs") or "outputs"), page),
                "selected_layout": "",
            },
        )
        slide["persona_repair_backup"] = {
            "path": _coerce_string(backup_path),
            "content_hash": _coerce_string(backup_hash),
            "created_at": _now(),
        }

    _mutate_state(workspace, _update)


def current_page_contract(workspace: Path | str | None = None) -> dict[str, Any]:
    summary = deck_progress_summary(workspace)
    if not summary.get("active"):
        return {}
    page = summary.get("next_page", {}) or {}
    if not page:
        return {}
    return {
        "page_index": int(page.get("page", 0) or 0),
        "page_title": _coerce_string(page.get("title", "")),
        "page_role": _coerce_string(page.get("page_role", "")),
        "persona_signal": _coerce_string(page.get("persona_signal", "")),
        "manuscript_anchor": _coerce_string(page.get("manuscript_anchor", "")),
        "required_component": _coerce_string(page.get("required_component", "")),
        "layout_bias": _coerce_string(page.get("layout_bias", "")),
        "hard_requirements": _coerce_string_list(page.get("hard_requirements", [])),
        "style_requirements": page.get("style_requirements", {}) if isinstance(page.get("style_requirements"), dict) else {},
        "component_requirements": _coerce_string_list(page.get("component_requirements", [])),
        "soft_signals": _coerce_string_list(page.get("soft_signals", [])),
        "source_boundary_notes": _coerce_string_list(page.get("source_boundary_notes", [])),
        "must_preserve": _coerce_string_list(page.get("must_preserve", [])),
        "layout_requirements": _coerce_string_list(page.get("layout_requirements", [])),
        "asset_requirements": _coerce_string_list(page.get("asset_requirements", [])),
        "repair_focus": _coerce_string(page.get("persona_repair_focus", "")),
        "persona_status": _coerce_string(page.get("persona_status", "pending")),
        "persona_retry_count": int(page.get("persona_retry_count", 0) or 0),
        "contract_mode": _coerce_string(page.get("contract_mode", "")),
    }


def deck_progress_summary(workspace: Path | str | None = None) -> dict[str, Any]:
    state = load_deck_execution_state(workspace)
    if not state:
        return {
            "active": False,
            "expected_slide_count": 0,
            "written_count": 0,
            "accepted_count": 0,
            "complete": False,
            "next_action": "",
        }

    slides = state.get("slides", {}) or {}
    ordered = sorted(
        (slide for slide in slides.values() if isinstance(slide, dict)),
        key=lambda item: int(item.get("page", 0) or 0),
    )
    expected = int(state.get("expected_slide_count", 0) or len(ordered) or 0)
    written = [slide for slide in ordered if slide.get("html_written")]
    accepted = [slide for slide in ordered if slide.get("accepted")]

    next_action = ""
    next_page: dict[str, Any] | None = None
    for slide in ordered:
        if not slide.get("html_written"):
            next_page = slide
            next_action = "write_html"
            break
    if next_page is None:
        for slide in ordered:
            if slide.get("html_written") and not slide.get("inspected"):
                next_page = slide
                next_action = "inspect_or_fix"
                break
    if next_page is None:
        for slide in ordered:
            if slide.get("inspected") and not slide.get("inspect_passed"):
                next_page = slide
                next_action = "inspect_or_fix"
                break
    if next_page is None:
        for slide in ordered:
            if not slide.get("accepted"):
                next_page = slide
                next_action = "inspect_or_fix"
                break
    if next_page is None and expected and len(accepted) >= expected:
        next_action = "finalize"

    return {
        "active": True,
        "mode": str(state.get("mode", "non_template") or "non_template"),
        "expected_slide_count": expected,
        "written_count": len(written),
        "accepted_count": len(accepted),
        "complete": bool(expected and len(accepted) >= expected),
        "next_action": next_action,
        "next_page": next_page or {},
        "slides": ordered,
    }


def render_deck_progress_prompt(workspace: Path | str | None = None) -> str:
    summary = deck_progress_summary(workspace)
    if not summary.get("active"):
        return ""

    next_page = summary.get("next_page", {}) or {}
    next_line = "finalize the deck"
    if summary.get("next_action") == "write_html":
        next_line = (
            f"write `{next_page.get('file')}` using layout "
            f"`{next_page.get('selected_layout') or 'from layout_mapping.yaml'}`"
        )
    elif summary.get("next_action") == "inspect_or_fix":
        next_line = f"inspect or fix `{next_page.get('file')}`"

    slide_lines = []
    for slide in (summary.get("slides") or [])[:12]:
        slide_lines.append(
            f"- p{slide.get('page')}: layout={slide.get('selected_layout') or '?'} "
            f"html={bool(slide.get('html_written'))} accepted={bool(slide.get('accepted'))} "
            f"persona={slide.get('persona_status') or 'pending'}"
        )
    current_contract = current_page_contract(workspace)
    contract_lines: list[str] = []
    if current_contract:
        contract_lines.extend(
            [
                "<current_page_contract>",
                f"page_index={current_contract.get('page_index')}",
                f"page_title={current_contract.get('page_title') or '(unspecified)'}",
                f"page_role={current_contract.get('page_role') or '(unspecified)'}",
                f"persona_signal={current_contract.get('persona_signal') or '(unspecified)'}",
                f"manuscript_anchor={current_contract.get('manuscript_anchor') or '(unspecified)'}",
                f"required_component={current_contract.get('required_component') or '(unspecified)'}",
                f"layout_bias={current_contract.get('layout_bias') or '(unspecified)'}",
                f"persona_status={current_contract.get('persona_status') or 'pending'}",
            ]
        )
        must_preserve = current_contract.get("must_preserve") or []
        if must_preserve:
            contract_lines.append("must_preserve:")
            contract_lines.extend(f"- {item}" for item in must_preserve[:4])
        hard_requirements = current_contract.get("hard_requirements") or []
        if hard_requirements:
            contract_lines.append("hard_requirements:")
            contract_lines.extend(f"- {item}" for item in hard_requirements[:4])
        style_requirements = current_contract.get("style_requirements") or {}
        if isinstance(style_requirements, dict) and style_requirements:
            contract_lines.append(
                "style_requirements="
                + json.dumps(style_requirements, ensure_ascii=False)[:500]
            )
        component_requirements = current_contract.get("component_requirements") or []
        if component_requirements:
            contract_lines.append("component_requirements:")
            contract_lines.extend(f"- {item}" for item in component_requirements[:4])
        soft_signals = current_contract.get("soft_signals") or []
        if soft_signals:
            contract_lines.append("soft_persona_signals:")
            contract_lines.extend(f"- {item}" for item in soft_signals[:4])
        layout_requirements = current_contract.get("layout_requirements") or []
        if layout_requirements:
            contract_lines.append("layout_requirements:")
            contract_lines.extend(f"- {item}" for item in layout_requirements[:4])
        asset_requirements = current_contract.get("asset_requirements") or []
        if asset_requirements:
            contract_lines.append("asset_requirements:")
            contract_lines.extend(f"- {item}" for item in asset_requirements[:4])
        repair_focus = _coerce_string(current_contract.get("repair_focus", ""))
        if repair_focus:
            contract_lines.append("repair_protocol=full_page_rewrite_only")
            contract_lines.append(f"repair_focus={repair_focus}")
            contract_lines.append(
                "repair_instruction=replace the complete slide HTML; do not append large fragments with apply_slide_patch"
            )
        contract_lines.append("</current_page_contract>")

    prompt = (
        "<deck_progress>\n"
        f"expected_slides={summary.get('expected_slide_count')} "
        f"written={summary.get('written_count')} accepted={summary.get('accepted_count')}\n"
        f"next_action={next_line}\n"
        + "\n".join(slide_lines)
        + "\n</deck_progress>"
    )
    if contract_lines:
        prompt += "\n" + "\n".join(contract_lines)
    return prompt


__all__ = [
    "DECK_EXECUTION_STATE_FILE",
    "deck_progress_summary",
    "initialize_deck_execution_state",
    "load_deck_execution_state",
    "current_page_contract",
    "record_html_written",
    "record_layout_query",
    "record_persona_verdict",
    "record_shell_query",
    "record_slide_inspected",
    "render_deck_progress_prompt",
    "state_path",
]
