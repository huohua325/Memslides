from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from memslides.utils.typings import InputRequest
from collections.abc import Iterable
from bs4 import BeautifulSoup


_DESIGN_PLAN_CANDIDATES = (
    "design_plan.md",
    "design-plan.md",
    "outputs/design_plan.md",
    "outputs/design-plan.md",
)


def find_existing_design_plan_rel(workspace: Path) -> str | None:
    for candidate in _DESIGN_PLAN_CANDIDATES:
        if (workspace / candidate).exists():
            return candidate
    return None


def render_profile_execution_contract_markdown(
    profile_execution_contract: dict[str, Any] | None,
) -> str:
    if not profile_execution_contract:
        return ""

    def _markdown_bullets(items: list[str], *, fallback: str = "- None") -> str:
        clean_items = [str(item).strip() for item in items if str(item or "").strip()]
        if not clean_items:
            return fallback
        return "\n".join(f"- {item}" for item in clean_items)

    deck_spine = [
        str(item).strip()
        for item in profile_execution_contract.get("deck_spine", []) or []
        if str(item or "").strip()
    ]
    style_contract = [
        str(item).strip()
        for item in profile_execution_contract.get("style_contract", []) or []
        if str(item or "").strip()
    ]
    content_contract = [
        str(item).strip()
        for item in profile_execution_contract.get("content_contract", []) or []
        if str(item or "").strip()
    ]
    source_boundary_contract = [
        str(item).strip()
        for item in profile_execution_contract.get("source_boundary_contract", []) or []
        if str(item or "").strip()
    ]
    proxy_realization_rules = [
        str(item).strip()
        for item in profile_execution_contract.get("proxy_realization_rules", []) or []
        if str(item or "").strip()
    ]
    do_not_force = [
        str(item).strip()
        for item in profile_execution_contract.get("do_not_force", []) or []
        if str(item or "").strip()
    ]
    hard_requirements = [
        str(item).strip()
        for item in profile_execution_contract.get("hard_requirements", []) or []
        if str(item or "").strip()
    ]
    component_requirements = [
        str(item).strip()
        for item in profile_execution_contract.get("component_requirements", []) or []
        if str(item or "").strip()
    ]
    soft_signals = [
        str(item).strip()
        for item in profile_execution_contract.get("soft_signals", []) or []
        if str(item or "").strip()
    ]
    style_requirements = (
        profile_execution_contract.get("style_requirements")
        if isinstance(profile_execution_contract.get("style_requirements"), dict)
        else {}
    )

    return (
        "## Profile-Derived Deck Contract\n"
        f"- Planning Focus: {profile_execution_contract.get('planning_focus', 'decision_brief')}\n"
        "- Requirement: this section is runtime-compiled from active profile preferences and must be preserved in the final design plan.\n\n"
        "### Deck Spine\n"
        f"{_markdown_bullets(deck_spine)}\n\n"
        "### Style Contract\n"
        f"{_markdown_bullets(style_contract)}\n\n"
        "### Content Contract\n"
        f"{_markdown_bullets(content_contract)}\n\n"
        "### Preference Realization Contract\n"
        f"- Hard Requirements:\n{_markdown_bullets(hard_requirements)}\n"
        f"- Style Requirements: {json.dumps(style_requirements, ensure_ascii=False) if style_requirements else '{}'}\n"
        f"- Component Requirements:\n{_markdown_bullets(component_requirements)}\n"
        f"- Soft Persona Signals:\n{_markdown_bullets(soft_signals)}\n\n"
        "### Source Boundary Contract\n"
        f"{_markdown_bullets(source_boundary_contract)}\n\n"
        "### Proxy Realization Rules\n"
        f"{_markdown_bullets(proxy_realization_rules)}\n\n"
        "### Do Not Force\n"
        f"{_markdown_bullets(do_not_force)}\n"
    )


def render_profile_execution_page_obligations_markdown(
    profile_execution_contract: dict[str, Any] | None,
) -> str:
    if not profile_execution_contract:
        return ""

    obligations = profile_execution_contract.get("page_obligations", []) or []
    lines = [
        "## Page-Level Persona Obligations",
        "- Requirement: expand these obligations into the final page mapping before writing any slide HTML.",
        "- Requirement: keep page roles persona-specific; do not collapse them into generic page types only.",
        "",
    ]
    if not obligations:
        lines.append("- No page obligations compiled.")
        return "\n".join(lines) + "\n"

    for idx, item in enumerate(obligations, start=1):
        if not isinstance(item, dict):
            continue
        page_role = str(item.get("page_role", "") or f"page_{idx}").strip()
        required_signal = str(item.get("required_signal", "") or "persona-aligned signal").strip()
        preferred_component = str(item.get("preferred_component", "") or "fit-for-content archetype").strip()
        priority = str(item.get("priority", "") or "medium").strip()
        lines.extend([
            f"{idx}. Persona Page Role: {page_role}",
            f"- Required Signal: {required_signal}",
            f"- Preferred Component / Archetype: {preferred_component}",
            f"- Priority: {priority}",
            "- Avoid Instead: generic page patterns that drop the persona signal or contradict the deck spine",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_profile_execution_plan_markdown(
    profile_execution_plan: dict[str, Any] | None,
) -> str:
    if not profile_execution_plan:
        return ""

    lines = [
        "## Persona Page Plan",
        "- Requirement: treat this as the page-by-page persona execution queue before writing any slide HTML.",
        "- Requirement: preserve source truth; if a persona signal is not source-supported, realize it through framing, hierarchy, sequencing, or component choice instead of inventing facts.",
        "",
    ]

    global_notes = [
        str(item).strip()
        for item in profile_execution_plan.get("global_execution_notes", []) or []
        if str(item or "").strip()
    ]
    if global_notes:
        lines.append("### Global Execution Notes")
        lines.extend(f"- {item}" for item in global_notes)
        lines.append("")

    for item in profile_execution_plan.get("page_plan", []) or []:
        if not isinstance(item, dict):
            continue
        page_index = int(item.get("page_index", 0) or 0)
        page_role = str(item.get("page_role", "") or f"page_{page_index}").strip()
        persona_signal = str(item.get("persona_signal", "") or "persona-aligned signal").strip()
        manuscript_anchor = str(item.get("manuscript_anchor", "") or "fit-for-source content cluster").strip()
        required_component = str(item.get("required_component", "") or "fit-for-content archetype").strip()
        layout_bias = str(item.get("layout_bias", "") or "single_focus").strip()
        lines.extend(
            [
                f"### Page {page_index}: {page_role}",
                f"- Persona Signal: {persona_signal}",
                f"- Manuscript Anchor: {manuscript_anchor}",
                f"- Required Component / Archetype: {required_component}",
                f"- Layout Bias: {layout_bias}",
            ]
        )
        must_preserve = [
            str(entry).strip()
            for entry in item.get("must_preserve", []) or []
            if str(entry or "").strip()
        ]
        nice_to_have = [
            str(entry).strip()
            for entry in item.get("nice_to_have", []) or []
            if str(entry or "").strip()
        ]
        if must_preserve:
            lines.append("- Must Preserve:")
            lines.extend(f"  - {entry}" for entry in must_preserve)
        hard_requirements = [
            str(entry).strip()
            for entry in item.get("hard_requirements", []) or []
            if str(entry or "").strip()
        ]
        style_requirements = item.get("style_requirements") if isinstance(item.get("style_requirements"), dict) else {}
        component_requirements = [
            str(entry).strip()
            for entry in item.get("component_requirements", []) or []
            if str(entry or "").strip()
        ]
        soft_signals = [
            str(entry).strip()
            for entry in item.get("soft_signals", []) or []
            if str(entry or "").strip()
        ]
        if hard_requirements:
            lines.append("- Hard Requirements:")
            lines.extend(f"  - {entry}" for entry in hard_requirements[:4])
        if style_requirements:
            lines.append("- Style Requirements:")
            lines.append(f"  - {json.dumps(style_requirements, ensure_ascii=False)[:500]}")
        if component_requirements:
            lines.append("- Component Requirements:")
            lines.extend(f"  - {entry}" for entry in component_requirements[:4])
        if soft_signals:
            lines.append("- Soft Persona Signals:")
            lines.extend(f"  - {entry}" for entry in soft_signals[:4])
        if nice_to_have:
            lines.append("- Nice To Have:")
            lines.extend(f"  - {entry}" for entry in nice_to_have)
        lines.append("")

    fallback_rules = [
        str(item).strip()
        for item in profile_execution_plan.get("fallback_rules", []) or []
        if str(item or "").strip()
    ]
    if fallback_rules:
        lines.append("### Fallback Rules")
        lines.extend(f"- {item}" for item in fallback_rules)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _persona_markers_for_bias(layout_bias: str) -> set[str]:
    bias = str(layout_bias or "").strip().lower()
    marker_map = {
        "comparison": {"comparison", "compare", "matrix", "table", "risk", "trade-off", "对比", "比较", "矩阵", "风险"},
        "process": {"process", "pipeline", "step", "workflow", "timeline", "流程", "步骤", "机制", "架构"},
        "evidence": {"evidence", "experiment", "result", "proof", "figure", "chart", "实验", "结果", "证据", "图"},
        "recap": {"recap", "summary", "takeaway", "key takeaway", "总结", "回顾", "要点"},
        "decision": {"decision", "recommendation", "next step", "action", "adopt", "结论", "建议", "下一步", "采用"},
        "single_focus": {"focus", "hero", "headline", "opening", "核心", "重点"},
    }
    return marker_map.get(bias, set())


def _component_markers(required_component: str) -> set[str]:
    text = str(required_component or "").strip().lower()
    markers = set()
    if any(token in text for token in ("table", "matrix", "comparison", "rank")):
        markers |= {"table", "matrix", "comparison", "对比", "矩阵"}
    if any(token in text for token in ("chart", "figure", "visual", "image", "diagram")):
        markers |= {"figure", "chart", "visual", "image", "图", "图表"}
    if any(token in text for token in ("process", "pipeline", "step", "timeline", "roadmap", "walkthrough")):
        markers |= {"process", "pipeline", "step", "timeline", "步骤", "流程"}
    if any(token in text for token in ("summary", "recap", "takeaway")):
        markers |= {"summary", "recap", "takeaway", "总结", "回顾"}
    if any(token in text for token in ("headline", "hero", "kpi", "callout")):
        markers |= {"headline", "hero", "title", "callout", "标题"}
    return markers


def _normalize_profile_css_value(property_name: str, value: str) -> str:
    prop = _normalize_css_property_name(str(property_name or ""))
    text = str(value or "").strip().lower().replace("!important", "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if prop in {"color", "background-color", "background"}:
        rgb = _color_token_to_rgb(text)
        if rgb is not None:
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return text.replace(" ", "")
    if prop == "font-weight":
        if text in {"bold", "bolder"}:
            return "700"
        if text in {"normal", "regular"}:
            return "400"
        numeric = re.fullmatch(r"\d+(?:\.0+)?", text)
        if numeric:
            return str(round(float(text)))
    if prop == "font-style":
        if "italic" in text:
            return "italic"
        if "oblique" in text:
            return "oblique"
    if prop == "text-decoration":
        tokens = {
            token
            for token in re.split(r"[\s,]+", text.replace(";", " "))
            if token and token not in {"solid", "auto"}
        }
        if "none" in tokens:
            return "none"
        ordered = [token for token in ("underline", "overline", "line-through") if token in tokens]
        return " ".join(ordered) if ordered else text
    return text


def _profile_css_values_equivalent(property_name: str, expected: str, observed: str) -> bool:
    return _normalize_profile_css_value(property_name, expected) == _normalize_profile_css_value(property_name, observed)


def _coerce_profile_style_requirements(*values: Any) -> dict[str, Any]:
    css_values: dict[str, str] = {}
    palette_values: list[str] = []
    soft_tokens: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        raw_css = value.get("css_values")
        if isinstance(raw_css, dict):
            for raw_prop, raw_value in raw_css.items():
                prop = _normalize_css_property_name(str(raw_prop or ""))
                val = str(raw_value or "").strip()
                if prop and val:
                    css_values[prop] = val
        raw_palette = value.get("palette_values")
        if isinstance(raw_palette, (list, tuple, set)):
            palette_values.extend(str(item).strip() for item in raw_palette if str(item or "").strip())
        elif isinstance(raw_palette, str) and raw_palette.strip():
            palette_values.append(raw_palette.strip())
        raw_soft = value.get("soft_tokens")
        if isinstance(raw_soft, (list, tuple, set)):
            soft_tokens.extend(str(item).strip() for item in raw_soft if str(item or "").strip())
        elif isinstance(raw_soft, str) and raw_soft.strip():
            soft_tokens.append(raw_soft.strip())
    result: dict[str, Any] = {}
    if css_values:
        result["css_values"] = css_values
    if palette_values:
        result["palette_values"] = list(dict.fromkeys(palette_values))[:8]
    if soft_tokens:
        result["soft_tokens"] = list(dict.fromkeys(soft_tokens))[:8]
    return result


def _extract_slide_style_observations(slide_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(slide_html or "", "html.parser")
    css_vars: dict[str, str] = {}
    observed: dict[str, list[str]] = {}
    raw_style_text = "\n".join(style_tag.string or style_tag.get_text() or "" for style_tag in soup.find_all("style"))

    declaration_blobs: list[str] = []
    for match in re.finditer(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", raw_style_text, re.DOTALL):
        selectors = str(match.group("selectors") or "").strip()
        if selectors.startswith("@"):
            continue
        declaration_blobs.append(str(match.group("body") or ""))
    for tag in soup.find_all(True):
        style = str(tag.get("style", "") or "").strip()
        if style:
            declaration_blobs.append(style)

    declarations: list[tuple[str, str]] = []
    for blob in declaration_blobs:
        for prop, value in _parse_style_declarations(blob).items():
            normalized_prop = _normalize_css_property_name(prop)
            if not normalized_prop:
                continue
            if normalized_prop.startswith("--"):
                css_vars[normalized_prop] = value
            declarations.append((normalized_prop, value))

    def _resolve(value: str, depth: int = 0) -> str:
        text = str(value or "").strip()
        if depth > 4 or not text:
            return text
        match = re.fullmatch(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+?)\s*)?\)", text, re.IGNORECASE)
        if not match:
            return text
        return _resolve(css_vars.get(match.group(1).strip(), (match.group(2) or "").strip()), depth + 1)

    for prop, value in declarations:
        if prop.startswith("--"):
            continue
        normalized_prop = "background-color" if prop == "background" else prop
        resolved_value = _resolve(value)
        if not resolved_value:
            continue
        observed.setdefault(normalized_prop, []).append(resolved_value)
        if normalized_prop == "background-color":
            observed.setdefault("background", []).append(resolved_value)

    all_values = list(dict.fromkeys(value for values in observed.values() for value in values))
    return {
        "properties": {prop: list(dict.fromkeys(values)) for prop, values in observed.items()},
        "all_values": all_values,
    }


def _evaluate_profile_style_requirements(
    slide_html: str,
    style_requirements: dict[str, Any],
) -> dict[str, Any]:
    css_values = style_requirements.get("css_values", {}) if isinstance(style_requirements, dict) else {}
    palette_values = style_requirements.get("palette_values", []) if isinstance(style_requirements, dict) else []
    observations = _extract_slide_style_observations(slide_html)
    observed_props = observations.get("properties", {}) or {}
    failures: list[str] = []
    matches: list[dict[str, str]] = []

    if isinstance(css_values, dict):
        for raw_prop, raw_expected in css_values.items():
            prop = _normalize_css_property_name(str(raw_prop or ""))
            expected = str(raw_expected or "").strip()
            if not prop or not expected:
                continue
            observed_values = list(observed_props.get(prop, []) or [])
            if prop == "background-color":
                observed_values.extend(observed_props.get("background", []) or [])
            if any(_profile_css_values_equivalent(prop, expected, observed) for observed in observed_values):
                matches.append({"property": prop, "expected": expected})
            else:
                failures.append(f"{prop} expected `{expected}` was not observed in slide CSS")

    palette_matches: list[str] = []
    for expected in palette_values if isinstance(palette_values, list) else []:
        token = str(expected or "").strip()
        if not token:
            continue
        if any(
            _profile_css_values_equivalent("color", token, observed)
            for observed in observations.get("all_values", []) or []
        ):
            palette_matches.append(token)

    return {
        "css_values": css_values if isinstance(css_values, dict) else {},
        "css_value_matches": matches,
        "css_value_failures": failures,
        "palette_values": palette_values if isinstance(palette_values, list) else [],
        "palette_matches": list(dict.fromkeys(palette_matches)),
        "observed_properties": observed_props,
        "style_matched": not failures,
    }


def _semantic_component_signals(slide_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(slide_html or "", "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    class_text = " ".join(
        " ".join(tag.get("class", [])) if isinstance(tag.get("class"), list) else str(tag.get("class", "") or "")
        for tag in soup.find_all(True)
    ).lower()
    aria_text = " ".join(str(tag.get("aria-label", "") or "") for tag in soup.find_all(True)).lower()
    combined = f"{text} {class_text} {aria_text}"
    card_like = soup.select(".card, [class*=card], .tile, [class*=tile], .panel, [class*=panel]")
    chip_like = soup.select(".chip, [class*=chip], .badge, [class*=badge], .pill, [class*=pill], .kpi, [class*=kpi]")
    callout_like = soup.select(".callout, [class*=callout], .summary, [class*=summary], .takeaway, [class*=takeaway], .decision, [class*=decision]")
    grid_like = soup.select(".matrix, [class*=matrix], .grid, [class*=grid], [role=grid]")
    has_table = soup.find("table") is not None
    list_count = len(soup.find_all("li"))
    numeric_short_tokens = re.findall(r"(?:\b\d+(?:\.\d+)?%?\b|top-\d+|roi|kpi|n=|指标|准确率|提升)", combined)
    return {
        "combined_text": combined,
        "has_table": has_table,
        "has_matrix": has_table or bool(grid_like) or any(token in combined for token in ("matrix", "矩阵", "risk", "风险", "comparison", "对比", "比较")),
        "has_kpi_chips": bool(chip_like) or ("kpi" in combined and len(numeric_short_tokens) >= 1) or len(numeric_short_tokens) >= 3,
        "has_cards": len(card_like) >= 2 or any(token in combined for token in ("cards", "卡片", "contribution", "贡献")),
        "has_callout": bool(callout_like) or any(token in combined for token in ("callout", "takeaway", "recommendation", "summary", "结论", "建议", "总结", "要点")),
        "has_process": list_count >= 3 or any(token in combined for token in ("process", "pipeline", "step", "timeline", "流程", "步骤", "机制")),
        "has_visual": soup.find(["img", "svg", "canvas"]) is not None,
        "chip_count": len(chip_like),
        "card_count": len(card_like),
        "list_count": list_count,
    }


def _component_requirement_matched(requirement: str, signals: dict[str, Any]) -> bool:
    text = str(requirement or "").strip().lower()
    if not text or text == "fit-for-content archetype":
        return True
    if any(token in text for token in ("kpi", "chip", "badge", "pill", "指标")):
        return bool(signals.get("has_kpi_chips"))
    if any(token in text for token in ("matrix", "table", "comparison", "rank", "risk", "矩阵", "表格", "对比", "比较", "风险")):
        return bool(signals.get("has_matrix"))
    if any(token in text for token in ("card", "contribution", "value ladder", "卡片", "贡献")):
        return bool(signals.get("has_cards"))
    if any(token in text for token in ("callout", "summary", "takeaway", "decision", "recommend", "结论", "建议", "总结", "要点")):
        return bool(signals.get("has_callout"))
    if any(token in text for token in ("process", "pipeline", "step", "timeline", "roadmap", "walkthrough", "流程", "步骤", "机制")):
        return bool(signals.get("has_process"))
    if any(token in text for token in ("chart", "figure", "visual", "image", "diagram", "图", "图表")):
        return bool(signals.get("has_visual") or signals.get("has_matrix"))
    combined = str(signals.get("combined_text", "") or "")
    return any(token and token in combined for token in re.split(r"[\s+/，,;；]+", text) if len(token) >= 4)


def _slide_observed_signals(slide_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(slide_html or "", "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    headings = [tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2", "h3"]) if tag.get_text(" ", strip=True)]
    has_table = soup.find("table") is not None
    has_visual = soup.find(["img", "svg", "canvas"]) is not None
    list_count = len(soup.find_all("li"))

    observed: set[str] = set()
    if has_table or any(token in text for token in ("compare", "comparison", "matrix", "对比", "矩阵", "风险")):
        observed.add("comparison")
    if list_count >= 3 or any(token in text for token in ("process", "pipeline", "step", "timeline", "流程", "步骤", "机制")):
        observed.add("process")
    if has_visual or has_table or any(token in text for token in ("experiment", "result", "evidence", "figure", "chart", "实验", "结果", "证据")):
        observed.add("evidence")
    if any(token in text for token in ("summary", "recap", "takeaway", "总结", "回顾", "要点")):
        observed.add("recap")
    if any(token in text for token in ("decision", "recommendation", "next step", "action", "结论", "建议", "下一步", "采用")):
        observed.add("decision")
    if headings:
        observed.add("single_focus")

    return {
        "text": text,
        "has_table": has_table,
        "has_visual": has_visual,
        "list_count": list_count,
        "heading_count": len(headings),
        "headings": headings[:4],
        "observed_biases": sorted(observed),
    }


def _extract_template_layout_annotation(slide_html: str) -> str:
    match = re.search(r"<!--\s*template-layout:\s*(.+?)\s*-->", slide_html or "", re.IGNORECASE)
    return str(match.group(1) or "").strip() if match else ""


def _page_contract_is_applicable(page_contract: dict[str, Any]) -> bool:
    for key in (
        "page_role",
        "persona_signal",
        "manuscript_anchor",
        "required_component",
        "layout_bias",
        "must_preserve",
        "hard_requirements",
        "style_requirements",
        "component_requirements",
        "soft_signals",
        "source_boundary_notes",
        "layout_requirements",
        "asset_requirements",
        "selected_layout",
        "bound_asset_path",
        "visual_requirement",
    ):
        value = page_contract.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, (list, tuple, set)) and any(str(item or "").strip() for item in value):
            return True
    return False


def evaluate_profile_execution_page(
    *,
    page_contract: dict[str, Any] | None,
    slide_html: str,
    slide_path: Path | None = None,
    design_plan_text: str = "",
    slide_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(page_contract or {})
    meta = dict(slide_meta or {})
    page_index = int(contract.get("page_index", meta.get("page", 0)) or 0)
    page_role = str(contract.get("page_role", "") or meta.get("page_role", "")).strip()
    required_component = str(contract.get("required_component", "") or meta.get("required_component", "")).strip()
    layout_bias = str(contract.get("layout_bias", "") or meta.get("layout_bias", "") or "single_focus").strip().lower()
    hard_requirements = [
        str(item).strip()
        for item in (contract.get("hard_requirements") or meta.get("hard_requirements") or [])
        if str(item or "").strip()
    ]
    component_requirements = [
        str(item).strip()
        for item in (contract.get("component_requirements") or meta.get("component_requirements") or [])
        if str(item or "").strip()
    ]
    if required_component and required_component not in component_requirements:
        component_requirements.insert(0, required_component)
    style_requirements = _coerce_profile_style_requirements(
        meta.get("style_requirements"),
        contract.get("style_requirements"),
    )
    soft_signals = [
        str(item).strip()
        for item in (contract.get("soft_signals") or meta.get("soft_signals") or [])
        if str(item or "").strip()
    ]
    source_boundary_notes = [
        str(item).strip()
        for item in (contract.get("source_boundary_notes") or meta.get("source_boundary_notes") or [])
        if str(item or "").strip()
    ]
    must_preserve = [
        str(item).strip()
        for item in (contract.get("must_preserve") or meta.get("must_preserve") or [])
        if str(item or "").strip()
    ]
    selected_layout = str(contract.get("selected_layout", "") or meta.get("selected_layout", "")).strip()
    bound_asset_path = str(contract.get("bound_asset_path", "") or meta.get("bound_asset_path", "")).strip()
    bound_asset_kind = str(contract.get("bound_asset_kind", "") or meta.get("bound_asset_kind", "")).strip()
    visual_requirement = str(contract.get("visual_requirement", "") or meta.get("visual_requirement", "") or "none").strip().lower()

    observed = _slide_observed_signals(slide_html or "")
    component_signals = _semantic_component_signals(slide_html or "")
    text = observed.get("text", "")
    observed_biases = observed.get("observed_biases", [])
    component_matches = [
        {
            "requirement": requirement,
            "matched": _component_requirement_matched(requirement, component_signals),
        }
        for requirement in component_requirements[:6]
    ]
    component_match = all(item["matched"] for item in component_matches) if component_matches else True

    bias_markers = _persona_markers_for_bias(layout_bias)
    layout_bias_match = layout_bias in observed_biases if layout_bias else True
    if not layout_bias_match and bias_markers:
        layout_bias_match = any(marker in text for marker in bias_markers)

    plan_text_match = any(
        token in design_plan_text.lower()
        for token in {page_role.lower(), required_component.lower()}
        if token
    )
    must_preserve_hits = [item for item in must_preserve if item.lower() in text]
    missing_must_preserve = [item for item in must_preserve if item not in must_preserve_hits]
    style_evaluation = _evaluate_profile_style_requirements(slide_html or "", style_requirements)

    soft_signal_hits = [
        signal
        for signal in soft_signals
        if signal.lower() in text or any(token in text for token in re.split(r"[\s,，;；]+", signal.lower()) if len(token) >= 4)
    ]

    explicit_failures: list[str] = []
    soft_gaps: list[str] = []

    if style_evaluation.get("css_value_failures"):
        explicit_failures.extend(str(item) for item in style_evaluation.get("css_value_failures", [])[:3])

    if selected_layout:
        annotated_layout = _extract_template_layout_annotation(slide_html)
        if annotated_layout and annotated_layout != selected_layout:
            explicit_failures.append(
                f"Annotated template layout is `{annotated_layout}` but expected `{selected_layout}`."
            )

    if visual_requirement == "required":
        if bound_asset_path and bound_asset_path not in (slide_html or ""):
            explicit_failures.append(
                f"Bound {bound_asset_kind or 'visual'} asset `{bound_asset_path}` is missing from the slide."
            )
        if not observed.get("has_visual") and not observed.get("has_table"):
            explicit_failures.append(
                "This page is required to stay visual, but no real image/table/chart signal was detected."
            )

    missing_components = [item["requirement"] for item in component_matches if not item.get("matched")]
    if missing_components:
        soft_gaps.append(
            "Component requirement was not clearly realized: " + ", ".join(missing_components[:3])
        )
    if layout_bias and not layout_bias_match:
        soft_gaps.append(
            f"Layout bias `{layout_bias}` was not clearly realized."
        )

    deterministic_verdict = "pass"
    if explicit_failures:
        deterministic_verdict = "retry_required"
    elif soft_gaps:
        deterministic_verdict = "needs_llm_review"

    repair_bits = explicit_failures or soft_gaps
    repair_focus = " ".join(repair_bits[:2]).strip()
    hard_structure_score = 0.0 if any("template layout" in item or "asset" in item or "visual" in item for item in explicit_failures) else 1.0
    concrete_style_score = 1.0 if not style_evaluation.get("css_value_failures") else 0.0
    if component_matches:
        semantic_component_score = sum(1 for item in component_matches if item.get("matched")) / len(component_matches)
    else:
        semantic_component_score = 1.0
    soft_persona_score = (
        min(1.0, len(soft_signal_hits) / max(1, min(3, len(soft_signals))))
        if soft_signals
        else 1.0
    )
    page_scores = {
        "hard_structure": round(hard_structure_score, 3),
        "concrete_style": round(concrete_style_score, 3),
        "semantic_component": round(semantic_component_score, 3),
        "soft_persona_signal": round(soft_persona_score, 3),
        "overall": round(
            (hard_structure_score * 0.35)
            + (concrete_style_score * 0.25)
            + (semantic_component_score * 0.3)
            + (soft_persona_score * 0.1),
            3,
        ),
    }

    return {
        "applicable": _page_contract_is_applicable(contract | meta),
        "page_index": page_index,
        "page_role": page_role,
        "expected_layout_bias": layout_bias,
        "expected_component": required_component,
        "hard_requirements": hard_requirements,
        "style_requirements": style_requirements,
        "component_requirements": component_requirements,
        "soft_signals": soft_signals,
        "source_boundary_notes": source_boundary_notes,
        "slide_path": slide_path.as_posix() if slide_path else "",
        "exists": bool((slide_html or "").strip()),
        "selected_layout": selected_layout,
        "bound_asset_path": bound_asset_path,
        "visual_requirement": visual_requirement,
        "observed_biases": observed_biases,
        "component_signals": {
            key: value
            for key, value in component_signals.items()
            if key != "combined_text"
        },
        "component_matches": component_matches,
        "component_matched": component_match,
        "layout_bias_matched": layout_bias_match,
        "plan_text_matched": plan_text_match,
        "must_preserve_hits": must_preserve_hits,
        "missing_must_preserve": missing_must_preserve,
        "must_preserve_checked_as": "compatibility_soft_signal",
        "style_evaluation": style_evaluation,
        "soft_signal_hits": soft_signal_hits,
        "page_scores": page_scores,
        "headings": observed.get("headings", []),
        "deterministic_verdict": deterministic_verdict,
        "explicit_failures": explicit_failures,
        "soft_gaps": soft_gaps,
        "needs_llm_review": bool(soft_gaps and not explicit_failures),
        "repair_focus": repair_focus,
    }


_PROFILE_AUDIT_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")


def _profile_audit_routing_dir(workspace: Path, round_index: int = 0) -> Path:
    preferred = workspace / ".memory" / "rounds" / f"round_{round_index:03d}" / "profile_routing"
    if preferred.exists():
        return preferred
    rounds_dir = workspace / ".memory" / "rounds"
    if rounds_dir.exists():
        candidates = sorted(path / "profile_routing" for path in rounds_dir.glob("round_*") if (path / "profile_routing").exists())
        if candidates:
            return candidates[0]
    return preferred


def _prioritized_profile_pptx_candidates(workspace: Path) -> list[Path]:
    preferred_names = ("round_00_initial.pptx", "manuscript.pptx")
    preferred = [workspace / name for name in preferred_names if (workspace / name).is_file()]
    others = sorted(
        path
        for path in workspace.glob("*.pptx")
        if path.is_file() and path.name not in {"template.pptx", *preferred_names}
    )
    return preferred + others


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _compact_audit_text(value: Any, *, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _profile_audit_tokens(text: str) -> list[str]:
    raw = str(text or "")
    lower = raw.lower()
    tokens: list[str] = []
    tokens.extend(match.group(0) for match in _PROFILE_AUDIT_COLOR_RE.finditer(raw))
    tokens.extend(
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+./-]{2,}", lower)
        if token not in {"the", "and", "for", "with", "this", "that", "from"}
    )
    for term in (
        "机制", "流程", "架构", "证据", "实验", "结果", "局限", "未来", "密度",
        "矩阵", "表格", "总结", "问题", "技术", "中文", "主色", "强调色", "字体",
        "背景", "图注", "图表", "贡献", "风险", "结论", "建议", "下一步",
    ):
        if term in lower:
            tokens.append(term)
    deduped = list(dict.fromkeys(token.strip() for token in tokens if token.strip()))
    return deduped[:18]


def _profile_audit_text_matches(haystack: str, tokens: list[str]) -> bool:
    lower = str(haystack or "").lower()
    return any(str(token or "").lower() in lower for token in tokens if str(token or "").strip())


def _profile_audit_snippets(haystack: str, tokens: list[str], *, limit: int = 3) -> list[str]:
    text = re.sub(r"\s+", " ", str(haystack or "")).strip()
    lower = text.lower()
    snippets: list[str] = []
    for token in tokens:
        needle = str(token or "").strip().lower()
        if not needle:
            continue
        idx = lower.find(needle)
        if idx < 0:
            continue
        start = max(0, idx - 48)
        end = min(len(text), idx + len(needle) + 72)
        snippets.append(_compact_audit_text(text[start:end], limit=180))
        if len(snippets) >= limit:
            break
    return list(dict.fromkeys(snippets))


def _profile_audit_flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            values.extend(_profile_audit_flatten(value, child))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]"
            values.extend(_profile_audit_flatten(value, child))
    elif obj is not None:
        text = str(obj).strip()
        if text:
            values.append((prefix, text))
    return values


def _classify_profile_audit_preference(dimension: str, preference: str, source: str = "") -> str:
    text = f"{dimension} {preference} {source}".lower()
    if _PROFILE_AUDIT_COLOR_RE.search(preference) or any(
        token in text
        for token in ("css_values", "palette", "color", "background-color", "font-style", "font-weight", "text-decoration", "主色", "强调色", "字体")
    ):
        return "concrete_style"
    if any(token in text for token in ("page count", "页数", "inspect", "export", "selected_layout", "bound_asset", "visual_requirement", "不得编造", "strictly based")):
        return "hard_structure"
    if any(token in text for token in ("pipeline", "diagram", "matrix", "table", "chart", "figure", "card", "chip", "callout", "summary", "流程", "机制", "矩阵", "表格", "图表", "卡片", "总结")):
        return "semantic_component"
    return "soft_persona_signal"


def _collect_profile_audit_preferences(
    *,
    wm_after_routing: dict[str, Any],
    profile_execution_contract: dict[str, Any] | None,
    profile_execution_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[int, ...]]] = set()

    def _add(
        *,
        dimension: str,
        preference: str,
        source: str,
        page_targets: list[int] | None = None,
        contract_field_hint: str = "",
        wm_present: bool = False,
    ) -> None:
        clean = _compact_audit_text(preference, limit=360)
        if not clean:
            return
        pages = sorted({int(page) for page in (page_targets or []) if int(page or 0) > 0})
        key = (str(dimension or ""), clean.lower(), str(source or ""), tuple(pages))
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "dimension": str(dimension or "general").strip() or "general",
            "profile_preference": clean,
            "source": source,
            "preference_type": _classify_profile_audit_preference(dimension, clean, source),
            "page_targets": pages,
            "contract_field_hint": contract_field_hint,
            "wm_present": wm_present,
        })

    for pref in wm_after_routing.get("preferences", []) or []:
        if not isinstance(pref, dict):
            continue
        _add(
            dimension=str(pref.get("dimension", "") or "profile"),
            preference=str(pref.get("content", "") or ""),
            source="wm_after_routing",
            wm_present=True,
        )

    contract = profile_execution_contract or {}
    for field in (
        "deck_spine",
        "style_contract",
        "content_contract",
        "source_boundary_contract",
        "proxy_realization_rules",
        "hard_requirements",
        "component_requirements",
        "soft_signals",
    ):
        for item in contract.get(field, []) or []:
            _add(
                dimension=f"contract.{field}",
                preference=str(item or ""),
                source="profile_execution_contract",
                contract_field_hint=field,
            )
    style_req = contract.get("style_requirements") if isinstance(contract.get("style_requirements"), dict) else {}
    css_items = (style_req.get("css_values") or {}).items() if isinstance(style_req.get("css_values"), dict) else []
    for prop, value in css_items:
        _add(
            dimension="contract.style_requirements.css_values",
            preference=f"{prop}: {value}",
            source="profile_execution_contract",
            contract_field_hint="style_requirements.css_values",
        )
    for value in style_req.get("palette_values", []) or []:
        _add(
            dimension="contract.style_requirements.palette_values",
            preference=f"palette value: {value}",
            source="profile_execution_contract",
            contract_field_hint="style_requirements.palette_values",
        )

    for page in (profile_execution_plan or {}).get("page_plan", []) or []:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index", 0) or 0)
        prefix = f"page_plan.p{page_index}"
        for field in ("page_role", "persona_signal", "manuscript_anchor", "required_component", "layout_bias"):
            _add(
                dimension=f"{prefix}.{field}",
                preference=str(page.get(field, "") or ""),
                source="profile_execution_plan",
                page_targets=[page_index],
                contract_field_hint=f"page_plan[{page_index}].{field}",
            )
        page_style = page.get("style_requirements") if isinstance(page.get("style_requirements"), dict) else {}
        page_css_items = (page_style.get("css_values") or {}).items() if isinstance(page_style.get("css_values"), dict) else []
        for prop, value in page_css_items:
            _add(
                dimension=f"{prefix}.style_requirements.css_values",
                preference=f"{prop}: {value}",
                source="profile_execution_plan",
                page_targets=[page_index],
                contract_field_hint=f"page_plan[{page_index}].style_requirements.css_values",
            )
        for value in page_style.get("palette_values", []) or []:
            _add(
                dimension=f"{prefix}.style_requirements.palette_values",
                preference=f"palette value: {value}",
                source="profile_execution_plan",
                page_targets=[page_index],
                contract_field_hint=f"page_plan[{page_index}].style_requirements.palette_values",
            )
        for field in ("hard_requirements", "component_requirements", "soft_signals", "source_boundary_notes", "must_preserve"):
            for item in page.get(field, []) or []:
                _add(
                    dimension=f"{prefix}.{field}",
                    preference=str(item or ""),
                    source="profile_execution_plan",
                    page_targets=[page_index],
                    contract_field_hint=f"page_plan[{page_index}].{field}",
                )

    return rows


def _infer_profile_audit_page_targets(
    row: dict[str, Any],
    profile_execution_plan: dict[str, Any] | None,
    all_pages: list[int],
) -> list[int]:
    explicit = [int(page) for page in row.get("page_targets", []) or [] if int(page or 0) > 0]
    if explicit:
        return explicit
    dimension = str(row.get("dimension", "") or "").lower()
    if dimension.startswith("theme") or "style" in dimension or row.get("preference_type") == "concrete_style":
        return all_pages
    tokens = _profile_audit_tokens(str(row.get("profile_preference", "") or ""))
    matched: list[int] = []
    for page in (profile_execution_plan or {}).get("page_plan", []) or []:
        if not isinstance(page, dict):
            continue
        page_text = json.dumps(page, ensure_ascii=False)
        if _profile_audit_text_matches(page_text, tokens):
            page_index = int(page.get("page_index", 0) or 0)
            if page_index > 0:
                matched.append(page_index)
    return sorted(set(matched)) or all_pages


def _profile_audit_contract_fields(
    row: dict[str, Any],
    flattened_contract: list[tuple[str, str]],
    flattened_plan: list[tuple[str, str]],
) -> list[str]:
    if row.get("contract_field_hint"):
        return [str(row["contract_field_hint"])]
    tokens = _profile_audit_tokens(str(row.get("profile_preference", "") or ""))
    fields: list[str] = []
    for path, value in flattened_contract + flattened_plan:
        if _profile_audit_text_matches(value, tokens):
            fields.append(path)
        if len(fields) >= 5:
            break
    return fields


def _profile_audit_slide_evidence(
    *,
    preference: str,
    preference_type: str,
    page_index: int,
    page_result: dict[str, Any],
    slide_html: str,
) -> list[str]:
    evidence: list[str] = []
    colors = [match.group(0) for match in _PROFILE_AUDIT_COLOR_RE.finditer(preference or "")]
    if colors:
        observations = _extract_slide_style_observations(slide_html)
        observed_values = observations.get("all_values", []) or []
        matched = [
            color
            for color in colors
            if any(_profile_css_values_equivalent("color", color, observed) for observed in observed_values)
        ]
        if matched:
            evidence.append(f"p{page_index}: CSS palette observed {', '.join(matched[:4])}")

    if preference_type == "semantic_component":
        pref_lower = str(preference or "").lower()
        signals = page_result.get("component_signals", {}) or {}
        signal_hits: list[str] = []
        if any(token in pref_lower for token in ("pipeline", "process", "step", "流程", "机制")) and signals.get("has_process"):
            signal_hits.append("process/pipeline")
        if any(token in pref_lower for token in ("matrix", "table", "comparison", "矩阵", "表格", "对比")) and signals.get("has_matrix"):
            signal_hits.append("matrix/table")
        if any(token in pref_lower for token in ("summary", "takeaway", "callout", "总结", "结论")) and signals.get("has_callout"):
            signal_hits.append("summary/callout")
        if any(token in pref_lower for token in ("chip", "kpi", "badge", "pill")) and signals.get("has_kpi_chips"):
            signal_hits.append("chip/KPI")
        if any(token in pref_lower for token in ("card", "contribution", "卡片", "贡献")) and signals.get("has_cards"):
            signal_hits.append("cards")
        if signal_hits:
            evidence.append(f"p{page_index}: semantic component signal {', '.join(signal_hits)}")

    tokens = _profile_audit_tokens(preference)
    slide_text = BeautifulSoup(slide_html or "", "html.parser").get_text(" ", strip=True)
    snippets = _profile_audit_snippets(slide_text, tokens, limit=2)
    if snippets:
        evidence.extend(f"p{page_index}: text `{snippet}`" for snippet in snippets)

    soft_hits = page_result.get("soft_signal_hits", []) or []
    if soft_hits:
        matching_soft = [hit for hit in soft_hits if _profile_audit_text_matches(hit, tokens)]
        if matching_soft:
            evidence.append(f"p{page_index}: soft signal hit `{_compact_audit_text(matching_soft[0], limit=120)}`")

    return list(dict.fromkeys(evidence))[:4]


def _profile_audit_verifier_result(page_result: dict[str, Any], page_meta: dict[str, Any]) -> str:
    status = str(page_meta.get("persona_status", "") or page_result.get("deterministic_verdict", "") or "unknown")
    deterministic = str(page_result.get("deterministic_verdict", "") or "")
    score = (page_result.get("page_scores", {}) or {}).get("overall")
    parts = [f"status={status}"]
    if deterministic:
        parts.append(f"deterministic={deterministic}")
    if score is not None:
        parts.append(f"score={score}")
    if page_result.get("soft_gaps"):
        parts.append("soft_gaps=" + "; ".join(str(item) for item in page_result.get("soft_gaps", [])[:2]))
    if page_result.get("explicit_failures"):
        parts.append("failures=" + "; ".join(str(item) for item in page_result.get("explicit_failures", [])[:2]))
    return ", ".join(parts)


def build_profile_realization_audit(
    *,
    workspace: Path,
    slide_dir: Path | str | None,
    profile_execution_contract: dict[str, Any] | None,
    profile_execution_plan: dict[str, Any] | None,
    coverage_by_page: list[dict[str, Any]] | None = None,
    design_plan_text: str = "",
    section_presence: dict[str, bool] | None = None,
    deck_state_slides: dict[str, Any] | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    workspace = Path(workspace)
    routing_dir = _profile_audit_routing_dir(workspace, round_index=round_index)
    profile_load_status = _safe_read_json(routing_dir / "profile_load_status.json")
    routing_decisions = _safe_read_json(routing_dir / "routing_decisions.json")
    wm_after_routing = _safe_read_json(routing_dir / "wm_after_routing.json")
    deck_state_path = workspace / "deck_execution_state.json"
    deck_state = _safe_read_json(deck_state_path)
    if deck_state_slides is None:
        deck_state_slides = deck_state.get("slides", {}) if isinstance(deck_state.get("slides"), dict) else {}

    slide_root = Path(slide_dir) if slide_dir else workspace / "outputs"
    if not slide_root.is_absolute():
        slide_root = workspace / slide_root
    slide_paths = sorted(slide_root.glob("slide_*.html")) if slide_root.exists() else []
    slide_html_by_page: dict[int, str] = {}
    for path in slide_paths:
        match = re.search(r"slide_(\d+)\.html$", path.name)
        if not match:
            continue
        try:
            slide_html_by_page[int(match.group(1))] = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            slide_html_by_page[int(match.group(1))] = ""

    if not design_plan_text:
        design_plan_rel = find_existing_design_plan_rel(workspace)
        design_plan_path = workspace / design_plan_rel if design_plan_rel else None
        if design_plan_path and design_plan_path.exists():
            try:
                design_plan_text = design_plan_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                design_plan_text = ""
    section_presence = section_presence or {
        "profile_derived_deck_contract": "## Profile-Derived Deck Contract" in design_plan_text,
        "page_level_persona_obligations": "## Page-Level Persona Obligations" in design_plan_text,
        "persona_page_plan": "## Persona Page Plan" in design_plan_text,
    }

    page_results = {int(item.get("page_index", 0) or 0): item for item in coverage_by_page or [] if isinstance(item, dict)}
    if not page_results and profile_execution_plan:
        for page in profile_execution_plan.get("page_plan", []) or []:
            if not isinstance(page, dict):
                continue
            page_index = int(page.get("page_index", 0) or 0)
            page_meta = deck_state_slides.get(str(page_index), {}) if isinstance(deck_state_slides, dict) else {}
            page_results[page_index] = evaluate_profile_execution_page(
                page_contract=page,
                slide_html=slide_html_by_page.get(page_index, ""),
                slide_path=slide_root / f"slide_{page_index:02d}.html",
                design_plan_text=design_plan_text,
                slide_meta=page_meta if isinstance(page_meta, dict) else None,
            )

    all_pages = sorted(
        {
            int(page.get("page_index", 0) or 0)
            for page in (profile_execution_plan or {}).get("page_plan", []) or []
            if isinstance(page, dict) and int(page.get("page_index", 0) or 0) > 0
        }
        or set(slide_html_by_page.keys())
    )
    expected_pages = len((profile_execution_plan or {}).get("page_plan", []) or []) or int(deck_state.get("expected_slide_count", 0) or 0)

    decisions_by_dimension = {
        str(item.get("dimension", "") or "").lower(): str(item.get("action", "") or "")
        for item in routing_decisions.get("decisions", []) or []
        if isinstance(item, dict)
    }
    flattened_contract = _profile_audit_flatten(profile_execution_contract or {})
    flattened_plan = _profile_audit_flatten(profile_execution_plan or {})
    raw_rows = _collect_profile_audit_preferences(
        wm_after_routing=wm_after_routing,
        profile_execution_contract=profile_execution_contract,
        profile_execution_plan=profile_execution_plan,
    )

    preference_matrix: list[dict[str, Any]] = []
    dimension_slide_evidence: dict[str, int] = {}
    for row in raw_rows:
        dimension = str(row.get("dimension", "") or "general")
        dimension_root = dimension.split(".", 1)[0].lower()
        router_action = decisions_by_dimension.get(dimension_root, "")
        page_targets = _infer_profile_audit_page_targets(row, profile_execution_plan, all_pages)
        tokens = _profile_audit_tokens(str(row.get("profile_preference", "") or ""))
        design_evidence = _profile_audit_snippets(design_plan_text, tokens, limit=3)
        contract_fields = _profile_audit_contract_fields(row, flattened_contract, flattened_plan)

        current_contract_evidence: list[str] = []
        slide_evidence: list[str] = []
        verifier_result: list[str] = []
        repair_history: list[str] = []
        explicit_failure_seen = False
        soft_gap_seen = False
        for page_index in page_targets[:8]:
            page_meta = deck_state_slides.get(str(page_index), {}) if isinstance(deck_state_slides, dict) else {}
            page_meta_text = json.dumps(page_meta, ensure_ascii=False)
            if _profile_audit_text_matches(page_meta_text, tokens):
                snippets = _profile_audit_snippets(page_meta_text, tokens, limit=1)
                current_contract_evidence.append(
                    f"p{page_index}: {_compact_audit_text(snippets[0] if snippets else 'page contract contains matching token', limit=160)}"
                )
            elif page_meta:
                current_contract_evidence.append(
                    f"p{page_index}: contract exists role={page_meta.get('page_role') or page_meta.get('title') or 'unspecified'}"
                )

            page_result = page_results.get(page_index, {})
            html = slide_html_by_page.get(page_index, "")
            page_slide_evidence = _profile_audit_slide_evidence(
                preference=str(row.get("profile_preference", "") or ""),
                preference_type=str(row.get("preference_type", "") or ""),
                page_index=page_index,
                page_result=page_result,
                slide_html=html,
            )
            slide_evidence.extend(page_slide_evidence)
            if page_result:
                verifier_result.append(f"p{page_index}: {_profile_audit_verifier_result(page_result, page_meta if isinstance(page_meta, dict) else {})}")
                explicit_failure_seen = explicit_failure_seen or bool(page_result.get("explicit_failures"))
                soft_gap_seen = soft_gap_seen or bool(page_result.get("soft_gaps"))
            retry_count = int(page_meta.get("persona_retry_count", 0) or 0) if isinstance(page_meta, dict) else 0
            if retry_count:
                repair_history.append(
                    f"p{page_index}: persona_retry_count={retry_count}, final={page_meta.get('persona_status') or 'unknown'}"
                )

        if slide_evidence:
            verdict = "realized"
            dimension_slide_evidence[dimension_root] = dimension_slide_evidence.get(dimension_root, 0) + 1
        elif explicit_failure_seen:
            verdict = "hard_failure"
        elif design_evidence or contract_fields or current_contract_evidence:
            verdict = "soft_gap" if row.get("preference_type") == "soft_persona_signal" or soft_gap_seen else "planned_only"
        else:
            verdict = "missing"

        preference_matrix.append({
            "dimension": dimension,
            "preference_type": row.get("preference_type", ""),
            "profile_preference": row.get("profile_preference", ""),
            "router_action": router_action,
            "wm_present": bool(row.get("wm_present")),
            "contract_field": contract_fields,
            "page_targets": page_targets,
            "design_plan_evidence": design_evidence,
            "current_page_contract_evidence": list(dict.fromkeys(current_contract_evidence))[:8],
            "slide_evidence": list(dict.fromkeys(slide_evidence))[:12],
            "verifier_result": list(dict.fromkeys(verifier_result))[:8],
            "repair_history": list(dict.fromkeys(repair_history))[:6],
            "verdict": verdict,
            "notes": (
                "Concrete slide-level evidence found."
                if verdict == "realized"
                else "Preference reached planning/contract layer but slide evidence is partial or indirect."
                if verdict in {"planned_only", "soft_gap"}
                else "Hard verifier failure observed."
                if verdict == "hard_failure"
                else "No clear route/planning/slide evidence found."
            ),
        })

    page_focus_checks: list[dict[str, Any]] = []
    for page_index in all_pages:
        page_result = page_results.get(page_index, {})
        page_meta = deck_state_slides.get(str(page_index), {}) if isinstance(deck_state_slides, dict) else {}
        page_focus_checks.append({
            "page_index": page_index,
            "page_role": page_result.get("page_role") or page_meta.get("page_role") or page_meta.get("title") or "",
            "required_component": page_result.get("expected_component") or page_meta.get("required_component") or "",
            "layout_bias": page_result.get("expected_layout_bias") or page_meta.get("layout_bias") or "",
            "persona_status": page_meta.get("persona_status", ""),
            "persona_retry_count": int(page_meta.get("persona_retry_count", 0) or 0) if isinstance(page_meta, dict) else 0,
            "inspect_passed": bool(page_meta.get("inspect_passed")) if isinstance(page_meta, dict) else False,
            "component_matched": bool(page_result.get("component_matched")),
            "layout_bias_matched": bool(page_result.get("layout_bias_matched")),
            "page_scores": page_result.get("page_scores", {}),
            "soft_gaps": page_result.get("soft_gaps", []),
            "explicit_failures": page_result.get("explicit_failures", []),
            "slide_evidence": [
                f"observed_biases={page_result.get('observed_biases', [])}",
                f"headings={page_result.get('headings', [])}",
            ],
        })

    pptx_candidates = _prioritized_profile_pptx_candidates(workspace)
    inspect_passed_count = sum(1 for item in (deck_state_slides or {}).values() if isinstance(item, dict) and item.get("inspect_passed"))
    page_contract_count = sum(
        1
        for item in (deck_state_slides or {}).values()
        if isinstance(item, dict) and any(item.get(key) for key in ("page_role", "persona_signal", "required_component", "layout_bias"))
    )

    required_checks = [
        {"name": "profile_loaded", "passed": bool(profile_load_status.get("has_content")), "evidence": profile_load_status.get("source", "")},
        {"name": "router_decisions_nonempty", "passed": bool(routing_decisions.get("decisions")), "evidence": routing_decisions.get("summary", {})},
        {"name": "wm_preferences_nonempty", "passed": int(wm_after_routing.get("total_prefs", 0) or len(wm_after_routing.get("preferences", []) or [])) > 0, "evidence": wm_after_routing.get("total_prefs", 0)},
        {"name": "profile_contract_compiled", "passed": bool(profile_execution_contract), "evidence": (profile_execution_contract or {}).get("planning_focus", "")},
        {"name": "profile_plan_compiled", "passed": bool((profile_execution_plan or {}).get("page_plan")), "evidence": len((profile_execution_plan or {}).get("page_plan", []) or [])},
        {"name": "design_plan_profile_sections", "passed": all(section_presence.values()), "evidence": section_presence},
        {"name": "page_contracts_registered", "passed": bool(expected_pages and page_contract_count >= expected_pages), "evidence": f"{page_contract_count}/{expected_pages}"},
        {"name": "html_pages_exist", "passed": bool(expected_pages and len(slide_paths) >= expected_pages), "evidence": f"{len(slide_paths)}/{expected_pages}"},
        {"name": "inspect_passed", "passed": bool(expected_pages and inspect_passed_count >= expected_pages), "evidence": f"{inspect_passed_count}/{expected_pages}"},
        {"name": "pptx_exported", "passed": bool(pptx_candidates), "evidence": pptx_candidates[0].as_posix() if pptx_candidates else ""},
    ]

    expected_dimension_roots = {"theme", "visual", "layout", "content", "general"}
    dimension_coverage = {
        dimension: {
            "rows_with_slide_evidence": dimension_slide_evidence.get(dimension, 0),
            "passed": dimension_slide_evidence.get(dimension, 0) > 0,
        }
        for dimension in sorted(expected_dimension_roots)
    }
    core_page_checks: list[dict[str, Any]] = []
    for page_index in (2, 5, 6, 8):
        if page_index not in page_results:
            continue
        result = page_results[page_index]
        core_page_checks.append({
            "page_index": page_index,
            "passed": bool(result.get("component_matched")),
            "required_component": result.get("expected_component", ""),
            "soft_gaps": result.get("soft_gaps", []),
        })

    palette_page_hits = 0
    for page_index, html in slide_html_by_page.items():
        observations = _extract_slide_style_observations(html)
        values = observations.get("all_values", []) or []
        if any(_PROFILE_AUDIT_COLOR_RE.fullmatch(str(value or "").strip()) for value in values):
            palette_page_hits += 1
        elif any(
            any(_profile_css_values_equivalent("color", color, value) for value in values)
            for color in ("#111827", "#2563EB", "#F59E0B", "#10B981", "#0F172A", "#047857")
        ):
            palette_page_hits += 1
    high_quality_checks = [
        {"name": "dimension_slide_evidence", "passed": all(item["passed"] for item in dimension_coverage.values()), "evidence": dimension_coverage},
        {"name": "core_pages_component_realized", "passed": bool(core_page_checks) and all(item["passed"] for item in core_page_checks), "evidence": core_page_checks},
        {"name": "palette_on_majority_pages", "passed": bool(expected_pages and palette_page_hits >= max(1, expected_pages // 2)), "evidence": f"{palette_page_hits}/{expected_pages}"},
        {"name": "persona_repair_not_stuck", "passed": not any(str((item or {}).get("persona_status", "")) == "retry_required" for item in (deck_state_slides or {}).values() if isinstance(item, dict)), "evidence": "no active retry_required pages"},
    ]

    soft_gaps = [
        {
            "page_index": item.get("page_index"),
            "page_role": item.get("page_role"),
            "soft_gaps": item.get("soft_gaps", []),
            "score": (item.get("page_scores", {}) or {}).get("overall"),
        }
        for item in page_focus_checks
        if item.get("soft_gaps") or ((item.get("page_scores", {}) or {}).get("overall", 1.0) < 0.8)
    ]
    required_passed = all(item["passed"] for item in required_checks)
    high_quality_passed = sum(1 for item in high_quality_checks if item["passed"])
    if required_passed and high_quality_passed == len(high_quality_checks):
        overall = "strong"
    elif required_passed:
        overall = "solid_with_soft_gaps"
    elif sum(1 for item in required_checks if item["passed"]) >= max(1, len(required_checks) - 2):
        overall = "partial"
    else:
        overall = "weak"

    return {
        "version": 1,
        "judgment_mode": "round0_stage_trace",
        "round_index": round_index,
        "workspace": workspace.as_posix(),
        "artifact_paths": {
            "routing_dir": routing_dir.as_posix(),
            "deck_execution_state": deck_state_path.as_posix(),
            "slide_dir": slide_root.as_posix(),
        },
        "chain_checks": {
            "required": required_checks,
            "high_quality": high_quality_checks,
            "soft_gaps": soft_gaps,
            "overall": overall,
        },
        "dimension_coverage": dimension_coverage,
        "preference_matrix": preference_matrix,
        "page_focus_checks": page_focus_checks,
        "summary": {
            "overall": overall,
            "matrix_rows": len(preference_matrix),
            "realized_rows": sum(1 for item in preference_matrix if item.get("verdict") == "realized"),
            "planned_only_rows": sum(1 for item in preference_matrix if item.get("verdict") == "planned_only"),
            "soft_gap_rows": sum(1 for item in preference_matrix if item.get("verdict") == "soft_gap"),
            "hard_failure_rows": sum(1 for item in preference_matrix if item.get("verdict") == "hard_failure"),
            "missing_rows": sum(1 for item in preference_matrix if item.get("verdict") == "missing"),
            "page_count_expected": expected_pages,
            "page_count_observed": len(slide_paths),
        },
    }


def render_profile_realization_audit_markdown(audit: dict[str, Any] | None) -> str:
    if not audit:
        return ""

    def _status(value: bool) -> str:
        return "PASS" if value else "FAIL"

    lines = [
        "# Profile Realization Audit",
        "",
        f"- Overall: `{audit.get('summary', {}).get('overall') or audit.get('chain_checks', {}).get('overall', '')}`",
        f"- Matrix rows: {audit.get('summary', {}).get('matrix_rows', 0)}",
        f"- Realized rows: {audit.get('summary', {}).get('realized_rows', 0)}",
        f"- Workspace: `{audit.get('workspace', '')}`",
        "",
        "## Required Checks",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in (audit.get("chain_checks", {}) or {}).get("required", []) or []:
        evidence = _compact_audit_text(json.dumps(item.get("evidence", ""), ensure_ascii=False), limit=160)
        lines.append(f"| {item.get('name', '')} | {_status(bool(item.get('passed')))} | {evidence} |")

    lines.extend([
        "",
        "## High Quality Checks",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ])
    for item in (audit.get("chain_checks", {}) or {}).get("high_quality", []) or []:
        evidence = _compact_audit_text(json.dumps(item.get("evidence", ""), ensure_ascii=False), limit=180)
        lines.append(f"| {item.get('name', '')} | {_status(bool(item.get('passed')))} | {evidence} |")

    lines.extend([
        "",
        "## Page Focus",
        "| Page | Role | Component | Layout | Persona | Score | Gaps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for item in audit.get("page_focus_checks", []) or []:
        score = (item.get("page_scores", {}) or {}).get("overall", "")
        gaps = _compact_audit_text("; ".join(str(gap) for gap in item.get("soft_gaps", []) or []), limit=120)
        lines.append(
            f"| p{item.get('page_index', '')} | {_compact_audit_text(item.get('page_role', ''), limit=48)} "
            f"| {_compact_audit_text(item.get('required_component', ''), limit=52)} "
            f"| {item.get('layout_bias', '')} | {item.get('persona_status', '')} | {score} | {gaps} |"
        )

    lines.extend([
        "",
        "## Preference Matrix",
        "| Dimension | Type | Preference | Pages | Verdict | Evidence | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in (audit.get("preference_matrix", []) or [])[:120]:
        pages = ", ".join(f"p{page}" for page in row.get("page_targets", [])[:8])
        evidence = row.get("slide_evidence") or row.get("current_page_contract_evidence") or row.get("design_plan_evidence") or []
        evidence_text = _compact_audit_text("; ".join(str(item) for item in evidence[:3]), limit=220)
        lines.append(
            f"| {_compact_audit_text(row.get('dimension', ''), limit=48)} "
            f"| {row.get('preference_type', '')} "
            f"| {_compact_audit_text(row.get('profile_preference', ''), limit=100)} "
            f"| {pages} | {row.get('verdict', '')} | {evidence_text} | {_compact_audit_text(row.get('notes', ''), limit=96)} |"
        )
    if len(audit.get("preference_matrix", []) or []) > 120:
        lines.append(f"\n_Only the first 120 rows are shown; JSON contains {len(audit.get('preference_matrix', []))} rows._")

    soft_gaps = (audit.get("chain_checks", {}) or {}).get("soft_gaps", []) or []
    if soft_gaps:
        lines.extend(["", "## Soft Gaps"])
        for item in soft_gaps:
            lines.append(
                f"- p{item.get('page_index')}: {_compact_audit_text(item.get('page_role', ''), limit=80)} "
                f"score={item.get('score')} gaps={_compact_audit_text('; '.join(str(g) for g in item.get('soft_gaps', []) or []), limit=160)}"
            )

    return "\n".join(lines).rstrip() + "\n"


def build_profile_realization_report(
    *,
    workspace: Path,
    slide_dir: Path | str | None,
    profile_execution_contract: dict[str, Any] | None,
    profile_execution_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not profile_execution_plan:
        return None

    design_plan_rel = find_existing_design_plan_rel(workspace)
    design_plan_path = workspace / design_plan_rel if design_plan_rel else None
    design_plan_text = ""
    if design_plan_path and design_plan_path.exists():
        try:
            design_plan_text = design_plan_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            design_plan_text = ""

    slide_root = Path(slide_dir) if slide_dir else workspace / "outputs"
    if not slide_root.is_absolute():
        slide_root = workspace / slide_root
    slide_paths = sorted(slide_root.glob("slide_*.html")) if slide_root.exists() else []

    section_presence = {
        "profile_derived_deck_contract": "## Profile-Derived Deck Contract" in design_plan_text,
        "page_level_persona_obligations": "## Page-Level Persona Obligations" in design_plan_text,
        "persona_page_plan": "## Persona Page Plan" in design_plan_text,
    }

    coverage_by_page: list[dict[str, Any]] = []
    observed_signal_union: set[str] = set()
    misses: list[dict[str, Any]] = []
    proxy_realizations: list[dict[str, Any]] = []
    blocking_failures: list[dict[str, Any]] = []
    deck_state_slides: dict[str, Any] = {}
    deck_state_expected = 0
    try:
        from memslides.runtime.deck_execution_state import load_deck_execution_state

        deck_state = load_deck_execution_state(workspace) or {}
        deck_state_slides = deck_state.get("slides", {}) or {}
        deck_state_expected = int(deck_state.get("expected_slide_count", 0) or 0)
    except Exception:
        deck_state_slides = {}
        deck_state_expected = 0

    persona_status_counts: dict[str, int] = {}
    inspected_pages = 0
    inspect_passed_pages = 0
    accepted_pages = 0
    for page_meta in (deck_state_slides.values() if isinstance(deck_state_slides, dict) else []):
        if not isinstance(page_meta, dict):
            continue
        status = str(page_meta.get("persona_status", "") or "unknown").strip() or "unknown"
        persona_status_counts[status] = persona_status_counts.get(status, 0) + 1
        inspected_pages += 1 if page_meta.get("inspected") else 0
        inspect_passed_pages += 1 if page_meta.get("inspect_passed") else 0
        accepted_pages += 1 if page_meta.get("accepted") else 0

    for page in profile_execution_plan.get("page_plan", []) or []:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index", 0) or 0)
        slide_path = slide_root / f"slide_{page_index:02d}.html"
        slide_html = ""
        if slide_path.exists():
            try:
                slide_html = slide_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                slide_html = ""
        page_meta = deck_state_slides.get(str(page_index), {}) if isinstance(deck_state_slides, dict) else {}
        page_result = evaluate_profile_execution_page(
            page_contract=page,
            slide_html=slide_html,
            slide_path=slide_path,
            design_plan_text=design_plan_text,
            slide_meta=page_meta if isinstance(page_meta, dict) else None,
        )
        page_result["persona_status"] = str(page_meta.get("persona_status", "") or "")
        page_result["persona_unresolved"] = bool(page_meta.get("persona_unresolved"))
        observed_signal_union.update(page_result.get("observed_biases", []))
        coverage_by_page.append(page_result)
        if page_result.get("explicit_failures"):
            blocking_failures.append({
                "page_index": page_index,
                "page_role": page_result["page_role"],
                "failures": page_result.get("explicit_failures", [])[:4],
            })
        if page_result.get("deterministic_verdict") == "retry_required" or (
            not page_result.get("component_matched") or not page_result.get("layout_bias_matched")
        ):
            misses.append({
                "page_index": page_index,
                "page_role": page_result["page_role"],
                "expected_component": page_result["expected_component"],
                "expected_layout_bias": page_result["expected_layout_bias"],
                "observed_biases": page_result["observed_biases"],
                "reason": (
                    "; ".join(page_result.get("explicit_failures", [])[:2])
                    or "; ".join(page_result.get("soft_gaps", [])[:2])
                    or "component or layout-bias signal not clearly realized"
                ),
            })
            if (
                page_result["observed_biases"]
                or page_result.get("must_preserve_hits")
                or page_result.get("plan_text_matched")
            ):
                proxy_realizations.append({
                    "page_index": page_index,
                    "page_role": page_result["page_role"],
                    "proxy_signal": (
                        page_result["observed_biases"]
                        or page_result.get("must_preserve_hits")
                        or ["design-plan-only"]
                    ),
                })

    aligned_pages = sum(
        1
        for item in coverage_by_page
        if item.get("component_matched") and item.get("layout_bias_matched")
    )
    total_pages = max(1, len(coverage_by_page))
    alignment_ratio = aligned_pages / total_pages
    design_plan_support = all(section_presence.values())
    unresolved_pages = [
        item
        for item in coverage_by_page
        if item.get("persona_unresolved") or item.get("soft_gaps")
    ]

    if alignment_ratio >= 0.75 and design_plan_support:
        overall_alignment = "strong"
    elif alignment_ratio >= 0.4:
        overall_alignment = "partial"
    else:
        overall_alignment = "weak"

    if blocking_failures:
        verification_status = "blocking_failures"
    elif unresolved_pages or misses:
        verification_status = "soft_gaps"
    else:
        verification_status = "hard_pass"

    export_candidates = _prioritized_profile_pptx_candidates(workspace)
    export_success = bool(export_candidates)
    if any(item.get("persona_unresolved") for item in coverage_by_page):
        persona_repair_status = "unresolved_release"
    elif any(item.get("persona_status") == "retry_required" for item in coverage_by_page):
        persona_repair_status = "retry_required"
    elif coverage_by_page and all(str(item.get("persona_status", "") or "pass") in {"", "pass"} for item in coverage_by_page):
        persona_repair_status = "pass"
    else:
        persona_repair_status = "mixed"

    audit = build_profile_realization_audit(
        workspace=workspace,
        slide_dir=slide_root,
        profile_execution_contract=profile_execution_contract,
        profile_execution_plan=profile_execution_plan,
        coverage_by_page=coverage_by_page,
        design_plan_text=design_plan_text,
        section_presence=section_presence,
        deck_state_slides=deck_state_slides,
        round_index=0,
    )

    return {
        "judgment_mode": "deterministic",
        "injection_success": bool(profile_execution_contract or profile_execution_plan),
        "planning_success": bool(section_presence.get("profile_derived_deck_contract")) and bool(section_presence.get("persona_page_plan")),
        "generation_signals": {
            "page_count_expected": len(profile_execution_plan.get("page_plan", []) or []),
            "page_count_observed": len(slide_paths),
            "observed_signal_union": sorted(observed_signal_union),
            "aligned_pages": aligned_pages,
            "alignment_ratio": round(alignment_ratio, 3),
        },
        "verification_status": verification_status,
        "persona_repair_status": persona_repair_status,
        "persona_status_counts": persona_status_counts,
        "structural_status": {
            "expected_pages": deck_state_expected or len(profile_execution_plan.get("page_plan", []) or []),
            "html_pages": len(slide_paths),
            "inspected_pages": inspected_pages,
            "inspect_passed_pages": inspect_passed_pages,
            "accepted_pages": accepted_pages,
        },
        "export_success": export_success,
        "export_candidates": [path.as_posix() for path in export_candidates[:5]],
        "page_scores": [
            {
                "page_index": item.get("page_index"),
                "page_role": item.get("page_role"),
                **(item.get("page_scores", {}) or {}),
            }
            for item in coverage_by_page
        ],
        "blocking_failures": blocking_failures,
        "design_plan_path": design_plan_path.as_posix() if design_plan_path else "",
        "slide_dir": slide_root.as_posix(),
        "section_presence": section_presence,
        "expected_signals": [
            {
                "page_index": int(page.get("page_index", 0) or 0),
                "page_role": str(page.get("page_role", "") or "").strip(),
                "persona_signal": str(page.get("persona_signal", "") or "").strip(),
                "required_component": str(page.get("required_component", "") or "").strip(),
                "layout_bias": str(page.get("layout_bias", "") or "").strip(),
            }
            for page in profile_execution_plan.get("page_plan", []) or []
            if isinstance(page, dict)
        ],
        "observed_signals": sorted(observed_signal_union),
        "coverage_by_page": coverage_by_page,
        "misses": misses,
        "proxy_realizations": proxy_realizations,
        "overall_alignment": overall_alignment,
        "alignment_ratio": round(alignment_ratio, 3),
        "deck_contract_focus": str((profile_execution_contract or {}).get("planning_focus", "") or ""),
        "page_count_expected": len(profile_execution_plan.get("page_plan", []) or []),
        "page_count_observed": len(slide_paths),
        "profile_realization_audit_summary": audit.get("summary", {}),
        "profile_realization_audit": audit,
        "profile_realization_audit_markdown": render_profile_realization_audit_markdown(audit),
    }


def build_profile_execution_source_evidence_summary(
    request: InputRequest,
    workspace: Path,
    md_file: Path | None,
) -> str:
    attachment_paths = [Path(path) for path in (request.attachments or []) if str(path or "").strip()]
    suffix_counter = Counter((path.suffix.lower() or "<no_suffix>") for path in attachment_paths)
    attachment_summary = ", ".join(
        f"{suffix}:{count}" for suffix, count in sorted(suffix_counter.items())
    ) or "none"

    manuscript_text = ""
    if md_file and md_file.exists():
        try:
            manuscript_text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            manuscript_text = ""

    heading_count = len(re.findall(r"(?m)^#{1,6}\s+", manuscript_text))
    table_line_count = len(re.findall(r"(?m)^\|.*\|\s*$", manuscript_text))
    image_ref_count = len(re.findall(r"!\[[^\]]*\]\([^)]+\)|<img\b", manuscript_text, flags=re.IGNORECASE))
    figure_mention_count = len(re.findall(r"\b(fig(?:ure)?|table)\b", manuscript_text, flags=re.IGNORECASE))
    manuscript_words = len(re.findall(r"\w+", manuscript_text))

    lower_context = " ".join(
        [
            str(request.instruction or ""),
            " ".join(path.name.lower() for path in attachment_paths),
            manuscript_text[:4000].lower(),
        ]
    ).lower()
    if ".pdf" in attachment_summary and any(
        token in lower_context
        for token in ("abstract", "introduction", "method", "results", "discussion", "conclusion")
    ):
        source_mode = "paper_like"
    elif any(token in lower_context for token in ("case study", "campaign", "business plan", "workflow", "roadmap")):
        source_mode = "case_like"
    else:
        source_mode = "mixed_or_unknown"

    no_extrapolation_terms = (
        "only use",
        "strictly based",
        "do not extrapolate",
        "do not fabricate",
        "仅使用",
        "严格根据",
        "不得补充",
        "不能外推",
        "不要杜撰",
        "只根据",
    )
    boundary_policy = (
        "strict_no_extrapolation"
        if any(term in lower_context for term in no_extrapolation_terms)
        else "not_explicit"
    )

    copied_attachment_dir = workspace / "attachments"
    copied_attachment_count = len(list(copied_attachment_dir.iterdir())) if copied_attachment_dir.exists() else 0

    lines = [
        f"- attachment_count: {len(attachment_paths)} (copied_to_workspace={copied_attachment_count})",
        f"- attachment_types: {attachment_summary}",
        f"- manuscript_word_count: {manuscript_words}",
        f"- manuscript_headings: {heading_count}",
        f"- figure_like_refs: {image_ref_count + figure_mention_count}",
        f"- markdown_table_lines: {table_line_count}",
        f"- inferred_source_mode: {source_mode}",
        f"- source_boundary_policy: {boundary_policy}",
    ]
    return "\n".join(lines)


def _resolve_all_slide_paths(self) -> list[Path]:
    """Return all slide HTML files in sorted order."""
    for slide_dir in self._candidate_slide_dirs():
        if slide_dir.exists() and slide_dir.is_dir():
            slide_paths = sorted(slide_dir.glob("slide_*.html"))
            if slide_paths:
                return slide_paths
    return []

def _candidate_slide_dirs(
    self,
    slide_dir: Path | str | None = None,
    *,
    include_workspace_root: bool = True,
) -> list[Path]:
    """Return de-duplicated slide directory candidates in fallback order."""
    candidate_dirs: list[Path] = []

    def _add(candidate: Path | str | None) -> None:
        if not candidate:
            return
        path = Path(candidate)
        if not path.is_absolute():
            path = self.workspace / path
        if path.suffix:
            candidate_dirs.append(path.parent)
        candidate_dirs.append(path)

    _add(slide_dir)
    _add(self.intermediate_output.get("slide_html_dir"))
    candidate_dirs.extend(
        self.workspace / rel for rel in ("outputs", "outputs/slides", "slides")
    )
    if include_workspace_root:
        candidate_dirs.append(self.workspace)

    seen: set[str] = set()
    deduped_dirs: list[Path] = []
    for candidate in candidate_dirs:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_dirs.append(candidate)
    return deduped_dirs

def _collect_exportable_slide_paths(self, slide_dir: Path) -> list[Path]:
    """Return canonical slide HTML files from a single directory."""
    if not slide_dir.exists() or not slide_dir.is_dir():
        return []

    indexed: list[tuple[int, Path]] = []
    for html in slide_dir.glob("slide_*.html"):
        m = re.fullmatch(r"slide_(\d+)\.html", html.name, re.IGNORECASE)
        if m:
            indexed.append((int(m.group(1)), html))
    indexed.sort(key=lambda x: x[0])
    return [p for _, p in indexed]

def _exportable_slide_dir_score(
    slide_paths: list[Path],
    candidate_index: int,
) -> tuple[int, float, int, int]:
    """Score an exportable slide directory.

    Preference order:
    1. More canonical slides.
    2. Newer slide set.
    3. Larger total HTML payload.
    4. Earlier fallback candidate when all else ties.
    """
    latest_mtime = 0.0
    total_size = 0
    for slide in slide_paths:
        try:
            stat = slide.stat()
        except OSError:
            continue
        latest_mtime = max(latest_mtime, stat.st_mtime)
        total_size += stat.st_size
    return (len(slide_paths), latest_mtime, total_size, -candidate_index)

def _resolve_exportable_slide_dir(self, slide_dir: Path | str | None = None) -> Path | None:
    """Return the best candidate directory that contains exportable slides.

    Some runs leave behind multiple canonical slide directories in the same
    workspace, such as a stale legacy output folder and a newer repaired
    slide set at the workspace root. We rank candidates so export uses the
    most complete, freshest canonical deck instead of blindly taking the
    first directory that happens to contain ``slide_<num>.html`` files.
    """
    best_candidate: Path | None = None
    best_score: tuple[int, float, int, int] | None = None

    for idx, candidate in enumerate(self._candidate_slide_dirs(slide_dir)):
        slide_paths = self._collect_exportable_slide_paths(candidate)
        if not slide_paths:
            continue

        score = self._exportable_slide_dir_score(slide_paths, idx)
        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate

def _resolve_exportable_slide_paths(self, slide_dir: Path | str | None = None) -> list[Path]:
    """Return visible slide HTMLs in numeric order.

    Export should only include canonical files named ``slide_<num>.html``.
    Auxiliary files such as ``slide_04a_hidden.html`` are intentionally
    excluded from PPTX/PDF generation.
    """
    resolved_dir = self._resolve_exportable_slide_dir(slide_dir)
    if resolved_dir is None:
        return []
    return self._collect_exportable_slide_paths(resolved_dir)

def _extract_tag_block(text: str, tag_name: str) -> str:
    """Extract the inner text of a simple XML-like prompt block."""
    source = str(text or "")
    if not source:
        return ""
    start = source.find(f"<{tag_name}")
    if start < 0:
        return ""
    open_end = source.find(">", start)
    if open_end < 0:
        return ""
    close = source.find(f"</{tag_name}>", open_end)
    if close < 0:
        return ""
    return source[open_end + 1 : close].strip()

def _parse_rule_specs_from_injection(cls, text: str) -> list[dict[str, Any]]:
    """Parse JSONL rule specs from <working_memory_rule_specs> injection text."""
    inner = cls._extract_tag_block(text, "working_memory_rule_specs")
    if not inner:
        return []
    specs: list[dict[str, Any]] = []
    for line in inner.splitlines():
        payload = line.strip()
        if not payload or not payload.startswith("{"):
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            specs.append(data)
    return specs

def _selector_hints_for_element_kind(element_kind: str) -> list[str]:
    kind = _canonical_future_preference_element_kind(element_kind)
    mapping = {
        "any_text": [
            "h1",
            "h2",
            "p",
            "li",
            "span",
            "strong",
            "em",
            "b",
            "u",
            "mark",
            "td",
            "th",
            ".title",
            ".subtitle",
            ".body",
            ".content",
            ".caption",
            ".badge",
            ".tag",
            ".callout",
            "[data-role]",
        ],
        "slide_title": ["h1", ".title", "[data-role='title']"],
        "subtitle": ["h2", ".subtitle", "[data-role='subtitle']"],
        "body_text": ["p", "li", "span", "strong", "em", "b", "u", "mark", ".body", ".content"],
        "footer": [".footer", "footer", "[data-role='footer']"],
        "pill_tag": [".badge", ".pill", ".pill-tag", ".tag", ".chip", "[data-role='pill-tag']"],
        "caption": [".caption", "figcaption", ".figure-caption", ".table-caption", ".image-caption", ".fig-caption", "[data-role='caption']"],
        "table_cell": ["td", "th", ".table-cell", "[data-role='table-cell']"],
        "legend_label": [".legend", ".legend-label", "[data-role='legend-label']"],
        "callout": [".callout", ".annotation", "[data-role='callout']"],
    }
    return mapping.get(kind, [])

def _rule_spec_applies_to_existing_slides(spec: dict[str, Any]) -> bool:
    """Return whether a WM rule expects edits on already-rendered slides."""
    if not isinstance(spec, dict):
        return True

    propagation = spec.get("propagation")
    if isinstance(propagation, dict) and "apply_existing_slides" in propagation:
        return _normalize_bool_flag(propagation.get("apply_existing_slides"))

    target = spec.get("target")
    if isinstance(target, dict):
        slide_scope = str(target.get("slide_scope", "") or "").strip().lower()
        if slide_scope in {
            "future",
            "future_only",
            "new",
            "new_only",
            "new_slides",
            "inserted_only",
        }:
            return False

    return True

def _rule_spec_applies_to_future_slides(spec: dict[str, Any]) -> bool:
    """Return whether a WM rule explicitly targets future/new slides."""
    if not isinstance(spec, dict):
        return False

    propagation = spec.get("propagation")
    if isinstance(propagation, dict) and "apply_future_slides" in propagation:
        return _normalize_bool_flag(propagation.get("apply_future_slides"))

    target = spec.get("target")
    if isinstance(target, dict):
        slide_scope = str(target.get("slide_scope", "") or "").strip().lower()
        if slide_scope in {
            "future",
            "future_only",
            "new",
            "new_only",
            "new_slides",
            "inserted_only",
        }:
            return True

    return False

def _parse_style_declarations(style_text: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for chunk in str(style_text or "").split(";"):
        if ":" not in chunk:
            continue
        name, value = chunk.split(":", 1)
        key = name.strip().lower()
        normalized_value = value.strip()
        if key and normalized_value:
            declarations[key] = normalized_value
    return declarations

def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start:end])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}

def _resolve_css_var_value(
    cls,
    value: str,
    css_vars: dict[str, str],
    depth: int = 0,
) -> str:
    if depth > 4:
        return str(value or "").strip()

    text = str(value or "").strip()
    if not text:
        return ""

    match = re.fullmatch(
        r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+?)\s*)?\)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return text

    var_name = match.group(1).strip()
    fallback = (match.group(2) or "").strip()
    resolved = css_vars.get(var_name, fallback)
    if not resolved:
        return ""
    return cls._resolve_css_var_value(resolved, css_vars, depth + 1)

def _extract_inserted_slide_path_from_result(result_text: str) -> str:
    text = str(result_text or "").strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}
    if isinstance(payload, dict):
        for key in ("new_slide_path", "slide_path", "file_path", "path"):
            candidate = str(payload.get(key, "") or "").strip()
            if candidate:
                return candidate

    match = re.search(
        r"(?:New slide path|new_slide_path|slide_path)\s*[:=]\s*[\"']?(?P<path>[^\n\r\"']+)",
        text,
        re.IGNORECASE,
    )
    return match.group("path").strip() if match else ""

def _color_token_to_rgb(color_value: str) -> tuple[int, int, int] | None:
    text = str(color_value or "").strip().lower().replace("!important", "").strip()
    if not text:
        return None

    named_colors = {
        "blue": (0, 0, 255),
        "navy": (0, 0, 128),
        "royalblue": (65, 105, 225),
        "dodgerblue": (30, 144, 255),
        "deepskyblue": (0, 191, 255),
        "cornflowerblue": (100, 149, 237),
        "steelblue": (70, 130, 180),
        "skyblue": (135, 206, 235),
        "lightskyblue": (135, 206, 250),
        "slateblue": (106, 90, 205),
        "mediumblue": (0, 0, 205),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "darkgreen": (0, 100, 0),
    }
    if text in named_colors:
        return named_colors[text]

    hex_match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", text)
    if hex_match:
        raw = hex_match.group(1)
        if len(raw) == 3:
            raw = "".join(ch * 2 for ch in raw)
        return tuple(int(raw[idx: idx + 2], 16) for idx in (0, 2, 4))

    rgb_match = re.fullmatch(r"rgba?\(([^)]+)\)", text)
    if rgb_match:
        parts = [part.strip() for part in rgb_match.group(1).split(",")]
        if len(parts) < 3:
            return None
        values: list[int] = []
        for part in parts[:3]:
            try:
                if part.endswith("%"):
                    values.append(round(float(part[:-1]) * 255 / 100))
                else:
                    values.append(round(float(part)))
            except ValueError:
                return None
        return tuple(max(0, min(255, value)) for value in values)

    return None

def _normalize_css_property_name(property_name: str) -> str:
    prop = str(property_name or "").strip().lower().replace("_", "-")
    return {
        "fontweight": "font-weight",
        "fontstyle": "font-style",
        "textdecoration": "text-decoration",
        "background": "background",
        "background-color": "background-color",
        "bg-color": "background-color",
        "fill": "background-color",
    }.get(prop, prop)

def _normalize_css_value(cls, property_name: str, value: str) -> str:
    prop = cls._normalize_css_property_name(property_name)
    text = str(value or "").strip().lower().replace("!important", "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if prop in {"color", "background-color", "background"}:
        rgb = cls._color_token_to_rgb(text)
        if rgb is not None:
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return text.replace(" ", "")
    if prop == "font-weight":
        if text in {"bold", "bolder"}:
            return "700"
        if text in {"normal", "regular"}:
            return "400"
        numeric = re.fullmatch(r"\d+(?:\.0+)?", text)
        if numeric:
            return str(round(float(text)))
    if prop == "font-style":
        if "italic" in text:
            return "italic"
        if "oblique" in text:
            return "oblique"
    if prop == "text-decoration":
        tokens = {
            token
            for token in re.split(r"[\s,]+", text.replace(";", " "))
            if token and token not in {"solid", "auto"}
        }
        if "none" in tokens:
            return "none"
        ordered = [token for token in ("underline", "overline", "line-through") if token in tokens]
        return " ".join(ordered) if ordered else text
    return text

def _css_values_equivalent(cls, property_name: str, expected: str, observed: str) -> bool:
    return cls._normalize_css_value(property_name, expected) == cls._normalize_css_value(property_name, observed)

def _canonical_future_preference_element_kind(element_kind: Any) -> str:
    value = str(element_kind or "").strip().lower().replace("-", "_")
    aliases = {
        "all_text": "any_text",
        "all_text_elements": "any_text",
        "text": "body_text",
        "body": "body_text",
        "paragraph": "body_text",
        "bullet": "body_text",
        "bullets": "body_text",
        "title": "slide_title",
        "heading": "slide_title",
        "h1": "slide_title",
        "badge": "pill_tag",
        "pill": "pill_tag",
        "pilltag": "pill_tag",
        "pill_tag": "pill_tag",
        "pill-tag": "pill_tag",
        "tag": "pill_tag",
        "chip": "pill_tag",
        "caption": "caption",
        "figcaption": "caption",
        "figure_caption": "caption",
        "image_caption": "caption",
        "table_caption": "caption",
        "legend": "legend_label",
    }
    return aliases.get(value, value)

def _rule_requirement_text(spec: dict[str, Any]) -> str:
    action = spec.get("action") if isinstance(spec, dict) else {}
    return (
        str(action.get("description", "") or "").strip()
        if isinstance(action, dict)
        else ""
    ) or str(spec.get("normalized_sentence", "") or "").strip() or str(spec.get("content", "") or "").strip()

def _verifiable_rule_css_values(cls, spec: dict[str, Any]) -> dict[str, str]:
    action = spec.get("action") if isinstance(spec, dict) else {}
    css_values = action.get("css_values", {}) if isinstance(action, dict) else {}
    if not isinstance(css_values, dict):
        return {}
    values: dict[str, str] = {}
    for raw_prop, raw_value in css_values.items():
        prop = cls._normalize_css_property_name(str(raw_prop or ""))
        value = str(raw_value or "").strip()
        if prop and value:
            values[prop] = value
    return values

def _future_slide_rule_needs_semantic_resolution(spec: dict[str, Any]) -> bool:
    target = spec.get("target") if isinstance(spec, dict) else {}
    condition = spec.get("condition") if isinstance(spec, dict) else {}
    semantic_target = target.get("semantic_target", "") if isinstance(target, dict) else ""
    return (
        bool(semantic_target)
        or str(condition.get("match_granularity", "") or "").strip().lower() == "text_span"
        or str(condition.get("apply_to", "") or "").strip().lower() == "semantic_span_only"
    )

def _extract_element_style_records(cls, html_text: str, element_kind: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    selectors = cls._selector_hints_for_element_kind(element_kind)
    if not selectors:
        return []
    nodes: list[Any] = []
    seen_nodes: set[int] = set()
    for selector in selectors:
        try:
            candidates = soup.select(selector)
        except Exception:
            candidates = []
        for node in candidates:
            if id(node) in seen_nodes:
                continue
            seen_nodes.add(id(node))
            nodes.append(node)

    css_vars: dict[str, str] = {}
    rules: list[tuple[str, dict[str, str]]] = []
    for style_tag in soup.find_all("style"):
        css_text = style_tag.string or style_tag.get_text() or ""
        for match in re.finditer(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", css_text, re.DOTALL):
            selectors_text = str(match.group("selectors") or "").strip()
            if not selectors_text or selectors_text.startswith("@"):
                continue
            declarations = cls._parse_style_declarations(match.group("body"))
            for key, value in declarations.items():
                if key.startswith("--"):
                    css_vars[key] = value
            for raw_selector in selectors_text.split(","):
                selector = raw_selector.strip()
                if selector:
                    rules.append((selector, declarations))

    body_declarations = cls._parse_style_declarations((soup.body or {}).get("style", "") if soup.body else "")
    records: list[dict[str, str]] = []
    for node in nodes:
        computed: dict[str, str] = {}
        inline = cls._parse_style_declarations(node.get("style", ""))
        for prop, value in body_declarations.items():
            normalized_prop = cls._normalize_css_property_name(prop)
            if normalized_prop:
                computed[normalized_prop] = cls._resolve_css_var_value(value, css_vars)
        matched_selectors: list[str] = []
        for selector, declarations in rules:
            try:
                matches = node in soup.select(selector)
            except Exception:
                matches = False
            if not matches:
                continue
            matched_selectors.append(selector)
            for prop, value in declarations.items():
                normalized_prop = cls._normalize_css_property_name(prop)
                if normalized_prop:
                    computed[normalized_prop] = cls._resolve_css_var_value(value, css_vars)
        for prop, value in inline.items():
            normalized_prop = cls._normalize_css_property_name(prop)
            if normalized_prop:
                computed[normalized_prop] = cls._resolve_css_var_value(value, css_vars)
        if not computed.get("background-color") and computed.get("background"):
            computed["background-color"] = computed["background"]
        record = {
            "text": " ".join(node.get_text(" ", strip=True).split()),
            "selectors": ", ".join(matched_selectors[:4]),
            "tag": str(getattr(node, "name", "") or ""),
        }
        record.update(computed)
        record.setdefault("color", "")
        record.setdefault("font-style", "")
        record.setdefault("font_style", record.get("font-style", ""))
        record.setdefault("font-weight", "")
        record.setdefault("font_weight", record.get("font-weight", ""))
        record.setdefault("text-decoration", "")
        record.setdefault("text_decoration", record.get("text-decoration", ""))
        record.setdefault("background-color", "")
        record.setdefault("background_color", record.get("background-color", ""))
        record.setdefault("background", record.get("background-color", ""))
        records.append(record)
    return records

def _record_css_value(cls, record: dict[str, str], property_name: str) -> str:
    prop = cls._normalize_css_property_name(property_name)
    if prop in record:
        return str(record.get(prop, "") or "")
    aliases = {
        "font-style": "font_style",
        "font-weight": "font_weight",
        "text-decoration": "text_decoration",
        "background-color": "background_color",
    }
    alias = aliases.get(prop, "")
    if alias and alias in record:
        return str(record.get(alias, "") or "")
    if prop == "background-color":
        return str(record.get("background", "") or "")
    return ""

def _build_css_value_failures(
    cls,
    *,
    slide_path: Path,
    rule_id: str,
    requirement: str,
    element_kind: str,
    records: list[dict[str, str]],
    expected_values: dict[str, str],
) -> list[dict[str, Any]]:
    if not expected_values:
        return []
    if not records:
        return [
            {
                "slide_path": slide_path,
                "rule_id": rule_id,
                "requirement": requirement,
                "observed_issue": f"no {element_kind} element was found for a concrete CSS WM rule",
            }
        ]
    failures: list[dict[str, Any]] = []
    for record in records:
        for prop, expected in expected_values.items():
            observed = _record_css_value(cls, record, prop)
            if observed and cls._css_values_equivalent(prop, expected, observed):
                continue
            failures.append(
                {
                    "slide_path": slide_path,
                    "rule_id": rule_id,
                    "requirement": requirement,
                    "observed_issue": (
                        f"new {element_kind} text '{record.get('text', '')[:80]}' "
                        f"uses {prop} {observed or '(no value found)'}, expected {expected}"
                    ),
                    "expected_property": prop,
                    "expected_value": expected,
                    "observed_value": observed or "",
                }
            )
    return failures

def _future_preference_verifiable_on_single_slide(cls, spec: dict[str, Any]) -> bool:
    if not isinstance(spec, dict) or not cls._rule_spec_applies_to_future_slides(spec):
        return False
    if not cls._verifiable_rule_css_values(spec):
        return False
    if cls._future_slide_rule_needs_semantic_resolution(spec):
        return False
    target = spec.get("target")
    if not isinstance(target, dict):
        return False
    element_kind = cls._canonical_future_preference_element_kind(target.get("element_kind", ""))
    if not element_kind or element_kind == "deck":
        return False
    return True

def _future_preference_judge_llm(self) -> Any | None:
    modify_agent = getattr(self, "modifyagent", None)
    llm = getattr(modify_agent, "llm", None) if modify_agent is not None else None
    if llm and hasattr(llm, "run"):
        return llm
    design_agent = getattr(self, "designagent", None)
    llm = getattr(design_agent, "llm", None) if design_agent is not None else None
    return llm if llm and hasattr(llm, "run") else None

def _collect_future_slide_preference_failures(
    self,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for raw_path in slide_paths:
        slide_path = Path(raw_path)
        for spec in rule_specs:
            if not isinstance(spec, dict) or not self._rule_spec_applies_to_future_slides(spec):
                continue
            expected_values = self._verifiable_rule_css_values(spec)
            if not expected_values:
                continue
            if self._future_slide_rule_needs_semantic_resolution(spec):
                continue
            target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
            element_kind = self._canonical_future_preference_element_kind(
                target.get("element_kind", "") if isinstance(target, dict) else "",
            )
            if not element_kind or element_kind == "deck":
                continue
            rule_id = str(spec.get("rule_id", "") or "").strip() or "future_preference"
            key = (str(slide_path), rule_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            if not slide_path.exists():
                failures.append(
                    {
                        "slide_path": slide_path,
                        "rule_id": rule_id,
                        "requirement": self._rule_requirement_text(spec),
                        "observed_issue": "slide file missing",
                    }
                )
                continue

            try:
                html_text = slide_path.read_text(encoding="utf-8")
            except OSError:
                failures.append(
                    {
                        "slide_path": slide_path,
                        "rule_id": rule_id,
                        "requirement": self._rule_requirement_text(spec),
                        "observed_issue": "slide file unreadable",
                    }
                )
                continue

            failures.extend(
                self._build_css_value_failures(
                    slide_path=slide_path,
                    rule_id=rule_id,
                    requirement=self._rule_requirement_text(spec),
                    element_kind=element_kind,
                    records=self._extract_element_style_records(html_text, element_kind),
                    expected_values=expected_values,
                )
            )

    return failures

async def _judge_future_slide_preferences_with_llm(
    self,
    slide_path: Path,
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return []

async def _collect_future_slide_preference_failures_async(
    self,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return self._collect_future_slide_preference_failures(slide_paths, rule_specs)

def _build_future_preference_followup(
    self,
    failures: list[dict[str, Any]],
    *,
    include_header: bool = True,
) -> str:
    if not failures:
        return ""

    lines = []
    for item in failures[:8]:
        observed_text = str(item.get("observed_issue", "") or "").strip()
        if not observed_text:
            observed_text = "concrete CSS preference is not satisfied"
        lines.append(
            f"- {self._workspace_relative_label(item['slide_path'])}: "
            f"{observed_text}; active future preference says {item['requirement']}"
        )
    prefix = (
        "SYSTEM: A newly inserted slide is not yet following an active remembered future preference.\n"
        if include_header
        else ""
    )
    return (
        prefix
        + "\n".join(lines)
        + "\nRepair only the new slide(s) that fail this remembered preference. "
        "Do not batch-edit old slides, and do not call `finalize` until the inserted slide(s) comply."
    )

def build_non_template_design_plan_scaffold(
    request: InputRequest,
    profile_execution_contract: dict[str, Any] | None = None,
    profile_execution_plan: dict[str, Any] | None = None,
) -> str:
    def _contract_lines(key: str) -> list[str]:
        values = (profile_execution_contract or {}).get(key, []) or []
        return [str(item).strip() for item in values if str(item or "").strip()]

    def _plan_lines(key: str) -> list[str]:
        values = (profile_execution_plan or {}).get(key, []) or []
        return [str(item).strip() for item in values if str(item or "").strip()]

    def _extract_hex_colors(text: str) -> list[str]:
        return re.findall(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})", str(text or ""))

    def _extract_preference_value(lines: list[str], keywords: tuple[str, ...]) -> str:
        for line in lines:
            lowered = line.lower()
            if not any(keyword in lowered for keyword in keywords):
                continue
            if "：" in line:
                return line.split("：", 1)[1].strip()
            if ":" in line:
                return line.split(":", 1)[1].strip()
            return line.strip()
        return ""

    def _hex_luminance(hex_color: str) -> float:
        token = str(hex_color or "").strip().lstrip("#")
        if len(token) == 3:
            token = "".join(ch * 2 for ch in token)
        if len(token) != 6:
            return 1.0

        def _channel(value: str) -> float:
            normalized = int(value, 16) / 255.0
            if normalized <= 0.03928:
                return normalized / 12.92
            return ((normalized + 0.055) / 1.055) ** 2.4

        try:
            r = _channel(token[0:2])
            g = _channel(token[2:4])
            b = _channel(token[4:6])
        except ValueError:
            return 1.0
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _pick_dark_color(candidates: list[str], fallback: str) -> str:
        for color in candidates:
            if _hex_luminance(color) < 0.3:
                return color
        return candidates[0] if candidates else fallback

    def _dedupe(items: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    planning_focus = str(
        (profile_execution_contract or {}).get("planning_focus")
        or (profile_execution_plan or {}).get("planning_focus")
        or ""
    ).strip()
    persona_ready_scaffold = bool(profile_execution_contract or profile_execution_plan)

    instruction = " ".join(str(request.instruction or "").split())
    if len(instruction) > 220:
        instruction = instruction[:217] + "..."

    style_lines = _contract_lines("style_contract")
    style_req = (
        profile_execution_contract.get("style_requirements")
        if isinstance((profile_execution_contract or {}).get("style_requirements"), dict)
        else {}
    )
    if isinstance(style_req, dict):
        palette_values = style_req.get("palette_values", [])
        if palette_values:
            style_lines.append("主色偏好: " + ", ".join(str(item) for item in palette_values[:6]))
        css_values = style_req.get("css_values", {})
        if isinstance(css_values, dict) and css_values:
            style_lines.append(
                "CSS values: "
                + ", ".join(f"{prop}: {value}" for prop, value in css_values.items())
            )
    content_lines = _contract_lines("content_contract")
    deck_spine_lines = _contract_lines("deck_spine")
    source_boundary_lines = _contract_lines("source_boundary_contract")
    proxy_lines = _contract_lines("proxy_realization_rules")
    do_not_force_lines = _contract_lines("do_not_force")
    global_notes = _plan_lines("global_execution_notes")

    primary_candidates = _extract_hex_colors(
        _extract_preference_value(style_lines, ("主色", "primary"))
    )
    accent_candidates = _extract_hex_colors(
        _extract_preference_value(style_lines, ("强调", "accent"))
    )
    body_text = _pick_dark_color(primary_candidates, "#0F172A")
    primary = (
        primary_candidates[1]
        if len(primary_candidates) >= 2
        else (primary_candidates[0] if primary_candidates else "#2563EB")
    )
    accent = accent_candidates[0] if accent_candidates else "#0EA5E9"
    background_pref = _extract_preference_value(
        style_lines, ("背景", "background")
    ).lower()
    background = "#FFFFFF" if "white" in background_pref or "白" in background_pref else "#FFFFFF"
    surface = "#F8FAFC"
    inverse_text = "#FFFFFF"
    muted_text = "#64748B"
    border = "#CBD5E1"

    font_intent = _extract_preference_value(style_lines, ("字体", "font"))
    font_stack = "Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif"
    title_font_line = f"- Title: {font_stack} | 36-40px | 700"
    subtitle_font_line = f"- Subtitle: {font_stack} | 24-28px | 600"
    body_font_line = f"- Body: {font_stack} | 20-24px | 400"
    caption_font_line = f"- Caption: {font_stack} | 16-18px | 400"
    font_intent_line = f"- Family Intent: {font_intent}" if font_intent else ""

    focus_keywords = {
        "decision_brief": [
            "decision-oriented",
            "evidence-led",
            "restrained",
            "executive-academic",
            "projection-friendly",
        ],
        "mechanism_deep_dive": [
            "mechanistic",
            "technical",
            "annotated",
            "structured",
            "projection-friendly",
        ],
        "teaching_walkthrough": [
            "stepwise",
            "didactic",
            "clear",
            "approachable",
            "projection-friendly",
        ],
    }
    theme_keywords = list(focus_keywords.get(planning_focus, []))
    keyword_line_pool = " ".join(style_lines + content_lines + deck_spine_lines + global_notes).lower()
    if "matrix" in keyword_line_pool or "矩阵" in keyword_line_pool:
        theme_keywords.append("matrix-based")
    if "risk" in keyword_line_pool or "风险" in keyword_line_pool:
        theme_keywords.append("risk-aware")
    if "comparison" in keyword_line_pool or "对比" in keyword_line_pool or "比较" in keyword_line_pool:
        theme_keywords.append("comparative")
    if "callout" in keyword_line_pool or "判断" in keyword_line_pool:
        theme_keywords.append("judgment-forward")
    if not theme_keywords:
        theme_keywords = ["clear", "structured", "modern", "restrained", "projection-friendly"]
    theme_keywords = _dedupe(theme_keywords)[:6]

    profile_pages = list((profile_execution_plan or {}).get("page_plan", []) or [])
    page_roles = _dedupe(
        str(item.get("page_role", "") or "").strip()
        for item in profile_pages
        if isinstance(item, dict)
    )
    if page_roles:
        page_archetypes = [
            f"- {role}: persona-driven page role from the compiled execution plan"
            for role in page_roles[:6]
        ]
    else:
        page_archetypes = [
            "- Cover page: strong title block with concise subtitle or supporting metadata",
            "- Content page: clear title + 1 dominant structure (list, comparison, process, figure-with-caption)",
            "- Ending page: short takeaway summary with high emphasis and low clutter",
        ]

    required_components = _dedupe(
        str(item.get("required_component", "") or "").strip()
        for item in profile_pages
        if isinstance(item, dict)
    )
    component_rules = [
        "- Title bar: short, high-contrast, consistent top alignment",
        "- Keep one dominant structure per slide; persona page role beats generic page shape.",
        "- When a preferred component is source-supported, realize it directly instead of collapsing everything into bullets.",
        "- If a preferred component is not source-supported, preserve the signal through grouping, labels, hierarchy, and emphasis rather than fabricated facts.",
        "- Image: preserve aspect ratio; use contain for informative figures and tables",
        "- Table / chart: keep labels readable, prefer simplified framing over decoration",
    ]
    if required_components:
        component_rules.insert(
            2,
            "- Preferred components in this deck include: "
            + ", ".join(f"`{item}`" for item in required_components[:6]) + ".",
        )

    do_rules = [
        "- Do: keep visual hierarchy obvious and preserve whitespace around dense content",
        "- Do: adapt layout per page content instead of forcing a single template",
    ]
    if any("one main judgment per slide" in line.lower() for line in style_lines + content_lines + deck_spine_lines):
        do_rules.append("- Do: keep one main judgment per slide when the persona plan calls for decision framing")
    if any("claim -> paper evidence -> implication or risk" in line.lower() for line in content_lines):
        do_rules.append("- Do: structure key pages as claim -> source evidence -> implication or risk")
    if proxy_lines:
        do_rules.append(
            "- Do: when the source cannot support a richer persona component, preserve the signal through framing, labels, and hierarchy."
        )

    dont_rules = [
        "- Don't: invent colors outside the palette without a manuscript-driven reason",
        "- Don't: overload a page with too many small text blocks or tiny figures",
        "- Don't: use decorative effects that reduce readability or projection contrast",
    ]
    dont_rules.extend(f"- Don't: {line}" for line in do_not_force_lines[:3])
    if source_boundary_lines:
        dont_rules.append(
            "- Don't: drift beyond source-backed facts, claims, and numbers even when the persona asks for stronger framing"
        )
    dont_rules = _dedupe(dont_rules)

    design_goal_lines = [
        "- Audience: inferred from manuscript and user instruction",
        "- Objective: translate the manuscript into clear, projection-friendly fixed-layout slides",
        "- Tone: professional, structured, high-contrast, concise",
    ]
    if persona_ready_scaffold:
        design_goal_lines.append(
            "- Persona Handling: treat the profile-derived contract and persona page plan as the first-order deck defaults; only refine generic sections if the manuscript creates a real conflict."
        )

    spacing_lines = [
        "- Page margin: 56-64px",
        "- Column gutter: 24-36px",
        "- Card padding: 18-24px",
        "- Default grid: use 1-column for single-focus judgment/mechanism pages and 2-column for comparisons, risk matrices, and benchmark evidence.",
    ]
    if "left aligned" in keyword_line_pool:
        spacing_lines.append("- Alignment bias: left-aligned reading flow unless the manuscript content clearly demands otherwise.")
    if "whitespace" in keyword_line_pool or "留白" in keyword_line_pool:
        spacing_lines.append("- Preserve deliberate whitespace around decision blocks and evidence groupings.")

    plan_action_line = (
        "- Action: this scaffold is already persona-aware. Read it first; if it already fits the manuscript, keep it and proceed. Only refine sections with concrete conflicts or missing detail before generating slide HTML.\n"
        if persona_ready_scaffold
        else "- Action: DeckDesigner agent should read this file first and refine it to match the manuscript before generating any slide HTML if needed.\n"
    )

    base_scaffold = (
        "# Design Plan\n\n"
        "## Plan Status\n"
        "- Source: system scaffold for non-template mode\n"
        f"{plan_action_line}"
        f"- Request Summary: {instruction or 'Generate projection-friendly slides that follow the manuscript closely.'}\n\n"
        "## Design Goal\n"
        + "\n".join(design_goal_lines)
        + "\n\n"
        "## Theme Keywords\n"
        + "\n".join(f"- {item}" for item in theme_keywords)
        + "\n\n"
        "## Color Palette\n"
        f"- Background: {background}\n"
        f"- Surface: {surface}\n"
        f"- Primary Text: {body_text}\n"
        f"- Inverse Text: {inverse_text}\n"
        f"- Primary: {primary}\n"
        f"- Accent: {accent}\n"
        f"- Muted Text: {muted_text}\n"
        f"- Border: {border}\n\n"
        "## Typography\n"
        f"{title_font_line}\n"
        f"{subtitle_font_line}\n"
        f"{body_font_line}\n"
        f"{caption_font_line}\n"
        + (f"{font_intent_line}\n\n" if font_intent_line else "\n")
        + "## Spacing & Grid\n"
        + "\n".join(spacing_lines)
        + "\n\n"
        "## Page Archetypes\n"
        + "\n".join(page_archetypes)
        + "\n\n"
        "## Component Rules\n"
        + "\n".join(component_rules)
        + "\n\n"
        "## Do / Don't\n"
        + "\n".join(do_rules + dont_rules)
        + "\n"
    )

    if not profile_execution_contract:
        return base_scaffold

    contract_sections = render_profile_execution_contract_markdown(profile_execution_contract) + "\n"
    contract_sections += render_profile_execution_page_obligations_markdown(profile_execution_contract) + "\n"
    if profile_execution_plan:
        contract_sections += render_profile_execution_plan_markdown(profile_execution_plan) + "\n"
    _, design_goal_and_rest = base_scaffold.split("## Design Goal\n", 1)
    return (
        "# Design Plan\n\n"
        "## Plan Status\n"
        "- Source: system scaffold for non-template mode\n"
        f"{plan_action_line}"
        f"- Request Summary: {instruction or 'Generate projection-friendly slides that follow the manuscript closely.'}\n\n"
        f"{contract_sections}"
        "## Design Goal\n"
        f"{design_goal_and_rest}"
    )


def normalize_memory_intent(value: Any) -> str:
    from memslides.memory.core.models import normalize_intent_label

    return normalize_intent_label(value)


def resolve_request_memory_intent(request: InputRequest) -> str:
    """Prefer explicit request intent and fall back to instruction classification."""
    from memslides.memory.core.models import classify_intent_by_keywords

    extra_info = request.extra_info or {}

    resolved_intent = normalize_memory_intent(extra_info.get("resolved_memory_intent", ""))
    if resolved_intent:
        return resolved_intent

    explicit_intent = normalize_memory_intent(request.memory_intent)
    if explicit_intent:
        return explicit_intent

    extra_intent = normalize_memory_intent(extra_info)
    if extra_intent:
        return extra_intent

    return classify_intent_by_keywords(request.instruction)


def normalize_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_text_flag(value: Any) -> str:
    return str(value or "").strip()


def render_design_plan_execution_plan(
    plan_path: str,
    *,
    scaffold_created: bool = False,
    scaffold_requires_refinement: bool = False,
    template_generated: bool = False,
    profile_contract_present: bool = False,
) -> str:
    if scaffold_created:
        if scaffold_requires_refinement:
            first_steps = (
                f"- Status: a system scaffold already exists at `{plan_path}`, but refinement is still required.\n"
                f"- Step 1: read `{plan_path}` with `read_file`.\n"
                f"- Step 2: overwrite `{plan_path}` with a manuscript-specific structured design plan using `write_markdown_file`.\n"
            )
        else:
            first_steps = (
                f"- Status: a persona-aware system scaffold already exists at `{plan_path}` and may already be usable.\n"
                f"- Step 1: read `{plan_path}` with `read_file`.\n"
                f"- Step 2: only if the manuscript reveals a concrete gap or conflict, lightly refine `{plan_path}` with `write_markdown_file`.\n"
            )
    elif template_generated:
        first_steps = (
            f"- Status: a template-generated design plan already exists at `{plan_path}`.\n"
            f"- Step 1: read `{plan_path}` with `read_file` before generating slides.\n"
        )
    else:
        first_steps = (
            f"- Status: a design plan should be created or updated at `{plan_path}`.\n"
            f"- Step 1: if it does not exist, create it with `write_markdown_file`; otherwise read it with `read_file`.\n"
        )

    parts = [
        "<design_plan_execution_plan>\n",
        "- Precondition: the incoming manuscript is read-only in DeckDesigner stage; do not rewrite it or any other markdown artifact except `design_plan.md`.\n",
        first_steps,
        "- Step 3: after the latest write, read back the latest design plan with `read_file`.\n",
    ]
    if profile_contract_present:
        parts.append(
            "- Step 4: keep the profile-derived contract sections intact and propagate them into page-level obligations before writing any slide HTML.\n"
        )
    parts.extend([
        "- Step 5: only after the design plan is present and validated, proceed to write slide HTML.\n",
        "</design_plan_execution_plan>",
    ])
    return "".join(parts)


resolve_all_slide_paths = _resolve_all_slide_paths
candidate_slide_dirs = _candidate_slide_dirs
collect_exportable_slide_paths = _collect_exportable_slide_paths
exportable_slide_dir_score = _exportable_slide_dir_score
resolve_exportable_slide_dir = _resolve_exportable_slide_dir
resolve_exportable_slide_paths = _resolve_exportable_slide_paths
extract_tag_block = _extract_tag_block
parse_rule_specs_from_injection = _parse_rule_specs_from_injection
selector_hints_for_element_kind = _selector_hints_for_element_kind
rule_spec_applies_to_existing_slides = _rule_spec_applies_to_existing_slides
rule_spec_applies_to_future_slides = _rule_spec_applies_to_future_slides
parse_style_declarations = _parse_style_declarations
extract_json_object = _extract_json_object
resolve_css_var_value = _resolve_css_var_value
extract_inserted_slide_path_from_result = _extract_inserted_slide_path_from_result
color_token_to_rgb = _color_token_to_rgb
normalize_css_property_name = _normalize_css_property_name
normalize_css_value = _normalize_css_value
css_values_equivalent = _css_values_equivalent
canonical_future_preference_element_kind = _canonical_future_preference_element_kind
rule_requirement_text = _rule_requirement_text
verifiable_rule_css_values = _verifiable_rule_css_values
future_slide_rule_needs_semantic_resolution = _future_slide_rule_needs_semantic_resolution
extract_element_style_records = _extract_element_style_records
record_css_value = _record_css_value
build_css_value_failures = _build_css_value_failures
future_preference_verifiable_on_single_slide = _future_preference_verifiable_on_single_slide
future_preference_judge_llm = _future_preference_judge_llm
collect_future_slide_preference_failures = _collect_future_slide_preference_failures
judge_future_slide_preferences_with_llm = _judge_future_slide_preferences_with_llm
collect_future_slide_preference_failures_async = _collect_future_slide_preference_failures_async
build_future_preference_followup = _build_future_preference_followup
