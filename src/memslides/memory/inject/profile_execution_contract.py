"""Compile routed profile preferences into a design-stage execution contract."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from memslides.memory.extract.llm_compat import extract_response_text, resolve_llm_retry_times

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover - optional dependency
    repair_json = None

logger = logging.getLogger(__name__)

ALLOWED_PLANNING_FOCI = {
    "decision_brief",
    "visual_system_first",
    "teaching_walkthrough",
    "technical_walkthrough",
    "execution_plan",
}

ALLOWED_LAYOUT_BIASES = {
    "single_focus",
    "comparison",
    "process",
    "evidence",
    "recap",
    "decision",
}

PERSONA_PLANNING_FOCUS = {
    "financial_manager": "decision_brief",
    "management_analyst": "decision_brief",
    "legislator": "decision_brief",
    "medical_health_services_manager": "decision_brief",
    "graphic_designer": "visual_system_first",
    "marketing_manager": "visual_system_first",
    "postsecondary_teacher": "teaching_walkthrough",
    "training_development_specialist": "teaching_walkthrough",
    "software_developer": "technical_walkthrough",
    "operations_manager": "execution_plan",
}

PROFILE_EXECUTION_CONTRACT_MAX_TOKENS = 1024
PROFILE_EXECUTION_PLAN_MAX_TOKENS = 1400
PROFILE_EXECUTION_CONTRACT_LLM_PROMPT_MAX_CHARS = 3200
PROFILE_EXECUTION_PLAN_LLM_PROMPT_MAX_CHARS = 4200

_EXPLICIT_COLOR_RE = re.compile(
    r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b|rgba?\(\s*[^)]+\)",
)
_EXPLICIT_CSS_PROP_RE = re.compile(
    r"(?i)\b(background-color|background|bg-color|bg|fill|color|font-style|font-weight|text-decoration)\s*[:=：]\s*(#[0-9a-fA-F]{3,6}\b|rgba?\([^)]+\)|[a-zA-Z0-9_.#%-]+)",
)
_COMPONENT_HINT_RE = re.compile(
    r"(?i)\b(kpi|chip|badge|pill|matrix|table|card|callout|summary|takeaway|decision|risk|contribution|evidence|timeline|pipeline|step|diagram|chart)\b|指标|矩阵|表格|卡片|总结|结论|风险|贡献|证据|流程|步骤|图表"
)

PROFILE_EXECUTION_CONTRACT_PROMPT = """你是“画像到PPT设计计划”的编译器。你的任务不是重复原始画像，而是把当前 active profile / working-memory preference 编译成 design stage 可以直接执行的 deck/page 级 contract。

请严格遵守下面目标：
1. 只保留这次任务里真正会影响 page planning 的内容。
2. 不要机械抄写原始 preference；要把它翻译成 deck spine、page obligations、组件偏好和边界约束。
3. 当前用户指令优先于历史画像；若 source 不支持某画像偏好，必须放进 proxy_realization_rules 或 do_not_force，不能硬套。
4. 输出必须适合 non-template 生成链路，帮助 design agent 直接写 design_plan.md。

## 当前任务上下文
- core_persona: {core_persona}
- current_task_intent: {task_intent}
- memory_read_intent: {read_intent}
- memory_write_intent: {write_intent}

## 当前用户指令
{instruction}

## 已路由后的 active preferences
{active_preferences}

## resolved_intent 摘要
{resolved_intent_summary}

## source evidence 摘要
{source_evidence_summary}

## planning_focus 可选值
- decision_brief: 决策导向、KPI/风险/结论优先
- visual_system_first: 视觉系统、展示顺序、showcase 页面优先
- teaching_walkthrough: 讲解顺序、理解路径、步骤拆解优先
- technical_walkthrough: 架构/机制/权衡/数据流解释优先
- execution_plan: owner-action-timeline-status 落地执行优先

## 输出 schema
输出一个 JSON object，字段必须齐全：
{{
  "planning_focus": "decision_brief | visual_system_first | teaching_walkthrough | technical_walkthrough | execution_plan",
  "deck_spine": ["3-6条，描述整套 deck 的 persona 可观察主线"],
  "page_obligations": [
    {{
      "page_role": "这类页面承担什么角色",
      "required_signal": "本页必须让人看到的 persona 信号",
      "preferred_component": "推荐组件/页型/archetype",
      "priority": "high | medium | low"
    }}
  ],
  "style_contract": ["只保留会影响页面规划的风格约束"],
  "content_contract": ["只保留会影响标题/结论句/要点组织的约束"],
  "source_boundary_contract": ["不能外推/必须基于 source/不能硬造数字 等硬约束"],
  "proxy_realization_rules": ["source 不足时允许怎样做代理实现"],
  "do_not_force": ["这次不该强行实现的画像偏好"],
  "hard_requirements": ["只有结构/资产/来源边界这类可硬执行要求"],
  "style_requirements": {{"css_values": {{"color": "#047857"}}, "palette_values": ["#0F172A"], "soft_tokens": ["academic restrained palette"]}},
  "component_requirements": ["KPI chips", "risk matrix"],
  "soft_signals": ["medium-low density", "one main judgment per slide"],
  "source_boundary_notes": ["source 不足时如何做代理实现"]
}}

## 额外要求
- page_obligations 至少 4 项，优先覆盖 opener / core evidence / synthesis / ending 等不同页型。
- 如果 persona 很强调视觉或教学逻辑，page_obligations 里要体现展示顺序或讲解顺序，而不是只有泛泛的“内容页”。
- 若某些原始 preference 更像 Modify 阶段规则或多轮流程规则，不要放进 contract。
- 只输出 JSON object，不要输出解释文字。
"""

PROFILE_EXECUTION_PLAN_PROMPT = """你是“画像执行计划”编译器。你的任务是把已编译好的 persona deck contract 转成逐页 page plan，让 DeckDesigner 直接知道每一页要承担什么 persona 信号。

要求：
1. page_plan 必须覆盖全部目标页数。
2. 每页都要有明确 page_role，不能写成泛泛的“内容页1/2/3”。
3. 优先把 persona 差异落到叙事顺序、页型、组件偏好和强调层级，而不是只改颜色。
4. 只能基于 source/instruction 允许的内容边界；source 不足时走 fallback_rules，不要编造事实。
5. template_mode=true 时，不要强行推翻模板结构；优先通过 page_role、component framing、slot emphasis 和 proxy realization 承载 persona。

输入上下文：
- target_page_count: {target_page_count}
- template_mode: {template_mode}
- instruction: {instruction}
- source_evidence_summary:
{source_evidence_summary}

已编译好的 deck contract:
{profile_execution_contract}

输出 JSON object，字段必须齐全：
{{
  "planning_focus": "decision_brief | visual_system_first | teaching_walkthrough | technical_walkthrough | execution_plan",
  "global_execution_notes": ["deck级执行说明"],
  "fallback_rules": ["source不足时的代理实现规则"],
  "page_plan": [
    {{
      "page_index": 1,
      "page_role": "本页角色",
      "persona_signal": "本页必须让人感知到的 persona 信号",
      "manuscript_anchor": "对应稿件主题锚点",
      "required_component": "推荐组件/页型",
      "layout_bias": "single_focus | comparison | process | evidence | recap | decision",
      "hard_requirements": ["真正可硬执行的结构/资产/来源边界，不能放模糊风格"],
      "style_requirements": {{"css_values": {{"color": "#047857"}}, "palette_values": ["#0F172A"], "soft_tokens": ["low-saturation academic palette"]}},
      "component_requirements": ["KPI chips", "risk matrix"],
      "soft_signals": ["medium-low density", "one main judgment per slide"],
      "source_boundary_notes": ["source 不支持时只做框架代理实现"],
      "must_preserve": ["兼容旧字段；只放真正应保留的短执行提示，不要塞模糊偏好"],
      "nice_to_have": ["可选增强点"]
    }}
  ]
}}

只输出 JSON object，不要解释。"""


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe_keep_order(items: list[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = _normalize_space(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def _coerce_string_list(value: Any, *, limit: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[\n\r]+|[;；]+", value)
        return _dedupe_keep_order(pieces, limit=limit)
    if isinstance(value, (list, tuple, set)):
        return _dedupe_keep_order([str(item) for item in value], limit=limit)
    return _dedupe_keep_order([str(value)], limit=limit)


def _normalize_css_property_name(property_name: str) -> str:
    prop = str(property_name or "").strip().lower().replace("_", "-")
    return {
        "fontweight": "font-weight",
        "fontstyle": "font-style",
        "textdecoration": "text-decoration",
        "background": "background",
        "bg": "background-color",
        "bg-color": "background-color",
        "fill": "background-color",
    }.get(prop, prop)


def _extract_concrete_style_requirements(*texts: str) -> dict[str, Any]:
    """Split literal CSS requirements from softer palette/style hints."""
    css_values: dict[str, str] = {}
    palette_values: list[str] = []
    soft_tokens: list[str] = []

    for raw_text in texts:
        text = str(raw_text or "").strip()
        if not text:
            continue

        consumed_spans: list[tuple[int, int]] = []
        for match in _EXPLICIT_CSS_PROP_RE.finditer(text):
            prop = _normalize_css_property_name(match.group(1))
            value = str(match.group(2) or "").strip().rstrip("。；;,，")
            if not prop or not value:
                continue
            if prop in {"background"}:
                prop = "background-color"
            if prop in {"color", "background-color"} and not _EXPLICIT_COLOR_RE.fullmatch(value):
                continue
            if prop in {"color", "background-color", "font-style", "font-weight", "text-decoration"}:
                css_values.setdefault(prop, value)
                consumed_spans.append(match.span(2))

        for match in _EXPLICIT_COLOR_RE.finditer(text):
            span = match.span()
            if any(start <= span[0] and span[1] <= end for start, end in consumed_spans):
                continue
            palette_values.append(match.group(0).strip())

        lowered = text.lower()
        if re.search(r"(?i)\bitalic\b|斜体", text):
            css_values.setdefault("font-style", "italic")
        if re.search(r"(?i)\bbold\b|加粗|粗体", text):
            css_values.setdefault("font-weight", "700")
        if re.search(r"(?i)\bunderline\b|下划线", text):
            css_values.setdefault("text-decoration", "underline")
        if any(token in lowered for token in ("academic", "executive", "density", "whitespace", "muted", "soft", "low-saturation")) or any(
            token in text for token in ("学术", "决策", "密度", "留白", "柔和", "低饱和", "克制")
        ):
            soft_tokens.append(text)

    payload: dict[str, Any] = {}
    if css_values:
        payload["css_values"] = css_values
    if palette_values:
        payload["palette_values"] = _dedupe_keep_order(palette_values, limit=8)
    if soft_tokens:
        payload["soft_tokens"] = _dedupe_keep_order(soft_tokens, limit=8)
    return payload


def _coerce_style_requirements(value: Any, *fallback_texts: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(value, dict):
        raw_css_values = value.get("css_values")
        if isinstance(raw_css_values, dict):
            css_values: dict[str, str] = {}
            for raw_prop, raw_value in raw_css_values.items():
                prop = _normalize_css_property_name(str(raw_prop or ""))
                val = _normalize_space(raw_value)
                if prop and val:
                    css_values[prop] = val
            if css_values:
                payload["css_values"] = css_values
        palette_values = _coerce_string_list(value.get("palette_values"), limit=8)
        if palette_values:
            payload["palette_values"] = palette_values
        soft_tokens = _coerce_string_list(value.get("soft_tokens"), limit=8)
        if soft_tokens:
            payload["soft_tokens"] = soft_tokens

    derived = _extract_concrete_style_requirements(*fallback_texts)
    if derived.get("css_values"):
        css_values = dict(payload.get("css_values", {}) or {})
        for prop, val in derived["css_values"].items():
            css_values.setdefault(prop, val)
        if css_values:
            payload["css_values"] = css_values
    if derived.get("palette_values"):
        payload["palette_values"] = _dedupe_keep_order(
            list(payload.get("palette_values", []) or []) + list(derived["palette_values"]),
            limit=8,
        )
    if derived.get("soft_tokens"):
        payload["soft_tokens"] = _dedupe_keep_order(
            list(payload.get("soft_tokens", []) or []) + list(derived["soft_tokens"]),
            limit=8,
        )
    return payload


def _derive_component_requirements(
    explicit: Any,
    *,
    required_component: str = "",
    preference_texts: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    items = _coerce_string_list(explicit, limit=limit)
    if required_component:
        items.append(required_component)
    for text in preference_texts or []:
        clean = _normalize_space(text)
        if clean and _COMPONENT_HINT_RE.search(clean):
            items.append(clean)
    return _dedupe_keep_order(items, limit=limit)


def _derive_soft_signals(
    explicit: Any,
    *,
    texts: list[str] | None = None,
    exclude: list[str] | None = None,
    limit: int = 8,
) -> list[str]:
    excluded = {_normalize_space(item) for item in (exclude or []) if _normalize_space(item)}
    items = _coerce_string_list(explicit, limit=limit)
    for text in texts or []:
        clean = _normalize_space(text)
        if not clean or clean in excluded:
            continue
        if _EXPLICIT_CSS_PROP_RE.search(clean) or _COMPONENT_HINT_RE.search(clean):
            continue
        items.append(clean)
    return _dedupe_keep_order(items, limit=limit)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        if repair_json is None:
            return None
        try:
            repaired = repair_json(candidate)
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _explicit_planning_focus_from_text(text: str) -> str:
    task_text = str(text or "").strip().lower().replace("-", "_")
    if not task_text:
        return ""
    if any(term in task_text for term in ("decision", "brief", "executive", "pi_", "kpi", "roi", "recommendation", "决策", "简报")):
        return "decision_brief"
    if any(term in task_text for term in ("teacher", "teaching", "training", "course", "tutorial", "lesson", "教育", "教学", "培训", "讲解", "课程")):
        return "teaching_walkthrough"
    if any(
        term in task_text
        for term in (
            "paper",
            "pdf",
            "academic",
            "research",
            "source",
            "source_derived",
            "source_grounded",
            "manuscript",
            "literature",
            "developer",
            "technical",
            "architecture",
            "mechanism",
            "deep_dive",
            "system",
            "data_flow",
            "model",
            "transformer",
            "attention",
            "multi_head",
            "q/k/v",
            "工程",
            "技术",
            "软件",
            "机制",
            "深挖",
            "架构",
            "模型",
            "公式",
            "论文",
            "学术",
            "文献",
            "实验证据",
        )
    ):
        return "technical_walkthrough"
    if any(term in task_text for term in ("design", "brand", "campaign", "visual", "营销", "视觉")):
        return "visual_system_first"
    if any(term in task_text for term in ("operation", "workflow", "execution", "owner", "运营", "执行")):
        return "execution_plan"
    return ""


def infer_planning_focus(core_persona: str = "", task_intent: str = "") -> str:
    explicit_task_focus = _explicit_planning_focus_from_text(task_intent)
    if explicit_task_focus:
        return explicit_task_focus

    persona_key = re.sub(r"[^a-z_]+", "", str(core_persona or "").strip().lower())
    if persona_key in PERSONA_PLANNING_FOCUS:
        return PERSONA_PLANNING_FOCUS[persona_key]

    explicit_persona_focus = _explicit_planning_focus_from_text(core_persona)
    if explicit_persona_focus:
        return explicit_persona_focus
    return "decision_brief"


def infer_planning_focus_from_context(
    *,
    core_persona: str = "",
    task_intent: str = "",
    instruction: str = "",
    active_preferences: dict[str, list[str]] | None = None,
    source_evidence_summary: str = "",
) -> str:
    """Infer planning focus without letting generic visual wording become branding.

    Round-0 paper decks often have intents such as ``paper_visual_*`` where
    "visual" means source-derived explanation, not a brand/campaign showcase.
    Prefer explicit task intent first, then the concrete instruction/source/profile
    context, and only then fall back to persona defaults.
    """
    explicit_task_focus = _explicit_planning_focus_from_text(task_intent)
    if explicit_task_focus:
        return explicit_task_focus

    preference_text = ""
    if active_preferences:
        preference_text = "\n".join(
            item
            for values in active_preferences.values()
            for item in values[:4]
        )
    contextual_focus = _explicit_planning_focus_from_text(
        "\n".join([instruction or "", source_evidence_summary or "", preference_text])
    )
    if contextual_focus:
        return contextual_focus

    return infer_planning_focus(core_persona, task_intent)


def _format_active_preferences(active_preferences: dict[str, list[str]]) -> str:
    if not active_preferences:
        return "(none)"

    ordered_dims = ("theme", "visual", "layout", "content", "general")
    lines: list[str] = []
    for dim in ordered_dims:
        items = active_preferences.get(dim, [])
        if not items:
            continue
        lines.append(f"### {dim}")
        for item in items[:8]:
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else "(none)"


def _compact_multiline_text(text: Any, *, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _bounded_retry_times(llm: Any) -> int:
    resolved = resolve_llm_retry_times(llm, minimum=1)
    return max(1, resolved)


def _pref_get(pref: Any, key: str, default: Any = "") -> Any:
    if isinstance(pref, dict):
        return pref.get(key, default)
    return getattr(pref, key, default)


def format_profile_execution_contract(contract: dict[str, Any] | None) -> str:
    if not contract:
        return ""

    lines = [
        '<profile_execution_contract priority="highest" note="Translate active profile into page-level design obligations. Obey current user instruction when conflicts occur; use proxy realization instead of forcing unsupported preferences.">',
        f"- Planning focus: {contract.get('planning_focus', 'decision_brief')}",
    ]

    deck_spine = _coerce_string_list(contract.get("deck_spine"), limit=6)
    if deck_spine:
        lines.append("- Deck spine:")
        for item in deck_spine:
            lines.append(f"  - {item}")

    obligations = contract.get("page_obligations") or []
    if obligations:
        lines.append("- Page obligations:")
        for idx, item in enumerate(obligations[:8], start=1):
            if not isinstance(item, dict):
                continue
            page_role = _normalize_space(item.get("page_role", "")) or f"page_{idx}"
            required_signal = _normalize_space(item.get("required_signal", "")) or "persona-aligned page signal"
            preferred_component = _normalize_space(item.get("preferred_component", "")) or "fit-for-content archetype"
            priority = _normalize_space(item.get("priority", "")) or "medium"
            lines.append(
                f"  - {idx}. role={page_role}; signal={required_signal}; component={preferred_component}; priority={priority}"
            )

    proxy_rules = _coerce_string_list(contract.get("proxy_realization_rules"), limit=6)
    if proxy_rules:
        lines.append("- Proxy realization rules:")
        for item in proxy_rules:
            lines.append(f"  - {item}")

    style_requirements = contract.get("style_requirements") if isinstance(contract.get("style_requirements"), dict) else {}
    css_values = style_requirements.get("css_values", {}) if isinstance(style_requirements, dict) else {}
    palette_values = style_requirements.get("palette_values", []) if isinstance(style_requirements, dict) else []
    component_requirements = _coerce_string_list(contract.get("component_requirements"), limit=8)
    soft_signals = _coerce_string_list(contract.get("soft_signals"), limit=8)
    if css_values or palette_values:
        lines.append("- Style realization contract:")
        if isinstance(css_values, dict) and css_values:
            lines.append(
                "  - concrete css_values: "
                + ", ".join(f"{prop}={value}" for prop, value in css_values.items())
            )
        if palette_values:
            lines.append("  - palette_values: " + ", ".join(str(item) for item in palette_values[:8]))
    if component_requirements:
        lines.append("- Semantic component contract:")
        for item in component_requirements[:8]:
            lines.append(f"  - {item}")
    if soft_signals:
        lines.append("- Soft persona signals:")
        for item in soft_signals[:8]:
            lines.append(f"  - {item}")

    do_not_force = _coerce_string_list(contract.get("do_not_force"), limit=6)
    if do_not_force:
        lines.append("- Do not force:")
        for item in do_not_force:
            lines.append(f"  - {item}")

    lines.append("</profile_execution_contract>")
    return "\n".join(lines)


def format_profile_execution_plan(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ""

    lines = [
        '<profile_execution_plan priority="highest" note="Use this page-by-page persona execution plan as the operative checklist when writing the deck.">',
        f"- Planning focus: {plan.get('planning_focus', 'decision_brief')}",
    ]

    notes = _coerce_string_list(plan.get("global_execution_notes"), limit=6)
    if notes:
        lines.append("- Global execution notes:")
        for item in notes:
            lines.append(f"  - {item}")

    for page in plan.get("page_plan", []) or []:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index", 0) or 0)
        page_role = _normalize_space(page.get("page_role", "")) or f"page_{page_index or 'x'}"
        persona_signal = _normalize_space(page.get("persona_signal", "")) or "persona-aligned page signal"
        manuscript_anchor = _normalize_space(page.get("manuscript_anchor", "")) or "fit-for-source content cluster"
        required_component = _normalize_space(page.get("required_component", "")) or "fit-for-content archetype"
        layout_bias = _normalize_space(page.get("layout_bias", "")) or "single_focus"
        lines.append(
            f"- Page {page_index}: role={page_role}; signal={persona_signal}; anchor={manuscript_anchor}; component={required_component}; layout_bias={layout_bias}"
        )
        must_preserve = _coerce_string_list(page.get("must_preserve"), limit=4)
        nice_to_have = _coerce_string_list(page.get("nice_to_have"), limit=3)
        hard_requirements = _coerce_string_list(page.get("hard_requirements"), limit=3)
        component_requirements = _coerce_string_list(page.get("component_requirements"), limit=4)
        soft_signals = _coerce_string_list(page.get("soft_signals"), limit=4)
        style_requirements = page.get("style_requirements") if isinstance(page.get("style_requirements"), dict) else {}
        css_values = style_requirements.get("css_values", {}) if isinstance(style_requirements, dict) else {}
        palette_values = style_requirements.get("palette_values", []) if isinstance(style_requirements, dict) else []
        if hard_requirements:
            lines.append("  - Hard requirements:")
            for item in hard_requirements:
                lines.append(f"    - {item}")
        if css_values or palette_values:
            lines.append("  - Style requirements:")
            if isinstance(css_values, dict) and css_values:
                lines.append(
                    "    - css_values: "
                    + ", ".join(f"{prop}={value}" for prop, value in css_values.items())
                )
            if palette_values:
                lines.append("    - palette_values: " + ", ".join(str(item) for item in palette_values[:6]))
        if component_requirements:
            lines.append("  - Component requirements:")
            for item in component_requirements:
                lines.append(f"    - {item}")
        if soft_signals:
            lines.append("  - Soft persona signals:")
            for item in soft_signals:
                lines.append(f"    - {item}")
        if must_preserve:
            lines.append("  - Must preserve:")
            for item in must_preserve:
                lines.append(f"    - {item}")
        if nice_to_have:
            lines.append("  - Nice to have:")
            for item in nice_to_have:
                lines.append(f"    - {item}")

    fallback_rules = _coerce_string_list(plan.get("fallback_rules"), limit=6)
    if fallback_rules:
        lines.append("- Fallback rules:")
        for item in fallback_rules:
            lines.append(f"  - {item}")

    lines.append("</profile_execution_plan>")
    return "\n".join(lines)


class ProfileExecutionContractCompiler:
    """Compile active profile preferences into a design-stage execution contract."""

    def __init__(self, llm: Any = None):
        self._llm = llm

    @staticmethod
    def collect_active_preferences(wm_preferences: list[Any] | None) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {
            "theme": [],
            "visual": [],
            "layout": [],
            "content": [],
            "general": [],
        }
        for pref in wm_preferences or []:
            if bool(_pref_get(pref, "superseded", False)):
                continue
            dimension = str(_pref_get(pref, "dimension", "") or "general")
            top_level = dimension.split(".")[0] if "." in dimension else dimension
            if top_level == "template":
                continue
            if top_level not in grouped:
                top_level = "general"
            content = _normalize_space(_pref_get(pref, "content", ""))
            if content:
                grouped[top_level].append(content)
        return {key: _dedupe_keep_order(value, limit=10) for key, value in grouped.items() if value}

    async def compile(
        self,
        *,
        wm_preferences: list[Any] | None,
        instruction: str,
        resolved_intent_artifact: dict[str, Any] | None = None,
        source_evidence_summary: str = "",
        core_persona: str = "",
        task_intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
    ) -> dict[str, Any] | None:
        active_preferences = self.collect_active_preferences(wm_preferences)
        if not active_preferences:
            return None

        resolved_intent_summary = json.dumps(
            resolved_intent_artifact or {},
            ensure_ascii=False,
            indent=2,
        )[:2000] or "{}"

        contract: dict[str, Any] | None = None
        if self._llm:
            prompt = PROFILE_EXECUTION_CONTRACT_PROMPT.format(
                core_persona=core_persona or "unspecified",
                task_intent=task_intent or "unspecified",
                read_intent=read_intent or "unspecified",
                write_intent=write_intent or "unspecified",
                instruction=_compact_multiline_text(instruction, limit=1200),
                active_preferences=_format_active_preferences(active_preferences),
                resolved_intent_summary=_compact_multiline_text(resolved_intent_summary, limit=900),
                source_evidence_summary=_compact_multiline_text(source_evidence_summary or "(none)", limit=700),
            )
            if len(prompt) <= PROFILE_EXECUTION_CONTRACT_LLM_PROMPT_MAX_CHARS:
                contract = await self._compile_with_llm(prompt)
            else:
                logger.info(
                    "ProfileExecutionContractCompiler skipping LLM refinement due to prompt size (%d chars)",
                    len(prompt),
                )

        normalized = self._normalize_contract(
            contract,
            active_preferences=active_preferences,
            core_persona=core_persona,
            task_intent=task_intent,
            instruction=instruction,
            source_evidence_summary=source_evidence_summary,
        )
        return normalized

    async def _compile_with_llm(self, prompt: str) -> dict[str, Any] | None:
        request_kwargs = {
            "max_tokens": PROFILE_EXECUTION_CONTRACT_MAX_TOKENS,
            "temperature": 0.0,
            "reasoning_effort": "minimal",
        }
        retry_times = _bounded_retry_times(self._llm)
        try:
            if hasattr(self._llm, "run"):
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                    retry_times=retry_times,
                    request_kwargs=request_kwargs,
                )
                text = extract_response_text(response)
            else:
                response = await self._llm(prompt)
                text = extract_response_text(response)
        except Exception as exc:
            logger.warning("ProfileExecutionContractCompiler LLM call failed: %s", exc)
            return None
        return _parse_json_object(text)

    def _normalize_contract(
        self,
        contract: dict[str, Any] | None,
        *,
        active_preferences: dict[str, list[str]],
        core_persona: str,
        task_intent: str,
        instruction: str,
        source_evidence_summary: str,
    ) -> dict[str, Any]:
        contract = contract or {}
        inferred_focus = infer_planning_focus_from_context(
            core_persona=core_persona,
            task_intent=task_intent,
            instruction=instruction,
            active_preferences=active_preferences,
            source_evidence_summary=source_evidence_summary,
        )
        preference_text_for_focus = "\n".join(
            item
            for values in active_preferences.values()
            for item in values[:4]
        )
        explicit_task_focus = _explicit_planning_focus_from_text(task_intent) or _explicit_planning_focus_from_text(
            "\n".join([instruction or "", source_evidence_summary or "", preference_text_for_focus])
        )
        raw_planning_focus = str(contract.get("planning_focus", "") or "").strip()
        focus_overridden = (
            bool(explicit_task_focus)
            and raw_planning_focus in ALLOWED_PLANNING_FOCI
            and raw_planning_focus != explicit_task_focus
        )
        planning_focus = raw_planning_focus
        if planning_focus not in ALLOWED_PLANNING_FOCI:
            planning_focus = inferred_focus
        elif explicit_task_focus and planning_focus != explicit_task_focus:
            planning_focus = explicit_task_focus
        if focus_overridden:
            contract = dict(contract)
            contract.pop("deck_spine", None)
            contract.pop("page_obligations", None)

        style_defaults = active_preferences.get("theme", [])[:3] + active_preferences.get("visual", [])[:3]
        content_defaults = active_preferences.get("content", [])[:3] + active_preferences.get("general", [])[:2]
        deck_spine = _coerce_string_list(contract.get("deck_spine"), limit=6)
        if not deck_spine:
            deck_spine = self._fallback_deck_spine(planning_focus, active_preferences)

        page_obligations = self._normalize_page_obligations(contract.get("page_obligations"))
        if not page_obligations:
            page_obligations = self._fallback_page_obligations(planning_focus, active_preferences)

        source_boundary_contract = _coerce_string_list(contract.get("source_boundary_contract"), limit=6)
        if not source_boundary_contract:
            source_boundary_contract = self._fallback_source_boundaries(source_evidence_summary)

        proxy_realization_rules = _coerce_string_list(contract.get("proxy_realization_rules"), limit=6)
        if not proxy_realization_rules:
            proxy_realization_rules = [
                "When the source does not support a preferred component, preserve the persona signal through page ordering, emphasis hierarchy, and labels instead of inventing new facts.",
                "Use typography, spacing, grouping, and section framing as fallback carriers when source evidence is too weak for a richer visual treatment.",
            ]

        do_not_force = _coerce_string_list(contract.get("do_not_force"), limit=6)
        if not do_not_force:
            do_not_force = [
                "Do not force persona preferences that would require unsupported numbers, unsupported claims, or fabricated evidence.",
                "Do not let a favorite archetype override the actual source structure when the content clearly calls for another page form.",
            ]

        style_contract = _coerce_string_list(contract.get("style_contract"), limit=6) or style_defaults
        content_contract = _coerce_string_list(contract.get("content_contract"), limit=6) or content_defaults
        all_profile_texts = [
            *style_contract,
            *content_contract,
            *active_preferences.get("layout", []),
            *active_preferences.get("general", []),
            *active_preferences.get("visual", []),
        ]
        component_requirements = _derive_component_requirements(
            contract.get("component_requirements"),
            preference_texts=[
                str(item.get("preferred_component", "") or "")
                for item in page_obligations
                if isinstance(item, dict)
            ]
            + all_profile_texts,
            limit=10,
        )
        style_requirements = _coerce_style_requirements(
            contract.get("style_requirements"),
            *style_contract,
            *active_preferences.get("theme", []),
            *active_preferences.get("visual", []),
        )
        hard_requirements = _coerce_string_list(contract.get("hard_requirements"), limit=8)
        soft_signals = _derive_soft_signals(
            contract.get("soft_signals"),
            texts=all_profile_texts,
            exclude=hard_requirements + component_requirements,
            limit=10,
        )

        return {
            "planning_focus": planning_focus,
            "deck_spine": deck_spine,
            "page_obligations": page_obligations,
            "style_contract": style_contract,
            "content_contract": content_contract,
            "source_boundary_contract": source_boundary_contract,
            "proxy_realization_rules": proxy_realization_rules,
            "do_not_force": do_not_force,
            "hard_requirements": hard_requirements,
            "style_requirements": style_requirements,
            "component_requirements": component_requirements,
            "soft_signals": soft_signals,
            "source_boundary_notes": source_boundary_contract,
        }

    @staticmethod
    def _normalize_page_obligations(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []

        normalized: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, str):
                text = _normalize_space(item)
                if not text:
                    continue
                normalized.append({
                    "page_role": text,
                    "required_signal": "persona-aligned page signal",
                    "preferred_component": "fit-for-content archetype",
                    "priority": "medium",
                })
                continue

            if not isinstance(item, dict):
                continue
            page_role = _normalize_space(item.get("page_role", ""))
            required_signal = _normalize_space(item.get("required_signal", ""))
            preferred_component = _normalize_space(item.get("preferred_component", ""))
            priority = _normalize_space(item.get("priority", "")).lower()
            if not page_role and not required_signal:
                continue
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            normalized.append({
                "page_role": page_role or "persona-specific page",
                "required_signal": required_signal or "persona-aligned page signal",
                "preferred_component": preferred_component or "fit-for-content archetype",
                "priority": priority,
            })

        return normalized[:10]

    @staticmethod
    def _fallback_deck_spine(
        planning_focus: str,
        active_preferences: dict[str, list[str]],
    ) -> list[str]:
        general = active_preferences.get("general", [])
        layout = active_preferences.get("layout", [])
        content = active_preferences.get("content", [])

        focus_specific = {
            "decision_brief": [
                "Open with the decision frame, scope boundary, and the success lens used to judge the deck.",
                "Move quickly into the highest-signal evidence pages that surface trade-offs, ROI, risk, or operational implications.",
                "End with a concise synthesis that converts evidence into a recommendation or next-step framing.",
            ],
            "visual_system_first": [
                "Establish the visual direction early so the deck reads as an intentional system rather than disconnected pages.",
                "Sequence pages as showcases of one dominant visual idea at a time, with clear escalation and contrast.",
                "Close with a distilled visual synthesis anchored in the source or task, rather than a generic summary slide.",
            ],
            "teaching_walkthrough": [
                "Start by orienting the audience to the learning target and the path of explanation.",
                "Unfold the topic through stepwise explanation pages that progressively build understanding.",
                "Finish with a recap or transfer page that helps the audience retain the key lesson.",
            ],
            "technical_walkthrough": [
                "Start with the system problem, architecture lens, or evaluation target before drilling into details.",
                "Use the middle pages to explain mechanisms, data flow, constraints, and trade-offs in a structured order.",
                "Close with limitations, engineering implications, or implementation guidance tied back to the source.",
            ],
            "execution_plan": [
                "Frame the deck around owners, actions, timing, and operational checkpoints instead of abstract description alone.",
                "Use middle pages to map workstreams, dependencies, blockers, and status signals into execution structure.",
                "Close with a clear execution summary that leaves no ambiguity about follow-up actions.",
            ],
        }

        extras = _dedupe_keep_order(general[:1] + layout[:1] + content[:1], limit=3)
        return _dedupe_keep_order(focus_specific.get(planning_focus, focus_specific["decision_brief"]) + extras, limit=6)

    @staticmethod
    def _fallback_page_obligations(
        planning_focus: str,
        active_preferences: dict[str, list[str]],
    ) -> list[dict[str, str]]:
        content_hint = active_preferences.get("content", ["Use concise, persona-aligned headlines."])[0]
        visual_hint = active_preferences.get("visual", ["Use one dominant component per page."])[0]
        layout_hint = active_preferences.get("layout", ["Keep page structure legible and uncluttered."])[0]

        focus_map = {
            "decision_brief": [
                ("decision opener", "Immediate decision framing and evaluation lens", "headline + KPI chips"),
                ("evidence comparison", "Comparable options, drivers, or trade-offs", "comparison table or ranked comparison"),
                ("risk and boundary page", "Risks, assumptions, and scope boundaries are explicit", "risk table or callout matrix"),
                ("recommendation close", "A concise recommendation or action recommendation", "summary box + next-step bullets"),
            ],
            "visual_system_first": [
                ("visual opener", "The visual direction is established immediately", "hero composition"),
                ("showcase page", "One dominant visual story rather than many small fragments", "large image or visual collage"),
                ("system detail page", "Typography/color/layout system becomes legible", "annotated layout or modular grid"),
                ("visual synthesis close", "The final page resolves into a crisp source-backed visual takeaway", "synthesis panel + key proof"),
            ],
            "teaching_walkthrough": [
                ("learning opener", "Learning target and lesson path are explicit", "goal block + agenda"),
                ("concept build page", "One concept is explained with progressive structure", "step diagram or concept card sequence"),
                ("worked example page", "A concrete example bridges abstract content to understanding", "example walkthrough"),
                ("recap page", "Key takeaways are easy to retain and repeat", "recap checklist or summary ladder"),
            ],
            "technical_walkthrough": [
                ("problem framing", "The technical problem and system lens are explicit", "architecture/problem frame"),
                ("source structure overview", "The source-derived architecture, object, or system boundary is visible", "system block diagram"),
                ("mechanism decomposition", "Core mechanism or pipeline is broken into understandable units", "pipeline diagram"),
                ("formula and term explanation", "Important formulas, variables, or terms get dedicated explanatory space", "formula callout + variable map"),
                ("evidence page", "Empirical evidence, benchmark, or source table is surfaced as structure", "benchmark table or result chart"),
                ("contribution synthesis", "The source-backed contribution is made explicit without generic praise", "contribution cards or value ladder"),
                ("assumption and limitation page", "Constraints, assumptions, or stated limitations are visible", "boundary matrix or limitation panel"),
                ("technical takeaway", "Implication, future work, or next research direction is tied back to the source", "summary table or conclusion panel"),
            ],
            "execution_plan": [
                ("execution opener", "Scope, owner lens, and outcome target are explicit", "status header + owner strip"),
                ("workstream page", "Actions are grouped into concrete workstreams", "workstream matrix"),
                ("timeline page", "Sequence and dependencies are visible", "timeline or phased roadmap"),
                ("checkpoint close", "Next actions and checkpoints are unambiguous", "action tracker"),
            ],
        }

        obligations: list[dict[str, str]] = []
        for idx, (page_role, required_signal, component) in enumerate(
            focus_map.get(planning_focus, focus_map["decision_brief"]),
            start=1,
        ):
            signal_suffix = content_hint if idx == 1 else visual_hint if idx == 2 else layout_hint
            obligations.append({
                "page_role": page_role,
                "required_signal": f"{required_signal}; {signal_suffix}",
                "preferred_component": component,
                "priority": "high" if idx <= 2 else "medium",
            })
        return obligations

    @staticmethod
    def _fallback_source_boundaries(source_evidence_summary: str) -> list[str]:
        lowered = source_evidence_summary.lower()
        if any(term in lowered for term in ("strict_no_extrapolation", "不得", "仅使用", "不能外推", "do not extrapolate")):
            return [
                "Use only source-backed facts, claims, and numbers; do not extrapolate beyond the provided material.",
                "If an important persona preference cannot be grounded by the source, translate it into framing or hierarchy rather than fabricated content.",
            ]
        return [
            "Prefer source-backed claims, numbers, and labels over generic filler.",
            "Keep any interpretation within the boundary signaled by the source evidence summary and the current instruction.",
        ]


class ProfileExecutionPlanCompiler:
    """Compile deck-level persona contract into a page-level execution plan."""

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def compile(
        self,
        *,
        contract: dict[str, Any] | None,
        instruction: str,
        source_evidence_summary: str = "",
        target_page_count: int = 0,
        template_mode: bool = False,
    ) -> dict[str, Any] | None:
        if not contract:
            return None

        normalized_count = int(target_page_count or 0)
        if normalized_count <= 0:
            normalized_count = max(6, min(10, len(contract.get("page_obligations", []) or []) + 2))

        plan: dict[str, Any] | None = None
        if self._llm:
            prompt = PROFILE_EXECUTION_PLAN_PROMPT.format(
                target_page_count=normalized_count,
                template_mode="true" if template_mode else "false",
                instruction=_compact_multiline_text(instruction, limit=1200),
                source_evidence_summary=_compact_multiline_text(source_evidence_summary or "(none)", limit=700),
                profile_execution_contract=_compact_multiline_text(
                    json.dumps(contract, ensure_ascii=False, indent=2),
                    limit=3200,
                ),
            )
            if len(prompt) <= PROFILE_EXECUTION_PLAN_LLM_PROMPT_MAX_CHARS:
                plan = await self._compile_with_llm(prompt)
            else:
                logger.info(
                    "ProfileExecutionPlanCompiler skipping LLM refinement due to prompt size (%d chars)",
                    len(prompt),
                )

        return self._normalize_plan(
            plan,
            contract=contract,
            target_page_count=normalized_count,
            template_mode=template_mode,
        )

    async def _compile_with_llm(self, prompt: str) -> dict[str, Any] | None:
        request_kwargs = {
            "max_tokens": PROFILE_EXECUTION_PLAN_MAX_TOKENS,
            "temperature": 0.0,
            "reasoning_effort": "minimal",
        }
        retry_times = _bounded_retry_times(self._llm)
        try:
            if hasattr(self._llm, "run"):
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                    retry_times=retry_times,
                    request_kwargs=request_kwargs,
                )
                text = extract_response_text(response)
            else:
                response = await self._llm(prompt)
                text = extract_response_text(response)
        except Exception as exc:
            logger.warning("ProfileExecutionPlanCompiler LLM call failed: %s", exc)
            return None
        return _parse_json_object(text)

    def _normalize_plan(
        self,
        plan: dict[str, Any] | None,
        *,
        contract: dict[str, Any],
        target_page_count: int,
        template_mode: bool,
    ) -> dict[str, Any]:
        contract_focus = str(contract.get("planning_focus") or "decision_brief").strip()
        if contract_focus not in ALLOWED_PLANNING_FOCI:
            contract_focus = "decision_brief"
        raw_plan_focus = str((plan or {}).get("planning_focus") or contract.get("planning_focus") or "decision_brief").strip()
        focus_overridden = (
            raw_plan_focus in ALLOWED_PLANNING_FOCI
            and contract_focus in ALLOWED_PLANNING_FOCI
            and raw_plan_focus != contract_focus
        )
        planning_focus = raw_plan_focus
        if planning_focus not in ALLOWED_PLANNING_FOCI:
            planning_focus = contract_focus
        elif contract_focus and planning_focus != contract_focus:
            planning_focus = contract_focus

        effective_contract = dict(contract)
        effective_contract["planning_focus"] = planning_focus
        if focus_overridden:
            effective_contract.pop("deck_spine", None)
            effective_contract.pop("page_obligations", None)

        page_plan = [] if focus_overridden else self._normalize_page_plan_items((plan or {}).get("page_plan"))
        if not page_plan:
            page_plan = self._fallback_page_plan(
                contract=effective_contract,
                target_page_count=target_page_count,
                template_mode=template_mode,
            )

        page_plan = self._expand_or_trim_page_plan(
            page_plan,
            contract=effective_contract,
            target_page_count=target_page_count,
            template_mode=template_mode,
        )

        global_execution_notes = _coerce_string_list((plan or {}).get("global_execution_notes"), limit=8)
        if not global_execution_notes:
            global_execution_notes = self._fallback_global_execution_notes(effective_contract, template_mode=template_mode)

        fallback_rules = _coerce_string_list((plan or {}).get("fallback_rules"), limit=8)
        if not fallback_rules:
            fallback_rules = _coerce_string_list(effective_contract.get("proxy_realization_rules"), limit=6)
        if template_mode:
            fallback_rules = _dedupe_keep_order(
                fallback_rules
                + [
                    "When a persona-preferred component does not fit the selected template layout, preserve the persona signal through slot emphasis, framing, and sequencing instead of replacing the template layout outright.",
                ],
                limit=8,
            )

        return {
            "planning_focus": planning_focus,
            "global_execution_notes": global_execution_notes,
            "fallback_rules": fallback_rules,
            "page_plan": page_plan,
        }

    @staticmethod
    def _normalize_page_plan_items(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                page_index = int(item.get("page_index", 0) or 0)
            except Exception:
                page_index = 0
            page_role = _normalize_space(item.get("page_role", ""))
            persona_signal = _normalize_space(item.get("persona_signal", ""))
            manuscript_anchor = _normalize_space(item.get("manuscript_anchor", ""))
            required_component = _normalize_space(item.get("required_component", ""))
            layout_bias = _normalize_space(item.get("layout_bias", "")).lower()
            if layout_bias not in ALLOWED_LAYOUT_BIASES:
                layout_bias = "single_focus"
            if not page_role and not persona_signal:
                continue
            must_preserve = _coerce_string_list(item.get("must_preserve"), limit=5)
            nice_to_have = _coerce_string_list(item.get("nice_to_have"), limit=4)
            page_texts = [
                page_role,
                persona_signal,
                manuscript_anchor,
                required_component,
                *must_preserve,
                *nice_to_have,
            ]
            hard_requirements = _coerce_string_list(item.get("hard_requirements"), limit=5)
            component_requirements = _derive_component_requirements(
                item.get("component_requirements"),
                required_component=required_component,
                preference_texts=page_texts,
                limit=5,
            )
            soft_signals = _derive_soft_signals(
                item.get("soft_signals"),
                texts=page_texts,
                exclude=hard_requirements + component_requirements,
                limit=6,
            )
            normalized.append({
                "page_index": page_index,
                "page_role": page_role or "persona-specific page",
                "persona_signal": persona_signal or "persona-aligned page signal",
                "manuscript_anchor": manuscript_anchor or "fit-for-source content cluster",
                "required_component": required_component or "fit-for-content archetype",
                "layout_bias": layout_bias,
                "hard_requirements": hard_requirements,
                "style_requirements": _coerce_style_requirements(item.get("style_requirements"), *page_texts),
                "component_requirements": component_requirements,
                "soft_signals": soft_signals,
                "source_boundary_notes": _coerce_string_list(item.get("source_boundary_notes"), limit=4),
                "must_preserve": must_preserve,
                "nice_to_have": nice_to_have,
            })
        normalized.sort(key=lambda item: (int(item.get("page_index", 0) or 0), item.get("page_role", "")))
        return normalized[:32]

    @staticmethod
    def _layout_bias_for_component(component: str) -> str:
        text = str(component or "").lower()
        if any(term in text for term in ("decision", "recommend", "action", "next-step")):
            return "decision"
        if any(term in text for term in ("matrix", "comparison", "compare", "table", "rank")):
            return "comparison"
        if any(term in text for term in ("process", "pipeline", "step", "walkthrough", "timeline", "roadmap")):
            return "process"
        if any(term in text for term in ("evidence", "proof", "chart", "figure", "experiment")):
            return "evidence"
        if any(term in text for term in ("recap", "summary", "takeaway")):
            return "recap"
        return "single_focus"

    @classmethod
    def _focus_role_palette(cls, planning_focus: str) -> list[dict[str, str]]:
        def _item(
            page_role: str,
            persona_signal: str,
            required_component: str,
            manuscript_anchor: str,
        ) -> dict[str, str]:
            return {
                "page_role": page_role,
                "persona_signal": persona_signal,
                "required_component": required_component,
                "manuscript_anchor": manuscript_anchor,
                "layout_bias": cls._layout_bias_for_component(required_component),
            }

        palettes: dict[str, list[dict[str, str]]] = {
            "decision_brief": [
                _item(
                    "decision opener",
                    "Immediate decision framing and evaluation lens",
                    "headline + KPI chips",
                    "opening frame and scope",
                ),
                _item(
                    "contribution framing",
                    "The main contribution is translated into a value proposition the audience can judge quickly",
                    "contribution cards or value ladder",
                    "paper contributions and problem framing",
                ),
                _item(
                    "mechanism leverage page",
                    "The core approach is connected to why it matters for the decision",
                    "mechanism diagram + implication callouts",
                    "core method and architecture",
                ),
                _item(
                    "evidence comparison",
                    "Comparable options, drivers, or trade-offs are made legible",
                    "comparison table or ranked comparison",
                    "core source evidence cluster",
                ),
                _item(
                    "benchmark proof page",
                    "The strongest empirical proof is surfaced with clear evidence hierarchy",
                    "benchmark table or proof chart",
                    "experimental results and benchmarks",
                ),
                _item(
                    "risk and boundary page",
                    "Risks, assumptions, and scope boundaries are explicit",
                    "risk table or callout matrix",
                    "limitations and boundary conditions",
                ),
                _item(
                    "implication page",
                    "The practical implication or next-step research value is explicit",
                    "implication grid + next-step bullets",
                    "implications and future work",
                ),
                _item(
                    "recommendation close",
                    "A concise recommendation or action recommendation",
                    "summary box + next-step bullets",
                    "final synthesis and next-step framing",
                ),
            ],
            "visual_system_first": [
                _item("visual opener", "The visual direction is established immediately", "hero composition", "opening frame and scope"),
                _item("showcase page", "One dominant visual story is allowed to lead", "large image or visual collage", "hero evidence or image-led source cluster"),
                _item("system detail page", "The visual system becomes legible through modular structure", "annotated layout or modular grid", "system detail and supporting source cluster"),
                _item("contrast page", "Contrast and progression make the sequence feel intentional", "before-after or contrast layout", "comparative source cluster"),
                _item("proof page", "Aesthetic direction is still grounded in evidence or proof", "evidence card wall or visual proof strip", "evidence and proof points"),
                _item("visual synthesis close", "The final page resolves into a distilled source-backed visual takeaway", "synthesis panel + key proof", "closing summary and takeaway"),
            ],
            "teaching_walkthrough": [
                _item("learning opener", "The learning target and path are explicit", "goal block + agenda", "opening frame and scope"),
                _item("definition page", "Key terms are grounded before complexity increases", "definition callout + key idea card", "core definitions and setup"),
                _item("concept build page", "One concept is unpacked step by step", "step diagram or concept card sequence", "core concept cluster"),
                _item("mechanism walkthrough", "The mechanism is explained in a progressive order", "process diagram or numbered sequence", "mechanism or process explanation"),
                _item("worked example page", "A concrete example bridges concept to understanding", "example walkthrough", "example, figure, or derivation cluster"),
                _item("boundary page", "Common confusions, limits, or scope notes are surfaced", "misconception callouts or boundary table", "limitations and caveats"),
                _item("recap page", "Key takeaways are easy to retain and repeat", "recap checklist or summary ladder", "final synthesis and recap"),
            ],
            "technical_walkthrough": [
                _item("problem framing", "The technical problem and evaluation lens are explicit", "architecture/problem frame", "opening frame and scope"),
                _item("source structure overview", "The source-derived architecture, object, or system boundary is visible", "system block diagram", "system overview and high-level architecture"),
                _item("mechanism decomposition", "A core mechanism is broken into understandable units", "pipeline diagram", "core mechanism and data flow"),
                _item("formula and term explanation", "Important formulas, variables, or terms get dedicated explanatory space", "formula callout + variable map", "formulas, definitions, or key terms"),
                _item("data flow page", "The flow of information or computation is explicit", "data-flow or sequence diagram", "operational data flow or processing sequence"),
                _item("evaluation evidence", "Evidence is tied to evaluation targets or benchmarks", "benchmark table or result chart", "experimental evaluation cluster"),
                _item("assumption and limitation page", "Constraints, assumptions, or stated limitations are visible", "boundary matrix or limitation panel", "constraints, limitations, and future work"),
                _item("technical takeaway", "Implication, future work, or next research direction is tied back to the source", "summary table or conclusion panel", "final synthesis and engineering takeaway"),
            ],
            "execution_plan": [
                _item("execution opener", "Scope, owner lens, and outcome target are explicit", "status header + owner strip", "opening frame and scope"),
                _item("workstream page", "Actions are grouped into concrete workstreams", "workstream matrix", "core action clusters"),
                _item("dependency page", "Dependencies and coordination needs are visible", "dependency map or swimlane", "dependencies and handoffs"),
                _item("timeline page", "Sequence and pacing are explicit", "timeline or phased roadmap", "timeline and sequencing"),
                _item("risk page", "Operational blockers and mitigation plans are explicit", "risk register or blocker matrix", "risks and blockers"),
                _item("checkpoint close", "Next actions and checkpoints are unambiguous", "action tracker", "final synthesis and next-step framing"),
            ],
        }
        return palettes.get(planning_focus, palettes["decision_brief"])

    @staticmethod
    def _obligation_to_page_spec(
        obligation: dict[str, Any] | None,
        *,
        default_anchor: str,
    ) -> dict[str, str]:
        item = obligation or {}
        required_component = _normalize_space(item.get("preferred_component", ""))
        return {
            "page_role": _normalize_space(item.get("page_role", "")),
            "persona_signal": _normalize_space(item.get("required_signal", "")),
            "required_component": required_component,
            "manuscript_anchor": default_anchor,
            "layout_bias": ProfileExecutionPlanCompiler._layout_bias_for_component(required_component),
        }

    @staticmethod
    def _merge_role_specs(
        primary: list[dict[str, str]],
        secondary: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        for collection in (primary, secondary):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                role = _normalize_space(item.get("page_role", "")).lower()
                component = _normalize_space(item.get("required_component", "")).lower()
                key = f"{role}|{component}"
                if not role or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _fallback_page_plan(
        self,
        *,
        contract: dict[str, Any],
        target_page_count: int,
        template_mode: bool,
    ) -> list[dict[str, Any]]:
        obligations = contract.get("page_obligations", []) or []
        planning_focus = str(contract.get("planning_focus") or "decision_brief")
        deck_spine = _coerce_string_list(contract.get("deck_spine"), limit=6)
        style_contract = _coerce_string_list(contract.get("style_contract"), limit=4)
        content_contract = _coerce_string_list(contract.get("content_contract"), limit=4)
        source_boundary_contract = _coerce_string_list(contract.get("source_boundary_contract"), limit=3)
        palette = self._focus_role_palette(planning_focus)

        fallback: list[dict[str, Any]] = []
        opener = self._obligation_to_page_spec(
            obligations[0] if obligations else (palette[0] if palette else {}),
            default_anchor="opening frame and scope",
        )
        closer = self._obligation_to_page_spec(
            obligations[-1] if obligations else (palette[-1] if palette else {}),
            default_anchor="final synthesis and next-step framing",
        )
        raw_middle_candidates = [
            self._obligation_to_page_spec(item, default_anchor="core source evidence cluster")
            for item in (
                obligations[1:-1]
                if len(obligations) > 2
                else obligations[1:] if len(obligations) > 1 else obligations[:1]
            )
        ]
        palette_middle = palette[1:-1] if len(palette) > 2 else palette
        middle_candidates = self._merge_role_specs(raw_middle_candidates, palette_middle)
        if not middle_candidates:
            middle_candidates = [opener or closer or {}]
        for idx in range(1, target_page_count + 1):
            if idx == 1:
                obligation = opener
            elif idx == target_page_count:
                obligation = closer
            else:
                obligation = middle_candidates[(idx - 2) % len(middle_candidates)] if middle_candidates else {}
            page_role = _normalize_space(obligation.get("page_role", "")) or (
                "persona opener" if idx == 1 else "persona close" if idx == target_page_count else f"persona page {idx}"
            )
            if idx not in {1, target_page_count} and middle_candidates:
                cycle_index = ((idx - 2) // max(1, len(middle_candidates))) + 1
                if cycle_index > 1:
                    page_role = f"{page_role} extension {cycle_index}"
            required_component = _normalize_space(obligation.get("required_component", "")) or (
                "headline + evidence panel" if idx == 1 else "summary panel" if idx == target_page_count else "fit-for-content archetype"
            )
            persona_signal = _normalize_space(obligation.get("persona_signal", "")) or (
                deck_spine[min(idx - 1, len(deck_spine) - 1)] if deck_spine else "persona-aligned page signal"
            )
            anchor = _normalize_space(obligation.get("manuscript_anchor", "")) or (
                "opening frame and scope"
                if idx == 1
                else "final synthesis and next-step framing"
                if idx == target_page_count
                else "core source evidence cluster"
            )
            must_preserve = []
            if idx == 1 and content_contract:
                must_preserve.append(content_contract[0])
            if idx == 2 and style_contract:
                must_preserve.append(style_contract[0])
            if "risk" in page_role.lower() and source_boundary_contract:
                must_preserve.append(source_boundary_contract[0])
            if any(token in page_role.lower() for token in ("evidence", "benchmark", "proof")) and style_contract:
                must_preserve.append(style_contract[min(1, len(style_contract) - 1)])
            if template_mode:
                must_preserve.append("Preserve the selected template layout and carry the persona signal through emphasis, grouping, and slot usage.")
            page_texts = [
                page_role,
                persona_signal,
                anchor,
                required_component,
                *must_preserve,
            ]
            hard_requirements = (
                ["Preserve selected template layout authority from layout_mapping.yaml."]
                if template_mode
                else []
            )
            component_requirements = _derive_component_requirements(
                [],
                required_component=required_component,
                preference_texts=page_texts,
                limit=5,
            )
            fallback.append({
                "page_index": idx,
                "page_role": page_role,
                "persona_signal": persona_signal,
                "manuscript_anchor": anchor,
                "required_component": required_component,
                "layout_bias": _normalize_space(obligation.get("layout_bias", "")) or self._layout_bias_for_component(required_component),
                "hard_requirements": hard_requirements,
                "style_requirements": _coerce_style_requirements({}, *page_texts),
                "component_requirements": component_requirements,
                "soft_signals": _derive_soft_signals(
                    [],
                    texts=page_texts,
                    exclude=hard_requirements + component_requirements,
                    limit=6,
                ),
                "source_boundary_notes": source_boundary_contract,
                "must_preserve": _dedupe_keep_order(must_preserve, limit=4),
                "nice_to_have": [],
            })
        return fallback

    def _expand_or_trim_page_plan(
        self,
        page_plan: list[dict[str, Any]],
        *,
        contract: dict[str, Any],
        target_page_count: int,
        template_mode: bool,
    ) -> list[dict[str, Any]]:
        normalized = list(page_plan[:target_page_count])
        if len(normalized) < target_page_count:
            fallback = self._fallback_page_plan(
                contract=contract,
                target_page_count=target_page_count,
                template_mode=template_mode,
            )
            existing = {int(item.get("page_index", 0) or 0) for item in normalized}
            for item in fallback:
                page_index = int(item.get("page_index", 0) or 0)
                if page_index in existing:
                    continue
                normalized.append(item)
                existing.add(page_index)
                if len(normalized) >= target_page_count:
                    break

        normalized.sort(key=lambda item: int(item.get("page_index", 0) or 0))
        reindexed: list[dict[str, Any]] = []
        for idx, item in enumerate(normalized[:target_page_count], start=1):
            copied = dict(item)
            copied["page_index"] = idx
            reindexed.append(copied)
        return reindexed

    @staticmethod
    def _fallback_global_execution_notes(contract: dict[str, Any], *, template_mode: bool) -> list[str]:
        notes = []
        notes.extend(_coerce_string_list(contract.get("deck_spine"), limit=3))
        notes.extend(_coerce_string_list(contract.get("content_contract"), limit=2))
        notes.extend(_coerce_string_list(contract.get("source_boundary_contract"), limit=2))
        if template_mode:
            notes.append("Template mode: keep the selected layout family stable; realize persona differences through ordering, framing, slot emphasis, and copy hierarchy.")
        else:
            notes.append("Non-template mode: use the persona page plan to make page roles explicit before writing any slide HTML.")
        return _dedupe_keep_order(notes, limit=8)
