from __future__ import annotations

import logging
import json
import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memslides.pipelines.generation_support import (
    find_existing_design_plan_rel as _find_existing_design_plan_rel,
    normalize_bool_flag as _normalize_bool_flag,
    normalize_memory_intent as _normalize_memory_intent,
    normalize_text_flag as _normalize_text_flag,
)
from memslides.templates.runtime_state import (
    TemplateRuntimeState,
    canonical_layout_names_from_profile,
    clear_template_runtime_state,
    save_template_runtime_state,
)
from memslides.templates.quality import (
    ADAPTIVE_STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    STRUCTURAL_TEMPLATE,
    assess_template_quality,
    write_template_quality_report,
)
from bs4 import BeautifulSoup
from memslides.utils.typings import InputRequest
from memslides.utils.webview import convert_html_to_pptx_with_retry

logger = logging.getLogger(__name__)

LIVE_WM_SNAPSHOT_FILENAME = "live_wm_snapshot.json"
PROFILE_SEED_SOURCE = "ltm_profile_inject"
PROFILE_OVERRIDE_SOURCE = "profile_override"
STRUCTURED_RULE_CARRYOVER_SOURCES = {
    "saved_structured_rule_carryover",
    "live_snapshot_structured_rule_carryover",
}
WM_PROFILE_WRITABLE_SOURCES = {
    PROFILE_OVERRIDE_SOURCE,
    "manual_wm_edit",
}


@dataclass
class ModifyExecutionPlan:
    scope: str
    reason: str
    target_slide_paths: list[Path]
    target_rule_ids: list[str]
    selector_hints: list[str]
    coverage_required: bool = False
    operation_kind: str = "style"
    applicable_rule_ids: list[str] | None = None
    new_element_rule_applications: list[dict[str, Any]] | None = None
    diagram_contract: dict[str, Any] | None = None
    rewrite_decision: dict[str, Any] | None = None
    rewrite_decision_source: str = ""


@dataclass
class ModifyToolPolicyPlan:
    scope: str
    operation_kind: str
    tool_groups: list[str]
    target_slide_paths: list[Path]
    expected_slide_delta: int = 0
    coverage_required: bool = False
    first_steps: list[str] | None = None
    risk_flags: list[str] | None = None
    requested_tools: list[str] | None = None
    reason: str = ""
    source: str = "fallback"
    raw_response: str = ""
    prompt_artifact: str = ""
    response_artifact: str = ""
    payload_artifact: str = ""
    new_element_rule_applications: list[dict[str, Any]] | None = None
    expected_slide_delta_source: str = "fallback"
    rewrite_decision: dict[str, Any] | None = None
    rewrite_decision_source: str = ""


@dataclass
class ModifyToolScopeContract:
    scope: str
    operation_kind: str
    allowed_tools: list[str]
    removed_tools: list[str]
    reason: str
    tool_groups: list[str] | None = None
    policy_source: str = ""
    system_added_tools: list[str] | None = None
    hard_removed_tools: list[str] | None = None


@dataclass
class PreferenceSemanticCompilation:
    payload: dict[str, Any]
    source: str
    memory_flow: str = "none"
    valid: bool = False
    fail_closed: bool = False
    reason: str = ""
    payloads: list[dict[str, Any]] | None = None
    compiler_prompt_artifact: str = ""
    compiler_response_artifact: str = ""
    compiler_payload_artifact: str = ""
    critic_prompt_artifact: str = ""
    critic_response_artifact: str = ""
    critic_payload_artifact: str = ""
    trace_artifact: str = ""


def safe_memory_json(value: Any) -> Any:
    """Return a JSON-safe representation for durable memory snapshots."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_memory_json(item) for item in value]
    if isinstance(value, set):
        return [safe_memory_json(item) for item in sorted(value, key=str)]
    if isinstance(value, dict):
        return {str(key): safe_memory_json(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        try:
            return safe_memory_json(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return safe_memory_json(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): safe_memory_json(item)
                for key, item in value.__dict__.items()
                if not str(key).startswith("_")
            }
        except Exception:
            pass
    return str(value)


def _wm_attr(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _wm_source(value: Any) -> str:
    return str(_wm_attr(value, "source_task_id", "") or "").strip()


def is_profile_seed_preference(value: Any) -> bool:
    return _wm_source(value) == PROFILE_SEED_SOURCE


def is_structured_rule_carryover_preference(value: Any) -> bool:
    return _wm_source(value) in STRUCTURED_RULE_CARRYOVER_SOURCES


def is_profile_writable_preference(value: Any) -> bool:
    if bool(_wm_attr(value, "superseded", False)):
        return False
    source = _wm_source(value)
    if source == PROFILE_SEED_SOURCE or source in STRUCTURED_RULE_CARRYOVER_SOURCES:
        return False
    return True


def _experience_source(value: Any) -> str:
    return str(_wm_attr(value, "source", "") or "").strip()


def classify_working_memory_for_save(wm: Any | None) -> dict[str, Any]:
    """Summarize WM content by writeback destination.

    Profile seeds and restored structured rules are useful for the current
    session, but they should not be counted as new profile memory.
    """
    if wm is None:
        return {
            "profile_seed_preferences": 0,
            "carryover_preferences": 0,
            "profile_writable_preferences": 0,
            "other_preferences": 0,
            "total_preferences": 0,
            "round_experiences_total": 0,
            "round_experience_preloads": 0,
            "round_experience_candidates": 0,
            "tool_chain_groups": 0,
            "tool_chain_segments": 0,
            "tool_chain_experience_groups": 0,
            "tool_chain_experiences": 0,
            "episodes": 0,
        }

    preferences = list(getattr(wm, "_temp_preferences", []) or [])
    experiences = (
        list(wm.get_experiences())
        if hasattr(wm, "get_experiences")
        else list(getattr(wm, "_temp_experiences", []) or [])
    )
    episodes = (
        list(wm.get_episodes())
        if hasattr(wm, "get_episodes")
        else list(getattr(wm, "_temp_episodes", []) or [])
    )
    chain_buffer = getattr(wm, "chain_buffer", None)
    chains: dict[str, list[Any]] = {}
    chain_exps: dict[str, list[Any]] = {}
    if chain_buffer is not None:
        try:
            chains = chain_buffer.get_new_chains_for_consolidation()
        except Exception:
            try:
                chains = chain_buffer.get_all_chains()
            except Exception:
                chains = {}
        try:
            chain_exps = chain_buffer.get_all_experiences()
        except Exception:
            chain_exps = {}

    seed_count = sum(1 for pref in preferences if is_profile_seed_preference(pref))
    carryover_count = sum(1 for pref in preferences if is_structured_rule_carryover_preference(pref))
    writable_count = sum(1 for pref in preferences if is_profile_writable_preference(pref))
    return {
        "profile_seed_preferences": seed_count,
        "carryover_preferences": carryover_count,
        "profile_writable_preferences": writable_count,
        "other_preferences": max(0, len(preferences) - seed_count - carryover_count - writable_count),
        "total_preferences": len(preferences),
        "round_experiences_total": len(experiences),
        "round_experience_preloads": sum(1 for exp in experiences if _experience_source(exp) == "preload"),
        "round_experience_candidates": sum(1 for exp in experiences if _experience_source(exp) != "preload"),
        "tool_chain_groups": len(chains),
        "tool_chain_segments": sum(len(items) for items in chains.values()),
        "tool_chain_experience_groups": len(chain_exps),
        "tool_chain_experiences": sum(len(items) for items in chain_exps.values()),
        "episodes": len(episodes),
    }


def working_memory_has_only_restore_seed_content(wm: Any | None) -> bool:
    """True when WM contains only profile seed/carryover/preload content."""
    if wm is None:
        return False
    preferences = list(getattr(wm, "_temp_preferences", []) or [])
    experiences = (
        list(wm.get_experiences())
        if hasattr(wm, "get_experiences")
        else list(getattr(wm, "_temp_experiences", []) or [])
    )
    episodes = list(getattr(wm, "_temp_episodes", []) or [])
    if episodes:
        return False
    if not preferences and not experiences:
        return False
    for pref in preferences:
        if not (is_profile_seed_preference(pref) or is_structured_rule_carryover_preference(pref)):
            return False
    return all(_experience_source(exp) == "preload" for exp in experiences)


def clear_restore_seed_content(wm: Any | None) -> dict[str, int]:
    """Remove seed/preload content that would block snapshot hydration."""
    if wm is None:
        return {"preferences_removed": 0, "experiences_removed": 0, "episodes_removed": 0}
    preferences = list(getattr(wm, "_temp_preferences", []) or [])
    experiences = (
        list(wm.get_experiences())
        if hasattr(wm, "get_experiences")
        else list(getattr(wm, "_temp_experiences", []) or [])
    )
    episodes = list(getattr(wm, "_temp_episodes", []) or [])
    kept_preferences = [
        pref
        for pref in preferences
        if not (is_profile_seed_preference(pref) or is_structured_rule_carryover_preference(pref))
    ]
    kept_experiences = [exp for exp in experiences if _experience_source(exp) != "preload"]
    try:
        wm._temp_preferences = kept_preferences
    except Exception:
        pass
    try:
        wm._temp_experiences = kept_experiences
    except Exception:
        pass
    if episodes:
        try:
            wm._temp_episodes = []
        except Exception:
            pass
    return {
        "preferences_removed": len(preferences) - len(kept_preferences),
        "experiences_removed": len(experiences) - len(kept_experiences),
        "episodes_removed": len(episodes),
    }


_MODIFY_SCOPE_SHARED_TOOLS = [
    "list_files",
    "read_file",
    "inspect_slide",
    "finalize",
    "thinking",
    "remember_lesson",
]

_MODIFY_TOOL_GROUPS = {
    "local_patch": [
        "plan_slide_patch",
        "read_slide_snapshot",
        "apply_slide_patch",
    ],
    "global_style": [
        "scan_slide_index",
        "batch_update_css_rule",
        "batch_update_semantic_style",
        "patch_semantic_inline_style",
    ],
    "structural": [
        "scan_slide_index",
        "insert_slide",
        "delete_slide",
        "write_new_slide_file",
    ],
    "post_write_repair": [
        "plan_slide_patch",
        "read_slide_snapshot",
        "apply_slide_patch",
    ],
    "asset_discovery": [
        "list_document_figures",
        "explore_workspace_images",
        "image_caption",
    ],
    "structured_visual": [
        "render_chart_asset",
        "render_table_asset",
        "render_flowchart_asset",
    ],
    "diagram_rewrite": [
        "read_slide_snapshot",
        "write_html_file",
        "render_flowchart_asset",
    ],
    "controlled_rewrite": [
        "read_slide_snapshot",
        "write_html_file",
    ],
    "external_acquisition": [
        "search_web",
        "fetch_url",
        "search_images",
        "download_file",
        "image_generation",
    ],
    "content_research": [
        "search_web",
        "fetch_url",
        "document_summary",
    ],
    "memory_lookup": [
        "search_experiences",
        "search_episodes",
        "remember_lesson",
    ],
    "progress": [
        "todo_create",
        "todo_update",
        "todo_list",
    ],
}

_MODIFY_POLICY_VALID_SCOPES = {"local", "global"}
_MODIFY_POLICY_VALID_OPERATION_KINDS = {
    "style",
    "content",
    "layout",
    "diagram_layout",
    "image_asset",
    "structural",
    "research_content",
    "mixed",
    "preference_update",
    "query",
    "export_repair",
    "corrupted_recovery",
    "controlled_rewrite",
}
_MODIFY_POLICY_VALID_TOOL_GROUPS = set(_MODIFY_TOOL_GROUPS)
_MODIFY_CAPABILITY_REQUIRED_TOOLS = {
    "local_content_edit": ["read_slide_snapshot", "apply_slide_patch", "inspect_slide"],
    "local_style_edit": ["plan_slide_patch", "read_slide_snapshot", "apply_slide_patch", "inspect_slide"],
    "local_layout_repair": ["plan_slide_patch", "read_slide_snapshot", "apply_slide_patch", "inspect_slide"],
    "deck_style_batch": ["scan_slide_index", "batch_update_css_rule", "batch_update_semantic_style", "patch_semantic_inline_style", "inspect_slide"],
    "slide_structure": ["scan_slide_index", "insert_slide", "write_new_slide_file", "delete_slide", "inspect_slide"],
    "asset_update": ["list_document_figures", "explore_workspace_images", "image_caption", "apply_slide_patch", "inspect_slide"],
    "chart_or_table_update": ["render_chart_asset", "render_table_asset", "render_flowchart_asset", "apply_slide_patch", "inspect_slide"],
    "diagram_layout": ["read_slide_snapshot", "write_html_file", "render_flowchart_asset", "inspect_slide"],
    "controlled_rewrite": ["read_slide_snapshot", "write_html_file", "inspect_slide"],
    "template_alignment": ["read_slide_snapshot", "apply_slide_patch", "inspect_slide"],
    "export_repair": ["plan_slide_patch", "read_slide_snapshot", "apply_slide_patch", "inspect_slide"],
    "preference_update": ["finalize", "remember_lesson"],
    "query_only": ["list_files", "read_file", "inspect_slide"],
}
_MODIFY_POLICY_DEFAULT_GROUPS_BY_OPERATION = {
    "style": ["local_patch"],
    "content": ["local_patch"],
    "layout": ["local_patch"],
    "diagram_layout": ["diagram_rewrite", "structured_visual", "local_patch"],
    "image_asset": ["asset_discovery", "local_patch"],
    "structural": ["structural", "local_patch"],
    "research_content": ["content_research", "local_patch"],
    "mixed": ["local_patch", "global_style"],
    "preference_update": [],
    "query": [],
    "export_repair": ["local_patch"],
    "corrupted_recovery": ["local_patch"],
    "controlled_rewrite": ["controlled_rewrite", "local_patch"],
}
_MODIFY_POLICY_HARD_REMOVE_COMMON = {
    "convert_to_markdown",
    "inspect_manuscript",
    "search_papers",
    "get_paper_authors",
    "get_scholar_details",
    "write_markdown_file",
    "list_memory_artifacts",
    "list_template_layouts",
    "recommend_template_layout",
    "query_slide_layout",
    "query_layout_geometry",
    "query_image_info",
    "query_template_shell",
}
_MODIFY_POLICY_FULL_WRITE_TOOL = "write_html_file"
PREFERENCE_UPDATE_MUTATION_TOOLS = {
    "apply_slide_patch",
    "write_html_file",
    "write_new_slide_file",
    "insert_slide",
    "delete_slide",
    "batch_update_css_rule",
    "batch_update_semantic_style",
    "patch_semantic_inline_style",
}

_PREFERENCE_SEMANTIC_COMPILER_PROMPT = """你是 MemSlides 的 working-memory preference semantic compiler。你的任务是根据用户本轮 feedback 做语义编译，不要靠关键词规则机械判断。

你必须把请求拆成四类语义：
1. current_actions: 本轮要立即执行的当前动作，例如只改首页、只改第10页、插入新页；没有则为空。
2. memory_updates: 本轮是否要把某个偏好/约束写入 working memory；没有则为空。
3. propagation_decision: 如果写入记忆，判断它是否作用于现有已生成 slides、未来新增/新生成 slides、或未来新建元素（例如已有 slide 中新增 pill tag/badge/title），或这些范围的组合。
4. evidence_spans: 每个关键判断都必须给出来自用户原文的短片段证据。没有证据就不要声称该传播语义成立。

重要约束：
- 你是语义判断器，不要把 intent classifier 的 target_slide=all 当成传播语义。
- 只有当用户语义上明确要求修改现有/已有/当前/整套旧页时，apply_existing_slides 才能为 true，并且必须提供 existing_evidence_spans。
- 只有当用户语义上要求后续/未来/新增/新生成/新插入页面继承时，apply_future_slides 才能为 true，并且必须提供 future_evidence_spans。
- 只有当用户语义上要求后续新建/新增的元素继承时，apply_future_elements 才能为 true，并且必须提供 future_element_evidence_spans 和 creation_scope。creation_scope 可包含 any_text、slide_title、pill_tag、body_text、caption、table_cell、chart_data_label、legend_label、callout 等。
- 当用户要求“以后所有表示某类语义的文本/数值/指标”继承样式时，不要局限到 body_text/caption；使用 target.semantic_target 表达语义对象，例如 {{"id":"experimental_metric_value_text","label":"实验结果/指标数值文本","positive_examples":["FFAcc 达到 37.6"],"negative_examples":["第 10 页"]}}，并使用 creation_scope=["any_text"]。
- 对开放语义目标必须尽量保留 general_sentence；如果分类表没有合适枚举，可以创建 user_defined:* 的 semantic_target.id，但 action 仍要保持机器可执行。
- action.css_values 是 finalize 前确定性硬校验的唯一 CSS 来源。只有当用户给出明确 CSS 值或明确 CSS 状态时才填写，例如 #065F46、rgb(6,95,70)、background-color: #DBEAFE、font-style: italic、font-weight: 700/bold、text-decoration: underline。
- 如果用户只说“深绿色/蓝色/白色/学术/低饱和/更柔和”等自然语言偏好，不要发明永久 CSS 值；保留在 action.description/general_sentence，必要时放入 action.design_tokens 作为软意图，不能写入 action.css_values。
- 对内容、布局、表达口吻、图片处理等模糊偏好，优先结构化 rule_type/target/condition/description，不要把它们误写成硬 CSS。
- 如果用户只是单页当前修改，不写记忆，memory_updates=[]。
- 如果用户说“try / this time / demo only / just for this demo / 试试 / 这次 / 本次 / 临时先这样”等，只能写成 session_dimension_note：它是本 session 的普通维度文字记忆，不生成 structured_rule，不进入 Temporary prefs。
- 只有“future / next rounds / later / if we add more / 后续 / 以后 / 未来 / 新增 / 新生成 / 如果后面再加”等明确未来传播证据，才允许生成 structured_rule。
- 一次性结构操作如 “add a new slide after slide 4 / 在第4页后新增一页 / 删除第3页” 是 current_edit_only；除非用户同时明确说后续也要继承某偏好，否则不写 memory_updates。
- 如果语义不确定，memory_flow 设为 ambiguous_fail_closed，memory_updates=[]。
- 返回 JSON only。

当前 deck slides:
{deck_status_json}

Intent classifier result:
{intent_json}

Existing WM structured rules:
{existing_rules}

User feedback:
{user_message}

返回格式：
{{
  "memory_flow": "none|memory_only|session_dimension_note|current_edit_plus_future_rule|existing_batch_plus_future_rule|structural_insert_with_future_rule|current_edit_only|ambiguous_fail_closed",
  "current_actions": [
    {{
      "action_type": "edit_existing|insert_slide|delete_slide|none",
      "target_slide": "slide_01|slide_10|all|",
      "target_element": "slide_title|body_text|footer|image|pill_tag|caption|figure_caption|table_caption|deck|",
      "description": "本轮当前动作",
      "evidence_spans": ["用户原文片段"]
    }}
  ],
  "memory_updates": [
    {{
      "preference": "一句偏好描述",
      "category": "style|color|layout|typography|content|general",
      "dimension": "theme.primary_colors|typography.text_color|theme.font_family|layout.slide_structure|content.language_style|general|",
      "general_sentence": "写入 WM 的规则句",
      "retention_scope": "job_local|intent_profile|default_profile",
      "evidence_spans": ["用户原文片段"],
      "structured_rule": {{
        "schema_version": "wm_rule_v1",
        "rule_type": "style|layout|content|constraint",
        "scope": "global",
        "target": {{"slide_scope": "all|future|existing", "element_kind": "any_text|slide_title|body_text|footer|image|pill_tag|caption|table_cell|chart_data_label|legend_label|callout|deck", "semantic_target": {{"id": "experimental_metric_value_text|user_defined:*", "label": "语义对象名称", "description": "如何判断该对象", "positive_examples": ["正例"], "negative_examples": ["反例"]}}}},
        "condition": {{"match_granularity": "text_span|element", "apply_to": "semantic_span_only|whole_element"}},
        "action": {{"op": "apply_preference", "description": "……", "css_properties": ["color"], "css_values": {{"color": "#065F46"}}, "design_tokens": {{"color.intent": "deep_green"}}}},
        "propagation": {{"apply_existing_slides": false, "apply_future_slides": true, "apply_future_elements": true, "creation_scope": ["slide_title"]}},
        "verification": {{"hard_check": "css_values_only"}},
        "dimension": "typography.text_color"
      }}
    }}
  ],
  "propagation_decision": {{
    "apply_existing_slides": false,
    "apply_future_slides": true,
    "apply_future_elements": true,
    "creation_scope": ["slide_title"],
    "existing_evidence_spans": [],
    "future_evidence_spans": ["用户原文片段"],
    "future_element_evidence_spans": ["用户原文片段"],
    "reason": "一句话解释"
  }},
  "confidence": 0.0,
  "ambiguities": []
}}"""

_PREFERENCE_SEMANTIC_CRITIC_PROMPT = """你是 MemSlides working-memory semantic critic。请检查 compiler 是否把用户 feedback 的传播语义判断错了，尤其要防止把 future preference 误解释成 existing deck-wide batch 修改。

你只做校验，不做工具计划。请回答：
1. should_modify_existing_slides: 用户是否明确要求修改已经生成的旧页？
2. should_only_write_memory: 本轮是否只是写入记忆，不应修改任何旧页？
3. should_apply_future_slides: 用户是否要求未来新增/新生成 slides 应用该偏好？
4. should_apply_future_elements: 用户是否要求未来新建元素（包括已有 slide 中新增 tag/badge/title 等）应用该偏好？

如果 compiler 的 propagation_decision 与你的判断冲突，approved=false，并说明 issues。
如果唯一问题是未来新建元素的 creation_scope 太窄，但用户的未来语义证据清楚，请在 corrected_propagation_decision 中给出更安全的宽 scope，例如 creation_scope=["any_text"] 或完整文本元素集合；系统会验证后采纳该修正。
只有当是否修改旧页/是否应用未来本身存在冲突时，才让 approved=false 表示不可采纳的失败。
返回 JSON only。

User feedback:
{user_message}

Compiler output:
{compiler_json}

返回格式：
{{
  "approved": true,
  "should_modify_existing_slides": false,
  "should_only_write_memory": true,
  "should_apply_future_slides": true,
  "should_apply_future_elements": true,
  "issues": [],
  "corrected_propagation_decision": {{
    "apply_existing_slides": false,
    "apply_future_slides": true,
    "apply_future_elements": true,
    "creation_scope": ["slide_title"],
    "existing_evidence_spans": [],
    "future_evidence_spans": ["用户原文片段"],
    "future_element_evidence_spans": ["用户原文片段"]
  }}
}}"""

_NEW_ELEMENT_RULE_APPLICABILITY_PROMPT = """你是 MemSlides 的 working-memory rule applicability compiler。你的任务不是重新解释用户偏好，而是判断“已经存在的结构化 WM rule”是否应该应用到本轮将新建的元素。

必须遵守：
- 只根据 active structured rules、用户本轮 feedback、intent、execution plan 判断 applicability。
- 不要用关键词机械匹配；要做语义判断。
- 只能把规则应用到本轮新创建的元素，不能回改旧元素或旧页面。
- 如果用户本轮没有创建相应元素，或证据不足，返回空 applications。
- 如果你不确定规则是否适用于本轮新增元素，或者无法从本轮反馈/计划中定位新增文本，返回空 applications；不要为了覆盖率猜测。
- 每个 application 必须包含来自本轮 feedback 的 evidence_spans，说明本轮会创建什么元素，例如“再加几个标签”。
- 如果 rule.target.semantic_target 或 rule.condition 指向局部语义文本，必须在 matched_text_spans 中列出本轮新文本里要应用规则的具体 span；不要把整句都视为目标，除非整句本身就是目标。
- matched_text_spans 只用于 semantic_target/text_span 规则；没有具体新文本 span 时跳过该 application。
- 如果规则要求的文字颜色与视觉质量冲突，优先保留规则要求，调整未被用户锁定的配套属性（如深色背景）；如果用户同时锁定冲突属性，标记 conflict_policy=fail_closed。
- 返回 JSON only。

User feedback:
{user_message}

Intent classifier result:
{intent_json}

Execution plan:
{plan_json}

Active structured WM rules:
{rule_specs_json}

返回格式：
{{
  "applications": [
    {{
      "rule_id": "wmr_...",
      "applies": true,
      "application_type": "future_element",
      "target_slide_paths": ["outputs/slide_06.html"],
      "element_kind": "pill_tag",
      "creation_scope": ["pill_tag"],
      "matched_text_spans": ["37.6"],
      "requirement": "新增 pill tag 文字为白色",
      "evidence_spans": ["再加几个标签"],
      "conflict_policy": "adjust_unlocked_companion_style|fail_closed|none",
      "repair_guidance": "保留白色文字；如果背景太浅，改成深色标签背景。"
    }}
  ],
  "ambiguities": []
}}"""

_NEW_ELEMENT_RULE_VERIFIER_PROMPT = """你是 MemSlides 的 working-memory new-element verifier。请检查本轮在已有 slide 中新增的元素是否满足 applicable WM rules。

核心职责：
- 只判断新创建的元素，不要求回改旧元素。
- 对 design_tokens / 模糊偏好（如“深绿色”“柔和”“学术”“低饱和”“不指定固定色号”）必须做语义判断，不要要求固定色号。
- 同一 slide 的多个新增元素和多个规则在本次调用中一起判断。
- 如果新增元素违反规则，返回 fail；如果证据不足或无法判断，返回 uncertain，系统会 fail closed。
- 如果规则要求文字颜色且当前背景对比不足，应该要求保留文字颜色并调整未锁定背景/容器样式；不要把用户偏好静默改掉。
- 返回 JSON only。

User feedback:
{user_message}

Slide path: {slide_path}

Applicable rule applications:
{applications_json}

Detected newly-created element records:
{new_records_json}

Before HTML:
{before_html}

After HTML:
{after_html}

返回格式：
{{
  "passed": false,
  "satisfied_elements": [
    {{"rule_id": "wmr_...", "element_text": "……", "evidence": "why it satisfies the fuzzy preference"}}
  ],
  "violations": [
    {{
      "rule_id": "wmr_...",
      "element_text": "……",
      "observed_issue": "……",
      "repair_instructions": "……"
    }}
  ],
  "judgements": [
    {{
      "rule_id": "wmr_...",
      "verdict": "pass|fail|uncertain",
      "reason": "……",
      "repair_guidance": "……"
    }}
  ]
}}"""


def _unique_tool_names(tool_names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in tool_names:
        name = str(name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def normalize_modify_scope(scope: str | None) -> str:
    value = str(scope or "").strip().lower()
    if value == "structural":
        return "global"
    if value in {"local", "global"}:
        return value
    return "local"


def normalize_modify_operation_kind(kind: str | None, *, scope: str | None = None) -> str:
    value = str(kind or "").strip().lower()
    if value in {"structural", "structure", "slide_structure", "deck_structure"}:
        return "structural"
    if value in {"diagram", "diagram_layout", "flowchart", "pipeline_layout", "process_diagram", "data_flow"}:
        return "diagram_layout"
    if value in {"preference", "memory_preference", "memory_update", "future_preference"}:
        return "preference_update"
    if value in {"controlled_rewrite", "controlled_full_rewrite", "full_regenerate", "full_regeneration"}:
        return "controlled_rewrite"
    if value in _MODIFY_POLICY_VALID_OPERATION_KINDS:
        return value
    if str(scope or "").strip().lower() == "structural":
        return "structural"
    return "style"


def normalize_modify_tool_group(group: str | None) -> str:
    value = str(group or "").strip().lower().replace("-", "_")
    aliases = {
        "patch": "local_patch",
        "local": "local_patch",
        "local_style": "local_patch",
        "batch": "global_style",
        "global": "global_style",
        "style_batch": "global_style",
        "structure": "structural",
        "slide_structure": "structural",
        "assets": "asset_discovery",
        "image": "asset_discovery",
        "images": "asset_discovery",
        "image_assets": "asset_discovery",
        "chart": "structured_visual",
        "charts": "structured_visual",
        "table": "structured_visual",
        "tables": "structured_visual",
        "visual": "structured_visual",
        "structured_visuals": "structured_visual",
        "diagram": "diagram_rewrite",
        "flowchart": "diagram_rewrite",
        "pipeline": "diagram_rewrite",
        "diagram_layout": "diagram_rewrite",
        "diagram_rewrite": "diagram_rewrite",
        "controlled_rewrite": "controlled_rewrite",
        "controlled_full_rewrite": "controlled_rewrite",
        "full_regenerate": "controlled_rewrite",
        "full_regeneration": "controlled_rewrite",
        "external": "external_acquisition",
        "search": "external_acquisition",
        "web": "external_acquisition",
        "research": "content_research",
        "memory": "memory_lookup",
        "todo": "progress",
    }
    value = aliases.get(value, value)
    return value if value in _MODIFY_POLICY_VALID_TOOL_GROUPS else ""


def normalize_modify_tool_groups(groups: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    for raw in groups or []:
        group = normalize_modify_tool_group(str(raw))
        if group and group not in normalized:
            normalized.append(group)
    return normalized


def build_modify_tool_allowlist(scope: str, operation_kind: str | None = None) -> list[str]:
    canonical_scope = normalize_modify_scope(scope)
    canonical_operation = normalize_modify_operation_kind(operation_kind, scope=scope)
    allowed = list(_MODIFY_SCOPE_SHARED_TOOLS)
    if canonical_scope == "global":
        allowed.extend(_MODIFY_TOOL_GROUPS["local_patch"])
        allowed.extend(_MODIFY_TOOL_GROUPS["global_style"])
        allowed.extend(_MODIFY_TOOL_GROUPS["structural"])
    else:
        allowed.extend(_MODIFY_TOOL_GROUPS["local_patch"])
    if canonical_operation == "diagram_layout":
        allowed.extend(_MODIFY_TOOL_GROUPS["diagram_rewrite"])
        allowed.extend(_MODIFY_TOOL_GROUPS["structured_visual"])
    if canonical_operation == "controlled_rewrite":
        allowed.extend(_MODIFY_TOOL_GROUPS["controlled_rewrite"])
    return _unique_tool_names(allowed)


def _available_tool_names(agent: Any) -> list[str]:
    return [
        str(tool.get("function", {}).get("name", "") or "").strip()
        for tool in getattr(agent, "tools", [])
        if isinstance(tool, dict) and str(tool.get("function", {}).get("name", "") or "").strip()
    ]


def restore_agent_base_tools(agent: Any) -> list[str]:
    """Restore the role-configured toolset before applying a per-turn policy."""
    if hasattr(agent, "restore_base_tools"):
        try:
            return list(agent.restore_base_tools())
        except Exception:
            logger.debug("Failed to restore agent captured base tools", exc_info=True)
    if hasattr(agent, "_setup_toolset"):
        try:
            agent._setup_toolset()
        except Exception:
            logger.debug("Failed to restore agent base tools", exc_info=True)
    return _available_tool_names(agent)


def _policy_hard_removed_tools(policy: ModifyToolPolicyPlan | None = None) -> set[str]:
    removed = set(_MODIFY_POLICY_HARD_REMOVE_COMMON)
    operation_kind = normalize_modify_operation_kind(
        getattr(policy, "operation_kind", None) if policy else None,
        scope=getattr(policy, "scope", None) if policy else None,
    )
    if operation_kind == "preference_update":
        removed.update(PREFERENCE_UPDATE_MUTATION_TOOLS)
    if operation_kind == "controlled_rewrite":
        removed.update(
            {
                "insert_slide",
                "delete_slide",
                "write_new_slide_file",
                "batch_update_css_rule",
                "batch_update_semantic_style",
                "patch_semantic_inline_style",
                "render_flowchart_asset",
            }
        )
    if operation_kind not in {
        "export_repair",
        "corrupted_recovery",
        "diagram_layout",
        "controlled_rewrite",
    }:
        removed.add(_MODIFY_POLICY_FULL_WRITE_TOOL)
    return removed


def _tools_for_policy_groups(groups: list[str]) -> list[str]:
    tools: list[str] = []
    for group in groups:
        tools.extend(_MODIFY_TOOL_GROUPS.get(group, []))
    return _unique_tool_names(tools)


def _infer_operation_kind_from_text(user_message: str, intent: dict[str, Any] | None = None) -> str:
    text = " ".join(str(user_message or "").split()).lower()
    intent_text = json.dumps(intent or {}, ensure_ascii=False).lower()
    combined = f"{text} {intent_text}"
    if is_structural_slide_operation(user_message, intent):
        return "structural"
    if classify_diagram_layout_intent(user_message, intent):
        return "diagram_layout"
    if any(
        token in combined
        for token in ("图表", "折线图", "柱状图", "饼图", "散点图", "三线表", "表格", "chart", "table", "line chart", "bar chart")
    ):
        return "content"
    if any(token in combined for token in ("图片", "图像", "换图", "替换图", "image", "picture", "photo", "figure")):
        return "image_asset"
    if any(token in combined for token in ("布局", "位置", "对齐", "间距", "overflow", "溢出", "layout", "spacing", "align")):
        return "layout"
    if any(token in combined for token in ("搜索", "查找", "补充资料", "新信息", "search", "research", "evidence")):
        return "research_content"
    if any(token in combined for token in ("文字", "内容", "改成", "替换", "删除这段", "content", "text")):
        return "content"
    return "style"


def _needs_structured_visual_tools(
    user_message: str,
    intent: dict[str, Any] | None = None,
    execution_plan: ModifyExecutionPlan | None = None,
) -> bool:
    text = " ".join(str(user_message or "").split()).lower()
    intent_text = json.dumps(intent or {}, ensure_ascii=False).lower()
    plan_text = json.dumps(
        {
            "reason": getattr(execution_plan, "reason", ""),
            "selector_hints": list(getattr(execution_plan, "selector_hints", []) or []),
        },
        ensure_ascii=False,
    ).lower()
    combined = f"{text} {intent_text} {plan_text}"
    if classify_diagram_layout_intent(user_message, intent):
        return True
    return any(
        token in combined
        for token in (
            "图表",
            "折线图",
            "柱状图",
            "饼图",
            "散点图",
            "趋势图",
            "三线表",
            "表格",
            "表1",
            "chart",
            "table",
            "line chart",
            "bar chart",
            "donut",
            "pie",
        )
    )


def classify_diagram_layout_intent(
    user_message: str,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a compact diagram contract when an existing slide should become a diagram."""
    text = " ".join(str(user_message or "").split())
    lowered = text.lower()
    intent_text = json.dumps(intent or {}, ensure_ascii=False).lower()
    combined = f"{lowered} {intent_text}"
    diagram_tokens = (
        "流程图",
        "管线",
        "机制图",
        "结构图",
        "架构图",
        "数据流",
        "重排成",
        "flowchart",
        "flow chart",
        "pipeline",
        "process diagram",
        "data flow",
        "diagram",
    )
    if not any(token in combined for token in diagram_tokens):
        return None
    if not (
        infer_specific_existing_slide_reference(user_message)
        or str((intent or {}).get("target_slide", "") or "").strip()
        or re.search(r"第\s*\d+\s*页", text)
    ):
        return None

    attention_tokens = ("attention", "q/k/v", "qkv", "w_q", "w_k", "w_v", "weighted sum", "softmax", "注意力")
    is_attention = any(token in combined for token in attention_tokens)
    branch_tokens = ("q/k/v", "qkv", "w_q", "w_k", "w_v", "分支", "branch")
    if is_attention:
        required_nodes = [
            "输入 token",
            "W_Q",
            "W_K",
            "W_V",
            "Q",
            "K",
            "V",
            "attention score",
            "softmax",
            "weighted sum",
            "output",
        ]
        required_edges = [
            "输入 token -> W_Q/W_K/W_V",
            "W_Q/W_K/W_V -> Q/K/V",
            "Q/K -> attention score",
            "attention score -> softmax",
            "softmax + V -> weighted sum",
            "weighted sum -> output",
        ]
        title_hint = "Attention Pipeline"
        diagram_kind = "branch_pipeline"
    else:
        required_nodes = _extract_diagram_nodes_from_text(text)
        required_edges = []
        title_hint = "Pipeline Diagram" if "pipeline" in combined or "管线" in combined else "Flowchart"
        diagram_kind = "branch_pipeline" if any(token in combined for token in branch_tokens) else "pipeline"
        if "architecture" in combined or "架构" in combined or "结构图" in combined:
            diagram_kind = "architecture"

    marker_match = re.search(r"TEMP-PREF-[A-Za-z0-9_-]+", text)
    return {
        "diagram_kind": diagram_kind,
        "title_hint": title_hint,
        "required_nodes": required_nodes,
        "required_edges": required_edges,
        "remove_conflicting_assets": True,
        "marker_text": marker_match.group(0) if marker_match else "",
        "source": "deterministic_classifier",
    }


def _extract_diagram_nodes_from_text(text: str) -> list[str]:
    compact = str(text or "").strip()
    match = re.search(r"从(.{1,120}?)(?:，|。|,|\.|$)", compact)
    if not match:
        return []
    segment = match.group(1)
    segment = re.sub(r"(尽量|最好|请|帮我|重排成|改成|像流程图).*", "", segment)
    raw_parts = re.split(r"\s*(?:到|->|→|、|,|，|和|以及)\s*", segment)
    nodes: list[str] = []
    for part in raw_parts:
        node = " ".join(part.split()).strip(" ：:;；")
        if node and len(node) <= 40 and node not in nodes:
            nodes.append(node)
    return nodes[:12]


_FUTURE_SLIDE_DELTA_CONTEXT_RE = re.compile(
    r"(后面|后续|以后|未来|之后|下次|接下来|future|next round|next rounds).{0,36}"
    r"(新增|新加|添加|增加|插入|重写|rewrite|regenerate|new slides?|future slides?)"
    r"|"
    r"(新增|新加|添加|增加|插入|new slides?|future slides?).{0,36}"
    r"(也|都|继续|沿用|遵守|应用|apply|follow|inherit)",
    re.IGNORECASE,
)


def _strip_future_slide_delta_context(text: str) -> str:
    """Remove future-propagation clauses before current-turn slide delta inference."""
    if not text:
        return ""
    previous = None
    stripped = text
    while previous != stripped:
        previous = stripped
        stripped = _FUTURE_SLIDE_DELTA_CONTEXT_RE.sub(" ", stripped)
    return " ".join(stripped.split())


def _expected_slide_delta_from_text(user_message: str, intent: dict[str, Any] | None = None) -> int:
    text = " ".join(str(user_message or "").split()).lower()
    intent_text = json.dumps(intent or {}, ensure_ascii=False).lower()
    current_text = _strip_future_slide_delta_context(text)
    combined = f"{current_text} {intent_text}"
    if re.search(r"(删除|删掉|移除|去掉|remove|delete).{0,24}(页面|页|slide|第\s*\d+\s*页)", combined):
        return -1
    if re.search(
        r"(新增|新加|添加|增加|插入|补|加)\s*(?:一|1|\d+)\s*(?:个|张)?\s*(?:页面|页|slide)",
        combined,
    ):
        return 1
    if re.search(
        r"(新增|新加|添加|增加|插入|补\s*(?:一|1|\d+)\s*页|加\s*(?:一|1|\d+)\s*页|append|insert|add)"
        r".{0,24}(页面|页|slide)",
        combined,
    ):
        return 1
    return 0


def _current_turn_slide_delta_signal(user_message: str, intent: dict[str, Any] | None = None) -> int:
    """High-confidence current-turn slide-count signal used to arbitrate LLM/fallback policy."""
    return _expected_slide_delta_from_text(user_message, intent)


def _coerce_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _normalize_rewrite_scope(value: Any) -> str:
    scope = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "single": "single_slide",
        "one_slide": "single_slide",
        "slide": "single_slide",
        "local": "single_slide",
        "multi": "multi_slide",
        "multiple": "multi_slide",
        "multiple_slides": "multi_slide",
        "global": "multi_slide",
        "deck": "multi_slide",
        "": "none",
        "false": "none",
        "no": "none",
    }
    scope = aliases.get(scope, scope)
    return scope if scope in {"single_slide", "multi_slide", "none"} else "none"


def _normalize_rewrite_decision_payload(
    raw_decision: Any,
    runtime: Any,
    *,
    fallback_targets: list[Path] | None = None,
    user_message: str = "",
    intent: dict[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    raw = raw_decision if isinstance(raw_decision, dict) else {}
    needs = bool(raw.get("needs_controlled_rewrite", False))
    scope = _normalize_rewrite_scope(raw.get("rewrite_scope"))
    if not needs:
        scope = "none"

    raw_targets = raw.get("rewrite_target_slides")
    if raw_targets is None:
        raw_targets = raw.get("target_slides")
    target_paths = _resolve_policy_target_paths(
        runtime,
        raw_targets if raw_targets not in (None, [], "") else None,
        intent=intent,
    )
    if not target_paths:
        target_paths = list(fallback_targets or [])
    if not target_paths:
        inferred = infer_specific_existing_slide_reference(user_message)
        if inferred:
            resolved = resolve_slide_path(runtime, inferred)
            if resolved is not None:
                target_paths = [resolved]

    confidence = _coerce_confidence(
        raw.get("confidence"),
        default=0.75 if needs and scope == "single_slide" else 0.0,
    )
    return {
        "needs_controlled_rewrite": bool(needs),
        "rewrite_reason": str(raw.get("rewrite_reason", "") or "").strip(),
        "rewrite_scope": scope,
        "rewrite_target_slides": [
            workspace_relative_label(runtime, path) for path in target_paths
        ],
        "confidence": confidence,
        "source": str(source or raw.get("source", "") or "").strip(),
    }


def _rewrite_decision_target_paths(
    runtime: Any,
    decision: dict[str, Any] | None,
    *,
    fallback_targets: list[Path] | None = None,
    intent: dict[str, Any] | None = None,
) -> list[Path]:
    if not isinstance(decision, dict):
        return []
    targets = _resolve_policy_target_paths(
        runtime,
        decision.get("rewrite_target_slides"),
        intent=intent,
    )
    if not targets:
        targets = list(fallback_targets or [])
    return targets


def _is_controlled_rewrite_decision_allowed(
    decision: dict[str, Any] | None,
    runtime: Any,
    *,
    expected_slide_delta: int,
    fallback_targets: list[Path] | None = None,
    intent: dict[str, Any] | None = None,
    min_confidence: float = 0.55,
) -> tuple[bool, list[Path]]:
    if expected_slide_delta != 0 or not isinstance(decision, dict):
        return False, []
    if not bool(decision.get("needs_controlled_rewrite", False)):
        return False, []
    if _normalize_rewrite_scope(decision.get("rewrite_scope")) != "single_slide":
        return False, []
    if _coerce_confidence(decision.get("confidence")) < min_confidence:
        return False, []
    targets = _rewrite_decision_target_paths(
        runtime,
        decision,
        fallback_targets=fallback_targets,
        intent=intent,
    )
    if len(targets) != 1:
        return False, targets
    if not targets[0].exists():
        return False, targets
    return True, targets


def classify_controlled_rewrite_intent(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    fallback_targets: list[Path] | None = None,
) -> dict[str, Any]:
    """Conservative fallback for page-level information-architecture rewrites."""
    text = " ".join(str(user_message or "").split()).lower()
    combined = f"{text} {json.dumps(intent or {}, ensure_ascii=False).lower()}"
    if not combined.strip() or _current_turn_slide_delta_signal(user_message, intent) != 0:
        return _normalize_rewrite_decision_payload({}, runtime, source="deterministic")
    if classify_diagram_layout_intent(user_message, intent):
        return _normalize_rewrite_decision_payload({}, runtime, source="deterministic")

    target_paths = list(fallback_targets or [])
    if len(target_paths) != 1:
        inferred = infer_specific_existing_slide_reference(user_message)
        if inferred:
            resolved = resolve_slide_path(runtime, inferred)
            target_paths = [resolved] if resolved is not None else []
    if len(target_paths) != 1:
        return _normalize_rewrite_decision_payload({}, runtime, source="deterministic")

    local_tweak_patterns = (
        r"(颜色|字体颜色|文字颜色|标题颜色|字号|加粗|斜体|下划线|背景色|改[为成].{0,12}(白色|黑色|红色|蓝色|绿色))",
        r"(替换|改成|删除|补充|增加).{0,24}(一句话|这句话|这个词|标题文本|文案|标点|错别字)",
        r"(换图|替换图|换图片|图片换成|image|picture|photo)",
    )
    if any(re.search(pattern, combined) for pattern in local_tweak_patterns):
        return _normalize_rewrite_decision_payload({}, runtime, source="deterministic")

    page_restructure_patterns = (
        r"(首页|封面|第一页|开场页|总结页|结论页|机制页|第\s*\d+\s*(页|张)|slide[_\s-]*0?\d+).{0,80}(太散|很散|收紧|扫到重点|一眼.*重点|可扫读|重新组织|信息架构|开场|汇报会|brief|decision brief)",
        r"(只保留|保留).{0,48}(三|3|几个|若干).{0,48}(块|点|判断点|问题|贡献|理由|部分)",
        r"(改成|做成|重构成|整理成|变成).{0,48}(三块|三列|三卡片|三段式|brief|decision brief|开场页|汇报会.*开场)",
    )
    if not any(re.search(pattern, combined) for pattern in page_restructure_patterns):
        return _normalize_rewrite_decision_payload({}, runtime, source="deterministic")

    confidence = 0.78
    if re.search(r"(只保留|三块|三列|三卡片|decision brief|扫到重点|信息架构)", combined):
        confidence = 0.88
    return _normalize_rewrite_decision_payload(
        {
            "needs_controlled_rewrite": True,
            "rewrite_scope": "single_slide",
            "rewrite_target_slides": [workspace_relative_label(runtime, target_paths[0])],
            "rewrite_reason": (
                "Deterministic fallback detected a single-slide page-level information "
                "architecture rewrite request."
            ),
            "confidence": confidence,
        },
        runtime,
        fallback_targets=target_paths,
        user_message=user_message,
        intent=intent,
        source="deterministic",
    )


def _resolve_policy_target_paths(
    runtime: Any,
    raw_targets: Any,
    *,
    fallback_plan: ModifyExecutionPlan | None = None,
    intent: dict[str, Any] | None = None,
) -> list[Path]:
    targets: list[Path] = []

    def _add(raw: Any) -> None:
        if raw is None:
            return
        text = str(raw or "").strip()
        if not text or text.lower() in {"all", "*"}:
            for path in resolve_all_slide_paths(runtime):
                if path not in targets:
                    targets.append(path)
            return
        resolved = resolve_slide_path(runtime, text)
        if resolved is None:
            path = Path(text)
            if not path.is_absolute():
                path = runtime.workspace / path
            if path.exists():
                resolved = path.resolve()
        if resolved is not None and resolved not in targets:
            targets.append(resolved)

    if isinstance(raw_targets, str):
        _add(raw_targets)
    elif isinstance(raw_targets, list):
        for item in raw_targets:
            _add(item)

    if not targets and fallback_plan is not None:
        targets.extend(list(getattr(fallback_plan, "target_slide_paths", []) or []))

    if not targets and isinstance(intent, dict):
        target = str(intent.get("target_slide", "") or "").strip()
        if target:
            _add(target)

    return targets


def build_fallback_modify_tool_policy_plan(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    execution_plan: ModifyExecutionPlan | None,
    reason: str = "",
) -> ModifyToolPolicyPlan:
    text_inferred_operation_kind = _infer_operation_kind_from_text(user_message, intent)
    operation_kind = normalize_modify_operation_kind(
        getattr(execution_plan, "operation_kind", None),
        scope=getattr(execution_plan, "scope", None),
    )
    if operation_kind == "style" and execution_plan is None:
        operation_kind = text_inferred_operation_kind
    elif operation_kind == "style":
        if text_inferred_operation_kind in {"image_asset", "layout", "diagram_layout", "research_content", "structural"}:
            operation_kind = text_inferred_operation_kind

    deterministic_rewrite_decision = classify_controlled_rewrite_intent(
        runtime,
        user_message=user_message,
        intent=intent,
        fallback_targets=list(getattr(execution_plan, "target_slide_paths", []) or []),
    )

    scope = normalize_modify_scope(getattr(execution_plan, "scope", None))
    if execution_plan is None:
        scope = "global" if operation_kind == "structural" else "local"
    if operation_kind in {"structural", "mixed"}:
        scope = "global"

    tool_groups = list(_MODIFY_POLICY_DEFAULT_GROUPS_BY_OPERATION.get(operation_kind, ["local_patch"]))
    if scope == "global" and operation_kind not in {"structural", "query", "preference_update"}:
        if "global_style" not in tool_groups:
            tool_groups.append("global_style")
    if operation_kind == "image_asset" and "asset_discovery" not in tool_groups:
        tool_groups.insert(0, "asset_discovery")
    if _needs_structured_visual_tools(user_message, intent, execution_plan) and "structured_visual" not in tool_groups:
        tool_groups.append("structured_visual")

    target_paths = list(getattr(execution_plan, "target_slide_paths", []) or [])
    if not target_paths and operation_kind not in {"query", "preference_update"}:
        target_paths = _resolve_policy_target_paths(
            runtime,
            None,
            fallback_plan=execution_plan,
            intent=intent,
        )

    expected_slide_delta = _expected_slide_delta_from_text(user_message, intent)
    expected_slide_delta_source = "text" if expected_slide_delta != 0 else "default_zero"
    if expected_slide_delta != 0:
        operation_kind = "structural"
        scope = "global"
        if "structural" not in tool_groups:
            tool_groups.insert(0, "structural")
        if "local_patch" not in tool_groups:
            tool_groups.append("local_patch")
    else:
        rewrite_allowed, rewrite_targets = _is_controlled_rewrite_decision_allowed(
            deterministic_rewrite_decision,
            runtime,
            expected_slide_delta=expected_slide_delta,
            fallback_targets=target_paths,
            intent=intent,
            min_confidence=0.80,
        )
        if operation_kind == "style" and rewrite_allowed:
            operation_kind = "controlled_rewrite"
            scope = "local"
            target_paths = rewrite_targets
            tool_groups = list(_MODIFY_POLICY_DEFAULT_GROUPS_BY_OPERATION["controlled_rewrite"])
    first_steps: list[str] = []
    if is_preference_update_plan(execution_plan):
        expected_slide_delta = 0
        first_steps = [
            "The future-only preference has already been captured in working memory.",
            "Do not edit existing slide files for this turn.",
            "Optionally call remember_lesson to leave a compact note, then call finalize.",
        ]
    elif operation_kind == "structural" and expected_slide_delta > 0:
        first_steps = [
            "Use scan_slide_index/read_file only if needed to choose the insertion anchor.",
            "Call insert_slide to create the canonical new slide file.",
            "If insert_slide reports PLACEHOLDER_CREATED, call write_new_slide_file on the returned write_target with complete slide HTML.",
            "Apply any active future/new-element working-memory preferences to the inserted slide content and styling.",
            "Inspect the completed new slide; use local patch tools only for small post-write repairs.",
        ]
    elif operation_kind == "structural" and expected_slide_delta < 0:
        first_steps = [
            "Use scan_slide_index/read_file only if needed to identify the deletion target.",
            "Call delete_slide with renumber=true, then inspect neighboring slide order before finalize.",
        ]
    elif operation_kind == "diagram_layout":
        first_steps = [
            "Call read_slide_snapshot on the target slide to get the current content_hash.",
            "Regenerate the target slide as one coherent diagram page with write_html_file(force_regenerate=true, expected_hash=content_hash).",
            "Use render_flowchart_asset when the request needs a reliable flowchart/pipeline visual.",
            "Remove old conflicting figures/captions/empty containers, then inspect_slide before finalize.",
        ]
    elif operation_kind == "controlled_rewrite":
        first_steps = [
            "Call read_slide_snapshot on the target slide to get the current content_hash.",
            "Rewrite the same target slide with write_html_file(force_regenerate=true, expected_hash=content_hash).",
            "Use a clearly structured page-level layout such as headline plus three cards/columns/section blocks.",
            "Preserve active Temporary preferences such as conclusion-title wording and TEMP-PREF markers.",
            "Run inspect_slide before finalize.",
        ]
    elif getattr(execution_plan, "new_element_rule_applications", None):
        app_text = json.dumps(
            _json_safe(getattr(execution_plan, "new_element_rule_applications", []) or []),
            ensure_ascii=False,
        ).lower()
        first_steps = [
            "Apply the listed WM rule application only to elements newly created in this turn.",
            "Keep the current edit local to the target slide(s); do not batch-edit old slides.",
            "If a text-color preference conflicts with contrast, preserve the preferred text color and adjust unlocked companion styling such as the tag background.",
        ]
        if "caption" in app_text or "图注" in app_text or "说明" in app_text:
            first_steps.append(
                "For newly created captions, apply the remembered caption color/font-style directly to the new caption element; fuzzy color intents such as deep green must be visibly satisfied and must not be replaced by an unrelated generic theme token."
            )

    return ModifyToolPolicyPlan(
        scope=scope,
        operation_kind=operation_kind,
        tool_groups=normalize_modify_tool_groups(tool_groups),
        target_slide_paths=target_paths,
        expected_slide_delta=expected_slide_delta,
        coverage_required=bool(getattr(execution_plan, "coverage_required", False)),
        first_steps=first_steps,
        risk_flags=[],
        requested_tools=[],
        reason=reason or getattr(execution_plan, "reason", "") or "Fallback modify tool policy generated.",
        source="fallback",
        new_element_rule_applications=_json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        ),
        expected_slide_delta_source=expected_slide_delta_source,
        rewrite_decision=_json_safe(deterministic_rewrite_decision),
        rewrite_decision_source=(
            "deterministic_override"
            if operation_kind == "controlled_rewrite"
            else "deterministic_fallback"
        ),
    )


def plan_has_future_only_rules(plan: ModifyExecutionPlan | None) -> bool:
    if plan is None:
        return False
    operation_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=getattr(plan, "scope", None),
    )
    if operation_kind == "preference_update":
        return True
    rule_ids = (
        list(getattr(plan, "target_rule_ids", []) or [])
        + list(getattr(plan, "applicable_rule_ids", []) or [])
    )
    if getattr(plan, "new_element_rule_applications", None):
        rule_ids.extend(
            str(app.get("rule_id", "") or "").strip()
            for app in getattr(plan, "new_element_rule_applications", []) or []
            if isinstance(app, dict)
        )
    return bool(rule_ids) and not bool(getattr(plan, "coverage_required", False))


def clamp_policy_to_execution_plan(
    policy: ModifyToolPolicyPlan,
    execution_plan: ModifyExecutionPlan | None,
    *,
    source_suffix: str = "",
) -> ModifyToolPolicyPlan:
    if execution_plan is None:
        return policy

    operation_kind = normalize_modify_operation_kind(
        getattr(execution_plan, "operation_kind", None),
        scope=getattr(execution_plan, "scope", None),
    )
    policy_expected_delta = int(getattr(policy, "expected_slide_delta", 0) or 0)
    plan_is_structural = operation_kind == "structural" or policy_expected_delta != 0
    plan_targets = list(getattr(execution_plan, "target_slide_paths", []) or [])
    changed = False
    clamped = deepcopy(policy)
    if policy_expected_delta != 0 and normalize_modify_operation_kind(clamped.operation_kind, scope=clamped.scope) != "structural":
        clamped.scope = "global"
        clamped.operation_kind = "structural"
        clamped.expected_slide_delta = policy_expected_delta
        clamped.tool_groups = normalize_modify_tool_groups(list(clamped.tool_groups or []) + ["structural", "local_patch"])
        changed = True

    if operation_kind == "preference_update":
        clamped.scope = "local"
        clamped.operation_kind = "preference_update"
        clamped.tool_groups = []
        clamped.target_slide_paths = []
        clamped.expected_slide_delta = 0
        clamped.coverage_required = False
        clamped.requested_tools = [
            name
            for name in (clamped.requested_tools or [])
            if name not in PREFERENCE_UPDATE_MUTATION_TOOLS
        ]
        changed = True
    elif plan_has_future_only_rules(execution_plan):
        clamped.target_slide_paths = plan_targets
        clamped.coverage_required = bool(getattr(execution_plan, "coverage_required", False))
        clamped_kind = normalize_modify_operation_kind(clamped.operation_kind, scope=clamped.scope)
        if clamped_kind == "controlled_rewrite":
            clamped.scope = "local"
            clamped.operation_kind = "controlled_rewrite"
            clamped.tool_groups = ["controlled_rewrite", "local_patch"]
            clamped.expected_slide_delta = 0
            clamped.requested_tools = [
                name
                for name in (clamped.requested_tools or [])
                if name
                not in {
                    "batch_update_css_rule",
                    "batch_update_semantic_style",
                    "patch_semantic_inline_style",
                    "insert_slide",
                    "delete_slide",
                    "write_new_slide_file",
                    "render_flowchart_asset",
                }
            ]
            for name in ("read_slide_snapshot", "write_html_file", "inspect_slide"):
                if name not in clamped.requested_tools:
                    clamped.requested_tools.append(name)
            changed = True
        elif not plan_is_structural:
            clamped.scope = "local"
            clamped.operation_kind = operation_kind
            clamped.tool_groups = [
                group
                for group in normalize_modify_tool_groups(clamped.tool_groups)
                if group not in {"global_style", "structural"}
            ]
            if "local_patch" not in clamped.tool_groups:
                clamped.tool_groups.insert(0, "local_patch")
            clamped.expected_slide_delta = 0
            clamped.requested_tools = [
                name
                for name in (clamped.requested_tools or [])
                if name not in {
                    "batch_update_css_rule",
                    "batch_update_semantic_style",
                    "patch_semantic_inline_style",
                    "insert_slide",
                    "delete_slide",
                    "write_new_slide_file",
                    _MODIFY_POLICY_FULL_WRITE_TOOL,
                    "render_flowchart_asset",
                }
            ]
            changed = True
        else:
            clamped.scope = "global"
            clamped.operation_kind = "structural"
            clamped.expected_slide_delta = policy_expected_delta
            clamped.tool_groups = normalize_modify_tool_groups(list(clamped.tool_groups or []) + ["structural", "local_patch"])
            changed = True

    if not changed:
        return policy

    suffix = f"+{source_suffix}" if source_suffix else "+semantic_guard"
    clamped.source = f"{policy.source}{suffix}"
    clamped.reason = (
        (policy.reason or "").rstrip()
        + " Semantic guard clamped tool scope to the LLM preference propagation decision."
    ).strip()
    return clamped


def align_modify_execution_plan_to_tool_policy(
    plan: ModifyExecutionPlan | None,
    policy: ModifyToolPolicyPlan | None,
) -> ModifyExecutionPlan | None:
    """Keep the model-facing execution plan consistent with the final tool policy.

    The semantic preference plan can be local/style while the policy planner correctly
    detects a structural slide-count operation from the raw user request. In that case
    Temporary prefs are constraints for newly created content, not a current-slide-only
    limiter, so downstream guards must see a structural execution plan too.
    """
    if plan is None or policy is None:
        return plan

    policy_kind = normalize_modify_operation_kind(
        getattr(policy, "operation_kind", None),
        scope=getattr(policy, "scope", None),
    )
    try:
        expected_delta = int(getattr(policy, "expected_slide_delta", 0) or 0)
    except (TypeError, ValueError):
        expected_delta = 0
    if policy_kind == "controlled_rewrite" and expected_delta == 0:
        policy_targets = list(getattr(policy, "target_slide_paths", []) or [])
        if len(policy_targets) != 1:
            return plan
        plan_kind = normalize_modify_operation_kind(
            getattr(plan, "operation_kind", None),
            scope=getattr(plan, "scope", None),
        )
        if (
            plan_kind == "controlled_rewrite"
            and normalize_modify_scope(getattr(plan, "scope", None)) == "local"
            and list(getattr(plan, "target_slide_paths", []) or []) == policy_targets
        ):
            return plan
        aligned = deepcopy(plan)
        aligned.scope = "local"
        aligned.operation_kind = "controlled_rewrite"
        aligned.coverage_required = True
        aligned.target_slide_paths = policy_targets
        aligned.rewrite_decision = _json_safe(getattr(policy, "rewrite_decision", None) or {})
        aligned.rewrite_decision_source = str(getattr(policy, "rewrite_decision_source", "") or "")
        reason = str(getattr(aligned, "reason", "") or "").strip()
        alignment_note = (
            "Tool policy identified a single-slide controlled rewrite; treat this as "
            "page-level information architecture restructuring, not a local patch or slide-count operation."
        )
        if alignment_note not in reason:
            aligned.reason = f"{reason} {alignment_note}".strip()
        return aligned

    if policy_kind != "structural" and expected_delta == 0:
        return plan

    plan_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=getattr(plan, "scope", None),
    )
    policy_targets = list(getattr(policy, "target_slide_paths", []) or [])
    if (
        plan_kind == "structural"
        and normalize_modify_scope(getattr(plan, "scope", None)) == "global"
        and not bool(getattr(plan, "coverage_required", False))
        and (not policy_targets or list(getattr(plan, "target_slide_paths", []) or []) == policy_targets)
    ):
        return plan

    aligned = deepcopy(plan)
    aligned.scope = "global"
    aligned.operation_kind = "structural"
    aligned.coverage_required = False
    if policy_targets:
        aligned.target_slide_paths = policy_targets
    reason = str(getattr(aligned, "reason", "") or "").strip()
    alignment_note = (
        "Tool policy identified a structural slide-count operation; active Temporary prefs "
        "apply as constraints to newly created content, not as a current-slide-only limiter."
    )
    if alignment_note not in reason:
        aligned.reason = f"{reason} {alignment_note}".strip()
    return aligned


def render_modify_tool_policy_prompt(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    execution_plan: ModifyExecutionPlan | None,
    wm_rule_specs_text: str = "",
    available_tools: list[str] | None = None,
) -> str:
    slide_paths = [
        workspace_relative_label(runtime, path)
        for path in resolve_all_slide_paths(runtime)[:40]
    ]
    tool_lines = "\n".join(
        f"- {group}: {', '.join(tools)}"
        for group, tools in _MODIFY_TOOL_GROUPS.items()
    )
    return (
        "You are planning the tool policy for a MemSlides modify turn. "
        "Return JSON only. Choose tool groups that are likely needed, but do not over-narrow: "
        "existing slide edits must keep local patch tools available; a placeholder inserted slide must be completed with "
        "`write_new_slide_file` before local patch tools are used for small repairs.\n\n"
        "Valid scopes: local, global.\n"
        "Valid operation_kind: style, content, layout, diagram_layout, controlled_rewrite, image_asset, structural, research_content, mixed, preference_update, query, export_repair, corrupted_recovery.\n"
        "Valid tool_groups:\n"
        f"{tool_lines}\n\n"
        "Important policy:\n"
        "- A concrete target slide number is not a structural operation by itself.\n"
        "- structural means add/delete/insert/move/reorder slides.\n"
        "- diagram_layout means rewriting an existing target slide into a flowchart/pipeline/mechanism/data-flow diagram; it is not a page-count operation.\n"
        "- controlled_rewrite means one existing slide needs page-level information architecture restructuring, not just a local patch. It must not change slide count.\n"
        "- image_asset should include asset_discovery; add external_acquisition only if local/document assets are likely insufficient.\n"
        "- chart/table/three-line-table/flowchart/pipeline requests should keep structured_visual available together with local_patch.\n"
        "- Do not request write_html_file for normal modify turns; existing healthy slides use patch/batch tools. The exceptions are diagram_layout and controlled_rewrite for exactly one target slide.\n"
        "- In rewrite_decision, distinguish local edits from page-level restructuring: changing wording/color/image/small spacing is local; reorganizing a homepage/summary/mechanism page into a scan-friendly brief with only a few blocks is controlled_rewrite.\n"
        "- Future/new-slide preference wording such as 'later new pages should also follow this rule' is not structural and not controlled_rewrite by itself.\n\n"
        "- If the rule execution plan has future-only preference rule IDs and coverage_required=false, keep the policy local to the listed current target slides. Do not add global_style, insert/delete, or deck-wide batch tools unless the plan itself is structural.\n"
        "- For add/insert-slide tasks, first_steps should prefer insert_slide, then write_new_slide_file for the inserted file, "
        "then inspect_slide; do not make apply_slide_patch the primary completion action for a placeholder new slide.\n\n"
    f"User message: {user_message}\n"
    f"Intent JSON: {json.dumps(intent or {}, ensure_ascii=False)}\n"
        f"Rule execution plan: {json.dumps({'scope': getattr(execution_plan, 'scope', ''), 'operation_kind': getattr(execution_plan, 'operation_kind', ''), 'targets': [workspace_relative_label(runtime, p) for p in getattr(execution_plan, 'target_slide_paths', []) or []], 'coverage_required': bool(getattr(execution_plan, 'coverage_required', False)), 'target_rule_ids': list(getattr(execution_plan, 'target_rule_ids', []) or []), 'applicable_rule_ids': list(getattr(execution_plan, 'applicable_rule_ids', []) or []), 'new_element_rule_applications': _json_safe(getattr(execution_plan, 'new_element_rule_applications', []) or []), 'reason': getattr(execution_plan, 'reason', '')}, ensure_ascii=False)}\n"
        f"Current slides: {json.dumps(slide_paths, ensure_ascii=False)}\n"
        f"Memory rule specs excerpt: {str(wm_rule_specs_text or '')[:3000]}\n"
        f"Available tools: {json.dumps(sorted(available_tools or []), ensure_ascii=False)}\n\n"
        "Return JSON with this shape:\n"
        "{\n"
        '  "scope": "local|global",\n'
        '  "operation_kind": "...",\n'
        '  "target_slides": ["outputs/slide_02.html"],\n'
        '  "expected_slide_delta": 0,\n'
        '  "coverage_required": false,\n'
        '  "tool_groups": ["local_patch"],\n'
        '  "requested_tools": [],\n'
        '  "first_steps": ["..."],\n'
        '  "risk_flags": ["..."],\n'
        '  "reason": "...",\n'
        '  "rewrite_decision": {\n'
        '    "needs_controlled_rewrite": false,\n'
        '    "rewrite_reason": "",\n'
        '    "rewrite_scope": "none|single_slide|multi_slide",\n'
        '    "rewrite_target_slides": [],\n'
        '    "confidence": 0.0\n'
        "  }\n"
        "}\n"
    )


async def build_modify_tool_policy_plan(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    execution_plan: ModifyExecutionPlan | None,
    wm_rule_specs_text: str = "",
    agent: Any | None = None,
) -> ModifyToolPolicyPlan:
    available_tools = _available_tool_names(agent) if agent is not None else []
    fallback = build_fallback_modify_tool_policy_plan(
        runtime,
        user_message=user_message,
        intent=intent,
        execution_plan=execution_plan,
    )
    fallback = clamp_policy_to_execution_plan(fallback, execution_plan, source_suffix="semantic_guard")

    llm = None
    try:
        llm_objects = getattr(getattr(runtime, "memory_system", None), "llm_objects_by_task", {}) or {}
        llm = (
            llm_objects.get("intent_classify")
            or llm_objects.get("style_classify")
            or getattr(getattr(runtime, "memory_system", None), "llm", None)
            or getattr(agent, "llm", None)
        )
    except Exception:
        llm = getattr(agent, "llm", None)
    if is_preference_update_plan(execution_plan):
        fallback.new_element_rule_applications = _json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        )
        return fallback
    if llm is None:
        fallback.new_element_rule_applications = _json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        )
        return fallback

    prompt = render_modify_tool_policy_prompt(
        runtime,
        user_message=user_message,
        intent=intent,
        execution_plan=execution_plan,
        wm_rule_specs_text=wm_rule_specs_text,
        available_tools=available_tools,
    )
    raw_response = ""
    try:
        if callable(llm) and not hasattr(llm, "run"):
            raw_response = await llm(prompt)
        else:
            response = await llm.run(messages=[{"role": "user", "content": prompt}])
            raw_response = response.choices[0].message.content or ""
        payload = extract_json_object(raw_response)
    except Exception as exc:
        logger.warning("Modify tool policy LLM planning failed; using fallback: %s", exc)
        fallback.raw_response = f"planner_error: {exc}"
        (
            fallback.prompt_artifact,
            fallback.response_artifact,
            fallback.payload_artifact,
        ) = write_modify_tool_policy_planner_artifacts(
            runtime,
            prompt=prompt,
            raw_response=f"planner_error: {exc}\n\n{raw_response or ''}",
            payload={},
            source="fallback_error",
        )
        fallback.new_element_rule_applications = _json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        )
        return fallback

    if not payload:
        fallback.raw_response = str(raw_response or "")
        (
            fallback.prompt_artifact,
            fallback.response_artifact,
            fallback.payload_artifact,
        ) = write_modify_tool_policy_planner_artifacts(
            runtime,
            prompt=prompt,
            raw_response=raw_response,
            payload={},
            source="fallback_empty",
        )
        fallback.new_element_rule_applications = _json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        )
        return fallback

    operation_kind = normalize_modify_operation_kind(
        payload.get("operation_kind"),
        scope=payload.get("scope"),
    )
    deterministic_rewrite_decision = classify_controlled_rewrite_intent(
        runtime,
        user_message=user_message,
        intent=intent,
        fallback_targets=list(getattr(execution_plan, "target_slide_paths", []) or []),
    )
    llm_rewrite_decision = _normalize_rewrite_decision_payload(
        payload.get("rewrite_decision"),
        runtime,
        fallback_targets=list(getattr(execution_plan, "target_slide_paths", []) or []),
        user_message=user_message,
        intent=intent,
        source="llm",
    )
    rewrite_decision = llm_rewrite_decision
    rewrite_decision_source = "llm"
    if operation_kind == "style":
        inferred = _infer_operation_kind_from_text(user_message, intent)
        if inferred in {"structural", "diagram_layout"}:
            operation_kind = "structural"
            if inferred == "diagram_layout":
                operation_kind = "diagram_layout"
    scope = normalize_modify_scope(payload.get("scope"))
    if operation_kind in {"structural", "mixed"}:
        scope = "global"

    tool_groups = normalize_modify_tool_groups(payload.get("tool_groups") if isinstance(payload.get("tool_groups"), list) else [])
    if not tool_groups:
        tool_groups = list(fallback.tool_groups)
    if _needs_structured_visual_tools(user_message, intent, execution_plan) and "structured_visual" not in tool_groups:
        tool_groups.append("structured_visual")
    if operation_kind == "diagram_layout":
        scope = "local"
        if "diagram_rewrite" not in tool_groups:
            tool_groups.insert(0, "diagram_rewrite")
        if "structured_visual" not in tool_groups:
            tool_groups.append("structured_visual")
    target_paths = _resolve_policy_target_paths(
        runtime,
        payload.get("target_slides"),
        fallback_plan=execution_plan,
        intent=intent,
    )

    payload_has_expected_delta = "expected_slide_delta" in payload
    expected_delta = payload.get("expected_slide_delta", fallback.expected_slide_delta)
    try:
        expected_delta_int = int(expected_delta)
    except (TypeError, ValueError):
        expected_delta_int = fallback.expected_slide_delta
    current_turn_delta_signal = _current_turn_slide_delta_signal(user_message, intent)
    expected_delta_source = "llm" if payload_has_expected_delta else getattr(fallback, "expected_slide_delta_source", "fallback")
    if (
        fallback.expected_slide_delta != 0
        and expected_delta_int == 0
        and current_turn_delta_signal != 0
        and not payload_has_expected_delta
    ):
        expected_delta_int = fallback.expected_slide_delta
        expected_delta_source = getattr(fallback, "expected_slide_delta_source", "fallback")
    elif expected_delta_int == 0 and current_turn_delta_signal == 0:
        expected_delta_source = "llm_zero" if payload_has_expected_delta else "default_zero"
    if expected_delta_int != 0:
        operation_kind = "structural"
        scope = "global"
        rewrite_decision = dict(rewrite_decision)
        rewrite_decision["needs_controlled_rewrite"] = False
        rewrite_decision["rewrite_scope"] = "none"
        rewrite_decision_source = "structural_priority"
        if current_turn_delta_signal != 0:
            expected_delta_source = "text"
        if "structural" not in tool_groups:
            tool_groups.insert(0, "structural")
        if "local_patch" not in tool_groups:
            tool_groups.append("local_patch")
    else:
        llm_rewrite_allowed, llm_rewrite_targets = _is_controlled_rewrite_decision_allowed(
            llm_rewrite_decision,
            runtime,
            expected_slide_delta=expected_delta_int,
            fallback_targets=target_paths,
            intent=intent,
            min_confidence=0.55,
        )
        deterministic_rewrite_allowed, deterministic_rewrite_targets = _is_controlled_rewrite_decision_allowed(
            deterministic_rewrite_decision,
            runtime,
            expected_slide_delta=expected_delta_int,
            fallback_targets=target_paths,
            intent=intent,
            min_confidence=0.80,
        )
        if llm_rewrite_allowed:
            operation_kind = "controlled_rewrite"
            scope = "local"
            target_paths = llm_rewrite_targets
            tool_groups = ["controlled_rewrite", "local_patch"]
            requested = payload.get("requested_tools", [])
            if not isinstance(requested, list):
                requested = []
            for name in ("read_slide_snapshot", "write_html_file", "inspect_slide"):
                if name not in requested:
                    requested.append(name)
            payload["requested_tools"] = requested
            rewrite_decision = llm_rewrite_decision
            rewrite_decision_source = "llm"
        elif deterministic_rewrite_allowed:
            operation_kind = "controlled_rewrite"
            scope = "local"
            target_paths = deterministic_rewrite_targets
            tool_groups = ["controlled_rewrite", "local_patch"]
            requested = payload.get("requested_tools", [])
            if not isinstance(requested, list):
                requested = []
            for name in ("read_slide_snapshot", "write_html_file", "inspect_slide"):
                if name not in requested:
                    requested.append(name)
            payload["requested_tools"] = requested
            rewrite_decision = deterministic_rewrite_decision
            rewrite_decision_source = "deterministic_override"

    if operation_kind == "controlled_rewrite":
        first_steps = [
            "Call read_slide_snapshot on the target slide to get the current content_hash.",
            "Rewrite the same target slide with write_html_file(force_regenerate=true, expected_hash=content_hash).",
            "Use a scan-friendly page-level structure such as headline plus three cards/columns/section blocks; do not leave three plain paragraphs in the top-left.",
            "Preserve active Temporary preferences such as conclusion-title wording and TEMP-PREF markers.",
            "Run inspect_slide before finalize.",
        ]
        tool_groups = ["controlled_rewrite", "local_patch"]
    else:
        first_steps = (
            [str(item) for item in payload.get("first_steps", []) if str(item).strip()]
            if isinstance(payload.get("first_steps"), list)
            else []
        )

    policy = ModifyToolPolicyPlan(
        scope=scope,
        operation_kind=operation_kind,
        tool_groups=tool_groups,
        target_slide_paths=target_paths,
        expected_slide_delta=expected_delta_int,
        coverage_required=bool(payload.get("coverage_required", fallback.coverage_required)),
        first_steps=first_steps,
        risk_flags=[str(item) for item in payload.get("risk_flags", []) if str(item).strip()] if isinstance(payload.get("risk_flags"), list) else [],
        requested_tools=[str(item) for item in payload.get("requested_tools", []) if str(item).strip()] if isinstance(payload.get("requested_tools"), list) else [],
        reason=str(payload.get("reason", "") or fallback.reason),
        source="llm",
        raw_response=str(raw_response or ""),
        new_element_rule_applications=_json_safe(
            getattr(execution_plan, "new_element_rule_applications", []) or []
        ),
        expected_slide_delta_source=expected_delta_source,
        rewrite_decision=_json_safe(rewrite_decision),
        rewrite_decision_source=rewrite_decision_source,
    )
    (
        policy.prompt_artifact,
        policy.response_artifact,
        policy.payload_artifact,
    ) = write_modify_tool_policy_planner_artifacts(
        runtime,
        prompt=prompt,
        raw_response=raw_response,
        payload=payload,
        source="llm",
    )
    return clamp_policy_to_execution_plan(policy, execution_plan, source_suffix="semantic_guard")


def build_tools_for_modify_policy(
    agent: Any,
    policy: ModifyToolPolicyPlan,
) -> tuple[list[str], list[str], list[str]]:
    base_tools = _available_tool_names(agent)
    base_set = set(base_tools)
    hard_removed = _policy_hard_removed_tools(policy)

    requested = _unique_tool_names(list(policy.requested_tools or []))
    tools = list(_MODIFY_SCOPE_SHARED_TOOLS)
    tools.extend(_tools_for_policy_groups(policy.tool_groups))
    tools.extend(requested)

    operation_kind = normalize_modify_operation_kind(policy.operation_kind, scope=policy.scope)
    scope = normalize_modify_scope(policy.scope)
    system_required = list(_MODIFY_SCOPE_SHARED_TOOLS)

    if operation_kind not in {"query", "preference_update"}:
        system_required.extend(_MODIFY_TOOL_GROUPS["local_patch"])
    if scope == "global" and operation_kind not in {"query", "preference_update", "structural"}:
        system_required.extend(_MODIFY_TOOL_GROUPS["global_style"])
    if operation_kind in {"structural", "mixed"}:
        system_required.extend(_MODIFY_TOOL_GROUPS["structural"])
        system_required.extend(_MODIFY_TOOL_GROUPS["local_patch"])
    if operation_kind == "image_asset":
        system_required.extend(_MODIFY_TOOL_GROUPS["asset_discovery"])
    if operation_kind == "research_content":
        system_required.extend(_MODIFY_TOOL_GROUPS["content_research"])
    if operation_kind == "diagram_layout":
        system_required.extend(_MODIFY_TOOL_GROUPS["diagram_rewrite"])
        system_required.extend(_MODIFY_TOOL_GROUPS["structured_visual"])
    if operation_kind == "controlled_rewrite":
        system_required.extend(_MODIFY_TOOL_GROUPS["controlled_rewrite"])
        system_required.extend(_MODIFY_TOOL_GROUPS["local_patch"])

    added_by_system = [
        name for name in _unique_tool_names(system_required)
        if name not in tools
    ]
    tools.extend(added_by_system)

    active = [
        name for name in _unique_tool_names(tools)
        if name in base_set and name not in hard_removed
    ]
    if "remember_lesson" in base_set and "remember_lesson" not in active:
        active.append("remember_lesson")
    return active, added_by_system, sorted(name for name in hard_removed if name in base_set)


def capability_for_modify_policy(
    policy: ModifyToolPolicyPlan | None,
    plan: ModifyExecutionPlan | None = None,
) -> str:
    operation_kind = normalize_modify_operation_kind(
        getattr(policy, "operation_kind", None) if policy else getattr(plan, "operation_kind", None),
        scope=getattr(policy, "scope", None) if policy else getattr(plan, "scope", None),
    )
    expected_delta = int(getattr(policy, "expected_slide_delta", 0) or 0) if policy else 0
    groups = set(normalize_modify_tool_groups(getattr(policy, "tool_groups", []) if policy else []))
    if expected_delta != 0 or operation_kind == "structural":
        return "slide_structure"
    if operation_kind == "preference_update":
        return "preference_update"
    if operation_kind == "query":
        return "query_only"
    if operation_kind == "image_asset":
        return "asset_update"
    if operation_kind == "diagram_layout":
        return "diagram_layout"
    if operation_kind == "controlled_rewrite" or "controlled_rewrite" in groups:
        return "controlled_rewrite"
    if "structured_visual" in groups:
        return "chart_or_table_update"
    if operation_kind == "layout":
        return "local_layout_repair"
    if operation_kind == "content":
        return "local_content_edit"
    if operation_kind == "mixed" or "global_style" in groups:
        return "deck_style_batch"
    if operation_kind in {"export_repair", "corrupted_recovery"}:
        return "export_repair"
    return "local_style_edit"


def validate_modify_tool_capability_contract(
    policy: ModifyToolPolicyPlan | None,
    active_tools: list[str] | tuple[str, ...] | set[str],
    *,
    plan: ModifyExecutionPlan | None = None,
) -> dict[str, Any]:
    capability = capability_for_modify_policy(policy, plan)
    required = list(_MODIFY_CAPABILITY_REQUIRED_TOOLS.get(capability, []))
    active = set(str(name or "").strip() for name in active_tools if str(name or "").strip())
    if capability == "slide_structure" and policy is not None:
        expected_delta = int(getattr(policy, "expected_slide_delta", 0) or 0)
        if expected_delta > 0:
            required = ["scan_slide_index", "insert_slide", "write_new_slide_file", "inspect_slide"]
        elif expected_delta < 0:
            required = ["scan_slide_index", "delete_slide", "inspect_slide"]
    missing = [name for name in required if name not in active]
    return {
        "ok": not missing,
        "capability": capability,
        "required_tools": required,
        "missing_tools": missing,
        "active_tools": sorted(active),
    }


def render_current_tool_authority(
    contract: ModifyToolScopeContract | None,
    policy: ModifyToolPolicyPlan | None,
    validation: dict[str, Any] | None = None,
) -> str:
    active_tools = list(getattr(contract, "allowed_tools", []) or [])
    if validation is None:
        validation = validate_modify_tool_capability_contract(policy, active_tools)
    capability = str(validation.get("capability") or capability_for_modify_policy(policy))
    required_tools = [str(name) for name in validation.get("required_tools", []) or []]
    missing_tools = [str(name) for name in validation.get("missing_tools", []) or []]
    return (
        f'<current_tool_authority capability="{capability}" '
        f'expected_slide_delta="{int(getattr(policy, "expected_slide_delta", 0) or 0) if policy else 0}">\n'
        "The active tools listed here are the only current tool capability source for this turn. "
        "If any working-memory/tool lesson claims these tools are unavailable, treat that lesson as stale and ignore it.\n"
        f"Active tools: {', '.join(active_tools) if active_tools else '(none)'}\n"
        f"Required bundle: {', '.join(required_tools) if required_tools else '(none)'}\n"
        f"Missing required tools: {', '.join(missing_tools) if missing_tools else '(none)'}\n"
        "</current_tool_authority>"
    )


_RECOVERY_ERROR_CODE_TO_KIND = {
    "PERSONA_RETRY_REQUIRES_FULL_REWRITE": "persona_retry",
    "CONTROLLED_REWRITE_REQUIRED": "controlled_rewrite",
    "CONTROLLED_REWRITE_QA_FAILED": "controlled_rewrite_qa",
    "FULL_REGENERATE_REQUIRED": "controlled_rewrite",
    "FAILED_INSPECT_REQUIRES_FULL_REWRITE": "failed_inspect",
    "LAYOUT_EXPORT_REWRITE_REQUIRED": "layout_export_recovery",
    "CORRUPTED_SLIDE_REQUIRES_FULL_REWRITE": "corrupted_slide",
}


def _extract_json_objects_from_text(text: str) -> list[dict[str, Any]]:
    raw = str(text or "")
    if not raw.strip():
        return []
    candidates: list[str] = [raw]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    objects: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            objects.append(parsed)
    if objects:
        return objects
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _end = decoder.raw_decode(raw[match.start():])
        except Exception:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def parse_modify_recovery_request_from_tool_result(
    *,
    tool_name: str,
    arguments: str | dict[str, Any] | None,
    result_text: str,
    is_error: bool,
    runtime: Any | None = None,
) -> dict[str, Any] | None:
    """Extract a tool-requested recovery capability from a tool result.

    Tool failures should not rely on natural-language warnings alone. This
    parser recognizes structured error payloads plus conservative textual
    fallbacks and converts them into a policy-level recovery request.
    """
    text = str(result_text or "")
    parsed_args: dict[str, Any] = {}
    if isinstance(arguments, dict):
        parsed_args = arguments
    elif isinstance(arguments, str) and arguments.strip():
        try:
            loaded = json.loads(arguments)
            if isinstance(loaded, dict):
                parsed_args = loaded
        except Exception:
            parsed_args = {}

    payloads = _extract_json_objects_from_text(text)
    payload = next(
        (
            item
            for item in payloads
            if str(item.get("error_code", "") or "").strip()
            or str(item.get("recovery_capability", "") or "").strip()
            or str(item.get("required_tool", "") or "").strip()
        ),
        {},
    )
    error_code = str(payload.get("error_code", "") or "").strip()
    recovery_capability = str(payload.get("recovery_capability", "") or "").strip().lower()
    lowered = text.lower()
    needs_controlled_rewrite = False
    recovery_kind = ""
    if error_code in _RECOVERY_ERROR_CODE_TO_KIND:
        needs_controlled_rewrite = True
        recovery_kind = _RECOVERY_ERROR_CODE_TO_KIND[error_code]
    elif recovery_capability in {"controlled_rewrite", "controlled_full_rewrite", "full_regenerate"}:
        needs_controlled_rewrite = True
        recovery_kind = recovery_capability
    elif (
        is_error
        and "write_html_file" in lowered
        and "force_regenerate" in lowered
        and (
            "persona" in lowered
            or "failed inspect" in lowered
            or "failed `inspect_slide`" in lowered
            or "repeated layout failure" in lowered
            or "patch repair is not converging" in lowered
            or "render/export error" in lowered
            or "layout/export" in lowered
            or "full rewrite protocol" in lowered
            or "corrupted" in lowered
            or "full-regenerate" in lowered
            or "full regenerate" in lowered
        )
    ):
        needs_controlled_rewrite = True
        if "persona" in lowered:
            recovery_kind = "persona_retry"
        elif "corrupted" in lowered:
            recovery_kind = "corrupted_slide"
        elif "layout" in lowered or "render/export" in lowered:
            recovery_kind = "layout_export_recovery"
        elif "inspect" in lowered:
            recovery_kind = "failed_inspect"
        else:
            recovery_kind = "controlled_rewrite"

    if not needs_controlled_rewrite:
        return None

    raw_slide_path = (
        payload.get("slide_path")
        or payload.get("file_path")
        or payload.get("target_slide")
        or parsed_args.get("slide_path")
        or parsed_args.get("file_path")
        or parsed_args.get("html_file")
        or ""
    )
    if not raw_slide_path:
        match = re.search(
            r"(?:file_path|slide_path|html_file)\s*=\s*[\"']([^\"']+\.html)[\"']",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            raw_slide_path = match.group(1)
    target_path: Path | None = None
    if raw_slide_path:
        try:
            candidate = Path(str(raw_slide_path))
            if runtime is not None and not candidate.is_absolute():
                candidate = Path(getattr(runtime, "workspace", Path("."))) / candidate
            target_path = candidate.resolve()
        except Exception:
            target_path = None

    current_hash = str(
        payload.get("current_content_hash")
        or payload.get("content_hash")
        or payload.get("expected_hash")
        or parsed_args.get("expected_hash")
        or parsed_args.get("content_hash")
        or ""
    ).strip()
    if not current_hash:
        match = re.search(
            r"(?:expected_hash|content_hash|current_content_hash)\s*=\s*[\"']([^\"']+)[\"']",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            current_hash = match.group(1).strip()

    return {
        "capability": "controlled_rewrite",
        "operation_kind": "controlled_rewrite",
        "recovery_kind": recovery_kind or "controlled_rewrite",
        "error_code": error_code,
        "source_tool": str(tool_name or ""),
        "target_slide_path": str(target_path) if target_path is not None else "",
        "target_slide_label": str(raw_slide_path or ""),
        "current_content_hash": current_hash,
        "required_tools": ["read_slide_snapshot", "write_html_file", "inspect_slide"],
        "blocked_ops": _json_safe(payload.get("blocked_ops", [])),
        "raw_error_excerpt": text[:1200],
    }


def build_controlled_rewrite_recovery_policy(
    runtime: Any,
    current_policy: ModifyToolPolicyPlan | None,
    recovery_request: dict[str, Any],
    *,
    reason: str = "",
) -> ModifyToolPolicyPlan:
    target_paths: list[Path] = []
    raw_target = str(recovery_request.get("target_slide_path", "") or "").strip()
    if raw_target:
        try:
            target_paths.append(Path(raw_target).resolve())
        except Exception:
            pass
    if not target_paths and current_policy is not None:
        target_paths = list(getattr(current_policy, "target_slide_paths", []) or [])
    hash_text = str(recovery_request.get("current_content_hash", "") or "").strip()
    target_label = str(recovery_request.get("target_slide_label", "") or "").strip()
    first_steps = [
        "Use read_slide_snapshot on the recovery target slide to obtain the latest content_hash.",
        "Rewrite only that target slide with write_html_file(file_path=target, content=<complete replacement HTML>, force_regenerate=true, expected_hash=content_hash).",
        "Preserve the user's requested content/layout constraints and active deck preferences while rebalancing the page as one coherent HTML document.",
        "Run inspect_slide on the rewritten target slide before finalize.",
    ]
    if hash_text:
        first_steps[0] = (
            f"The failed tool reported current_content_hash={hash_text}; still prefer read_slide_snapshot "
            "if any file may have changed before writing."
        )
    return ModifyToolPolicyPlan(
        scope="local",
        operation_kind="controlled_rewrite",
        tool_groups=["controlled_rewrite", "local_patch"],
        target_slide_paths=target_paths,
        expected_slide_delta=int(getattr(current_policy, "expected_slide_delta", 0) or 0)
        if current_policy is not None
        else 0,
        coverage_required=bool(getattr(current_policy, "coverage_required", False))
        if current_policy is not None
        else bool(target_paths),
        first_steps=first_steps,
        risk_flags=[
            "controlled_full_rewrite_required",
            str(recovery_request.get("recovery_kind", "") or "tool_requested_recovery"),
        ],
        requested_tools=["read_slide_snapshot", "write_html_file", "inspect_slide"],
        reason=(
            reason
            or "A tool returned a structured recovery request that requires controlled full-slide rewrite."
        )
        + (f" Target: {target_label}." if target_label else ""),
        source="runtime_recovery",
        raw_response=json.dumps(_json_safe(recovery_request), ensure_ascii=False),
        new_element_rule_applications=_json_safe(
            getattr(current_policy, "new_element_rule_applications", []) or []
        )
        if current_policy is not None
        else [],
        expected_slide_delta_source=str(
            getattr(current_policy, "expected_slide_delta_source", "runtime_recovery")
            if current_policy is not None
            else "runtime_recovery"
        ),
    )


def build_controlled_rewrite_recovery_followup(
    runtime: Any,
    recovery_request: dict[str, Any],
    policy: ModifyToolPolicyPlan,
) -> str:
    target_paths = list(getattr(policy, "target_slide_paths", []) or [])
    target_label = (
        workspace_relative_label(runtime, target_paths[0])
        if target_paths
        else str(recovery_request.get("target_slide_label", "") or "(target slide)")
    )
    hash_text = str(recovery_request.get("current_content_hash", "") or "").strip()
    hash_guidance = (
        f"The failed tool reported current_content_hash `{hash_text}`, but re-read the snapshot if the file may have changed. "
        if hash_text
        else ""
    )
    return (
        "SYSTEM: The previous tool returned a structured recovery request. "
        "The current patch-only path is no longer sufficient for this slide. "
        f"Target slide: {target_label}. "
        f"Recovery kind: {str(recovery_request.get('recovery_kind', '') or 'controlled_rewrite')}. "
        f"{hash_guidance}"
        "Use `read_slide_snapshot` on that target to obtain the latest `content_hash`, then call "
        "`write_html_file(file_path=<same target>, content=<complete replacement HTML>, "
        "force_regenerate=true, expected_hash=<content_hash>)`. "
        "Rewrite only this target slide, keep the requested content/layout constraints, do not insert/delete slides, "
        "then run `inspect_slide` before `finalize`."
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _modify_policy_history_dir(runtime: Any) -> Path:
    history_dir = runtime.workspace / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def _modify_policy_artifact_dir(runtime: Any) -> Path:
    artifact_dir = _modify_policy_history_dir(runtime) / "modify_tool_policy"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _current_modify_turn(runtime: Any) -> int:
    try:
        return int(getattr(runtime, "_modify_turn_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _next_modify_policy_event_id(runtime: Any) -> int:
    current = int(getattr(runtime, "_modify_tool_policy_event_id", 0) or 0) + 1
    try:
        setattr(runtime, "_modify_tool_policy_event_id", current)
    except Exception:
        pass
    return current


def _runtime_relative_artifact(runtime: Any, path: Path) -> str:
    try:
        return path.resolve().relative_to(runtime.workspace.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _write_modify_policy_text_artifact(
    runtime: Any,
    *,
    label: str,
    text: str,
    suffix: str = "txt",
) -> str:
    try:
        turn = _current_modify_turn(runtime)
        event_id = _next_modify_policy_event_id(runtime)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "event")).strip("_") or "event"
        suffix = suffix.lstrip(".") or "txt"
        path = _modify_policy_artifact_dir(runtime) / f"turn_{turn:02d}_event_{event_id:03d}_{safe_label}.{suffix}"
        path.write_text(str(text or ""), encoding="utf-8")
        return _runtime_relative_artifact(runtime, path)
    except Exception:
        logger.debug("Failed to write modify tool policy text artifact", exc_info=True)
        return ""


def _modify_policy_plan_payload(policy: ModifyToolPolicyPlan | None) -> dict[str, Any]:
    if policy is None:
        return {}
    return {
        "scope": policy.scope,
        "operation_kind": policy.operation_kind,
        "tool_groups": list(policy.tool_groups or []),
        "target_slide_paths": [str(path.resolve()) for path in policy.target_slide_paths or []],
        "expected_slide_delta": policy.expected_slide_delta,
        "expected_slide_delta_source": getattr(policy, "expected_slide_delta_source", ""),
        "rewrite_decision": _json_safe(getattr(policy, "rewrite_decision", None) or {}),
        "rewrite_decision_source": str(getattr(policy, "rewrite_decision_source", "") or ""),
        "coverage_required": policy.coverage_required,
        "first_steps": list(policy.first_steps or []),
        "risk_flags": list(policy.risk_flags or []),
        "requested_tools": list(policy.requested_tools or []),
        "reason": policy.reason,
        "source": policy.source,
        "raw_response": policy.raw_response[:4000],
        "prompt_artifact": policy.prompt_artifact,
        "response_artifact": policy.response_artifact,
        "payload_artifact": policy.payload_artifact,
        "new_element_rule_applications": _json_safe(policy.new_element_rule_applications or []),
    }


def write_modify_tool_policy_planner_artifacts(
    runtime: Any,
    *,
    prompt: str,
    raw_response: str,
    payload: dict[str, Any] | None = None,
    source: str,
) -> tuple[str, str, str]:
    """Persist planner IO so a single experiment can be audited after the run."""
    try:
        turn = _current_modify_turn(runtime)
        base_dir = _modify_policy_artifact_dir(runtime)
        event_id = _next_modify_policy_event_id(runtime)
        prefix = f"turn_{turn:02d}_planner_{event_id:03d}_{source}"
        prompt_path = base_dir / f"{prefix}_prompt.txt"
        response_path = base_dir / f"{prefix}_response.txt"
        payload_path = base_dir / f"{prefix}_payload.json"
        prompt_path.write_text(prompt or "", encoding="utf-8")
        response_path.write_text(raw_response or "", encoding="utf-8")
        payload_path.write_text(
            json.dumps(_json_safe(payload or {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (
            _runtime_relative_artifact(runtime, prompt_path),
            _runtime_relative_artifact(runtime, response_path),
            _runtime_relative_artifact(runtime, payload_path),
        )
    except Exception:
        logger.debug("Failed to write modify tool policy planner artifacts", exc_info=True)
        return "", "", ""


def write_preference_semantic_artifacts(
    runtime: Any,
    *,
    prompt: str,
    raw_response: str,
    payload: dict[str, Any] | None,
    stage: str,
    source: str,
) -> tuple[str, str, str]:
    try:
        turn = _current_modify_turn(runtime)
        base_dir = _modify_policy_artifact_dir(runtime) / "preference_semantics"
        base_dir.mkdir(parents=True, exist_ok=True)
        event_id = _next_modify_policy_event_id(runtime)
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage or "stage")).strip("_") or "stage"
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source or "unknown")).strip("_") or "unknown"
        prefix = f"turn_{turn:02d}_{event_id:03d}_{safe_stage}_{safe_source}"
        prompt_path = base_dir / f"{prefix}_prompt.txt"
        response_path = base_dir / f"{prefix}_response.txt"
        payload_path = base_dir / f"{prefix}_payload.json"
        prompt_path.write_text(prompt or "", encoding="utf-8")
        response_path.write_text(raw_response or "", encoding="utf-8")
        payload_path.write_text(
            json.dumps(_json_safe(payload or {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (
            _runtime_relative_artifact(runtime, prompt_path),
            _runtime_relative_artifact(runtime, response_path),
            _runtime_relative_artifact(runtime, payload_path),
        )
    except Exception:
        logger.debug("Failed to write preference semantic artifacts", exc_info=True)
        return "", "", ""


def append_preference_semantic_trace(
    runtime: Any,
    compilation: PreferenceSemanticCompilation,
    *,
    user_message: str,
    intent: dict[str, Any] | None = None,
    critic_payload: dict[str, Any] | None = None,
    note: str = "",
) -> str:
    try:
        history_dir = _modify_policy_history_dir(runtime)
        payload = {
            "timestamp": _utc_timestamp(),
            "turn": _current_modify_turn(runtime),
            "event": "preference_semantic_compilation",
            "user_message_excerpt": str(user_message or "")[:500],
            "intent": _json_safe(intent or {}),
            "memory_flow": compilation.memory_flow,
            "valid": compilation.valid,
            "fail_closed": compilation.fail_closed,
            "reason": compilation.reason,
            "source": compilation.source,
            "payload": _json_safe(compilation.payload),
            "payloads": _json_safe(compilation.payloads or []),
            "critic": _json_safe(critic_payload or {}),
            "artifacts": {
                "compiler_prompt": compilation.compiler_prompt_artifact,
                "compiler_response": compilation.compiler_response_artifact,
                "compiler_payload": compilation.compiler_payload_artifact,
                "critic_prompt": compilation.critic_prompt_artifact,
                "critic_response": compilation.critic_response_artifact,
                "critic_payload": compilation.critic_payload_artifact,
            },
            "note": note,
        }
        trace_path = history_dir / "preference_semantic_trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return _runtime_relative_artifact(runtime, trace_path)
    except Exception:
        logger.debug("Failed to append preference semantic trace", exc_info=True)
        return ""


def append_preference_semantic_stage_trace(
    runtime: Any,
    *,
    event: str,
    user_message: str,
    intent: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        history_dir = _modify_policy_history_dir(runtime)
        payload = {
            "timestamp": _utc_timestamp(),
            "turn": _current_modify_turn(runtime),
            "event": str(event or "preference_semantic_stage"),
            "user_message_excerpt": str(user_message or "")[:500],
            "intent": _json_safe(intent or {}),
            "extra": _json_safe(extra or {}),
        }
        trace_path = history_dir / "preference_semantic_trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to append preference semantic stage trace", exc_info=True)


def append_modify_tool_policy_trace(
    runtime: Any,
    *,
    event: str,
    plan: ModifyExecutionPlan | None = None,
    policy: ModifyToolPolicyPlan | None = None,
    allowed_tools: list[str] | None = None,
    removed_tools: list[str] | None = None,
    system_added_tools: list[str] | None = None,
    hard_removed_tools: list[str] | None = None,
    base_tools: list[str] | None = None,
    user_message: str = "",
    intent: dict[str, Any] | None = None,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        history_dir = _modify_policy_history_dir(runtime)
        payload = {
            "timestamp": _utc_timestamp(),
            "turn": _current_modify_turn(runtime),
            "event": str(event or "policy"),
            "user_message_excerpt": str(user_message or "")[:500],
            "intent": _json_safe(intent or {}),
            "execution_plan": {
                "scope": getattr(plan, "scope", "") if plan else "",
                "operation_kind": getattr(plan, "operation_kind", "") if plan else "",
                "reason": getattr(plan, "reason", "") if plan else "",
                "target_slide_paths": [
                    str(path.resolve()) for path in getattr(plan, "target_slide_paths", []) or []
                ],
                "target_rule_ids": list(getattr(plan, "target_rule_ids", []) or []),
                "applicable_rule_ids": list(getattr(plan, "applicable_rule_ids", []) or []),
                "new_element_rule_applications": _json_safe(
                    getattr(plan, "new_element_rule_applications", []) or []
                ),
                "diagram_contract": _json_safe(getattr(plan, "diagram_contract", None) or {}),
                "rewrite_decision": _json_safe(getattr(plan, "rewrite_decision", None) or {}),
                "rewrite_decision_source": str(getattr(plan, "rewrite_decision_source", "") or ""),
                "coverage_required": bool(getattr(plan, "coverage_required", False)) if plan else False,
            },
            "policy": _modify_policy_plan_payload(policy),
            "base_tools": list(base_tools or []),
            "allowed_tools": list(allowed_tools or []),
            "removed_tools": list(removed_tools or []),
            "system_added_tools": list(system_added_tools or []),
            "hard_removed_tools": list(hard_removed_tools or []),
            "note": note,
            "extra": _json_safe(extra or {}),
        }
        trace_path = history_dir / "modify_tool_policy_trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to append modify tool policy trace", exc_info=True)


def write_modify_tool_scope_artifact(
    runtime: Any,
    plan: ModifyExecutionPlan | None,
    allowed_tools: list[str],
    removed_tools: list[str],
    policy: ModifyToolPolicyPlan | None = None,
    system_added_tools: list[str] | None = None,
    hard_removed_tools: list[str] | None = None,
) -> None:
    try:
        history_dir = runtime.workspace / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": getattr(plan, "scope", "") if plan else "",
            "operation_kind": getattr(plan, "operation_kind", "") if plan else "",
            "reason": getattr(plan, "reason", "") if plan else "",
            "allowed_tools": allowed_tools,
            "removed_tools": removed_tools,
            "target_slide_paths": [
                str(path.resolve()) for path in getattr(plan, "target_slide_paths", []) or []
            ],
            "target_rule_ids": list(getattr(plan, "target_rule_ids", []) or []),
            "applicable_rule_ids": list(getattr(plan, "applicable_rule_ids", []) or []),
            "new_element_rule_applications": _json_safe(
                getattr(plan, "new_element_rule_applications", []) or []
            ),
            "diagram_contract": _json_safe(getattr(plan, "diagram_contract", None) or {}),
            "rewrite_decision": _json_safe(getattr(plan, "rewrite_decision", None) or {}),
            "rewrite_decision_source": str(getattr(plan, "rewrite_decision_source", "") or ""),
            "coverage_required": bool(getattr(plan, "coverage_required", False)) if plan else False,
        }
        if policy is not None:
            payload["policy"] = _modify_policy_plan_payload(policy)
            payload["system_added_tools"] = list(system_added_tools or [])
            payload["hard_removed_tools"] = list(hard_removed_tools or [])
        (history_dir / "tool_scope.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (history_dir / "tool_policy.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Failed to write modify tool scope artifact", exc_info=True)


def apply_modify_tool_policy(
    runtime: Any,
    agent: Any,
    plan: ModifyExecutionPlan | None,
    policy: ModifyToolPolicyPlan,
    *,
    event: str = "policy_applied",
    user_message: str = "",
    intent: dict[str, Any] | None = None,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> ModifyToolScopeContract:
    restored_tools = restore_agent_base_tools(agent)
    before_tools = _available_tool_names(agent)
    allowed_tools, system_added, hard_removed = build_tools_for_modify_policy(agent, policy)
    if hasattr(agent, "set_tools_by_names"):
        active_tools = agent.set_tools_by_names(allowed_tools)
    else:
        active_tools = [
            tool_name
            for tool_name in allowed_tools
            if tool_name in getattr(getattr(agent, "agent_env", None), "_tools_dict", {})
        ]
    removed_tools = sorted(set(name for name in before_tools if name) - set(active_tools))
    write_modify_tool_scope_artifact(
        runtime,
        plan,
        active_tools,
        removed_tools,
        policy=policy,
        system_added_tools=system_added,
        hard_removed_tools=hard_removed,
    )
    append_modify_tool_policy_trace(
        runtime,
        event=event,
        plan=plan,
        policy=policy,
        allowed_tools=active_tools,
        removed_tools=removed_tools,
        system_added_tools=system_added,
        hard_removed_tools=hard_removed,
        base_tools=restored_tools or before_tools,
        user_message=user_message,
        intent=intent,
        note=note,
        extra=extra,
    )
    return ModifyToolScopeContract(
        scope=normalize_modify_scope(policy.scope),
        operation_kind=normalize_modify_operation_kind(policy.operation_kind, scope=policy.scope),
        allowed_tools=active_tools,
        removed_tools=removed_tools,
        reason=policy.reason or (getattr(plan, "reason", "") if plan else ""),
        tool_groups=list(policy.tool_groups or []),
        policy_source=policy.source,
        system_added_tools=system_added,
        hard_removed_tools=hard_removed,
    )


def apply_modify_tool_scope(
    runtime: Any,
    agent: Any,
    plan: ModifyExecutionPlan | None,
) -> ModifyToolScopeContract:
    policy = build_fallback_modify_tool_policy_plan(
        runtime,
        user_message="",
        intent=None,
        execution_plan=plan,
        reason=getattr(plan, "reason", "") if plan else "",
    )
    return apply_modify_tool_policy(runtime, agent, plan, policy)


def apply_modify_tool_scope_legacy(
    runtime: Any,
    agent: Any,
    plan: ModifyExecutionPlan | None,
) -> ModifyToolScopeContract:
    raw_scope = getattr(plan, "scope", None)
    scope = normalize_modify_scope(raw_scope)
    operation_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=raw_scope,
    )
    allowed_tools = build_modify_tool_allowlist(scope, operation_kind)
    before_tools = [
        str(tool.get("function", {}).get("name", "") or "").strip()
        for tool in getattr(agent, "tools", [])
        if isinstance(tool, dict)
    ]
    if hasattr(agent, "set_tools_by_names"):
        active_tools = agent.set_tools_by_names(allowed_tools)
    else:
        active_tools = [
            tool_name
            for tool_name in allowed_tools
            if tool_name in getattr(getattr(agent, "agent_env", None), "_tools_dict", {})
        ]
    removed_tools = sorted(
        set(name for name in before_tools if name)
        - set(active_tools)
    )
    write_modify_tool_scope_artifact(runtime, plan, active_tools, removed_tools)
    return ModifyToolScopeContract(
        scope=scope,
        operation_kind=operation_kind,
        allowed_tools=active_tools,
        removed_tools=removed_tools,
        reason=getattr(plan, "reason", "") if plan else "",
    )


@dataclass
class IntentResolutionResult:
    intent: str
    source: str
    scenario: str = ""
    confidence: float | None = None
    raw_response: str = ""


def confidence_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def workspace_relative_label(runtime: Any, path: Path | str) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(runtime.workspace.resolve()).as_posix()
    except Exception:
        return target.as_posix()


def resolve_workspace_context_slide_dir(
    runtime: Any,
    slide_dir: Path | str | None = None,
) -> Path | None:
    raw_dir = slide_dir
    if raw_dir is None:
        raw_dir = runtime.intermediate_output.get("slide_html_dir")
    if not raw_dir:
        return None
    resolved = Path(raw_dir)
    if not resolved.is_absolute():
        resolved = runtime.workspace / resolved
    return resolved.resolve()


def prime_design_slide_output_dir(
    runtime: Any,
    manuscript_path: Path | str | None = None,
) -> Path:
    active_dir = resolve_workspace_context_slide_dir(runtime)
    if active_dir is None:
        manuscript_dir = None
        if manuscript_path is not None:
            manuscript_dir = Path(manuscript_path)
            if not manuscript_dir.is_absolute():
                manuscript_dir = runtime.workspace / manuscript_dir
            manuscript_dir = manuscript_dir.resolve().parent

        if manuscript_dir is not None and manuscript_dir.name.lower() == "outputs":
            active_dir = manuscript_dir
        else:
            active_dir = (runtime.workspace / "outputs").resolve()

    active_dir.mkdir(parents=True, exist_ok=True)
    runtime.intermediate_output["slide_html_dir"] = str(active_dir)
    runtime.save_results()
    return active_dir


def build_workspace_context_block(
    runtime: Any,
    *,
    slide_dir: Path | str | None = None,
    manuscript_path: Path | str | None = None,
    include_active_slide_dir: bool = False,
) -> str:
    ws_context_parts: list[str] = []

    attach_dir = runtime.workspace / "attachments"
    if attach_dir.exists():
        files = sorted(attach_dir.iterdir())
        if files:
            file_list = ", ".join(f.name for f in files[:30])
            ws_context_parts.append(
                f"Available attachments ({len(files)} files, path: attachments/): {file_list}"
            )

    resolved_slide_dir = resolve_workspace_context_slide_dir(runtime, slide_dir)
    if resolved_slide_dir is not None:
        slide_rel = workspace_relative_label(runtime, resolved_slide_dir)
        slides = sorted(resolved_slide_dir.glob("*.html")) if resolved_slide_dir.exists() else []
        if slides:
            slide_list = ", ".join(s.name for s in slides)
            ws_context_parts.append(
                f"Current slides ({len(slides)} files, path: {slide_rel}/): {slide_list}\n"
                f"  Use `read_file(\"{slide_rel}/slide_01.html\")` or `inspect_slide(\"{slide_rel}/slide_01.html\")` to access."
            )
        elif include_active_slide_dir:
            ws_context_parts.append(
                f"Active slide output directory: {slide_rel}/\n"
                f"  Write new slides with `write_html_file(\"{slide_rel}/slide_01.html\", ...)` and do not invent another slide directory."
            )

    if manuscript_path is not None:
        manuscript = Path(manuscript_path)
        if not manuscript.is_absolute():
            manuscript = runtime.workspace / manuscript
        ws_context_parts.append(
            f"Incoming manuscript (read-only): {workspace_relative_label(runtime, manuscript.resolve())}"
        )

    design_plan_rel = _find_existing_design_plan_rel(runtime.workspace)
    if design_plan_rel:
        ws_context_parts.append(f"DeckDesigner plan: {design_plan_rel}")
    else:
        ws_context_parts.append(
            "DeckDesigner plan: MISSING\n"
            "  No `design_plan.md` was found in workspace root or outputs/."
        )

    converted_dir = runtime.workspace / "converted"
    if converted_dir.exists():
        subdirs = [d.name for d in sorted(converted_dir.iterdir()) if d.is_dir()]
        if subdirs:
            ws_context_parts.append(
                f"Converted documents: {', '.join(subdirs[:10])}\n"
                f"  Use `list_document_figures(\"converted/{subdirs[0]}\")` to view extracted figures."
            )

    if not ws_context_parts:
        return ""

    return (
        "<workspace_context>\n"
        + "\n".join(ws_context_parts)
        + f"\nWorkspace root: {runtime.workspace}"
        + "\nAll tool paths are relative to workspace root. Use `list_files(\".\")` to explore."
        + "\n</workspace_context>"
    )


def get_request_extra_info(request: InputRequest) -> dict[str, Any]:
    if request.extra_info is None:
        request.extra_info = {}
    return request.extra_info


def get_request_core_persona(request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    for key in ("core_persona", "user_persona", "user_role"):
        value = _normalize_text_flag(extra_info.get(key, ""))
        if value:
            return value
    return ""


def get_request_task_intent(runtime: Any, request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    for key in ("resolved_task_intent", "task_intent", "role_intent_id", "role_intent", "intent"):
        candidate = _normalize_memory_intent(extra_info.get(key, ""))
        if candidate:
            return candidate
    return runtime._get_runtime_request_intent(request)


def get_request_memory_read_intent(runtime: Any, request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    for key in (
        "resolved_memory_read_intent",
        "memory_read_intent",
        "memory_bucket_id",
        "bucket_id",
        "profile_bucket_id",
        "memory_intent",
        "resolved_memory_intent",
    ):
        candidate = _normalize_memory_intent(extra_info.get(key, ""))
        if candidate:
            return candidate
    explicit_request_intent = _normalize_memory_intent(request.memory_intent)
    if explicit_request_intent:
        return explicit_request_intent
    return runtime._get_runtime_request_intent(request)


def get_request_memory_write_intent(runtime: Any, request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    for key in (
        "resolved_memory_write_intent",
        "memory_write_intent",
        "memory_bucket_id",
        "bucket_id",
        "profile_bucket_id",
        "write_intent",
    ):
        candidate = _normalize_memory_intent(extra_info.get(key, ""))
        if candidate:
            return candidate
    return get_request_task_intent(runtime, request)


def should_freeze_preference_writeback(
    runtime: Any,
    request: InputRequest | None = None,
) -> bool:
    candidate = request or getattr(runtime, "_last_request", None)
    if not candidate:
        return False
    extra_info = candidate.extra_info or {}
    return _normalize_bool_flag(extra_info.get("freeze_preference_writeback", False))


def get_intent_resolution_mode(request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    mode = str(extra_info.get("intent_resolution_mode", "explicit_first") or "").strip().lower()
    if mode in {"explicit_first", "llm_first", "keyword_only"}:
        return mode
    return "explicit_first"


def get_explicit_extra_info_intent(request: InputRequest) -> str:
    extra_info = get_request_extra_info(request)
    for key in (
        "resolved_memory_read_intent",
        "memory_read_intent",
        "memory_bucket_id",
        "bucket_id",
        "profile_bucket_id",
        "resolved_memory_intent",
        "memory_intent",
        "resolved_task_intent",
        "task_intent",
        "role_intent_id",
        "role_intent",
        "intent",
    ):
        candidate = _normalize_memory_intent(extra_info.get(key, ""))
        if candidate:
            return candidate
    return ""


def resolve_all_slide_paths(runtime: Any) -> list[Path]:
    """Return all slide HTML files in sorted order."""
    for slide_dir in candidate_slide_dirs(runtime):
        if slide_dir.exists() and slide_dir.is_dir():
            slide_paths = sorted(slide_dir.glob("slide_*.html"))
            if slide_paths:
                return slide_paths
    return []


def candidate_slide_dirs(
    runtime: Any,
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
            path = runtime.workspace / path
        if path.suffix:
            candidate_dirs.append(path.parent)
        candidate_dirs.append(path)

    _add(slide_dir)
    _add(runtime.intermediate_output.get("slide_html_dir"))
    candidate_dirs.extend(
        runtime.workspace / rel for rel in ("outputs", "outputs/slides", "slides")
    )
    if include_workspace_root:
        candidate_dirs.append(runtime.workspace)

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


def collect_exportable_slide_paths(slide_dir: Path) -> list[Path]:
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


def exportable_slide_dir_score(
    slide_paths: list[Path],
    candidate_index: int,
) -> tuple[int, float, int, int]:
    """Score an exportable slide directory."""
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


def resolve_exportable_slide_dir(runtime: Any, slide_dir: Path | str | None = None) -> Path | None:
    """Return the best candidate directory that contains exportable slides."""
    best_candidate: Path | None = None
    best_score: tuple[int, float, int, int] | None = None

    for idx, candidate in enumerate(candidate_slide_dirs(runtime, slide_dir)):
        slide_paths = collect_exportable_slide_paths(candidate)
        if not slide_paths:
            continue

        score = exportable_slide_dir_score(slide_paths, idx)
        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate


def resolve_exportable_slide_paths(
    runtime: Any,
    slide_dir: Path | str | None = None,
) -> list[Path]:
    """Return visible slide HTMLs in numeric order."""
    resolved_dir = resolve_exportable_slide_dir(runtime, slide_dir)
    if resolved_dir is None:
        return []
    return collect_exportable_slide_paths(resolved_dir)


def session_preference_has_general_signal(user_message: str) -> bool:
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False
    patterns = (
        r"(所有|全部|整套|全局|统一).{0,12}(页|页面|标题|配色|颜色|字体|版式|风格|正文)",
        r"(以后|未来|后续|后面|之后|新增页|新加页|新增幻灯片|新生成|新建|新插入|future).{0,20}(继续|沿用|保持|都用|都要|都保持|都为|均为|设为|改成|使用|应用|继承|默认|遵守|生效)",
        r"(每次|总是|一直|默认).{0,16}(都|用|保持|遵守)",
        r"(来源|页数|格式|约束).{0,16}(必须|只能|不要|禁止|严格)",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def infer_session_rule_element_kind(*texts: str) -> str:
    haystack = " ".join(str(text or "") for text in texts).lower()
    if any(keyword in haystack for keyword in ("标题", "title", "h1", "heading")):
        return "slide_title"
    if any(keyword in haystack for keyword in ("pill tag", "pill_tag", "pill-tag", "badge", "chip", "标签", "圆角矩形标签")):
        return "pill_tag"
    if any(keyword in haystack for keyword in ("正文", "bullet", "body", "段落", "要点")):
        return "body_text"
    if any(keyword in haystack for keyword in ("页脚", "footer")):
        return "footer"
    if any(keyword in haystack for keyword in ("图片", "image", "图像")):
        return "image"
    return "deck"


def request_mentions_text_color_without_background(user_message: str) -> bool:
    """Detect requests that mean text color, not theme/background color."""
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False

    text_target_terms = (
        "字体",
        "文字",
        "文本",
        "标题",
        "正文",
        "字色",
        "font",
        "text",
        "title",
        "heading",
        "body",
    )
    color_terms = (
        "颜色",
        "color",
        "蓝",
        "红",
        "黑",
        "白",
        "灰",
        "绿",
        "紫",
        "黄",
        "blue",
        "red",
        "black",
        "white",
        "gray",
        "grey",
        "green",
        "purple",
        "yellow",
    )
    background_terms = (
        "背景",
        "底色",
        "底板",
        "色块",
        "高亮",
        "填充",
        "background",
        "surface",
        "highlight",
        "fill",
    )
    mentions_text_target = any(term in normalized for term in text_target_terms)
    mentions_color = any(term in normalized for term in color_terms) or bool(
        re.search(r"#[0-9a-f]{3,8}\b", normalized)
    )
    mentions_background = any(term in normalized for term in background_terms)
    if not mentions_text_target or not mentions_color or mentions_background:
        return False
    return True


_EXPLICIT_COLOR_RE = re.compile(
    r"(?i)(?:#[0-9a-f]{3}(?:[0-9a-f]{3})?|rgba?\(\s*[^)]+\))"
)
_EXPLICIT_BACKGROUND_CONTEXT_RE = re.compile(
    r"(?i)(?:background(?:-color)?|bg|fill|surface|highlight|背景|底色|填充|高亮|色块)"
)
_EXPLICIT_TEXT_CONTEXT_RE = re.compile(
    r"(?i)(?:color|text|font|foreground|文字|文本|字体|字色|前景|标题|正文|图注|说明)"
)

_FUZZY_DESIGN_TOKEN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("color.intent", ("深绿色", "墨绿", "dark green", "deep green")),
    ("color.intent", ("蓝色", "blue")),
    ("color.intent", ("白色", "white")),
    ("color.intent", ("紫色", "purple")),
    ("color.intent", ("红色", "red")),
    ("color.intent", ("灰色", "gray", "grey")),
    ("tone.intent", ("学术", "academic")),
    ("tone.intent", ("克制", "谨慎", "保守", "cautious", "conservative")),
    ("color.saturation", ("低饱和", "low saturation", "muted")),
    ("color.temperature", ("柔和", "soft")),
)


def _near_text(text: str, start: int, end: int, *, window: int = 32) -> str:
    return text[max(0, start - window) : min(len(text), end + window)]


def _explicit_color_property_for_context(context: str) -> str:
    if _EXPLICIT_BACKGROUND_CONTEXT_RE.search(context):
        return "background-color"
    if _EXPLICIT_TEXT_CONTEXT_RE.search(context):
        return "color"
    return "color"


def extract_explicit_css_values_from_text(*texts: str) -> dict[str, str]:
    """Extract only literal CSS-like values that are safe to hard-check later."""
    css_values: dict[str, str] = {}
    for raw_text in texts:
        text = str(raw_text or "")
        if not text:
            continue

        for match in re.finditer(
            r"(?i)(background-color|background|bg-color|bg|fill|color|font-style|font-weight|text-decoration)\s*[:=：]\s*([^,，;；。)\]\s]+(?:\([^)]*\))?)",
            text,
        ):
            prop = normalize_css_property_name(match.group(1))
            value = str(match.group(2) or "").strip().rstrip("。；;，,")
            if not prop or not value:
                continue
            if prop in {"color", "background-color", "background"}:
                if _EXPLICIT_COLOR_RE.fullmatch(value):
                    css_values["background-color" if prop == "background" else prop] = value
                continue
            normalized = normalize_css_value(prop, value)
            if prop == "font-style" and normalized in {"italic", "oblique"}:
                css_values[prop] = value
            elif prop == "font-weight" and (normalized in {"400", "700"} or re.fullmatch(r"\d{3}", normalized)):
                css_values[prop] = value
            elif prop == "text-decoration" and normalized in {"underline", "overline", "line-through", "none"}:
                css_values[prop] = value

        for match in _EXPLICIT_COLOR_RE.finditer(text):
            token = match.group(0).strip()
            context = _near_text(text, match.start(), match.end())
            prop = _explicit_color_property_for_context(context)
            css_values.setdefault(prop, token)

        lowered = text.lower()
        if re.search(r"(?i)\bitalic\b|斜体", text):
            css_values.setdefault("font-style", "italic")
        if re.search(r"(?i)\bbold\b|加粗|粗体", text):
            css_values.setdefault("font-weight", "700")
        if re.search(r"(?i)\bunderline\b|下划线", text):
            css_values.setdefault("text-decoration", "underline")
        if re.search(r"(?i)\bline-through\b|删除线", lowered):
            css_values.setdefault("text-decoration", "line-through")
    return css_values


def extract_soft_design_tokens_from_text(*texts: str) -> dict[str, str]:
    """Capture fuzzy personalization intent without making it a hard verifier input."""
    haystack = " ".join(str(text or "") for text in texts).lower()
    tokens: dict[str, str] = {}
    if not haystack:
        return tokens
    for token_key, patterns in _FUZZY_DESIGN_TOKEN_PATTERNS:
        for pattern in patterns:
            if pattern.lower() in haystack:
                aliases = {
                    "深绿色": "deep_green",
                    "墨绿": "deep_green",
                    "dark green": "deep_green",
                    "deep green": "deep_green",
                    "蓝色": "blue",
                    "白色": "white",
                    "紫色": "purple",
                    "红色": "red",
                    "灰色": "gray",
                    "学术": "academic",
                    "克制": "cautious",
                    "谨慎": "cautious",
                    "保守": "conservative",
                    "低饱和": "muted",
                    "柔和": "soft",
                }
                token_value = aliases.get(pattern.lower(), pattern.lower().replace(" ", "_"))
                tokens.setdefault(token_key, token_value)
                break
    return tokens


def intent_value_mentions_structural_slide_op(intent: dict[str, Any] | None) -> bool:
    if not intent:
        return False
    intent_type = str(intent.get("intent_type", intent.get("type", "")) or "").lower()
    action = str(
        intent.get("action", intent.get("op", intent.get("operation", ""))) or ""
    ).lower()
    element_type = str(intent.get("element_type", "") or "").lower()
    structural_text = f"{intent_type} {action}"
    if any(token in structural_text for token in ("insert", "delete", "move", "reorder", "add_slide", "new_slide", "remove_slide")):
        return True
    if any(token in element_type for token in ("slide_structure", "deck_structure")):
        return True
    return False


def is_structural_slide_operation(
    user_message: str,
    intent: dict[str, Any] | None,
) -> bool:
    if intent_value_mentions_structural_slide_op(intent):
        return True

    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False

    if re.search(r"(后续|以后|未来|默认).{0,10}(新增页|新增页面|新加页|新页面)", normalized):
        if not re.search(r"(在|于).{0,12}(前|后|之前|之后|结尾|最后|第\d+页)", normalized):
            return False

    structural_patterns = (
        r"(插入|新增一页|新增一张|添加一页|添加一张|增加一页|增加一张|补一页|加一页|append a slide|insert a slide|add a slide)",
        r"(在|于).{0,12}(前|后|之前|之后|结尾|最后|最后一页|结论页|第\d+页).{0,10}(插入|新增|添加|增加|补充|append|insert|add)",
        r"(删除|删掉|移除|去掉|remove|delete).{0,10}(第\d+页|该页|这页|这一页|页面|页|slide)",
        r"(移动|挪到|调到|放到|移到|重排|调整顺序|重新排序|move|reorder).{0,20}(第\d+页|前面|后面|最后|开头|页面|页|slide)",
    )
    return any(re.search(pattern, normalized) for pattern in structural_patterns)


def resolve_slide_path(runtime: Any, target_slide: str) -> Path | None:
    """Resolve target_slide to an actual slide path in the current slide dir."""
    for slide_dir in candidate_slide_dirs(runtime):
        exact = slide_dir / f"{target_slide}.html"
        if exact.exists():
            return exact
        m = re.search(r"(\d+)$", target_slide)
        if m:
            num = int(m.group(1))
            padded = target_slide[:m.start()] + f"{num:02d}"
            padded_path = slide_dir / f"{padded}.html"
            if padded_path.exists():
                return padded_path
    return None


def extract_tag_block(text: str, tag_name: str) -> str:
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


def parse_rule_specs_from_injection(text: str) -> list[dict[str, Any]]:
    """Parse JSONL rule specs from <working_memory_rule_specs> injection text."""
    source = str(text or "")
    blocks = re.findall(
        r"<working_memory_rule_specs\b[^>]*>(.*?)</working_memory_rule_specs>",
        source,
        flags=re.DOTALL,
    )
    if not blocks:
        inner = extract_tag_block(source, "working_memory_rule_specs")
        blocks = [inner] if inner else []
    if not blocks:
        return []
    specs: list[dict[str, Any]] = []
    for inner in blocks:
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


def format_rule_specs_text(
    rule_specs: list[dict[str, Any]],
    *,
    source: str = "control_plane",
    note: str = "Active structured WM rules injected as control-plane constraints.",
) -> str:
    lines: list[str] = []
    seen_rule_keys: set[str] = set()
    fallback_index = 0
    for raw_spec in rule_specs:
        if not isinstance(raw_spec, dict):
            continue
        spec = deepcopy(raw_spec)
        if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
            continue
        fallback_index += 1
        spec.setdefault("schema_version", "wm_rule_v1")
        rule_id = str(spec.get("rule_id", "") or "").strip() or f"wm_rule_{fallback_index:03d}"
        if rule_id in seen_rule_keys:
            continue
        seen_rule_keys.add(rule_id)
        spec["rule_id"] = rule_id
        lines.append(json.dumps(spec, ensure_ascii=False))
    if not lines:
        return ""
    return "\n".join(
        [
            (
                f'<working_memory_rule_specs format="jsonl" priority="highest" '
                f'source="{source}" note="{note}">'
            ),
            *lines,
            "</working_memory_rule_specs>",
        ]
    )


def _structured_rule_identity(raw_spec: Any) -> str:
    if not isinstance(raw_spec, dict):
        return ""
    spec = raw_spec.get("rule_spec") if isinstance(raw_spec.get("rule_spec"), dict) else raw_spec
    if not isinstance(spec, dict):
        return ""
    rule_id = str(spec.get("rule_id", "") or "").strip()
    if rule_id:
        return f"id:{rule_id}"
    target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    propagation = spec.get("propagation") if isinstance(spec.get("propagation"), dict) else {}
    fingerprint = {
        "dimension": str(spec.get("dimension", "") or ""),
        "target": target,
        "action": action,
        "propagation": propagation,
        "normalized_sentence": str(spec.get("normalized_sentence", "") or ""),
    }
    return "fp:" + json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, default=str)


def collect_saved_structured_wm_rule_specs(runtime: Any) -> list[dict[str, Any]]:
    """Collect structured WM rules from the durable session snapshot.

    External workers do not share the web process' live WorkingMemory object. When
    a user leaves and resumes a session, the session-level live snapshot is the
    durable source for session-scoped future rules. Legacy round snapshots are
    only used as a compatibility fallback when the session snapshot does not
    exist yet.
    """
    workspace = getattr(runtime, "workspace", None)
    if not workspace:
        return []
    memory_root = Path(workspace) / ".memory"
    live_snapshot_path = memory_root / LIVE_WM_SNAPSHOT_FILENAME
    candidates: list[Path] = []
    if live_snapshot_path.exists():
        candidates.append(live_snapshot_path)
    rounds_root = memory_root / "rounds"
    if rounds_root.exists():
        try:
            candidates.extend(
                sorted(
                    rounds_root.glob("round_*/wm_snapshot.json"),
                    key=lambda item: item.stat().st_mtime if item.exists() else 0,
                    reverse=True,
                )
            )
        except Exception:
            logger.debug("Failed to list saved WM snapshots", exc_info=True)
    if not candidates:
        return []

    specs: list[dict[str, Any]] = []
    seen_rule_keys: set[str] = set()

    def _add(raw_spec: Any, *, fallback_content: str = "", fallback_dimension: str = "") -> None:
        if not isinstance(raw_spec, dict):
            return
        spec = deepcopy(raw_spec.get("rule_spec")) if isinstance(raw_spec.get("rule_spec"), dict) else deepcopy(raw_spec)
        if not isinstance(spec, dict):
            return
        if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
            return
        spec.setdefault("schema_version", "wm_rule_v1")
        spec.setdefault("source", "saved_round_snapshot")
        if fallback_content:
            spec.setdefault("content", fallback_content)
            spec.setdefault("normalized_sentence", fallback_content)
        if fallback_dimension:
            spec.setdefault("dimension", fallback_dimension)
        rule_id = str(spec.get("rule_id", "") or "").strip() or f"saved_wm_rule_{len(specs) + 1:03d}"
        spec["rule_id"] = rule_id
        rule_key = _structured_rule_identity(spec) or f"id:{rule_id}"
        if rule_key in seen_rule_keys:
            return
        seen_rule_keys.add(rule_key)
        specs.append(spec)

    for snapshot_path in candidates:
        try:
            raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read saved WM snapshot: %s", snapshot_path, exc_info=True)
            continue
        if not isinstance(raw, dict):
            continue
        for _index, pref in enumerate(raw.get("temp_preferences", []) or []):
            if not isinstance(pref, dict) or bool(pref.get("superseded", False)):
                continue
            structured = pref.get("structured_data")
            _add(
                structured,
                fallback_content=str(pref.get("content", "") or ""),
                fallback_dimension=str(pref.get("dimension", "") or ""),
            )
        structured_text = str(raw.get("structured_rules_text") or "")
        if not structured_text and snapshot_path.parent.exists():
            try:
                injection_path = snapshot_path.parent / "memory_injection.txt"
                if injection_path.exists():
                    structured_text = injection_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                structured_text = ""
        for spec in parse_rule_specs_from_injection(structured_text):
            _add(spec)

    return specs


def _live_wm_snapshot_path(runtime: Any) -> Path | None:
    workspace = getattr(runtime, "workspace", None)
    if not workspace:
        return None
    return Path(workspace) / ".memory" / LIVE_WM_SNAPSHOT_FILENAME


def _read_wm_snapshot_file(snapshot_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read saved WM snapshot: %s", snapshot_path, exc_info=True)
        return None
    return raw if isinstance(raw, dict) else None


def _structured_rule_temp_preference_payload(
    spec: dict[str, Any],
    *,
    timestamp: str = "",
    source_task_id: str = "saved_structured_rule_carryover",
) -> dict[str, Any]:
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    content = (
        str(spec.get("normalized_sentence") or "").strip()
        or str(spec.get("content") or "").strip()
        or str(action.get("description") or "").strip()
        or str(spec.get("rule_id") or "Structured temporary preference").strip()
    )
    return {
        "content": content,
        "dimension": str(spec.get("dimension") or "general").strip() or "general",
        "preference_type": "value",
        "source_task_id": source_task_id,
        "scope": str(spec.get("scope") or "global").strip() or "global",
        "scope_value": "",
        "superseded": False,
        "structured_data": deepcopy(spec),
        "timestamp": timestamp,
    }


def merge_structured_rule_specs_into_temp_preferences(
    temp_preferences: list[Any],
    rule_specs: list[dict[str, Any]],
    *,
    timestamp: str = "",
    source_task_id: str = "saved_structured_rule_carryover",
) -> list[Any]:
    merged = list(temp_preferences or [])
    seen: set[str] = set()
    for pref in merged:
        if not isinstance(pref, dict) or bool(pref.get("superseded", False)):
            continue
        key = _structured_rule_identity(pref.get("structured_data"))
        if key:
            seen.add(key)
    for raw_spec in rule_specs or []:
        if not isinstance(raw_spec, dict):
            continue
        spec = deepcopy(raw_spec.get("rule_spec")) if isinstance(raw_spec.get("rule_spec"), dict) else deepcopy(raw_spec)
        if not isinstance(spec, dict):
            continue
        if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
            continue
        spec.setdefault("schema_version", "wm_rule_v1")
        key = _structured_rule_identity(spec)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(
            _structured_rule_temp_preference_payload(
                spec,
                timestamp=timestamp,
                source_task_id=source_task_id,
            )
        )
    return merged


def _persist_live_wm_snapshot_payload(
    *,
    snapshot_path: Path,
    orchestrator: Any | None,
    runtime: Any | None = None,
    reason: str = "",
    session_id: str = "",
    user_id: str = "",
    fallback_intent: str = "",
) -> bool:
    job_mgr = getattr(orchestrator, "_job_mgr", None) if orchestrator is not None else None
    wm = getattr(job_mgr, "working_memory", None) if job_mgr is not None else None
    active_job = getattr(job_mgr, "_active_job", None) if job_mgr is not None else None
    if wm is None:
        return False
    try:
        round_count = int(job_mgr.round_count()) if hasattr(job_mgr, "round_count") else 0
    except Exception:
        round_count = 0
    try:
        saved_at = datetime.now(timezone.utc).astimezone().isoformat()
        temp_preferences = safe_memory_json(getattr(wm, "_temp_preferences", []) or [])
        active_structured_specs = collect_active_structured_wm_rule_specs(runtime, orchestrator) if runtime is not None else []
        if active_structured_specs:
            temp_preferences = merge_structured_rule_specs_into_temp_preferences(
                temp_preferences if isinstance(temp_preferences, list) else [],
                active_structured_specs,
                timestamp=saved_at,
                source_task_id="live_snapshot_structured_rule_carryover",
            )
        structured_rules_text = (
            format_rule_specs_text(
                active_structured_specs,
                source="live_session_snapshot",
                note="Merged active structured WM rules, including restored prior-session rules.",
            )
            if active_structured_specs
            else str(orchestrator.get_wm_structured_rules_text() or "") if orchestrator is not None and hasattr(orchestrator, "get_wm_structured_rules_text") else ""
        )
        payload = {
            "schema_version": 1,
            "snapshot_source": "live_session",
            "snapshot_reason": str(reason or ""),
            "saved_at": saved_at,
            "session_id": str(session_id or getattr(active_job, "project_id", "") or ""),
            "user_id": str(user_id or getattr(active_job, "user_id", "") or ""),
            "job": {
                "id": str(getattr(active_job, "id", "") or ""),
                "intent": str(getattr(active_job, "intent", "") or fallback_intent or ""),
                "read_intent": str(getattr(active_job, "read_intent", "") or ""),
                "write_intent": str(getattr(active_job, "write_intent", "") or ""),
                "started_at": str(getattr(active_job, "started_at", "") or ""),
                "round_count": round_count,
            },
            "counts": {
                "preferences": len(temp_preferences),
                "experiences": len(wm.get_experiences() if hasattr(wm, "get_experiences") else getattr(wm, "_temp_experiences", []) or []),
                "episodes": len(wm.get_episodes() if hasattr(wm, "get_episodes") else getattr(wm, "_temp_episodes", []) or []),
                "rounds": round_count,
                "structured_rules": len(active_structured_specs),
            },
            "temp_preferences": temp_preferences,
            "temp_experiences": safe_memory_json(
                wm.get_experiences() if hasattr(wm, "get_experiences") else getattr(wm, "_temp_experiences", []) or []
            ),
            "temp_episodes": safe_memory_json(
                wm.get_episodes() if hasattr(wm, "get_episodes") else getattr(wm, "_temp_episodes", []) or []
            ),
            "chain_buffer": safe_memory_json(getattr(wm, "chain_buffer", None)),
            "general_rules_text": str(orchestrator.get_wm_general_rules_text() or "") if orchestrator is not None and hasattr(orchestrator, "get_wm_general_rules_text") else "",
            "structured_rules_text": structured_rules_text,
            "preferences_text": str(orchestrator.get_wm_preferences_text(relevant_dims=None, include_general=False) or "") if orchestrator is not None and hasattr(orchestrator, "get_wm_preferences_text") else "",
        }
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = snapshot_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(snapshot_path)
        return True
    except Exception:
        logger.debug("Failed to persist live WM snapshot", exc_info=True)
        return False


def persist_live_wm_snapshot(runtime: Any, *, reason: str = "") -> bool:
    """Persist the current active WM to a session-level snapshot file.

    Round artifacts remain useful for debugging, but this file is the durable
    state used when resuming local sessions after navigation/restart.
    """
    snapshot_path = _live_wm_snapshot_path(runtime)
    if snapshot_path is None:
        return False
    return _persist_live_wm_snapshot_payload(
        snapshot_path=snapshot_path,
        orchestrator=getattr(runtime, "_memory_orchestrator_instance", None),
        runtime=runtime,
        reason=reason,
        session_id=str(getattr(runtime, "session_id", "") or ""),
        user_id=str(getattr(runtime, "user_id", "") or ""),
        fallback_intent=str(getattr(runtime, "_resolved_request_intent", "") or ""),
    )


def persist_live_wm_snapshot_from_orchestrator(orchestrator: Any, *, reason: str = "") -> bool:
    job_mgr = getattr(orchestrator, "_job_mgr", None)
    workspace = getattr(job_mgr, "_workspace", None) if job_mgr is not None else None
    if not workspace:
        return False
    return _persist_live_wm_snapshot_payload(
        snapshot_path=Path(workspace) / ".memory" / LIVE_WM_SNAPSHOT_FILENAME,
        orchestrator=orchestrator,
        runtime=None,
        reason=reason,
    )


def _saved_wm_snapshot_payloads(runtime: Any) -> list[tuple[dict[str, Any], Path]]:
    workspace = getattr(runtime, "workspace", None)
    if not workspace:
        return []
    payloads: list[tuple[dict[str, Any], Path]] = []
    live_snapshot_path = _live_wm_snapshot_path(runtime)
    if live_snapshot_path is not None and live_snapshot_path.exists():
        live_payload = _read_wm_snapshot_file(live_snapshot_path)
        if live_payload is not None:
            payloads.append((live_payload, live_snapshot_path))
    rounds_root = Path(workspace) / ".memory" / "rounds"
    if not rounds_root.exists():
        return payloads
    try:
        candidates = sorted(
            rounds_root.glob("round_*/wm_snapshot.json"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
    except Exception:
        logger.debug("Failed to list saved WM snapshots", exc_info=True)
        return payloads
    for snapshot_path in candidates:
        if live_snapshot_path is not None and snapshot_path == live_snapshot_path:
            continue
        raw = _read_wm_snapshot_file(snapshot_path)
        if raw is not None:
            payloads.append((raw, snapshot_path))
    return payloads


async def hydrate_active_wm_from_latest_snapshot(
    runtime: Any,
    orchestrator: Any | None,
    *,
    reason: str = "session_resume",
    allow_seed_only_replace: bool = False,
) -> dict[str, Any]:
    """Seed an empty active WM from the latest saved round snapshot.

    This makes same-session working memory survive logout/re-entry and external
    worker process boundaries. It intentionally only runs on an empty active WM.
    """
    wm = _get_orchestrator_wm(orchestrator)
    if wm is None:
        return {"hydrated": False, "reason": "no_active_wm"}
    if (
        getattr(wm, "_temp_preferences", None)
        or getattr(wm, "_temp_experiences", None)
        or getattr(wm, "_temp_episodes", None)
    ):
        if allow_seed_only_replace and working_memory_has_only_restore_seed_content(wm):
            cleared = clear_restore_seed_content(wm)
        else:
            return {"hydrated": False, "reason": "active_wm_not_empty"}
    else:
        cleared = {"preferences_removed": 0, "experiences_removed": 0, "episodes_removed": 0}

    all_snapshots = _saved_wm_snapshot_payloads(runtime)
    snapshots = list(all_snapshots)
    if not snapshots:
        return {
            "hydrated": False,
            "reason": "no_saved_snapshot",
            "snapshot_path": "",
        }

    try:
        from memslides.memory.core.models import DesignEpisode, RoundExperience, TempPreference
    except Exception:
        logger.debug("Failed to import WM model classes for snapshot hydration", exc_info=True)
        return {"hydrated": False, "reason": "model_import_failed", "snapshot_path": ""}

    primary_snapshot_path = snapshots[0][1]
    is_live_primary = primary_snapshot_path.name == LIVE_WM_SNAPSHOT_FILENAME
    plain_snapshots = [snapshots[0]] if is_live_primary else snapshots

    pref_count = 0
    seen_preferences: set[str] = set()
    snapshot_paths: list[str] = []
    for raw, snapshot_path in reversed(plain_snapshots):
        snapshot_paths.append(str(snapshot_path))
        for item in raw.get("temp_preferences", []) or []:
            if not isinstance(item, dict):
                continue
            key_payload = {
                "content": item.get("content", ""),
                "dimension": item.get("dimension", ""),
                "scope": item.get("scope", ""),
                "scope_value": item.get("scope_value", ""),
                "structured_data": item.get("structured_data"),
            }
            key = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen_preferences:
                continue
            seen_preferences.add(key)
            try:
                wm._temp_preferences.append(TempPreference(**item))
                pref_count += 1
            except Exception:
                logger.debug("Failed to hydrate TempPreference from saved snapshot", exc_info=True)

    latest_payload = snapshots[0][0] if snapshots else {}
    latest_structured_ids = {
        _structured_rule_identity(item.get("structured_data"))
        for item in latest_payload.get("temp_preferences", []) or []
        if isinstance(item, dict) and not bool(item.get("superseded", False))
    }
    latest_structured_ids.discard("")
    carried_rule_count = 0
    carryover_sources = all_snapshots[1:] if len(all_snapshots) > 1 else []
    for raw, _snapshot_path in carryover_sources:
        for item in raw.get("temp_preferences", []) or []:
            if not isinstance(item, dict) or bool(item.get("superseded", False)):
                continue
            structured = item.get("structured_data")
            rule_key = _structured_rule_identity(structured)
            if not rule_key or rule_key in latest_structured_ids:
                continue
            carried = deepcopy(item)
            carried["source_task_id"] = str(carried.get("source_task_id") or "saved_structured_rule_carryover")
            carried_structured = carried.get("structured_data")
            if isinstance(carried_structured, dict):
                carried_structured = deepcopy(carried_structured)
                carried_structured["source"] = str(carried_structured.get("source") or "saved_round_snapshot")
                carried_structured["restored_from_prior_snapshot"] = True
                carried["structured_data"] = carried_structured
            try:
                wm._temp_preferences.append(TempPreference(**carried))
                latest_structured_ids.add(rule_key)
                carried_rule_count += 1
            except Exception:
                logger.debug("Failed to carry over structured WM rule from prior snapshot", exc_info=True)

    exp_count = 0
    seen_experiences: set[str] = set()
    for raw, _snapshot_path in reversed(plain_snapshots):
        for item in raw.get("temp_experiences", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "") or json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen_experiences:
                continue
            seen_experiences.add(key)
            try:
                wm._temp_experiences.append(RoundExperience.from_dict(item))
                exp_count += 1
            except Exception:
                logger.debug("Failed to hydrate RoundExperience from saved snapshot", exc_info=True)

    episode_count = 0
    seen_episodes: set[str] = set()
    for raw, _snapshot_path in reversed(plain_snapshots):
        for item in raw.get("temp_episodes", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("episode_id") or item.get("id") or "") or json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen_episodes:
                continue
            seen_episodes.add(key)
            try:
                wm._temp_episodes.append(DesignEpisode.from_dict(item))
                episode_count += 1
            except Exception:
                logger.debug("Failed to hydrate DesignEpisode from saved snapshot", exc_info=True)

    if hasattr(wm, "_normalize_active_general_preferences"):
        try:
            wm._normalize_active_general_preferences()
        except Exception:
            logger.debug("Failed to normalize hydrated general WM preferences", exc_info=True)

    hydrated = pref_count > 0 or exp_count > 0 or episode_count > 0
    result = {
        "hydrated": hydrated,
        "reason": reason,
        "snapshot_path": snapshot_paths[-1] if snapshot_paths else "",
        "snapshot_count": len(snapshots),
        "preferences": pref_count,
        "experiences": exp_count,
        "episodes": episode_count,
        "structured_rules": len(collect_active_structured_wm_rule_specs(runtime, orchestrator)) if hydrated else 0,
        "carried_structured_rules": carried_rule_count,
        "cleared_seed_content": cleared,
    }
    if hydrated:
        logger.info(
            "Hydrated active WM from saved snapshots (%s): preferences=%s experiences=%s episodes=%s structured_rules=%s",
            len(snapshots),
            pref_count,
            exp_count,
            episode_count,
            result["structured_rules"],
        )
    return result


def collect_active_structured_wm_rule_specs(runtime: Any, orchestrator: Any | None = None) -> list[dict[str, Any]]:
    """Collect every active job-local structured WM rule as control-plane data."""
    specs: list[dict[str, Any]] = []
    seen_rule_keys: set[str] = set()

    def _add(raw_spec: Any, *, fallback_content: str = "", fallback_dimension: str = "") -> None:
        if not isinstance(raw_spec, dict):
            return
        spec = deepcopy(raw_spec.get("rule_spec")) if isinstance(raw_spec.get("rule_spec"), dict) else deepcopy(raw_spec)
        if not isinstance(spec, dict):
            return
        if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
            return
        spec.setdefault("schema_version", "wm_rule_v1")
        if fallback_content:
            spec.setdefault("content", fallback_content)
            spec.setdefault("normalized_sentence", fallback_content)
        if fallback_dimension:
            spec.setdefault("dimension", fallback_dimension)
        rule_id = str(spec.get("rule_id", "") or "").strip() or f"wm_rule_{len(specs) + 1:03d}"
        spec["rule_id"] = rule_id
        rule_key = _structured_rule_identity(spec) or f"id:{rule_id}"
        if rule_key in seen_rule_keys:
            return
        seen_rule_keys.add(rule_key)
        specs.append(spec)

    orch = orchestrator or getattr(runtime, "_memory_orchestrator_instance", None)
    if orch is not None and hasattr(orch, "get_wm_structured_rules_text"):
        try:
            for spec in parse_rule_specs_from_injection(orch.get_wm_structured_rules_text() or ""):
                _add(spec)
        except Exception:
            logger.debug("Failed to collect structured WM rules from orchestrator text", exc_info=True)

    try:
        wm = _get_orchestrator_wm(orch) if orch is not None else None
    except Exception:
        wm = None
    if wm is not None:
        for _index, pref in enumerate(getattr(wm, "_temp_preferences", []) or []):
            if getattr(pref, "superseded", False):
                continue
            structured = getattr(pref, "structured_data", None)
            _add(
                structured,
                fallback_content=str(getattr(pref, "content", "") or ""),
                fallback_dimension=str(getattr(pref, "dimension", "") or ""),
            )

    fallback_text = get_session_preference_fallback_prompt(runtime)
    if fallback_text:
        for spec in parse_rule_specs_from_injection(fallback_text):
            _add(spec)

    for spec in collect_saved_structured_wm_rule_specs(runtime):
        _add(spec)

    return specs


def merge_wm_rule_specs_text(
    runtime: Any,
    existing_text: str,
    *,
    orchestrator: Any | None = None,
    source: str = "control_plane",
) -> tuple[str, dict[str, Any]]:
    active_specs = collect_active_structured_wm_rule_specs(runtime, orchestrator)
    existing_specs = parse_rule_specs_from_injection(existing_text)
    existing_ids = {
        str(spec.get("rule_id", "") or "").strip()
        for spec in existing_specs
        if isinstance(spec, dict) and str(spec.get("rule_id", "") or "").strip()
    }
    skipped: list[dict[str, str]] = []
    injected: list[dict[str, Any]] = []
    for spec in active_specs:
        rule_id = str(spec.get("rule_id", "") or "").strip()
        if rule_id and rule_id in existing_ids:
            skipped.append({"rule_id": rule_id, "reason": "already_present"})
            continue
        injected.append(spec)
    control_text = format_rule_specs_text(
        injected,
        source=source,
        note="All active structured WM rules; not filtered by dimension matcher/top-k.",
    )
    merged = "\n\n".join(filter(None, [existing_text, control_text]))
    trace = {
        "active_rule_count": len(active_specs),
        "existing_rule_count": len(existing_specs),
        "injected_rule_ids": [
            str(spec.get("rule_id", "") or "").strip()
            for spec in injected
            if str(spec.get("rule_id", "") or "").strip()
        ],
        "skipped_rules": skipped,
        "source": source,
    }
    return merged, trace


def append_memory_flow_trace(
    runtime: Any,
    *,
    event: str,
    user_message: str = "",
    intent: dict[str, Any] | None = None,
    plan: ModifyExecutionPlan | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        history_dir = _modify_policy_history_dir(runtime)
        payload = {
            "timestamp": _utc_timestamp(),
            "turn": _current_modify_turn(runtime),
            "event": str(event or "memory_flow"),
            "user_message_excerpt": str(user_message or "")[:500],
            "intent": _json_safe(intent or {}),
            "plan": {
                "scope": getattr(plan, "scope", "") if plan else "",
                "operation_kind": getattr(plan, "operation_kind", "") if plan else "",
                "target_slide_paths": [
                    workspace_relative_label(runtime, path)
                    for path in getattr(plan, "target_slide_paths", []) or []
                ],
                "target_rule_ids": list(getattr(plan, "target_rule_ids", []) or []),
                "applicable_rule_ids": list(getattr(plan, "applicable_rule_ids", []) or []),
                "new_element_rule_applications": _json_safe(
                    getattr(plan, "new_element_rule_applications", []) or []
                ),
                "coverage_required": bool(getattr(plan, "coverage_required", False)) if plan else False,
                "reason": getattr(plan, "reason", "") if plan else "",
            },
            "extra": _json_safe(extra or {}),
        }
        trace_path = history_dir / "memory_flow_trace.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to append memory-flow trace", exc_info=True)


def selector_hints_for_element_kind(element_kind: str) -> list[str]:
    kind = canonical_session_element_kind(element_kind)
    mapping = {
        ANY_TEXT_ELEMENT_KIND: [
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
        "table_cell": ["td", "th", ".table-cell", "[data-role='table-cell']"],
        "chart_axis_label": [".axis-label", ".chart-axis-label", "[data-role='chart-axis-label']"],
        "chart_data_label": [".data-label", ".chart-data-label", "[data-role='chart-data-label']"],
        "legend_label": [".legend", ".legend-label", "[data-role='legend-label']"],
        "callout": [".callout", ".annotation", "[data-role='callout']"],
        "caption": [
            ".caption",
            "figcaption",
            ".figure-caption",
            ".table-caption",
            ".image-caption",
            ".fig-caption",
            ".caption.note",
            ".note",
            "[data-role='caption']",
            "[data-role='figure-caption']",
            "[data-role='table-caption']",
        ],
    }
    return mapping.get(kind, [])


def rule_spec_applies_to_existing_slides(spec: dict[str, Any]) -> bool:
    """Return whether a WM rule expects edits on already-rendered slides."""
    if not isinstance(spec, dict):
        return False

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

    return False


def rule_spec_applies_to_future_slides(spec: dict[str, Any]) -> bool:
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


def rule_spec_applies_to_future_elements(spec: dict[str, Any]) -> bool:
    """Return whether a WM rule explicitly targets elements created in later turns."""
    if not isinstance(spec, dict):
        return False
    propagation = spec.get("propagation")
    if isinstance(propagation, dict) and "apply_future_elements" in propagation:
        return _normalize_bool_flag(propagation.get("apply_future_elements"))
    return False


def rule_spec_creation_scope(spec: dict[str, Any]) -> list[str]:
    if not isinstance(spec, dict):
        return []
    propagation = spec.get("propagation")
    raw_scope: Any = []
    if isinstance(propagation, dict):
        raw_scope = propagation.get("creation_scope", [])
    if isinstance(raw_scope, str):
        candidates = [raw_scope]
    elif isinstance(raw_scope, list):
        candidates = raw_scope
    else:
        candidates = []
    target = spec.get("target")
    semantic_target = ""
    if isinstance(target, dict):
        semantic_target = target.get("semantic_target", "") or ""
    scope = normalize_session_creation_scope(
        candidates,
        str(spec.get("normalized_sentence", "") or ""),
        str(spec.get("natural_language", "") or ""),
        semantic_target=semantic_target,
    )
    if scope:
        return scope
    if isinstance(target, dict):
        kind = canonical_session_element_kind(target.get("element_kind", ""))
        if kind and kind != "deck":
            return [kind]
    return []


def future_element_candidate_rule_specs(rule_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active rules worth presenting to the LLM applicability pass."""
    candidates: list[dict[str, Any]] = []
    for spec in rule_specs:
        if not isinstance(spec, dict):
            continue
        if rule_spec_applies_to_existing_slides(spec):
            continue
        if rule_spec_applies_to_future_elements(spec) or rule_spec_applies_to_future_slides(spec):
            candidates.append(spec)
    return candidates


def request_explicitly_targets_existing_slides(user_message: str) -> bool:
    """Return whether wording asks to mutate already-rendered slides."""
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False
    patterns = (
        r"(现有|已有|当前|已经生成|现在的|整套).{0,16}(页|页面|幻灯片|标题)",
        r"(所有|全部|整套|全局).{0,16}(页|页面|幻灯片|slide|标题|正文|图注|配色|颜色|字体|版式|风格).{0,24}(改|修改|设置|设为|变成|换成|统一|apply|change|set)",
        r"(首页|封面|目录页|这一页|这页|本页|当前页|第\s*\d+\s*(页|张)|slide\s*\d+)",
        r"(把|将|改|修改|设置|设为|变成|换成|apply|change|set).{0,24}(现有|已有|当前|所有|全部|整套)",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def infer_specific_existing_slide_reference(user_message: str) -> str:
    """Infer a concrete existing slide id from common user wording."""
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return ""
    if re.search(r"(首页|封面|第一页|第\s*1\s*(页|张)|slide[_\s-]*0?1\b)", normalized):
        return "slide_01"
    match = re.search(r"第\s*(\d{1,3})\s*(页|张)", normalized)
    if not match:
        match = re.search(r"slide[_\s-]*(\d{1,3})\b", normalized)
    if match:
        try:
            return f"slide_{int(match.group(1)):02d}"
        except (TypeError, ValueError):
            return ""
    return ""


def collect_rule_ids_and_selector_hints(specs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Collect model-facing rule ids and likely selectors from structured rules."""
    selector_hints: list[str] = []
    rule_ids: list[str] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        rule_id = str(spec.get("rule_id", "") or "").strip()
        if rule_id and rule_id not in rule_ids:
            rule_ids.append(rule_id)
        target = spec.get("target")
        if isinstance(target, dict):
            for hint in selector_hints_for_element_kind(
                str(target.get("element_kind", "") or "").strip()
            ):
                if hint not in selector_hints:
                    selector_hints.append(hint)
    return rule_ids, selector_hints


def is_preference_update_plan(plan: ModifyExecutionPlan | None) -> bool:
    if plan is None:
        return False
    operation_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=getattr(plan, "scope", None),
    )
    return (
        operation_kind == "preference_update"
        and not list(getattr(plan, "target_slide_paths", []) or [])
        and not bool(getattr(plan, "coverage_required", False))
    )


def is_future_only_preference_plan(plan: ModifyExecutionPlan | None) -> bool:
    """Backward-compatible alias for the future-only preference-update plan shape."""
    return is_preference_update_plan(plan)


def parse_style_declarations(style_text: str) -> dict[str, str]:
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


def extract_json_object(text: str) -> dict[str, Any]:
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


def _extract_path_candidates_from_text(text: str, field_names: tuple[str, ...]) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: list[str] = []

    def _add(candidate: Any) -> None:
        text_value = str(candidate or "").strip()
        if text_value and text_value not in candidates:
            candidates.append(text_value)

    payload = extract_json_object(raw)
    if isinstance(payload, dict):
        for field_name in field_names:
            _add(payload.get(field_name, ""))
        results = payload.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                for field_name in field_names:
                    _add(item.get(field_name, ""))

    for field_name in field_names:
        key_pattern = r"[_\s-]*".join(re.escape(part) for part in field_name.split("_") if part)
        if not key_pattern:
            continue
        pattern = re.compile(
            rf'["\']?(?:{key_pattern})["\']?\s*[:=]\s*["\']?(?P<path>[^\n\r"\'`{{}}\[\],]+)',
            re.IGNORECASE,
        )
        for match in pattern.finditer(raw):
            _add(match.group("path").strip().rstrip(".,;:)"))

    return candidates


def resolve_css_var_value(
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
    return resolve_css_var_value(resolved, css_vars, depth + 1)


def extract_inserted_slide_path_from_result(result_text: str) -> str:
    candidates = _extract_path_candidates_from_text(
        result_text,
        ("new_slide_path", "slide_path", "file_path", "write_target", "path"),
    )
    return candidates[0] if candidates else ""


_INSERTED_SLIDE_EXPLICIT_PLACEHOLDER_MARKERS = (
    "新插入页面",
    "占位",
    "请用 write_new_slide_file",
    "write_new_slide_file 填充最终内容",
    "placeholder created",
)
_INSERTED_SLIDE_GENERIC_PLACEHOLDER_MARKERS = (
    "新页面",
    "请在此添加内容",
    "请添加内容",
    "添加内容",
    "new slide",
    "add content",
    "placeholder",
    "todo",
    "tbd",
)


def _normalize_inserted_slide_completion_text(text: str) -> str:
    normalized = str(text or "").lower().replace("…", "...")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[\[\]\(\)\{\}<>'\"`“”‘’.,，:：;；!！?？、/\\|+=_-]+", "", normalized)
    return normalized


def _visible_body_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    for node in soup(["script", "style", "noscript", "template"]):
        node.decompose()
    root = soup.body or soup
    return " ".join(root.get_text(" ", strip=True).split())


def inspect_inserted_slide_completion(path: Path | str) -> dict[str, Any]:
    """Return whether a newly inserted slide still looks like an unfinished placeholder."""
    slide_path = Path(path)
    status: dict[str, Any] = {
        "slide_path": str(slide_path),
        "is_pending": False,
        "reason": "",
        "signals": [],
        "visible_text_excerpt": "",
    }
    if not slide_path.exists():
        status.update(
            {
                "is_pending": True,
                "reason": "missing_inserted_slide_file",
                "signals": ["missing_file"],
            }
        )
        return status

    try:
        html_text = slide_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        status.update(
            {
                "is_pending": True,
                "reason": "unreadable_inserted_slide_file",
                "signals": [str(exc)],
            }
        )
        return status

    visible_text = _visible_body_text_from_html(html_text)
    status["visible_text_excerpt"] = visible_text[:240]
    normalized_html = _normalize_inserted_slide_completion_text(html_text)
    normalized_visible = _normalize_inserted_slide_completion_text(visible_text)

    signals: list[str] = []
    for marker in _INSERTED_SLIDE_EXPLICIT_PLACEHOLDER_MARKERS:
        normalized_marker = _normalize_inserted_slide_completion_text(marker)
        if normalized_marker and normalized_marker in normalized_html:
            signals.append(marker)
    if signals:
        status.update(
            {
                "is_pending": True,
                "reason": "inserted_slide_placeholder_marker",
                "signals": signals,
            }
        )
        return status

    generic_hits: list[str] = []
    meaningful_text = normalized_visible
    for marker in _INSERTED_SLIDE_GENERIC_PLACEHOLDER_MARKERS:
        normalized_marker = _normalize_inserted_slide_completion_text(marker)
        if normalized_marker and normalized_marker in normalized_visible:
            generic_hits.append(marker)
            meaningful_text = meaningful_text.replace(normalized_marker, "")

    meaningful_text = re.sub(r"(slide|page|第)?\d+(?:/\d+)?(页)?", "", meaningful_text)
    if generic_hits and len(meaningful_text) < 24:
        status.update(
            {
                "is_pending": True,
                "reason": "inserted_slide_generic_placeholder",
                "signals": generic_hits,
            }
        )
        return status

    if not normalized_visible:
        status.update(
            {
                "is_pending": True,
                "reason": "inserted_slide_empty_body",
                "signals": ["empty_body"],
            }
        )
    return status


def collect_pending_inserted_slides(
    runtime: Any,
    inserted_paths: list[Path] | tuple[Path, ...] | set[Path],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for raw_path in inserted_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = runtime.workspace / path
        status = inspect_inserted_slide_completion(path)
        if status.get("is_pending"):
            pending.append(status)
    return pending


def build_inserted_slide_completion_followup(
    runtime: Any,
    pending_slides: list[dict[str, Any]],
) -> str:
    if not pending_slides:
        return ""
    lines = []
    for item in pending_slides[:8]:
        slide_path = Path(str(item.get("slide_path", "") or ""))
        label = workspace_relative_label(runtime, slide_path) if str(slide_path) else "(unknown slide)"
        reason = str(item.get("reason", "") or "unfinished_inserted_slide")
        signals = ", ".join(str(signal) for signal in item.get("signals", [])[:4]) or "no meaningful body text"
        excerpt = str(item.get("visible_text_excerpt", "") or "").strip()
        suffix = f"; visible text: {excerpt[:120]}" if excerpt else ""
        lines.append(f"- {label}: {reason}; signals: {signals}{suffix}")
    return (
        "SYSTEM: One or more newly inserted slides are still placeholders or underfilled, so this modify turn is not complete.\n"
        + "\n".join(lines)
        + "\nPreferred next action: use `write_new_slide_file` on each listed inserted slide to write a complete HTML slide "
        "with real content and layout. You may read neighboring slides first for continuity. "
        "Use `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch` only after the complete slide HTML exists, "
        "for small post-write repairs. Do not call `finalize` while these inserted slides remain unfinished."
    )


def color_token_to_rgb(color_value: str) -> tuple[int, int, int] | None:
    text = str(color_value or "").strip().lower().replace("!important", "").strip()
    if not text:
        return None
    if text.startswith("linear-gradient") or text.startswith("radial-gradient"):
        match = re.search(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", text)
        if match:
            text = match.group(0)
        else:
            return None
    if " " in text and text.startswith("#"):
        text = text.split()[0]

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
        return tuple(int(raw[idx : idx + 2], 16) for idx in (0, 2, 4))

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


def css_color_tokens_equivalent(expected: str, observed: str) -> bool:
    expected_rgb = color_token_to_rgb(expected)
    observed_rgb = color_token_to_rgb(observed)
    if expected_rgb is not None and observed_rgb is not None:
        return expected_rgb == observed_rgb
    return normalize_css_value("color", expected) == normalize_css_value("color", observed)


def normalize_css_value(property_name: str, value: str) -> str:
    prop = normalize_css_property_name(property_name)
    text = str(value or "").strip().lower().replace("!important", "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if prop in {"color", "background-color", "background"}:
        rgb = color_token_to_rgb(text)
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
        return text
    if prop == "font-style":
        if "italic" in text:
            return "italic"
        if "oblique" in text:
            return "oblique"
        return text
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


def normalize_css_property_name(property_name: str) -> str:
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


def css_values_equivalent(property_name: str, expected: str, observed: str) -> bool:
    prop = normalize_css_property_name(property_name)
    if prop in {"color", "background-color", "background"}:
        return css_color_tokens_equivalent(expected, observed)
    return normalize_css_value(prop, expected) == normalize_css_value(prop, observed)


def css_value_is_machine_verifiable(property_name: str, value: str) -> bool:
    prop = normalize_css_property_name(property_name)
    text = str(value or "").strip()
    if not prop or not text:
        return False
    if prop in {"color", "background-color", "background"}:
        return _EXPLICIT_COLOR_RE.fullmatch(text) is not None
    normalized = normalize_css_value(prop, text)
    if prop == "font-style":
        return normalized in {"italic", "oblique", "normal"}
    if prop == "font-weight":
        return normalized in {"400", "700"} or re.fullmatch(r"\d{3}", normalized) is not None
    if prop == "text-decoration":
        return normalized in {"underline", "overline", "line-through", "none"}
    return prop in {"font-size", "font-family", "opacity"} and bool(normalized)


def future_preference_verifiable_on_single_slide(spec: dict[str, Any]) -> bool:
    if not isinstance(spec, dict) or not rule_spec_applies_to_future_slides(spec):
        return False
    if not verifiable_rule_css_values(spec):
        return False
    if future_slide_rule_needs_semantic_resolution(spec):
        return False
    target = spec.get("target")
    if not isinstance(target, dict):
        return False
    element_kind = canonical_session_element_kind(
        target.get("element_kind", ""),
        str(spec.get("normalized_sentence", "") or ""),
        str(spec.get("content", "") or ""),
    )
    return bool(element_kind and element_kind != "deck")


def future_preference_judge_llm(runtime: Any) -> Any | None:
    modify_agent = getattr(runtime, "modifyagent", None)
    llm = getattr(modify_agent, "llm", None) if modify_agent is not None else None
    if llm and hasattr(llm, "run"):
        return llm
    design_agent = getattr(runtime, "designagent", None)
    llm = getattr(design_agent, "llm", None) if design_agent is not None else None
    return llm if llm and hasattr(llm, "run") else None


def collect_future_slide_preference_failures(
    runtime: Any,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for raw_path in slide_paths:
        slide_path = Path(raw_path)
        for spec in rule_specs:
            if not isinstance(spec, dict) or not rule_spec_applies_to_future_slides(spec):
                continue
            expected_values = verifiable_rule_css_values(spec)
            if not expected_values:
                continue
            if future_slide_rule_needs_semantic_resolution(spec):
                continue
            target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
            element_kind = canonical_session_element_kind(
                target.get("element_kind", "") if isinstance(target, dict) else "",
                str(spec.get("normalized_sentence", "") or ""),
                str(spec.get("content", "") or ""),
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
                        "requirement": rule_requirement_text(spec),
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
                        "requirement": rule_requirement_text(spec),
                        "observed_issue": "slide file unreadable",
                    }
                )
                continue

            records = extract_element_style_records(html_text, element_kind)
            matching_records = filter_records_for_expected_spans(records, [])
            failures.extend(
                build_css_value_failures(
                    slide_path=slide_path,
                    rule_id=rule_id,
                    requirement=rule_requirement_text(spec),
                    repair_guidance="",
                    element_kind=element_kind,
                    records=matching_records,
                    expected_values=expected_values,
                )
            )

    return failures


async def judge_future_slide_preferences_with_llm(
    runtime: Any,
    slide_path: Path,
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return []


async def collect_future_slide_preference_failures_async(
    runtime: Any,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return collect_future_slide_preference_failures(runtime, slide_paths, rule_specs)


def render_new_element_rule_applicability_prompt(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    plan: ModifyExecutionPlan | None,
    rule_specs: list[dict[str, Any]],
) -> str:
    return _NEW_ELEMENT_RULE_APPLICABILITY_PROMPT.format(
        user_message=str(user_message or "")[:2000],
        intent_json=json.dumps(_json_safe(intent or {}), ensure_ascii=False, indent=2),
        plan_json=json.dumps(
            {
                "scope": getattr(plan, "scope", "") if plan else "",
                "operation_kind": getattr(plan, "operation_kind", "") if plan else "",
                "target_slide_paths": [
                    workspace_relative_label(runtime, path)
                    for path in getattr(plan, "target_slide_paths", []) or []
                ],
                "coverage_required": bool(getattr(plan, "coverage_required", False)) if plan else False,
                "reason": getattr(plan, "reason", "") if plan else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        rule_specs_json=json.dumps(_json_safe(rule_specs), ensure_ascii=False, indent=2)[:8000],
    )


def normalize_new_element_rule_applications(
    runtime: Any,
    payload: dict[str, Any],
    *,
    user_message: str,
    plan: ModifyExecutionPlan | None,
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_apps = payload.get("applications", [])
    if not isinstance(raw_apps, list):
        return []

    specs_by_id = {
        str(spec.get("rule_id", "") or "").strip(): spec
        for spec in rule_specs
        if isinstance(spec, dict) and str(spec.get("rule_id", "") or "").strip()
    }
    plan_targets = list(getattr(plan, "target_slide_paths", []) or [])
    plan_target_labels = [workspace_relative_label(runtime, path) for path in plan_targets]
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    for raw_app in raw_apps:
        if not isinstance(raw_app, dict) or not _normalize_bool_flag(raw_app.get("applies", True)):
            continue
        rule_id = str(raw_app.get("rule_id", "") or "").strip()
        spec = specs_by_id.get(rule_id)
        if not spec:
            continue
        if rule_spec_applies_to_existing_slides(spec):
            continue
        evidence_spans = normalize_evidence_spans(raw_app.get("evidence_spans", []), user_message)
        if not evidence_spans:
            continue
        element_kind = canonical_session_element_kind(
            raw_app.get("element_kind", ""),
            str(raw_app.get("requirement", "") or ""),
            str(spec.get("normalized_sentence", "") or ""),
            str(spec.get("content", "") or ""),
        )
        if not element_kind or element_kind == "deck":
            continue
        raw_scope = raw_app.get("creation_scope", [])
        if isinstance(raw_scope, str):
            raw_scope = [raw_scope]
        if not isinstance(raw_scope, list):
            raw_scope = []
        creation_scope = []
        for raw_kind in raw_scope:
            kind = canonical_session_element_kind(raw_kind, element_kind)
            if kind and kind != "deck" and kind not in creation_scope:
                creation_scope.append(kind)
        if not creation_scope:
            creation_scope = rule_spec_creation_scope(spec) or [element_kind]
        condition = spec.get("condition") if isinstance(spec.get("condition"), dict) else {}
        target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
        semantic_target = target.get("semantic_target", "") if isinstance(target, dict) else ""
        raw_matched_spans = raw_app.get("matched_text_spans", [])
        if isinstance(raw_matched_spans, str):
            raw_matched_spans = [raw_matched_spans]
        if not isinstance(raw_matched_spans, list):
            raw_matched_spans = []
        matched_text_spans: list[str] = []
        for item in raw_matched_spans:
            span = " ".join(str(item or "").split()).strip()
            if span and span not in matched_text_spans:
                matched_text_spans.append(span[:120])
            if len(matched_text_spans) >= 8:
                break
        requires_span_match = (
            bool(semantic_target)
            or str(condition.get("match_granularity", "") or "").strip().lower() == "text_span"
            or str(condition.get("apply_to", "") or "").strip().lower() == "semantic_span_only"
        )
        if requires_span_match and not matched_text_spans:
            continue

        raw_targets = raw_app.get("target_slide_paths", [])
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list):
            raw_targets = []
        resolved_labels: list[str] = []
        for raw_target in raw_targets:
            text = str(raw_target or "").strip()
            if not text:
                continue
            resolved = resolve_slide_path(runtime, Path(text).stem if text.endswith(".html") else text)
            if resolved is None:
                candidate = Path(text)
                if not candidate.is_absolute():
                    candidate = runtime.workspace / candidate
                if candidate.exists():
                    resolved = candidate
            if resolved is not None:
                label = workspace_relative_label(runtime, resolved)
                if label not in resolved_labels:
                    resolved_labels.append(label)
        if not resolved_labels:
            resolved_labels = list(plan_target_labels)

        conflict_policy = str(raw_app.get("conflict_policy", "") or "none").strip().lower()
        if conflict_policy not in {"adjust_unlocked_companion_style", "fail_closed", "none"}:
            conflict_policy = "none"
        requirement = (
            str(raw_app.get("requirement", "") or "").strip()
            or str((spec.get("action") or {}).get("description", "") or "").strip()
            or str(spec.get("normalized_sentence", "") or "").strip()
            or str(spec.get("content", "") or "").strip()
        )
        app = {
            "rule_id": rule_id,
            "application_type": "future_element",
            "target_slide_paths": resolved_labels,
            "element_kind": element_kind,
            "creation_scope": creation_scope,
            "matched_text_spans": matched_text_spans,
            "requirement": requirement,
            "evidence_spans": evidence_spans,
            "conflict_policy": conflict_policy,
            "repair_guidance": str(raw_app.get("repair_guidance", "") or "").strip(),
            "rule_spec": deepcopy(spec),
        }
        key = (rule_id, element_kind, tuple(resolved_labels))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(app)
    return normalized


async def build_new_element_rule_applications(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    plan: ModifyExecutionPlan | None,
    rule_specs: list[dict[str, Any]],
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    candidate_specs = future_element_candidate_rule_specs(rule_specs)
    if not candidate_specs or is_preference_update_plan(plan):
        return []
    if plan is not None and bool(getattr(plan, "coverage_required", False)):
        return []
    if llm is None:
        llm = future_preference_judge_llm(runtime)
    if llm is None:
        append_memory_flow_trace(
            runtime,
            event="new_element_applicability_unavailable",
            user_message=user_message,
            intent=intent,
            plan=plan,
            extra={"candidate_rule_count": len(candidate_specs), "reason": "llm_unavailable"},
        )
        return []

    prompt = render_new_element_rule_applicability_prompt(
        runtime,
        user_message=user_message,
        intent=intent,
        plan=plan,
        rule_specs=candidate_specs,
    )
    raw_response = ""
    payload: dict[str, Any] = {}
    try:
        raw_response = await _run_llm_text(llm, prompt)
        payload = extract_json_object(raw_response)
    except Exception as exc:
        payload = {"error": str(exc)}
        logger.warning("New-element rule applicability LLM failed: %s", exc)
    artifacts = write_preference_semantic_artifacts(
        runtime,
        prompt=prompt,
        raw_response=raw_response,
        payload=payload,
        stage="new_element_applicability",
        source="llm" if "error" not in payload else "error",
    )
    applications = normalize_new_element_rule_applications(
        runtime,
        payload,
        user_message=user_message,
        plan=plan,
        rule_specs=candidate_specs,
    )
    append_memory_flow_trace(
        runtime,
        event="new_element_rule_applicability",
        user_message=user_message,
        intent=intent,
        plan=plan,
        extra={
            "candidate_rule_ids": [
                str(spec.get("rule_id", "") or "").strip()
                for spec in candidate_specs
            ],
            "applicable_rule_ids": [app["rule_id"] for app in applications],
            "artifacts": {
                "prompt": artifacts[0],
                "response": artifacts[1],
                "payload": artifacts[2],
            },
            "raw_payload": payload,
        },
    )
    return applications


def _rule_text_for_matching(spec: dict[str, Any]) -> str:
    action = spec.get("action") if isinstance(spec, dict) else {}
    return " ".join(
        str(item or "").strip()
        for item in [
            action.get("description", "") if isinstance(action, dict) else "",
            spec.get("normalized_sentence", "") if isinstance(spec, dict) else "",
            spec.get("content", "") if isinstance(spec, dict) else "",
        ]
        if str(item or "").strip()
    ).lower()


def rule_requirement_text(spec: dict[str, Any]) -> str:
    action = spec.get("action") if isinstance(spec, dict) else {}
    return (
        str(action.get("description", "") or "").strip()
        if isinstance(action, dict)
        else ""
    ) or str(spec.get("normalized_sentence", "") or "").strip() or str(spec.get("content", "") or "").strip()


def verifiable_rule_css_values(spec: dict[str, Any]) -> dict[str, str]:
    action = spec.get("action") if isinstance(spec, dict) else {}
    css_values = action.get("css_values", {}) if isinstance(action, dict) else {}
    if not isinstance(css_values, dict):
        return {}

    values: dict[str, str] = {}
    for raw_prop, raw_value in css_values.items():
        prop = normalize_css_property_name(str(raw_prop or ""))
        value = str(raw_value or "").strip()
        if prop and value and css_value_is_machine_verifiable(prop, value):
            values[prop] = value
    return values


def rule_design_tokens(spec: dict[str, Any]) -> dict[str, str]:
    action = spec.get("action") if isinstance(spec, dict) else {}
    tokens = action.get("design_tokens", {}) if isinstance(action, dict) else {}
    if not isinstance(tokens, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in tokens.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if key and value:
            result[key] = value
    return result


def rule_needs_llm_new_element_verification(spec: dict[str, Any]) -> bool:
    if not isinstance(spec, dict):
        return False
    if rule_design_tokens(spec):
        return True
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    css_properties = normalize_css_property_list(action.get("css_properties", []) if isinstance(action, dict) else [])
    expected_values = verifiable_rule_css_values(spec)
    if css_properties and not expected_values:
        return True
    requirement = " ".join(
        str(item or "")
        for item in [
            action.get("description", "") if isinstance(action, dict) else "",
            spec.get("normalized_sentence", ""),
            spec.get("content", ""),
        ]
        if str(item or "").strip()
    )
    return bool(extract_soft_design_tokens_from_text(requirement))


def wm_personalization_mode(runtime: Any | None = None) -> str:
    raw = ""
    if runtime is not None:
        raw = str(getattr(runtime, "wm_personalization_mode", "") or "").strip().lower()
    if not raw:
        raw = os.environ.get("MEMSLIDES_WM_PERSONALIZATION_MODE", "").strip().lower()
    if raw in {"off", "light", "eval", "strict"}:
        return raw
    return "light"


def _soft_rule_text(spec: dict[str, Any]) -> str:
    action = spec.get("action") if isinstance(spec, dict) else {}
    parts = [
        str(spec.get("normalized_sentence", "") or ""),
        str(spec.get("content", "") or ""),
        str(action.get("description", "") or "") if isinstance(action, dict) else "",
    ]
    return " ".join(part for part in parts if part).lower()


def _style_value_mentions_blur_or_dim(value: str) -> bool:
    text = str(value or "").strip().lower()
    if "blur(" in text:
        return True
    if "brightness(" in text:
        match = re.search(r"brightness\(\s*([0-9.]+)%?\s*\)", text)
        if match:
            try:
                return float(match.group(1)) < (80 if "%" in text else 0.8)
            except ValueError:
                return True
        return True
    return False


def collect_soft_preference_evaluations(
    runtime: Any,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return non-blocking preference observations for fuzzy/content/layout rules."""
    if wm_personalization_mode(runtime) in {"off", "strict"}:
        return []
    evaluations: list[dict[str, Any]] = []
    candidate_rules = [
        spec
        for spec in rule_specs
        if isinstance(spec, dict)
        and not verifiable_rule_css_values(spec)
        and str(spec.get("rule_type", "") or "").strip().lower() in {"content", "layout", "constraint", "style"}
    ]
    if not candidate_rules or not slide_paths:
        return []
    for raw_path in slide_paths[:20]:
        slide_path = Path(raw_path)
        try:
            html_text = slide_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        soup = BeautifulSoup(html_text, "html.parser")
        visible_text = " ".join(soup.get_text(" ", strip=True).split())
        styles = [parse_style_declarations(tag.get("style", "")) for tag in soup.find_all(True)]
        for spec in candidate_rules:
            rule_text = _soft_rule_text(spec)
            if not rule_text:
                continue
            rule_id = str(spec.get("rule_id", "") or "").strip() or "wm_soft_rule"
            if re.search(r"(最多|不超过|少于|at most|no more than).{0,8}(3|三).{0,12}(要点|bullet|项目)", rule_text):
                bullet_count = len(soup.find_all("li"))
                if bullet_count > 3:
                    evaluations.append(
                        {
                            "severity": "soft_warning",
                            "rule_id": rule_id,
                            "slide_path": workspace_relative_label(runtime, slide_path),
                            "signal": "bullet_count_exceeds_soft_preference",
                            "observed": bullet_count,
                            "expected": "at most 3 bullets",
                        }
                    )
            if re.search(r"(谨慎|克制|不要夸大|避免夸大|cautious|conservative|avoid hype)", rule_text):
                hype_terms = ["革命性", "显著碾压", "dramatically", "revolutionary", "breakthrough"]
                found_terms = [term for term in hype_terms if term.lower() in visible_text.lower()]
                if found_terms:
                    evaluations.append(
                        {
                            "severity": "soft_warning",
                            "rule_id": rule_id,
                            "slide_path": workspace_relative_label(runtime, slide_path),
                            "signal": "possibly_overstated_language",
                            "observed": found_terms[:5],
                        }
                    )
            if re.search(r"(不要|避免|禁止).{0,12}(模糊|暗化|裁剪|blur|dim|crop)", rule_text):
                for style in styles:
                    if any(_style_value_mentions_blur_or_dim(value) for value in style.values()):
                        evaluations.append(
                            {
                                "severity": "soft_warning",
                                "rule_id": rule_id,
                                "slide_path": workspace_relative_label(runtime, slide_path),
                                "signal": "possibly_dimmed_or_blurred_media",
                            }
                        )
                        break
    return evaluations


def _normalize_title_prefix_value(value: Any) -> str:
    prefix = re.sub(r"\s+", "", str(value or "")).strip(" \"'`“”‘’")
    prefix = re.sub(r"(?:\.\.\.|…)+$", "", prefix)
    if not prefix:
        return ""
    if prefix == "结论":
        return "结论："
    return prefix


def _title_prefix_from_rule_spec(spec: dict[str, Any]) -> str:
    if not isinstance(spec, dict):
        return ""
    try:
        blob = json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        blob = str(spec)
    target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    design_tokens = action.get("design_tokens") if isinstance(action.get("design_tokens"), dict) else {}
    element_kind = str(target.get("element_kind", "") or "").strip().lower()
    if element_kind not in {"slide_title", "title", "heading", "h1"} and "slide_title" not in blob:
        return ""

    for key in ("title.prefix", "title_prefix", "prefix"):
        prefix = _normalize_title_prefix_value(design_tokens.get(key))
        if prefix:
            return prefix

    quoted_prefix = re.search(
        r"(?:标题|slide_title|title)[^。；\n]{0,80}?[\"“']([^\"”'\n]{1,32}[:：])(?:\.\.\.|…)?[\"”']",
        blob,
        flags=re.IGNORECASE,
    )
    if quoted_prefix:
        prefix = _normalize_title_prefix_value(quoted_prefix.group(1))
        if prefix:
            return prefix

    prefix_before_word = re.search(
        r"[\"“']([^\"”'\n]{1,32}[:：])(?:\.\.\.|…)?[\"”'][^。；\n]{0,40}(?:开头|前缀|prefix)",
        blob,
        flags=re.IGNORECASE,
    )
    if prefix_before_word:
        prefix = _normalize_title_prefix_value(prefix_before_word.group(1))
        if prefix:
            return prefix

    if "结论：" in blob and ("标题" in blob or "slide_title" in blob or "title" in blob.lower()):
        return "结论："
    return ""


def _title_prefix_rule_applies_to_existing_slides(spec: dict[str, Any]) -> bool:
    target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
    propagation = spec.get("propagation") if isinstance(spec.get("propagation"), dict) else {}
    verification = spec.get("verification") if isinstance(spec.get("verification"), dict) else {}
    slide_scope = str(target.get("slide_scope", "") or "").strip().lower()
    apply_existing = propagation.get("apply_existing_slides")
    return (
        slide_scope in {"", "all", "existing", "current", "deck"}
        or bool(verification.get("must_cover_all_targets"))
        or apply_existing is True
    )


def _title_with_required_prefix(text: str, prefix: str) -> str:
    title = re.sub(r"\s+", " ", str(text or "")).strip()
    prefix = _normalize_title_prefix_value(prefix)
    if not title or not prefix:
        return title
    if title.startswith(prefix):
        return title
    prefix_stem = prefix.rstrip(":：")
    if prefix_stem and title.startswith(prefix_stem):
        remainder = title[len(prefix_stem):].lstrip(" \t:：-—–")
        if len(remainder) > 1 and remainder[0] in {"与", "和", "及"}:
            remainder = remainder[1:].lstrip(" \t:：-—–")
        return prefix + (remainder or prefix_stem)
    return prefix + title


def enforce_hard_title_prefix_preferences(
    runtime: Any,
    slide_paths: list[Path],
    rule_specs: list[dict[str, Any]],
    *,
    user_message: str = "",
) -> list[dict[str, Any]]:
    """Deterministically enforce existing-slide title-prefix WM rules before export.

    This intentionally handles only concrete prefix rules such as title.prefix =
    "结论：". Fuzzy language-style preferences remain non-blocking observations.
    """
    prefixes: list[str] = []
    for spec in rule_specs or []:
        prefix = _title_prefix_from_rule_spec(spec)
        if prefix and _title_prefix_rule_applies_to_existing_slides(spec):
            prefixes.append(prefix)
    if not prefixes:
        synthetic = _title_prefix_from_rule_spec(
            {
                "target": {"element_kind": "slide_title", "slide_scope": "all"},
                "action": {"description": user_message or ""},
                "natural_language": user_message or "",
            }
        )
        if synthetic:
            prefixes.append(synthetic)
    if not prefixes:
        return []

    prefix = prefixes[0]
    results: list[dict[str, Any]] = []
    for raw_path in slide_paths or []:
        path = Path(raw_path)
        entry: dict[str, Any] = {
            "slide_path": workspace_relative_label(runtime, path),
            "prefix": prefix,
        }
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            entry.update({"status": "error", "error": str(exc)})
            results.append(entry)
            continue
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("h1")
        if title_tag is None:
            title_tag = soup.find(attrs={"data-role": "title"})
        if title_tag is None:
            for candidate in soup.find_all(True):
                classes = {str(item).lower() for item in (candidate.get("class") or [])}
                if classes & {"title", "heading", "slide-title"}:
                    title_tag = candidate
                    break
        if title_tag is None:
            entry.update({"status": "error", "error": "slide_title_not_found"})
            results.append(entry)
            continue

        old_title = title_tag.get_text(" ", strip=True)
        new_title = _title_with_required_prefix(old_title, prefix)
        if new_title == old_title:
            continue
        title_tag.clear()
        title_tag.append(new_title)
        head_title = soup.find("title")
        if head_title is not None:
            head_title.clear()
            head_title.append(new_title)
        path.write_text(str(soup), encoding="utf-8")
        entry.update(
            {
                "status": "repaired",
                "old_title": old_title,
                "new_title": new_title,
            }
        )
        results.append(entry)
    return results


def future_slide_rule_needs_semantic_resolution(spec: dict[str, Any]) -> bool:
    target = spec.get("target") if isinstance(spec, dict) else {}
    condition = spec.get("condition") if isinstance(spec, dict) else {}
    if not isinstance(condition, dict):
        condition = {}
    semantic_target = target.get("semantic_target", "") if isinstance(target, dict) else ""
    return (
        bool(semantic_target)
        or str(condition.get("match_granularity", "") or "").strip().lower() == "text_span"
        or str(condition.get("apply_to", "") or "").strip().lower() == "semantic_span_only"
    )


def record_css_value(record: dict[str, str], property_name: str) -> str:
    prop = normalize_css_property_name(property_name)
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


def filter_records_for_expected_spans(
    records: list[dict[str, str]],
    matched_text_spans: list[str],
) -> list[dict[str, str]]:
    spans = [
        " ".join(str(span or "").split()).strip()
        for span in matched_text_spans
        if str(span or "").strip()
    ]
    if not spans:
        return records
    matching: list[dict[str, str]] = []
    for span in spans:
        exact = [
            record
            for record in records
            if " ".join(str(record.get("text", "") or "").split()) == span
        ]
        if exact:
            for record in exact:
                if record not in matching:
                    matching.append(record)
            continue
        containing = [
            record
            for record in records
            if span in " ".join(str(record.get("text", "") or "").split())
        ]
        containing.sort(key=lambda record: len(str(record.get("text", "") or "")))
        for record in containing[:1]:
            if record not in matching:
                matching.append(record)
    return matching


def build_css_value_failures(
    *,
    slide_path: Path,
    rule_id: str,
    requirement: str,
    repair_guidance: str,
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
                "repair_guidance": repair_guidance,
                "element_kind": element_kind,
                "element_text": "",
            }
        ]

    failures: list[dict[str, Any]] = []
    for record in records:
        for prop, expected in expected_values.items():
            observed = record_css_value(record, prop)
            if observed and css_values_equivalent(prop, expected, observed):
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
                    "repair_guidance": repair_guidance,
                    "expected_property": prop,
                    "expected_value": expected,
                    "observed_value": observed or "",
                    "element_kind": element_kind,
                    "element_text": str(record.get("text", "") or ""),
                }
            )
    return failures


def _css_rule_declarations(soup: BeautifulSoup) -> tuple[dict[str, str], list[tuple[str, dict[str, str]]]]:
    css_vars: dict[str, str] = {}
    rules: list[tuple[str, dict[str, str]]] = []
    for style_tag in soup.find_all("style"):
        css_text = style_tag.string or style_tag.get_text() or ""
        for match in re.finditer(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", css_text, re.DOTALL):
            selectors_text = str(match.group("selectors") or "").strip()
            if not selectors_text or selectors_text.startswith("@"):
                continue
            declarations = parse_style_declarations(match.group("body"))
            for key, value in declarations.items():
                if key.startswith("--"):
                    css_vars[key] = value
            for raw_selector in selectors_text.split(","):
                selector = raw_selector.strip()
                if selector:
                    rules.append((selector, declarations))
    return css_vars, rules


def _node_matches_selector(soup: BeautifulSoup, node: Any, selector: str) -> bool:
    try:
        return node in soup.select(selector)
    except Exception:
        selector_text = str(selector or "").strip().lower()
        if not selector_text:
            return False
        classes = {
            str(item).strip().lower()
            for item in (node.get("class", []) or [])
            if str(item).strip()
        }
        name = str(getattr(node, "name", "") or "").strip().lower()
        data_role = str(node.get("data-role", "") or "").strip().lower()
        if selector_text.startswith("."):
            return selector_text[1:] in classes
        if selector_text == name:
            return True
        if "data-role" in selector_text:
            return data_role and data_role in selector_text
        return False


def extract_element_style_records(html_text: str, element_kind: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    selectors = selector_hints_for_element_kind(element_kind)
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

    css_vars, rules = _css_rule_declarations(soup)
    body_declarations = parse_style_declarations((soup.body or {}).get("style", "") if soup.body else "")
    records: list[dict[str, str]] = []
    for node in nodes:
        inline = parse_style_declarations(node.get("style", ""))
        computed: dict[str, str] = {}
        for prop, value in body_declarations.items():
            normalized_prop = normalize_css_property_name(prop)
            if normalized_prop:
                computed[normalized_prop] = resolve_css_var_value(value, css_vars)
        matched_selectors: list[str] = []
        for selector, declarations in rules:
            if not _node_matches_selector(soup, node, selector):
                continue
            matched_selectors.append(selector)
            for prop, value in declarations.items():
                normalized_prop = normalize_css_property_name(prop)
                if normalized_prop:
                    computed[normalized_prop] = resolve_css_var_value(value, css_vars)
        for prop, value in inline.items():
            normalized_prop = normalize_css_property_name(prop)
            if normalized_prop:
                computed[normalized_prop] = resolve_css_var_value(value, css_vars)
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


def _new_element_records(before_html: str, after_html: str, element_kind: str) -> list[dict[str, str]]:
    before_records = extract_element_style_records(before_html, element_kind)
    after_records = extract_element_style_records(after_html, element_kind)
    before_counts: Counter[str] = Counter(record.get("text", "") for record in before_records)
    new_records: list[dict[str, str]] = []
    for record in after_records:
        key = record.get("text", "")
        if before_counts[key] > 0:
            before_counts[key] -= 1
            continue
        new_records.append(record)
    return new_records


def render_new_element_rule_verifier_prompt(
    runtime: Any,
    *,
    user_message: str,
    slide_path: Path,
    before_html: str,
    after_html: str,
    applications: list[dict[str, Any]],
    new_records_by_rule: dict[str, list[dict[str, str]]],
) -> str:
    return _NEW_ELEMENT_RULE_VERIFIER_PROMPT.format(
        user_message=str(user_message or "")[:2000],
        slide_path=workspace_relative_label(runtime, slide_path),
        applications_json=json.dumps(_json_safe(applications), ensure_ascii=False, indent=2)[:10000],
        new_records_json=json.dumps(_json_safe(new_records_by_rule), ensure_ascii=False, indent=2)[:8000],
        before_html=str(before_html or "")[:20000],
        after_html=str(after_html or "")[:24000],
    )


def _new_element_app_matches_slide(runtime: Any, app: dict[str, Any], slide_path: Path) -> bool:
    labels = {
        str(item or "").strip()
        for item in app.get("target_slide_paths", []) or []
        if str(item or "").strip()
    }
    return not labels or workspace_relative_label(runtime, slide_path) in labels


def _application_element_text_hint(app: dict[str, Any]) -> str:
    for span in app.get("matched_text_spans", []) or []:
        text = " ".join(str(span or "").split()).strip()
        if text:
            return text
    return ""


def _new_element_applications_requiring_llm(applications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        spec = app.get("rule_spec") if isinstance(app.get("rule_spec"), dict) else {}
        if rule_needs_llm_new_element_verification(spec):
            result.append(app)
    return result


def _new_element_records_for_application(before_html: str, after_html: str, app: dict[str, Any]) -> list[dict[str, str]]:
    element_kind = canonical_session_element_kind(app.get("element_kind", ""))
    records = _new_element_records(before_html, after_html, element_kind)
    return filter_records_for_expected_spans(
        records,
        [
            str(span or "")
            for span in app.get("matched_text_spans", []) or []
            if str(span or "").strip()
        ],
    )


def _new_element_llm_failures_from_payload(
    *,
    slide_path: Path,
    applications: list[dict[str, Any]],
    payload: dict[str, Any],
    fallback_reason: str = "",
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        return [
            {
                "slide_path": slide_path,
                "rule_id": str(app.get("rule_id", "") or "wm_rule"),
                "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                "observed_issue": fallback_reason or "LLM verifier returned no parseable judgement for fuzzy/design-token WM rule",
                "repair_guidance": str(app.get("repair_guidance", "") or ""),
                "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                "element_text": _application_element_text_hint(app),
                "verifier": "llm",
                "fail_closed": True,
            }
            for app in applications
        ]

    by_rule = {
        str(app.get("rule_id", "") or "").strip(): app
        for app in applications
        if str(app.get("rule_id", "") or "").strip()
    }
    failures: list[dict[str, Any]] = []
    raw_violations = payload.get("violations", [])
    if isinstance(raw_violations, list):
        for item in raw_violations:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id", "") or "").strip()
            app = by_rule.get(rule_id, {})
            failures.append(
                {
                    "slide_path": slide_path,
                    "rule_id": rule_id or str(app.get("rule_id", "") or "wm_rule"),
                    "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                    "observed_issue": str(item.get("observed_issue", "") or item.get("reason", "") or "LLM verifier judged the new element violates the fuzzy WM rule"),
                    "repair_guidance": str(item.get("repair_instructions", "") or item.get("repair_guidance", "") or app.get("repair_guidance", "") or ""),
                    "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                    "element_text": str(item.get("element_text", "") or _application_element_text_hint(app)),
                    "verifier": "llm",
                }
            )

    raw_judgements = payload.get("judgements", [])
    if isinstance(raw_judgements, list):
        for item in raw_judgements:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "") or "").strip().lower()
            if verdict not in {"fail", "uncertain"}:
                continue
            rule_id = str(item.get("rule_id", "") or "").strip()
            app = by_rule.get(rule_id, {})
            failures.append(
                {
                    "slide_path": slide_path,
                    "rule_id": rule_id or str(app.get("rule_id", "") or "wm_rule"),
                    "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                    "observed_issue": str(item.get("reason", "") or f"LLM verifier returned {verdict} for fuzzy/design-token WM rule"),
                    "repair_guidance": str(item.get("repair_guidance", "") or app.get("repair_guidance", "") or ""),
                    "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                    "element_text": str(item.get("element_text", "") or _application_element_text_hint(app)),
                    "verifier": "llm",
                    "fail_closed": verdict == "uncertain",
                }
            )

    if failures:
        return failures
    if payload.get("passed") is True:
        return []
    if payload.get("passed") is False:
        return [
            {
                "slide_path": slide_path,
                "rule_id": str(app.get("rule_id", "") or "wm_rule"),
                "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                "observed_issue": "LLM verifier reported the new slide elements do not satisfy fuzzy/design-token WM preference, but returned no detailed violation",
                "repair_guidance": str(app.get("repair_guidance", "") or ""),
                "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                "element_text": _application_element_text_hint(app),
                "verifier": "llm",
            }
            for app in applications
        ]
    return [
        {
            "slide_path": slide_path,
            "rule_id": str(app.get("rule_id", "") or "wm_rule"),
            "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
            "observed_issue": fallback_reason or "LLM verifier did not explicitly pass fuzzy/design-token WM rule",
            "repair_guidance": str(app.get("repair_guidance", "") or ""),
            "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
            "element_text": _application_element_text_hint(app),
            "verifier": "llm",
            "fail_closed": True,
        }
        for app in applications
    ]


def collect_new_element_preference_failures(
    runtime: Any,
    *,
    before_html_by_path: dict[str, str],
    changed_slide_paths: list[Path],
    applications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not applications or not changed_slide_paths:
        return failures
    changed_by_label = {
        workspace_relative_label(runtime, path): path
        for path in changed_slide_paths
    }
    for app in applications:
        if not isinstance(app, dict):
            continue
        rule_id = str(app.get("rule_id", "") or "").strip()
        spec = app.get("rule_spec") if isinstance(app.get("rule_spec"), dict) else {}
        expected_values = verifiable_rule_css_values(spec)
        element_kind = canonical_session_element_kind(app.get("element_kind", ""))
        if not rule_id or not expected_values:
            continue
        target_labels = {
            str(item or "").strip()
            for item in app.get("target_slide_paths", []) or []
            if str(item or "").strip()
        }
        candidate_paths = [
            path
            for label, path in changed_by_label.items()
            if not target_labels or label in target_labels
        ]
        for slide_path in candidate_paths:
            before_html = before_html_by_path.get(str(slide_path.resolve()), "")
            try:
                after_html = slide_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                failures.append(
                    {
                        "slide_path": slide_path,
                        "rule_id": rule_id,
                        "requirement": app.get("requirement", ""),
                        "observed_issue": "changed slide file is unreadable",
                        "repair_guidance": app.get("repair_guidance", ""),
                    }
                )
                continue
            new_records = _new_element_records(before_html, after_html, element_kind)
            if not new_records:
                if expected_values:
                    failures.append(
                        {
                            "slide_path": slide_path,
                            "rule_id": rule_id,
                            "requirement": app.get("requirement", ""),
                            "observed_issue": f"no newly created {element_kind} element was detected for an applicable concrete CSS rule",
                            "repair_guidance": app.get("repair_guidance", ""),
                            "element_kind": element_kind,
                            "element_text": _application_element_text_hint(app),
                        }
                    )
                continue
            matched_records = filter_records_for_expected_spans(
                new_records,
                [
                    str(span or "")
                    for span in app.get("matched_text_spans", []) or []
                    if str(span or "").strip()
                ],
            )
            failures.extend(
                build_css_value_failures(
                    slide_path=slide_path,
                    rule_id=rule_id,
                    requirement=str(app.get("requirement", "") or rule_requirement_text(spec)),
                    repair_guidance=str(app.get("repair_guidance", "") or ""),
                    element_kind=element_kind,
                    records=matched_records,
                    expected_values=expected_values,
                )
            )
    return failures


async def judge_new_element_preferences_with_llm(
    runtime: Any,
    *,
    slide_path: Path,
    before_html: str,
    applications: list[dict[str, Any]],
    user_message: str = "",
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    if not applications:
        return []
    try:
        after_html = Path(slide_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [
            {
                "slide_path": slide_path,
                "rule_id": str(app.get("rule_id", "") or "wm_rule"),
                    "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                    "observed_issue": "changed slide file is unreadable before LLM verifier",
                    "repair_guidance": str(app.get("repair_guidance", "") or ""),
                    "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                    "element_text": _application_element_text_hint(app),
                    "verifier": "llm",
                    "fail_closed": True,
                }
            for app in applications
        ]

    llm_apps = _new_element_applications_requiring_llm(applications)
    if not llm_apps:
        return []
    llm_apps = [
        app
        for app in llm_apps
        if _new_element_app_matches_slide(runtime, app, Path(slide_path))
    ]
    if not llm_apps:
        return []

    new_records_by_rule: dict[str, list[dict[str, str]]] = {}
    missing_new_element_failures: list[dict[str, Any]] = []
    for app in llm_apps:
        rule_id = str(app.get("rule_id", "") or "wm_rule")
        records = _new_element_records_for_application(before_html, after_html, app)
        new_records_by_rule[rule_id] = records
        if not records:
            missing_new_element_failures.append(
                {
                    "slide_path": slide_path,
                    "rule_id": rule_id,
                    "requirement": str(app.get("requirement", "") or rule_requirement_text(app.get("rule_spec", {}))),
                    "observed_issue": f"no newly created {canonical_session_element_kind(app.get('element_kind', ''))} element was detected for fuzzy/design-token WM verification",
                    "repair_guidance": str(app.get("repair_guidance", "") or ""),
                    "element_kind": canonical_session_element_kind(app.get("element_kind", "")),
                    "element_text": _application_element_text_hint(app),
                    "verifier": "llm",
                    "fail_closed": True,
                }
            )
    if missing_new_element_failures:
        return missing_new_element_failures

    if llm is None:
        llm = future_preference_judge_llm(runtime)
    if llm is None:
        failures = _new_element_llm_failures_from_payload(
            slide_path=Path(slide_path),
            applications=llm_apps,
            payload={},
            fallback_reason="LLM verifier unavailable for fuzzy/design-token WM rule",
        )
        append_memory_flow_trace(
            runtime,
            event="new_element_llm_verifier_failed",
            extra={
                "slide_path": workspace_relative_label(runtime, slide_path),
                "reason": "llm_unavailable",
                "failures": _json_safe(failures[:12]),
            },
        )
        return failures

    prompt = render_new_element_rule_verifier_prompt(
        runtime,
        user_message=user_message,
        slide_path=Path(slide_path),
        before_html=before_html,
        after_html=after_html,
        applications=llm_apps,
        new_records_by_rule=new_records_by_rule,
    )
    append_memory_flow_trace(
        runtime,
        event="new_element_llm_verifier_started",
        extra={
            "slide_path": workspace_relative_label(runtime, slide_path),
            "rule_ids": [str(app.get("rule_id", "") or "") for app in llm_apps],
            "new_record_counts": {rule_id: len(records) for rule_id, records in new_records_by_rule.items()},
        },
    )
    raw_response = ""
    payload: dict[str, Any] = {}
    source = "llm"
    try:
        raw_response = await _run_llm_text(llm, prompt)
        payload = extract_json_object(raw_response)
        if not payload:
            source = "empty"
    except Exception as exc:
        source = "error"
        payload = {"error": str(exc)}
        logger.warning("New-element LLM verifier failed: %s", exc)
    artifacts = write_preference_semantic_artifacts(
        runtime,
        prompt=prompt,
        raw_response=raw_response,
        payload=payload,
        stage="new_element_verifier",
        source=source,
    )
    failures = _new_element_llm_failures_from_payload(
        slide_path=Path(slide_path),
        applications=llm_apps,
        payload=payload,
        fallback_reason=(
            str(payload.get("error", "") or "")
            if isinstance(payload, dict) and payload.get("error")
            else "LLM verifier failed closed for fuzzy/design-token WM rule"
        ),
    )
    append_memory_flow_trace(
        runtime,
        event="new_element_llm_verifier_failed" if failures else "new_element_llm_verifier_succeeded",
        extra={
            "slide_path": workspace_relative_label(runtime, slide_path),
            "rule_ids": [str(app.get("rule_id", "") or "") for app in llm_apps],
            "passed": not bool(failures),
            "failures": _json_safe(failures[:12]),
            "artifacts": {
                "prompt": artifacts[0],
                "response": artifacts[1],
                "payload": artifacts[2],
            },
            "raw_payload": _json_safe(payload),
        },
    )
    return failures


async def collect_new_element_preference_failures_async(
    runtime: Any,
    *,
    before_html_by_path: dict[str, str],
    changed_slide_paths: list[Path],
    applications: list[dict[str, Any]],
    user_message: str = "",
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    failures = collect_new_element_preference_failures(
        runtime,
        before_html_by_path=before_html_by_path,
        changed_slide_paths=changed_slide_paths,
        applications=applications,
    )
    changed_by_label = {
        workspace_relative_label(runtime, path): path
        for path in changed_slide_paths
    }
    for label, slide_path in changed_by_label.items():
        slide_apps = [
            app
            for app in applications
            if isinstance(app, dict)
            and _new_element_app_matches_slide(runtime, app, slide_path)
            and rule_needs_llm_new_element_verification(
                app.get("rule_spec") if isinstance(app.get("rule_spec"), dict) else {}
            )
        ]
        if not slide_apps:
            continue
        failures.extend(
            await judge_new_element_preferences_with_llm(
                runtime,
                slide_path=slide_path,
                before_html=before_html_by_path.get(str(slide_path.resolve()), ""),
                applications=slide_apps,
                user_message=user_message,
                llm=llm,
            )
        )
    if failures:
        append_memory_flow_trace(
            runtime,
            event="new_element_verifier_failed",
            plan=None,
            extra={
                "failures": _json_safe(failures[:12]),
                "changed_slide_paths": [workspace_relative_label(runtime, path) for path in changed_slide_paths],
            },
        )
    return failures


def _new_element_preference_repair_hint(runtime: Any, item: dict[str, Any]) -> str:
    element_text = " ".join(str(item.get("element_text", "") or "").split()).strip()
    element_kind = str(item.get("element_kind", "") or "").strip()
    slide_label = workspace_relative_label(runtime, item.get("slide_path", ""))
    if not element_text and not element_kind:
        return ""
    snapshot_args = [json.dumps(slide_label, ensure_ascii=False)]
    planner_args = [json.dumps(slide_label, ensure_ascii=False)]
    planner_kwargs = [
        "edit_intent=" + json.dumps(
            str(item.get("repair_guidance", "") or item.get("requirement", "") or "repair new element preference"),
            ensure_ascii=False,
        )
    ]
    if element_text:
        snapshot_args.append("focus_text=" + json.dumps(element_text, ensure_ascii=False))
        planner_kwargs.append("focus_text=" + json.dumps(element_text, ensure_ascii=False))
    if element_kind:
        snapshot_args.append("focus_kind=" + json.dumps(element_kind, ensure_ascii=False))
        planner_kwargs.append("target_scope=" + json.dumps(element_kind, ensure_ascii=False))
        planner_kwargs.append("focus_kind=" + json.dumps(element_kind, ensure_ascii=False))
    expected_property = str(item.get("expected_property", "") or "").strip()
    if expected_property:
        planner_kwargs.append("requested_properties=[" + json.dumps(expected_property, ensure_ascii=False) + "]")
    return (
        " Next repair path: call `read_slide_snapshot("
        + ", ".join(snapshot_args)
        + ")`, then `plan_slide_patch("
        + ", ".join(planner_args + planner_kwargs)
        + ")`, then apply a concrete `apply_slide_patch` op to the matched_focus target."
    )


def build_new_element_preference_followup(
    runtime: Any,
    failures: list[dict[str, Any]],
    *,
    include_header: bool = True,
) -> str:
    if not failures:
        return ""
    lines: list[str] = []
    has_conflict = False
    for item in failures[:8]:
        has_conflict = has_conflict or str(item.get("conflict", "") or "") == "preference_quality_conflict"
        guidance = str(item.get("repair_guidance", "") or "").strip()
        suffix = f"; repair: {guidance}" if guidance else ""
        lines.append(
            f"- {workspace_relative_label(runtime, item.get('slide_path', ''))}: "
            f"{str(item.get('observed_issue', '') or '').strip()}; active WM rule says "
            f"{str(item.get('requirement', '') or item.get('rule_id', '')).strip()}{suffix}"
            f"{_new_element_preference_repair_hint(runtime, item)}"
        )
    prefix = (
        "SYSTEM: New elements created in this modify turn do not yet follow applicable remembered working-memory preferences.\n"
        if include_header
        else ""
    )
    conflict_line = (
        "\nWhen contrast conflicts with the remembered text-color preference, preserve the preference and adjust unlocked companion styles such as the tag background. If the user explicitly locked the conflicting companion style, stop and report the conflict."
        if has_conflict
        else ""
    )
    return (
        prefix
        + "\n".join(lines)
        + "\nRepair only the newly created element(s) on the changed current slide(s). "
        "Do not batch-edit old slides or old matching elements, and do not call `finalize` until these new elements comply."
        + conflict_line
    )


def build_future_preference_followup(
    runtime: Any,
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
            f"- {workspace_relative_label(runtime, item['slide_path'])}: "
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


def build_modify_execution_plan(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    wm_rule_specs_text: str,
) -> ModifyExecutionPlan | None:
    """Compile current-turn global intent into an explicit modify plan."""
    all_slide_paths = resolve_all_slide_paths(runtime)
    if not all_slide_paths:
        return None

    semantic_compilation = getattr(runtime, "_last_preference_semantic_compilation", None)
    if (
        isinstance(semantic_compilation, PreferenceSemanticCompilation)
        and semantic_compilation.fail_closed
    ):
        structural_slide_operation = is_structural_slide_operation(user_message, intent or {})
        if structural_slide_operation:
            return ModifyExecutionPlan(
                scope="global",
                reason=(
                    "Preference semantic compiler failed closed; proceed only with the requested "
                    "structural/current action and do not propagate a memory preference."
                ),
                target_slide_paths=[],
                target_rule_ids=[],
                selector_hints=[],
                coverage_required=False,
                operation_kind="structural",
            )
        intent_target = str((intent or {}).get("target_slide", "") or "").strip()
        specific_slide_path = None
        if intent_target and intent_target.lower() != "all":
            specific_slide_path = resolve_slide_path(runtime, intent_target)
        if specific_slide_path is None:
            inferred_slide_ref = infer_specific_existing_slide_reference(user_message)
            if inferred_slide_ref:
                specific_slide_path = resolve_slide_path(runtime, inferred_slide_ref)
        if specific_slide_path is not None:
            return ModifyExecutionPlan(
                scope="local",
                reason=(
                    "Preference semantic compiler failed closed; execute only the explicit current "
                    "slide edit and do not write or propagate a memory rule."
                ),
                target_slide_paths=[specific_slide_path],
                target_rule_ids=[],
                selector_hints=[],
                coverage_required=False,
                operation_kind="style",
            )
        return None

    text_color_boundary = request_mentions_text_color_without_background(user_message)
    intent = intent or {}
    intent_target = str(intent.get("target_slide", "") or "").strip()
    specific_slide_path = None
    if intent_target and intent_target.lower() != "all":
        specific_slide_path = resolve_slide_path(runtime, intent_target)
    if specific_slide_path is None:
        inferred_slide_ref = infer_specific_existing_slide_reference(user_message)
        if inferred_slide_ref:
            specific_slide_path = resolve_slide_path(runtime, inferred_slide_ref)

    all_rule_specs = parse_rule_specs_from_injection(wm_rule_specs_text)
    current_turn_specs: list[dict[str, Any]] = []
    active_future_element_specs = future_element_candidate_rule_specs(all_rule_specs)
    active_future_element_rule_ids, active_future_element_selector_hints = collect_rule_ids_and_selector_hints(
        active_future_element_specs
    )
    for spec in all_rule_specs:
        try:
            source_turn = int(spec.get("source_turn", -1))
        except (TypeError, ValueError):
            source_turn = -1
        if source_turn == runtime._modify_turn_count:
            current_turn_specs.append(spec)

    structural_slide_operation = is_structural_slide_operation(user_message, intent)
    diagram_contract = classify_diagram_layout_intent(user_message, intent)
    if diagram_contract is not None and specific_slide_path is not None:
        diagram_contract = dict(diagram_contract)
        diagram_contract["slide_path"] = workspace_relative_label(runtime, specific_slide_path)
        return ModifyExecutionPlan(
            scope="local",
            reason=(
                "Existing-slide diagram/pipeline layout request detected; allow one "
                "controlled full-slide rewrite for the resolved target slide."
            ),
            target_slide_paths=[specific_slide_path],
            target_rule_ids=[],
            selector_hints=[".flowchart", ".pipeline", ".diagram", "svg", "img"],
            coverage_required=True,
            operation_kind="diagram_layout",
            applicable_rule_ids=active_future_element_rule_ids,
            diagram_contract=diagram_contract,
        )
    user_has_global_signal = (
        session_preference_has_general_signal(user_message)
        or intent_target.lower() == "all"
    ) and not structural_slide_operation
    current_turn_global_specs = [
        spec
        for spec in current_turn_specs
        if str(spec.get("scope", "") or "").strip().lower() == "global"
    ]
    current_turn_existing_specs = [
        spec for spec in current_turn_global_specs
        if rule_spec_applies_to_existing_slides(spec)
    ]
    current_turn_future_only_specs = [
        spec for spec in current_turn_global_specs
        if not rule_spec_applies_to_existing_slides(spec)
    ]
    future_rule_ids, future_selector_hints = collect_rule_ids_and_selector_hints(
        current_turn_future_only_specs
    )
    if semantic_compilation is None and not current_turn_existing_specs and not current_turn_future_only_specs:
        explicit_existing_batch = (
            intent_target.lower() == "all"
            and request_explicitly_targets_existing_slides(user_message)
            and not re.search(
                r"(以后|未来|后续|新增|新生成|新建|新插入|future|new).{0,24}(都|使用|应用|继承|默认|保持|设为|改成|为)",
                " ".join(str(user_message or "").split()).lower(),
            )
        )
        if not explicit_existing_batch:
            if specific_slide_path is not None:
                return ModifyExecutionPlan(
                    scope="local",
                    reason=(
                        "No reliable preference semantic compilation is available; execute only "
                        "the explicit current slide target and do not infer deck-wide propagation "
                        "from future/general wording."
                    ),
                    target_slide_paths=[specific_slide_path],
                    target_rule_ids=[],
                    selector_hints=selector_hints_for_element_kind(
                        canonical_session_element_kind(
                            str((intent or {}).get("element_type", "") or ""),
                            user_message,
                        )
                    ),
                    coverage_required=False,
                    operation_kind="style",
                    applicable_rule_ids=active_future_element_rule_ids,
                )
            if session_preference_has_general_signal(user_message):
                return ModifyExecutionPlan(
                    scope="local",
                    reason=(
                        "No reliable preference semantic compilation is available; this looks like "
                        "a memory/general preference turn, so no existing slide targets are inferred."
                    ),
                    target_slide_paths=[],
                    target_rule_ids=[],
                    selector_hints=[],
                    coverage_required=False,
                    operation_kind="preference_update",
                    applicable_rule_ids=active_future_element_rule_ids,
                )

    if (
        current_turn_future_only_specs
        and not current_turn_existing_specs
        and not structural_slide_operation
    ):
        if specific_slide_path is not None:
            return ModifyExecutionPlan(
                scope="local",
                reason=(
                    "The current request includes a concrete existing-slide edit plus "
                    "future-only preference rule(s). Edit only the resolved existing slide "
                    "now; keep the future-only rule for later inserted/generated slides."
                ),
                target_slide_paths=[specific_slide_path],
                target_rule_ids=future_rule_ids,
                selector_hints=future_selector_hints,
                coverage_required=False,
                operation_kind="style",
                applicable_rule_ids=active_future_element_rule_ids,
            )
        return ModifyExecutionPlan(
            scope="local",
            reason=(
                "LLM semantic compiler extracted future-only preference rule(s) for this turn; "
                "there are no existing slide targets."
            ),
            target_slide_paths=[],
            target_rule_ids=future_rule_ids,
            selector_hints=future_selector_hints,
            coverage_required=False,
            operation_kind="preference_update",
            applicable_rule_ids=active_future_element_rule_ids,
        )

    if structural_slide_operation and not current_turn_existing_specs:
        rule_ids, selector_hints = future_rule_ids, future_selector_hints

        if specific_slide_path is not None:
            reason = (
                "Structural slide operation detected; treat this as a deck-level page operation "
                "anchored to the resolved slide, without requiring batch edits on existing slides."
            )
            if current_turn_future_only_specs:
                reason += " Future-only global rules will apply to the inserted slide once it exists."
            return ModifyExecutionPlan(
                scope="global",
                reason=reason,
                target_slide_paths=[specific_slide_path],
                target_rule_ids=rule_ids,
                selector_hints=selector_hints,
                coverage_required=False,
                operation_kind="structural",
                applicable_rule_ids=active_future_element_rule_ids,
            )

        if current_turn_future_only_specs:
            return ModifyExecutionPlan(
                scope="global",
                reason=(
                    "Structural slide operation detected with future-only global rules; "
                    "do not require deck-wide edits on existing slides."
                ),
                target_slide_paths=[],
                target_rule_ids=rule_ids,
                selector_hints=selector_hints,
                coverage_required=False,
                operation_kind="structural",
                applicable_rule_ids=active_future_element_rule_ids,
            )

        return ModifyExecutionPlan(
            scope="global",
            reason=(
                "Structural slide operation detected; use deck-level insert/delete/new-slide tools "
                "instead of narrowing this turn to existing-slide patch tools."
            ),
            target_slide_paths=[],
            target_rule_ids=rule_ids,
            selector_hints=selector_hints,
            coverage_required=False,
            operation_kind="structural",
            applicable_rule_ids=active_future_element_rule_ids,
        )

    if not user_has_global_signal and not current_turn_existing_specs:
        if specific_slide_path is not None:
            selector_hints = []
            for hint in active_future_element_selector_hints:
                if hint not in selector_hints:
                    selector_hints.append(hint)
            return ModifyExecutionPlan(
                scope="local",
                reason=(
                    "Current request resolves to a specific slide without a new deck-level rule."
                    + (
                        " Active future-element WM rule(s) are available for LLM applicability checking."
                        if active_future_element_rule_ids
                        else ""
                    )
                ),
                target_slide_paths=[specific_slide_path],
                target_rule_ids=[],
                selector_hints=selector_hints,
                coverage_required=False,
                operation_kind="style",
                applicable_rule_ids=active_future_element_rule_ids,
            )
        return None

    scope = "global"
    target_slide_paths = list(all_slide_paths)

    selector_hints: list[str] = []
    rule_ids: list[str] = []
    active_specs = current_turn_existing_specs or all_rule_specs
    inferred_element_kind = infer_session_rule_element_kind(user_message)

    for spec in active_specs:
        rule_id = str(spec.get("rule_id", "") or "").strip()
        if rule_id and rule_id not in rule_ids:
            rule_ids.append(rule_id)
        target = spec.get("target")
        if isinstance(target, dict):
            for hint in selector_hints_for_element_kind(
                str(target.get("element_kind", "") or "").strip()
            ):
                if hint not in selector_hints:
                    selector_hints.append(hint)

    if not selector_hints:
        for hint in selector_hints_for_element_kind(inferred_element_kind):
            if hint not in selector_hints:
                selector_hints.append(hint)
    if text_color_boundary:
        for hint in ("h1", "h2", "p", "li", ".title", ".body", ".content"):
            if hint not in selector_hints:
                selector_hints.append(hint)

    reason_parts: list[str] = []
    if user_has_global_signal:
        reason_parts.append("The current user message contains deck-level/global language.")
    if current_turn_existing_specs:
        reason_parts.append(
            f"{len(current_turn_existing_specs)} structured existing-slide rule(s) were extracted for this turn."
        )
    elif current_turn_future_only_specs:
        reason_parts.append(
            f"{len(current_turn_future_only_specs)} future-only rule(s) were extracted for this turn."
        )
    if scope == "hybrid" and specific_slide_path is not None:
        reason_parts.append(
            f"A specific target slide was also inferred ({specific_slide_path.name})."
        )
    if text_color_boundary:
        reason_parts.append(
            "The current wording targets text/font color; preserve existing backgrounds and surfaces."
        )

    return ModifyExecutionPlan(
        scope=scope,
        reason=" ".join(reason_parts) or "Deck-level modification plan generated.",
        target_slide_paths=target_slide_paths,
        target_rule_ids=rule_ids,
        selector_hints=selector_hints,
        coverage_required=True,
        operation_kind="style",
        applicable_rule_ids=active_future_element_rule_ids,
    )


def render_modify_execution_plan(plan: ModifyExecutionPlan, *, workspace_label_fn=workspace_relative_label, runtime: Any | None = None) -> str:
    """Render a model-facing execution contract for modify turns."""
    if runtime is None:
        runtime = type("_RuntimeProxy", (), {"workspace": Path("."),})()
    operation_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=getattr(plan, "scope", None),
    )
    target_lines = "\n".join(
        f"- {workspace_label_fn(runtime, path)}" for path in plan.target_slide_paths
    )
    if not target_lines and operation_kind == "structural":
        target_lines = "- (deck structure operation; choose the insertion/deletion anchor with scan_slide_index or the current slide list)"
    elif not target_lines and operation_kind == "preference_update":
        target_lines = "- (no existing slide targets; this turn records a future/new-slide preference)"
    selector_hint_line = ", ".join(plan.selector_hints) if plan.selector_hints else "(inspect one representative slide first)"
    rule_line = ", ".join(plan.target_rule_ids) if plan.target_rule_ids else "(current-turn global request without explicit rule id)"
    if operation_kind == "preference_update":
        contract_lines = [
            "- This is a future-only preference update, not an existing-slide style change.",
            "- Do not call batch style tools and do not patch already-rendered slide files.",
            "- The preference should be applied later when a new slide is inserted/generated.",
            "- If useful, call `remember_lesson` once to leave a concise note, then call `finalize`.",
        ]
    elif operation_kind == "diagram_layout":
        diagram_contract = getattr(plan, "diagram_contract", None) or {}
        contract_lines = [
            "- This is an existing-slide diagram layout rewrite, not a deck page-count structural operation.",
            "- Start with `read_slide_snapshot` on the target slide to obtain `content_hash`.",
            "- Use `write_html_file(file_path=target, content=<complete replacement HTML>, force_regenerate=true, expected_hash=content_hash)` for one coherent page rewrite.",
            "- If the diagram is flowchart/pipeline-like, prefer `render_flowchart_asset` and place the generated SVG/PNG or HTML fragment as the main visual.",
            "- Remove old conflicting figures, captions, bullet containers, and empty legacy cards unless the user explicitly asks to keep them.",
            "- Preserve active visual preferences in the rewritten page, including markers such as TEMP-PREF tags.",
            "- After writing, call `inspect_slide`; if diagram diagnostics fail, use the deterministic flowchart fallback rather than appending a small fragment to the old layout.",
            "- Do not insert/delete/reorder slides for this request.",
            "Diagram contract: " + json.dumps(diagram_contract, ensure_ascii=False),
        ]
    elif operation_kind == "controlled_rewrite":
        rewrite_decision = getattr(plan, "rewrite_decision", None) or {}
        contract_lines = [
            "- This is a single existing-slide controlled rewrite for page-level information architecture restructuring.",
            "- Start with `read_slide_snapshot` on the target slide to obtain `content_hash`.",
            "- Use `write_html_file(file_path=target, content=<complete replacement HTML>, force_regenerate=true, expected_hash=content_hash)` and write back to the same target slide.",
            "- Do not insert/delete/reorder slides and do not batch-edit unrelated slides.",
            "- Build a scan-friendly page structure, such as headline plus three cards, three columns, or distinct section blocks.",
            "- For homepage/decision-brief/opening-slide requests, preserve three semantic blocks when requested and make them visually structured, not three plain paragraphs in the top-left.",
            "- Preserve active Temporary preferences, including `结论：` title prefixes and TEMP-PREF markers when present.",
            "- After writing, call `inspect_slide`; use local patch tools only for small inspection repairs.",
            "Rewrite decision: " + json.dumps(_json_safe(rewrite_decision), ensure_ascii=False),
        ]
    elif operation_kind == "structural":
        contract_lines = [
            "- This is a deck-level structural operation, and global modify turns expose the full core modify toolset.",
            "- Prefer `insert_slide`/`delete_slide` for the file-level page operation.",
            "- For an inserted slide, fill the brand-new file with `write_new_slide_file`; do not simulate a new page by rewriting an existing slide.",
            "- Use `scan_slide_index` first if you need to choose an insertion/deletion anchor from the current deck order.",
            "- If inspection finds a problem after the new slide is written, use `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch` to repair it.",
            "- `batch_update_css_rule` and `batch_update_semantic_style` are available, but use them only when the user also requested a deck-wide style/content change.",
            "- Do not call `finalize` until the requested page-count operation has actually changed the canonical slide files and the affected slide(s) have been checked.",
        ]
    else:
        contract_lines = [
            "- Interpret `font color`, `text color`, `字体颜色`, `文字颜色`, `标题颜色`, and similar wording as CSS text `color` on the requested text targets. Do not infer a background/surface/theme rewrite from that wording; preserve existing backgrounds, background images, panels, and layout unless the user explicitly asks to change them.",
            "- If the rule targets title/body/footer semantics across structurally different slides, prefer `batch_update_semantic_style`.",
            "- If these slides share the same explicit selector, prefer `batch_update_css_rule` over rewriting full HTML.",
            "- For existing-slide repairs, inspect `repair_candidates`, `risk_flags`, exposed `rules`, and layout targets from `read_slide_snapshot` before deciding the patch.",
            "- If you need to change existing slide content or a single slide's local structure, use `read_slide_snapshot` + `apply_slide_patch` instead of full-page rewrite.",
            "- For canvas/container/overflow issues, prefer `merge_style`, `merge_css_rule`, or `replace_css_rule` on `slide_canvas`, `layout_container`, or exposed rules before shrinking text.",
            "- Use `set_attr` / `remove_attr` only for safe attribute repairs (`class`, `style`, `src`, `alt`, `width`, `height`).",
            "- Never call `apply_slide_patch` without a non-empty `patch_ops` list built from `repair_candidates`, exposed `rules`, or `read_slide_snapshot.targets`.",
            "- If `apply_slide_patch` succeeds after auto-rebinding stale references, continue normally; if it returns `STALE_SNAPSHOT`, use the returned fresh snapshot and `rebind_hints` instead of falling back to full rewrite.",
            "- After a deck-level batch style pass, prefer `patch_semantic_inline_style` for one-off exception repairs on a single slide.",
            f"- Likely selector hints: {selector_hint_line}",
            "- Use `scan_slide_index` first when you need a deck-level map of titles/selectors.",
            "- Do not call `finalize` until every target slide above has been modified or explicitly confirmed compliant.",
            "- If there is also a local exception request, apply the deck-level rule first, then repair the exception slide with `patch_semantic_inline_style` when possible.",
        ]
    new_element_applications = getattr(plan, "new_element_rule_applications", None) or []
    if new_element_applications and operation_kind != "preference_update":
        rule_lines = []
        for app in new_element_applications[:8]:
            target_labels = ", ".join(
                str(item)
                for item in (app.get("target_slide_paths", []) if isinstance(app, dict) else [])[:4]
            )
            requirement = str(app.get("requirement", "") or app.get("rule_id", "") or "").strip()
            guidance = str(app.get("repair_guidance", "") or "").strip()
            matched_spans = ", ".join(
                str(item)
                for item in (app.get("matched_text_spans", []) if isinstance(app, dict) else [])[:6]
            )
            spec = app.get("rule_spec") if isinstance(app.get("rule_spec"), dict) else {}
            tokens = rule_design_tokens(spec)
            token_text = (
                f"; design tokens: {json.dumps(tokens, ensure_ascii=False)}"
                if tokens
                else ""
            )
            rule_lines.append(
                f"- {app.get('rule_id', 'wm_rule')}: apply to newly created {app.get('element_kind', 'element')} only"
                + (f" on {target_labels}" if target_labels else "")
                + (f"; matched spans: {matched_spans}" if matched_spans else "")
                + (f"; requirement: {requirement}" if requirement else "")
                + token_text
                + (f"; repair guidance: {guidance}" if guidance else "")
            )
        contract_lines.extend(
            [
                "- Active WM new-element applications for this turn:",
                *rule_lines,
        "- These rules constrain only elements newly created in this turn; do not backfill old matching elements.",
        "- If a text-color preference would be low contrast on the current tag/background, preserve the preferred text color and adjust an unlocked companion style such as background-color.",
        "- For caption rules, apply the remembered caption typography to the newly created caption element itself, such as color and font-style, without restyling old captions.",
        "- For fuzzy color intents in design_tokens, choose a visible local implementation that satisfies the semantic color intent; do not use unrelated variables like a heading color token unless that token actually matches the requested intent.",
        "- For semantic text-span rules, style only the listed matched_text_spans inside the newly created text unless the application explicitly says the whole element is the target.",
            ]
        )
    contract_text = "\n".join(contract_lines)
    return (
        f"<modify_execution_plan scope=\"{normalize_modify_scope(plan.scope)}\" operation_kind=\"{operation_kind}\" coverage_required=\"{str(plan.coverage_required).lower()}\">\n"
        f"Reason: {plan.reason}\n"
        f"Rule IDs: {rule_line}\n"
        "Target slides for this turn:\n"
        f"{target_lines}\n"
        "Execution contract:\n"
        f"{contract_text}\n"
        "</modify_execution_plan>"
    )


def render_modify_tool_policy_plan(
    policy: ModifyToolPolicyPlan,
    *,
    workspace_label_fn=workspace_relative_label,
    runtime: Any | None = None,
) -> str:
    if runtime is None:
        runtime = type("_RuntimeProxy", (), {"workspace": Path("."),})()
    operation_kind = normalize_modify_operation_kind(policy.operation_kind, scope=policy.scope)
    target_lines = "\n".join(
        f"- {workspace_label_fn(runtime, path)}" for path in policy.target_slide_paths
    ) or "- (no fixed target; inspect workspace/deck first)"
    if operation_kind == "preference_update" and not policy.target_slide_paths:
        target_lines = "- (no existing slide targets; memory-only/future preference turn)"
    first_steps = "\n".join(f"- {step}" for step in policy.first_steps or []) or "- choose the smallest effective modifying tool"
    risks = ", ".join(policy.risk_flags or []) or "(none declared)"
    new_element_apps = list(getattr(policy, "new_element_rule_applications", []) or [])
    safety_lines = [
        "- Existing healthy slides must be modified with semantic batch tools or `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch`.",
        "- If `insert_slide` creates a placeholder, the next expected modifying action is `write_new_slide_file` on that new file; do not use `apply_slide_patch` as the primary way to fill a new slide.",
        "- After a brand-new slide has complete HTML from `write_new_slide_file`, repair that slide with local patch tools only for small issues found by inspection.",
        "- Do not keep inserting slides once the requested slide-count change is complete.",
        "- If image/material tools are available and the task involves images, inspect local/document assets before external acquisition.",
    ]
    if new_element_apps or (getattr(policy, "reason", "") and "new-element" in str(policy.reason).lower()):
        safety_lines.append(
            "- Apply active WM new-element rules to elements created in this turn; preserve user-specified text color and adjust unlocked backgrounds if contrast requires it."
        )
    if new_element_apps:
        for app in new_element_apps[:6]:
            if not isinstance(app, dict):
                continue
            spec = app.get("rule_spec") if isinstance(app.get("rule_spec"), dict) else {}
            tokens = rule_design_tokens(spec)
            token_text = (
                f" design_tokens={json.dumps(tokens, ensure_ascii=False)}"
                if tokens
                else ""
            )
            safety_lines.append(
                "- Applicable WM rule "
                f"{str(app.get('rule_id', '') or 'wm_rule').strip()}: "
                f"{str(app.get('requirement', '') or rule_requirement_text(spec)).strip()}"
                f"{token_text}. Apply it to newly created {str(app.get('element_kind', '') or 'element')} only."
            )
    if new_element_apps or (getattr(policy, "reason", "") and "caption" in str(policy.reason).lower()):
        safety_lines.append(
            "- When creating a new caption, apply active WM caption rules such as color and font-style to the new caption only; fuzzy color intents must be visibly satisfied, not replaced by unrelated CSS variables."
        )
    if operation_kind == "preference_update":
        safety_lines = [
            "- This tool policy is read/memory-only; do not mutate existing slide files.",
            "- Use `remember_lesson` only if an extra concise memory note is needed.",
            "- Call `finalize` once the preference update is acknowledged.",
        ]
    elif operation_kind == "controlled_rewrite":
        safety_lines = [
            "- This tool policy allows a single-slide controlled rewrite because the planner identified page-level information architecture restructuring.",
            "- Start with `read_slide_snapshot`; use its `content_hash` as `expected_hash` for `write_html_file(force_regenerate=true)`.",
            "- Rewrite only the listed target slide(s); do not insert/delete/reorder slides and do not batch-edit unrelated slides.",
            "- Produce a clear structured layout such as headline plus three cards/columns/section blocks; do not leave sparse plain text in the upper-left.",
            "- Preserve the user-requested content/layout constraints and active deck preferences while making the page internally coherent.",
            "- `inspect_slide` must pass before finalize; use local patch tools afterward only for small inspection repairs.",
        ]
    elif operation_kind == "diagram_layout":
        safety_lines = [
            "- This tool policy allows a controlled full rewrite only for the listed diagram target slide(s).",
            "- Start with `read_slide_snapshot`; use its `content_hash` as `expected_hash` for `write_html_file(force_regenerate=true)`.",
            "- Use `render_flowchart_asset` when a deterministic flowchart/pipeline visual is safer than hand-written fragments.",
            "- Do not insert/delete slides, and do not batch-edit unrelated old slides.",
            "- `inspect_slide` must pass both export/layout checks and diagram diagnostics before finalize.",
        ]
    safety_contract = "\n".join(safety_lines)
    return (
        f"<modify_tool_policy source=\"{policy.source}\" scope=\"{normalize_modify_scope(policy.scope)}\" "
        f"operation_kind=\"{operation_kind}\" "
        f"expected_slide_delta=\"{policy.expected_slide_delta}\">\n"
        f"Reason: {policy.reason}\n"
        f"Tool groups: {', '.join(policy.tool_groups or []) or '(none)'}\n"
        f"Risk flags: {risks}\n"
        "Target slides:\n"
        f"{target_lines}\n"
        "First steps:\n"
        f"{first_steps}\n"
        "Runtime safety contract:\n"
        f"{safety_contract}\n"
        "</modify_tool_policy>"
    )


def resolve_modify_coverage_path(runtime: Any, raw_path: str) -> Path | None:
    """Resolve a tool argument path back to the current workspace slide file."""
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = runtime.workspace / path
    if path.exists():
        return path.resolve()
    if path.suffix.lower() == ".html":
        alias = resolve_slide_path(runtime, path.stem)
        if alias is not None:
            return alias.resolve()
    return None


def extract_modify_coverage_paths(
    runtime: Any,
    *,
    tool_name: str,
    arguments: str | dict[str, Any] | None,
    result_text: str,
) -> list[Path]:
    """Collect slide paths covered by a successful modify tool call."""
    parsed_args: dict[str, Any] = {}
    if isinstance(arguments, dict):
        parsed_args = arguments
    elif isinstance(arguments, str) and arguments.strip():
        try:
            parsed_args = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_args = {}

    paths: list[Path] = []
    if tool_name in {"write_html_file", "write_new_slide_file", "patch_semantic_inline_style", "apply_slide_patch"}:
        if tool_name in {"patch_semantic_inline_style", "apply_slide_patch"}:
            payload = extract_json_object(result_text or "")
            if isinstance(payload, dict) and payload.get("success") is False:
                return []
            if not payload and re.search(r'["\']?success["\']?\s*[:=]\s*false', str(result_text or ""), re.IGNORECASE):
                return []

        field_names = ("slide_path", "file_path") if tool_name == "apply_slide_patch" else ("file_path", "path")
        for raw_path in _extract_path_candidates_from_text(result_text, field_names):
            candidate = resolve_modify_coverage_path(runtime, raw_path)
            if candidate is not None and candidate not in paths:
                paths.append(candidate)
        if not paths and isinstance(parsed_args, dict):
            candidate = resolve_modify_coverage_path(runtime, parsed_args.get("file_path", ""))
            if candidate is not None:
                paths.append(candidate)
        return paths

    if tool_name == "insert_slide":
        anchor_path = resolve_modify_coverage_path(
            runtime,
            parsed_args.get("target_slide", "") if isinstance(parsed_args, dict) else "",
        )
        if anchor_path is not None:
            paths.append(anchor_path)

        match = re.search(
            r"New slide path:\s*(?P<path>[^\n\r]+)",
            str(result_text or ""),
            re.IGNORECASE,
        )
        if match:
            new_path = resolve_modify_coverage_path(runtime, match.group("path").strip())
            if new_path is not None and new_path not in paths:
                paths.append(new_path)
        return paths

    if tool_name in {"batch_update_css_rule", "batch_update_semantic_style"}:
        payload = extract_json_object(result_text or "")
        if isinstance(payload, dict) and payload.get("success") is False:
            return []
        if not payload and re.search(r'["\']?success["\']?\s*[:=]\s*false', str(result_text or ""), re.IGNORECASE):
            return []

        results = payload.get("results", []) if isinstance(payload, dict) else []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "") or "").strip().lower()
                if status == "error":
                    continue
                candidate = resolve_modify_coverage_path(runtime, item.get("file_path", ""))
                if candidate is not None and candidate not in paths:
                    paths.append(candidate)
        if not paths:
            for raw_path in _extract_path_candidates_from_text(result_text, ("file_path",)):
                candidate = resolve_modify_coverage_path(runtime, raw_path)
                if candidate is not None and candidate not in paths:
                    paths.append(candidate)
        if paths:
            return paths
        for raw_path in parsed_args.get("file_paths", []) if isinstance(parsed_args, dict) else []:
            candidate = resolve_modify_coverage_path(runtime, raw_path)
            if candidate is not None:
                paths.append(candidate)
    return paths


def build_modify_plan_followup(runtime: Any, uncovered_paths: list[Path]) -> str:
    labels = ", ".join(workspace_relative_label(runtime, path) for path in uncovered_paths[:12])
    return (
        "SYSTEM: The global modify plan is still incomplete. "
        f"Remaining target slides: {labels}. "
        "Use `scan_slide_index` if you still need a deck summary, then prefer "
        "`read_slide_snapshot` + `apply_slide_patch` for existing-slide content or local structure edits, "
        "prioritizing `repair_candidates`, `risk_flags`, exposed `rules`, `slide_canvas`, and `layout_container` for layout fixes, "
        "using `merge_css_rule` / `replace_css_rule` / `merge_style` before shrinking text, "
        "using `set_attr` / `remove_attr` only for safe attributes, "
        "and making sure `apply_slide_patch` includes a non-empty `patch_ops` list built from repair candidates, rules, or snapshot targets. "
        "If `apply_slide_patch` returns `STALE_SNAPSHOT`, use the fresh snapshot and `rebind_hints` to retry instead of switching straight to full rewrite. "
        "`batch_update_semantic_style` for semantic batch edits, "
        "`batch_update_css_rule` for explicit-selector edits, "
        "`patch_semantic_inline_style` for single-slide exception repair, "
        "and reserve `write_new_slide_file` for brand-new inserted slides while using `write_html_file` only for generation or corrupted-slide recovery. "
        "Do not call `finalize` until these slides are covered."
    )


def build_modify_no_mutation_followup(runtime: Any, plan: ModifyExecutionPlan | None) -> str:
    """Prompt the editor out of redundant read loops without forcing mutation."""
    scope = normalize_modify_scope(getattr(plan, "scope", None))
    operation_kind = normalize_modify_operation_kind(
        getattr(plan, "operation_kind", None),
        scope=getattr(plan, "scope", None),
    )
    if is_preference_update_plan(plan):
        return (
            "SYSTEM: This turn is a future-only memory preference update with no existing slide targets. "
            "Do not patch or batch-edit current slide files. If the preference has been recorded, call `finalize` now."
        )
    if scope == "global" and operation_kind == "structural":
        return (
            "SYSTEM: Recent read-only turns repeated information without changing the canonical slide files. "
            "For this page-count request, choose an anchor if needed with `scan_slide_index`, then call "
            "`insert_slide` or `delete_slide`. If a new slide is inserted as a placeholder, fill it with "
            "`write_new_slide_file` as the primary completion action; after inspection, repair with "
            "`plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch` only if needed. "
            "Do not wait for another turn or call `finalize` before the canonical slide files actually change."
        )
    if operation_kind == "image_asset":
        return (
            "SYSTEM: Recent read-only turns repeated the same resource/hash/scope. "
            "Do not reread the same slide snapshot or file window unless the file hash changed. "
            "If the image choice is still unresolved, use a genuinely new asset step such as "
            "`list_document_figures`, `explore_workspace_images`, `image_caption`, `search_images`, or "
            "`image_generation`; once the asset/path is known, use local patch tools to update the target slide, "
            "then inspect and finalize."
        )
    if operation_kind == "research_content":
        return (
            "SYSTEM: Recent read-only turns repeated the same information. "
            "Use a new research/read scope if more evidence is needed (`search_web`, `fetch_url`, or a different "
            "`read_file` offset/limit), or apply the requested local change if the needed content is already known. "
            "Do not keep rereading the same slide/hash/window."
        )
    if scope == "global":
        return (
            "SYSTEM: Recent read-only turns repeated the same information without progressing the global edit. "
            "Use a new information scope only if it is actually needed; otherwise use "
            "`batch_update_semantic_style`, `batch_update_css_rule`, or local patch tools "
            "to make the requested change, then inspect and finalize."
        )
    return (
        "SYSTEM: Recent read-only turns repeated the same slide/hash or file window. "
        "Do not reread that same scope. If more context is genuinely needed, read a different file range or use "
        "the appropriate asset/search tool for this request. If the target and desired edit are already clear, "
        "use `plan_slide_patch` / `read_slide_snapshot` only as needed, then call `apply_slide_patch` with concrete "
        "`patch_ops`; inspect after the patch and finalize if it is correct."
    )


async def build_user_state_snapshot(
    runtime: Any,
    request: InputRequest | None = None,
    *,
    task_intent: str = "",
    read_intent: str = "",
    write_intent: str = "",
    core_persona: str = "",
    include_cross_intent: bool = True,
) -> dict[str, Any]:
    if not runtime.memory_system:
        return {}

    resolver = getattr(runtime.memory_system, "user_state_resolver", None)
    if resolver is None:
        return {}

    candidate = request or getattr(runtime, "_last_request", None)
    resolved_task_intent = task_intent
    resolved_read_intent = read_intent
    resolved_write_intent = write_intent
    resolved_core_persona = core_persona

    if candidate is not None:
        resolved_task_intent = resolved_task_intent or get_request_task_intent(runtime, candidate)
        resolved_read_intent = resolved_read_intent or get_request_memory_read_intent(runtime, candidate)
        resolved_write_intent = resolved_write_intent or get_request_memory_write_intent(runtime, candidate)
        resolved_core_persona = resolved_core_persona or get_request_core_persona(candidate)

    job_mgr = getattr(runtime, "_job_mgr", None)
    working_memory = getattr(job_mgr, "working_memory", None) if job_mgr else None

    snapshot = await resolver.build_snapshot(
        user_id=runtime.user_id,
        task_intent=resolved_task_intent,
        read_intent=resolved_read_intent,
        write_intent=resolved_write_intent,
        core_persona=resolved_core_persona,
        working_memory=working_memory,
        include_cross_intent=include_cross_intent,
    )
    payload = snapshot.to_dict()
    payload["prompt_blocks"] = resolver.build_prompt_blocks(snapshot)
    return payload


async def call_runtime_intent_llm(llm_client: Any, prompt: str) -> str:
    from memslides.memory.core.models import call_intent_classifier_llm

    return await call_intent_classifier_llm(llm_client, prompt)


async def classify_request_intent_with_llm(
    runtime: Any,
    request: InputRequest,
):
    from memslides.memory.core.models import (
        IntentClassificationResult,
        build_intent_classification_prompt,
        classify_intent_details_with_llm,
    )

    if not runtime.memory_system:
        return IntentClassificationResult(intent="", scenario="", raw_response="")

    llm_objects = getattr(runtime.memory_system, "llm_objects_by_task", {}) or {}
    llm_client = (
        llm_objects.get("intent_classify")
        or getattr(runtime.memory_system, "llm", None)
    )
    if llm_client is None:
        return IntentClassificationResult(intent="", scenario="", raw_response="")

    attachment_names = [Path(path).name for path in request.attachments if path]
    template_name = Path(request.template).stem if request.template else ""
    prompt = build_intent_classification_prompt(
        request.instruction,
        attachment_names=attachment_names,
        template_name=template_name,
        num_pages=request.num_pages,
    )
    result = await classify_intent_details_with_llm(
        request.instruction,
        llm_client=llm_client,
        fallback_to_keywords=False,
        attachment_names=attachment_names,
        template_name=template_name,
        num_pages=request.num_pages,
    )

    artifact_writer = getattr(runtime.memory_system, "artifact_writer", None)
    if artifact_writer and hasattr(artifact_writer, "write_intent_classification"):
        try:
            artifact_writer.write_intent_classification(
                turn_id=0,
                user_message=request.instruction,
                prompt=prompt,
                response=result.raw_response,
                intent_result={
                    "intent": result.intent,
                    "scenario": result.scenario,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "source": "runtime_request",
                },
            )
        except Exception as e:
            logger.debug("Runtime request intent artifact write failed: %s", e)

    return result


async def resolve_request_intent_runtime(
    runtime: Any,
    request: InputRequest,
) -> IntentResolutionResult:
    from memslides.memory.core.models import (
        DEFAULT_INTENT,
        build_intent_signal_text,
        classify_intent_by_keywords,
        infer_intent_scenario,
    )

    if getattr(runtime, "_resolved_request_intent", ""):
        return IntentResolutionResult(
            intent=runtime._resolved_request_intent,
            scenario=runtime._resolved_request_intent_scenario,
            source=runtime._resolved_request_intent_source,
            confidence=runtime._resolved_request_intent_confidence,
            raw_response=runtime._resolved_request_intent_raw_response,
        )

    explicit_request_intent = _normalize_memory_intent(request.memory_intent)
    explicit_extra_info_intent = get_explicit_extra_info_intent(request)
    resolution_mode = get_intent_resolution_mode(request)
    intent_signal_text = build_intent_signal_text(
        request.instruction,
        attachment_names=[Path(path).name for path in request.attachments if path],
        template_name=Path(request.template).stem if request.template else "",
        num_pages=request.num_pages,
    )

    if resolution_mode == "explicit_first":
        if explicit_request_intent:
            return IntentResolutionResult(
                intent=explicit_request_intent,
                scenario=infer_intent_scenario(intent_signal_text, explicit_request_intent),
                source="explicit_request",
                confidence=1.0,
            )
        if explicit_extra_info_intent:
            return IntentResolutionResult(
                intent=explicit_extra_info_intent,
                scenario=infer_intent_scenario(intent_signal_text, explicit_extra_info_intent),
                source="explicit_extra_info",
                confidence=1.0,
            )

    llm_result = None
    if resolution_mode != "keyword_only":
        llm_result = await classify_request_intent_with_llm(runtime, request)
        if llm_result.intent and llm_result.intent != DEFAULT_INTENT:
            return IntentResolutionResult(
                intent=llm_result.intent,
                scenario=llm_result.scenario,
                source="llm",
                confidence=llm_result.confidence,
                raw_response=llm_result.raw_response,
            )

    keyword_intent = classify_intent_by_keywords(request.instruction)
    keyword_scenario = infer_intent_scenario(intent_signal_text, keyword_intent)
    if keyword_intent != DEFAULT_INTENT:
        return IntentResolutionResult(
            intent=keyword_intent,
            scenario=keyword_scenario,
            source="keyword",
            raw_response=llm_result.raw_response if llm_result else "",
        )

    if llm_result and llm_result.intent:
        return IntentResolutionResult(
            intent=llm_result.intent,
            scenario=llm_result.scenario,
            source="llm",
            confidence=llm_result.confidence,
            raw_response=llm_result.raw_response,
        )

    return IntentResolutionResult(
        intent=keyword_intent or DEFAULT_INTENT,
        scenario=keyword_scenario,
        source="keyword",
        raw_response=llm_result.raw_response if llm_result else "",
    )


def write_runtime_json(runtime: Any, filename: str, payload: dict[str, Any]) -> None:
    try:
        with open(runtime.workspace / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug("Failed to write %s: %s", filename, e)


def cache_resolved_request_intent(
    runtime: Any,
    request: InputRequest,
    result: IntentResolutionResult,
) -> None:
    extra_info = get_request_extra_info(request)
    runtime._resolved_request_intent = result.intent
    runtime._resolved_request_intent_scenario = result.scenario
    runtime._resolved_request_intent_source = result.source
    runtime._resolved_request_intent_confidence = result.confidence
    runtime._resolved_request_intent_raw_response = result.raw_response

    extra_info["resolved_memory_intent"] = result.intent
    extra_info["resolved_scenario_intent"] = result.scenario
    extra_info["resolved_memory_intent_source"] = result.source
    extra_info["resolved_task_intent"] = get_request_task_intent(runtime, request)
    extra_info["resolved_memory_read_intent"] = get_request_memory_read_intent(runtime, request)
    extra_info["resolved_memory_write_intent"] = get_request_memory_write_intent(runtime, request)
    core_persona = get_request_core_persona(request)
    if core_persona:
        extra_info["core_persona"] = core_persona
    if result.confidence is not None:
        extra_info["resolved_memory_intent_confidence"] = result.confidence
    elif "resolved_memory_intent_confidence" in extra_info:
        extra_info.pop("resolved_memory_intent_confidence", None)

    payload = {
        "resolved_memory_intent": result.intent,
        "resolved_scenario_intent": result.scenario,
        "resolved_task_intent": extra_info.get("resolved_task_intent", ""),
        "resolved_memory_read_intent": extra_info.get("resolved_memory_read_intent", ""),
        "resolved_memory_write_intent": extra_info.get("resolved_memory_write_intent", ""),
        "core_persona": extra_info.get("core_persona", ""),
        "source": result.source,
        "confidence": result.confidence,
        "raw_response": result.raw_response,
        "resolution_mode": get_intent_resolution_mode(request),
        "explicit_request_intent": _normalize_memory_intent(request.memory_intent),
        "explicit_extra_info_intent": get_explicit_extra_info_intent(request),
        "instruction_preview": request.instruction[:500],
    }
    runtime._resolved_request_intent_payload = payload
    write_runtime_json(runtime, "resolved_intent.json", payload)


def get_runtime_request_intent(runtime: Any, request: InputRequest) -> str:
    return runtime._resolved_request_intent or _resolve_request_memory_intent(request)


def get_resolved_intent_artifact(runtime: Any) -> dict[str, Any]:
    return dict(getattr(runtime, "_resolved_request_intent_payload", {}) or {})


def get_template_match_artifact(runtime: Any) -> dict[str, Any]:
    return dict(getattr(runtime, "_template_match_state", {}) or {})


def initialize_template_match_state(runtime: Any, request: InputRequest) -> None:
    runtime._template_match_state = {
        "resolved_memory_intent": get_runtime_request_intent(runtime, request),
        "resolved_scenario_intent": runtime._resolved_request_intent_scenario,
        "resolved_task_intent": get_request_task_intent(runtime, request),
        "resolved_memory_read_intent": get_request_memory_read_intent(runtime, request),
        "resolved_memory_write_intent": get_request_memory_write_intent(runtime, request),
        "core_persona": get_request_core_persona(request),
        "template_intent": "",
        "template_intent_confidence": None,
        "template_use_decision": "",
        "template_use_basis": "",
        "template_use_intent": "",
        "template_activation_allowed": False,
        "template_activation_reason": "",
        "template_runtime_mode": "disabled",
        "template_structure_score": None,
        "template_quality_issues": [],
        "selected_template_id": "",
        "selected_template_name": "",
        "selection_source": "",
        "selection_confidence": None,
        "template_reasoning": "",
        "selection_reasoning": "",
        "style_memory_hint": "",
        "template_query": "",
        "matched_by_history": False,
        "matched_history_intent": "",
        "request_used_explicit_template": bool(
            request.template or request.template_id or request.template_as_reference
        ),
        "instruction_preview": request.instruction[:500],
    }
    write_runtime_json(runtime, "template_match.json", runtime._template_match_state)


def update_template_match_state(
    runtime: Any,
    request: InputRequest,
    **updates: Any,
) -> None:
    if not getattr(runtime, "_template_match_state", None):
        initialize_template_match_state(runtime, request)
    runtime._template_match_state.update(updates)
    runtime._template_match_state["resolved_memory_intent"] = get_runtime_request_intent(runtime, request)
    runtime._template_match_state["resolved_task_intent"] = get_request_task_intent(runtime, request)
    runtime._template_match_state["resolved_memory_read_intent"] = get_request_memory_read_intent(runtime, request)
    runtime._template_match_state["resolved_memory_write_intent"] = get_request_memory_write_intent(runtime, request)
    runtime._template_match_state["core_persona"] = get_request_core_persona(request)
    runtime._template_match_state["instruction_preview"] = request.instruction[:500]
    write_runtime_json(runtime, "template_match.json", runtime._template_match_state)


def template_user_candidates(runtime: Any) -> list[str]:
    user_id = (getattr(runtime, "user_id", "") or "").strip()
    return [user_id] if user_id else []


def normalize_template_lookup_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("模板", "")
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)
    return normalized


def resolve_template_from_selection(
    templates: list[Any],
    *,
    template_id: str = "",
    template_name: str = "",
) -> Any | None:
    template_id = (template_id or "").strip()
    template_name = (template_name or "").strip()

    if template_id:
        for template in templates:
            if getattr(template, "id", "") == template_id:
                return template

    if template_name:
        return pick_template_by_name_hint(templates, template_name)

    return None


async def list_accessible_templates(runtime: Any, store: Any, limit: int = 20) -> list[Any]:
    templates: list[Any] = []
    seen_names: set[str] = set()
    per_user_limit = max(limit, 1)
    for user_id in template_user_candidates(runtime):
        try:
            for template in await store.list_by_user(user_id, limit=per_user_limit):
                key = normalize_template_lookup_text(getattr(template, "name", ""))
                if key and key in seen_names:
                    continue
                if key:
                    seen_names.add(key)
                templates.append(template)
                if len(templates) >= limit:
                    return templates
        except Exception as e:
            logger.debug("Failed to list templates for %s: %s", user_id, e)
    return templates


def pick_template_by_name_hint(
    templates: list[Any],
    template_query: str,
) -> Any | None:
    normalized_query = normalize_template_lookup_text(template_query)
    if not normalized_query:
        return None

    for template in templates:
        normalized_name = normalize_template_lookup_text(getattr(template, "name", ""))
        if normalized_name and normalized_name == normalized_query:
            return template

    for template in templates:
        normalized_name = normalize_template_lookup_text(getattr(template, "name", ""))
        if not normalized_name:
            continue
        if normalized_query in normalized_name or normalized_name in normalized_query:
            return template
    return None


def filter_template_preferences_for_matching(
    preferences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from memslides.memory.core.models import is_template_related_preference

    filtered: list[dict[str, Any]] = []
    for pref in preferences:
        if is_template_related_preference(
            dimension=str(pref.get("conflict_group", "") or ""),
            content=pref.get("preference", ""),
            trigger=pref.get("trigger", ""),
            rationale=pref.get("rationale", ""),
        ):
            continue
        filtered.append(pref)
    return filtered


def template_profile_context_from_profile(profile: Any | None) -> dict[str, Any]:
    if profile is None:
        return {}
    context: dict[str, Any] = {}

    template_pref = getattr(profile, "template", None)
    if template_pref is not None:
        if hasattr(template_pref, "to_usage_selection_context"):
            context.update(template_pref.to_usage_selection_context())
        elif hasattr(template_pref, "to_selection_context"):
            context.update(template_pref.to_selection_context())

    def _collect_texts(values: list[Any]) -> list[str]:
        texts: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text:
                texts.append(text)
        return texts

    def _contains_any(source: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in source for keyword in keywords)

    theme = getattr(profile, "theme", None)
    visual = getattr(profile, "visual", None)
    layout = getattr(profile, "layout", None)
    content = getattr(profile, "content", None)
    general = getattr(profile, "general", None)

    raw_texts: list[str] = []
    if theme is not None:
        raw_texts.extend(_collect_texts(getattr(theme, "primary_colors", [])))
        raw_texts.extend(_collect_texts(getattr(theme, "accent_colors", [])))
        raw_texts.extend(_collect_texts([
            getattr(theme, "font_family", ""),
            getattr(theme, "background_style", ""),
        ]))
        raw_texts.extend(_collect_texts(getattr(theme, "notes", [])))
    if visual is not None:
        raw_texts.extend(_collect_texts([
            getattr(visual, "image_style", ""),
            getattr(visual, "icon_usage", ""),
            getattr(visual, "animation_preference", ""),
        ]))
        raw_texts.extend(_collect_texts(getattr(visual, "chart_type_priority", [])))
        raw_texts.extend(_collect_texts(getattr(visual, "notes", [])))
    if layout is not None:
        raw_texts.extend(_collect_texts([
            getattr(layout, "content_density", ""),
            getattr(layout, "alignment_style", ""),
            getattr(layout, "spacing_preference", ""),
            getattr(layout, "slide_structure", ""),
        ]))
        raw_texts.extend(_collect_texts(getattr(layout, "notes", [])))
    if content is not None:
        raw_texts.extend(_collect_texts([
            getattr(content, "text_density", ""),
            getattr(content, "language_style", ""),
            getattr(content, "bullet_point_style", ""),
            getattr(content, "title_length", ""),
        ]))
        raw_texts.extend(_collect_texts(getattr(content, "notes", [])))
    if general is not None:
        raw_texts.extend(_collect_texts(getattr(general, "preferences", [])))

    combined = " ".join(raw_texts).lower()
    style_defaults: dict[str, Any] = {}

    narrative_style_map = {
        "academic": ("academic", "学术", "论文", "research", "研究", "progress review"),
        "business": ("business", "商务", "商业", "管理", "executive", "汇报"),
        "technical": ("technical", "技术", "method", "workflow", "方法", "流程"),
        "educational": ("educational", "教育", "教学", "课堂", "course unit", "课程"),
        "creative": ("creative", "创意", "storytelling", "叙事", "品牌化"),
    }
    for label, keywords in narrative_style_map.items():
        if _contains_any(combined, keywords):
            style_defaults["narrative_style"] = label
            break

    density_source = " ".join(_collect_texts([
        getattr(layout, "content_density", "") if layout is not None else "",
        getattr(content, "text_density", "") if content is not None else "",
        getattr(content, "title_length", "") if content is not None else "",
    ])).lower()
    if _contains_any(density_source, ("low", "sparse", "minimal", "简洁", "简约", "留白", "短标题")):
        style_defaults["info_density"] = "low"
    elif _contains_any(density_source, ("high", "dense", "detailed", "紧凑", "详细", "丰富", "长标题")):
        style_defaults["info_density"] = "high"
    elif density_source:
        style_defaults["info_density"] = "medium"

    layout_notes = getattr(layout, "notes", []) if layout is not None else []
    layout_source = " ".join(_collect_texts([
        getattr(layout, "slide_structure", "") if layout is not None else "",
        getattr(layout, "alignment_style", "") if layout is not None else "",
        getattr(layout, "spacing_preference", "") if layout is not None else "",
        *layout_notes,
    ])).lower()
    if _contains_any(layout_source, ("minimal", "简约", "极简", "留白")):
        style_defaults["layout_preference"] = "minimal"
    elif _contains_any(layout_source, ("dense", "紧凑", "多栏", "grid", "网格")):
        style_defaults["layout_preference"] = "dense"
    elif layout_source:
        style_defaults["layout_preference"] = "balanced"

    theme_primary_colors = getattr(theme, "primary_colors", []) if theme is not None else []
    theme_accent_colors = getattr(theme, "accent_colors", []) if theme is not None else []
    theme_notes = getattr(theme, "notes", []) if theme is not None else []
    theme_source = " ".join(_collect_texts([
        getattr(theme, "background_style", "") if theme is not None else "",
        *theme_primary_colors,
        *theme_accent_colors,
        *theme_notes,
    ])).lower()
    if _contains_any(theme_source, ("dark", "深色", "黑", "navy", "墨")):
        style_defaults["color_tone"] = "dark"
    elif _contains_any(theme_source, ("warm", "暖", "橙", "红", "金")):
        style_defaults["color_tone"] = "warm"
    elif _contains_any(theme_source, ("cool", "冷", "蓝", "青", "绿")):
        style_defaults["color_tone"] = "cool"
    elif _contains_any(theme_source, ("vibrant", "鲜艳", "高饱和", "亮色")):
        style_defaults["color_tone"] = "vibrant"
    elif theme_source:
        style_defaults["color_tone"] = "neutral"

    reference_style = ""
    for text in raw_texts:
        lowered = text.lower()
        if "像" in text or "类似" in text or "like " in lowered or "similar to" in lowered:
            reference_style = text
            break
    if reference_style:
        style_defaults["reference_style"] = reference_style

    query_terms: list[str] = []
    for value in [
        style_defaults.get("narrative_style", ""),
        style_defaults.get("layout_preference", ""),
        style_defaults.get("color_tone", ""),
        getattr(layout, "slide_structure", "") if layout is not None else "",
        getattr(theme, "background_style", "") if theme is not None else "",
    ]:
        text = str(value or "").strip()
        if text and text not in query_terms:
            query_terms.append(text)

    if style_defaults:
        context["profile_style_defaults"] = style_defaults
        context["profile_style_signals"] = raw_texts[:8]
    if query_terms:
        context["profile_template_query_terms"] = query_terms[:5]

    return context


def build_runtime_template_summary_from_usage_history(
    usage_history: list[Any],
) -> Any | None:
    if not usage_history:
        return None
    from memslides.memory.core.models import TemplatePreference

    runtime_template_pref = TemplatePreference()
    runtime_template_pref.refresh_from_usage_records(
        usage_history,
        reset_semantic_fields=True,
    )
    return runtime_template_pref


async def load_usage_history_for_template_memory(
    runtime: Any,
    store: Any,
    *,
    request_intent: str,
    aspect_ratio: str,
    limit: int = 20,
    success_only: bool = False,
    allow_unscoped_fallback: bool = True,
) -> list[Any]:
    attempts: list[tuple[str, str]] = [
        (request_intent, aspect_ratio),
        (request_intent, ""),
    ]
    if allow_unscoped_fallback:
        attempts.extend([
            ("", aspect_ratio),
            ("", ""),
        ])

    deduped_attempts: list[tuple[str, str]] = []
    for candidate in attempts:
        if candidate not in deduped_attempts:
            deduped_attempts.append(candidate)

    for memory_intent, ratio in deduped_attempts:
        if (
            allow_unscoped_fallback
            and not memory_intent
            and request_intent == "default"
            and ratio == aspect_ratio
        ):
            continue
        try:
            history = await store.get_usage_history(
                user_id=runtime.user_id,
                limit=limit,
                success_only=success_only,
                memory_intent=memory_intent,
                aspect_ratio=ratio,
            )
        except Exception as e:
            logger.debug(
                "Failed to load template usage history (intent=%s, ratio=%s): %s",
                memory_intent,
                ratio,
                e,
            )
            continue
        if history:
            return history
    return []


async def sync_template_profile_from_usage_history(
    runtime: Any,
    request: InputRequest,
) -> None:
    if not runtime.memory_system:
        return
    if getattr(runtime, "_freeze_preference_writeback_current_job", False):
        return

    store = getattr(runtime.memory_system, "template_store", None)
    profile_store = getattr(runtime.memory_system, "profile_store", None)
    if not store or not profile_store:
        return

    write_intent = get_request_memory_write_intent(runtime, request)
    aspect_ratio = getattr(request.powerpoint_type, "value", "") or ""
    usage_history = await load_usage_history_for_template_memory(
        runtime,
        store,
        request_intent=write_intent,
        aspect_ratio=aspect_ratio,
        limit=50,
        success_only=True,
        allow_unscoped_fallback=False,
    )
    if not usage_history:
        return

    try:
        profile = await profile_store.get(runtime.user_id, intent=write_intent)
        template_pref = getattr(profile, "template", None)
        if (
            template_pref is not None
            and hasattr(template_pref, "refresh_from_usage_records")
            and template_pref.refresh_from_usage_records(
                usage_history,
                reset_semantic_fields=True,
            )
        ):
            await profile_store.save(profile, intent=write_intent)
            logger.info(
                "[Stage 14] Refreshed template profile summary from usage history "
                "(write_intent=%s, records=%s)",
                write_intent,
                len(usage_history),
            )
    except Exception as e:
        logger.debug("template profile sync failed: %s", e)


async def search_templates_by_style(
    runtime: Any,
    store: Any,
    *,
    narrative_style: str = "",
    info_density: str = "",
    color_tone: str = "",
    aspect_ratio: str | None = None,
    limit: int = 5,
) -> list[Any]:
    for user_id in template_user_candidates(runtime):
        try:
            results = await store.search_by_style(
                narrative_style=narrative_style,
                info_density=info_density,
                color_tone=color_tone,
                user_id=user_id,
                aspect_ratio=aspect_ratio,
                limit=limit,
            )
            if results:
                return results
        except Exception as e:
            logger.debug("search_by_style failed for %s: %s", user_id, e)
    return []


async def search_templates_by_query(
    runtime: Any,
    store: Any,
    query: str,
    *,
    aspect_ratio: str | None = None,
    limit: int = 5,
) -> list[Any]:
    for user_id in template_user_candidates(runtime):
        try:
            results = await store.search(
                query=query,
                user_id=user_id,
                aspect_ratio=aspect_ratio,
                limit=limit,
            )
            if results:
                return results
        except Exception as e:
            logger.debug("template search failed for %s: %s", user_id, e)
    return []


async def get_recent_template(runtime: Any, store: Any, limit: int = 1) -> list[Any]:
    for user_id in template_user_candidates(runtime):
        try:
            results = await store.get_recent(user_id=user_id, limit=limit)
            if results:
                return results
        except Exception as e:
            logger.debug("get_recent failed for %s: %s", user_id, e)
    return []


async def auto_match_template_profile(
    runtime: Any,
    request: InputRequest,
) -> tuple[Any | None, Any | None]:
    if not runtime.memory_system:
        return None, None

    store = getattr(runtime.memory_system, "template_store", None)
    if not store:
        return None, None

    try:
        from memslides.memory.style_intent_classifier import (
            StyleIntentClassifier,
            TemplateIntent,
        )
    except ImportError as e:
        logger.warning("StyleIntentClassifier not available: %s", e)
        return None, None

    accessible_templates = await list_accessible_templates(runtime, store, limit=50)
    available_names = [
        t.name for t in accessible_templates if getattr(t, "name", "")
    ]
    read_intent = get_request_memory_read_intent(runtime, request)
    aspect_ratio = getattr(request.powerpoint_type, "value", "") or ""

    user_preferences: list[dict] = []
    pref_store = getattr(runtime.memory_system, "preference_store", None)
    if pref_store:
        try:
            prefs = await pref_store.get_active(runtime.user_id, limit=10)
            user_preferences = filter_template_preferences_for_matching(
                [p.to_dict() for p in prefs]
            )
        except Exception as e:
            logger.debug("Failed to load preferences for style classify: %s", e)

    profile = None
    template_profile_context: dict[str, Any] = {}
    profile_store = getattr(runtime.memory_system, "profile_store", None)
    if profile_store:
        try:
            profile = await profile_store.get(runtime.user_id, intent=read_intent)
        except Exception as e:
            logger.debug("Failed to load user profile for template matching: %s", e)

    usage_history = await load_usage_history_for_template_memory(
        runtime,
        store,
        request_intent=read_intent,
        aspect_ratio=aspect_ratio,
        limit=12,
    )
    runtime_template_summary = build_runtime_template_summary_from_usage_history(
        usage_history
    )
    base_profile_context = template_profile_context_from_profile(profile)
    if runtime_template_summary is not None and hasattr(
        runtime_template_summary, "to_usage_selection_context"
    ):
        template_profile_context = {
            **base_profile_context,
            **runtime_template_summary.to_usage_selection_context(),
        }
    else:
        template_profile_context = base_profile_context

    llm_objects = getattr(runtime.memory_system, "llm_objects_by_task", {}) or {}
    style_llm = (
        llm_objects.get("style_classify")
        or llm_objects.get("intent_classify")
        or getattr(runtime.memory_system, "llm", None)
    )
    classifier = StyleIntentClassifier(
        llm=style_llm,
        artifact_writer=getattr(runtime.memory_system, "artifact_writer", None),
        use_llm=style_llm is not None,
    )
    style_intent_result = await classifier.classify(
        user_message=request.instruction,
        user_preferences=user_preferences,
        available_templates=available_names,
        template_store=store,
        user_id=runtime.user_id,
        embedding_func=getattr(runtime.memory_system, "embedding_func", None),
        memory_intent=read_intent,
        template_profile_context=template_profile_context,
    )
    from memslides.templates.activation import decide_template_activation

    activation_decision = decide_template_activation(request, style_intent_result)
    logger.info(
        "[Stage 14] Style intent: %s, gate=%s/%s (confidence=%.2f)",
        style_intent_result.template_intent.value,
        getattr(style_intent_result.template_use_decision, "value", ""),
        getattr(style_intent_result.template_use_basis, "value", ""),
        style_intent_result.confidence,
    )
    update_template_match_state(
        runtime,
        request,
        template_intent=style_intent_result.template_intent.value,
        template_intent_confidence=style_intent_result.confidence,
        template_use_decision=getattr(style_intent_result.template_use_decision, "value", ""),
        template_use_basis=getattr(style_intent_result.template_use_basis, "value", ""),
        template_reasoning=style_intent_result.reasoning,
        style_memory_hint=style_intent_result.memory_hint,
        template_query=style_intent_result.template_query,
        template_use_intent=activation_decision.use_intent.value,
        template_activation_allowed=activation_decision.allowed,
        template_activation_reason=activation_decision.reason,
        template_activation_evidence=activation_decision.evidence,
    )

    if not activation_decision.allowed:
        update_template_match_state(
            runtime,
            request,
            selection_source=f"skip_{activation_decision.use_intent.value}",
            selection_confidence=style_intent_result.confidence,
            selection_reasoning=activation_decision.reason,
            matched_by_history=False,
            matched_history_intent="",
        )
        return None, style_intent_result

    selector_llm = (
        llm_objects.get("template_analyze")
        or llm_objects.get("style_classify")
        or llm_objects.get("intent_classify")
        or getattr(runtime.memory_system, "llm", None)
    )
    if accessible_templates and selector_llm is not None:
        try:
            from memslides.memory.template_selector import TemplateSelector

            selector = TemplateSelector(
                llm=selector_llm,
                artifact_writer=getattr(runtime.memory_system, "artifact_writer", None),
            )
            selection = await selector.select(
                user_message=request.instruction,
                style_intent=style_intent_result.to_dict(),
                templates=accessible_templates,
                usage_history=[record.to_dict() for record in usage_history],
                user_preferences=user_preferences,
                template_profile_context=template_profile_context,
                user_id=runtime.user_id,
                aspect_ratio=aspect_ratio,
            )
            matched = resolve_template_from_selection(
                accessible_templates,
                template_id=selection.selected_template_id,
                template_name=selection.selected_template_name,
            )
            if matched is not None and selection.should_use_template:
                matched_by_history = style_intent_result.memory_hint in {
                    "similar_scenario",
                    "last_template",
                    "previous_template",
                    "that_template",
                }
                update_template_match_state(
                    runtime,
                    request,
                    selected_template_id=getattr(matched, "id", "") or getattr(matched, "template_id", ""),
                    selected_template_name=getattr(matched, "name", ""),
                    selection_source="llm_selector",
                    selection_confidence=selection.confidence,
                    selection_reasoning=selection.reasoning,
                    matched_by_history=matched_by_history,
                    matched_history_intent=read_intent if matched_by_history else "",
                )
                logger.info(
                    "[Stage 14] LLM-selected template: %s (confidence=%.2f)",
                    matched.name,
                    selection.confidence,
                )
                return matched, style_intent_result
            if selection.should_use_template:
                logger.warning(
                    "LLM template selector returned an unknown template (id=%s, name=%s); falling back to heuristic matching",
                    selection.selected_template_id,
                    selection.selected_template_name,
                )
            else:
                update_template_match_state(
                    runtime,
                    request,
                    selection_source="llm_selector_skip",
                    selection_confidence=selection.confidence,
                    selection_reasoning=selection.reasoning,
                    selected_template_id="",
                    selected_template_name="",
                    matched_by_history=False,
                    matched_history_intent="",
                )
                logger.info(
                    "[Stage 14] LLM template selector chose to skip template (reason=%s)",
                    selection.reasoning or "no suitable template",
                )
                return None, style_intent_result
        except Exception as e:
            logger.warning(
                "LLM template selection failed; falling back only to explicit name query: %s",
                e,
            )

    matched_templates: list[Any] = []
    selection_source = ""

    if style_intent_result.template_query:
        exact_match = pick_template_by_name_hint(
            accessible_templates, style_intent_result.template_query
        )
        if exact_match is not None:
            matched_templates = [exact_match]
            selection_source = "name_hint"

    if not matched_templates and style_intent_result.template_query:
        matched_templates = await search_templates_by_query(
            runtime,
            store,
            style_intent_result.template_query,
            aspect_ratio=request.powerpoint_type.value,
            limit=3,
        )
        if matched_templates:
            selection_source = "query_search"

    if matched_templates:
        matched = matched_templates[0]
        update_template_match_state(
            runtime,
            request,
            selected_template_id=getattr(matched, "id", "") or getattr(matched, "template_id", ""),
            selected_template_name=getattr(matched, "name", ""),
            selection_source=selection_source or "heuristic",
            selection_confidence=style_intent_result.confidence,
            selection_reasoning=style_intent_result.reasoning,
            matched_by_history=False,
            matched_history_intent="",
        )
        logger.info(
            "[Stage 14] Name-matched template fallback: %s (user=%s)",
            matched.name,
            getattr(matched, "user_id", "") or "default",
        )
        return matched, style_intent_result

    update_template_match_state(
        runtime,
        request,
        selection_source="freeform_fallback",
        selection_confidence=style_intent_result.confidence,
        selection_reasoning=style_intent_result.reasoning,
        selected_template_id="",
        selected_template_name="",
        matched_by_history=False,
        matched_history_intent="",
    )
    logger.info("[Stage 14] No template matched, falling back to freeform generation")
    return None, style_intent_result


def activate_template_profile(
    runtime: Any,
    request: InputRequest,
    template_profile: Any,
) -> tuple[Any, list[dict]]:
    from memslides.memory.inject.template_guide_builder import TemplateGuideBuilder

    # Reference-only template generation deliberately does not materialize or
    # expose template shell assets.  Shell extraction remains available for
    # offline audits, but active generation consumes only canonical layouts,
    # density hints, style tokens, and real attachment assets.
    guide_builder = TemplateGuideBuilder(template_profile)
    runtime._template_profile = template_profile
    runtime._guide_builder = guide_builder
    runtime._current_template_id = (
        getattr(template_profile, "id", "")
        or getattr(template_profile, "template_id", "")
        or ""
    )

    if runtime._current_template_id and not request.template_id:
        request.template_id = runtime._current_template_id

    template_source = getattr(template_profile, "template_source", "") or ""
    if template_source and Path(template_source).exists():
        request.template = template_source
        request.template_as_reference = True

    template_conflicts = guide_builder.detect_conflicts(request.instruction)
    template_match_state = getattr(runtime, "_template_match_state", {}) or {}
    quality_report = assess_template_quality(template_profile)
    quality_path = ""
    try:
        quality_path = str(write_template_quality_report(runtime.workspace, quality_report).resolve())
    except Exception as e:
        logger.warning("Failed to persist template quality report: %s", e)

    update_template_match_state(
        runtime,
        request,
        template_runtime_mode=quality_report.mode,
        template_structure_score=quality_report.structure_score,
        template_visual_safety_score=quality_report.visual_safety_score,
        template_quality_issues=quality_report.issues,
        template_quality_reason=quality_report.reason,
    )
    template_state = TemplateRuntimeState(
        active=True,
        mode=quality_report.mode,
        selected_template_id=runtime._current_template_id,
        selected_template_name=getattr(template_profile, "name", "") or "",
        profile_path=str((runtime.workspace / ".template_profile.json").resolve()),
        quality_path=quality_path,
        structure_score=quality_report.structure_score,
        visual_safety_score=quality_report.visual_safety_score,
        canonical_layout_names=canonical_layout_names_from_profile(template_profile),
        selection_source=str(
            template_match_state.get("selection_source", "")
            or template_match_state.get("template_intent", "")
        ),
    )
    try:
        save_template_runtime_state(runtime.workspace, template_state)
        runtime._template_runtime_state = template_state
    except Exception as e:
        logger.warning("Failed to persist template runtime state: %s", e)
    return guide_builder, template_conflicts


async def record_template_usage(
    runtime: Any,
    request: InputRequest,
    template_profile: Any,
    style_intent_result: Any = None,
    success: bool = True,
) -> None:
    if not runtime.memory_system or not template_profile:
        return

    store = getattr(runtime.memory_system, "template_store", None)
    if not store:
        return

    template_id = getattr(template_profile, "id", "") or getattr(
        template_profile, "template_id", ""
    )
    if not template_id:
        return

    if success:
        try:
            await store.mark_used(template_id)
        except Exception as e:
            logger.debug("mark_used failed for %s: %s", template_id, e)

    try:
        from memslides.memory.core.models import TemplateUsageRecord
    except ImportError:
        return

    message_embedding = None
    embedding_func = getattr(runtime.memory_system, "embedding_func", None)
    if embedding_func:
        try:
            embeddings = await embedding_func([request.instruction[:500]])
            if len(embeddings) > 0:
                first = embeddings[0]
                message_embedding = first.tolist() if hasattr(first, "tolist") else list(first)
        except Exception as e:
            logger.debug("template usage embedding failed: %s", e)

    attachment_type = ""
    if request.attachments:
        suffix = Path(request.attachments[0]).suffix.lower().lstrip(".")
        if suffix in {"ppt", "pptx"}:
            attachment_type = "pptx"
        elif suffix in {"pdf"}:
            attachment_type = "pdf"
        elif suffix in {"doc", "docx", "md", "txt"}:
            attachment_type = "doc"
        elif suffix in {"png", "jpg", "jpeg", "webp"}:
            attachment_type = "image"

    intent = "explicit"
    if style_intent_result is not None and hasattr(style_intent_result, "template_intent"):
        intent = style_intent_result.template_intent.value

    record = TemplateUsageRecord(
        user_id=runtime.user_id,
        template_id=template_id,
        template_name=getattr(template_profile, "name", ""),
        user_message=request.instruction[:500],
        user_message_embedding=message_embedding,
        intent=intent,
        memory_intent=get_request_memory_write_intent(runtime, request),
        aspect_ratio=getattr(request.powerpoint_type, "value", "") or "",
        has_attachment=bool(request.attachments),
        attachment_type=attachment_type,
        success=success,
    )
    try:
        await store.add_usage_record(record)
    except Exception as e:
        logger.debug("add_usage_record failed: %s", e)
        return

    await sync_template_profile_from_usage_history(runtime, request)


async def record_explicit_template_seed_usage(
    runtime: Any,
    request: InputRequest,
    template_profile: Any,
) -> bool:
    if runtime._template_usage_seeded or not request.template_as_reference or not template_profile:
        return False

    await record_template_usage(
        runtime,
        request=request,
        template_profile=template_profile,
        style_intent_result=None,
        success=True,
    )
    runtime._template_usage_seeded = True
    logger.info(
        "[Stage 14] Seeded explicit template usage history (template=%s, write_intent=%s)",
        getattr(template_profile, "name", "") or getattr(template_profile, "id", ""),
        get_request_memory_write_intent(runtime, request),
    )
    return True


def session_preference_has_general_signal(user_message: str) -> bool:
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False
    patterns = (
        r"(所有|全部|整套|全局|统一).{0,12}(页|页面|标题|配色|颜色|字体|版式|风格|正文)",
        r"(以后|未来|后续|后面|之后|新增页|新加页|新增幻灯片|新生成|新建|新插入|future).{0,20}(继续|沿用|保持|都用|都要|都保持|都为|均为|设为|改成|使用|应用|继承|默认|遵守|生效)",
        r"(每次|总是|一直|默认).{0,16}(都|用|保持|遵守)",
        r"(来源|页数|格式|约束).{0,16}(必须|只能|不要|禁止|严格)",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def session_preference_targets_future_slides(user_message: str) -> bool:
    normalized = " ".join(str(user_message or "").split()).lower()
    if not normalized:
        return False
    return bool(
        re.search(
            r"(新增页|新加页|新增加的?页|新增的?幻灯片|新增加的?幻灯片|新生成的?幻灯片|新建的?幻灯片|新插入的?幻灯片|后续新增|以后新增|future slides?|new slides?)",
            normalized,
        )
    )


def session_preference_is_future_slide_only(user_message: str) -> bool:
    normalized = " ".join(str(user_message or "").split()).lower()
    if not session_preference_targets_future_slides(normalized):
        return False
    return not bool(
        re.search(
            r"(现有|已有|当前|已经生成|现在的|整套).{0,12}(页|页面|幻灯片|标题)",
            normalized,
        )
    )


def normalize_evidence_spans(raw_spans: Any, user_message: str, *, limit: int = 6) -> list[str]:
    source = str(user_message or "")
    if isinstance(raw_spans, str):
        candidates = [raw_spans]
    elif isinstance(raw_spans, list):
        candidates = [str(item or "") for item in raw_spans]
    else:
        candidates = []
    spans: list[str] = []
    for candidate in candidates:
        span = " ".join(candidate.split()).strip()
        if not span:
            continue
        if source and span not in source:
            compact_source = " ".join(source.split())
            if span not in compact_source:
                continue
        if span not in spans:
            spans.append(span[:120])
        if len(spans) >= limit:
            break
    return spans


ANY_TEXT_ELEMENT_KIND = "any_text"
TEXT_BEARING_ELEMENT_KINDS = [
    "slide_title",
    "subtitle",
    "body_text",
    "table_cell",
    "caption",
    "chart_axis_label",
    "chart_data_label",
    "legend_label",
    "callout",
    "pill_tag",
    "footer",
]
BROAD_TEXT_SCOPE_ALIASES = {
    ANY_TEXT_ELEMENT_KIND,
    "all_text",
    "all_text_elements",
    "text_element",
    "text_elements",
    "textual_element",
    "textual_elements",
    "semantic_text",
    "any_text_element",
    "any_text_elements",
}


def infer_semantic_text_target(*texts: str) -> str:
    haystack = " ".join(str(text or "") for text in texts).lower()
    if not haystack:
        return ""
    metric_signals = (
        "实验结果",
        "结果指标",
        "指标数值",
        "关键数值",
        "实验指标",
        "性能指标",
        "metric value",
        "metric values",
        "experimental result",
        "experimental results",
        "result metric",
        "result metrics",
    )
    if any(signal in haystack for signal in metric_signals):
        return "experimental_metric_value_text"
    if (
        any(signal in haystack for signal in ("ffacc", "acc", "accuracy", "auc", "f1"))
        and any(signal in haystack for signal in ("指标", "结果", "数值", "metric", "result"))
    ):
        return "experimental_metric_value_text"
    return ""


def semantic_target_id(raw_target: Any) -> str:
    if isinstance(raw_target, dict):
        return (
            str(raw_target.get("id", "") or raw_target.get("name", "") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
    return str(raw_target or "").strip().lower().replace("-", "_")


def normalize_semantic_target_payload(
    raw_target: Any,
    *,
    user_message: str,
    preference: str,
    general_sentence: str,
    fallback_id: str = "",
) -> dict[str, Any] | str:
    target_id = semantic_target_id(raw_target) or str(fallback_id or "").strip().lower().replace("-", "_")
    if not target_id:
        target_id = infer_semantic_text_target(user_message, preference, general_sentence)
    if not target_id:
        return ""

    if isinstance(raw_target, dict):
        normalized: dict[str, Any] = deepcopy(raw_target)
    else:
        normalized = {"id": target_id}
    normalized["id"] = target_id

    if target_id == "experimental_metric_value_text":
        normalized.setdefault("label", "实验结果/指标数值文本")
        normalized.setdefault(
            "description",
            "表示实验结果、性能指标、消融结果、数据集指标值的数字或短文本。",
        )
        normalized.setdefault(
            "positive_examples",
            ["FFAcc 达到 37.6", "Acc=92.1", "FID 降至 13.5"],
        )
        normalized.setdefault(
            "negative_examples",
            ["第 10 页", "2026 年", "Figure 3", "方法编号 1.2"],
        )
    else:
        normalized.setdefault("label", target_id.replace("user_defined:", "").replace("_", " "))
        normalized.setdefault("description", str(general_sentence or preference or "").strip())

    for list_key in ("positive_examples", "negative_examples"):
        raw_examples = normalized.get(list_key, [])
        if isinstance(raw_examples, str):
            raw_examples = [raw_examples]
        if not isinstance(raw_examples, list):
            raw_examples = []
        examples: list[str] = []
        for item in raw_examples:
            text = " ".join(str(item or "").split()).strip()
            if text and text not in examples:
                examples.append(text[:120])
            if len(examples) >= 8:
                break
        normalized[list_key] = examples
    return normalized


def semantic_target_matches(raw_target: Any, target_id: str) -> bool:
    return semantic_target_id(raw_target) == str(target_id or "").strip().lower().replace("-", "_")


def normalize_css_property_list(raw_properties: Any, *, force_color: bool = False) -> list[str]:
    if isinstance(raw_properties, str):
        candidates = [raw_properties]
    elif isinstance(raw_properties, list):
        candidates = raw_properties
    else:
        candidates = []
    properties: list[str] = []
    for raw_prop in candidates:
        prop = str(raw_prop or "").strip().lower()
        if not prop:
            continue
        prop = {
            "font_weight": "font-weight",
            "fontweight": "font-weight",
            "text_decoration": "text-decoration",
            "textdecoration": "text-decoration",
            "font_style": "font-style",
            "fontstyle": "font-style",
        }.get(prop, prop)
        if prop not in properties:
            properties.append(prop)
    if force_color and "color" not in properties:
        properties.insert(0, "color")
    return properties


def request_targets_broad_semantic_text(*texts: str) -> bool:
    haystack = " ".join(str(text or "") for text in texts).lower()
    if not infer_semantic_text_target(haystack):
        return False
    return bool(
        re.search(
            r"(所有|全部|任意|每个|以后|未来|后续|新增|新生成).{0,24}(表示|属于|实验|指标|结果|数值|文本|文字|metric|result)",
            haystack,
        )
    )


def normalize_session_creation_scope(
    raw_scope: Any,
    *texts: str,
    semantic_target: Any = "",
) -> list[str]:
    if isinstance(raw_scope, str):
        candidates = [raw_scope]
    elif isinstance(raw_scope, list):
        candidates = raw_scope
    else:
        candidates = []

    scope: list[str] = []
    saw_broad_text_scope = False
    for raw_kind in candidates:
        raw_value = str(raw_kind or "").strip().lower().replace("-", "_")
        if raw_value in BROAD_TEXT_SCOPE_ALIASES:
            saw_broad_text_scope = True
            continue
        kind = canonical_session_element_kind(raw_kind, *texts)
        if kind in BROAD_TEXT_SCOPE_ALIASES:
            saw_broad_text_scope = True
            continue
        if kind and kind != "deck" and kind not in scope:
            scope.append(kind)

    if saw_broad_text_scope:
        return [ANY_TEXT_ELEMENT_KIND]

    target_id = semantic_target_id(semantic_target) or infer_semantic_text_target(*texts)
    if target_id and request_targets_broad_semantic_text(*texts):
        if not scope or all(kind in TEXT_BEARING_ELEMENT_KINDS for kind in scope):
            return [ANY_TEXT_ELEMENT_KIND]
    return scope


def canonical_session_element_kind(raw: Any, *texts: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "all_text": ANY_TEXT_ELEMENT_KIND,
        "all_text_elements": ANY_TEXT_ELEMENT_KIND,
        "any_text": ANY_TEXT_ELEMENT_KIND,
        "any_text_element": ANY_TEXT_ELEMENT_KIND,
        "any_text_elements": ANY_TEXT_ELEMENT_KIND,
        "semantic_text": ANY_TEXT_ELEMENT_KIND,
        "text_element": ANY_TEXT_ELEMENT_KIND,
        "text_elements": ANY_TEXT_ELEMENT_KIND,
        "textual_element": ANY_TEXT_ELEMENT_KIND,
        "textual_elements": ANY_TEXT_ELEMENT_KIND,
        "title": "slide_title",
        "heading": "slide_title",
        "h1": "slide_title",
        "body": "body_text",
        "paragraph": "body_text",
        "bullet": "body_text",
        "bullets": "body_text",
        "text": "body_text",
        "badge": "pill_tag",
        "pill": "pill_tag",
        "pilltag": "pill_tag",
        "pill_tag": "pill_tag",
        "pill-tag": "pill_tag",
        "tag": "pill_tag",
        "chip": "pill_tag",
        "label": "pill_tag",
        "caption": "caption",
        "figcaption": "caption",
        "figure_caption": "caption",
        "figurecaption": "caption",
        "image_caption": "caption",
        "imagecaption": "caption",
        "table_caption": "caption",
        "tablecaption": "caption",
        "table_cell": "table_cell",
        "tablecell": "table_cell",
        "cell": "table_cell",
        "td": "table_cell",
        "th": "table_cell",
        "chart_axis_label": "chart_axis_label",
        "axis_label": "chart_axis_label",
        "axis": "chart_axis_label",
        "chart_data_label": "chart_data_label",
        "data_label": "chart_data_label",
        "data_labels": "chart_data_label",
        "legend": "legend_label",
        "legend_label": "legend_label",
        "callout": "callout",
        "call_out": "callout",
        "图注": "caption",
        "图片说明": "caption",
        "图表说明": "caption",
        "表格说明": "caption",
    }
    if value in aliases:
        return aliases[value]
    if value in {
        ANY_TEXT_ELEMENT_KIND,
        "slide_title",
        "body_text",
        "footer",
        "image",
        "pill_tag",
        "deck",
        "subtitle",
        "caption",
        "table_cell",
        "chart_axis_label",
        "chart_data_label",
        "legend_label",
        "callout",
    }:
        return value
    return infer_session_rule_element_kind(*texts)


def infer_session_preference_retention_scope(
    user_message: str,
    raw_scope: str,
    *,
    write_general: bool,
) -> str:
    normalized_scope = str(raw_scope or "").strip().lower()
    if normalized_scope in {"job_local", "intent_profile", "default_profile"}:
        return normalized_scope
    return "job_local"


def infer_session_preference_category(dimension: str, fallback: str = "general") -> str:
    dim = str(dimension or "").strip().lower()
    if dim.startswith("layout"):
        return "layout"
    if dim.startswith("content"):
        return "content"
    if dim.startswith("theme.primary_colors") or dim in {"color", "typography.text_color", "theme.text_color"}:
        return "color"
    if dim.startswith("theme.font") or dim == "typography" or dim.startswith("typography."):
        return "typography"
    if dim.startswith("theme") or dim.startswith("visual"):
        return "style"
    if dim == "general":
        return "general"
    normalized_fallback = str(fallback or "").strip().lower()
    return normalized_fallback if normalized_fallback in {"style", "color", "layout", "typography", "content", "general"} else "general"


def slugify_session_rule(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower())
    return normalized.strip("_")[:40] or "rule"


def extract_session_general_preference_text(user_message: str) -> str:
    text = " ".join(str(user_message or "").split()).strip()
    if not text:
        return ""

    pieces = re.split(r"[。；;，,\n]+|(?:并且|然后|同时|另外|以及)", text)
    candidates: list[str] = []
    for piece in pieces:
        cleaned = re.sub(
            r"^\s*(?:(?:请你|请|帮我|麻烦|并|且|和模型说|告诉模型|跟模型说|记住|模型要|说)\s*)+",
            "",
            piece,
        ).strip()
        if cleaned:
            candidates.append(cleaned)
    candidates.append(text)

    for candidate in candidates:
        if session_preference_has_general_signal(candidate):
            return candidate[:200]
    for candidate in candidates:
        if (
            session_preference_targets_future_slides(candidate)
            and request_mentions_text_color_without_background(candidate)
        ):
            return candidate[:200]
    return ""


def infer_session_rule_element_kind(*texts: str) -> str:
    haystack = " ".join(str(text or "") for text in texts).lower()
    if any(keyword in haystack for keyword in ("标题", "title", "h1", "heading")):
        return "slide_title"
    if any(
        keyword in haystack
        for keyword in (
            "图注",
            "图片说明",
            "图表说明",
            "表格说明",
            "说明文字",
            "caption",
            "figcaption",
            "figure caption",
            "image caption",
            "table caption",
        )
    ):
        return "caption"
    if any(keyword in haystack for keyword in ("pill tag", "pill_tag", "pill-tag", "badge", "chip", "标签", "圆角矩形标签")):
        return "pill_tag"
    if any(keyword in haystack for keyword in ("正文", "bullet", "body", "段落", "要点")):
        return "body_text"
    if any(keyword in haystack for keyword in ("页脚", "footer")):
        return "footer"
    if any(keyword in haystack for keyword in ("图片", "image", "图像")):
        return "image"
    return "deck"


def infer_session_rule_type(dimension: str, *texts: str) -> str:
    dim = str(dimension or "").strip().lower()
    haystack = " ".join(str(text or "") for text in texts).lower()
    if dim.startswith("layout"):
        return "layout"
    if dim.startswith("content"):
        return "content"
    if dim == "general" and any(keyword in haystack for keyword in ("必须", "只能", "不要", "禁止", "strict", "only")):
        return "constraint"
    return "style"


def build_session_rule_spec(
    *,
    user_message: str,
    preference: str,
    dimension: str,
    general_sentence: str,
    retention_scope: str,
    turn: int,
    structured_rule: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = deepcopy(structured_rule) if isinstance(structured_rule, dict) else {}
    target = spec.get("target") if isinstance(spec.get("target"), dict) else {}
    element_kind = canonical_session_element_kind(
        target.get("element_kind", "") if isinstance(target, dict) else "",
        user_message,
        preference,
        general_sentence,
    )
    normalized_dim = str(spec.get("dimension") or dimension or "general").strip()
    spec.setdefault("schema_version", "wm_rule_v1")
    spec.setdefault(
        "rule_id",
        f"wmr_session_turn_{turn:03d}_{slugify_session_rule(preference or general_sentence)}",
    )
    spec["source"] = "session_preference"
    spec["source_turn"] = turn
    spec.setdefault("natural_language", str(user_message or "").strip()[:200] or preference)
    spec["normalized_sentence"] = general_sentence
    spec["dimension"] = normalized_dim
    spec.setdefault(
        "rule_type",
        infer_session_rule_type(normalized_dim, user_message, preference, general_sentence),
    )
    spec["scope"] = "global"

    if not isinstance(target, dict):
        target = {}
    raw_semantic_target = target.get("semantic_target", "")
    semantic_target = normalize_semantic_target_payload(
        raw_semantic_target,
        user_message=user_message,
        preference=preference,
        general_sentence=general_sentence,
        fallback_id=infer_semantic_text_target(user_message, preference, general_sentence),
    )
    target.setdefault("slide_scope", "all")
    target["element_kind"] = canonical_session_element_kind(
        target.get("element_kind", element_kind),
        user_message,
        preference,
        general_sentence,
    )
    if (
        str(target.get("element_kind", "") or "").strip() == "deck"
        and semantic_target
        and request_targets_broad_semantic_text(user_message, preference, general_sentence)
    ):
        target["element_kind"] = ANY_TEXT_ELEMENT_KIND
    if semantic_target:
        target["semantic_target"] = semantic_target
    spec["target"] = target

    action = spec.get("action")
    if not isinstance(action, dict):
        action = {}
    action.setdefault("op", "apply_preference")
    action.setdefault("description", preference or general_sentence)
    existing_css_values = action.get("css_values", {})
    if not isinstance(existing_css_values, dict):
        existing_css_values = {}
    normalized_css_values: dict[str, str] = {}
    for raw_prop, raw_value in existing_css_values.items():
        prop = normalize_css_property_name(str(raw_prop or ""))
        value = str(raw_value or "").strip()
        if prop and value and css_value_is_machine_verifiable(prop, value):
            normalized_css_values[prop] = value
    existing_css_properties = normalize_css_property_list(action.get("css_properties", []))
    css_source_texts = [
        preference,
        general_sentence,
        str(action.get("description", "") or ""),
    ]
    if normalized_dim.lower() in {"typography.text_color", "theme.text_color"} or existing_css_properties:
        css_source_texts.append(user_message)
    extracted_css_values = extract_explicit_css_values_from_text(*css_source_texts)
    for prop, value in extracted_css_values.items():
        normalized_css_values.setdefault(prop, value)
    if normalized_css_values:
        action["css_values"] = normalized_css_values
        action["css_properties"] = normalize_css_property_list(
            existing_css_properties + list(normalized_css_values.keys())
        )
    else:
        action.pop("css_values", None)

    design_tokens = action.get("design_tokens", {})
    if not isinstance(design_tokens, dict):
        design_tokens = {}
    for key, value in extract_soft_design_tokens_from_text(
        user_message,
        preference,
        general_sentence,
        str(action.get("description", "") or ""),
    ).items():
        design_tokens.setdefault(key, value)
    if design_tokens:
        action["design_tokens"] = design_tokens

    if normalized_dim.lower() in {"typography.text_color", "theme.text_color"}:
        action["css_properties"] = normalize_css_property_list(
            action.get("css_properties", []),
            force_color=True,
        )
        action["preserve_css_properties"] = [
            "background",
            "background-color",
            "background-image",
        ]
        action["preserve_semantic_roles"] = ["background", "surface"]
    spec["action"] = action

    condition = spec.get("condition")
    if not isinstance(condition, dict):
        condition = {}
    if semantic_target:
        condition.setdefault("match_granularity", "text_span")
        condition.setdefault("apply_to", "semantic_span_only")
    else:
        condition.setdefault("match_granularity", "element")
        condition.setdefault("apply_to", "whole_element")
    spec["condition"] = condition

    propagation = spec.get("propagation")
    if not isinstance(propagation, dict):
        propagation = {}
    propagation["apply_existing_slides"] = _normalize_bool_flag(
        propagation.get("apply_existing_slides")
    )
    propagation["apply_future_slides"] = _normalize_bool_flag(
        propagation.get("apply_future_slides")
    )
    propagation["apply_future_elements"] = _normalize_bool_flag(
        propagation.get("apply_future_elements")
    )
    raw_creation_scope = propagation.get("creation_scope", [])
    creation_scope = normalize_session_creation_scope(
        raw_creation_scope,
        user_message,
        preference,
        general_sentence,
        semantic_target=semantic_target,
    )
    if propagation["apply_future_elements"] and not creation_scope and element_kind != "deck":
        creation_scope.append(element_kind)
    propagation["creation_scope"] = creation_scope
    spec["propagation"] = propagation
    if semantic_target:
        target["semantic_target"] = semantic_target
        spec["target"] = target

    verification = spec.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    verification["must_cover_all_targets"] = bool(propagation["apply_existing_slides"])
    verification.setdefault("hard_check", "css_values_only")
    spec["verification"] = verification

    spec["retention_scope"] = retention_scope
    try:
        spec["confidence"] = float(spec.get("confidence", 0.8))
    except (TypeError, ValueError):
        spec["confidence"] = 0.8
    return spec


def empty_session_preference_payload() -> dict[str, Any]:
    return {
        "preference": "",
        "category": "general",
        "dimension": "",
        "write_general": False,
        "general_sentence": "",
        "retention_scope": "job_local",
        "structured_rule": None,
    }


def _first_memory_update(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {}
    updates = raw_payload.get("memory_updates")
    if isinstance(updates, list):
        for item in updates:
            if isinstance(item, dict):
                return item
    return raw_payload


def normalize_session_preference_payload(
    raw_payload: Any,
    *,
    user_message: str,
    turn: int,
) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return empty_session_preference_payload()
    raw_payload = _first_memory_update(raw_payload)
    if not raw_payload:
        return empty_session_preference_payload()

    preference = str(raw_payload.get("preference", "") or "").strip()
    category = str(raw_payload.get("category", "general") or "general").strip().lower()
    dimension = str(raw_payload.get("dimension", "") or "").strip()

    text_color_request = request_mentions_text_color_without_background(user_message)
    if not dimension and category in {"color", "layout", "typography", "content"}:
        dimension = category
    if text_color_request:
        if dimension.lower() in {"", "color", "general", "theme.primary_colors"}:
            dimension = "typography.text_color"

    structured_rule = raw_payload.get("structured_rule")
    if not isinstance(structured_rule, dict):
        structured_rule = None
    else:
        structured_rule = deepcopy(structured_rule)
        target = structured_rule.get("target")
        if not isinstance(target, dict):
            target = {}
        target["element_kind"] = canonical_session_element_kind(
            target.get("element_kind", ""),
            user_message,
            preference,
            str(raw_payload.get("general_sentence", "") or ""),
        )
        structured_rule["target"] = target

        structured_dim = str(structured_rule.get("dimension", "") or "").strip().lower()
        css_properties = (
            normalize_css_property_list(structured_rule.get("action", {}).get("css_properties", []))
            if isinstance(structured_rule.get("action"), dict)
            else []
        )
        if (
            text_color_request
            or "color" in css_properties
        ):
            if structured_dim in {"", "color", "general", "theme.primary_colors", "theme.text_color"}:
                structured_rule["dimension"] = "typography.text_color"
                dimension = "typography.text_color"
        elif structured_dim and not dimension:
            dimension = structured_dim

    propagation = structured_rule.get("propagation", {}) if structured_rule else {}
    write_general = _normalize_bool_flag(raw_payload.get("write_general"))
    if not write_general and isinstance(propagation, dict) and structured_rule is not None:
        write_general = (
            _normalize_bool_flag(propagation.get("apply_future_slides"))
            or _normalize_bool_flag(propagation.get("apply_future_elements"))
            or (
                str(structured_rule.get("scope", "") or "").strip().lower() == "global"
                and _normalize_bool_flag(propagation.get("apply_existing_slides"))
            )
        )

    general_sentence = str(raw_payload.get("general_sentence", "") or "").strip()
    if write_general and not general_sentence and preference:
        general_sentence = preference

    retention_scope = infer_session_preference_retention_scope(
        user_message,
        str(raw_payload.get("retention_scope", "") or ""),
        write_general=write_general,
    )
    category = infer_session_preference_category(dimension, fallback=category)

    if write_general and general_sentence and structured_rule is not None:
        structured_rule = build_session_rule_spec(
            user_message=user_message,
            preference=preference,
            dimension=dimension or "general",
            general_sentence=general_sentence,
            retention_scope=retention_scope,
            turn=turn,
            structured_rule=structured_rule,
        )
    elif not write_general:
        structured_rule = None

    return {
        "preference": preference,
        "category": category,
        "dimension": dimension,
        "write_general": write_general,
        "general_sentence": general_sentence,
        "retention_scope": retention_scope,
        "structured_rule": structured_rule,
    }


def preference_semantic_deck_status(runtime: Any) -> list[dict[str, Any]]:
    slides: list[dict[str, Any]] = []
    for path in resolve_all_slide_paths(runtime)[:40]:
        slides.append(
            {
                "slide": path.stem,
                "path": workspace_relative_label(runtime, path),
            }
        )
    return slides


def render_preference_semantic_compiler_prompt(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    existing_rules_text: str = "",
) -> str:
    return _PREFERENCE_SEMANTIC_COMPILER_PROMPT.format(
        deck_status_json=json.dumps(preference_semantic_deck_status(runtime), ensure_ascii=False, indent=2),
        intent_json=json.dumps(intent or {}, ensure_ascii=False, indent=2),
        existing_rules=str(existing_rules_text or "")[:6000] or "(none)",
        user_message=str(user_message or "")[:2000],
    )


def render_preference_semantic_critic_prompt(
    *,
    user_message: str,
    compiler_payload: dict[str, Any],
) -> str:
    return _PREFERENCE_SEMANTIC_CRITIC_PROMPT.format(
        user_message=str(user_message or "")[:2000],
        compiler_json=json.dumps(_json_safe(compiler_payload), ensure_ascii=False, indent=2)[:8000],
    )


async def _run_llm_text(llm: Any, prompt: str) -> str:
    if llm is None:
        return ""
    if callable(llm) and not hasattr(llm, "run"):
        result = await llm(prompt)
        return str(result or "")
    try:
        response = await llm.run(prompt)
    except TypeError:
        response = await llm.run(messages=[{"role": "user", "content": prompt}])
    return str(response.choices[0].message.content or "")


def semantic_compilation_fail_closed(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    reason: str,
    source: str,
    compiler_payload: dict[str, Any] | None = None,
    critic_payload: dict[str, Any] | None = None,
    compiler_artifacts: tuple[str, str, str] = ("", "", ""),
    critic_artifacts: tuple[str, str, str] = ("", "", ""),
) -> PreferenceSemanticCompilation:
    payload = compiler_payload if isinstance(compiler_payload, dict) else {}
    payload.setdefault("memory_flow", "ambiguous_fail_closed")
    payload.setdefault("memory_updates", [])
    compilation = PreferenceSemanticCompilation(
        payload=payload,
        source=source,
        memory_flow="ambiguous_fail_closed",
        valid=False,
        fail_closed=True,
        reason=reason,
        compiler_prompt_artifact=compiler_artifacts[0],
        compiler_response_artifact=compiler_artifacts[1],
        compiler_payload_artifact=compiler_artifacts[2],
        critic_prompt_artifact=critic_artifacts[0],
        critic_response_artifact=critic_artifacts[1],
        critic_payload_artifact=critic_artifacts[2],
    )
    compilation.trace_artifact = append_preference_semantic_trace(
        runtime,
        compilation,
        user_message=user_message,
        intent=intent,
        critic_payload=critic_payload,
        note=reason,
    )
    append_preference_semantic_stage_trace(
        runtime,
        event="preference_semantic_compilation_failed",
        user_message=user_message,
        intent=intent,
        extra={
            "reason": reason,
            "source": source,
            "compiler_payload_artifact": compiler_artifacts[2],
            "critic_payload_artifact": critic_artifacts[2],
        },
    )
    logger.warning("Preference semantic compilation fail-closed: %s", reason)
    return compilation


def _semantic_compiler_update_payload(
    compiler_payload: dict[str, Any],
) -> dict[str, Any]:
    updates = compiler_payload.get("memory_updates")
    if isinstance(updates, list):
        for item in updates:
            if isinstance(item, dict):
                return item
    return {}


def _semantic_compiler_update_payloads(
    compiler_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    updates = compiler_payload.get("memory_updates")
    if isinstance(updates, list):
        return [item for item in updates if isinstance(item, dict)]
    update = _semantic_compiler_update_payload(compiler_payload)
    return [update] if update else []


def _semantic_propagation_payload(
    compiler_payload: dict[str, Any],
) -> dict[str, Any]:
    propagation = compiler_payload.get("propagation_decision")
    return propagation if isinstance(propagation, dict) else {}


def _semantic_current_existing_batch_spans(
    compiler_payload: dict[str, Any],
    user_message: str,
) -> list[str]:
    actions = compiler_payload.get("current_actions")
    if not isinstance(actions, list):
        return []

    batch_targets = {
        "all",
        "deck",
        "global",
        "current_deck",
        "current deck",
        "all_slides",
        "all slides",
        "existing_slides",
        "existing slides",
        "现有页面",
        "已有页面",
        "当前页面",
        "所有页面",
        "全部页面",
        "整套",
    }
    spans: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type", "") or "").strip().lower()
        target_slide = str(action.get("target_slide", "") or "").strip().lower()
        if action_type != "edit_existing" or target_slide not in batch_targets:
            continue
        for span in normalize_evidence_spans(action.get("evidence_spans", []), user_message):
            if span not in spans:
                spans.append(span)
    return spans


def _structured_rule_dimension(structured_rule: dict[str, Any], fallback: str = "") -> str:
    dimension = str(structured_rule.get("dimension") or fallback or "").strip()
    if dimension:
        return dimension
    action = structured_rule.get("action") if isinstance(structured_rule.get("action"), dict) else {}
    css_properties = normalize_css_property_list(action.get("css_properties", [])) if isinstance(action, dict) else []
    if "color" in css_properties:
        return "typography.text_color"
    rule_type = str(structured_rule.get("rule_type") or "").strip().lower()
    if rule_type == "content":
        return "content.language_style"
    if rule_type == "layout":
        return "layout.slide_structure"
    return ""


def _rule_text_for_value_extraction(
    *,
    user_message: str,
    update: dict[str, Any],
    structured_rule: dict[str, Any],
    dimension: str,
) -> list[str]:
    action = structured_rule.get("action") if isinstance(structured_rule.get("action"), dict) else {}
    target = structured_rule.get("target") if isinstance(structured_rule.get("target"), dict) else {}
    target_text = " ".join(
        str(value or "")
        for value in (
            target.get("element_kind"),
            target.get("semantic_target", {}).get("id") if isinstance(target.get("semantic_target"), dict) else "",
            target.get("semantic_target", {}).get("label") if isinstance(target.get("semantic_target"), dict) else "",
            action.get("description") if isinstance(action, dict) else "",
            update.get("preference"),
            update.get("general_sentence"),
        )
    )
    dimension_text = str(dimension or structured_rule.get("dimension") or update.get("dimension") or "").lower()
    wants_text_style = (
        dimension_text in {"typography.text_color", "theme.text_color"}
        or "typography" in dimension_text
        or bool(re.search(r"(title|heading|标题|text|文字|文本|font|字体|caption|图注|说明)", target_text, flags=re.IGNORECASE))
    )
    if wants_text_style:
        return [
            str(update.get("preference") or ""),
            str(update.get("general_sentence") or ""),
            str(action.get("description") if isinstance(action, dict) else ""),
            str(user_message or ""),
        ]
    return [
        str(update.get("preference") or ""),
        str(update.get("general_sentence") or ""),
        str(action.get("description") if isinstance(action, dict) else ""),
    ]


def _normalize_single_preference_semantic_update(
    compiler_payload: dict[str, Any],
    update: dict[str, Any],
    *,
    user_message: str,
    critic_payload: dict[str, Any],
    memory_flow: str,
    turn: int,
    apply_existing: bool,
    apply_future: bool,
    apply_future_elements: bool,
    existing_spans: list[str],
    future_spans: list[str],
    future_element_spans: list[str],
    propagation_decision: dict[str, Any],
    used_critic_correction: bool,
) -> tuple[dict[str, Any], str] | tuple[None, str]:
    update_spans = normalize_evidence_spans(update.get("evidence_spans", []), user_message)
    preference_text = str(update.get("preference", "") or "")
    general_sentence_text = str(update.get("general_sentence", "") or "")
    raw_structured_rule = update.get("structured_rule")
    raw_target = raw_structured_rule.get("target") if isinstance(raw_structured_rule, dict) else {}
    raw_semantic_target = raw_target.get("semantic_target", "") if isinstance(raw_target, dict) else ""
    semantic_target = normalize_semantic_target_payload(
        raw_semantic_target,
        user_message=user_message,
        preference=preference_text,
        general_sentence=general_sentence_text,
        fallback_id=infer_semantic_text_target(
            user_message,
            preference_text,
            general_sentence_text,
        ),
    )
    raw_rule_propagation = (
        raw_structured_rule.get("propagation")
        if isinstance(raw_structured_rule, dict) and isinstance(raw_structured_rule.get("propagation"), dict)
        else {}
    )
    raw_creation_scope = raw_rule_propagation.get("creation_scope", propagation_decision.get("creation_scope", []))
    creation_scope = normalize_session_creation_scope(
        raw_creation_scope,
        user_message,
        preference_text,
        general_sentence_text,
        semantic_target=semantic_target,
    )
    if apply_future_elements and not creation_scope:
        update_rule_target = raw_structured_rule.get("target") if isinstance(raw_structured_rule, dict) else {}
        target_kind = (
            canonical_session_element_kind(
                update_rule_target.get("element_kind", "") if isinstance(update_rule_target, dict) else "",
                user_message,
                preference_text,
                general_sentence_text,
            )
            if isinstance(update_rule_target, dict)
            else "deck"
        )
        if target_kind != "deck":
            creation_scope = [target_kind]
    if apply_future_elements and not creation_scope:
        return None, "future-element propagation lacks creation_scope"

    if not update_spans and not existing_spans and not future_spans and not future_element_spans:
        return None, "memory update lacks user evidence span"

    structured_rule = update.get("structured_rule")
    if not isinstance(structured_rule, dict):
        structured_rule = {}
    else:
        structured_rule = deepcopy(structured_rule)
    target = structured_rule.get("target")
    if not isinstance(target, dict):
        target = {}
    target["element_kind"] = canonical_session_element_kind(
        target.get("element_kind", ""),
        user_message,
        preference_text,
        general_sentence_text,
    )
    if semantic_target:
        target["semantic_target"] = semantic_target
    if (
        target["element_kind"] == "deck"
        and semantic_target
        and request_targets_broad_semantic_text(
            user_message,
            preference_text,
            general_sentence_text,
        )
    ):
        target["element_kind"] = ANY_TEXT_ELEMENT_KIND
    if (apply_future or apply_future_elements) and not apply_existing:
        target["slide_scope"] = "future"
    elif apply_existing and not (apply_future or apply_future_elements):
        target["slide_scope"] = "existing"
    else:
        target["slide_scope"] = "all"
    structured_rule["target"] = target

    propagation = structured_rule.get("propagation")
    if not isinstance(propagation, dict):
        propagation = {}
    propagation["apply_existing_slides"] = apply_existing
    propagation["apply_future_slides"] = apply_future
    propagation["apply_future_elements"] = apply_future_elements
    propagation["creation_scope"] = creation_scope
    structured_rule["propagation"] = propagation
    structured_rule["evidence_spans"] = {
        "preference": update_spans,
        "existing": existing_spans,
        "future": future_spans,
        "future_elements": future_element_spans,
    }

    dimension = _structured_rule_dimension(
        structured_rule,
        str(update.get("dimension", "") or ""),
    )
    action = structured_rule.get("action") if isinstance(structured_rule.get("action"), dict) else {}
    css_props = normalize_css_property_list(action.get("css_properties", [])) if isinstance(action, dict) else []
    if "color" in css_props and dimension.lower() in {"", "color", "general", "theme.primary_colors", "theme.text_color"}:
        dimension = "typography.text_color"
    if dimension:
        structured_rule["dimension"] = dimension

    payload = {
        "preference": preference_text.strip(),
        "category": str(update.get("category", "") or "general").strip().lower(),
        "dimension": dimension,
        "write_general": True,
        "general_sentence": str(update.get("general_sentence", "") or update.get("preference", "") or "").strip(),
        "retention_scope": str(update.get("retention_scope", "") or "job_local").strip(),
        "structured_rule": structured_rule,
    }
    normalized = normalize_session_preference_payload(
        payload,
        user_message=user_message,
        turn=turn,
    )
    if not normalized.get("preference") and not normalized.get("general_sentence"):
        return None, "normalized compiler payload is empty"
    if not isinstance(normalized.get("structured_rule"), dict):
        return None, "normalized compiler payload has no structured rule"

    normalized_rule = normalized["structured_rule"]
    action = normalized_rule.get("action") if isinstance(normalized_rule.get("action"), dict) else {}
    if action and (
        str(normalized.get("dimension") or "").strip().lower() in {"typography.text_color", "theme.text_color"}
        or normalize_css_property_list(action.get("css_properties", []))
    ):
        existing_css_values = action.get("css_values", {})
        if not isinstance(existing_css_values, dict):
            existing_css_values = {}
        css_values = {
            normalize_css_property_name(str(prop or "")): str(value or "").strip()
            for prop, value in existing_css_values.items()
            if normalize_css_property_name(str(prop or ""))
            and css_value_is_machine_verifiable(str(prop or ""), str(value or "").strip())
        }
        for prop, value in extract_explicit_css_values_from_text(
            *_rule_text_for_value_extraction(
                user_message=user_message,
                update=update,
                structured_rule=normalized_rule,
                dimension=str(normalized.get("dimension") or ""),
            )
        ).items():
            css_values.setdefault(prop, value)
        css_properties = normalize_css_property_list(action.get("css_properties", []))
        if css_values:
            action["css_values"] = css_values
            action["css_properties"] = normalize_css_property_list(
                css_properties + list(css_values.keys())
            )
        else:
            action.pop("css_values", None)
            if css_properties:
                action["css_properties"] = css_properties
        normalized_rule["action"] = action

    normalized_rule["semantic_compiler"] = {
        "memory_flow": memory_flow,
        "source": "llm",
        "confidence": compiler_payload.get("confidence"),
        "critic_correction_applied": used_critic_correction,
        "evidence_spans": {
            "preference": update_spans,
            "existing": existing_spans,
            "future": future_spans,
            "future_elements": future_element_spans,
        },
        "creation_scope": creation_scope,
    }
    return normalized, "approved_with_critic_correction" if used_critic_correction else "approved"


def _normalize_preference_semantic_payload(
    runtime: Any,
    compiler_payload: dict[str, Any],
    *,
    user_message: str,
    intent: dict[str, Any] | None,
    critic_payload: dict[str, Any],
    turn: int,
) -> tuple[list[dict[str, Any]], str, str]:
    if not isinstance(compiler_payload, dict) or not compiler_payload:
        return [], "ambiguous_fail_closed", "compiler returned no JSON object"

    memory_flow = str(compiler_payload.get("memory_flow", "") or "").strip().lower()
    if not memory_flow:
        memory_flow = "none"
    if memory_flow == "ambiguous":
        memory_flow = "ambiguous_fail_closed"
    if memory_flow == "ambiguous_fail_closed":
        return [], memory_flow, "compiler marked the request ambiguous"

    updates = _semantic_compiler_update_payloads(compiler_payload)
    if memory_flow in {"none", "current_edit_only"} or not updates:
        return [], memory_flow, "no memory update requested by compiler"

    if memory_flow == "session_dimension_note":
        normalized_payloads: list[dict[str, Any]] = []
        reasons: list[str] = []
        for update in updates:
            update_spans = normalize_evidence_spans(update.get("evidence_spans", []), user_message)
            preference = str(update.get("preference", "") or update.get("general_sentence", "") or "").strip()
            if not preference:
                reasons.append("session note preference is empty")
                continue
            if not update_spans:
                reasons.append("session note lacks user evidence span")
                continue
            dimension = str(update.get("dimension", "") or "").strip()
            if not dimension:
                category = str(update.get("category", "") or "general").strip().lower()
                dimension = category if category in {"style", "color", "layout", "typography", "content"} else "general"
            payload = {
                "preference": preference,
                "category": str(update.get("category", "") or infer_session_preference_category(dimension)).strip().lower(),
                "dimension": dimension,
                "write_general": False,
                "general_sentence": str(update.get("general_sentence", "") or "").strip(),
                "retention_scope": str(update.get("retention_scope", "") or "job_local").strip(),
                "structured_rule": None,
                "note_kind": "session_dimension_note",
            }
            normalized = normalize_session_preference_payload(
                payload,
                user_message=user_message,
                turn=turn,
            )
            normalized["write_general"] = False
            normalized["structured_rule"] = None
            normalized["note_kind"] = "session_dimension_note"
            if normalized.get("preference") or normalized.get("general_sentence"):
                normalized_payloads.append(normalized)
            else:
                reasons.append("normalized session note is empty")
        if not normalized_payloads:
            return [], "ambiguous_fail_closed", reasons[0] if reasons else "session dimension note payload is empty"
        return normalized_payloads, memory_flow, "session_dimension_note"

    propagation_decision = _semantic_propagation_payload(compiler_payload)
    apply_existing = _normalize_bool_flag(propagation_decision.get("apply_existing_slides"))
    apply_future = _normalize_bool_flag(propagation_decision.get("apply_future_slides"))
    apply_future_elements = _normalize_bool_flag(
        propagation_decision.get("apply_future_elements")
    )
    existing_spans = normalize_evidence_spans(
        propagation_decision.get("existing_evidence_spans", []),
        user_message,
    )
    future_spans = normalize_evidence_spans(
        propagation_decision.get("future_evidence_spans", []),
        user_message,
    )
    future_element_spans = normalize_evidence_spans(
        propagation_decision.get("future_element_evidence_spans", []),
        user_message,
    )
    update_spans_all: list[str] = []
    for update in updates:
        for span in normalize_evidence_spans(update.get("evidence_spans", []), user_message):
            if span not in update_spans_all:
                update_spans_all.append(span)

    current_existing_spans = _semantic_current_existing_batch_spans(
        compiler_payload,
        user_message,
    )
    if (
        memory_flow == "existing_batch_plus_future_rule"
        and not apply_existing
        and (apply_future or apply_future_elements)
        and current_existing_spans
    ):
        apply_existing = True
        existing_spans = list(existing_spans)
        for span in current_existing_spans:
            if span not in existing_spans:
                existing_spans.append(span)
        propagation_decision = deepcopy(propagation_decision)
        propagation_decision["apply_existing_slides"] = True
        propagation_decision["existing_evidence_spans"] = existing_spans

    critic_approved = _normalize_bool_flag(critic_payload.get("approved"))
    critic_corrected = (
        critic_payload.get("corrected_propagation_decision")
        if isinstance(critic_payload.get("corrected_propagation_decision"), dict)
        else {}
    )
    used_critic_correction = False
    if not critic_approved and critic_corrected:
        corrected_existing = _normalize_bool_flag(critic_corrected.get("apply_existing_slides"))
        corrected_future = _normalize_bool_flag(critic_corrected.get("apply_future_slides"))
        corrected_future_elements = _normalize_bool_flag(
            critic_corrected.get("apply_future_elements")
        )
        corrected_existing_spans = normalize_evidence_spans(
            critic_corrected.get("existing_evidence_spans", []),
            user_message,
        )
        corrected_future_spans = normalize_evidence_spans(
            critic_corrected.get("future_evidence_spans", []),
            user_message,
        )
        corrected_future_element_spans = normalize_evidence_spans(
            critic_corrected.get("future_element_evidence_spans", []),
            user_message,
        )
        corrected_creation_scope = normalize_session_creation_scope(
            critic_corrected.get("creation_scope", []),
            user_message,
        )
        if (
            corrected_existing == apply_existing
            and corrected_future == apply_future
            and corrected_future_elements == apply_future_elements
            and (
                not apply_future_elements
                or corrected_creation_scope
                or not propagation_decision.get("creation_scope")
            )
        ):
            if corrected_existing_spans or not apply_existing:
                existing_spans = corrected_existing_spans if apply_existing else []
            if corrected_future_spans or not apply_future:
                future_spans = corrected_future_spans if apply_future else []
            if corrected_future_element_spans or not apply_future_elements:
                future_element_spans = (
                    corrected_future_element_spans if apply_future_elements else []
                )
            propagation_decision = deepcopy(propagation_decision)
            propagation_decision.update(
                {
                    "apply_existing_slides": apply_existing,
                    "apply_future_slides": apply_future,
                    "apply_future_elements": apply_future_elements,
                    "creation_scope": corrected_creation_scope,
                    "existing_evidence_spans": existing_spans,
                    "future_evidence_spans": future_spans,
                    "future_element_evidence_spans": future_element_spans,
                }
            )
            used_critic_correction = True
            critic_approved = True

    if apply_existing and not existing_spans:
        return [], "ambiguous_fail_closed", "existing-slide propagation lacks user evidence span"
    if apply_future and not future_spans:
        return [], "ambiguous_fail_closed", "future-slide propagation lacks user evidence span"
    if apply_future_elements and not future_element_spans:
        return [], "ambiguous_fail_closed", "future-element propagation lacks user evidence span"
    if apply_existing and memory_flow in {"memory_only", "current_edit_plus_future_rule", "structural_insert_with_future_rule"}:
        return [], "ambiguous_fail_closed", f"memory_flow={memory_flow} cannot propagate to existing slides"
    if memory_flow == "existing_batch_plus_future_rule" and not apply_existing:
        return [], "ambiguous_fail_closed", "existing batch flow lacks existing-slide propagation"
    if memory_flow in {"current_edit_plus_future_rule", "structural_insert_with_future_rule"} and not (apply_future or apply_future_elements):
        return [], "ambiguous_fail_closed", f"memory_flow={memory_flow} lacks future propagation"
    if not apply_existing and not apply_future and not apply_future_elements:
        return [], "ambiguous_fail_closed", "memory update has no propagation target"
    if not update_spans_all and not existing_spans and not future_spans and not future_element_spans:
        return [], "ambiguous_fail_closed", "memory update lacks user evidence span"
    if not critic_approved:
        return [], "ambiguous_fail_closed", "critic rejected compiler propagation"
    if used_critic_correction:
        critic_existing = apply_existing
        critic_future = apply_future
        critic_future_elements = apply_future_elements
    else:
        critic_existing = _normalize_bool_flag(critic_payload.get("should_modify_existing_slides"))
        critic_future = _normalize_bool_flag(critic_payload.get("should_apply_future_slides"))
        critic_future_elements = _normalize_bool_flag(
            critic_payload.get("should_apply_future_elements")
        )
    if apply_existing and not critic_existing:
        return [], "ambiguous_fail_closed", "critic says existing slides should not be modified"
    if apply_future and not critic_future:
        return [], "ambiguous_fail_closed", "critic says future slides should not apply"
    if apply_future_elements and not critic_future_elements:
        return [], "ambiguous_fail_closed", "critic says future elements should not apply"

    normalized_payloads: list[dict[str, Any]] = []
    reasons: list[str] = []
    for update in updates:
        normalized, update_reason = _normalize_single_preference_semantic_update(
            compiler_payload,
            update,
            user_message=user_message,
            critic_payload=critic_payload,
            memory_flow=memory_flow,
            turn=turn,
            apply_existing=apply_existing,
            apply_future=apply_future,
            apply_future_elements=apply_future_elements,
            existing_spans=existing_spans,
            future_spans=future_spans,
            future_element_spans=future_element_spans,
            propagation_decision=propagation_decision,
            used_critic_correction=used_critic_correction,
        )
        if normalized is not None:
            normalized_payloads.append(normalized)
        else:
            reasons.append(update_reason)

    if not normalized_payloads:
        return [], "ambiguous_fail_closed", reasons[0] if reasons else "normalized compiler payload is empty"
    return normalized_payloads, memory_flow, "approved_with_critic_correction" if used_critic_correction else "approved"


async def compile_preference_semantics_with_llm(
    runtime: Any,
    *,
    user_message: str,
    intent: dict[str, Any] | None = None,
    existing_rules_text: str = "",
    llm: Any | None = None,
) -> PreferenceSemanticCompilation:
    prompt = render_preference_semantic_compiler_prompt(
        runtime,
        user_message=user_message,
        intent=intent,
        existing_rules_text=existing_rules_text,
    )
    append_preference_semantic_stage_trace(
        runtime,
        event="preference_semantic_compilation_started",
        user_message=user_message,
        intent=intent,
        extra={"existing_rules_text_length": len(str(existing_rules_text or ""))},
    )
    raw_response = ""
    compiler_payload: dict[str, Any] = {}
    compiler_artifacts = ("", "", "")
    try:
        raw_response = await _run_llm_text(llm, prompt)
        compiler_payload = extract_json_object(raw_response)
        compiler_artifacts = write_preference_semantic_artifacts(
            runtime,
            prompt=prompt,
            raw_response=raw_response,
            payload=compiler_payload,
            stage="compiler",
            source="llm",
        )
    except Exception as exc:
        compiler_artifacts = write_preference_semantic_artifacts(
            runtime,
            prompt=prompt,
            raw_response=f"compiler_error: {exc}\n\n{raw_response or ''}",
            payload={},
            stage="compiler",
            source="error",
        )
        return semantic_compilation_fail_closed(
            runtime,
            user_message=user_message,
            intent=intent,
            reason=f"compiler LLM failed: {exc}",
            source="llm_error",
            compiler_artifacts=compiler_artifacts,
        )

    if not compiler_payload:
        return semantic_compilation_fail_closed(
            runtime,
            user_message=user_message,
            intent=intent,
            reason="compiler returned empty or non-JSON response",
            source="llm_empty",
            compiler_payload={},
            compiler_artifacts=compiler_artifacts,
        )

    critic_prompt = render_preference_semantic_critic_prompt(
        user_message=user_message,
        compiler_payload=compiler_payload,
    )
    critic_response = ""
    critic_payload: dict[str, Any] = {}
    critic_artifacts = ("", "", "")
    try:
        critic_response = await _run_llm_text(llm, critic_prompt)
        critic_payload = extract_json_object(critic_response)
        critic_artifacts = write_preference_semantic_artifacts(
            runtime,
            prompt=critic_prompt,
            raw_response=critic_response,
            payload=critic_payload,
            stage="critic",
            source="llm",
        )
    except Exception as exc:
        critic_artifacts = write_preference_semantic_artifacts(
            runtime,
            prompt=critic_prompt,
            raw_response=f"critic_error: {exc}\n\n{critic_response or ''}",
            payload={},
            stage="critic",
            source="error",
        )
        return semantic_compilation_fail_closed(
            runtime,
            user_message=user_message,
            intent=intent,
            reason=f"critic LLM failed: {exc}",
            source="critic_error",
            compiler_payload=compiler_payload,
            compiler_artifacts=compiler_artifacts,
            critic_artifacts=critic_artifacts,
        )

    if not critic_payload:
        return semantic_compilation_fail_closed(
            runtime,
            user_message=user_message,
            intent=intent,
            reason="critic returned empty or non-JSON response",
            source="critic_empty",
            compiler_payload=compiler_payload,
            compiler_artifacts=compiler_artifacts,
            critic_artifacts=critic_artifacts,
        )

    normalized_payloads, memory_flow, reason = _normalize_preference_semantic_payload(
        runtime,
        compiler_payload,
        user_message=user_message,
        intent=intent,
        critic_payload=critic_payload,
        turn=_current_modify_turn(runtime),
    )
    if memory_flow == "ambiguous_fail_closed":
        return semantic_compilation_fail_closed(
            runtime,
            user_message=user_message,
            intent=intent,
            reason=reason,
            source="semantic_guard",
            compiler_payload=compiler_payload,
            critic_payload=critic_payload,
            compiler_artifacts=compiler_artifacts,
            critic_artifacts=critic_artifacts,
        )

    normalized_payload = normalized_payloads[0] if normalized_payloads else empty_session_preference_payload()
    compilation = PreferenceSemanticCompilation(
        payload=normalized_payload,
        source="llm",
        memory_flow=memory_flow,
        valid=(
            any(bool(payload.get("write_general")) for payload in normalized_payloads)
            or (
                memory_flow == "session_dimension_note"
                and any(
                    bool(payload.get("preference") or payload.get("general_sentence"))
                    and not isinstance(payload.get("structured_rule"), dict)
                    for payload in normalized_payloads
                )
            )
        ),
        fail_closed=False,
        reason=reason,
        payloads=normalized_payloads,
        compiler_prompt_artifact=compiler_artifacts[0],
        compiler_response_artifact=compiler_artifacts[1],
        compiler_payload_artifact=compiler_artifacts[2],
        critic_prompt_artifact=critic_artifacts[0],
        critic_response_artifact=critic_artifacts[1],
        critic_payload_artifact=critic_artifacts[2],
    )
    compilation.trace_artifact = append_preference_semantic_trace(
        runtime,
        compilation,
        user_message=user_message,
        intent=intent,
        critic_payload=critic_payload,
        note=reason,
    )
    append_preference_semantic_stage_trace(
        runtime,
        event="preference_semantic_compilation_succeeded",
        user_message=user_message,
        intent=intent,
        extra={
            "memory_flow": compilation.memory_flow,
            "valid": compilation.valid,
            "reason": compilation.reason,
            "trace_artifact": compilation.trace_artifact,
        },
    )
    logger.info(
        "Preference semantic compilation: flow=%s valid=%s reason=%s trace=%s",
        compilation.memory_flow,
        compilation.valid,
        compilation.reason,
        compilation.trace_artifact,
    )
    return compilation


def _get_orchestrator_wm(memory_orchestrator: Any) -> Any | None:
    try:
        job_mgr = getattr(memory_orchestrator, "_job_mgr", None)
        return getattr(job_mgr, "working_memory", None) if job_mgr is not None else None
    except Exception:
        return None


async def _ensure_session_preference_orchestrator(
    runtime: Any,
    *,
    user_message: str,
) -> Any | None:
    memory_orchestrator = getattr(runtime, "_memory_orchestrator_instance", None)
    if memory_orchestrator is None:
        create_orchestrator = getattr(runtime, "_create_orchestrator", None)
        if callable(create_orchestrator):
            try:
                memory_orchestrator = create_orchestrator()
            except Exception as exc:
                logger.warning(
                    "Session preference could not create MemoryOrchestrator (non-fatal): %s",
                    exc,
                )
                memory_orchestrator = None

    if memory_orchestrator is None:
        return None
    if _get_orchestrator_wm(memory_orchestrator) is not None:
        return memory_orchestrator

    ensure_job_started = getattr(runtime, "_ensure_memory_job_started", None)
    if callable(ensure_job_started):
        try:
            await ensure_job_started(
                user_prompt=user_message,
                intent=getattr(runtime, "_resolved_request_intent", ""),
                read_intent=getattr(runtime, "_resolved_request_intent", ""),
                write_intent=getattr(runtime, "_resolved_request_intent", ""),
                reason="session_preference_priming",
            )
        except Exception as exc:
            logger.warning(
                "Session preference could not start memory job through runtime (non-fatal): %s",
                exc,
            )
        memory_orchestrator = getattr(runtime, "_memory_orchestrator_instance", None)
        if memory_orchestrator is not None and _get_orchestrator_wm(memory_orchestrator) is not None:
            return memory_orchestrator

    if hasattr(memory_orchestrator, "on_job_start"):
        try:
            await memory_orchestrator.on_job_start(
                user_id=getattr(runtime, "user_id", "default"),
                project_id=getattr(getattr(runtime, "workspace", None), "stem", "")
                or getattr(runtime, "session_id", ""),
                user_prompt=user_message or "",
                intent=getattr(runtime, "_resolved_request_intent", "") or "",
                read_intent=getattr(runtime, "_resolved_request_intent", "") or "",
                write_intent=getattr(runtime, "_resolved_request_intent", "") or "",
                core_persona="",
            )
        except Exception as exc:
            logger.warning(
                "Session preference could not start memory job through orchestrator (non-fatal): %s",
                exc,
            )

    return memory_orchestrator


def _format_session_preference_message(
    *,
    turn: int,
    category: str,
    dimension: str,
    write_general: bool,
    retention_scope: str,
    preference: str,
    general_sentence: str,
) -> str:
    session_lines = [line for line in (preference, general_sentence) if line]
    return (
        f'<session_preference turn="{turn}" category="{category}" '
        f'dimension="{dimension or ""}" general="{str(write_general).lower()}" '
        f'retention_scope="{retention_scope}">\n'
        + "\n".join(session_lines)
        + "\n</session_preference>"
    )


def _cache_session_preference_fallback(
    runtime: Any,
    *,
    turn: int,
    category: str,
    dimension: str,
    write_general: bool,
    general_sentence: str,
    retention_scope: str,
    structured_rule: dict[str, Any] | None,
    preference: str,
) -> bool:
    fallback_text = (general_sentence or preference or "").strip()
    if not fallback_text:
        return False

    cache = getattr(runtime, "_session_preference_fallback_cache", None)
    if not isinstance(cache, list):
        cache = []
        try:
            setattr(runtime, "_session_preference_fallback_cache", cache)
        except Exception:
            return False

    normalized_text = " ".join(fallback_text.split())
    normalized_dimension = str(dimension or "").strip().lower()
    for existing in cache:
        if not isinstance(existing, dict):
            continue
        existing_text = " ".join(str(existing.get("text", "") or "").split())
        existing_dimension = str(existing.get("dimension", "") or "").strip().lower()
        if existing_text == normalized_text and existing_dimension == normalized_dimension:
            return False

    cache.append(
        {
            "turn": turn,
            "category": category,
            "dimension": dimension,
            "write_general": bool(write_general),
            "text": fallback_text,
            "preference": preference,
            "general_sentence": general_sentence,
            "retention_scope": retention_scope,
            "structured_rule": deepcopy(structured_rule) if isinstance(structured_rule, dict) else None,
        }
    )
    return True


def get_session_preference_fallback_prompt(runtime: Any) -> str:
    cache = getattr(runtime, "_session_preference_fallback_cache", None)
    if not isinstance(cache, list) or not cache:
        return ""

    general_lines: list[str] = []
    preference_lines: list[str] = []
    rule_lines: list[str] = []
    seen_general: set[str] = set()
    seen_preferences: set[str] = set()
    seen_rule_ids: set[str] = set()

    for item in cache:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "") or "").split()).strip()
        dimension = str(item.get("dimension", "") or "").strip()
        is_general = bool(item.get("write_general")) or dimension.lower() == "general"
        if text and is_general and text not in seen_general:
            seen_general.add(text)
            general_lines.append(f"- {text}")
        elif text and not is_general:
            preference_key = f"{dimension.lower()}::{text}"
            if preference_key not in seen_preferences:
                seen_preferences.add(preference_key)
                preference_lines.append(f"- [{dimension or 'general'}] {text}")

        spec = item.get("structured_rule")
        if not isinstance(spec, dict):
            continue
        spec = deepcopy(spec)
        spec.setdefault("schema_version", "wm_rule_v1")
        spec.setdefault("dimension", str(item.get("dimension", "") or "general"))
        spec.setdefault("scope", "global")
        spec.setdefault("content", text)
        spec.setdefault("normalized_sentence", text)
        spec.setdefault(
            "retention_scope",
            str(item.get("retention_scope", "") or "").strip() or "job_local",
        )
        rule_id = str(spec.get("rule_id", "") or "").strip()
        if not rule_id:
            rule_id = f"wmr_session_fallback_turn_{int(item.get('turn') or 0):03d}_{slugify_session_rule(text)}"
        if rule_id in seen_rule_ids:
            continue
        seen_rule_ids.add(rule_id)
        spec["rule_id"] = rule_id
        rule_lines.append(json.dumps(spec, ensure_ascii=False))

    parts: list[str] = []
    if general_lines:
        parts.append(
            "\n".join(
                [
                    '<working_memory_general_rules priority="highest" source="session_preference_fallback" note="当前会话全局偏好/约束；后续轮次与新增页应优先遵守">',
                    *general_lines,
                    "</working_memory_general_rules>",
                ]
            )
        )
    if preference_lines:
        parts.append(
            "\n".join(
                [
                    '<working_memory_preferences source="session_preference_fallback" note="当前会话偏好补充">',
                    *preference_lines,
                    "</working_memory_preferences>",
                ]
            )
        )
    if rule_lines:
        parts.append(
            "\n".join(
                [
                    '<working_memory_rule_specs format="jsonl" priority="highest" source="session_preference_fallback" note="当前会话结构化全局偏好/约束（可机读子集）">',
                    *rule_lines,
                    "</working_memory_rule_specs>",
                ]
            )
        )
    return "\n\n".join(parts)


async def append_session_preference(
    runtime: Any,
    user_message: str,
    agent_response: str,
    append_to_history: bool = True,
    intent: dict[str, Any] | None = None,
) -> bool:
    try:
        modify_agent = getattr(runtime, "modifyagent", None)
        llm = getattr(modify_agent, "llm", None) if modify_agent is not None else None
        if (not llm or not hasattr(llm, "run")):
            design_agent = getattr(runtime, "designagent", None)
            llm = getattr(design_agent, "llm", None) if design_agent is not None else None
        if not llm or (not hasattr(llm, "run") and not callable(llm)):
            semantic_compilation = semantic_compilation_fail_closed(
                runtime,
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                reason="preference semantic compiler LLM unavailable",
                source="llm_unavailable",
            )
            setattr(runtime, "_last_preference_semantic_compilation", semantic_compilation)
            return False

        existing_rules_text = ""
        orchestrator_for_rules = getattr(runtime, "_memory_orchestrator_instance", None)
        try:
            existing_rules_text = format_rule_specs_text(
                collect_active_structured_wm_rule_specs(runtime, orchestrator_for_rules),
                source="session_preference_existing_rules",
                note="Existing structured WM rules visible to the semantic compiler.",
            )
        except Exception:
            logger.debug("Failed to collect merged structured rules for semantic compiler", exc_info=True)
            existing_rules_text = ""
        if (
            not existing_rules_text
            and orchestrator_for_rules is not None
            and hasattr(orchestrator_for_rules, "get_wm_structured_rules_text")
        ):
            try:
                existing_rules_text = orchestrator_for_rules.get_wm_structured_rules_text()
            except Exception:
                existing_rules_text = ""

        semantic_compilation = await compile_preference_semantics_with_llm(
            runtime,
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            existing_rules_text=existing_rules_text,
            llm=llm,
        )
        setattr(runtime, "_last_preference_semantic_compilation", semantic_compilation)
        if semantic_compilation.fail_closed or not semantic_compilation.valid:
            logger.warning(
                "Session preference not written: semantic compiler did not approve memory write "
                "(flow=%s, reason=%s)",
                semantic_compilation.memory_flow,
                semantic_compilation.reason,
            )
            append_memory_flow_trace(
                runtime,
                event="preference_write_failed",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                extra={
                    "memory_flow": semantic_compilation.memory_flow,
                    "reason": semantic_compilation.reason,
                    "source": semantic_compilation.source,
                    "valid": semantic_compilation.valid,
                    "fail_closed": semantic_compilation.fail_closed,
                },
            )
            return False

        turn = runtime._modify_turn_count
        normalized_payloads = [
            payload for payload in (semantic_compilation.payloads or [semantic_compilation.payload])
            if isinstance(payload, dict)
            and (payload.get("preference") or payload.get("general_sentence"))
        ]

        if not normalized_payloads:
            return False

        memory_orchestrator = await _ensure_session_preference_orchestrator(
            runtime,
            user_message=user_message,
        )
        wrote_to_wm = False
        if memory_orchestrator is not None:
            def _has_active_pref(target_content: str, target_dimension: str) -> bool:
                normalized_content = " ".join(str(target_content or "").split()).strip()
                normalized_dimension = str(target_dimension or "").strip().lower()
                wm = _get_orchestrator_wm(memory_orchestrator)
                if not normalized_content or wm is None:
                    return False
                for existing in getattr(wm, "_temp_preferences", []):
                    if getattr(existing, "superseded", False):
                        continue
                    existing_content = " ".join(
                        str(getattr(existing, "content", "") or "").split()
                    ).strip()
                    existing_dimension = str(getattr(existing, "dimension", "") or "").strip().lower()
                    if existing_content != normalized_content:
                        continue
                    if existing_dimension == normalized_dimension:
                        return True
                return False

            def _active_pref_count() -> int:
                wm = _get_orchestrator_wm(memory_orchestrator)
                if wm is None:
                    return 0
                return sum(
                    1
                    for existing in getattr(wm, "_temp_preferences", [])
                    if not getattr(existing, "superseded", False)
                )

            try:
                for normalized_payload in normalized_payloads:
                    preference = str(normalized_payload.get("preference", "") or "").strip()
                    resolved_dimension = str(normalized_payload.get("dimension", "") or "").strip()
                    write_general = bool(normalized_payload.get("write_general"))
                    general_sentence = str(normalized_payload.get("general_sentence", "") or "").strip()
                    retention_scope = str(normalized_payload.get("retention_scope", "") or "job_local").strip()
                    structured_rule = normalized_payload.get("structured_rule")

                    specific_structured_data = {
                        "source": "session_preference",
                        "retention_scope": retention_scope,
                        "semantic_compiler": {
                            "memory_flow": semantic_compilation.memory_flow,
                            "source": semantic_compilation.source,
                            "reason": semantic_compilation.reason,
                            "trace_artifact": semantic_compilation.trace_artifact,
                            "compiler_payload_artifact": semantic_compilation.compiler_payload_artifact,
                            "critic_payload_artifact": semantic_compilation.critic_payload_artifact,
                        },
                    }
                    if general_sentence:
                        specific_structured_data["linked_general_sentence"] = general_sentence
                    if isinstance(structured_rule, dict) and structured_rule.get("rule_id"):
                        specific_structured_data["linked_rule_id"] = structured_rule.get("rule_id")
                    if str(normalized_payload.get("note_kind") or "") == "session_dimension_note":
                        specific_structured_data["note_kind"] = "session_dimension_note"

                    general_structured_data = None
                    if write_general and general_sentence:
                        general_structured_data = deepcopy(structured_rule) if isinstance(structured_rule, dict) else {}
                        general_structured_data["source"] = "session_preference"
                        general_structured_data["retention_scope"] = retention_scope
                        general_structured_data["normalized_sentence"] = general_sentence
                        general_structured_data["semantic_compiler_artifacts"] = {
                            "memory_flow": semantic_compilation.memory_flow,
                            "trace_artifact": semantic_compilation.trace_artifact,
                            "compiler_payload_artifact": semantic_compilation.compiler_payload_artifact,
                            "critic_payload_artifact": semantic_compilation.critic_payload_artifact,
                        }

                    specific_dimension = str(resolved_dimension or "").strip()
                    payload_wrote = False
                    if preference and specific_dimension.lower() != "general":
                        if specific_dimension and _has_active_pref(preference, specific_dimension):
                            logger.info(
                                "Session preference deduplicated (dim=%s): %s",
                                specific_dimension,
                                preference[:80],
                            )
                            wrote_to_wm = True
                            payload_wrote = True
                        else:
                            before_count = _active_pref_count()
                            await memory_orchestrator.on_user_preference(
                                preference,
                                dimension=specific_dimension,
                                scope="global",
                                structured_data=specific_structured_data,
                            )
                            if (
                                _has_active_pref(preference, specific_dimension)
                                or _active_pref_count() > before_count
                            ):
                                wrote_to_wm = True
                                payload_wrote = True

                    if write_general and general_sentence:
                        if _has_active_pref(general_sentence, "general"):
                            logger.info(
                                "Session general rule deduplicated (scope=%s): %s",
                                retention_scope,
                                general_sentence[:80],
                            )
                            wrote_to_wm = True
                            payload_wrote = True
                        else:
                            before_count = _active_pref_count()
                            await memory_orchestrator.on_user_preference(
                                general_sentence,
                                dimension="general",
                                scope="global",
                                structured_data=general_structured_data,
                            )
                            if (
                                _has_active_pref(general_sentence, "general")
                                or _active_pref_count() > before_count
                            ):
                                wrote_to_wm = True
                                payload_wrote = True

                    if preference and not payload_wrote and specific_dimension.lower() == "general":
                        if _has_active_pref(preference, "general"):
                            logger.info(
                                "Session preference deduplicated (general): %s",
                                preference[:80],
                            )
                            wrote_to_wm = True
                        else:
                            before_count = _active_pref_count()
                            await memory_orchestrator.on_user_preference(
                                preference,
                                dimension="general",
                                scope="global",
                                structured_data=specific_structured_data,
                            )
                            if (
                                _has_active_pref(preference, "general")
                                or _active_pref_count() > before_count
                            ):
                                wrote_to_wm = True
            except Exception as mem_e:
                logger.warning(
                    "Session preference WM write failed (non-fatal): %s", mem_e,
                )

        fallback_cached = False
        if not wrote_to_wm:
            logger.warning(
                "Session preference was approved by semantic compiler but was not written to WM; "
                "not caching fallback to avoid unverified propagation.",
            )
            append_memory_flow_trace(
                runtime,
                event="preference_write_failed",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                extra={
                    "memory_flow": semantic_compilation.memory_flow,
                    "reason": "approved_by_semantic_compiler_but_not_written_to_wm",
                    "rule_ids": [
                        payload.get("structured_rule", {}).get("rule_id", "")
                        for payload in normalized_payloads
                        if isinstance(payload.get("structured_rule"), dict)
                    ],
                },
            )

        if append_to_history and modify_agent is not None:
            from memslides.utils.typings import ChatMessage, Role

            for normalized_payload in normalized_payloads:
                pref_msg = ChatMessage(
                    role=Role.SYSTEM,
                    content=_format_session_preference_message(
                        turn=turn,
                        category=str(normalized_payload.get("category", "") or "general"),
                        dimension=str(normalized_payload.get("dimension", "") or ""),
                        write_general=bool(normalized_payload.get("write_general")),
                        retention_scope=str(normalized_payload.get("retention_scope", "") or "job_local"),
                        preference=str(normalized_payload.get("preference", "") or ""),
                        general_sentence=str(normalized_payload.get("general_sentence", "") or ""),
                    ),
                )
                modify_agent.chat_history.append(pref_msg)
        first_payload = normalized_payloads[0]
        logger.info(
            "Session preference extracted (turn=%s, updates=%s, first_cat=%s, first_dim=%s, append_to_history=%s, wrote_to_wm=%s, fallback_cached=%s): %s",
            turn,
            len(normalized_payloads),
            first_payload.get("category", "general"),
            first_payload.get("dimension") or "auto",
            append_to_history,
            wrote_to_wm,
            fallback_cached,
            (str(first_payload.get("general_sentence") or first_payload.get("preference") or ""))[:80],
        )
        for normalized_payload in normalized_payloads:
            structured_rule = normalized_payload.get("structured_rule")
            if not isinstance(structured_rule, dict):
                continue
            target = structured_rule.get("target") if isinstance(structured_rule.get("target"), dict) else {}
            propagation = (
                structured_rule.get("propagation")
                if isinstance(structured_rule.get("propagation"), dict)
                else {}
            )
            logger.info(
                "Session preference rule spec: rule_id=%s scope=%s slide_scope=%s element=%s apply_existing=%s apply_future=%s",
                structured_rule.get("rule_id", ""),
                structured_rule.get("scope", ""),
                target.get("slide_scope", ""),
                target.get("element_kind", ""),
                propagation.get("apply_existing_slides", ""),
                propagation.get("apply_future_slides", ""),
            )
        if wrote_to_wm:
            persist_live_wm_snapshot(runtime, reason="session_preference")
            append_memory_flow_trace(
                runtime,
                event="preference_write_succeeded",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                extra={
                    "memory_flow": semantic_compilation.memory_flow,
                    "update_count": len(normalized_payloads),
                    "categories": [str(payload.get("category", "") or "") for payload in normalized_payloads],
                    "dimensions": [str(payload.get("dimension", "") or "") for payload in normalized_payloads],
                    "retention_scopes": [str(payload.get("retention_scope", "") or "") for payload in normalized_payloads],
                    "rule_ids": [
                        payload.get("structured_rule", {}).get("rule_id", "")
                        for payload in normalized_payloads
                        if isinstance(payload.get("structured_rule"), dict)
                    ],
                },
            )

        return wrote_to_wm or fallback_cached
    except Exception as e:
        logger.warning("Session preference extraction failed (non-fatal): %s", e)
        append_memory_flow_trace(
            runtime,
            event="preference_write_failed",
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            extra={
                "stage": "append_session_preference",
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        return False


def dump_current_round_wm_snapshot(runtime: Any) -> bool:
    live_dumped = persist_live_wm_snapshot(runtime, reason="current_round")
    try:
        orchestrator = getattr(runtime, "_memory_orchestrator_instance", None)
        if orchestrator is None:
            return live_dumped
        job_mgr = getattr(orchestrator, "_job_mgr", None)
        if job_mgr is None:
            return live_dumped
        wm = getattr(job_mgr, "working_memory", None)
        if wm is None:
            return live_dumped

        dumper = getattr(runtime.memory_system, "artifact_dumper", None) if runtime.memory_system else None
        if dumper is None:
            return live_dumped

        round_index = job_mgr.round_count()
        if round_index <= 0:
            return live_dumped

        dumper.dump_wm_snapshot(
            round_index,
            chain_buffer=wm.chain_buffer,
            temp_preferences=wm._temp_preferences,
            temp_experiences=wm.get_experiences(),
            temp_episodes=wm.get_episodes(),
        )
        return True
    except Exception as e:
        logger.debug("Post-review wm_snapshot dump failed (non-fatal): %s", e)
        return False


def dump_manual_wm_snapshot(runtime: Any) -> bool:
    """Persist edited WM even when no normal round has completed yet."""
    live_dumped = persist_live_wm_snapshot(runtime, reason="manual_edit")
    try:
        orchestrator = getattr(runtime, "_memory_orchestrator_instance", None)
        if orchestrator is None:
            return live_dumped
        job_mgr = getattr(orchestrator, "_job_mgr", None)
        if job_mgr is None:
            return live_dumped
        wm = getattr(job_mgr, "working_memory", None)
        if wm is None:
            return live_dumped
        dumper = getattr(runtime.memory_system, "artifact_dumper", None) if runtime.memory_system else None
        if dumper is None:
            return live_dumped
        rounds_root = Path(runtime.workspace) / ".memory" / "rounds"
        existing_indices: list[int] = []
        if rounds_root.exists():
            for item in rounds_root.glob("round_*"):
                match = re.match(r"round_(\d+)$", item.name)
                if match:
                    existing_indices.append(int(match.group(1)))
        round_count = int(job_mgr.round_count() or 0)
        round_index = max(existing_indices or [0], default=0)
        if round_count > 0:
            round_index = max(round_index, round_count)
        else:
            round_index = max(round_index, 1)
        dumper.dump_wm_snapshot(
            round_index,
            chain_buffer=wm.chain_buffer,
            temp_preferences=wm._temp_preferences,
            temp_experiences=wm.get_experiences(),
            temp_episodes=wm.get_episodes(),
        )
        return True
    except Exception as e:
        logger.debug("Manual wm_snapshot dump failed (non-fatal): %s", e)
        return False


async def record_review_feedback_preference(runtime: Any, feedback: str) -> bool:
    feedback_text = str(feedback or "").strip()
    if not feedback_text:
        return False

    wrote = await append_session_preference(
        runtime,
        feedback_text,
        "",
        append_to_history=False,
    )
    if not wrote:
        return False

    runtime._skip_next_session_preference_priming = True
    dumped = dump_current_round_wm_snapshot(runtime)
    logger.info(
        "Post-review session preference recorded (feedback_len=%s, snapshot_dumped=%s).",
        len(feedback_text),
        dumped,
    )
    return True


def ensure_modifyagent_loaded(runtime: Any, *, load_reason: str) -> RevisionEditor | None:
    if runtime.modifyagent is not None:
        if isinstance(runtime.modifyagent, RevisionEditor):
            return runtime.modifyagent
        raise RuntimeError(
            "RevisionEditor initialization failed because the cached modify agent "
            f"is not a RevisionEditor instance during {load_reason}: "
            f"{type(runtime.modifyagent).__name__}"
        )

    if runtime.agent_env is None:
        logger.warning(
            "RevisionEditor is unavailable for %s because the agent environment is not initialized.",
            load_reason,
        )
        return None

    try:
        logger.info("Loading RevisionEditor for %s...", load_reason)
        runtime.modifyagent = RevisionEditor(
            runtime.config,
            runtime.agent_env,
            runtime.workspace,
            runtime.language,
        )
        logger.info("RevisionEditor loaded with %s tools", len(runtime.modifyagent.tools))
    except RoleToolContractError as e:
        logger.error("RevisionEditor protocol mismatch during %s: %s", load_reason, e)
        raise RuntimeError(
            "RevisionEditor initialization failed because the runtime tool schema "
            "does not match the required existing-slide patch protocol. "
            f"{e}"
        ) from e
    except Exception as e:
        logger.warning("Failed to load RevisionEditor for %s: %s", load_reason, e)
        raise RuntimeError(
            f"RevisionEditor initialization failed during {load_reason}: {e}"
        ) from e

    return runtime.modifyagent


def resolve_export_repair_agent(runtime: Any, preferred_agent: Any | None = None) -> RevisionEditor | None:
    if isinstance(preferred_agent, RevisionEditor):
        return preferred_agent
    return ensure_modifyagent_loaded(runtime, load_reason="export repair")


def extract_export_failure_slide(error_text: str) -> Path | None:
    match = re.search(r"(/\S+\.html)", error_text)
    if not match:
        return None
    try:
        return Path(match.group(1)).resolve()
    except Exception:
        return Path(match.group(1))


def _export_failure_kind(error_text: str) -> str:
    text = " ".join(str(error_text or "").lower().split())
    if "overflows body" in text or "clipped outside" in text:
        return "overflow"
    if "too close to bottom edge" in text:
        return "bottom_safe_zone"
    if "don't match presentation layout" in text or "expected body css" in text:
        return "canvas_dimensions"
    if "image not found" in text or "background image not found" in text:
        return "missing_image"
    return text[:120] or "export_failure"


def _export_failure_signature(error_text: str, failing_slide: Path | None) -> str:
    slide_key = str(failing_slide or extract_export_failure_slide(error_text) or "unknown")
    return f"{slide_key}:{_export_failure_kind(error_text)}"


def _format_export_diagnostics(diagnostics: Any, *, limit: int = 4) -> str:
    if not isinstance(diagnostics, list) or not diagnostics:
        return ""
    lines: list[str] = []
    for item in diagnostics[:limit]:
        if not isinstance(item, dict):
            continue
        parts = [
            str(item.get("code") or item.get("severity") or "diagnostic"),
            str(item.get("message") or "").strip(),
        ]
        selector = str(item.get("target_selector_hint") or "").strip()
        dom_path = str(item.get("target_dom_path") or "").strip()
        repair_paths = item.get("repair_dom_paths")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if selector:
            parts.append(f"selector={selector}")
        if dom_path:
            parts.append(f"dom_path={dom_path}")
        if isinstance(repair_paths, list) and repair_paths:
            parts.append(f"repair_dom_paths={', '.join(str(path) for path in repair_paths[:3])}")
        if isinstance(metadata, dict):
            overflow = metadata.get("overflow")
            if overflow:
                parts.append(f"overflow={overflow}")
            text_preview = metadata.get("text_preview") or item.get("text_preview")
            if text_preview:
                parts.append(f"text={str(text_preview)[:120]}")
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines)


def build_export_repair_message(
    runtime: Any,
    *,
    error_text: str,
    context_label: str,
    failing_slide: Path | None,
    slide_dir: Path | str | None,
    diagnostics: Any = None,
    failure_count: int = 1,
) -> str:
    cleaned_error = " ".join(str(error_text or "").split())
    if len(cleaned_error) > 1800:
        cleaned_error = cleaned_error[:1797] + "..."
    slide_label = workspace_relative_label(runtime, failing_slide) if failing_slide is not None else "unknown"
    workspace_context = build_workspace_context_block(
        runtime,
        slide_dir=slide_dir,
        include_active_slide_dir=True,
    )
    force_rewrite = failure_count >= 2
    repair_strategy = (
        "This same slide/export error has repeated. Do not attempt another tiny patch. "
        "Rewrite the complete failing slide with `write_html_file`, preserving the deck style but simplifying layout density, "
        "then inspect the rewritten slide before `finalize`."
        if force_rewrite
        else "First attempt may use `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch`; if layout targets are broad or ambiguous, rewrite the failing slide."
    )
    message = (
        f"SYSTEM: PPT export validation failed during {context_label}. "
        "Use the RevisionEditor-stage local repair protocol on the current on-disk HTML: "
        "inspect the failing slide and relevant current HTML first, run `plan_slide_patch` when available, "
        "prioritize `offending_targets`, `repair_context`, `recommended_patch_strategy`, `repair_candidates`, `rules`, and target selectors before patching, "
        "never call `apply_slide_patch` with empty or invalid `patch_ops`, reserve "
        "`write_html_file` for controlled regenerate only, run `inspect_slide` on "
        "every changed slide, and only call `finalize` after the current on-disk HTML "
        "passes inspection. `inspect_slide` is diagnostic only and never rewrites the "
        "disk HTML.\n"
        f"Failing slide: {slide_label}\n"
        f"Failure count for this slide/error kind: {failure_count}\n"
        f"Repair strategy: {repair_strategy}\n"
        f"Export error: {cleaned_error}\n"
        "Do not restate the plan in prose. Fix the slide now."
    )
    diagnostic_text = _format_export_diagnostics(diagnostics)
    if diagnostic_text:
        message += f"\nStructured pptx_export diagnostics:\n{diagnostic_text}"
    if workspace_context:
        message += f"\n{workspace_context}"
    return message


async def run_slide_export_repair(
    runtime: Any,
    *,
    current_slide_dir: Path | str,
    error: Exception | str,
    context_label: str,
    preferred_agent: Any | None = None,
    max_turns: int = 3,
    failure_count: int = 1,
    diagnostics: Any = None,
) -> Path | None:
    repair_agent = resolve_export_repair_agent(runtime, preferred_agent)
    if repair_agent is None:
        return None

    from memslides.utils.typings import ChatMessage, Role

    repaired_dir = resolve_exportable_slide_dir(runtime, current_slide_dir)
    if repaired_dir is None:
        repaired_dir = Path(current_slide_dir)
        if not repaired_dir.is_absolute():
            repaired_dir = runtime.workspace / repaired_dir
        repaired_dir = repaired_dir.resolve()

    error_text = str(error)
    failing_slide = extract_export_failure_slide(error_text)
    diagnostics = diagnostics if diagnostics is not None else getattr(error, "pptx_export_diagnostics", None)
    repair_agent.chat_history.append(
        ChatMessage(
            role=Role.USER,
            content=build_export_repair_message(
                runtime,
                error_text=error_text,
                context_label=context_label,
                failing_slide=failing_slide,
                slide_dir=repaired_dir,
                diagnostics=diagnostics,
                failure_count=failure_count,
            ),
        )
    )
    set_current_agent(
        repair_agent.name,
        workspace=runtime.workspace,
        model_ref=getattr(repair_agent, "model_ref", "modify_agent"),
    )

    try:
        for repair_turn in range(max_turns):
            agent_message = await repair_agent.action()
            if not agent_message.tool_calls:
                logger.warning(
                    "%s export-repair turn %s returned no tool calls; stopping repair loop.",
                    repair_agent.name,
                    repair_turn + 1,
                )
                break

            outcome = await repair_agent.execute(agent_message.tool_calls)

            candidate_dir = repaired_dir
            if isinstance(outcome, str):
                candidate_dir = Path(outcome)
                if not candidate_dir.is_absolute():
                    candidate_dir = runtime.workspace / candidate_dir
                candidate_dir = resolve_exportable_slide_dir(runtime, candidate_dir) or candidate_dir.resolve()
            elif repaired_dir is not None:
                candidate_dir = resolve_exportable_slide_dir(runtime, repaired_dir) or repaired_dir

            if candidate_dir is not None:
                repaired_dir = candidate_dir
                runtime.intermediate_output["slide_html_dir"] = str(repaired_dir)
                runtime.save_results()

            if isinstance(outcome, str):
                return repaired_dir
    finally:
        repair_agent.save_history()
        runtime.save_results()

    return repaired_dir


async def export_slides_with_agent_repair(
    runtime: Any,
    slide_html_dir: Path | str,
    output_pptx: Path | str,
    *,
    aspect_ratio: str,
    context_label: str,
    preferred_agent: Any | None = None,
    max_repair_rounds: int = 2,
    max_agent_turns: int = 3,
) -> tuple[Path, list[Path]]:
    try:
        max_repair_rounds = max(0, int(os.getenv("MEMSLIDES_EXPORT_STRICT_REPAIR_ROUNDS", str(max_repair_rounds)) or max_repair_rounds))
    except ValueError:
        max_repair_rounds = max(0, max_repair_rounds)
    allow_relaxed_fallback = str(os.getenv("MEMSLIDES_EXPORT_ALLOW_RELAXED_FALLBACK", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    failure_counts: dict[str, int] = {}
    resolved_slide_dir = resolve_exportable_slide_dir(runtime, slide_html_dir)
    if resolved_slide_dir is None:
        searched_dirs = ", ".join(
            str(path) for path in candidate_slide_dirs(runtime, slide_html_dir)
        )
        raise RuntimeError(
            "DeckDesigner agent did not produce exportable slides. "
            f"Searched: {searched_dirs}"
        )

    repair_agent = preferred_agent if isinstance(preferred_agent, RevisionEditor) else None
    last_error: Exception | None = None

    for repair_round in range(max_repair_rounds + 1):
        resolved_slide_dir = resolve_exportable_slide_dir(runtime, resolved_slide_dir) or resolved_slide_dir
        runtime.intermediate_output["slide_html_dir"] = str(resolved_slide_dir)
        runtime.save_results()

        export_html_files = resolve_exportable_slide_paths(runtime, resolved_slide_dir)
        if not export_html_files:
            raise RuntimeError(
                f"No exportable slides found in {resolved_slide_dir}; expected slide_XX.html files."
            )

        try:
            exp_writer = runtime._make_exp_writer()
            await convert_html_to_pptx_with_retry(
                export_html_files,
                output_pptx,
                aspect_ratio=aspect_ratio,
                session_id=str(runtime.workspace.stem),
                experience_writer=exp_writer,
                allow_skip_layout_validation_fallback=False,
            )
            runtime.intermediate_output["export_status"] = "strict"
            runtime.intermediate_output["export_warnings"] = []
            runtime.save_results()
            return resolved_slide_dir, export_html_files
        except Exception as e:
            last_error = e
            if repair_round >= max_repair_rounds:
                break

            logger.warning(
                "PPT export failed during %s (repair round %s/%s): %s",
                context_label,
                repair_round + 1,
                max_repair_rounds,
                e,
            )
            error_text = str(e)
            failing_slide = extract_export_failure_slide(error_text)
            signature = _export_failure_signature(error_text, failing_slide)
            failure_counts[signature] = failure_counts.get(signature, 0) + 1
            diagnostics = getattr(e, "pptx_export_diagnostics", None)
            repaired_dir = await run_slide_export_repair(
                runtime,
                current_slide_dir=resolved_slide_dir,
                error=e,
                context_label=context_label,
                preferred_agent=repair_agent,
                max_turns=max_agent_turns,
                failure_count=failure_counts[signature],
                diagnostics=diagnostics,
            )
            if isinstance(runtime.modifyagent, RevisionEditor):
                repair_agent = runtime.modifyagent
            if repaired_dir is None:
                break
            resolved_slide_dir = repaired_dir

    export_html_files = resolve_exportable_slide_paths(runtime, resolved_slide_dir)
    if allow_relaxed_fallback and export_html_files:
        warning_text = f"PPTX strict export failed during {context_label}: {last_error}"
        try:
            exp_writer = runtime._make_exp_writer()
            await convert_html_to_pptx_with_retry(
                export_html_files,
                output_pptx,
                aspect_ratio=aspect_ratio,
                session_id=str(runtime.workspace.stem),
                experience_writer=exp_writer,
                allow_skip_layout_validation_fallback=True,
            )
            warning_text += " Relaxed PPTX fallback was used."
        except Exception as relaxed_error:
            warning_text += f" Relaxed PPTX fallback also failed: {relaxed_error}. HTML/PDF fallback remains available."
            logger.warning(warning_text)
        warnings = list(runtime.intermediate_output.get("export_warnings") or [])
        if warning_text not in warnings:
            warnings.append(warning_text)
        runtime.intermediate_output["export_status"] = "partial"
        runtime.intermediate_output["export_warnings"] = warnings
        runtime.save_results()
        return resolved_slide_dir, export_html_files

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"PPT export failed during {context_label}")
