"""ProfileInjectionRouter — 用户画像维度级智能注入路由

根据用户第一轮 prompt 判断每个维度与历史画像的关系：
- inject:   不冲突、不相关 → 原样注入 LTM 值到 WM
- override: 冲突 → 提炼用户 prompt 中的新偏好，覆盖该维度写入 WM
- skip:     该维度 LTM 无数据且用户 prompt 也没提及 → 不写入

结果统一写入 WorkingMemory，供 Research/Design/Modify 全程使用。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from ..core.models import UserProfile, TempPreference
from ..extract.llm_compat import extract_message_text, resolve_llm_retry_times

logger = logging.getLogger(__name__)

# LTM 注入来源标记，Consolidator 归档时据此过滤，避免循环写回
LTM_INJECT_SOURCE = "ltm_profile_inject"
PROFILE_OVERRIDE_SOURCE = "profile_override"
PROFILE_ROUTE_MAX_TOKENS = 512


class ProfileInjectionRoutingError(RuntimeError):
    """Raised when profile injection routing cannot produce trustworthy decisions."""

PROFILE_ROUTE_PROMPT = """你在做一个5维路由任务。目标：比较“当前用户指令”与“历史画像摘要”，为每个维度输出 inject / enrich / override / skip。

上下文:
- core_persona: {core_persona}
- current_task_intent: {task_intent}
- memory_read_intent: {target_intent}
- memory_write_intent: {write_intent}

维度:
- theme: 配色/字体/背景/整体主题
- visual: 图片图表图形风格
- layout: 版式结构/密度/留白/节奏
- content: 标题/文案/语言/信息组织
- general: 页数/来源约束/全局执行要求

动作规则:
- inject = 历史有，用户没提
- skip = 历史无，用户没提
- enrich = 历史有，用户提的是兼容补充
- override = 用户明确替换/冲突；或历史无但用户提了

硬约束:
- 历史无数据时不能输出 enrich
- inject/skip 的 override_content 必须是 ""
- enrich/override 的 override_content 只提炼该维度信息
- 不要把页数、来源约束误判到 theme

用户指令:
{user_prompt}

历史画像摘要:
{per_dimension_summary}

只输出 JSON 数组:
[{{"dimension":"theme","action":"inject","reason":"...","override_content":""}}]"""

PROFILE_ROUTING_DEBUG_FILENAME = "profile_routing_llm_response_debug.json"


@dataclass
class DimensionDecision:
    """单个维度的注入决策"""
    dimension: str           # "theme" / "visual" / "layout" / "content" / "template" / "general"
    action: str              # "inject" | "enrich" | "override" | "skip"
    reason: str = ""         # 决策原因（供日志/调试）
    override_content: str = ""  # action="override" 时，从用户 prompt 提炼的新值


class ProfileInjectionRouter:
    """用户画像维度级智能注入路由器。

    在 Research Agent 启动前调用 route() + apply()，
    将画像按维度写入 WM，供全程使用。
    """

    def __init__(self, llm: Callable | None = None):
        self._llm = llm

    @staticmethod
    def allow_fallback() -> bool:
        strict = str(os.getenv("MEMSLIDES_PROFILE_ROUTER_STRICT", "")).strip().lower()
        if strict in {"1", "true", "yes", "on"}:
            return False
        configured = str(os.getenv("MEMSLIDES_PROFILE_ROUTER_ALLOW_FALLBACK", "")).strip().lower()
        if configured in {"0", "false", "no", "off"}:
            return False
        if configured in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        return True

    @staticmethod
    def _format_list(values: list[Any]) -> str:
        return ", ".join(str(value).strip() for value in values if str(value or "").strip())

    @staticmethod
    def _truncate_text(text: str, limit: int = 220) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @staticmethod
    def _compress_prompt_text(text: str, limit: int = 520) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        head = max(120, limit // 2)
        tail = max(80, limit - head - 5)
        return f"{normalized[:head]} ... {normalized[-tail:]}"

    @classmethod
    def _dimension_entries_for_injection(
        cls,
        profile: UserProfile,
        dim: str,
    ) -> list[tuple[str, str]]:
        """返回写入 WM 的原子偏好列表。

        对结构化画像优先拆成可执行的子维度条目，减少“整段画像=一条偏好”的稀释问题。
        general 维度按原子规则逐条写入；其他维度在缺少结构化字段时才退回块级文本。
        """
        if dim == "general":
            return [
                ("general", item.strip())
                for item in profile.general.preferences
                if str(item or "").strip()
            ]

        entries: list[tuple[str, str]] = []

        if dim == "theme":
            if profile.theme.primary_colors:
                entries.append((
                    "theme.primary_colors",
                    f"主色偏好: {cls._format_list(profile.theme.primary_colors)}",
                ))
            if profile.theme.accent_colors:
                entries.append((
                    "theme.accent_colors",
                    f"强调色偏好: {cls._format_list(profile.theme.accent_colors)}",
                ))
            if profile.theme.font_family:
                entries.append(("theme.font_family", f"字体偏好: {profile.theme.font_family}"))
            if profile.theme.font_size_range:
                low, high = profile.theme.font_size_range
                entries.append(("theme.font_size_range", f"字号范围偏好: {low}-{high}pt"))
            if profile.theme.background_style:
                entries.append(("theme.background_style", f"背景风格: {profile.theme.background_style}"))
            for note in profile.theme.notes:
                note_text = str(note or "").strip()
                if note_text:
                    entries.append(("theme.notes", note_text))

        elif dim == "visual":
            if profile.visual.image_style:
                entries.append(("visual.image_style", f"图片风格: {profile.visual.image_style}"))
            if profile.visual.chart_type_priority:
                entries.append((
                    "visual.chart_type_priority",
                    f"图表优先级: {cls._format_list(profile.visual.chart_type_priority)}",
                ))
            if profile.visual.icon_usage:
                entries.append(("visual.icon_usage", f"图标使用偏好: {profile.visual.icon_usage}"))
            if profile.visual.animation_preference:
                entries.append((
                    "visual.animation_preference",
                    f"动画偏好: {profile.visual.animation_preference}",
                ))
            for note in profile.visual.notes:
                note_text = str(note or "").strip()
                if note_text:
                    entries.append(("visual.notes", note_text))

        elif dim == "layout":
            if profile.layout.content_density:
                entries.append(("layout.content_density", f"内容密度: {profile.layout.content_density}"))
            if profile.layout.alignment_style:
                entries.append(("layout.alignment_style", f"对齐方式: {profile.layout.alignment_style}"))
            if profile.layout.spacing_preference:
                entries.append((
                    "layout.spacing_preference",
                    f"间距偏好: {profile.layout.spacing_preference}",
                ))
            if profile.layout.slide_structure:
                entries.append(("layout.slide_structure", f"页面结构: {profile.layout.slide_structure}"))
            for note in profile.layout.notes:
                note_text = str(note or "").strip()
                if note_text:
                    entries.append(("layout.notes", note_text))

        elif dim == "content":
            if profile.content.text_density:
                entries.append(("content.text_density", f"文本密度: {profile.content.text_density}"))
            if profile.content.language_style:
                entries.append(("content.language_style", f"语言风格: {profile.content.language_style}"))
            if profile.content.bullet_point_style:
                entries.append((
                    "content.bullet_point_style",
                    f"要点组织偏好: {profile.content.bullet_point_style}",
                ))
            if profile.content.title_length:
                entries.append(("content.title_length", f"标题长度偏好: {profile.content.title_length}"))
            for note in profile.content.notes:
                note_text = str(note or "").strip()
                if note_text:
                    entries.append(("content.notes", note_text))

        if entries:
            return entries

        dim_text = profile.to_prompt_text(dimensions={dim}, include_general=False)
        return [(dim, dim_text)] if dim_text else []

    @classmethod
    def _dimension_summary_text(
        cls,
        profile: UserProfile,
        dim: str,
    ) -> str:
        entries = cls._dimension_entries_for_injection(profile, dim)
        if not entries:
            return "(无历史数据)"
        compact = [cls._truncate_text(content, 120) for _, content in entries[:3]]
        return "\n".join(f"- {content}" for content in compact)

    @staticmethod
    def _llm_debug_payload(response: Any = None, *, text: str = "", error: str = "") -> dict[str, Any]:
        message = None
        choices = getattr(response, "choices", None)
        if choices:
            try:
                message = choices[0].message
            except Exception:
                message = None
        content = getattr(message, "content", None) if message is not None else None
        extra = getattr(message, "__pydantic_extra__", None) if message is not None else None
        payload: dict[str, Any] = {
            "error": error,
            "text_excerpt": ProfileInjectionRouter._truncate_text(text, 400),
            "has_choices": bool(choices),
            "content_type": type(content).__name__ if content is not None else "NoneType",
            "content_excerpt": ProfileInjectionRouter._truncate_text(extract_message_text(message), 400),
            "tool_calls_present": bool(getattr(message, "tool_calls", None)) if message is not None else False,
            "reasoning_content_present": bool(getattr(message, "reasoning_content", None)) if message is not None else False,
            "provider_extra_keys": sorted(list((extra or {}).keys())) if isinstance(extra, dict) else [],
        }
        if message is not None and hasattr(message, "model_dump"):
            try:
                payload["message_dump"] = message.model_dump(mode="json")
            except Exception:
                payload["message_dump"] = str(message)
        return payload

    @staticmethod
    def _build_dimension_summary_text(dim_summaries: dict[str, str]) -> str:
        ordered: list[str] = []
        for dim_name, summary in dim_summaries.items():
            ordered.append(f"{dim_name}: {summary}")
        return "\n".join(ordered)

    @staticmethod
    def _write_llm_debug_artifact(payload: dict[str, Any]) -> None:
        output_path = str(os.getenv("MEMSLIDES_PROFILE_ROUTING_DEBUG_PATH", "") or "").strip()
        if not output_path:
            return
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Failed to write profile routing LLM debug artifact", exc_info=True)

    @staticmethod
    def _supersede_active_dimension_prefix(wm: Any, dim: str) -> None:
        """覆盖整维偏好时，废弃该顶层维度下所有当前生效条目。"""
        for existing in getattr(wm, "_temp_preferences", []):
            if existing.superseded:
                continue
            dim_prefix = existing.dimension.split(".")[0] if existing.dimension else ""
            if dim_prefix == dim or existing.dimension == dim:
                existing.superseded = True

    @staticmethod
    def _append_reason(reason: str, suffix: str) -> str:
        """在保留原 reason 的基础上补充归一化说明。"""
        reason = reason.strip()
        return f"{reason}; {suffix}" if reason else suffix

    @classmethod
    def _normalize_decision(
        cls,
        *,
        dim: str,
        action: str,
        reason: str,
        override_content: str,
        has_history: bool,
    ) -> DimensionDecision:
        """归一化 LLM 决策，减少空画像场景下的语义漂移。"""
        normalized_action = action.strip()
        normalized_reason = reason.strip()
        normalized_content = override_content.strip()

        if normalized_action in {"inject", "skip"}:
            normalized_content = ""

        if not has_history:
            if normalized_action == "inject":
                normalized_action = "skip"
                normalized_reason = cls._append_reason(
                    normalized_reason,
                    "历史无数据，inject 归一化为 skip",
                )
            elif normalized_action == "enrich":
                normalized_action = "override"
                normalized_reason = cls._append_reason(
                    normalized_reason,
                    "历史无数据，enrich 归一化为 override",
                )

        if normalized_action == "enrich" and not normalized_content:
            if has_history:
                normalized_action = "inject"
                normalized_reason = cls._append_reason(
                    normalized_reason,
                    "未提炼出补充内容，enrich 归一化为 inject",
                )
            else:
                normalized_action = "skip"
                normalized_reason = cls._append_reason(
                    normalized_reason,
                    "历史无数据且未提炼出补充内容，enrich 归一化为 skip",
                )

        if normalized_action == "override" and not normalized_content:
            normalized_action = "skip"
            normalized_reason = cls._append_reason(
                normalized_reason,
                "未提炼出新偏好内容，override 归一化为 skip",
            )

        if normalized_action in {"inject", "skip"}:
            normalized_content = ""

        return DimensionDecision(
            dimension=dim,
            action=normalized_action,
            reason=normalized_reason,
            override_content=normalized_content,
        )

    async def route(
        self,
        user_prompt: str,
        profile: UserProfile,
        target_intent: str = "",
        task_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
    ) -> list[DimensionDecision]:
        """对 profile 的每个维度判断 inject / override / skip。

        Args:
            user_prompt: 用户第一轮指令
            profile: 从 LTM 加载的 UserProfile
            target_intent: 当前目标画像所属 intent

        Returns:
            每个维度的决策列表
        """
        # Template reuse is handled by template_usage_history + Stage 14 selector.
        # Do not inject template profile text as a generic SYSTEM preference.
        dim_map = {
            dim_name: sub_pref
            for dim_name, sub_pref in profile.get_dimension_map().items()
            if dim_name != "template"
        }

        # 构建每个维度的摘要文本
        dim_summaries: dict[str, str] = {}
        dim_has_data: dict[str, bool] = {}
        for dim_name, sub_pref in dim_map.items():
            summary_text = self._dimension_summary_text(profile, dim_name)
            dim_summaries[dim_name] = summary_text
            dim_has_data[dim_name] = summary_text != "(无历史数据)"

        if not any(dim_has_data.values()):
            return [
                DimensionDecision(
                    dimension=dim_name,
                    action="skip",
                    reason="历史画像为空，无需调用 LLM 路由",
                )
                for dim_name in dim_summaries
            ]

        # LLM 路由
        if self._llm:
            try:
                decisions = await self._llm_route(
                    user_prompt,
                    dim_summaries,
                    dim_has_data,
                    target_intent=target_intent,
                    task_intent=task_intent,
                    write_intent=write_intent,
                    core_persona=core_persona,
                )
                if decisions:
                    return decisions
            except Exception as e:
                if self.allow_fallback():
                    logger.warning("ProfileInjectionRouter LLM route failed, fallback enabled: %s", e)
                    return self._fallback_all_inject(dim_has_data, reason=f"LLM 路由失败，显式 fallback: {e}")
                raise ProfileInjectionRoutingError(f"ProfileInjectionRouter LLM route failed: {e}") from e

        if self.allow_fallback():
            return self._fallback_all_inject(dim_has_data, reason="无 LLM，显式 fallback")
        raise ProfileInjectionRoutingError("ProfileInjectionRouter requires an LLM or MEMSLIDES_PROFILE_ROUTER_ALLOW_FALLBACK=1")

    async def _llm_route(
        self,
        user_prompt: str,
        dim_summaries: dict[str, str],
        dim_has_data: dict[str, bool],
        *,
        target_intent: str = "",
        task_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
    ) -> list[DimensionDecision] | None:
        """LLM 分类路由。"""
        dim_order = list(dim_summaries.keys())
        summary_text = self._build_dimension_summary_text(dim_summaries)
        compact_prompt = self._compress_prompt_text(user_prompt, 520)

        prompt = PROFILE_ROUTE_PROMPT.format(
            target_intent=target_intent or "default",
            task_intent=task_intent or target_intent or "default",
            write_intent=write_intent or task_intent or target_intent or "default",
            core_persona=core_persona or "unspecified",
            user_prompt=compact_prompt,
            per_dimension_summary=summary_text,
        )

        request_kwargs = {
            "max_tokens": PROFILE_ROUTE_MAX_TOKENS,
            "temperature": 0.0,
            "reasoning_effort": "minimal",
        }
        retry_times = resolve_llm_retry_times(self._llm, minimum=1)
        last_response: Any = None

        try:
            if hasattr(self._llm, "run"):
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                    retry_times=retry_times,
                    request_kwargs=request_kwargs,
                )
                last_response = response
                text = extract_message_text(response.choices[0].message).strip()
            else:
                response = await self._llm(prompt)
                last_response = response
                text = str(response).strip() if response else ""
        except Exception:
            response = await self._llm.run(
                messages=[{"role": "user", "content": prompt}],
                retry_times=retry_times,
                request_kwargs=request_kwargs,
            )
            last_response = response
            text = extract_message_text(response.choices[0].message).strip()

        # 提取 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            self._write_llm_debug_artifact(
                self._llm_debug_payload(
                    last_response,
                    text=text,
                    error="LLM response has no JSON array",
                )
            )
            preview = text[:160].replace("\n", " ")
            raise ProfileInjectionRoutingError(f"LLM response has no JSON array: {preview!r}")

        try:
            items = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            self._write_llm_debug_artifact(
                self._llm_debug_payload(
                    last_response,
                    text=text,
                    error=f"JSON parse failed: {e}",
                )
            )
            raise ProfileInjectionRoutingError(f"JSON parse failed: {e}") from e

        decisions: list[DimensionDecision] = []
        valid_actions = {"inject", "enrich", "override", "skip"}
        valid_dims = set(dim_summaries.keys())

        for item in items:
            if not isinstance(item, dict):
                continue
            dim = item.get("dimension", "")
            action = item.get("action", "")
            if dim not in valid_dims or action not in valid_actions:
                continue
            decisions.append(self._normalize_decision(
                dim=dim,
                action=action,
                reason=item.get("reason", ""),
                override_content=item.get("override_content", ""),
                has_history=dim_has_data.get(dim, False),
            ))

        # 确保所有维度都有决策
        covered = {d.dimension for d in decisions}
        for dim_name in dim_order:
            if dim_name not in covered:
                if dim_has_data[dim_name]:
                    decisions.append(DimensionDecision(
                        dimension=dim_name, action="inject",
                        reason="LLM 未返回该维度决策，默认 inject",
                    ))
                else:
                    decisions.append(DimensionDecision(
                        dimension=dim_name, action="skip",
                        reason="LLM 未返回该维度决策且无数据，skip",
                    ))

        ordered_decisions = sorted(
            decisions,
            key=lambda d: dim_order.index(d.dimension),
        )
        return ordered_decisions

    @staticmethod
    def _fallback_all_inject(
        dim_has_data: dict[str, bool],
        *,
        reason: str = "显式 fallback",
    ) -> list[DimensionDecision]:
        """显式降级：有数据的维度全量 inject，无数据的 skip。"""
        decisions = []
        for dim_name, has_data in dim_has_data.items():
            if has_data:
                decisions.append(DimensionDecision(
                    dimension=dim_name, action="inject",
                    reason=reason,
                ))
            else:
                decisions.append(DimensionDecision(
                    dimension=dim_name, action="skip",
                    reason="无数据，skip",
                ))
        return decisions

    @staticmethod
    def _append_profile_injected_preference(wm: Any, pref: TempPreference) -> None:
        """Append router-approved LTM profile preferences without per-item LLM conflict checks.

        The router has already made the dimension-level inject/enrich/override decision.
        Running a second LLM conflict check for every atomized profile entry is expensive
        and can erase compatible notes that happen to share a sub-dimension.
        """
        existing_items = getattr(wm, "_temp_preferences", None)
        if not isinstance(existing_items, list):
            return

        normalized_content = " ".join(str(pref.content or "").split()).strip()
        normalized_dimension = str(pref.dimension or "").strip()
        if not normalized_content:
            return

        for existing in existing_items:
            if getattr(existing, "superseded", False):
                continue
            if str(getattr(existing, "dimension", "") or "").strip() != normalized_dimension:
                continue
            existing_content = " ".join(str(getattr(existing, "content", "") or "").split()).strip()
            if existing_content == normalized_content:
                return

        existing_items.append(pref)

    @staticmethod
    async def apply(
        decisions: list[DimensionDecision],
        wm: Any,
        profile: UserProfile,
        llm: Callable | None = None,
    ) -> dict[str, str]:
        """将路由决策应用到 WorkingMemory。

        使用 add_preference 进行精准冲突检测（LLM 判断同维度条目
        是否冲突，只 supersede 冲突的，不冲突的共存）。

        Args:
            decisions: route() 返回的决策列表
            wm: WorkingMemory 实例
            profile: 用于获取维度文本的 UserProfile
            llm: LLM callable，传递给 add_preference

        Returns:
            应用结果摘要 {dimension: action}
        """
        result_summary: dict[str, str] = {}

        for decision in decisions:
            dim = decision.dimension
            action = decision.action

            if action == "skip":
                result_summary[dim] = "skip"
                continue

            if action == "inject":
                # 将 LTM 维度值原样写入 WM；general 逐条写入，避免块级文本污染后续 merge
                dim_entries = ProfileInjectionRouter._dimension_entries_for_injection(profile, dim)
                for entry_dim, entry_text in dim_entries:
                    ProfileInjectionRouter._append_profile_injected_preference(
                        wm,
                        TempPreference(
                            content=entry_text,
                            dimension=entry_dim,
                            preference_type="value",
                            scope="global",
                            source_task_id=LTM_INJECT_SOURCE,
                        ),
                    )
                if dim_entries:
                    result_summary[dim] = "inject"
                else:
                    result_summary[dim] = "skip (no data)"

            elif action == "enrich":
                # LTM 原值 + 用户补充内容，作为独立的两条写入 WM
                dim_entries = ProfileInjectionRouter._dimension_entries_for_injection(profile, dim)
                for entry_dim, entry_text in dim_entries:
                    ProfileInjectionRouter._append_profile_injected_preference(
                        wm,
                        TempPreference(
                            content=entry_text,
                            dimension=entry_dim,
                            preference_type="value",
                            scope="global",
                            source_task_id=LTM_INJECT_SOURCE,
                        ),
                    )
                enrich_content = decision.override_content
                if enrich_content:
                    await wm.add_preference(
                        TempPreference(
                            content=enrich_content,
                            dimension=dim,
                            preference_type="value",
                            scope="global",
                            source_task_id=PROFILE_OVERRIDE_SOURCE,
                        ),
                        llm=llm,
                    )
                result_summary[dim] = "enrich"

            elif action == "override":
                content = decision.override_content
                if not content:
                    # override 但无提炼内容，降级为 skip（不注入旧值）
                    result_summary[dim] = "override (skip, no content)"
                    continue
                ProfileInjectionRouter._supersede_active_dimension_prefix(wm, dim)
                await wm.add_preference(
                    TempPreference(
                        content=content,
                        dimension=dim,
                        preference_type="value",
                        scope="global",
                        source_task_id=PROFILE_OVERRIDE_SOURCE,
                    ),
                    llm=llm,
                )
                result_summary[dim] = "override"

        return result_summary

    @staticmethod
    def format_decisions_for_log(decisions: list[DimensionDecision]) -> str:
        """格式化决策列表为日志文本。"""
        lines = []
        for d in decisions:
            line = f"  {d.dimension}: {d.action}"
            if d.reason:
                line += f" ({d.reason})"
            if d.override_content:
                line += f" → \"{d.override_content[:80]}\""
            lines.append(line)
        return "\n".join(lines)
