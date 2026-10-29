"""LLM-based template selector for generation-time auto matching."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from memslides.templates.semantic_access import canonical_layout_names

logger = logging.getLogger(__name__)


@dataclass
class TemplateSelectionResult:
    """Selection result returned by the LLM template selector."""

    should_use_template: bool = False
    selected_template_id: str = ""
    selected_template_name: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    considered_template_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_use_template": self.should_use_template,
            "selected_template_id": self.selected_template_id,
            "selected_template_name": self.selected_template_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "considered_template_ids": self.considered_template_ids,
        }


class TemplateSelector:
    """Use an LLM to select a template from the current user's pool."""

    def __init__(
        self,
        llm: Any,
        artifact_writer: Any = None,
        max_templates: int = 50,
        max_history: int = 12,
    ) -> None:
        self.llm = llm
        self._artifact_writer = artifact_writer
        self._max_templates = max_templates
        self._max_history = max_history

    async def select(
        self,
        *,
        user_message: str,
        style_intent: dict[str, Any],
        templates: list[Any],
        usage_history: list[dict[str, Any]] | None = None,
        user_preferences: list[dict[str, Any]] | None = None,
        template_profile_context: dict[str, Any] | None = None,
        user_id: str = "",
        aspect_ratio: str = "",
    ) -> TemplateSelectionResult:
        """Select the best template or decide to skip template usage."""
        candidate_summaries = [
            self._summarize_template(template)
            for template in templates[: self._max_templates]
        ]
        history_summaries = self._summarize_usage_history(
            usage_history[: self._max_history] if usage_history else []
        )

        prompt = self._build_prompt(
            user_message=user_message,
            style_intent=style_intent,
            candidates=candidate_summaries,
            usage_history=history_summaries,
            user_preferences=user_preferences or [],
            template_profile_context=template_profile_context or {},
            user_id=user_id,
            aspect_ratio=aspect_ratio,
        )

        response_text = await self._call_llm(prompt)
        selection = self._parse_result(response_text)

        if self._artifact_writer and hasattr(self._artifact_writer, "write_template_selection"):
            try:
                self._artifact_writer.write_template_selection(
                    user_message=user_message,
                    prompt=prompt,
                    response=response_text,
                    selection_result=selection.to_dict(),
                    candidates=candidate_summaries,
                    usage_history=history_summaries,
                )
            except Exception as e:
                logger.debug("Template selection artifact write failed: %s", e)

        return selection

    def _summarize_template(self, template: Any) -> dict[str, Any]:
        slide_induction = getattr(template, "slide_induction", {}) or {}
        content_patterns = getattr(template, "content_patterns", None)
        layout_names = canonical_layout_names(template)[:6]

        return {
            "id": getattr(template, "id", ""),
            "name": getattr(template, "name", ""),
            "description": getattr(template, "description", ""),
            "user_id": getattr(template, "user_id", ""),
            "aspect_ratio": getattr(template, "aspect_ratio", ""),
            "slide_count": getattr(template, "slide_count", 0),
            "functional_keys": list(slide_induction.get("functional_keys", []))[:8],
            "layout_examples": layout_names,
            "layout_capability": getattr(
                getattr(template, "semantic_model", None),
                "layout_capability",
                {},
            ),
            "content_patterns": {
                "narrative_style": getattr(content_patterns, "narrative_style", ""),
                "info_density": getattr(content_patterns, "info_density", ""),
                "typical_sections": list(getattr(content_patterns, "typical_sections", []) or [])[:8],
                "bullet_style": getattr(content_patterns, "bullet_style", ""),
                "max_bullets_per_slide": getattr(content_patterns, "max_bullets_per_slide", 0),
            },
        }

    @staticmethod
    def _summarize_usage_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not history:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        for item in history:
            key = item.get("template_id") or item.get("template_name") or "__unknown__"
            entry = grouped.setdefault(
                key,
                {
                    "template_id": item.get("template_id", ""),
                    "template_name": item.get("template_name", ""),
                    "usage_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "recent_examples": [],
                    "recent_attachment_types": [],
                    "recent_intents": [],
                },
            )
            entry["usage_count"] += 1
            if item.get("success", True):
                entry["success_count"] += 1
            else:
                entry["failure_count"] += 1
            if len(entry["recent_examples"]) < 3 and item.get("user_message"):
                entry["recent_examples"].append(item["user_message"][:200])
            attachment_type = item.get("attachment_type", "")
            if attachment_type and attachment_type not in entry["recent_attachment_types"]:
                entry["recent_attachment_types"].append(attachment_type)
            intent = item.get("intent", "")
            if intent and intent not in entry["recent_intents"]:
                entry["recent_intents"].append(intent)

        return sorted(
            grouped.values(),
            key=lambda x: (-x["success_count"], -x["usage_count"], x["template_name"]),
        )

    def _build_prompt(
        self,
        *,
        user_message: str,
        style_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
        usage_history: list[dict[str, Any]],
        user_preferences: list[dict[str, Any]],
        template_profile_context: dict[str, Any],
        user_id: str,
        aspect_ratio: str,
    ) -> str:
        pref_lines = [
            p.get("preference", "")
            for p in user_preferences[:8]
            if p.get("preference")
        ]

        return (
            "你是一个 PPT 模板选择器。第一阶段 gate 已经判定本轮允许进入模板选择；"
            "你的任务不是重新判断普通风格词是否该套模板，而是在候选模板中选择最合适的一个，"
            "或在候选明显不匹配时返回 should_use_template=false。\n\n"
            "选择原则：\n"
            "1. 必须尊重第一阶段 gate 的 template_use_decision/template_use_basis。\n"
            "2. 必须只从候选模板中选择，不要编造模板 ID 或名称。\n"
            "3. 若 template_use_basis=memory_reuse，应重点根据历史模板使用记录、用户模板稳定摘要和当前请求选择；不要固定取最近模板。\n"
            "4. 若 template_use_basis=strong_reference_style，应选择最接近该强参考风格的候选；没有合适候选就 skip。\n"
            "5. 历史记录和用户模板稳定摘要可作为先验，但不能压过当前请求。\n"
            "6. usage_history 中 success_count 高的模板优先级更高；failure_count 仅表示该模板曾被尝试但后续流程报错，只能作为弱参考。\n"
            "7. 如果多个模板名字相似，请结合 description、content_patterns、functional_keys 和历史示例判断。\n"
            "8. 如果没有足够合适的模板，返回 should_use_template=false。\n\n"
            "输出要求：只输出 JSON，不要输出解释性文字。\n"
            "JSON schema:\n"
            "{\n"
            '  "should_use_template": true,\n'
            '  "selected_template_id": "候选中的 template id，若不使用模板则为空",\n'
            '  "selected_template_name": "候选中的模板名，若不使用模板则为空",\n'
            '  "confidence": 0.0,\n'
            '  "reasoning": "简短理由",\n'
            '  "considered_template_ids": ["被重点比较过的候选 id"]\n'
            "}\n\n"
            f"## 当前 user_id\n{user_id}\n\n"
            f"## 当前请求比例\n{aspect_ratio}\n\n"
            f"## 用户消息\n{user_message}\n\n"
            f"## 风格/模板意图分析\n{json.dumps(style_intent, ensure_ascii=False, indent=2)}\n\n"
            f"## 当前用户历史偏好\n{json.dumps(pref_lines, ensure_ascii=False, indent=2)}\n\n"
            f"## 用户模板稳定摘要\n{json.dumps(template_profile_context, ensure_ascii=False, indent=2)}\n\n"
            f"## 历史模板使用记录摘要\n{json.dumps(usage_history, ensure_ascii=False, indent=2)}\n\n"
            f"## 候选模板（仅允许从这些模板中选择）\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        )

    async def _call_llm(self, prompt: str) -> str:
        if callable(self.llm) and not hasattr(self.llm, "run"):
            return await self.llm(prompt)
        response = await self.llm.run(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _parse_result(self, result_text: str) -> TemplateSelectionResult:
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if not json_match:
            logger.warning("No JSON found in template selection response")
            return TemplateSelectionResult()

        try:
            data = json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to parse template selection response: %s", e)
            return TemplateSelectionResult()

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return TemplateSelectionResult(
            should_use_template=self._coerce_bool(data.get("should_use_template", False)),
            selected_template_id=str(data.get("selected_template_id", "") or ""),
            selected_template_name=str(data.get("selected_template_name", "") or ""),
            confidence=confidence,
            reasoning=str(data.get("reasoning", "") or ""),
            considered_template_ids=[
                str(x)
                for x in (data.get("considered_template_ids") or [])
                if str(x).strip()
            ],
        )

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False
