"""StyleIntentClassifier — 风格意图分类器 (Stage 14)

用于 Design 阶段之前：判断用户是否有模板/风格意图，提取风格属性。
与 IntentClassifier（用于 Modify 阶段）并行存在。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)
STYLE_INTENT_MAX_TOKENS = 384

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


class TemplateIntent(StrEnum):
    """模板意图类型"""
    EXPLICIT = "explicit"      # 明确要求模板: "用商务模板"、"用上次那个模板"
    IMPLICIT = "implicit"      # 隐含风格偏好: "简约风格"、"学术风格"
    NONE = "none"              # 无风格/模板意图: 纯内容描述
    ANTI = "anti"              # 明确拒绝模板: "不要用模板"、"自由发挥"


class TemplateUseDecision(StrEnum):
    """第一阶段模板启用 gate。"""
    USE_TEMPLATE = "use_template"
    NO_TEMPLATE = "no_template"
    FORBID_TEMPLATE = "forbid_template"


class TemplateUseBasis(StrEnum):
    """第一阶段模板启用或跳过的依据。"""
    EXPLICIT_TEXT = "explicit_text"
    EXPLICIT_FILE = "explicit_file"
    MEMORY_REUSE = "memory_reuse"
    STRONG_REFERENCE_STYLE = "strong_reference_style"
    STYLE_ONLY = "style_only"
    CONTENT_ONLY = "content_only"
    ANTI_TEMPLATE = "anti_template"


@dataclass
class StyleAttributes:
    """从用户输入中提取的风格属性"""
    narrative_style: str = ""      # academic | business | creative | technical | educational
    info_density: str = ""         # low | medium | high
    color_tone: str = ""           # warm | cool | neutral | vibrant | dark
    layout_preference: str = ""    # minimal | balanced | dense
    reference_style: str = ""      # 参考风格描述 (如 "像苹果发布会")

    def is_empty(self) -> bool:
        return not any([
            self.narrative_style, self.info_density, self.color_tone,
            self.layout_preference, self.reference_style
        ])


@dataclass
class StyleIntentResult:
    """风格意图分类结果"""
    template_intent: TemplateIntent = TemplateIntent.NONE
    template_use_decision: TemplateUseDecision = TemplateUseDecision.NO_TEMPLATE
    template_use_basis: TemplateUseBasis = TemplateUseBasis.CONTENT_ONLY
    style_attributes: StyleAttributes = field(default_factory=StyleAttributes)
    template_query: str = ""           # 用于模板检索的关键词
    memory_hint: str = ""              # 记忆相关提示: "上次的模板"、"之前用过的"
    confidence: float = 0.5
    reasoning: str = ""                # LLM 推理过程

    def to_dict(self) -> dict:
        return {
            "template_intent": self.template_intent.value,
            "template_use_decision": self.template_use_decision.value,
            "template_use_basis": self.template_use_basis.value,
            "style_attributes": asdict(self.style_attributes),
            "template_query": self.template_query,
            "memory_hint": self.memory_hint,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 规则匹配器 (快速路径，无需 LLM)
# ══════════════════════════════════════════════════════════════════════════════

# 明确模板意图的关键词
EXPLICIT_TEMPLATE_PATTERNS = [
    r"(?:请)?用\s*[^，。,.]{0,40}?模板",      # "用商务模板"、"用这个模板"
    r"套用\s*[^，。,.]{0,40}?模板",           # "套用XX模板"
    r"使用\s*[^，。,.]{0,40}?模板",           # "使用XX模板"
    r"按照\s*[^，。,.]{0,60}?模板",           # "按照XX模板"
    r"按\s*[^，。,.]{0,40}?模板",             # "按商务模板生成"
    r"用上次.{0,4}模板",       # "用上次的模板"
    r"用之前.{0,4}模板",       # "用之前的模板"
    r"用那个模板",
    r"\buse\b.{0,80}\btemplate\b",
    r"\bapply\b.{0,80}\btemplate\b",
]

# 明确拒绝模板的关键词
ANTI_TEMPLATE_PATTERNS = [
    r"不要.{0,2}模板",         # "不要用模板"
    r"不用模板",
    r"自由发挥",
    r"自由设计",
    r"不套模板",
    r"do not use any template",
    r"no template at all",
    r"without any template",
    r"freeform mode",
    r"freeform",
]

# 隐含风格偏好的关键词映射
STYLE_KEYWORD_MAP = {
    # narrative_style
    "学术": {"narrative_style": "academic"},
    "academic": {"narrative_style": "academic"},
    "商务": {"narrative_style": "business"},
    "business": {"narrative_style": "business"},
    "商业": {"narrative_style": "business"},
    "创意": {"narrative_style": "creative"},
    "creative": {"narrative_style": "creative"},
    "技术": {"narrative_style": "technical"},
    "technical": {"narrative_style": "technical"},
    "教育": {"narrative_style": "educational"},
    "educational": {"narrative_style": "educational"},
    "教学": {"narrative_style": "educational"},
    # info_density
    "简约": {"info_density": "low", "layout_preference": "minimal"},
    "极简": {"info_density": "low", "layout_preference": "minimal"},
    "minimal": {"info_density": "low", "layout_preference": "minimal"},
    "简洁": {"info_density": "low"},
    "详细": {"info_density": "high"},
    "丰富": {"info_density": "high"},
    "dense": {"info_density": "high"},
    # color_tone
    "暖色": {"color_tone": "warm"},
    "warm": {"color_tone": "warm"},
    "冷色": {"color_tone": "cool"},
    "cool": {"color_tone": "cool"},
    "深色": {"color_tone": "dark"},
    "dark": {"color_tone": "dark"},
    "浅色": {"color_tone": "light"},
    "light": {"color_tone": "light"},
    "鲜艳": {"color_tone": "vibrant"},
    "vibrant": {"color_tone": "vibrant"},
}

# 记忆相关提示
MEMORY_HINT_PATTERNS = [
    (r"上次.{0,4}模板", "last_template"),
    (r"之前.{0,4}模板", "previous_template"),
    (r"以前.{0,4}模板", "previous_template"),
    (r"那个模板", "that_template"),
    (r"last.*template", "last_template"),
    (r"previous.*template", "previous_template"),
]

# 参考风格模式
REFERENCE_STYLE_PATTERNS = [
    r"像(.{2,20})一样",        # "像苹果发布会一样"
    r"类似(.{2,20})风格",      # "类似TED风格"
    r"参考(.{2,20})的?风格",   # "参考苹果的风格"
    r"like (.{2,30})",         # "like Apple keynote"
    r"similar to (.{2,30})",   # "similar to TED talks"
]


class StyleIntentClassifier:
    """风格意图分类器

    支持两种模式:
    1. 规则匹配 (快速路径): 基于关键词匹配，无需 LLM
    2. LLM 分类 (精确路径): 调用 LLM 进行深度分析
    """

    def __init__(
        self,
        llm: Any = None,
        artifact_writer: Any = None,
        use_llm: bool = True,
    ):
        """
        Args:
            llm: LLM 调用接口 (async callable 或有 run 方法的对象)
            artifact_writer: 中间产物写入器
            use_llm: 是否使用 LLM (False 时仅用规则匹配)
        """
        self.llm = llm
        self._artifact_writer = artifact_writer
        self._use_llm = use_llm and llm is not None

    async def classify(
        self,
        user_message: str,
        user_preferences: list[dict] | None = None,
        available_templates: list[str] | None = None,
        template_store: Any | None = None,
        user_id: str = "default",
        embedding_func: Any | None = None,
        memory_intent: str = "",
        template_profile_context: dict[str, Any] | None = None,
    ) -> StyleIntentResult:
        """分类用户消息的风格意图

        Args:
            user_message: 用户输入
            user_preferences: 用户历史偏好（从记忆系统获取）
            available_templates: 可用模板列表（用于提示 LLM）
            template_store: TemplateStore 实例（用于场景匹配）
            user_id: 用户 ID（用于查询历史记录）
            embedding_func: 嵌入函数（用于生成查询向量）

        Returns:
            StyleIntentResult
        """
        # 1. 快速规则匹配
        rule_result = self._rule_based_classify(user_message)
        rule_result = self._apply_profile_defaults(
            rule_result,
            template_profile_context,
        )

        # 如果规则匹配置信度高，优先保留强规则，但允许 LLM 补充风格属性
        if (
            rule_result.template_use_decision
            in (TemplateUseDecision.USE_TEMPLATE, TemplateUseDecision.FORBID_TEMPLATE)
            or rule_result.confidence >= 0.8
        ):
            if self._use_llm and rule_result.template_intent in (
                TemplateIntent.EXPLICIT,
                TemplateIntent.IMPLICIT,
            ):
                try:
                    llm_result = await self._llm_classify(
                        user_message,
                        user_preferences,
                        available_templates,
                        template_profile_context,
                    )
                    final_result = self._apply_profile_defaults(
                        self._merge_results(rule_result, llm_result),
                        template_profile_context,
                    )
                    # Persist the merged result so the latest artifact matches
                    # the final Stage 14 routing decision seen in runtime logs.
                    self._save_artifact(user_message, "", "", final_result, source="final")
                    return final_result
                except Exception as e:
                    logger.warning(
                        f"LLM enrichment failed, using rule result: {e}"
                    )
            self._save_artifact(user_message, "", "", rule_result, source="rule")
            return rule_result

        # 2. 场景匹配 (如果 template_store 可用)
        if template_store is not None and embedding_func is not None:
            try:
                matched_record, similarity = await self._find_similar_usage(
                    user_message=user_message,
                    template_store=template_store,
                    user_id=user_id,
                    embedding_func=embedding_func,
                    memory_intent=memory_intent,
                    threshold=0.75,
                )
                if matched_record is not None:
                    # 场景匹配成功，返回 IMPLICIT 意图
                    scenario_result = StyleIntentResult(
                        template_intent=TemplateIntent.IMPLICIT,
                        template_use_decision=TemplateUseDecision.NO_TEMPLATE,
                        template_use_basis=TemplateUseBasis.STYLE_ONLY,
                        template_query=matched_record.template_name,
                        memory_hint="similar_scenario",
                        confidence=similarity,
                        reasoning=(
                            "Matched historical scenario for style context only; "
                            f"similarity {similarity:.3f}: {matched_record.user_message[:100]}"
                        ),
                    )
                    logger.debug(
                        f"[STYLE_INTENT] Scenario match: template={matched_record.template_name}, "
                        f"similarity={similarity:.3f}, historical_msg={matched_record.user_message[:50]}..."
                    )
                    self._save_artifact(user_message, "", "", scenario_result, source="scenario")
                    return scenario_result
            except Exception as e:
                logger.warning(f"Scenario matching failed: {e}")

        # 3. LLM 深度分析 (如果启用)
        if self._use_llm:
            try:
                llm_result = await self._llm_classify(
                    user_message,
                    user_preferences,
                    available_templates,
                    template_profile_context,
                )
                # 合并规则结果和 LLM 结果
                final_result = self._apply_profile_defaults(
                    self._merge_results(rule_result, llm_result),
                    template_profile_context,
                )
                # Persist the merged result so downstream inspection can read the
                # final classifier output directly without inferring merge logic.
                self._save_artifact(user_message, "", "", final_result, source="final")
                return final_result
            except Exception as e:
                logger.warning(f"LLM classification failed, using rule result: {e}")

        # 降级到规则结果
        return rule_result

    def _rule_based_classify(self, user_message: str) -> StyleIntentResult:
        """基于规则的快速分类"""
        msg_lower = user_message.lower()

        if self._looks_like_template_reuse_request(msg_lower):
            memory_hint = (
                self._extract_template_reuse_memory_hint(msg_lower)
                or self._extract_memory_hint(user_message)
                or "previous_template"
            )
            template_query = self._extract_template_query(user_message, None)
            return StyleIntentResult(
                template_intent=TemplateIntent.IMPLICIT,
                template_use_decision=TemplateUseDecision.USE_TEMPLATE,
                template_use_basis=TemplateUseBasis.MEMORY_REUSE,
                template_query=template_query,
                memory_hint=memory_hint,
                confidence=0.78,
                reasoning="Matched template-reuse semantics before anti-template patterns",
            )

        # 检查明确拒绝模板
        for pattern in ANTI_TEMPLATE_PATTERNS:
            if re.search(pattern, msg_lower):
                return StyleIntentResult(
                    template_intent=TemplateIntent.ANTI,
                    template_use_decision=TemplateUseDecision.FORBID_TEMPLATE,
                    template_use_basis=TemplateUseBasis.ANTI_TEMPLATE,
                    confidence=0.9,
                    reasoning="Matched anti-template pattern",
                )

        # 检查明确要求模板
        template_query = ""
        for pattern in EXPLICIT_TEMPLATE_PATTERNS:
            match = re.search(pattern, msg_lower)
            if match:
                # 提取模板关键词
                template_query = self._extract_template_query(user_message, match)
                memory_hint = self._extract_memory_hint(user_message)
                return StyleIntentResult(
                    template_intent=TemplateIntent.EXPLICIT,
                    template_use_decision=TemplateUseDecision.USE_TEMPLATE,
                    template_use_basis=TemplateUseBasis.EXPLICIT_TEXT,
                    template_query=template_query,
                    memory_hint=memory_hint,
                    confidence=0.85,
                    reasoning="Matched explicit template pattern",
                )

        reference_style = ""
        reference_matches: list[str] = []
        for pattern in REFERENCE_STYLE_PATTERNS:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                reference_style = match.group(1).strip()
                reference_matches.append(f"reference:{reference_style}")
                break
        if reference_style:
            return StyleIntentResult(
                template_intent=TemplateIntent.IMPLICIT,
                template_use_decision=TemplateUseDecision.USE_TEMPLATE,
                template_use_basis=TemplateUseBasis.STRONG_REFERENCE_STYLE,
                style_attributes=StyleAttributes(reference_style=reference_style),
                template_query=reference_style,
                confidence=0.78,
                reasoning=f"Matched strong reference style: {reference_matches}",
            )

        # 检查隐含风格偏好
        style_attrs = StyleAttributes()
        matched_keywords = []
        for keyword, attrs in STYLE_KEYWORD_MAP.items():
            if keyword in msg_lower:
                matched_keywords.append(keyword)
                for attr_name, attr_value in attrs.items():
                    if not getattr(style_attrs, attr_name):
                        setattr(style_attrs, attr_name, attr_value)

        if not style_attrs.is_empty():
            return StyleIntentResult(
                template_intent=TemplateIntent.IMPLICIT,
                template_use_decision=TemplateUseDecision.NO_TEMPLATE,
                template_use_basis=TemplateUseBasis.STYLE_ONLY,
                style_attributes=style_attrs,
                template_query="",
                confidence=0.7,
                reasoning=f"Matched style keywords: {matched_keywords}",
            )

        # 无明确意图
        return StyleIntentResult(
            template_intent=TemplateIntent.NONE,
            template_use_decision=TemplateUseDecision.NO_TEMPLATE,
            template_use_basis=TemplateUseBasis.CONTENT_ONLY,
            confidence=0.6,
            reasoning="No template/style intent detected",
        )

    @staticmethod
    def _looks_like_template_reuse_request(msg_lower: str) -> bool:
        reuse_patterns = (
            r"\b(previously learned|learned|memory|historical|previous|prior|last)\b.{0,60}\b(template|layout|template preference)\b",
            r"\b(template|layout)\b.{0,60}\b(reuse|learned|memory|history|historical)\b",
            r"(复用|沿用|使用|套用).{0,24}(上次|之前|先前|历史|记忆|学到).{0,24}(模板|版式|布局)",
            r"(上次|之前|先前|历史|记忆|学到).{0,24}(模板|版式|布局)",
        )
        if not any(re.search(pattern, msg_lower) for pattern in reuse_patterns):
            return False
        if any(
            re.search(pattern, msg_lower)
            for pattern in (
                r"(do not use any template|no template at all|freeform mode|自由发挥|自由设计|不套模板)",
                r"不要.{0,4}模板",
                r"不用模板",
            )
        ):
            return False
        return True

    @staticmethod
    def _extract_template_reuse_memory_hint(msg_lower: str) -> str:
        if any(
            re.search(pattern, msg_lower)
            for pattern in (r"(上次|之前|先前|以前|last|previous|prior|earlier)",)
        ):
            return "previous_template"
        if any(
            re.search(pattern, msg_lower)
            for pattern in (r"(reuse|learned|history|historical|memory)",)
        ):
            return "previous_template"
        return ""

    @staticmethod
    def _extract_profile_defaults(
        template_profile_context: dict[str, Any] | None,
    ) -> tuple[StyleAttributes, list[str], bool]:
        if not template_profile_context:
            return StyleAttributes(), [], False

        raw_defaults = template_profile_context.get("profile_style_defaults", {})
        if not isinstance(raw_defaults, dict):
            raw_defaults = {}

        attrs = StyleAttributes(
            narrative_style=str(raw_defaults.get("narrative_style", "") or "").strip(),
            info_density=str(raw_defaults.get("info_density", "") or "").strip(),
            color_tone=str(raw_defaults.get("color_tone", "") or "").strip(),
            layout_preference=str(raw_defaults.get("layout_preference", "") or "").strip(),
            reference_style=str(raw_defaults.get("reference_style", "") or "").strip(),
        )

        query_terms: list[str] = []
        for value in template_profile_context.get("profile_template_query_terms", []) or []:
            text = str(value or "").strip()
            if text and text not in query_terms:
                query_terms.append(text)

        last_successful_template_name = str(
            template_profile_context.get("last_successful_template_name", "") or ""
        ).strip()
        if last_successful_template_name and last_successful_template_name not in query_terms:
            query_terms.append(last_successful_template_name)

        history_templates = template_profile_context.get("history_preferred_templates", []) or []
        for template_name in history_templates:
            text = str(template_name or "").strip()
            if text and text not in query_terms:
                query_terms.append(text)

        has_template_history = any(
            [
                last_successful_template_name,
                history_templates,
                template_profile_context.get("history_preferred_template_ids"),
                template_profile_context.get("history_supporting_usage_count"),
            ]
        )
        return attrs, query_terms[:5], has_template_history

    def _apply_profile_defaults(
        self,
        result: StyleIntentResult,
        template_profile_context: dict[str, Any] | None,
    ) -> StyleIntentResult:
        """用稳定画像补齐风格属性，并在必要时提供弱隐式风格意图。"""
        if result.template_intent == TemplateIntent.ANTI:
            return result

        profile_attrs, query_terms, has_template_history = self._extract_profile_defaults(
            template_profile_context
        )
        if profile_attrs.is_empty() and not query_terms and not has_template_history:
            return result

        if result.style_attributes.is_empty():
            result.style_attributes = profile_attrs
        else:
            for attr_name in (
                "narrative_style",
                "info_density",
                "color_tone",
                "layout_preference",
                "reference_style",
            ):
                if not getattr(result.style_attributes, attr_name) and getattr(profile_attrs, attr_name):
                    setattr(result.style_attributes, attr_name, getattr(profile_attrs, attr_name))

        if not result.template_query and query_terms:
            result.template_query = " ".join(query_terms[:3])

        if result.template_use_decision == TemplateUseDecision.USE_TEMPLATE and not result.template_query and query_terms:
            result.template_query = " ".join(query_terms[:3])

        if (
            result.template_intent == TemplateIntent.IMPLICIT
            and result.template_use_decision == TemplateUseDecision.USE_TEMPLATE
            and has_template_history
        ):
            result.confidence = max(result.confidence, 0.72)
        elif result.template_use_decision != TemplateUseDecision.USE_TEMPLATE:
            profile_reason = "Profile defaults used only as style context, not template permission"
            result.reasoning = (
                f"{result.reasoning}; {profile_reason}"
                if result.reasoning else profile_reason
            )

        return result

    def _extract_template_query(self, message: str, match: re.Match | None) -> str:
        """从匹配中提取模板查询关键词"""
        msg = (message or "").strip()

        name_patterns = [
            r"(?:请)?(?:帮我|给我)?(?:用|套用|使用|按照|按)\s*([^，。,.]{1,60}?)\s*模板",
            r"模板[：:]\s*([^，。,.]{1,60})",
            r"\b(?:use|apply)\b\s+(?:the\s+)?(.{1,60}?)\s+template",
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, msg, re.IGNORECASE)
            if not name_match:
                continue
            candidate = name_match.group(1).strip()
            cleaned = self._clean_template_query(candidate)
            if cleaned:
                return cleaned

        if match is not None:
            fallback = self._clean_template_query(match.group(0))
            if fallback and fallback not in {"上次", "之前", "那个"}:
                return fallback
        return ""

    @staticmethod
    def _clean_template_query(raw_query: str) -> str:
        """清洗模板查询词，尽量保留模板名，去掉动作词和噪声。"""
        query = (raw_query or "").strip()
        if not query:
            return ""

        query = re.sub(r"(?i)\b(?:use|apply|the|template)\b", " ", query)
        query = query.replace("模板", " ")
        query = re.sub(
            r"^(请|帮我|给我|麻烦|想要|我要|我想|请帮我)?\s*"
            r"(用|套用|使用|按照|按)\s*",
            "",
            query,
        )
        query = re.sub(
            r"\s*(生成|制作|做|做个|做一个|做一份|做个关于.+|生成一份|制作一份).*$",
            "",
            query,
        )
        query = re.sub(r"\s+", " ", query).strip(" ：:,-_，。")

        normalized = re.sub(r"\s+", "", query).replace("的", "")
        if normalized in {
            "上次",
            "之前",
            "以前",
            "那个",
            "这个",
            "上次那个",
            "之前那个",
            "以前那个",
            "上次这个",
            "之前这个",
            "以前这个",
        }:
            return ""
        return query

    def _extract_memory_hint(self, message: str) -> str:
        """提取记忆相关提示"""
        msg_lower = message.lower()
        for pattern, hint_type in MEMORY_HINT_PATTERNS:
            if re.search(pattern, msg_lower):
                return hint_type
        return ""

    async def _find_similar_usage(
        self,
        user_message: str,
        template_store: Any,
        user_id: str,
        embedding_func: Any,
        memory_intent: str = "",
        threshold: float = 0.75,
    ) -> tuple[Any | None, float]:
        """通过嵌入相似度查找最相似的历史使用记录

        Args:
            user_message: 当前用户输入
            template_store: TemplateStore 实例
            user_id: 用户 ID
            embedding_func: 嵌入函数
            threshold: 最小相似度阈值（默认 0.75）

        Returns:
            (matched_record, similarity_score) 或 (None, 0.0)

        Algorithm:
            1. 获取使用历史: template_store.get_usage_history(user_id, limit=20)
            2. 生成查询嵌入: embedding_func([user_message])
            3. 计算与每条记录的余弦相似度
            4. 跳过 embedding 为 None 的记录
            5. 找到最大相似度
            6. 如果 max >= threshold，返回 (record, score)
            7. 否则返回 (None, 0.0)
        """
        from memslides.memory.core.embedding import cosine_similarity
        import numpy as np

        # 1. 获取使用历史
        usage_history = await template_store.get_usage_history(
            user_id=user_id,
            limit=20,
            success_only=False,
            memory_intent=memory_intent,
        )
        if not usage_history and memory_intent:
            usage_history = await template_store.get_usage_history(
                user_id=user_id,
                limit=20,
                success_only=False,
            )

        if not usage_history:
            logger.debug("[STYLE_INTENT] No usage history found for scenario matching")
            return (None, 0.0)

        # 2. 生成查询嵌入
        try:
            query_embeddings = await embedding_func([user_message])
            if query_embeddings is None or len(query_embeddings) == 0:
                logger.warning("[STYLE_INTENT] Failed to generate query embedding")
                return (None, 0.0)
            query_vec = np.array(query_embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[STYLE_INTENT] Embedding generation failed: {e}")
            return (None, 0.0)

        # 3. 计算相似度并找到最佳匹配
        max_similarity = 0.0
        best_match = None

        for record in usage_history:
            # 跳过没有嵌入的记录
            if record.user_message_embedding is None:
                continue

            try:
                record_vec = np.array(record.user_message_embedding, dtype=np.float32)
                similarity = cosine_similarity(query_vec, record_vec)

                # Failed runs can still indicate a useful template choice when
                # the downstream crash happened after template injection, but
                # they should remain weaker evidence than fully successful runs.
                weighted_similarity = (
                    similarity
                    if getattr(record, "success", True)
                    else similarity * 0.98
                )

                if weighted_similarity > max_similarity:
                    max_similarity = weighted_similarity
                    best_match = record
            except Exception as e:
                logger.debug(f"[STYLE_INTENT] Failed to compute similarity for record {record.id}: {e}")
                continue

        # 4. 检查是否超过阈值
        if max_similarity >= threshold and best_match is not None:
            logger.info(
                f"[STYLE_INTENT] Found similar scenario: template={best_match.template_name}, "
                f"similarity={max_similarity:.3f}, success={getattr(best_match, 'success', True)}, "
                f"msg='{best_match.user_message[:50]}...'"
            )
            return (best_match, max_similarity)

        logger.debug(f"[STYLE_INTENT] No scenario match above threshold (max={max_similarity:.3f})")
        return (None, 0.0)

    async def _llm_classify(
        self,
        user_message: str,
        user_preferences: list[dict] | None,
        available_templates: list[str] | None,
        template_profile_context: dict[str, Any] | None = None,
    ) -> StyleIntentResult:
        """使用 LLM 进行深度分类"""
        prompt = self._build_prompt(
            user_message,
            user_preferences,
            available_templates,
            template_profile_context,
        )
        result_text = await self._call_llm(prompt)
        result = self._parse_llm_result(result_text)

        # 保存中间产物
        self._save_artifact(user_message, prompt, result_text, result, source="llm")

        return result

    def _build_prompt(
        self,
        user_message: str,
        user_preferences: list[dict] | None,
        available_templates: list[str] | None,
        template_profile_context: dict[str, Any] | None = None,
    ) -> str:
        pref_section = ""
        if user_preferences:
            pref_lines = [f"- {p.get('preference', '')}" for p in user_preferences[:5]]
            pref_section = "\n## 用户历史偏好\n" + "\n".join(pref_lines)

        template_profile_section = ""
        if template_profile_context:
            template_profile_section = (
                "\n## 用户模板稳定摘要\n"
                + json.dumps(template_profile_context, ensure_ascii=False, indent=2)
            )

        template_section = ""
        if available_templates:
            template_section = "\n## 可用模板\n" + ", ".join(available_templates[:10])

        return f"""你是一个 PPT 风格意图分类器。你的第一职责是判断本轮是否应该进入模板选择。

## 用户消息
{user_message}
{pref_section}
{template_profile_section}
{template_section}

## 第一阶段模板 gate
- template_use_decision=use_template: 用户明确要求模板、明确复用 previous/learned template，或给出强参考风格（如 like Apple keynote / 参考某明确品牌或模板风格）。
- template_use_decision=no_template: 用户只有普通风格词或纯内容请求；不要进入模板选择。
- template_use_decision=forbid_template: 用户明确拒绝模板或要求 freeform。

## template_use_basis
- explicit_text: 文本明确要求使用模板。
- memory_reuse: 明确要求复用之前/历史/学到的模板或布局。
- strong_reference_style: 明确要求类似某个品牌、发布会、已知视觉对象或强参考风格。
- style_only: 只有 calm/professional/academic/blue-green/minimal 等普通风格词。
- content_only: 纯内容请求，无模板或风格需求。
- anti_template: 明确不要模板。

## 兼容旧字段 template_intent
- explicit: 对应 use_template 且 basis 为 explicit_text。
- implicit: 对应 use_template 的 memory_reuse/strong_reference_style，或 no_template 的 style_only。
- none: 对应 no_template/content_only。
- anti: 对应 forbid_template/anti_template。

## 风格属性（仅当 intent 为 explicit 或 implicit 时提取）
- narrative_style: academic | business | creative | technical | educational | ""
- info_density: low | medium | high | ""
- color_tone: warm | cool | neutral | vibrant | dark | ""
- layout_preference: minimal | balanced | dense | ""
- reference_style: 参考风格描述（如"像苹果发布会"）

## 使用画像摘要的规则
- template_profile_context 只能补充风格属性，不能单独构成模板启用许可
- 只有用户明确要求模板、提供模板、或要求复用 previous/learned template 时，才可视为模板使用意图
- 当用户只说 calm/professional/academic/blue-green 等普通风格词时，输出 no_template/style_only
- 但如果用户本轮明确提出了相反要求，必须以用户本轮要求为准

## 输出 JSON（仅输出 JSON，不要其他内容）
{{
    "template_intent": "explicit|implicit|none|anti",
    "template_use_decision": "use_template|no_template|forbid_template",
    "template_use_basis": "explicit_text|memory_reuse|strong_reference_style|style_only|content_only|anti_template",
    "style_attributes": {{
        "narrative_style": "",
        "info_density": "",
        "color_tone": "",
        "layout_preference": "",
        "reference_style": ""
    }},
    "template_query": "用于模板检索的关键词（如有）",
    "memory_hint": "记忆相关提示（如'last_template'）",
    "confidence": 0.0-1.0,
    "reasoning": "简要推理过程"
}}"""

    def _parse_llm_result(self, result_text: str) -> StyleIntentResult:
        """解析 LLM 返回的 JSON"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return StyleIntentResult(confidence=0.3)

        try:
            data = json.loads(json_match.group())
            style_attrs_data = data.get("style_attributes", {})
            template_intent = TemplateIntent(data.get("template_intent", "none"))
            decision_raw = data.get("template_use_decision")
            basis_raw = data.get("template_use_basis")
            if decision_raw:
                template_use_decision = TemplateUseDecision(decision_raw)
            elif template_intent == TemplateIntent.ANTI:
                template_use_decision = TemplateUseDecision.FORBID_TEMPLATE
            elif template_intent == TemplateIntent.EXPLICIT:
                template_use_decision = TemplateUseDecision.USE_TEMPLATE
            else:
                template_use_decision = TemplateUseDecision.NO_TEMPLATE

            if basis_raw:
                template_use_basis = TemplateUseBasis(basis_raw)
            elif template_use_decision == TemplateUseDecision.FORBID_TEMPLATE:
                template_use_basis = TemplateUseBasis.ANTI_TEMPLATE
            elif template_intent == TemplateIntent.EXPLICIT:
                template_use_basis = TemplateUseBasis.EXPLICIT_TEXT
            elif template_use_decision == TemplateUseDecision.NO_TEMPLATE and template_intent == TemplateIntent.IMPLICIT:
                template_use_basis = TemplateUseBasis.STYLE_ONLY
            elif template_use_decision == TemplateUseDecision.NO_TEMPLATE:
                template_use_basis = TemplateUseBasis.CONTENT_ONLY
            else:
                template_use_basis = TemplateUseBasis.STRONG_REFERENCE_STYLE

            return StyleIntentResult(
                template_intent=template_intent,
                template_use_decision=template_use_decision,
                template_use_basis=template_use_basis,
                style_attributes=StyleAttributes(
                    narrative_style=style_attrs_data.get("narrative_style", ""),
                    info_density=style_attrs_data.get("info_density", ""),
                    color_tone=style_attrs_data.get("color_tone", ""),
                    layout_preference=style_attrs_data.get("layout_preference", ""),
                    reference_style=style_attrs_data.get("reference_style", ""),
                ),
                template_query=data.get("template_query", ""),
                memory_hint=data.get("memory_hint", ""),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM result: {e}")
            return StyleIntentResult(confidence=0.3)

    def _merge_results(
        self, rule_result: StyleIntentResult, llm_result: StyleIntentResult
    ) -> StyleIntentResult:
        """合并规则结果和 LLM 结果"""
        if (
            rule_result.template_use_decision
            in (TemplateUseDecision.USE_TEMPLATE, TemplateUseDecision.FORBID_TEMPLATE)
            and rule_result.confidence >= 0.78
        ):
            if rule_result.style_attributes.is_empty() and not llm_result.style_attributes.is_empty():
                rule_result.style_attributes = llm_result.style_attributes
            if not rule_result.memory_hint and llm_result.memory_hint:
                rule_result.memory_hint = llm_result.memory_hint
            if not rule_result.template_query and llm_result.template_query:
                rule_result.template_query = llm_result.template_query
            if llm_result.reasoning:
                rule_result.reasoning = (
                    f"{rule_result.reasoning}; LLM={llm_result.reasoning}"
                    if rule_result.reasoning else llm_result.reasoning
                )
            return rule_result

        if (
            llm_result.template_use_decision
            in (TemplateUseDecision.USE_TEMPLATE, TemplateUseDecision.FORBID_TEMPLATE)
            and llm_result.confidence >= rule_result.confidence
        ):
            return llm_result

        # 否则使用规则结果，但补充 LLM 提取的风格属性
        if rule_result.style_attributes.is_empty() and not llm_result.style_attributes.is_empty():
            rule_result.style_attributes = llm_result.style_attributes
        if not rule_result.memory_hint and llm_result.memory_hint:
            rule_result.memory_hint = llm_result.memory_hint
        if not rule_result.template_query and llm_result.template_query:
            rule_result.template_query = llm_result.template_query

        return rule_result

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if callable(self.llm) and not hasattr(self.llm, "run"):
            return await self.llm(prompt)
        from memslides.memory.extract.llm_compat import extract_response_text, resolve_llm_retry_times
        response = await self.llm.run(
            messages=[{"role": "user", "content": prompt}],
            retry_times=resolve_llm_retry_times(self.llm, minimum=1),
            request_kwargs={
                "max_tokens": STYLE_INTENT_MAX_TOKENS,
                "temperature": 0.0,
            },
        )
        return extract_response_text(response)

    def _save_artifact(
        self,
        user_message: str,
        prompt: str,
        response: str,
        result: StyleIntentResult,
        source: str,
    ) -> None:
        """保存中间产物"""
        if not self._artifact_writer:
            return

        try:
            if hasattr(self._artifact_writer, "write_style_intent"):
                self._artifact_writer.write_style_intent(
                    user_message=user_message,
                    prompt=prompt,
                    response=response,
                    result=result.to_dict(),
                    source=source,
                )
        except Exception as e:
            logger.debug(f"Style intent artifact write failed: {e}")
