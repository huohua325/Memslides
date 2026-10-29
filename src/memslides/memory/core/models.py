"""记忆系统核心数据模型。

活跃类:
- Message             (对话消息)
- ExperienceTrace     (经验轨迹)
- DesignEpisode       (情景记忆)
- AtomicPreference    (原子级用户偏好)

已删除 (Stage B 清理，0 外部调用):
- EditSegment, Episode, Rule, AtomicFact, Foresight, FactualRecord
- RuleScope, RuleStatus, MemoryType, IntentType
- ReactionType, RuleType, ConstraintType, DesignCategory, EpisodeStatus
"""

from __future__ import annotations

import json
import inspect
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any


_CANONICAL_EXPERIENCE_TYPES = {
    "tool_error",
    "tool_misuse",
    "tool_limitation",
    "constraint",
    "pattern",
    "chain",
    "agent_run",
    "generic",
}

_TASK_EXPERIENCE_CATEGORIES = {
    "tool_error",
    "tool_misuse",
    "tool_limitation",
    "pattern",
    "generic",
}

_EXPERIENCE_TYPE_ALIASES = {
    "tool_error": "tool_error",
    "hard_error": "tool_error",
    "error": "tool_error",
    "failed": "tool_error",
    "tool_misuse": "tool_misuse",
    "misuse": "tool_misuse",
    "soft_failure": "tool_misuse",
    "tool_limitation": "tool_limitation",
    "tool_limitations": "tool_limitation",
    "limitation": "tool_limitation",
    "limit": "tool_limitation",
    "constraint": "constraint",
    "pattern": "pattern",
    "pipeline": "pattern",
    "workflow": "pattern",
    "effective_pattern": "pattern",
    "effective_patterns": "pattern",
    "chain": "chain",
    "agent_run": "agent_run",
    "generic": "generic",
}

_EXPERIENCE_TYPE_PRIORITY = {
    "tool_error": 0,
    "tool_misuse": 1,
    "tool_limitation": 2,
    "constraint": 3,
    "pattern": 4,
    "chain": 5,
    "agent_run": 6,
    "generic": 7,
}


def normalize_experience_type(value: Any) -> str:
    """将经验类别标签归一为稳定的语义类型。"""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("-", "_").replace(" ", "_")
    return _EXPERIENCE_TYPE_ALIASES.get(normalized, normalized if normalized in _CANONICAL_EXPERIENCE_TYPES else "")


def _parse_experience_scenarios(scenarios: Any) -> list[str]:
    if scenarios is None:
        return []
    parsed = scenarios
    if isinstance(parsed, str):
        stripped = parsed.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                parsed = [stripped]
        else:
            parsed = [stripped]
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def infer_experience_type(
    experience_type: Any = "",
    scenarios: Any = None,
    final_outcome: str = "",
) -> str:
    """根据显式类型、场景标签和结果状态推断统一经验类型。"""
    explicit = normalize_experience_type(experience_type)
    if explicit and explicit != "generic":
        return explicit

    inferred: list[str] = []
    for scenario in _parse_experience_scenarios(scenarios):
        normalized = normalize_experience_type(scenario)
        if normalized:
            inferred.append(normalized)
            continue

        lowered = scenario.lower().replace("-", "_").replace(" ", "_")
        if lowered.startswith("agent_") or lowered == "initial_generation":
            inferred.append("agent_run")
        elif lowered.startswith("chain"):
            inferred.append("chain")

    if inferred:
        return min(inferred, key=lambda item: _EXPERIENCE_TYPE_PRIORITY.get(item, 99))

    outcome = str(final_outcome or "").strip().lower()
    if outcome in {"failed", "error"}:
        return "tool_error"
    if explicit:
        return explicit
    return "generic"


def default_outcome_for_experience_type(experience_type: Any, fallback: str = "") -> str:
    """为经验类型提供合理的默认 outcome。"""
    normalized = infer_experience_type(experience_type)
    if normalized == "tool_error":
        return "failed"
    if normalized in {"tool_misuse", "tool_limitation"}:
        return "partial"
    if normalized in {"pattern", "constraint", "chain", "agent_run"}:
        return "success"
    return fallback or "success"


def _infer_legacy_constraint_task_category(
    scenarios: Any = None,
    final_outcome: str = "",
    content: str = "",
) -> str:
    """将历史 constraint 标签折叠进新的 task-experience 分类体系。"""
    outcome = str(final_outcome or "").strip().lower()
    if outcome in {"failed", "error"}:
        return "tool_error"

    scenario_text = " ".join(_parse_experience_scenarios(scenarios))
    lowered = f"{scenario_text} {content}".lower()
    if any(
        term in lowered
        for term in (
            "tool_limitation",
            "limitation",
            "blind",
            "blindspot",
            "unsupported",
            "not support",
            "can't",
            "cannot",
            "unable to",
            "看不到",
            "不支持",
            "无法",
            "只能",
            "盲区",
            "限制",
        )
    ):
        return "tool_limitation"
    if any(
        term in lowered
        for term in (
            "pattern",
            "pipeline",
            "workflow",
            "最佳实践",
            "流程",
        )
    ):
        return "pattern"
    return "tool_misuse"


def infer_task_experience_category(
    experience_type: Any = "",
    scenarios: Any = None,
    final_outcome: str = "",
    content: str = "",
) -> str:
    """推断 RoundExperience 的轮次级分类，排除 chain 等独立注入路径。"""
    normalized = infer_experience_type(experience_type, scenarios, final_outcome)
    if normalized == "constraint":
        return _infer_legacy_constraint_task_category(scenarios, final_outcome, content)
    if normalized in _TASK_EXPERIENCE_CATEGORIES:
        return normalized
    return "generic"


# ═══════════════════════════════════════════════
# 会话消息
# ═══════════════════════════════════════════════


@dataclass
class Message:
    """单条对话消息"""
    role: str = ""                          # "user" | "assistant" | "system"
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)  # intent_type, slide_index等

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# 经验轨迹
# ═══════════════════════════════════════════════


@dataclass
class ExperienceTrace:
    """经验轨迹（Experiential Memory）

    记录Agent推理过程，辅助新Session中类似任务的PPT生成。

    Stage 12 扩展: 支持链级经验的完整信息（零损失转换）
    - experience_type: 区分来源（tool_error | chain | agent_run）
    - 链级字段（可选）: chain_name, tool_sequence, anti_pattern, applicable_when
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    task_description: str = ""         # 任务描述
    reasoning_steps: str = "[]"        # JSON: [{step, thought, action, result}]
    tools_used: str = "[]"             # JSON: 工具列表（兼容旧代码）
    final_outcome: str = ""            # success/partial/failed
    lessons_learned: str = ""          # LLM总结的经验教训
    applicable_scenarios: str = "[]"   # JSON: 场景标签
    confidence: float = 0.7
    reuse_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"             # active | superseded | archived
    superseded_by: str = ""            # 被哪条新经验替代
    superseded_at: str = ""            # 被替代时间
    merged_from_ids: str = "[]"        # JSON: 来源 RoundExperience / ExperienceTrace IDs
    source_types: str = "[]"           # JSON: ["agent_lesson", "auto_extract", ...]
    consolidation_version: str = ""    # Stage 15 归档版本标记
    template_id: str = ""              # Stage 9: 关联的模板 ID

    # Stage 12: 链级扩展字段（可选，向后兼容）
    experience_type: str = "generic"   # "tool_error" | "chain" | "agent_run"
    chain_name: str = ""               # 工具链名称（仅 chain 类型）
    tool_sequence: str = "[]"          # JSON: list[str] 工具序列（替代 tools_used）
    anti_pattern: str = ""             # 反模式描述（仅 chain 类型）
    applicable_when: str = ""          # 详细适用条件（比 applicable_scenarios 更具体）
    source_chain_ids: str = "[]"       # JSON: list[str] 来源链ID（仅 chain 类型）

    def __post_init__(self) -> None:
        self.experience_type = infer_experience_type(
            self.experience_type,
            self.applicable_scenarios,
            self.final_outcome,
        )

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceTrace":
        return _from_dict(cls, d)


# ═══════════════════════════════════════════════
# WM 统一经验模型 — RoundExperience
# ═══════════════════════════════════════════════


@dataclass
class RoundExperience:
    """WM 统一经验模型，描述一轮交互中得到的可复用经验。

    所有来源共用同一模型，通过 source 字段区分：
    - "agent_lesson": remember_lesson 主动记录
    - "auto_extract": ExperienceTraceWriter 自动提取
    - "tool_error":   工具调用失败时即时记录
    - "preload":      Job 开始时从 LTM 预加载

    生命周期 (Stage 15 统一):
    1. remember_lesson / 自动提取 / 工具错误 / LTM 预加载 → 写入 WM
    2. 每个 Round 开始 → 向量检索 → 注入 System Prompt
    3. Job 结束 → 所有非 preload 的经验按 category 分组合并 → 写入 LTM
       (preload 本身来自 LTM，跳过)
    4. 下一个 Job 开始 → 从 LTM 预加载到 WM
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                   # 经验内容（核心字段）
    tool_name: str = ""                 # 相关工具名
    keywords: list[str] = field(default_factory=list)  # 检索关键词
    category: str = "tool_error"        # 轮次级语义类型: tool_error | tool_misuse | tool_limitation | pattern | generic
    source: str = "agent_lesson"        # "agent_lesson" | "auto_extract" | "tool_error" | "preload"
    source_task_id: str = ""            # 历史字段：来源 Round ID，经 DB adapter 映射到旧列名
    source_job_id: str = ""             # 来源 Job ID
    confidence: float = 0.8
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 自动提取时保留的额外上下文（agent_lesson 时为空）
    tools_used: list[str] = field(default_factory=list)
    outcome: str = ""                   # "success" | "partial" | "failed"

    # 向量检索支持（延迟计算）
    _embedding: list[float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.category = infer_task_experience_category(
            self.category,
            self.keywords,
            self.outcome,
            self.content,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_embedding", None)  # 不序列化 embedding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoundExperience":
        valid = {f.name for f in fields(cls) if not f.name.startswith("_")}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def to_prompt_text(self) -> str:
        """生成注入 prompt 的文本"""
        prefix = f"[{self.tool_name}] " if self.tool_name else ""
        suffix = ""
        if self.source == "auto_extract" and self.outcome:
            suffix = f" (outcome: {self.outcome})"
        return f"- {prefix}{self.content}{suffix}"


TempExperienceTrace = RoundExperience


# ═══════════════════════════════════════════════
# 序列化辅助
# ═══════════════════════════════════════════════

# dataclass 字段中，哪些是 JSON 编码的（dict / list 存为 str）
_JSON_FIELDS: dict[type, set[str]] = {}


def _register_json_fields(cls: type, field_names: set[str]) -> None:
    _JSON_FIELDS[cls] = field_names


_register_json_fields(
    ExperienceTrace,
    {
        "reasoning_steps",
        "tools_used",
        "applicable_scenarios",
        "tool_sequence",
        "source_chain_ids",
        "merged_from_ids",
        "source_types",
    },
)


def _to_dict(obj: Any) -> dict[str, Any]:
    """将 dataclass 转为 dict，JSON 字段自动 dumps。"""
    d = asdict(obj)
    json_fields = _JSON_FIELDS.get(type(obj), set())
    for k in json_fields:
        if k in d and not isinstance(d[k], str):
            d[k] = json.dumps(d[k], ensure_ascii=False)
    return d


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """从 dict 重建 dataclass，JSON 字段自动 loads。"""
    json_fields = _JSON_FIELDS.get(cls, set())
    valid_field_names = {f.name for f in fields(cls)}
    cleaned = {}
    for k, v in data.items():
        if k not in valid_field_names:
            continue
        if k in json_fields and isinstance(v, str):
            try:
                cleaned[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cls(**cleaned)


# ═══════════════════════════════════════════════
# 情景记忆 — DesignEpisode
# ═══════════════════════════════════════════════

@dataclass
class DesignEpisode:
    """一次有意义的交互事件 — 因果链记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    source_round_id: int = 0

    # 因果链
    user_intent: str = ""
    interpretation_gap: str = ""
    action_outcome: str = ""

    # 洞察
    design_insight: str = ""

    # 元数据
    category: str = ""
    confidence: float = 0.7
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    context: str = "{}"                # Stage 9: JSON, e.g. {"template_id": "tpl_xxx"}

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignEpisode:
        cleaned = dict(data)
        # 忽略 SQLite 中残留的已废弃列
        for legacy in ("used_for_profile_update", "used_for_rule_induction"):
            cleaned.pop(legacy, None)
        return _from_dict(cls, cleaned)


# ═══════════════════════════════════════════════
# 原子级用户偏好 — AtomicPreference
# ═══════════════════════════════════════════════

@dataclass
class AtomicPreference:
    """原子级用户偏好 — 细粒度、可独立检索

    设计理念:
    1. 每个偏好是独立的、原子的，可单独检索和验证
    2. preference_type 区分"价值观偏好"和"策略偏好"
    3. scope 支持全局、幻灯片类型、元素类型三级作用域
    4. conflict_group 实现同类偏好的自动覆盖
    """

    # ── 标识 ──
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    # ── 核心内容 ──
    preference_type: str = "value"          # "value" | "strategy"
    trigger: str = ""                        # 触发条件（何时适用）
    preference: str = ""                     # 偏好内容（用户想要什么）
    rationale: str = ""                      # 来源解释（为什么有这个偏好）

    # ── 作用域 ──
    scope: str = "global"                    # "global" | "slide_type" | "element_type"
    scope_value: str = ""                    # scope 的具体值，如 "title", "text", "image"

    # ── 来源追踪 ──
    source_job_id: str = ""                  # 产生此偏好的 Job ID
    source_session_id: str = ""              # 产生此偏好的 Session ID

    # ── 证据追踪 ──
    evidence_episode_ids: str = "[]"         # JSON: 支撑此偏好的 Episode IDs

    # ── 置信度管理 ──
    confidence: float = 0.5                  # 0.0 - 1.0
    verified_count: int = 0                  # 正向验证次数
    contradiction_count: int = 0             # 反向矛盾次数

    # ── 冲突处理 ──
    conflict_group: str = ""                 # 冲突组标识，同组内新偏好覆盖旧偏好

    # ── 状态管理 ──
    status: str = "active"                   # "active" | "deprecated" | "superseded"

    # ── 时间戳 ──
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicPreference":
        return _from_dict(cls, data)

    def matches_context(self, context: dict) -> bool:
        """检查偏好是否适用于给定上下文

        Args:
            context: {"slide_type": "title", "element_type": "text", ...}
        """
        if self.scope == "global":
            return True
        if self.scope == "slide_type":
            return context.get("slide_type") == self.scope_value
        if self.scope == "element_type":
            return context.get("element_type") == self.scope_value
        return False


_register_json_fields(AtomicPreference, {"evidence_episode_ids"})


# ═══════════════════════════════════════════════
# 维度归一化
# ═══════════════════════════════════════════════

VALID_DIMENSIONS: dict[str, str] = {
    # Coarse experiment / session-preference axes.
    # They are mapped to stable canonical subfields so WM/LTM can use a
    # consistent storage schema while experiment prompts can still speak in
    # simpler human-facing dimension names.
    "layout": "layout.slide_structure",
    "color": "theme.primary_colors",
    "typography": "theme.font_family",
    "theme.primary_colors": "theme.primary_colors",
    "theme.colors": "theme.primary_colors",
    "theme.accent_colors": "theme.accent_colors",
    "theme.font_family": "theme.font_family",
    "theme.font": "theme.font_family",
    "theme.font_size_range": "theme.font_size_range",
    "theme.background_style": "theme.background_style",
    "typography.font_family": "theme.font_family",
    "typography.font": "theme.font_family",
    "visual.image_style": "visual.image_style",
    "visual.chart_type_priority": "visual.chart_type_priority",
    "visual.icon_usage": "visual.icon_usage",
    "visual.animation_preference": "visual.animation_preference",
    "layout.content_density": "layout.content_density",
    "layout.alignment_style": "layout.alignment_style",
    "layout.spacing_preference": "layout.spacing_preference",
    "layout.slide_structure": "layout.slide_structure",
    "content.text_density": "content.text_density",
    "content.language_style": "content.language_style",
    "content.bullet_point_style": "content.bullet_point_style",
    "content.title_length": "content.title_length",
    "template.preferred_templates": "template.preferred_templates",
    "template.avoid_templates": "template.avoid_templates",
    "template.selection_criteria": "template.selection_criteria",
}

_TEMPLATE_KEYWORD_HINTS = (
    "模板",
    "template",
    "母版",
    "slide master",
    "主题模板",
)

_STABLE_TEMPLATE_PREFERENCE_PATTERNS = (
    r"以后.{0,20}(默认|优先|尽量|继续|一直|还是|都用|使用|选用).{0,20}模板",
    r"之后.{0,20}(默认|优先|尽量|继续|一直|还是|都用|使用|选用).{0,20}模板",
    r"未来.{0,20}(默认|优先|尽量|继续|一直|还是|都用|使用|选用).{0,20}模板",
    r"默认.{0,30}模板",
    r"优先.{0,30}模板",
    r"尽量.{0,30}模板",
    r"总是.{0,30}模板",
    r"一直.{0,30}模板",
    r"习惯.{0,30}模板",
    r"长期.{0,30}模板",
    r"避免.{0,30}模板",
    r"不要再用.{0,30}模板",
    r"以后别用.{0,30}模板",
    r"prefer.{0,30}template",
    r"default.{0,30}template",
    r"avoid.{0,30}template",
)

_TRANSIENT_TEMPLATE_REQUEST_PATTERNS = (
    r"(?:请)?用\s*[^，。,.]{0,60}?模板",
    r"套用\s*[^，。,.]{0,60}?模板",
    r"使用\s*[^，。,.]{0,60}?模板",
    r"按照\s*[^，。,.]{0,60}?模板",
    r"按\s*[^，。,.]{0,60}?模板",
    r"用上次.{0,10}模板",
    r"用之前.{0,10}模板",
    r"用那个模板",
)


def normalize_dimension(raw_dim: str) -> str:
    """将 LLM 输出的 dimension 归一化到标准值。

    策略：精确匹配 → 前缀匹配 → 原样返回（兜底）
    """
    raw_dim = raw_dim.strip().lower()
    if raw_dim in VALID_DIMENSIONS:
        return VALID_DIMENSIONS[raw_dim]
    # 前缀匹配：找到与 raw_dim 同一类别前缀的最长 key
    best_match = ""
    for key in VALID_DIMENSIONS:
        prefix = key.rsplit(".", 1)[0] + "."
        if raw_dim.startswith(prefix):
            if len(key) > len(best_match):
                best_match = key
    if best_match:
        return VALID_DIMENSIONS[best_match]
    return raw_dim


def looks_like_template_related_text(text: Any) -> bool:
    """Whether the text refers to template selection or template constraints."""
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _TEMPLATE_KEYWORD_HINTS)


def has_stable_template_preference_signal(text: Any) -> bool:
    """Whether the text expresses a lasting template preference instead of a one-off task choice."""
    normalized = str(text or "").strip().lower()
    if not normalized or not looks_like_template_related_text(normalized):
        return False
    return any(re.search(pattern, normalized) for pattern in _STABLE_TEMPLATE_PREFERENCE_PATTERNS)


def is_transient_template_instruction(text: Any) -> bool:
    """Whether the text is a one-off task-time template instruction."""
    normalized = str(text or "").strip().lower()
    if not normalized or not looks_like_template_related_text(normalized):
        return False
    if has_stable_template_preference_signal(normalized):
        return False
    if any(re.search(pattern, normalized) for pattern in _TRANSIENT_TEMPLATE_REQUEST_PATTERNS):
        return True
    return True


def is_template_related_preference(
    *,
    dimension: str = "",
    content: Any = "",
    trigger: Any = "",
    rationale: Any = "",
) -> bool:
    """Whether a preference item belongs to template memory semantics."""
    normalized_dim = normalize_dimension(str(dimension or "")) if dimension else ""
    if normalized_dim == "template" or normalized_dim.startswith("template."):
        return True
    return any(
        looks_like_template_related_text(text)
        for text in (content, trigger, rationale)
    )


# ═══════════════════════════════════════════════
# 执行层级模型 — Job / Round / Operation
# ═══════════════════════════════════════════════


@dataclass
class Operation:
    """Round 中的单次工具调用或内部动作（不入库，仅在 WM 中临时存储）"""
    tool_name: str
    args_summary: str = ""
    result_summary: str = ""
    is_error: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Operation":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


@dataclass
class Round:
    """Job 中的一轮用户输入到 agent 响应的交互单元。"""
    id: str
    job_id: str
    user_message: str = ""
    agent_response: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str = ""
    operations: list[Operation] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["operations"] = [op.to_dict() for op in self.operations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Round":
        data = dict(d)
        ops_raw = data.pop("operations", [])
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        round_obj = cls(**filtered)
        round_obj.operations = [Operation.from_dict(op) for op in ops_raw]
        return round_obj

@dataclass
class Job:
    """一次多轮对话会话"""
    id: str
    user_id: str
    project_id: str
    status: str = "active"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str = ""
    rounds: list[Round] = field(default_factory=list)
    intent: str = ""  # Job级别的用户意图（academic/business/education等），在Job创建时确定一次
    read_intent: str = ""   # 画像读取 intent（优先参考哪条历史画像）
    write_intent: str = ""  # 画像写回 intent（本次形成的新稳定偏好应沉淀到哪条画像）
    core_persona: str = ""  # 用户稳定人格底色（如 scholar / executive / designer）

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rounds"] = [r.to_dict() for r in self.rounds]
        d.pop("tasks", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        data = dict(d)
        rounds_raw = data.pop("rounds", None)
        if rounds_raw is None:
            rounds_raw = data.pop("tasks", [])
        else:
            data.pop("tasks", None)
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        job = cls(**filtered)
        job.rounds = [Round.from_dict(r) for r in rounds_raw]
        return job

# ═══════════════════════════════════════════════
# WM 临时偏好 — TempPreference
# ═══════════════════════════════════════════════


@dataclass
class TempPreference:
    """WM 中的临时偏好，带维度归一化"""
    content: str
    dimension: str = ""
    preference_type: str = "value"  # "value" | "strategy"
    source_task_id: str = ""
    scope: str = "global"
    scope_value: str = ""
    superseded: bool = False
    structured_data: dict | None = None  # 结构化参数（路径 B 视觉指纹提取直接写入）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.dimension:
            self.dimension = normalize_dimension(self.dimension)

    def matches_context(self, context: dict) -> bool:
        if self.scope == "global":
            return True
        if self.scope == "slide_type":
            return context.get("slide_type") == self.scope_value
        if self.scope == "element_type":
            return context.get("element_type") == self.scope_value
        return False


# ═══════════════════════════════════════════════
# 工具链模型 — ToolChain / ChainExperience
# ═══════════════════════════════════════════════


def make_chain_signature(tool_sequence: list[str]) -> str:
    """从工具序列生成确定性签名，去重连续相同工具后用 ``-`` 连接。

    三层命名体系：
    - Tool Trace: 单个 Round 内完整的原始工具调用序列
    - Chain Segment (原 ToolChain): Tool Trace 经 LLM 按语义切分后的片段
    - Chain Experience: 从 Chain Segment 蒸馏出的经验条目，存储于 LTM

    例::

        >>> make_chain_signature(["read_file", "read_file", "write_html", "inspect_slide"])
        'read_file-write_html-inspect_slide'
    """
    deduped: list[str] = []
    for tool in tool_sequence:
        if not deduped or deduped[-1] != tool:
            deduped.append(tool)
    return "-".join(deduped)



@dataclass
class ChainSegment:
    """一条工具链片段 — 若干连续 Cycle 的聚合（原 ToolChain）

    三层命名体系中的第二层：Tool Trace → Chain Segment → Chain Experience
    """
    chain_id: str
    chain_name: str                    # 确定性签名：make_chain_signature(tool_sequence)
    tool_sequence: list[str]
    cycle_indices: list[int]
    rich_traces: list[dict] = field(default_factory=list)
    """完整执行轨迹列表，每个元素:
    {
        "reasoning": str,       # 可观测的工具使用理由（显式文本或合成兜底）
        "tool_reason": str,     # 与 reasoning 同义，供新代码显式消费
        "reason_source": str,   # reasoning_content | assistant_content | synthesized_from_tool_context | ...
        "reason_quality": str,  # high | medium | low | none
        "tool_name": str,       # 工具名
        "arguments": str,       # 工具参数（大文本类做摘要）
        "observation": str,     # 工具返回结果（智能截断）
        "is_error": bool,
    }
    """
    semantic_label: str = ""           # LLM 给出的可读描述（可选，不参与匹配逻辑）
    outcome: str = "success"
    source_round_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChainSegment":
        data = dict(d)
        if "source_round_id" not in data and "source_task_id" in data:
            data["source_round_id"] = data.pop("source_task_id")
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @property
    def source_task_id(self) -> str:
        """Deprecated compatibility alias for ``source_round_id``."""
        return self.source_round_id

    @source_task_id.setter
    def source_task_id(self, value: str) -> None:
        self.source_round_id = value


# 向后兼容别名
ToolChain = ChainSegment



@dataclass
class ChainExperience:
    """从工具链中提炼的经验条目"""
    chain_name: str
    tool_pipeline: list[str]
    lesson: str
    applicable_when: str
    anti_pattern: str = ""
    source_chain_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 新增字段
    keywords: list[str] = field(default_factory=list)        # LLM 生成的检索关键词
    subkey: str = ""                                          # 同签名下唯一标识
    inject_summary: str = ""                                  # Operation 级短摘要
    support_count: int = 1                                    # 支持该经验的来源链数量
    merge_count: int = 0                                      # 被合并更新次数
    source_users: list[str] = field(default_factory=list)     # 来源用户（预留审计）
    source_job_ids: list[str] = field(default_factory=list)   # 来源任务/作业（预留审计）
    keyword_embedding: list[float] | None = field(default=None, repr=False)  # 向量嵌入

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("keyword_embedding", None)  # 不序列化到 JSON
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChainExperience":
        valid = {f.name for f in fields(cls) if not f.name.startswith("_")}
        filtered = {k: v for k, v in d.items() if k in valid and k != "keyword_embedding"}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# Round 摘要 — RoundSummary
# ═══════════════════════════════════════════════


@dataclass
class RoundSummary:
    """Round 结束时生成的结构化摘要，用于注入后续 Round 的 context。"""
    round_id: str
    round_index: int
    user_request: str
    outcome: str
    slides_modified: list[str]
    key_actions: list[str]
    unresolved_issues: list[str]
    user_feedback: str = ""
    design_episode: DesignEpisode | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_prompt_text(self) -> str:
        lines = [f"### 第 {self.round_index} 轮修改（{self.outcome}）"]
        lines.append(f"用户请求: {self.user_request}")
        if self.slides_modified:
            lines.append(f"修改的幻灯片: {', '.join(self.slides_modified)}")
        if self.key_actions:
            lines.append("关键操作:")
            for action in self.key_actions:
                lines.append(f"  - {action}")
        if self.unresolved_issues:
            lines.append("未解决:")
            for issue in self.unresolved_issues:
                lines.append(f"  - {issue}")
        if self.user_feedback:
            lines.append(f"用户反馈: {self.user_feedback}")
        # 因果链：DesignEpisode 提供 intent→gap→outcome 完整上下文
        if self.design_episode:
            ep = self.design_episode
            if ep.user_intent:
                lines.append(f"意图理解: {ep.user_intent}")
            if ep.interpretation_gap:
                lines.append(f"理解偏差: {ep.interpretation_gap}")
            if ep.action_outcome:
                lines.append(f"执行结果: {ep.action_outcome}")
            if ep.design_insight:
                lines.append(f"设计洞察: {ep.design_insight}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.design_episode is not None:
            d["design_episode"] = self.design_episode.to_dict()
        else:
            d["design_episode"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoundSummary":
        data = dict(d)
        if "round_id" not in data and "task_id" in data:
            data["round_id"] = data.pop("task_id")
        if "round_index" not in data and "task_index" in data:
            data["round_index"] = data.pop("task_index")
        ep_raw = data.pop("design_episode", None)
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        obj = cls(**filtered)
        if ep_raw is not None:
            obj.design_episode = DesignEpisode.from_dict(ep_raw)
        return obj


# ═══════════════════════════════════════════════
# Operation 级记忆分发结果 — OperationMemory
# ═══════════════════════════════════════════════


@dataclass
class OperationMemory:
    """RoundCache 为单个 Operation 筛选后的记忆集合。"""
    tool_experiences: list = field(default_factory=list)
    preferences: list = field(default_factory=list)


# ═══════════════════════════════════════════════
# UserProfile 结构化画像模型（Design Doc §4.5）
# ═══════════════════════════════════════════════


@dataclass
class ThemePreference:
    primary_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    font_family: str = ""
    font_size_range: tuple[int, int] | None = None
    background_style: str = ""
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [
        "颜色", "配色", "色彩", "主题色", "背景色", "字体", "字号", "font",
        "color", "theme", "palette", "accent", "background", "style",
    ])
    notes: list[str] = field(default_factory=list)


@dataclass
class VisualPreference:
    image_style: str = ""
    chart_type_priority: list[str] = field(default_factory=list)
    icon_usage: str = ""
    animation_preference: str = ""
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [
        "图片", "图表", "图标", "动画", "视觉", "icon", "chart", "image",
        "animation", "visual", "illustration", "diagram", "图形",
    ])
    notes: list[str] = field(default_factory=list)


@dataclass
class LayoutPreference:
    content_density: str = ""
    alignment_style: str = ""
    spacing_preference: str = ""
    slide_structure: str = ""
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [
        "布局", "排版", "对齐", "间距", "密度", "结构", "layout", "align",
        "spacing", "margin", "padding", "grid", "column", "分栏", "留白",
    ])
    notes: list[str] = field(default_factory=list)


@dataclass
class ContentPreference:
    text_density: str = ""
    language_style: str = ""
    bullet_point_style: str = ""
    title_length: str = ""
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [
        "文字", "文本", "标题", "正文", "要点", "bullet", "text", "title",
        "content", "paragraph", "语言", "风格", "简洁", "详细", "摘要",
    ])
    notes: list[str] = field(default_factory=list)


@dataclass
class TemplatePreference:
    preferred_templates: list[str] = field(default_factory=list)
    avoid_templates: list[str] = field(default_factory=list)
    selection_criteria: str = ""
    history_preferred_templates: list[str] = field(default_factory=list)
    history_preferred_template_ids: list[str] = field(default_factory=list)
    history_reuse_scenarios: list[str] = field(default_factory=list)
    history_supported_aspect_ratios: list[str] = field(default_factory=list)
    last_successful_template_id: str = ""
    last_successful_template_name: str = ""
    last_successful_at: str = ""
    history_supporting_usage_count: int = 0
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [
        "模板", "template", "母版", "master", "slide master", "主题模板",
    ])
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _normalize_text_list(values: Any) -> list[str]:
        normalized: list[str] = []
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def normalize(self) -> bool:
        """规范化模板画像字段，兼容旧 JSON 载荷。"""
        changed = False
        list_fields = (
            "preferred_templates",
            "avoid_templates",
            "history_preferred_templates",
            "history_preferred_template_ids",
            "history_reuse_scenarios",
            "history_supported_aspect_ratios",
            "notes",
            "keywords",
        )
        for field_name in list_fields:
            current = getattr(self, field_name, [])
            normalized = self._normalize_text_list(current)
            if normalized != current:
                setattr(self, field_name, normalized)
                changed = True

        int_fields = ("history_supporting_usage_count",)
        for field_name in int_fields:
            current = getattr(self, field_name, 0)
            try:
                normalized = int(current or 0)
            except (TypeError, ValueError):
                normalized = 0
            if normalized != current:
                setattr(self, field_name, normalized)
                changed = True

        scalar_fields = (
            "selection_criteria",
            "last_successful_template_id",
            "last_successful_template_name",
            "last_successful_at",
        )
        for field_name in scalar_fields:
            current = getattr(self, field_name, "")
            normalized = str(current or "").strip()
            if normalized != current:
                setattr(self, field_name, normalized)
                changed = True

        return changed

    def has_history_summary(self) -> bool:
        return any(
            [
                self.history_preferred_templates,
                self.history_preferred_template_ids,
                self.history_reuse_scenarios,
                self.history_supported_aspect_ratios,
                self.last_successful_template_id,
                self.last_successful_template_name,
                self.last_successful_at,
                self.history_supporting_usage_count,
            ]
        )

    def to_selection_context(self) -> dict[str, Any]:
        self.normalize()
        context = {
            "preferred_templates": list(self.preferred_templates),
            "avoid_templates": list(self.avoid_templates),
            "selection_criteria": self.selection_criteria,
            "history_preferred_templates": list(self.history_preferred_templates),
            "history_preferred_template_ids": list(self.history_preferred_template_ids),
            "history_reuse_scenarios": list(self.history_reuse_scenarios),
            "history_supported_aspect_ratios": list(self.history_supported_aspect_ratios),
            "last_successful_template_id": self.last_successful_template_id,
            "last_successful_template_name": self.last_successful_template_name,
            "last_successful_at": self.last_successful_at,
            "history_supporting_usage_count": self.history_supporting_usage_count,
            "notes": list(self.notes),
        }
        return {
            key: value
            for key, value in context.items()
            if value not in ("", [], 0, None)
        }

    def to_usage_selection_context(self) -> dict[str, Any]:
        """Selection-only template context derived from usage history."""
        self.normalize()
        context = {
            "history_preferred_templates": list(self.history_preferred_templates),
            "history_preferred_template_ids": list(self.history_preferred_template_ids),
            "history_reuse_scenarios": list(self.history_reuse_scenarios),
            "history_supported_aspect_ratios": list(self.history_supported_aspect_ratios),
            "last_successful_template_id": self.last_successful_template_id,
            "last_successful_template_name": self.last_successful_template_name,
            "last_successful_at": self.last_successful_at,
            "history_supporting_usage_count": self.history_supporting_usage_count,
        }
        return {
            key: value
            for key, value in context.items()
            if value not in ("", [], 0, None)
        }

    def refresh_from_usage_records(
        self,
        records: list["TemplateUsageRecord"],
        *,
        stable_threshold: int = 2,
        max_templates: int = 3,
        max_scenarios: int = 4,
        reset_semantic_fields: bool = False,
    ) -> bool:
        """从 template_usage_history 派生稳定模板摘要。

        这里不会修改手工语义字段（preferred_templates / selection_criteria / avoid_templates），
        只更新 history_* 与最近成功模板相关的系统维护字段。
        当 reset_semantic_fields=True 时，会清空旧的语义模板字段，
        让运行时模板选择完全由 usage history 驱动。
        """
        grouped: dict[str, dict[str, Any]] = {}
        supporting_records: list["TemplateUsageRecord"] = []

        for record in records or []:
            if not getattr(record, "success", True):
                continue
            template_id = str(getattr(record, "template_id", "") or "").strip()
            template_name = str(getattr(record, "template_name", "") or "").strip()
            if not template_id and not template_name:
                continue

            supporting_records.append(record)
            group_key = template_id or template_name
            entry = grouped.setdefault(
                group_key,
                {
                    "template_id": template_id,
                    "template_name": template_name,
                    "count": 0,
                    "latest_at": "",
                    "examples": [],
                    "aspect_ratios": [],
                },
            )
            entry["count"] += 1
            created_at = str(getattr(record, "created_at", "") or "").strip()
            if created_at and created_at > entry["latest_at"]:
                entry["latest_at"] = created_at
            if template_id and not entry["template_id"]:
                entry["template_id"] = template_id
            if template_name and not entry["template_name"]:
                entry["template_name"] = template_name

            user_message = str(getattr(record, "user_message", "") or "").strip()
            if user_message:
                truncated = user_message[:160]
                if (
                    truncated not in entry["examples"]
                    and len(entry["examples"]) < max_scenarios
                ):
                    entry["examples"].append(truncated)

            aspect_ratio = str(getattr(record, "aspect_ratio", "") or "").strip()
            if aspect_ratio and aspect_ratio not in entry["aspect_ratios"]:
                entry["aspect_ratios"].append(aspect_ratio)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda item: (
                int(item["count"]),
                str(item["latest_at"] or ""),
                str(item["template_name"] or item["template_id"] or ""),
            ),
            reverse=True,
        )

        stable_groups = [
            item for item in ordered_groups
            if int(item["count"]) >= max(stable_threshold, 1)
        ][:max_templates]

        history_preferred_templates = self._normalize_text_list(
            [item["template_name"] or item["template_id"] for item in stable_groups]
        )
        history_preferred_template_ids = self._normalize_text_list(
            [item["template_id"] for item in stable_groups if item["template_id"]]
        )

        history_reuse_scenarios: list[str] = []
        scenario_sources = stable_groups or ordered_groups[:max_templates]
        for item in scenario_sources:
            template_label = item["template_name"] or item["template_id"]
            for example in item["examples"]:
                scenario = f"{template_label}: {example}" if template_label else example
                if scenario not in history_reuse_scenarios:
                    history_reuse_scenarios.append(scenario)
                if len(history_reuse_scenarios) >= max_scenarios:
                    break
            if len(history_reuse_scenarios) >= max_scenarios:
                break

        history_supported_aspect_ratios: list[str] = []
        for item in ordered_groups:
            for aspect_ratio in item["aspect_ratios"]:
                if aspect_ratio not in history_supported_aspect_ratios:
                    history_supported_aspect_ratios.append(aspect_ratio)

        last_record = supporting_records[0] if supporting_records else None
        updates = {
            "history_preferred_templates": history_preferred_templates,
            "history_preferred_template_ids": history_preferred_template_ids,
            "history_reuse_scenarios": history_reuse_scenarios,
            "history_supported_aspect_ratios": history_supported_aspect_ratios,
            "last_successful_template_id": (
                str(getattr(last_record, "template_id", "") or "").strip()
                if last_record else ""
            ),
            "last_successful_template_name": (
                str(getattr(last_record, "template_name", "") or "").strip()
                if last_record else ""
            ),
            "last_successful_at": (
                str(getattr(last_record, "created_at", "") or "").strip()
                if last_record else ""
            ),
            "history_supporting_usage_count": len(supporting_records),
        }

        changed = False
        if reset_semantic_fields:
            semantic_resets = {
                "preferred_templates": [],
                "avoid_templates": [],
                "selection_criteria": "",
            }
            for field_name, new_value in semantic_resets.items():
                if getattr(self, field_name) != new_value:
                    setattr(self, field_name, new_value)
                    changed = True

        for field_name, new_value in updates.items():
            if getattr(self, field_name) != new_value:
                setattr(self, field_name, new_value)
                changed = True

        changed = self.normalize() or changed
        return changed


@dataclass
class GeneralPreference:
    """兜底维度 — 无法归入其他维度的通用偏好。"""
    preferences: list[str] = field(default_factory=list)
    confidence: float = 0.5
    keywords: list[str] = field(default_factory=lambda: [])
    notes: list[str] = field(default_factory=list)


@dataclass
class UserCoreProfile:
    """跨 intent 共享的稳定用户底色画像。"""
    user_id: str = ""
    core_persona: str = ""
    theme: ThemePreference = field(default_factory=ThemePreference)
    visual: VisualPreference = field(default_factory=VisualPreference)
    layout: LayoutPreference = field(default_factory=LayoutPreference)
    content: ContentPreference = field(default_factory=ContentPreference)
    general: GeneralPreference = field(default_factory=GeneralPreference)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    def get_dimension_map(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "visual": self.visual,
            "layout": self.layout,
            "content": self.content,
            "general": self.general,
        }

    def to_prompt_text(
        self,
        dimensions: set[str] | None = None,
        include_general: bool = True,
        include_persona: bool = True,
    ) -> str:
        include_all = dimensions is None
        sections: list[str] = []

        if include_persona and self.core_persona:
            sections.append("## Stable Persona\n- " + self.core_persona)

        dim_parts: list[str] = []
        if (include_all or "theme" in dimensions) and (
            self.theme.primary_colors or self.theme.font_family or self.theme.notes
        ):
            parts = []
            if self.theme.primary_colors:
                parts.append(f"颜色 {self.theme.primary_colors}")
            if self.theme.accent_colors:
                parts.append(f"强调色 {self.theme.accent_colors}")
            if self.theme.font_family:
                parts.append(f"字体 {self.theme.font_family}")
            if self.theme.background_style:
                parts.append(f"背景 {self.theme.background_style}")
            for note in self.theme.notes:
                parts.append(note)
            dim_parts.append(f"- **主题**: {'; '.join(parts)}")
        if (include_all or "visual" in dimensions) and (
            self.visual.image_style
            or self.visual.chart_type_priority
            or self.visual.icon_usage
            or self.visual.animation_preference
            or self.visual.notes
        ):
            parts = []
            if self.visual.image_style:
                parts.append(f"图片风格 {self.visual.image_style}")
            if self.visual.chart_type_priority:
                parts.append(f"图表偏好 {self.visual.chart_type_priority}")
            if self.visual.icon_usage:
                parts.append(f"图标使用 {self.visual.icon_usage}")
            if self.visual.animation_preference:
                parts.append(f"动画 {self.visual.animation_preference}")
            for note in self.visual.notes:
                parts.append(note)
            dim_parts.append(f"- **视觉**: {'; '.join(parts)}")
        if (include_all or "layout" in dimensions) and (
            self.layout.content_density
            or self.layout.alignment_style
            or self.layout.spacing_preference
            or self.layout.slide_structure
            or self.layout.notes
        ):
            parts = []
            if self.layout.content_density:
                parts.append(f"密度 {self.layout.content_density}")
            if self.layout.alignment_style:
                parts.append(f"对齐 {self.layout.alignment_style}")
            if self.layout.spacing_preference:
                parts.append(f"间距 {self.layout.spacing_preference}")
            if self.layout.slide_structure:
                parts.append(f"页面结构 {self.layout.slide_structure}")
            for note in self.layout.notes:
                parts.append(note)
            dim_parts.append(f"- **布局**: {'; '.join(parts)}")
        if (include_all or "content" in dimensions) and (
            self.content.text_density
            or self.content.language_style
            or self.content.bullet_point_style
            or self.content.title_length
            or self.content.notes
        ):
            parts = []
            if self.content.text_density:
                parts.append(f"文本密度 {self.content.text_density}")
            if self.content.language_style:
                parts.append(f"语言风格 {self.content.language_style}")
            if self.content.bullet_point_style:
                parts.append(f"要点风格 {self.content.bullet_point_style}")
            if self.content.title_length:
                parts.append(f"标题长度 {self.content.title_length}")
            for note in self.content.notes:
                parts.append(note)
            dim_parts.append(f"- **内容**: {'; '.join(parts)}")
        if dim_parts:
            sections.append("## Cross-Intent Stable Preferences\n" + "\n".join(dim_parts))

        if include_general and self.general.preferences:
            lines = ["## Cross-Intent General Preferences"]
            for pref in self.general.preferences:
                lines.append(f"- {pref}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _safe_construct(cls, data: dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_dict(cls, d: dict) -> "UserCoreProfile":
        profile = cls(
            user_id=d.get("user_id", ""),
            core_persona=d.get("core_persona", ""),
        )
        if "theme" in d and isinstance(d["theme"], dict):
            profile.theme = cls._safe_construct(ThemePreference, d["theme"])
        if "visual" in d and isinstance(d["visual"], dict):
            profile.visual = cls._safe_construct(VisualPreference, d["visual"])
        if "layout" in d and isinstance(d["layout"], dict):
            profile.layout = cls._safe_construct(LayoutPreference, d["layout"])
        if "content" in d and isinstance(d["content"], dict):
            profile.content = cls._safe_construct(ContentPreference, d["content"])
        if "general" in d and isinstance(d["general"], dict):
            profile.general = cls._safe_construct(GeneralPreference, d["general"])
        profile.last_updated = d.get("last_updated", profile.last_updated)
        profile.version = d.get("version", 1)
        return profile


@dataclass
class UserProfile:
    """用户偏好画像 — 按维度组织，LTM 偏好检索的唯一源。"""
    user_id: str = ""
    theme: ThemePreference = field(default_factory=ThemePreference)
    visual: VisualPreference = field(default_factory=VisualPreference)
    layout: LayoutPreference = field(default_factory=LayoutPreference)
    content: ContentPreference = field(default_factory=ContentPreference)
    template: TemplatePreference = field(default_factory=TemplatePreference)
    general: GeneralPreference = field(default_factory=GeneralPreference)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    def merge_from_temp_preferences(self, temp_prefs: list[TempPreference]) -> None:
        """根据 dimension 前缀分发到对应子结构。"""
        for pref in temp_prefs:
            dim = pref.dimension or ""
            if dim.startswith("theme."):
                self._update_field(self.theme, pref)
            elif dim.startswith("visual."):
                self._update_field(self.visual, pref)
            elif dim.startswith("layout."):
                self._update_field(self.layout, pref)
            elif dim.startswith("content."):
                self._update_field(self.content, pref)
            elif dim.startswith("template."):
                self._update_field(self.template, pref)
            else:
                # 兜底：无法映射维度前缀的偏好归入 general
                if pref.content not in self.general.preferences:
                    self.general.preferences.append(pref.content)

    def _update_field(self, sub_pref: Any, temp_pref: TempPreference) -> None:
        """简单字段写入（仅用于 merge_from_temp_preferences 兜底，正式晋升走 Consolidator）。"""
        field_name = temp_pref.dimension.split(".", 1)[1] if "." in temp_pref.dimension else ""
        if not field_name or not hasattr(sub_pref, field_name):
            return
        setattr(sub_pref, field_name, temp_pref.content)
        if hasattr(sub_pref, "notes") and temp_pref.content not in sub_pref.notes:
            sub_pref.notes.append(temp_pref.content)

    def get_dimension_map(self) -> dict[str, Any]:
        """返回维度名称 → 子结构的映射（含 general 兜底）。"""
        return {
            "theme": self.theme,
            "visual": self.visual,
            "layout": self.layout,
            "content": self.content,
            "template": self.template,
            "general": self.general,
        }

    def match_dimension(self, text: str) -> str:
        """根据关键词匹配最佳维度名称。返回维度名或 'general'。"""
        text_lower = text.lower()
        best_dim = "general"
        best_score = 0
        for dim_name, sub_pref in self.get_dimension_map().items():
            if dim_name == "general":
                continue
            kws = getattr(sub_pref, "keywords", [])
            score = sum(1 for kw in kws if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_dim = dim_name
        return best_dim

    def to_prompt_text(
        self,
        dimensions: set[str] | None = None,
        include_general: bool = True,
    ) -> str:
        """生成注入 Agent prompt 的偏好文本（结构化 + 通用约束）。

        Args:
            dimensions: 要包含的维度名称集合（如 {"theme", "layout"}）。
                        None 表示全量输出所有维度。
            include_general: 是否附带输出 general.preferences。
                             Agent 注入场景应为 True，逐维分析场景应为 False，
                             避免把 general 偏好混入每个 specific 维度摘要。
        """
        include_all = dimensions is None
        sections: list[str] = []
        # 各维度偏好（按 dimensions 过滤）
        dim_parts: list[str] = []
        if (include_all or "theme" in dimensions) and (
            self.theme.primary_colors or self.theme.font_family or self.theme.notes
        ):
            parts = []
            if self.theme.primary_colors:
                parts.append(f"颜色 {self.theme.primary_colors}")
            if self.theme.accent_colors:
                parts.append(f"强调色 {self.theme.accent_colors}")
            if self.theme.font_family:
                parts.append(f"字体 {self.theme.font_family}")
            if self.theme.background_style:
                parts.append(f"背景 {self.theme.background_style}")
            for note in self.theme.notes:
                parts.append(note)
            dim_parts.append(f"- **主题**: {'; '.join(parts)}")
        if (include_all or "visual" in dimensions) and (
            self.visual.image_style
            or self.visual.chart_type_priority
            or self.visual.icon_usage
            or self.visual.animation_preference
            or self.visual.notes
        ):
            parts = []
            if self.visual.image_style:
                parts.append(f"图片风格 {self.visual.image_style}")
            if self.visual.chart_type_priority:
                parts.append(f"图表偏好 {self.visual.chart_type_priority}")
            if self.visual.icon_usage:
                parts.append(f"图标使用 {self.visual.icon_usage}")
            if self.visual.animation_preference:
                parts.append(f"动画 {self.visual.animation_preference}")
            for note in self.visual.notes:
                parts.append(note)
            dim_parts.append(f"- **视觉**: {'; '.join(parts)}")
        if (include_all or "layout" in dimensions) and (
            self.layout.content_density
            or self.layout.alignment_style
            or self.layout.spacing_preference
            or self.layout.slide_structure
            or self.layout.notes
        ):
            parts = []
            if self.layout.content_density:
                parts.append(f"密度 {self.layout.content_density}")
            if self.layout.alignment_style:
                parts.append(f"对齐 {self.layout.alignment_style}")
            if self.layout.spacing_preference:
                parts.append(f"间距 {self.layout.spacing_preference}")
            if self.layout.slide_structure:
                parts.append(f"页面结构 {self.layout.slide_structure}")
            for note in self.layout.notes:
                parts.append(note)
            dim_parts.append(f"- **布局**: {'; '.join(parts)}")
        if (include_all or "content" in dimensions) and (
            self.content.text_density
            or self.content.language_style
            or self.content.bullet_point_style
            or self.content.title_length
            or self.content.notes
        ):
            parts = []
            if self.content.text_density:
                parts.append(f"文本密度 {self.content.text_density}")
            if self.content.language_style:
                parts.append(f"语言风格 {self.content.language_style}")
            if self.content.bullet_point_style:
                parts.append(f"要点风格 {self.content.bullet_point_style}")
            if self.content.title_length:
                parts.append(f"标题长度 {self.content.title_length}")
            for note in self.content.notes:
                parts.append(note)
            dim_parts.append(f"- **内容**: {'; '.join(parts)}")
        template_context = self.template.to_selection_context()
        if (include_all or "template" in dimensions) and template_context:
            parts = []
            if self.template.preferred_templates:
                parts.append(f"偏好模板 {', '.join(self.template.preferred_templates)}")
            if self.template.avoid_templates:
                parts.append(f"避免模板 {', '.join(self.template.avoid_templates)}")
            if self.template.history_preferred_templates:
                parts.append(
                    f"历史稳定模板 {', '.join(self.template.history_preferred_templates)}"
                )
            if self.template.last_successful_template_name:
                parts.append(f"最近成功模板 {self.template.last_successful_template_name}")
            if self.template.history_supported_aspect_ratios:
                parts.append(
                    f"历史适用比例 {', '.join(self.template.history_supported_aspect_ratios)}"
                )
            if self.template.history_reuse_scenarios:
                parts.append(
                    f"历史适用场景 {', '.join(self.template.history_reuse_scenarios[:3])}"
                )
            for note in self.template.notes:
                parts.append(note)
            dim_parts.append(f"- **模板**: {'; '.join(parts)}")
        if dim_parts:
            sections.append("## User Preferences\n" + "\n".join(dim_parts))
        # 通用偏好：在 Agent 注入时始终包含；逐维分析场景可显式关闭
        if include_general and self.general.preferences:
            lines = ["## General Preferences"]
            for p in self.general.preferences:
                lines.append(f"- {p}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _safe_construct(cls, data: dict):
        """从 dict 安全构建 dataclass 实例，只保留 cls 已知的字段。"""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        profile = cls(user_id=d.get("user_id", ""))
        if "theme" in d and isinstance(d["theme"], dict):
            profile.theme = cls._safe_construct(ThemePreference, d["theme"])
        if "visual" in d and isinstance(d["visual"], dict):
            profile.visual = cls._safe_construct(VisualPreference, d["visual"])
        if "layout" in d and isinstance(d["layout"], dict):
            profile.layout = cls._safe_construct(LayoutPreference, d["layout"])
        if "content" in d and isinstance(d["content"], dict):
            profile.content = cls._safe_construct(ContentPreference, d["content"])
        if "template" in d and isinstance(d["template"], dict):
            profile.template = cls._safe_construct(TemplatePreference, d["template"])
        if "general" in d and isinstance(d["general"], dict):
            profile.general = cls._safe_construct(GeneralPreference, d["general"])
        profile.last_updated = d.get("last_updated", profile.last_updated)
        profile.version = d.get("version", 1)
        return profile


@dataclass
class UserStateSnapshot:
    """运行时拼装后的共享用户状态视图。"""
    user_id: str = ""
    core_persona: str = ""
    task_intent: str = ""
    read_intent: str = ""
    write_intent: str = ""
    core_profile: UserCoreProfile = field(default_factory=UserCoreProfile)
    intent_profile: UserProfile = field(default_factory=UserProfile)
    session_preferences: list[str] = field(default_factory=list)
    cross_intent_hints: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "core_persona": self.core_persona,
            "task_intent": self.task_intent,
            "read_intent": self.read_intent,
            "write_intent": self.write_intent,
            "core_profile": self.core_profile.to_dict(),
            "intent_profile": self.intent_profile.to_dict(),
            "session_preferences": list(self.session_preferences),
            "cross_intent_hints": list(self.cross_intent_hints),
            "meta": dict(self.meta),
        }


# ═══════════════════════════════════════════════
# 模板使用记录 — TemplateUsageRecord
# ═══════════════════════════════════════════════


@dataclass
class TemplateUsageRecord:
    """模板使用历史记录 — 用于场景匹配的智能模板推荐

    记录用户在特定场景下使用模板的完整上下文，包括：
    - 用户原始请求（user_message）及其向量嵌入
    - 模板元数据（template_id, template_name）
    - 意图类型（explicit, implicit, anti）
    - 附件信息（has_attachment, attachment_type）
    - 成功标记（success，预留用于未来增强）

    通过向量相似度匹配历史场景，实现隐式模板推荐。
    """

    # ── 标识 ──
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    template_id: str = ""
    template_name: str = ""

    # ── 场景上下文 ──
    user_message: str = ""                              # 用户原始请求（截断至 500 字符）
    user_message_embedding: list[float] | None = None   # 用于相似度匹配的向量嵌入

    # ── 意图元数据 ──
    intent: str = "explicit"                            # "explicit" | "implicit" | "anti"
    memory_intent: str = ""                             # "academic" | "business" | ...
    aspect_ratio: str = ""                              # "16:9" | "4:3" | ...
    has_attachment: bool = False
    attachment_type: str = ""                           # "pptx" | "pdf" | "doc" | "image"

    # ── 结果追踪 ──
    success: bool = True                                # 预留用于未来增强（失败追踪）

    # ── 时间戳 ──
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，用于数据库存储"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "template_id": self.template_id,
            "template_name": self.template_name,
            "user_message": self.user_message,
            "user_message_embedding": self.user_message_embedding,
            "intent": self.intent,
            "memory_intent": self.memory_intent,
            "aspect_ratio": self.aspect_ratio,
            "has_attachment": self.has_attachment,
            "attachment_type": self.attachment_type,
            "success": self.success,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateUsageRecord":
        """从 dict 反序列化，仅保留 dataclass 已知字段"""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ═══════════════════════════════════════════════
# Intent-Based 用户画像
# ═══════════════════════════════════════════════

# 预定义 intent 类别及其关键词（用于规则匹配降级）
INTENT_CATEGORIES: dict[str, list[str]] = {
    "academic": [
        "学术", "论文", "研究", "paper", "research", "thesis", "学位",
        "答辩", "defense", "conference", "期刊", "journal", "摘要",
        "abstract", "文献", "literature", "实验", "experiment", "科研",
    ],
    "business": [
        "商务", "商业", "business", "公司", "company", "企业", "corporate",
        "季度", "quarter", "营收", "revenue", "市场", "market", "客户",
        "client", "投资", "invest", "融资", "pitch", "BP", "商业计划",
    ],
    "education": [
        "教学", "教育", "课件", "课程", "lecture", "教案", "lesson",
        "培训", "training", "workshop", "学生", "student", "教师",
        "teacher", "知识点", "考试", "exam",
    ],
    "creative": [
        "创意", "设计", "creative", "艺术", "art", "作品集", "portfolio",
        "展示", "showcase", "个人", "personal", "故事", "story",
    ],
    "report": [
        "报告", "汇报", "report", "总结", "summary", "周报", "月报",
        "年报", "annual", "进度", "progress", "复盘", "review", "OKR",
    ],
}

DEFAULT_INTENT = "default"
DEFAULT_INTENT_SCENARIO = "default_general"

INTENT_SCENARIOS: dict[str, list[str]] = {
    "academic": [
        "academic_report",
        "academic_defense",
        "academic_general",
    ],
    "business": [
        "business_pitch",
        "business_proposal",
        "business_report",
        "business_general",
    ],
    "education": [
        "education_courseware",
        "education_training",
        "education_general",
    ],
    "creative": [
        "creative_portfolio",
        "creative_showcase",
        "creative_general",
    ],
    "report": [
        "report_progress",
        "report_review",
        "report_general",
    ],
    DEFAULT_INTENT: [
        DEFAULT_INTENT_SCENARIO,
    ],
}

SCENARIO_TO_INTENT = {
    scenario: intent
    for intent, scenarios in INTENT_SCENARIOS.items()
    for scenario in scenarios
}

_SCENARIO_ALIASES = {
    "research_report": "academic_report",
    "paper_report": "academic_report",
    "thesis_defense": "academic_defense",
    "proposal_defense": "academic_defense",
    "business_plan": "business_proposal",
    "roadshow": "business_pitch",
    "quarterly_report": "business_report",
    "courseware": "education_courseware",
    "lesson": "education_courseware",
    "training": "education_training",
    "portfolio": "creative_portfolio",
    "showcase": "creative_showcase",
    "progress_report": "report_progress",
    "work_review": "report_review",
}

_INTENT_LABEL_PATTERN = re.compile(
    r"\b(" + "|".join(sorted([*INTENT_CATEGORIES.keys(), DEFAULT_INTENT], key=len, reverse=True)) + r")\b"
)
_SCENARIO_LABEL_PATTERN = re.compile(
    r"\b("
    + "|".join(
        sorted(
            [
                *SCENARIO_TO_INTENT.keys(),
                *list(_SCENARIO_ALIASES.keys()),
            ],
            key=len,
            reverse=True,
        )
    )
    + r")\b"
)
_EXPLICIT_INTENT_TOKEN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{1,119}$")


@dataclass
class IntentClassificationResult:
    intent: str = DEFAULT_INTENT
    scenario: str = DEFAULT_INTENT_SCENARIO
    confidence: float | None = None
    reasoning: str = ""
    raw_response: str = ""


def _normalize_explicit_intent_token(value: Any) -> str:
    """Normalize explicit role-intent / memory-bucket ids without collapsing them."""
    if isinstance(value, dict):
        for key in (
            "resolved_memory_read_intent",
            "memory_read_intent",
            "memory_bucket_id",
            "bucket_id",
            "profile_bucket_id",
            "resolved_memory_intent",
            "memory_intent",
            "resolved_memory_write_intent",
            "memory_write_intent",
            "write_intent",
            "read_intent",
            "resolved_task_intent",
            "task_intent",
            "role_intent_id",
            "role_intent",
            "intent",
            "primary_intent",
            "category",
        ):
            candidate = _normalize_explicit_intent_token(value.get(key, ""))
            if candidate:
                return candidate
        return ""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        try:
            return _normalize_explicit_intent_token(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            return ""

    normalized = text.replace("-", "_").replace(" ", "_").strip("_")
    if not normalized:
        return ""
    if normalized in INTENT_CATEGORIES or normalized == DEFAULT_INTENT:
        return normalized
    if normalized in SCENARIO_TO_INTENT or normalized in _SCENARIO_ALIASES:
        return ""
    if _EXPLICIT_INTENT_TOKEN_PATTERN.fullmatch(normalized):
        return normalized
    return ""


def normalize_intent_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "resolved_memory_read_intent",
            "memory_read_intent",
            "memory_bucket_id",
            "bucket_id",
            "profile_bucket_id",
            "resolved_memory_intent",
            "memory_intent",
            "resolved_memory_write_intent",
            "memory_write_intent",
            "write_intent",
            "read_intent",
            "resolved_task_intent",
            "task_intent",
            "role_intent_id",
            "role_intent",
            "intent",
            "primary_intent",
            "category",
        ):
            candidate = normalize_intent_label(value.get(key, ""))
            if candidate:
                return candidate
        scenario = normalize_scenario_label(value)
        if scenario:
            return SCENARIO_TO_INTENT.get(scenario, scenario or DEFAULT_INTENT)
        return ""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text in INTENT_CATEGORIES or text == DEFAULT_INTENT:
        return text
    if text in SCENARIO_TO_INTENT:
        return SCENARIO_TO_INTENT[text]

    normalized = text.replace("-", "_").replace(" ", "_")
    if normalized in SCENARIO_TO_INTENT:
        return SCENARIO_TO_INTENT[normalized]
    if normalized in _SCENARIO_ALIASES:
        return SCENARIO_TO_INTENT[_SCENARIO_ALIASES[normalized]]

    if text.startswith("{") and text.endswith("}"):
        try:
            return normalize_intent_label(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            pass

    explicit = _normalize_explicit_intent_token(text)
    if explicit:
        return explicit

    match = _INTENT_LABEL_PATTERN.search(text)
    if match:
        return match.group(1)

    scenario_match = _SCENARIO_LABEL_PATTERN.search(text.replace("-", "_").replace(" ", "_"))
    if scenario_match:
        scenario = normalize_scenario_label(scenario_match.group(1))
        if scenario:
            return SCENARIO_TO_INTENT.get(scenario, DEFAULT_INTENT)
    return ""


def normalize_scenario_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "scenario",
            "scenario_intent",
            "sub_intent",
            "secondary_intent",
            "scene",
            "resolved_task_intent",
            "task_intent",
            "role_intent_id",
            "role_intent",
            "memory_bucket_id",
            "bucket_id",
        ):
            candidate = normalize_scenario_label(value.get(key, ""))
            if candidate:
                return candidate
        return ""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("-", "_").replace(" ", "_")
    if normalized in SCENARIO_TO_INTENT:
        return normalized
    if normalized in _SCENARIO_ALIASES:
        return _SCENARIO_ALIASES[normalized]

    if text.startswith("{") and text.endswith("}"):
        try:
            return normalize_scenario_label(json.loads(text))
        except (TypeError, json.JSONDecodeError):
            pass

    explicit = _normalize_explicit_intent_token(text)
    if explicit:
        return explicit

    match = _SCENARIO_LABEL_PATTERN.search(normalized)
    if match:
        matched = match.group(1)
        return _SCENARIO_ALIASES.get(matched, matched)
    return ""


def infer_intent_scenario(text: str, intent: str = "") -> str:
    primary_intent = normalize_intent_label(intent) or classify_intent_by_keywords(text)
    if primary_intent and primary_intent not in INTENT_CATEGORIES and primary_intent != DEFAULT_INTENT:
        return primary_intent
    text_lower = str(text or "").lower()

    def _contains_any(keywords: list[str]) -> bool:
        return any(keyword.lower() in text_lower for keyword in keywords)

    if primary_intent == "academic":
        if _contains_any(["答辩", "defense", "开题", "毕业答辩", "中期答辩"]):
            return "academic_defense"
        if _contains_any(["研究报告", "学术报告", "论文汇报", "论文", "paper", "research", "实验结果"]):
            return "academic_report"
        return "academic_general"

    if primary_intent == "business":
        if _contains_any(["融资", "路演", "pitch", "投资人", "investor"]):
            return "business_pitch"
        if _contains_any(["提案", "proposal", "商业计划", "bp", "business plan"]):
            return "business_proposal"
        if _contains_any(["汇报", "季度", "营收", "revenue", "市场分析", "review"]):
            return "business_report"
        return "business_general"

    if primary_intent == "education":
        if _contains_any(["培训", "training", "workshop"]):
            return "education_training"
        if _contains_any(["课件", "课程", "lesson", "lecture", "教案"]):
            return "education_courseware"
        return "education_general"

    if primary_intent == "creative":
        if _contains_any(["作品集", "portfolio"]):
            return "creative_portfolio"
        if _contains_any(["展示", "showcase", "发布会", "故事", "story"]):
            return "creative_showcase"
        return "creative_general"

    if primary_intent == "report":
        if _contains_any(["进度", "周报", "月报", "季报", "年报", "okr"]):
            return "report_progress"
        if _contains_any(["复盘", "总结", "review", "retrospective"]):
            return "report_review"
        return "report_general"

    return DEFAULT_INTENT_SCENARIO


def build_intent_signal_text(
    user_message: str,
    *,
    attachment_names: list[str] | None = None,
    template_name: str = "",
    num_pages: str | None = None,
) -> str:
    parts = [str(user_message or "").strip()]
    if attachment_names:
        parts.append("附件: " + ", ".join(str(name).strip() for name in attachment_names if str(name).strip()))
    if template_name:
        parts.append(f"模板: {template_name}")
    if num_pages:
        parts.append(f"页数: {num_pages}")
    return "\n".join(part for part in parts if part).strip()


def build_intent_classification_prompt(
    user_message: str,
    *,
    attachment_names: list[str] | None = None,
    template_name: str = "",
    num_pages: str | None = None,
) -> str:
    context_lines: list[str] = []
    if attachment_names:
        context_lines.append("- 附件文件名: " + ", ".join(attachment_names[:5]))
    if template_name:
        context_lines.append(f"- 当前模板名: {template_name}")
    if num_pages:
        context_lines.append(f"- 目标页数: {num_pages}")
    if not context_lines:
        context_lines.append("- 无额外运行时上下文")

    return """你是 PPT 任务意图分类器。请根据用户请求判断一级意图，并尽量给出更细的二级场景。

## 一级意图（intent，必须从中选择一个）
- academic: 学术论文、研究报告、学术答辩、会议演讲
- business: 商业计划、企业汇报、投融资路演、市场分析
- education: 教学课件、培训材料、知识讲解、教案
- creative: 创意展示、艺术作品集、个人故事、设计展示
- report: 工作汇报、项目总结、进度报告、复盘分析
- default: 无法明确归类的通用场景

## 二级场景（scenario，优先选择最贴切的）
- academic_report: 论文汇报、研究报告、学术报告
- academic_defense: 开题答辩、中期答辩、毕业答辩、论文答辩
- academic_general
- business_pitch: 融资路演、投资人推介
- business_proposal: 商业计划、项目提案、方案提报
- business_report: 商务汇报、经营分析、季度复盘
- business_general
- education_courseware: 课程课件、课堂讲义、知识讲解
- education_training: 培训材料、Workshop
- education_general
- creative_portfolio: 作品集、案例集、艺术展示
- creative_showcase: 创意展示、品牌展示、故事表达
- creative_general
- report_progress: 周报、月报、进度汇报、OKR
- report_review: 复盘总结、项目总结
- report_general
- default_general

## 输出要求
1. 只输出一个 JSON 对象，不要输出 Markdown，不要解释。
2. JSON 格式必须为:
{{"intent":"academic","scenario":"academic_defense","confidence":0.96,"reasoning":"一句话理由"}}
3. confidence 取值 0 到 1。
4. 如果无法确定 scenario，请返回对应 intent 的 *_general。

## 用户请求
{user_message}

## 运行时上下文
{context_block}
""".format(
        user_message=str(user_message or "").strip()[:800],
        context_block="\n".join(context_lines),
    )


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    candidates = [raw]

    code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    if code_match:
        candidates.append(code_match.group(1))

    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _normalize_confidence(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def parse_intent_classification_response(
    response_text: str,
    *,
    request_text: str = "",
) -> IntentClassificationResult:
    payload = _extract_json_payload(response_text)
    intent = normalize_intent_label(payload) or normalize_intent_label(response_text)
    scenario = normalize_scenario_label(payload)
    if not scenario and intent:
        scenario = infer_intent_scenario(request_text or response_text, intent)
    if scenario and not intent:
        intent = SCENARIO_TO_INTENT.get(scenario, scenario or DEFAULT_INTENT)

    if not intent:
        intent = DEFAULT_INTENT
    if not scenario:
        scenario = infer_intent_scenario(request_text or response_text, intent)

    reasoning = str(
        payload.get("reasoning")
        or payload.get("rationale")
        or payload.get("reason")
        or ""
    ).strip()
    confidence = _normalize_confidence(payload.get("confidence"))
    return IntentClassificationResult(
        intent=intent,
        scenario=scenario or DEFAULT_INTENT_SCENARIO,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=str(response_text or ""),
    )


def _is_llm_signature_compatibility_error(exc: TypeError) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "unexpected keyword argument",
            "positional argument",
            "required positional argument",
        )
    )


def _get_callable_parameters(func: Any) -> list[inspect.Parameter]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [param for param in signature.parameters.values() if param.name != "self"]


def _extract_text_from_llm_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            text = _extract_text_from_llm_content(item)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts)
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)
    if isinstance(content, dict):
        for key in ("text", "content", "output_text", "value"):
            if key in content:
                text = _extract_text_from_llm_content(content.get(key))
                if text:
                    return text
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)
    for attr in ("text", "content", "output_text", "value"):
        if hasattr(content, attr):
            text = _extract_text_from_llm_content(getattr(content, attr))
            if text:
                return text
    return str(content)


def _extract_text_from_llm_response(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, (str, list, dict)):
        return _extract_text_from_llm_content(response)

    choices = getattr(response, "choices", None)
    if choices:
        try:
            message = choices[0].message
        except (AttributeError, IndexError, TypeError):
            message = None
        text = _extract_text_from_llm_content(message)
        if text:
            return text

    for attr in ("message", "output_text", "content", "text"):
        if hasattr(response, attr):
            text = _extract_text_from_llm_content(getattr(response, attr))
            if text:
                return text

    return str(response)


async def call_intent_classifier_llm(llm_client: Any, prompt: str) -> str:
    if callable(llm_client) and not hasattr(llm_client, "run") and not hasattr(llm_client, "chat"):
        return _extract_text_from_llm_response(await llm_client(prompt))

    messages = [{"role": "user", "content": prompt}]
    if hasattr(llm_client, "chat"):
        try:
            response = await llm_client.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=80,
            )
        except TypeError as exc:
            if not _is_llm_signature_compatibility_error(exc):
                raise
            logger.debug("Intent llm_client.chat fallback without sampling kwargs: %s", exc)
            response = await llm_client.chat(messages=messages)
        return _extract_text_from_llm_response(response)

    if hasattr(llm_client, "run"):
        run_params = _get_callable_parameters(llm_client.run)
        supports_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in run_params
        )
        supports_messages_kw = supports_var_keyword or any(
            param.name == "messages"
            for param in run_params
        )
        first_param_name = run_params[0].name if run_params else ""
        last_exc: TypeError | None = None
        run_attempts: list[Any] = []
        if supports_messages_kw:
            run_attempts.extend(
                [
                    lambda: llm_client.run(
                        messages=messages,
                        temperature=0.0,
                        max_tokens=80,
                    ),
                    lambda: llm_client.run(messages=messages),
                ]
            )
        if first_param_name in {"prompt", "message", "text", "input", "query"}:
            run_attempts.append(lambda: llm_client.run(prompt))
        else:
            run_attempts.extend(
                [
                    lambda: llm_client.run(messages),
                    lambda: llm_client.run(prompt),
                ]
            )

        for runner in run_attempts:
            try:
                response = await runner()
                return _extract_text_from_llm_response(response)
            except TypeError as exc:
                if not _is_llm_signature_compatibility_error(exc):
                    raise
                last_exc = exc
                logger.debug("Intent llm_client.run compatibility fallback: %s", exc)
        if last_exc is not None:
            raise last_exc

    if callable(llm_client):
        return _extract_text_from_llm_response(await llm_client(prompt))

    raise TypeError(f"Unsupported llm client for intent classification: {type(llm_client)!r}")


def classify_intent_by_keywords(text: str) -> str:
    """基于关键词的 intent 分类（规则降级，零 LLM 成本）。

    返回匹配度最高的 intent 类别名，无匹配时返回 "default"。
    """
    if not text:
        return DEFAULT_INTENT
    text_lower = text.lower()
    best_intent = DEFAULT_INTENT
    best_score = 0
    for intent_name, keywords in INTENT_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > best_score:
            best_score = score
            best_intent = intent_name
    # 至少匹配 1 个关键词才认为有效（"学术"、"商务"等核心词已足够明确）
    return best_intent if best_score >= 1 else DEFAULT_INTENT


async def classify_intent_with_llm(
    text: str,
    llm_client: Any = None,
    fallback_to_keywords: bool = True,
) -> str:
    """基于 LLM 的 intent 分类（主策略），关键词匹配作为降级。

    Args:
        text: 用户输入文本
        llm_client: LLM 客户端（需要有 chat() 方法）
        fallback_to_keywords: LLM 失败时是否降级到关键词匹配

    Returns:
        intent 类别名（academic/business/education/creative/report/default）
    """
    if not text:
        return DEFAULT_INTENT

    result = await classify_intent_details_with_llm(
        text,
        llm_client=llm_client,
        fallback_to_keywords=fallback_to_keywords,
    )
    return result.intent


async def classify_intent_details_with_llm(
    text: str,
    llm_client: Any = None,
    *,
    fallback_to_keywords: bool = True,
    attachment_names: list[str] | None = None,
    template_name: str = "",
    num_pages: str | None = None,
) -> IntentClassificationResult:
    if not text:
        return IntentClassificationResult()

    response_text = ""
    if llm_client:
        try:
            prompt = build_intent_classification_prompt(
                text,
                attachment_names=attachment_names,
                template_name=template_name,
                num_pages=num_pages,
            )
            response_text = await call_intent_classifier_llm(llm_client, prompt)
            result = parse_intent_classification_response(
                response_text,
                request_text=build_intent_signal_text(
                    text,
                    attachment_names=attachment_names,
                    template_name=template_name,
                    num_pages=num_pages,
                ),
            )
            if result.intent != DEFAULT_INTENT or not fallback_to_keywords:
                return result
            logger.debug(
                "LLM returned default intent for text=%r, falling back to keywords",
                text[:80],
            )
        except Exception as e:
            logger.debug("LLM intent classification failed: %s, falling back to keywords", e)

    if fallback_to_keywords:
        intent = classify_intent_by_keywords(text)
        return IntentClassificationResult(
            intent=intent,
            scenario=infer_intent_scenario(
                build_intent_signal_text(
                    text,
                    attachment_names=attachment_names,
                    template_name=template_name,
                    num_pages=num_pages,
                ),
                intent=intent,
            ),
            raw_response=response_text,
        )

    return IntentClassificationResult(raw_response=response_text)


INTENT_CLASSIFICATION_PROMPT = """你是用户意图分类器。根据用户的 PPT 制作请求，判断其所属的场景类别。

## 可选类别（必须从中选择一个）
- academic: 学术论文、研究报告、学术答辩、会议演讲
- business: 商业计划、企业汇报、投融资路演、市场分析
- education: 教学课件、培训材料、知识讲解、教案
- creative: 创意展示、艺术作品集、个人故事、设计展示
- report: 工作汇报、项目总结、进度报告、复盘分析
- default: 无法明确归类的通用场景

## 分类规则
1. 必须从上述 6 个类别中选择一个
2. 优先选择最匹配的具体类别（academic/business/education/creative/report）
3. 只有在完全无法判断时才选择 default
4. 只输出类别名（如 "academic"），不要输出任何其他内容

## 用户请求
{user_message}

请输出类别名："""


@dataclass
class IntentProfile:
    """Intent-Based 用户画像容器。

    层级结构: user_id → {intent → UserProfile}
    同一用户在不同场景（学术/商务/教育等）下可以有不同的偏好画像。
    "default" intent 作为兜底，当无法判断 intent 时使用。
    """
    user_id: str = ""
    profiles: dict[str, UserProfile] = field(default_factory=dict)

    def get_profile(self, intent: str = DEFAULT_INTENT) -> UserProfile:
        """获取指定 intent 的画像，不存在时创建空画像（不从 default 继承，避免偏好污染）。"""
        if intent in self.profiles:
            return self.profiles[intent]
        # 新 intent：返回空画像，由 ProfileInjectionRouter 按需注入
        profile = UserProfile(user_id=self.user_id)
        self.profiles[intent] = profile
        return profile

    def set_profile(self, intent: str, profile: UserProfile) -> None:
        """设置指定 intent 的画像。"""
        self.profiles[intent] = profile

    def list_intents(self) -> list[str]:
        """返回所有已有 intent 列表。"""
        return list(self.profiles.keys())

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "profiles": {
                intent: profile.to_dict()
                for intent, profile in self.profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentProfile":
        ip = cls(user_id=d.get("user_id", ""))
        for intent, profile_data in d.get("profiles", {}).items():
            ip.profiles[intent] = UserProfile.from_dict(profile_data)
        return ip


# ═══════════════════════════════════════════════
# 视觉指纹 — SlideVisualFingerprint
# ═══════════════════════════════════════════════


@dataclass
class SlideVisualFingerprint:
    """单张幻灯片的确定性视觉指纹（从 HTML/CSS 直接解析，零 LLM 成本）。

    这是 PPT 场景特有的数据模型 — 用于多模态偏好提取的 Layer 1。
    """

    # ── 颜色 ──
    colors_used: list[str] = field(default_factory=list)       # 所有非安全色的标准化颜色
    background_color: str = ""                                  # body 背景色
    background_type: str = ""                                   # "solid" | "gradient" | "image" | "none"

    # ── 字体 ──
    fonts_used: list[tuple[str, float]] = field(default_factory=list)  # [(font_name, font_size)]
    title_font: str = ""
    title_size: float = 0.0
    body_font: str = ""
    body_size: float = 0.0

    # ── 布局 ──
    element_counts: dict[str, int] = field(default_factory=dict)  # {text: N, image: N, ...}
    content_area_ratio: float = 0.0       # 内容面积 / 幻灯片面积
    image_text_ratio: float = 0.0         # 图片面积 / 文本面积
    has_bullet_points: bool = False
    bullet_point_count: int = 0

    # ── 文本密度 ──
    title_char_count: int = 0
    body_char_count: int = 0
    total_text_density: float = 0.0       # 总字符数 / 万像素

    # ── 元数据 ──
    slide_index: int = 0
    layout_name: str = ""
    html_path: str = ""
    is_finalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlideVisualFingerprint":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# 注：TaskSnapshot 已删除。指纹+多模态分析不再在 mid-job 构建快照，
# 而是在 Consolidation 时直接从最终 PPT HTML 文件提取。


# ═══════════════════════════════════════════════
# 便捷导出
# ═══════════════════════════════════════════════

__all__ = [
    "Message",
    "ExperienceTrace",
    "infer_experience_type",
    "infer_task_experience_category",
    "normalize_experience_type",
    "default_outcome_for_experience_type",
    "RoundExperience",
    "TempExperienceTrace",  # 向后兼容别名
    "DesignEpisode",
    "AtomicPreference",
    "VALID_DIMENSIONS",
    "normalize_dimension",
    "looks_like_template_related_text",
    "has_stable_template_preference_signal",
    "is_transient_template_instruction",
    "is_template_related_preference",
    "Operation",
    "Round",
    "Job",
    "TempPreference",
    "ToolChain",
    "ChainExperience",
    "RoundSummary",
    "OperationMemory",
    "ThemePreference",
    "VisualPreference",
    "LayoutPreference",
    "ContentPreference",
    "TemplatePreference",
    "GeneralPreference",
    "UserCoreProfile",
    "UserProfile",
    "UserStateSnapshot",
    "TemplateUsageRecord",
    "SlideVisualFingerprint",
]
logger = logging.getLogger(__name__)
