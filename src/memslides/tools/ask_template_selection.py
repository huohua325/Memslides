"""ask_template_selection 工具 - 专用模板选择

Stage 9: 当用户未指定模板时，向用户展示推荐的模板列表，让用户选择。

限制：
- 仅用于模板选择，不能用于其他澄清
- 每个 run (PPT生成任务) 最多调用一次
- 选项固定包含"不使用模板"选项
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class TemplateOption:
    """单个模板选项"""
    template_id: str = ""
    template_name: str = ""
    recommendation_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "recommendation_reason": self.recommendation_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateOption":
        return cls(
            template_id=d.get("template_id", ""),
            template_name=d.get("template_name", ""),
            recommendation_reason=d.get("recommendation_reason", ""),
        )


@dataclass
class TemplateSelectionRequest:
    """模板选择请求"""
    id: str = field(default_factory=lambda: f"tpl_sel_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    question: str = ""
    template_options: list[TemplateOption] = field(default_factory=list)
    allow_no_template: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "template_options": [o.to_dict() for o in self.template_options],
            "allow_no_template": self.allow_no_template,
            "created_at": self.created_at,
        }

    @classmethod
    def from_tool_call(cls, arguments: dict) -> "TemplateSelectionRequest":
        """从工具调用参数创建请求"""
        options = [
            TemplateOption.from_dict(o)
            for o in arguments.get("template_options", [])
        ]
        return cls(
            question=arguments.get("question", ""),
            template_options=options,
            allow_no_template=arguments.get("allow_no_template", True),
        )

    def format_display_options(self) -> list[str]:
        """格式化为用户可见的选项列表"""
        display = []
        for i, opt in enumerate(self.template_options):
            label = chr(65 + i)  # A, B, C, ...
            reason = f" - {opt.recommendation_reason}" if opt.recommendation_reason else ""
            display.append(f"{label}. {opt.template_name}{reason}")
        if self.allow_no_template:
            label = chr(65 + len(self.template_options))
            display.append(f"{label}. 不使用模板，从头设计")
        return display


@dataclass
class TemplateSelectionResponse:
    """模板选择响应"""
    request_id: str = ""
    selected_template_id: str = ""  # 空字符串表示"不使用模板"
    selected_template_name: str = ""
    selected_option_index: Optional[int] = None
    responded_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "selected_template_id": self.selected_template_id,
            "selected_template_name": self.selected_template_name,
            "selected_option_index": self.selected_option_index,
            "responded_at": self.responded_at,
        }


# ═══════════════════════════════════════════════
# 工具 Schema（供 AgentEnv 注册）
# ═══════════════════════════════════════════════

ASK_TEMPLATE_SELECTION_SCHEMA = {
    "name": "ask_template_selection",
    "description": """向用户展示推荐的模板列表，让用户选择使用哪个模板。

仅在以下场景使用：
- 用户未指定模板，但系统检测到有适合的模板

限制：
- 每个 PPT 生成任务最多调用一次，重复调用将返回错误
- 必须包含"不使用模板"选项
- 用户选择后直接执行，不再确认""",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "向用户提出的选择问题，如'检测到以下模板适合您的任务，请选择：'"
            },
            "template_options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "template_id": {"type": "string"},
                        "template_name": {"type": "string"},
                        "recommendation_reason": {"type": "string"}
                    },
                    "required": ["template_id", "template_name"]
                },
                "description": "推荐的模板列表 (最多3个)"
            },
            "allow_no_template": {
                "type": "boolean",
                "default": True,
                "description": "是否允许用户选择'不使用模板'，默认为 true"
            }
        },
        "required": ["question", "template_options"]
    }
}


# ═══════════════════════════════════════════════
# 工具调用检测
# ═══════════════════════════════════════════════

TOOL_NAME = "ask_template_selection"


def is_template_selection_tool_call(tool_call: Any) -> bool:
    """检测是否为 ask_template_selection 工具调用"""
    if hasattr(tool_call, 'function'):
        return getattr(tool_call.function, 'name', '') == TOOL_NAME
    if isinstance(tool_call, dict):
        func = tool_call.get('function', {})
        return func.get('name', '') == TOOL_NAME
    return False


def parse_template_selection_arguments(tool_call: Any) -> dict:
    """解析工具调用参数"""
    if hasattr(tool_call, 'function'):
        args_str = getattr(tool_call.function, 'arguments', '{}')
    elif isinstance(tool_call, dict):
        args_str = tool_call.get('function', {}).get('arguments', '{}')
    else:
        return {}

    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse template selection arguments: %s", args_str)
        return {}


def format_tool_response(
    response: TemplateSelectionResponse,
    request: TemplateSelectionRequest | None = None,
) -> str:
    """格式化工具响应，返回给 Agent"""
    parts = []

    if request:
        parts.append(f"【问题】{request.question}")

    if response.selected_template_id:
        parts.append(f"【用户选择模板】{response.selected_template_name} (ID: {response.selected_template_id})")
        parts.append("使用该模板进行设计。不要再次提问或要求确认。")
    else:
        parts.append("【用户选择】不使用模板，从头设计。")
        parts.append("直接开始设计，不要再次提问或要求确认。")

    return "\n".join(parts)


# ═══════════════════════════════════════════════
# 模板选择管理器
# ═══════════════════════════════════════════════

class TemplateSelectionManager:
    """管理模板选择工具的调用状态

    限制每个 run (PPT生成任务) 最多调用一次。
    """

    def __init__(self):
        self._pending_request: TemplateSelectionRequest | None = None
        self._completed: tuple[TemplateSelectionRequest, TemplateSelectionResponse] | None = None
        self._used: bool = False  # 当前 run 是否已使用过

    def can_call(self) -> bool:
        """检查当前 run 是否还能调用此工具"""
        return not self._used

    def create_request(
        self,
        question: str,
        template_options: list[TemplateOption],
        allow_no_template: bool = True,
    ) -> TemplateSelectionRequest | None:
        """创建模板选择请求

        Returns:
            请求对象，如果已使用过则返回 None
        """
        if self._used:
            logger.warning("ask_template_selection already used in this run")
            return None

        request = TemplateSelectionRequest(
            question=question,
            template_options=template_options,
            allow_no_template=allow_no_template,
        )
        self._pending_request = request
        self._used = True
        logger.info("Created template selection request: %s", request.id)
        return request

    def submit_response(
        self,
        selected_index: int,
    ) -> TemplateSelectionResponse | None:
        """提交用户选择

        Args:
            selected_index: 用户选择的选项索引 (0-indexed)

        Returns:
            响应对象
        """
        if not self._pending_request:
            logger.warning("No pending template selection request")
            return None

        request = self._pending_request
        options = request.template_options

        # 判断是否选择了"不使用模板"
        if selected_index >= len(options):
            response = TemplateSelectionResponse(
                request_id=request.id,
                selected_template_id="",
                selected_template_name="",
                selected_option_index=selected_index,
            )
        else:
            selected = options[selected_index]
            response = TemplateSelectionResponse(
                request_id=request.id,
                selected_template_id=selected.template_id,
                selected_template_name=selected.template_name,
                selected_option_index=selected_index,
            )

        self._completed = (request, response)
        self._pending_request = None
        logger.info(
            "Template selection completed: %s (template_id=%s)",
            request.id,
            response.selected_template_id or "none",
        )
        return response

    def get_pending_request(self) -> TemplateSelectionRequest | None:
        """获取待处理的请求"""
        return self._pending_request

    def get_completed(self) -> tuple[TemplateSelectionRequest, TemplateSelectionResponse] | None:
        """获取已完成的请求-响应对"""
        return self._completed

    def get_selected_template_id(self) -> str:
        """获取当前 run 选择的模板 ID（空字符串表示未选择或不使用模板）"""
        if self._completed:
            return self._completed[1].selected_template_id
        return ""

    def reset(self) -> None:
        """重置状态 — 新的 run() 调用时重置"""
        self._pending_request = None
        self._completed = None
        self._used = False
