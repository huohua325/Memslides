"""ask_user_clarification 工具 - 实现 Pre-action Feedback

当 Agent 遇到模糊指令或多种设计选择时，主动向用户提问。
用户回答后，立刻提取为 AtomicPreference 并存储。

Stage 5: 交互范式演进 - PAHF (Personalized Agents from Human Feedback)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClarificationRequest:
    """用户澄清请求"""
    id: str = field(default_factory=lambda: f"clarify_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    question: str = ""
    options: list[str] = field(default_factory=list)
    context: str = ""  # 触发询问的上下文
    agent_type: str = ""  # "research" / "design" / "modify"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "context": self.context,
            "agent_type": self.agent_type,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_tool_call(cls, arguments: dict, agent_type: str = "", context: str = "") -> "ClarificationRequest":
        """从工具调用参数创建请求"""
        return cls(
            question=arguments.get("question", ""),
            options=arguments.get("options", []),
            context=context,
            agent_type=agent_type,
        )


@dataclass
class ClarificationResponse:
    """用户澄清回复"""
    request_id: str = ""
    answer: str = ""
    selected_option: Optional[int] = None  # 如果是选项，记录选择的索引 (0-indexed)
    responded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "selected_option": self.selected_option,
            "responded_at": self.responded_at,
        }


# ═══════════════════════════════════════════════
# 工具定义（供 AgentEnv 注册）
# ═══════════════════════════════════════════════

ASK_CLARIFICATION_TOOL_SCHEMA = {
    "name": "ask_user_clarification",
    "description": """当用户指令模糊或存在多种设计选择时，主动向用户提问以获取澄清。

使用场景：
- 用户说"专业点"但未指定风格（商务/学术/简约）
- 配色选择不明确（深色/浅色/品牌色）
- 布局选择不确定（单栏/双栏/全图）
- 模板约束与用户指令存在潜在冲突

注意：
- 仅在真正需要澄清时使用，避免过度打扰用户
- 优先使用已知的用户偏好（如有）
- 问题应简洁明确，选项应互斥且全面""",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "向用户提出的具体问题，应简洁明确"
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选的选项列表（2-4个），格式如 ['A. 深色商务风', 'B. 浅色简约风']。如果是开放式问题可省略。"
            }
        },
        "required": ["question"]
    }
}


def format_tool_response(
    response: ClarificationResponse,
    request: "ClarificationRequest | None" = None,
) -> str:
    """格式化工具响应，返回给 Agent
    
    返回问题 + 用户选择，但不回显全部选项列表（防止模型重复确认）。
    """
    parts = []
    
    # 包含原始问题作为上下文，但不列出所有选项
    if request:
        parts.append(f"【问题】{request.question}")
    
    # 用户的回答/选择
    if response.selected_option is not None:
        parts.append(f"【用户确认】{response.answer}")
    else:
        parts.append(f"【用户回答】{response.answer}")
    
    parts.append("立即根据用户选择执行操作。不要再次提问或要求确认。")
    
    return "\n".join(parts)


# ═══════════════════════════════════════════════
# 工具调用检测
# ═══════════════════════════════════════════════

def is_clarification_tool_call(tool_call: Any) -> bool:
    """检测是否为 ask_user_clarification 工具调用"""
    if hasattr(tool_call, 'function'):
        return getattr(tool_call.function, 'name', '') == 'ask_user_clarification'
    if isinstance(tool_call, dict):
        func = tool_call.get('function', {})
        return func.get('name', '') == 'ask_user_clarification'
    return False


def parse_clarification_arguments(tool_call: Any) -> dict:
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
        logger.warning(f"Failed to parse clarification arguments: {args_str}")
        return {}


# ═══════════════════════════════════════════════
# 澄清会话管理器
# ═══════════════════════════════════════════════

class ClarificationManager:
    """管理澄清请求和响应的生命周期"""
    
    def __init__(self):
        self._pending_requests: dict[str, ClarificationRequest] = {}
        self._completed_requests: dict[str, tuple[ClarificationRequest, ClarificationResponse]] = {}
    
    def create_request(
        self,
        question: str,
        options: list[str] = None,
        context: str = "",
        agent_type: str = "",
    ) -> ClarificationRequest:
        """创建新的澄清请求"""
        request = ClarificationRequest(
            question=question,
            options=options or [],
            context=context,
            agent_type=agent_type,
        )
        self._pending_requests[request.id] = request
        logger.info(f"Created clarification request: {request.id}")
        return request
    
    def submit_response(
        self,
        request_id: str,
        answer: str,
        selected_option: int = None,
    ) -> ClarificationResponse | None:
        """提交用户响应"""
        request = self._pending_requests.pop(request_id, None)
        if not request:
            logger.warning(f"Clarification request not found: {request_id}")
            return None
        
        response = ClarificationResponse(
            request_id=request_id,
            answer=answer,
            selected_option=selected_option,
        )
        
        self._completed_requests[request_id] = (request, response)
        logger.info(f"Submitted clarification response: {request_id}")
        return response
    
    def get_pending_request(self, request_id: str) -> ClarificationRequest | None:
        """获取待处理的请求"""
        return self._pending_requests.get(request_id)
    
    def get_completed(self, request_id: str) -> tuple[ClarificationRequest, ClarificationResponse] | None:
        """获取已完成的请求-响应对"""
        return self._completed_requests.get(request_id)
    
    def get_recent_completed(self, limit: int = 5) -> list[tuple[ClarificationRequest, ClarificationResponse]]:
        """获取最近完成的请求-响应对（用于记忆提取）"""
        items = list(self._completed_requests.values())
        return items[-limit:] if len(items) > limit else items
