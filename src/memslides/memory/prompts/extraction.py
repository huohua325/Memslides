"""Extraction Prompts — 意图分类、Episode提取、工具学习、用户消息分析、原子偏好提取"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# 意图分类 (每条用户消息)
# ══════════════════════════════════════════════════════════════════════════════

def _format_context(messages: list[dict] | None) -> str:
    if not messages:
        return "无历史上下文"
    lines = []
    for m in messages[-5:]:
        role = m.get("role", "unknown")
        content = m.get("content", "")[:200]
        lines.append(f"[{role}] {content}")
    return "最近对话：\n" + "\n".join(lines)


def build_intent_classification_prompt(
    user_message: str,
    context_messages: list[dict] | None = None,
    current_slide: str = "",
) -> str:
    """构建意图分类Prompt

    Args:
        user_message: 用户当前消息
        context_messages: 最近N条对话历史
        current_slide: 当前焦点幻灯片

    返回的prompt要求LLM输出JSON：
    {
        "intent_type": "modify_style|modify_content|modify_layout|query|confirm|reject|undo|export|chat",
        "confidence": 0.0-1.0,
        "target_slide": "slide_3" | "all" | "",
        "modification_description": "用户想要做什么的简要描述"
    }
    """
    return f"""你是一个PPT修改意图分类器。请分析用户消息的意图。

## 上下文
当前焦点幻灯片: {current_slide or '未指定'}
{_format_context(context_messages)}

## 用户消息
{user_message}

## 意图类型
- modify_style: 修改样式（颜色、字号、字体、背景等）
- modify_content: 修改内容（文字、图片替换等）
- modify_layout: 修改布局（位置、大小、对齐等）
- query: 询问/查看（"第3页是什么内容"）
- confirm: 确认/满意（"好的"、"可以"）
- reject: 不满意/否定（"不行"、"不是这样"）
- undo: 撤销/回滚（"改回去"、"撤销"）
- export: 导出请求
- chat: 闲聊/其他

## target_slide 判定规则
判断用户想修改哪些幻灯片。请根据语义理解判断，不要依赖关键词匹配。

输出 "all" 的情况：
- 用户明确说了"所有"、"全部"、"每一页"、"all slides"等
- 用户请求的是**全局样式修改**（如颜色、配色方案、字体、背景风格）且**没有指定具体页码**。
  这类修改天然具有全局性，用户期望整套幻灯片风格统一。
  例如："把颜色改成蓝色"、"换个字体"、"背景改成深色"→ 输出 "all"

输出 "slide_N" 的情况：
- 用户明确指定了页码（如"第3页"、"slide 5"、"封面"）
- 用户请求的是**内容修改**（如改文字、换图片、调整某个元素），即使没指定页码，
  也应根据上下文推断最可能的目标页，输出具体页码

输出 "" 的情况：
- 无法判断目标（如闲聊、确认、查询类消息）

请输出JSON（不要输出其他内容）：
{{
    "intent_type": "...",
    "confidence": 0.0-1.0,
    "target_slide": "slide_N" | "all" | "",
    "modification_description": "..."
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# Episode 提取 (每批修改)
# ══════════════════════════════════════════════════════════════════════════════

EPISODE_EXTRACTION_PROMPT = """You are analyzing PPT modification interactions to extract episodes \
that reveal the user's UNOBSERVABLE mental model.

## Input: Compressed modification rounds
{batch_text}

## What to extract (per round)
1. **user_intent** — TRUE intent beyond literal words
2. **interpretation_gap** — Mismatch between intent and action ("none" if correct)
3. **action_outcome** — What changed (brief)
4. **design_insight** — Reusable knowledge about this user's design thinking

## Hard rules
- NEVER extract observable PPT state (colors, sizes, positions, content)
- Every episode MUST have a non-empty design_insight
- Focus on WHY, not WHAT
- If a round involves bulk operations across many slides, extract the user's UNDERLYING PREFERENCE \
that motivated the bulk change (e.g. "User prioritizes visual density over whitespace"), \
NOT what was changed (never list which slides were modified or what CSS values were set)

## Output
Return a JSON object:
{{"episodes": [{{"source_round_id": N, "user_intent": "...", "interpretation_gap": "...", \
"action_outcome": "...", "design_insight": "...", "category": "color|typography|layout|content|workflow"}}]}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 用户消息分析 (每条用户消息)
# 注：以下 prompt 属于早期 user_message_analyzer 方案遗留，目前 modify
# 阶段主链已改为 session_preference -> on_user_preference -> WM.add_preference。
# 这里暂时保留定义，仅用于兼容历史代码/实验回溯，不作为当前主路径。
# ══════════════════════════════════════════════════════════════════════════════

USER_MESSAGE_SYSTEM_CONTEXT = (
    "You are a memory extractor for a PPT design assistant (MemSlides). "
    "The assistant helps users create and modify PowerPoint presentations through "
    "natural language instructions."
)

USER_MESSAGE_CLASSIFY_PROMPT = """{system_context}

Analyze the following user message and determine if it contains any PERSISTENT constraints
that should be remembered across multiple modification sessions.

User message: "{message}"

Distinguish between:
1. Persistent constraints: rules that should ALWAYS apply (e.g. "always use Microsoft YaHei font",
   "company policy: blue theme only", "never change background images")
2. Persistent preferences: style preferences that apply broadly (e.g. "I prefer minimalist style",
   "I like dark backgrounds")
3. One-time actions: single modification requests (e.g. "make the title bigger", "change slide 3")
   → do NOT extract these

Output JSON only:
{{"has_constraint": true/false, "constraint": "normalized constraint description or empty string",
  "has_preference": true/false, "preference": {{"content": "...", "category": "color|font|layout|style|other"}} or null}}

If neither constraint nor preference found, set both has_constraint and has_preference to false.
Do not invent information not present in the message."""


# ══════════════════════════════════════════════════════════════════════════════
# 原子偏好提取 (从 Episode)
# ══════════════════════════════════════════════════════════════════════════════

ATOMIC_PREFERENCE_EXTRACTION_PROMPT = """你是一个用户偏好分析专家。根据以下设计交互记录，提取用户的原子级偏好。

## 输入 Episode
{episode_content}

## 任务
从这个 Episode 中提取用户的偏好。每个偏好应该是独立的、原子的。

## 输出格式
返回 JSON 数组，每个元素包含:
- preference_type: "value"(价值观偏好) 或 "strategy"(操作策略)
- trigger: 何时适用这个偏好（条件描述）
- preference: 用户想要什么（简洁描述）
- rationale: 为什么得出这个偏好（基于 Episode 的解释）
- scope: "global"(全局) / "slide_type"(幻灯片类型) / "element_type"(元素类型)
- scope_value: scope 的具体值（如果 scope 不是 global）
- conflict_group: 冲突组标识（同类偏好应该有相同的 conflict_group）
- confidence: 置信度 0.0-1.0

## 示例输出
```json
[
  {{
    "preference_type": "value",
    "trigger": "设计任何幻灯片时",
    "preference": "偏好简洁的设计风格，避免元素过多",
    "rationale": "用户在 Episode 中明确表示不喜欢复杂的布局",
    "scope": "global",
    "scope_value": "",
    "conflict_group": "design_complexity",
    "confidence": 0.7
  }}
]
```

## 注意事项
1. 只提取有明确证据支持的偏好，不要臆测
2. 如果 Episode 中没有明显的偏好信号，返回空数组 []
3. 同一个 Episode 可能包含多个独立偏好
4. preference 应该简洁（<50字），trigger 描述触发条件

请分析并输出 JSON:
"""


ATOMIC_PREFERENCE_FEEDBACK_PROMPT = """分析用户消息，提取明确表达的设计偏好。

用户消息: {user_message}
Agent响应: {agent_response}
满意度: {satisfaction}

如果用户明确表达了偏好（如"我喜欢..."、"不要..."、"以后都..."），提取为 JSON:
```json
[{{"preference_type": "value或strategy", "trigger": "触发条件", "preference": "偏好内容",
   "scope": "global 或 slide_type 或 element_type",
   "scope_value": "scope具体值如title/text/image；scope=global时留空",
   "confidence": 0.6}}]
```

## scope 判断规则
- scope=global: 适用于所有场景（如"我喜欢简洁风格"）
- scope=slide_type + scope_value=title/content/agenda/cover: 仅适用于特定幻灯片类型
- scope=element_type + scope_value=text/image/chart/table: 仅适用于特定元素类型

如果没有明确偏好，返回空数组 []
"""
