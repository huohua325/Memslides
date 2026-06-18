"""Experience Prompts — 经验轨迹提取、工具错误合并"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# 经验轨迹提取 (每轮 modify)
# ══════════════════════════════════════════════════════════════════════════════

TURN_EXPERIENCE_PROMPT = """你是PPT设计助手的工具调用经验分析器。
分析以下一轮工具调用的完整序列，从中提炼出有价值的可复用经验。

请将经验统一归类到以下 4 类：
- hard_errors: 工具硬错误、直接失败、参数或输入不满足工具要求
- tool_misuse: 工具角色误用、流程误用、软失败
- tool_limitations: 工具盲区、能力边界、观测缺口
- effective_patterns: 有效工作模式、可复用 pipeline

用户任务: {user_task}

工具调用序列（按时间顺序）:
{tool_sequence}

最终结果: {outcome}

输出 JSON（不要任何其他文字）：
{{
  "hard_errors": [
    {{"tool": "工具名", "error_type": "失败类型", "description": "失败原因（1句话）", "lesson": "避免该失败的教训（1句话）"}}
  ],
  "tool_misuse": [
    {{"tool": "工具名", "misuse_type": "误用类型", "description": "具体描述（1句话）", "lesson": "正确的使用方式（1句话）"}}
  ],
  "tool_limitations": [
    {{"tool": "工具名", "limitation": "工具的具体盲区或能力边界（1句话）", "workaround": "绕过方法（1句话）"}}
  ],
  "effective_patterns": [
    {{"task_type": "任务类型标签", "pipeline": ["tool1","tool2","tool3"], "lesson": "此编排的关键要点（1句话）", "applicable_when": "适用场景"}}
  ]
}}

提取规则：
- hard_errors: 只有当失败具备可复用价值时才填写，例如参数格式错误、输入前置条件不满足、工具明确报错。不要复述整段报错，要提炼成可执行教训。
- tool_misuse: 从整体工具链中发现的工具角色或使用边界问题，包括但不限于：
  · 把单向执行工具（write_html_file）当作交互工具使用（在其中写入疑问、等待用户反馈）
  · 对健康已有页绕过 `read_slide_snapshot` / `apply_slide_patch`，直接 `write_html_file` 整页重写
  · 忽略快照中的 `repair_candidates` / `rules`，盲目只改正文或单个文本节点
  · 收到 `STALE_SNAPSHOT` 后不使用 fresh snapshot / `rebind_hints` 重绑，而是直接整页重写
  · 反复调用 inspect/read 但不做任何修改决定（应自主判断后行动）
  · 在 finalize 之后继续写入（已完成的流程不应再操作）
  · 任何体现出"模型不理解工具设计意图"的使用方式
  注意：不要基于"调用次数多"来判断误用——复杂任务本身需要大量工具调用。
  关注的是"工具的使用方式是否违反其设计意图"。
- tool_limitations: 发现工具有盲区或能力限制（如 inspect_slide 不报告 CSS 背景图）才填写
- effective_patterns: 完成复杂任务的多工具编排流程（涉及3种以上不同工具），例如：
  · 用户要求单页溢出修复 → read_slide_snapshot → apply_slide_patch → inspect_slide
  · 用户要求批量统一标题颜色 → scan_slide_index → batch_update_css_rule → inspect_slide
  · 用户要求批量修改后修补例外页 → batch_update_semantic_style → read_slide_snapshot → apply_slide_patch → inspect_slide
  注意：只有场景特定的完整编排才值得记录，不要记录通用的"先读后写后验证"。
- 没有发现则返回空数组，不要强行填充
"""


# ══════════════════════════════════════════════════════════════════════════════
# 工具错误合并 (工具重复失败时)
# ══════════════════════════════════════════════════════════════════════════════

MERGE_TOOL_ERROR_PROMPT = """你是一个经验合并助手。以下是同一工具多次失败的错误记录，请合并成1-3条简洁的教训（每条不超过80字）。
重点关注：失败的根本原因、触发条件、如何避免。去掉重复内容，保留最有价值的信息。

工具名称: {tool_name}
已有教训:
{existing_lessons}
新错误:
{new_error}
新参数: {new_args}

请直接输出合并后的教训文本（不要JSON），用换行分隔多条。"""
