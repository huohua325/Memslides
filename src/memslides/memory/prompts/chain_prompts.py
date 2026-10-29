"""LLM Prompt 模板 — 工具链分割、经验提取、Task 摘要

Design Doc §6.1.1, §5.7, §8.8
"""

TASK_SUMMARY_PROMPT = """你是一个幻灯片编辑助手的记忆模块。请根据以下一轮对话的执行记录，生成结构化摘要。

## 用户请求
{user_message}

## 工具调用记录（共 {tool_count} 次）
{tool_calls_summary}

## Agent 最终回复
{agent_response}

请输出 JSON 格式：
{{
  "key_actions": ["最多5条关键操作描述"],
  "unresolved_issues": ["未解决的问题（如有）"],
  "slides_affected": ["受影响的幻灯片ID列表"],
  "pattern_observed": "观察到的操作模式（如 inspect-fix 循环）"
}}
"""

CHAIN_SEGMENTATION_PROMPT = """你是PPT设计助手的工具调用分析器。
将以下一轮对话中的工具调用序列划分为若干"工具链"。
每条工具链代表一个完整的子任务（如"修改某张幻灯片的标题"、"插入一张图片"）。

用户任务: {user_task}

工具调用序列（共 {cycle_count} 个 cycle）:
{cycle_summaries}

划分规则：
1. 围绕同一目标（如同一张幻灯片、同一个文件）的连续操作归为一条链
2. thinking/todo 类工具视为下一条链的开始
3. finalize 单独成链
4. write → inspect(问题) → write 的模式，标记 outcome 为 "retry"
5. 无明确目标的工具归入最近的链
6. semantic_label 简洁描述子任务（人类可读描述）
7. Reasoning 字段包含 Agent 的决策逻辑，可帮助判断子任务边界
8. Observation 字段包含工具返回结果（文件内容、错误信息、渲染状态等），可帮助判断操作是否成功以及后续操作的动机

输出 JSON 数组：
[{{"semantic_label": "子任务描述", "cycle_indices": [0, 1, 2], "outcome": "success"}}]

outcome: "success" | "retry" | "partial"
"""

CHAIN_EXPERIENCE_PROMPT = """你是PPT设计助手的工具链经验分析器。
以下是同一类子任务（"{chain_name}"）的 {chain_count} 次执行记录，请提炼可复用的经验。

{existing_experience_section}

执行记录：
{chains_text}

输出 JSON：
{{"experiences": [{{
  "tool_pipeline": ["tool1", "tool2"],
  "lesson": "关键经验（1-2句话）",
  "applicable_when": "适用场景",
  "anti_pattern": "应避免的模式（来自 retry 链，否则为空）"
}}]}}

规则：
- 对比多条链找共性模式和差异
- lesson 应具体可操作
- anti_pattern 只从 outcome="retry" 的链中提取
- 没有有价值的经验则返回空数组
"""

PROFILE_UPDATE_PROMPT = """你是用户偏好画像管理器。请将以下新偏好合并到目标 intent={target_intent} 的「{dimension_name}」维度现有状态中。

## 当前目标画像 intent
{target_intent}

## 当前会话语义上下文
- core_persona: {core_persona}
- current_task_intent: {task_intent}
- memory_write_intent: {target_intent}

## 当前「{dimension_name}」维度状态
{current_state}

## 新提取的偏好
{new_preferences}

## 合并规则
1. 如果新偏好与现有字段含义一致，更新该字段值
2. 如果新偏好是对现有值的补充/修正，合并或替换
3. 无法映射到固定字段的偏好，放入 "notes" 列表（保留所有有价值的条目，去除完全重复）
4. "keywords" 字段可以根据新偏好适当扩展
5. confidence 根据偏好一致性调整（多次一致提升，矛盾降低）
6. 当前更新目标是 intent={target_intent} 的画像，只更新这条 intent 的偏好表示，不要误写到其他 intent 语境
7. core_persona 表示用户长期稳定底色，current_task_intent 表示本次任务场景；若两者不完全一致，不要抹掉长期人格底色，但只把对 target_intent 稳定成立的偏好沉淀到目标画像

请输出更新后的维度 JSON（保持结构不变，只更新有变化的字段）。只输出 JSON，无需解释。
"""

PROFILE_CONSOLIDATION_RAW_PROMPT = """你是用户偏好画像管理器。请根据用户在本次会话中的**原始消息**和**最终幻灯片视觉分析**，直接更新目标 intent={target_intent} 的用户画像。

## 当前目标画像 intent
{target_intent}

## 当前会话语义上下文
- core_persona: {core_persona}
- current_task_intent: {task_intent}
- memory_write_intent: {target_intent}

## 用户在本次会话中的全部消息（原文）
{all_user_messages}

## 最终幻灯片的视觉参数分析
{slide_visual_analysis}

## 当前用户画像
{current_profile}

## 画像维度结构说明
用户画像包含以下 6 个维度，每个维度有固定字段：

1. **theme**（主题）: primary_colors(list[str]), accent_colors(list[str]), font_family(str), font_size_range(tuple|null), background_style(str), confidence(float), notes(list[str])
2. **visual**（视觉）: image_style(str), chart_type_priority(list[str]), icon_usage(str), animation_preference(str), confidence(float), notes(list[str])
3. **layout**（布局）: content_density(str), alignment_style(str), spacing_preference(str), slide_structure(str), confidence(float), notes(list[str])
4. **content**（内容）: text_density(str), language_style(str), bullet_point_style(str), title_length(str), confidence(float), notes(list[str])
5. **template**（模板）: preferred_templates(list[str]), avoid_templates(list[str]), selection_criteria(str), history_preferred_templates(list[str]), history_preferred_template_ids(list[str]), history_reuse_scenarios(list[str]), history_supported_aspect_ratios(list[str]), last_successful_template_id(str), last_successful_template_name(str), last_successful_at(str), history_supporting_usage_count(int), confidence(float), notes(list[str])
6. **general**（通用）: preferences(list[str]), confidence(float), notes(list[str])

## 更新规则
1. 先做内部“二次提炼”（仅作为你的思考脚手架，不要输出该中间结果）：把用户反馈尽量拆成「动作 + 原因 + 作用范围」三元组。
2. 二次提炼是软约束，不是硬过滤：若某条高价值偏好无法完整拆成三元组，但明显是长期偏好，仍可保留到对应维度（必要时写入 notes 或 general.preferences，并降低 confidence）。
3. 只沉淀长期偏好；以下通常视为 job-local/会话内约束，不写入长期画像：第X页、本轮、上一轮、当前修改轮次、附件、PDF、仅使用附件、不得补充附件外事实、回滚、优先处理等流程性描述。
4. 从用户原始消息中提取显式和隐式偏好（如“我喜欢简洁风格” → layout.content_density="sparse"）。
5. 视觉分析仅作次级证据；用户原始消息权重更高（用户明确表达优先）。
6. 论文/任务专有词要抽象化：不要把论文名、模型名、特定数据集名、超参数原词直接沉淀；改写为可迁移表达（如“关键指标”“实验依据”“关键取舍”“资源约束”“风险与边界”）。
7. 只更新有证据支撑的维度；没有相关信息的维度保持原样。
8. confidence 按证据强度设置：用户明确表达≈0.9，跨轮一致信号≈0.8，视觉或弱推断≈0.5-0.7。
9. 不要删除现有画像中已有的有效信息，只做增量更新或修正。
10. 单次任务里的“请用XX模板/这次用某模板”通常只是本轮执行约束，不是长期模板偏好；除非用户明确表达“以后默认/长期/总是/优先/避免某模板”，否则不要更新 template 维度，也不要把这类句子写入 general.preferences。
11. template 下的 history_*、last_successful_* 字段是系统根据 template_usage_history 自动维护的派生摘要，不要在输出 JSON 中主动生成或修改这些字段。
12. 当前更新目标是 intent={target_intent} 的画像，只沉淀与该 intent 场景稳定成立的长期偏好表达。
13. core_persona 表示用户长期稳定底色，current_task_intent 表示本次任务场景；两者不完全一致时，不要抹掉 core_persona，但也不要把当前场景偏好误写到其他 intent。
14. 如果一条规则只是当前这份 PPT / 当前会话里的临时全局要求，即便作用于“所有页”，但用户未明确长期化（如“以后默认/长期/总是/每次都这样”），不要写入长期画像的 general.preferences，也不要据此覆盖稳定字段。
15. 无法归入固定字段但确有长期价值的偏好，可放入对应维度 notes 或 general.preferences，避免丢失高价值信号。

## 输出格式
只输出需要更新的维度的 JSON，格式如下（只包含有变化的维度）：
```json
{{
  "theme": {{"primary_colors": [...], "confidence": 0.7, "notes": [...]}},
  "layout": {{"content_density": "sparse", "confidence": 0.8}},
  "general": {{"preferences": ["偏好1", "偏好2"]}}
}}
```

重要：
- 只输出 JSON，无需解释
- 只包含有变化的维度和字段
- 不要输出 keywords 字段（系统自动管理）
- 没有任何可提取的偏好信息时，输出空对象 {{}}
"""


CHAIN_INDEPENDENT_DISTILL_PROMPT = """你是PPT设计助手的工具链经验分析器。
请从以下单个工具链片段中提炼一条可复用的经验。

## 工具链名称（签名）
{chain_name}

## 工具序列
{tool_sequence}

## 完整执行记录
以下是 Agent 的完整执行轨迹，包含推理过程（Reasoning）、工具调用参数（Arguments）和执行结果（Observation）：

{execution_record}

## 执行结果
{outcome}

请输出 JSON 格式：
{{
  "lesson": "关键经验（1-2句话，具体可操作）",
  "applicable_when": "适用场景描述",
  "anti_pattern": "应避免的模式（来自失败/重试链，否则为空字符串）",
  "tool_pipeline": ["tool1", "tool2"],
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}

规则：
- lesson 应具体可操作，不要泛泛而谈
- 重点关注 Reasoning 中的决策逻辑和 Observation 中的关键反馈
- anti_pattern 只从 outcome 为 retry/partial/failed 的链中提取，否则为空字符串
- keywords 必须包含 3-8 个检索关键词，覆盖工具名、操作类型和适用场景
- 没有有价值的经验则返回空 JSON {{}}
- 只输出 JSON，无需解释
"""

CHAIN_EQUIVALENCE_CHECK_PROMPT = """你是PPT设计助手的经验等价判定器。
请判断以下两条工具链经验是否描述的是同一类问题的同一类解决方案。

## 新经验 A
- 经验教训: {new_lesson}
- 适用场景: {new_applicable_when}
- 工具管道: {new_tool_pipeline}

## 已有经验 B
- 经验教训: {existing_lesson}
- 适用场景: {existing_applicable_when}
- 工具管道: {existing_tool_pipeline}

## 等价判定标准
两个经验在以下条件下视为等价：
1. 描述的是同一类问题（相同的错误类型、相同的操作目标）
2. 提出的是同一类解决方案（相同的工具使用模式、相同的处理策略）
3. 适用场景高度重叠

请输出 JSON 格式：
{{
  "equivalent": true/false,
  "merged_lesson": "合并后的经验文本（仅在 equivalent=true 时有值，综合两条经验的精华；equivalent=false 时为空字符串）"
}}

只输出 JSON，无需解释。
"""
