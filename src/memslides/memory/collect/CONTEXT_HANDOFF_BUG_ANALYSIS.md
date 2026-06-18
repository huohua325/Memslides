# Context Handoff Bug 分析报告

## 问题现象

DeckDesigner 在 context 压缩（compact）后，陷入无限写 `state_summary.md` 的循环，
产生 43+ 次无意义的 `write_markdown_file` 调用，浪费了整个 compact 后的 context 预算，
最终被 `MAX_AGENT_ITERATIONS` 强制 finalize，没有完成任何实际的 slide 生成工作。

## 证据链（来自 `20260306_cache/2c565cdf/`）

### 时间线

| 阶段 | 文件 | 消息范围 | 行为 |
|------|------|----------|------|
| Design-00 | 52 msgs | msg[0]-msg[51] | 正常工作：生成 slide_01 到 slide_14 |
| Design-01 compact前 | msg[0]-msg[28] | 继续生成 slide，到 msg[26] 完成 slide_14 |
| Design-01 compact | msg[29] | 注入 `MEMORY_COMPACT_MSG`，要求保存状态摘要 |
| Design-01 循环 | msg[30]-msg[116] | **43次** write_markdown_file，0次 write_html_file |
| Design-01 强制结束 | msg[117]-msg[119] | `FORCE_FINALIZE_MSG` → finalize → slides |
| Design-02 | 84 msgs | 从头开始，重新生成所有 15 张 slide |

### 关键数据

- Design-01 compact 后写了 **29 个不同文件名** 的 state_summary 变体
- 文件名从 `state_summary.md` 逐渐升级到 `STATE_SUMMARY_CANONICAL_FINAL.md`、
  `PROJECT_CONTINUITY_STATE_SUMMARY.md` 等越来越夸张的名字
- `state_summary.md` 本身被写了 **11 次**
- compact 后 **0 次** `read_file` 调用 — agent 从未读取自己写的 summary
- compact 后 **0 次** `write_html_file` — 没有继续做任何 slide 工作

## 根因分析

### 机制说明

`compact_history()` 不是创建新 session，而是在同一个 Agent 实例内做 context 压缩：

```
compact_history() 流程:
1. save_history(message_only=True)  → 保存当前快照为 Design-{iter}.jsonl
2. research_iter += 1
3. _split_history(keep_head=10, keep_tail=8)
   - head: 保留前 10 条消息（system + user + 最初几轮工具调用）
   - tail: 保留最后 8 条消息（最近的工具调用）
   - 中间部分全部丢弃
4. 向 LLM 发送 MEMORY_COMPACT_MSG，要求生成状态摘要
5. LLM 回复（通常调用 write_markdown_file 保存摘要）
6. 执行工具调用，在结果中追加 CONTINUE_MSG
7. 新的 chat_history = head + tail + [compact_ask, compact_response, tool_results]
```

### Bug 1: "写了但不读" — compact 回调的结构性缺陷

`MEMORY_COMPACT_MSG` 告诉 agent：
> "save it to the working directory to ensure seamless continuation in subsequent conversations"

但 `CONTINUE_MSG` 只说：
> "History has been compacted. Refer to the saved summary and continue your work"

**问题**：compact 后的 `head` 保留了最初的 system prompt + user message + 前几轮交互。
这些消息中没有任何指令告诉 agent 去 `read_file(state_summary.md)`。
agent 的 `head` 上下文看到的是"从零开始做 slide"的初始指令，
而 `tail` 只有最近 8 条消息（compact 摘要 + CONTINUE_MSG）。

agent 处于一个矛盾状态：
- `CONTINUE_MSG` 说"参考保存的摘要继续工作"
- 但 agent 不知道摘要文件在哪里（没有 read_file 指令）
- agent 也无法验证摘要是否完整保存
- 于是 agent 不断重写摘要，试图"确保"状态被保存

### Bug 2: 无终止条件的写入循环

compact 回调执行 LLM 的工具调用后，将结果追加到 `chat_history`，
然后 `execute()` 方法的主循环继续运行。agent 看到 `CONTINUE_MSG` 后，
不是去做 slide 工作，而是继续写 state_summary — 因为：

1. `head` 中的原始任务上下文已经过时（slide 进度信息在被丢弃的中间部分）
2. agent 无法确认自己的摘要是否被"后续对话"读取
3. 每次写完 summary，tool result 说 "Successfully wrote"，但没有任何信号说"够了，开始工作"
4. agent 于是换个文件名再写一次，希望"这次能被读到"

### Bug 3: Design-02 从零开始 — compact 后的 head 没有进度信息

Design-02（第二次 compact 后的快照）的行为证实了这个问题：

```
msg[0] system: (原始 system prompt，和 Design-00 完全相同)
msg[1] user: (原始 user message，和 Design-00 完全相同)
msg[2] assistant: [list_files, list_files, inspect_manuscript]  ← 从零开始探索
msg[3] tool: Directory not found: outputs
msg[4] tool: Directory 'slides' is empty  ← slides 目录被报告为空！
```

`_split_history(keep_head=10)` 保留了最初的 10 条消息，
这些消息是 agent 刚启动时的探索阶段（list_files → 空目录 → 读 manuscript）。
compact 后 agent 看到的上下文是"slides 目录是空的"，
所以它重新生成了所有 15 张 slide。

**注意**：msg[4] 说 "Directory 'slides' is empty" 是因为 head 保留的是
最初的 tool result（当时 slides 确实是空的），不是当前的文件系统状态。

## 影响评估

| 维度 | 影响 |
|------|------|
| Token 浪费 | compact 后 43 次 write_markdown_file ≈ 每次 ~6KB 内容 ≈ 258KB 纯文本 |
| 时间浪费 | 43 次 LLM 调用 + 43 次工具执行，估计 3-5 分钟 |
| 工作重复 | Design-02 从零重新生成 15 张 slide，完全浪费了 Design-00/01 的工作 |
| 用户体验 | 用户等待时间翻倍，最终产出质量可能更差（因为 context 预算已耗尽） |

## 修复建议

### 方案 A: compact 回调后注入 read_file 指令（最小改动）

在 `compact_history()` 的 `CONTINUE_MSG` 之后，追加一条明确的指令：

```python
# 在 compact_history() 的 observations 构建之后
READ_SUMMARY_MSG = {
    "text": (
        "<INSTRUCTION>Your state summary has been saved. "
        "Do NOT write any more summary files. "
        "Read the summary with `read_file('state_summary.md')` if needed, "
        "then IMMEDIATELY continue your primary task.</INSTRUCTION>"
    ),
    "type": "text",
}
```

### 方案 B: compact 时自动注入摘要内容到 head（推荐）

不依赖 agent 自己去 read_file，而是在 compact 后直接把摘要内容
注入到 `head` 的最后一条消息中：

```python
# compact_history() 中，在构建 new_tail 之前
summary_content = summary_message.text  # LLM 生成的摘要
# 不需要 agent 调用 write_markdown_file，直接把摘要嵌入 context
injected_summary = ChatMessage(
    role=Role.SYSTEM,
    content=f"<state_summary>\n{summary_content}\n</state_summary>"
)
# new_tail 中不执行 write_markdown_file，直接注入摘要
```

### 方案 C: 利用 WorkingMemory 替代 MEMORY_COMPACT_MSG（长期方案）

`working_memory` 标签已经在 tool result 中注入了进度信息：
```
<working_memory>
📋 ☑manuscript ☑design_plan ☑s01,s04,s08,s11,s15,s16 ◐s02,s03,s05,s06,...
</working_memory>
```

可以让 compact 直接使用 WorkingMemory 的状态，而不是让 LLM 自己总结：

```python
if self._working_memory:
    summary = self._working_memory.build_reminder("full", ...)
    # 直接注入，跳过 LLM 总结步骤
```

### 方案 D: 限制 compact 后的 write_markdown_file 调用次数

作为防御性措施，在 compact 回调的工具执行中限制 `write_markdown_file` 只能调用 1 次：

```python
# compact_history() 中执行 LLM 工具调用时
_compact_tcs = summary_message.tool_calls or []
# 只保留第一个 write_markdown_file，丢弃其余
_write_count = 0
filtered_tcs = []
for tc in _compact_tcs:
    if tc.function.name == 'write_markdown_file':
        _write_count += 1
        if _write_count > 1:
            continue
    filtered_tcs.append(tc)
```

## 推荐优先级

1. **方案 B**（compact 时注入摘要）— 从根本上解决问题，agent 不需要自己读写文件
2. **方案 D**（限制写入次数）— 作为防御性措施立即部署
3. **方案 A**（注入 read 指令）— 简单但依赖 agent 遵守指令
4. **方案 C**（WorkingMemory 替代）— 长期最优，但改动较大

## 相关文件

- `memslides/agents/agent.py` — `compact_history()`, `execute()`, `_split_history()`
- `memslides/utils/constants.py` — `MEMORY_COMPACT_MSG`, `CONTINUE_MSG`
- `memslides/main.py` — `AgentLoop.run()` 中 DeckDesigner 的创建和调用
- `memslides/agents/deck_designer.py` — DeckDesigner 的 loop 实现
