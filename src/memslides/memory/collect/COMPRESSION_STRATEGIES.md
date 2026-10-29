# ToolCallSegment 压缩策略说明

## 概述

`ToolCallSegment.from_raw_tool_call()` 负责将 Agent 的原始工具调用（参数 + 返回值，通常数百到数千字符）压缩为属性级摘要（通常 50~150 字符），供下游 EpisodeExtractor 和 ExperienceWriter 消费。

基于 `20260306_cache/2c565cdf` 的 225 次真实工具调用数据，整体压缩率为 **3.5%**（544,017 → 18,955 chars）。

## 各工具压缩策略

### 1. write_html_file（×95，压缩率 19.0%）

**策略**: 提取可读文字内容（前 15 词）+ 关键 CSS 属性（font-size/color/background 等）。如果有 `prev_content`，还会计算 before→after 的 CSS 属性 diff。

```
原始: {"content": "<!DOCTYPE html><html>...(1.6KB HTML)...", "file_path": "slides/slide_05.html"}
压缩: write_html_file (slide_05.html) : text="Scholar DAG（学术DAG）"; font-size=46px; color=#111111
```

### 2. write_markdown_file（×80，压缩率 1.5%）

**策略**: 区分有意义内容和噪声。
- **Manuscript / Plan 文件**: 提取文件名 + 首行标题内容
- **State Summary 文件**: 标记为 `[state]` 噪声（占 63/71 = 89%）

噪声检测关键词: `state_summary`, `continuity`, `continuation`, `handoff`, `session_state`, `working_state`, `context_handoff`, `project_state`, `canonical_state`

```
有意义: write paperx_ppt_manuscript.md: # PaperX：多模态学术汇报生成框架 [0.5m]
噪声:   [state] STATE_SUMMARY_FOR_CONTINUATION.md
```

### 3. read_file / read_html_file（×62，压缩率 7.9%）

**策略**: 保留文件名 + offset/limit 范围信息（如果有）。

```
有范围: read PaperX.md [L0:2000]
无范围: read slide_2.html
```

### 4. inspect_slide（×60，压缩率 29.8%）

**策略**: 从 result 中提取修改状态（CHANGED/UNCHANGED）、title、元素数、图片数、截图路径。

```
inspect_slide (slide_04.html) : CHANGED; title=Scholar DAG; elements=5; screenshot=xxx.png
```

### 5. query_slide_layout（×37）

**策略**: 提取 layout_name + 从 result 表格中统计元素数量。

```
query_slide_layout : layout=opening (3 elements)
```

### 6. image_caption（×41，压缩率 31.7%）

**策略**: 从 result JSON 中提取图片文件名 + caption 文本（前 80 字符）。

```
image_caption : p24_region_8.png: Picture: Four research posters (labeled a-d) with text, diagrams
```

### 7. convert_to_markdown（×2，压缩率 21.7%）

**策略**: 提取源文件名 + 从 result 中提取图片数量。

```
convert_to_markdown (PaperX.pdf) : convert PaperX.pdf; Found 15 images
```

### 8. list_document_figures（×5，压缩率 12.2%）

**策略**: 从 result JSON 中提取 total 数量 + 前 5 个文件名。

```
list_document_figures : 15 figures: p2_region_0.png, p12_region_2.png
```

### 9. inspect_manuscript（×8，压缩率 49.2%）

**策略**: 提取文件名 + num_pages + language + warnings 数量。错误时保留 error 信息。

```
正常: inspect paperx_ppt_manuscript.md; 16 pages; lang=zh
错误: inspect bad.md [ERROR: file does not exist: bad.md]
```

### 10. remember_lesson（×7，压缩率 7.8%）

**策略**: 提取 tool_name 标签 + content 前 80 字符摘要。

```
remember_lesson : [lesson:whitespace_density] User feedback indicates the current slides are under-filled
```

### 11. 其他工具（兜底）

**策略**: `finalize` 提取 outcome；`thinking` 提取 thought 前 120 字符；`todo_*` 提取任务内容；`list_files` 提取目录名；`image_generation`/`search_image` 提取 prompt/query。未识别的工具截断 args 到 80 字符。

## 压缩效果总览（基于真实数据）

| 工具 | 调用次数 | 原始大小 | 压缩后 | 压缩率 |
|------|---------|---------|--------|--------|
| write_markdown_file | 71 | 476,097 | 6,958 | 1.5% |
| image_caption | 41 | 14,529 | 4,599 | 31.7% |
| write_html_file | 41 | 15,370 | 2,916 | 19.0% |
| read_file | 26 | 22,588 | 1,793 | 7.9% |
| inspect_slide | 23 | 4,091 | 1,219 | 29.8% |
| list_files | 7 | 2,408 | 209 | 8.7% |
| finalize | 5 | 315 | 178 | 56.5% |
| list_document_figures | 4 | 2,256 | 276 | 12.2% |
| inspect_manuscript | 3 | 528 | 260 | 49.2% |
| remember_lesson | 3 | 5,170 | 403 | 7.8% |
| convert_to_markdown | 1 | 665 | 144 | 21.7% |
| **合计** | **225** | **544,017** | **18,955** | **3.5%** |

## 设计原则

1. **信息保留优先级**: 用户意图 > 设计属性变化 > 文件操作 > 工具元数据
2. **噪声过滤**: Agent 内部状态续传文件（占 write_markdown_file 的 89%）被标记为 `[state]` 噪声，下游 EpisodeExtractor 可据此跳过
3. **结构化提取优于截断**: 每种工具都有专门的提取逻辑，从 args/result 中提取语义关键信息，而非简单截断 JSON
4. **向后兼容**: 未识别的工具仍走兜底截断路径，新增工具不会导致崩溃

## 上游修复：Context Compact 无限写 state_summary 问题

### 问题

`write_markdown_file` 的 71 次调用中有 63 次（89%）是 `[state]` 噪声，根因是 `compact_history()` 的设计缺陷：
- compact 后 `CONTINUE_MSG` 告诉 agent "参考保存的摘要继续工作"
- 但 agent 的 context 中没有摘要内容，也没有 `read_file` 指令
- agent 无法验证摘要是否被保存/读取，于是不断换文件名重写（43 次），耗尽整个 context 预算

### 修复（方案 B — 已实施）

在 `agents/agent.py` 的 `compact_history()` 中：
1. LLM 生成的摘要文本直接用 `<state_summary>` 标签嵌入 compact 后的 context
2. 如果 LLM 把摘要写在 `write_markdown_file` 参数中（content 为空），从工具参数提取
3. 追加 `<INSTRUCTION>` 明确禁止再写 summary 文件，要求立即继续主任务
4. 工具调用仍执行（文件保存到磁盘用于调试），但 agent 不再需要自己读取

### 预期效果

修复后 `write_markdown_file` 的 `[state]` 噪声应大幅减少（从 63 次降至 ≤1 次），
整体压缩率会进一步改善（`write_markdown_file` 原始大小从 476KB 降至约 20KB）。

## 数据来源

压缩策略基于 `20260306_cache/2c565cdf` 的真实运行数据设计，该 session 包含：
- Research Agent: 文档解析（convert_to_markdown, read_file, list_document_figures, image_caption）
- DeckDesigner: 幻灯片生成（write_html_file, write_markdown_file, inspect_slide, query_slide_layout, finalize）
- RevisionEditor: 交互修改（write_html_file, inspect_slide, remember_lesson, inspect_manuscript）
