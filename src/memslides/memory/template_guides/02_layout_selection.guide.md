---
name: layout-selection
description: "Design 阶段启动时 — 如何从 slide_induction 中选择合适的布局"
phase: [design]
data_fields: [slide_induction]
inject_timing: 启动时
---

# 布局选择指南

## 数据字段

`slide_induction` 是一个 JSON 对象：

```json
{
  "<布局名称A>": { "template_id": 1, "slides": [1], "elements": [...] },
  "<布局名称B>": {...},
  "functional_keys": ["opening", "table of contents", "ending"],
  "language": {"lid": "zh"}
}
```

## 使用规则

### 布局名称格式

#### 格式 1：功能页名称（短名称）

```
"opening"
"table of contents"
"ending"
```

这些是**功能性页面**的固定名称，同时出现在 `functional_keys` 列表中。

#### 格式 2：布局描述 + 内容类型后缀

```
"Top Title Row with Logo, Central Tall Image...:image"
"Header with three column text blocks:text"
```

内容页的名称是**一句话布局描述**，后面跟 `:image` 或 `:text` 后缀。

**后缀含义**：
- `:image` → 多媒体布局（含图片元素）
- `:text` → 纯文本布局

### 布局选择方法

| 内容类型 | 应选择的布局 |
|---------|------------|
| 封面 | `"opening"` |
| 目录 | `"table of contents"` |
| 结尾/致谢 | `"ending"` |
| 三栏对比 | 描述含 "three column" / "three panels" |
| 图文混排 | 描述含 "image" + "text"，后缀 `:image` |
| 时间线/步骤 | 描述含 "timeline" / "step" |
| 图表+说明 | 描述含 "chart" / "graph" |
| 纯文本 | 后缀 `:text` |

### 特殊 key（不是布局）

```
"functional_keys": [...]  → 功能页名称列表
"language": {...}         → 模板语言信息
```

遍历布局时跳过这两个 key。
