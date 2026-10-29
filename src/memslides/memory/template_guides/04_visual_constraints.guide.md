---
name: visual-constraints
description: "Design 阶段全程 — 语义配色、字体和可读性约束"
phase: [design]
data_fields: [design_constraints]
inject_timing: 启动时
---

# 视觉约束指南

## 数据字段

### design_constraints

```json
{
  "color_palette": {
    "primary": "#004aad",
    "secondary": "#a6a6a6",
    "accent": "#737373",
    "background": "#004aad",
    "surface": "#f4f7fb",
    "primary_text": "#111827",
    "inverse_text": "#ffffff",
    "muted_text": "#545454",
    "border": "#a6a6a6",
    "text": "#111827",
    "additional": ["#545454", "#343434"]
  },
  "typography": {
    "title_font": "微软雅黑",
    "title_size": 38,
    "body_font": "微软雅黑",
    "body_size": 32,
    "line_height": 1.5
  },
  "spacing": { "margin": "10pt", "padding": "5pt" }
}
```

## 使用规则

### 配色层级

```
background — 页面壳层、标题带、装饰底色
surface — 正文、公式、表格、图表的安全承载面
primary_text — surface 上的正文主色
inverse_text — 深色壳层上的反白标题/短标签
muted_text — 次级说明、caption、辅助信息
border — 线条、分隔、卡片边框
primary — 标题/重点区域/关键图形
secondary — 次要元素、辅助文字、分割线
accent — 图标着色、高亮标记
text — 旧字段兼容，新链路优先使用 primary_text/inverse_text
additional — 次级图表或装饰色
```

**规则**：
1. **语义角色优先** — 不要把 background 当成正文承载面的唯一颜色。
2. **可读性优先** — 正文、公式、表格、caption 必须使用 `surface + primary_text` 或等价高对比组合。
3. **壳层分离** — 封面、标题带、短标签可以使用 `background + inverse_text`。
4. **允许安全派生** — 当模板没有可读 surface 时，可从模板色系派生浅色内容面，但不得制造低对比组合。

### 字号层级推导

```
主标题:    title_size（如 38pt）
副标题:    title_size × 0.7（如 27pt）
小节标题:  body_size × 1.1（如 35pt）
正文:      body_size（如 32pt）
注释/标签: body_size × 0.75（如 24pt）
步骤编号:  title_size × 1.2（如 46pt）
```

### 字体规则

- 所有标题元素使用 `title_font`
- 所有正文元素使用 `body_font`
- 中文字体名在 CSS 中需要加引号：`font-family: "微软雅黑"`

### 间距规则

```
margin: "10pt"  → 元素到页面边缘的最小距离
padding: "5pt"  → 元素内部的最小留白
```

### 图表配色

当 `chart_style.color_scheme` 为空时：
1. 图表底面 → surface
2. 图表文字 → primary_text
3. 第一系列 → primary
4. 第二系列 → accent
5. 第三系列 → secondary
6. 更多系列 → additional 中的颜色
