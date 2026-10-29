---
name: field-map
description: "design_skills 表字段总览"
phase: [research, design]
data_fields: [slide_count, aspect_ratio, slide_induction, design_constraints, content_patterns, image_stats]
inject_timing: 启动时
---

# 字段地图

你正在**参考一个已有的 PPT 模板**来生成新的 PPT。模板数据存储在 `design_skills` 表中。

## 字段总览

| 字段 | 存储内容 | 阶段 | 指导 Guide |
|------|---------|------|-----------|
| `slide_count` | 模板总页数 | Research + Design | 本文档 |
| `aspect_ratio` | 宽高比（如 16:9） | Research + Design | 本文档 |
| `content_patterns` | 信息密度、叙事风格、章节结构 | **Research** | `01_content_planning` |
| `slide_induction` | 每种布局的元素清单 | **Design** | `02_layout_selection` + `03_element_filling` |
| `design_constraints` | 配色方案、字体规范 | **Design** | `04_visual_constraints` |
| `image_stats` | 模板中每张图片的尺寸、出现频率 | **Design** | `05_image_handling` |

## 阶段分工

### Research 阶段
- 查阅：`content_patterns`, `slide_induction.functional_keys`, `slide_count`
- 输出：内容大纲（章节划分、每页内容要点、推荐布局类型）

### Design 阶段
- 查阅：`slide_induction`, `design_constraints`, `image_stats`
- 输出：符合模板规范的 HTML 幻灯片代码

## 关键原则

1. **slide_induction 是核心数据源** — 直接告诉你每种布局有哪些元素
2. **design_constraints 是硬约束** — 配色和字体必须严格遵守
3. **用户指令优先于模板** — 当用户明确要求与模板不同时，以用户为准
