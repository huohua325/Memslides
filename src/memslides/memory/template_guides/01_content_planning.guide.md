---
name: content-planning
description: "Research 阶段专用 — 如何使用 content_patterns 字段规划 PPT 内容大纲"
phase: [research]
data_fields: [content_patterns]
inject_timing: 启动时
---

# 内容规划指南

## 数据字段

`content_patterns` 字段结构：

```json
{
  "info_density": "medium",
  "narrative_style": "business",
  "typical_sections": ["opening", "table_of_contents", "content", "ending"],
  "bullet_style": "bulleted",
  "max_bullets_per_slide": 5
}
```

## 使用规则

### info_density — 信息密度

| 值 | 含义 | 实操指导 |
|----|------|---------|
| `"low"` | 精简风格 | 每页 2-3 个要点，大量留白 |
| `"medium"` | 平衡风格 | 每页 4-5 个要点，图文混排 |
| `"high"` | 信息密集 | 每页 6+ 个要点 |

### narrative_style — 叙事风格

| 值 | 含义 | 内容组织方式 |
|----|------|-------------|
| `"academic"` | 学术风格 | 引用严谨、术语专业、逻辑递进 |
| `"business"` | 商务风格 | 数据驱动、简洁直接、先结论后论据 |
| `"creative"` | 创意风格 | 故事叙述、视觉优先 |
| `"educational"` | 教育风格 | 循序渐进、解释清晰 |
| `"technical"` | 技术风格 | 架构图、代码片段、实现细节 |

### typical_sections — 典型章节

| 值 | 含义 |
|----|------|
| `"opening"` | 封面/开场（必有） |
| `"table_of_contents"` | 目录页 |
| `"section_divider"` | 章节分隔页 |
| `"content"` | 内容页（PPT 主体） |
| `"ending"` | 结尾/致谢页（必有） |

**规则**：你的大纲应包含 typical_sections 中列出的所有章节类型。如果模板没有某类型，不要规划该类型的页面。

### max_bullets_per_slide

每页的要点/列表项不超过此值。超过时应拆分到多页。

### bullet_style

- `"bulleted"` → 圆点列表
- `"numbered"` → 数字编号
