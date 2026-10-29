---
name: element-filling
description: "Design 阶段每页 — 如何读懂 slide_induction.elements 并正确填充内容"
phase: [design]
data_fields: [slide_induction]
inject_timing: 每页（MCP tool）
---

# 元素填充指南

## 数据字段

每个布局的 `elements` 数组：

```json
{
  "elements": [
    { "name": "main title", "type": "text", "data": ["输入标题"] },
    { "name": "left column body text", "type": "text", "data": ["单击此处输入您的正文..."] },
    { "name": "decorative image", "type": "image", "data": ["[image: 图片 136]"] }
  ]
}
```

## 使用规则

### name — 元素语义名称

| 名称模式 | 含义 |
|---------|------|
| `"main title"`, `"header main title"` | 页面主标题 |
| `"subtitle"`, `"secondary title"` | 副标题 |
| `"left column body text"` | 左栏正文 |
| `"step number"` | 步骤编号（如 01, 02） |
| `"decorative image"` | 装饰性图片（不替换） |
| `"main image"`, `"hero image"` | 内容图片（可替换） |

### type — 元素类型

- `"text"` — 文本框，需要填入文字
- `"image"` — 图片框

### data — 示例内容

`data` 数组告诉你：
1. **建议字符数** = `max(len(d) for d in data)`
2. **默认数量** = `len(data)`

### 字符限制规则

| 元素类型 | 规则 |
|---------|------|
| 标题（name 含 "title"） | 严格控制在示例字符数 ±20% |
| 正文（name 含 "body"/"text"） | 不超过示例字符数的 150% |
| 编号（name 含 "number"） | 必须与示例长度一致 |
| 标签（name 含 "label"/"tag"） | 不超过示例字符数 |

### 元素数量规则

布局有 3 个 step 元素，你必须生成恰好 3 组步骤。不能增减。
