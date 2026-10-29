---
name: image-handling
description: "Design 阶段 — 区分装饰图 vs 内容图，理解几何坐标"
phase: [design]
data_fields: [image_stats]
inject_timing: 启动时（摘要）+ 每页（详情）
---

# 图片处理指南

## 数据字段

### image_stats

```json
{
  "a96d997921e6...png": {
    "size": [2441, 1097],
    "appear_times": 1,
    "slide_numbers": [1],
    "relative_area": 24.56
  },
  "34cb01dd3864...png": {
    "size": [800, 800],
    "appear_times": 10,
    "slide_numbers": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "relative_area": 0.95
  }
}
```

## 使用规则

### 区分装饰图 vs 内容图

| 特征组合 | 判断 | 处理方式 |
|---------|------|---------|
| `appear_times` ≥ 3 且 `relative_area` < 5% | 图标/Logo | **不替换** |
| `appear_times` = 1 且 `relative_area` > 20% | 封面/全屏背景图 | 根据内容决定 |
| `appear_times` = 1 且 10% < `relative_area` < 50% | 内容配图 | **可替换** |
| `appear_times` ≥ 2 且 `relative_area` > 20% | 通用背景/装饰 | **不替换** |

**核心规则**：
- 多次出现 + 小面积 = 装饰图标，不替换
- 单次出现 + 中等面积 = 内容图，可替换

### 与 slide_induction 的对应

- `"decorative image"`, `"background/decorative image"` → 装饰图，保持不变
- `"main image"`, `"hero image"` → 内容图，可替换

**冲突时**：以 slide_induction 的 elements 为准。
