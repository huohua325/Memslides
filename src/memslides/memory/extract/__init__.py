"""Extract — 记忆提取模块

模块结构:
    episode_extractor.py        — 情景记忆提取 (DesignEpisode)
    preference_extractor.py     — Episode/round 级偏好提取 (AtomicPreference)
                                  注：modify 阶段当前不再走 user_message_analyzer +
                                  constraint/preference 二分写入链路
    visual_fingerprint.py       — 视觉指纹提取 (Stage 12)
                                  SlideVisualFingerprintExtractor (Layer 1: HTML 确定性提取)
                                  VLM_VISUAL_PREFERENCE_PROMPT (VLM 分析 prompt)
                                  注：统计聚合+编排层已移入 Consolidator
    template_skill_extractors.py — 模板技能提取，合并自
                                  style_extractor + content_pattern_extractor
    template_analyzer.py        — 模板整体分析
    layout_skill_extractor.py   — 布局技能提取
    tool_knowledge_learner.py   — 工具知识学习 (DEPRECATED Stage 15, 由 orchestrator.on_tool_error 取代)
    state_extractor.py          — PPT 状态提取 (DEPRECATED, 仍被 main.py 引用)
    state_coordinator.py        — 状态协调器 (DEPRECATED, 仍被 main.py 引用)
"""
