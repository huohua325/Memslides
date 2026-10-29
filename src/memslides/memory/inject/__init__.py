# memory/inject — 记忆注入层 (Stage 2 / Stage 7+8 重构)
#
# 注入能力：
# 1. MemoryOrchestrator.get_memory_components() — 通用记忆注入（WM偏好 + WM经验 + WM任务历史 + LTM工具链经验）
# 2. template_guide_builder — 模板技能注入（可插拔，仅模板驱动模式使用）
# 3. profile_injection_router — ProfileInjectionRouter Job Start 智能路由

from .template_guide_builder import TemplateGuideBuilder
from .design_plan_generator import DesignPlanGenerator, _classify_layout_type

__all__ = [
    # 模板驱动（可插拔）
    "TemplateGuideBuilder",
    "DesignPlanGenerator",
    "_classify_layout_type",
]
