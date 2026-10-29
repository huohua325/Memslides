from memslides.memory.compliance.template_checker import (
    ComplianceIssue,
    TemplateComplianceChecker,
)
from memslides.memory.core.template_models import TemplateProfile
from memslides.memory.extract.template_analyzer import TemplateAnalyzer, TemplateAnalysis
from memslides.memory.inject.template_guide_builder import TemplateGuideBuilder
from memslides.memory.template_selector import TemplateSelectionResult, TemplateSelector
from memslides.templates.activation import (
    TemplateActivationDecision,
    TemplateUseIntent,
    decide_template_activation,
)
from memslides.templates.induction import induct_template
from memslides.templates.layout_planner import (
    LayoutRecommendation,
    PageBrief,
    TemplateLayoutPlanner,
    dump_layout_mapping_json,
    load_layout_mapping,
)
from memslides.templates.page_asset_planner import PageAssetBinding, PageAssetPlanner
from memslides.templates.quality import (
    ADAPTIVE_STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    STRUCTURAL_TEMPLATE,
    STYLE_REFERENCE,
    TemplateQualityReport,
    assess_template_quality,
)
from memslides.templates.shell import (
    LAYERED_SHELL,
    NATIVE_BACKGROUND,
    NO_SHELL,
    RASTER_SHELL,
    TemplateShellView,
    resolve_template_shell,
    shell_summary,
    template_shell_by_layout,
    template_shell_model,
)
from memslides.templates.shell_assets import materialize_template_shell_assets

__all__ = [
    "TemplateProfile",
    "TemplateAnalysis",
    "TemplateAnalyzer",
    "TemplateSelector",
    "TemplateSelectionResult",
    "TemplateComplianceChecker",
    "ComplianceIssue",
    "TemplateGuideBuilder",
    "PageBrief",
    "PageAssetBinding",
    "PageAssetPlanner",
    "LayoutRecommendation",
    "TemplateLayoutPlanner",
    "TemplateActivationDecision",
    "TemplateUseIntent",
    "TemplateQualityReport",
    "TemplateShellView",
    "NATIVE_BACKGROUND",
    "LAYERED_SHELL",
    "RASTER_SHELL",
    "NO_SHELL",
    "STRICT_STRUCTURAL_TEMPLATE",
    "ADAPTIVE_STRUCTURAL_TEMPLATE",
    "STRUCTURAL_TEMPLATE",
    "STYLE_REFERENCE",
    "assess_template_quality",
    "decide_template_activation",
    "induct_template",
    "template_shell_model",
    "template_shell_by_layout",
    "resolve_template_shell",
    "shell_summary",
    "materialize_template_shell_assets",
    "load_layout_mapping",
    "dump_layout_mapping_json",
]
