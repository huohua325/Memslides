from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class TemplateUseIntent(StrEnum):
    EXPLICIT = "explicit"
    MEMORY_REUSE = "memory_reuse"
    STRONG_REFERENCE_STYLE = "strong_reference_style"
    NONE = "none"
    DISABLED = "disabled"


@dataclass
class TemplateActivationDecision:
    use_intent: TemplateUseIntent
    allowed: bool
    reason: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["use_intent"] = self.use_intent.value
        return data


_REUSE_PATTERNS = (
    r"\b(reuse|use|apply)\b.{0,40}\b(previous|prior|last|learned|memory|historical)\b.{0,30}\b(template|layout)\b",
    r"\b(previous|prior|last|learned|memory|historical)\b.{0,40}\b(template|layout)\b",
    r"\b(template|layout)\b.{0,40}\b(reuse|learned|memory|history|historical)\b",
    r"(复用|沿用|使用|套用).{0,20}(上次|之前|先前|历史|记忆|学到).{0,20}(模板|版式|布局)",
    r"(上次|之前|先前|历史|记忆|学到).{0,20}(模板|版式|布局)",
)

_DISABLE_PATTERNS = (
    r"\b(do not use any template|no template at all|without any template|freeform mode|freeform)\b",
    r"(不要|不用|不使用|别用).{0,8}(模板|版式|布局)",
    r"(自由发挥|自由设计|不套模板)",
)


def _matched(patterns: tuple[str, ...], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


def decide_template_activation(
    request: Any,
    style_intent_result: Any | None = None,
) -> TemplateActivationDecision:
    """Decide whether a run is allowed to activate a template.

    Style words such as "academic", "professional", or "blue-green accents"
    are intentionally not template-use permission. A template may enter the
    runtime only through explicit input or clear memory-reuse language.
    """

    style_decision = str(getattr(style_intent_result, "template_use_decision", "") or "")
    style_basis = str(getattr(style_intent_result, "template_use_basis", "") or "")
    if style_decision == "forbid_template":
        return TemplateActivationDecision(
            use_intent=TemplateUseIntent.DISABLED,
            allowed=False,
            reason="Style intent gate forbids template usage.",
            evidence=[f"template_use_basis:{style_basis or 'unknown'}"],
        )
    if style_decision == "use_template":
        if style_basis == "memory_reuse":
            use_intent = TemplateUseIntent.MEMORY_REUSE
        elif style_basis == "strong_reference_style":
            use_intent = TemplateUseIntent.STRONG_REFERENCE_STYLE
        else:
            use_intent = TemplateUseIntent.EXPLICIT
        return TemplateActivationDecision(
            use_intent=use_intent,
            allowed=True,
            reason="Style intent gate allows template selection.",
            evidence=[f"template_use_basis:{style_basis or 'unknown'}"],
        )
    if style_decision == "no_template":
        return TemplateActivationDecision(
            use_intent=TemplateUseIntent.NONE,
            allowed=False,
            reason="Style intent gate did not find template-use permission.",
            evidence=[f"template_use_basis:{style_basis or 'unknown'}"],
        )

    instruction = str(getattr(request, "instruction", "") or "")
    lowered = instruction.lower()
    disabled = _matched(_DISABLE_PATTERNS, lowered)
    if disabled:
        return TemplateActivationDecision(
            use_intent=TemplateUseIntent.DISABLED,
            allowed=False,
            reason="User explicitly disabled template usage.",
            evidence=disabled,
        )

    if (
        getattr(request, "template", None)
        or getattr(request, "template_id", None)
        or bool(getattr(request, "template_as_reference", False))
    ):
        return TemplateActivationDecision(
            use_intent=TemplateUseIntent.EXPLICIT,
            allowed=True,
            reason="Request provided an explicit template input.",
            evidence=["request.template/template_id/template_as_reference"],
        )

    reuse = _matched(_REUSE_PATTERNS, lowered)
    memory_hint = str(getattr(style_intent_result, "memory_hint", "") or "")
    if memory_hint in {"last_template", "previous_template", "that_template"}:
        reuse.append(f"memory_hint:{memory_hint}")
    if reuse:
        return TemplateActivationDecision(
            use_intent=TemplateUseIntent.MEMORY_REUSE,
            allowed=True,
            reason="Request clearly asks to reuse a previous or learned template.",
            evidence=reuse,
        )

    return TemplateActivationDecision(
        use_intent=TemplateUseIntent.NONE,
        allowed=False,
        reason="No explicit template input or template-reuse language was found.",
        evidence=[],
    )


__all__ = [
    "TemplateActivationDecision",
    "TemplateUseIntent",
    "decide_template_activation",
]
