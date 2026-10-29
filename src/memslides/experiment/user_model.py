from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memslides.experiment.config import UserModelProfile


class UserModelResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    feedback: str
    satisfied: bool
    score: float


class ReviewContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    slide_image_paths: list[str] = Field(default_factory=list)
    round_num: int = 0
    instruction: str = ""
    feedback_history: list[str] = Field(default_factory=list)
    task_intent: str = ""
    role_intent_id: str = ""
    role_intent_label_zh: str = ""
    role_intent_prompt: str = ""
    task_primary_anchor: str = ""
    task_secondary_anchor: str = ""
    memory_bucket_id: str = ""
    core_persona: str = ""
    read_intent: str = ""
    write_intent: str = ""
    shared_user_state: dict[str, Any] = Field(default_factory=dict)
    user_role: str = ""
    role_preference_schema_id: str = ""
    role_preference_profile_id: str = ""
    role_preference_profile: dict[str, Any] = Field(default_factory=dict)
    condition: str = ""
    zone: str = ""
    phase: str = ""


class ScriptedRoundSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    feedback: str
    score: float = 0.5
    satisfied: bool = False
    history_relation: str = "new_issue"
    target_dimension: str = ""
    reference_dimension: str = ""
    inferred_reference_dimension: str = ""
    target_history_round: int | None = None
    target_history_feedback: str = ""
    prompt_guidance: str = ""


def resolve_task_intent(intent: str, instruction: str) -> str:
    explicit = str(intent or "").strip().lower()
    if explicit:
        return explicit

    text = str(instruction or "").lower()
    if "architecture" in text or "架构" in text:
        return "architecture_walkthrough"
    if "research" in text or "方法" in text:
        return "research_method_explanation"
    if "template" in text or "模板" in text:
        return "template_generation"
    if "modify" in text or "修改" in text:
        return "hard_modify"
    return "general"


def _round_feedback(profile: UserModelProfile, round_num: int, instruction: str) -> tuple[str, float, bool]:
    base_score = min(0.92, 0.55 + 0.08 * max(round_num - 1, 0))
    if profile.strictness >= 0.8:
        base_score -= 0.05
    if profile.style == "suggestive":
        base_score += 0.03
    score = max(0.0, min(0.99, base_score))
    satisfied = score >= profile.satisfaction_threshold

    if satisfied:
        return (
            "The deck is in good shape; the remaining gap is minor.",
            score,
            True,
        )

    lead = "Slide 1" if round_num <= 1 else f"Slide {round_num}"
    feedback = f"{lead} should tighten the main point because the current draft is still too diffuse."
    if "template" in instruction.lower():
        feedback = f"{lead} should align the template choice more closely with the task because the current look is still off."
    return feedback, score, False


@dataclass
class UserModel:
    profile: UserModelProfile
    scripted_rounds: list[ScriptedRoundSpec] | None = None

    @classmethod
    def from_config(
        cls,
        profile: UserModelProfile,
        strategy_config: dict[str, Any] | None = None,
    ) -> "UserModel":
        scripted_rounds: list[ScriptedRoundSpec] | None = None
        if strategy_config:
            raw_rounds = strategy_config.get("scripted_rounds")
            if isinstance(raw_rounds, list) and raw_rounds:
                scripted_rounds = [
                    ScriptedRoundSpec.model_validate(item)
                    for item in raw_rounds
                    if isinstance(item, dict)
                ]
        return cls(profile=profile, scripted_rounds=scripted_rounds)

    async def review(self, context: ReviewContext) -> UserModelResponse:
        if self.scripted_rounds:
            index = max(0, min(context.round_num - 1, len(self.scripted_rounds) - 1))
            spec = self.scripted_rounds[index]
            return UserModelResponse(
                feedback=spec.feedback,
                satisfied=bool(spec.satisfied),
                score=float(spec.score),
            )

        feedback, score, satisfied = _round_feedback(
            self.profile,
            context.round_num,
            context.instruction,
        )
        return UserModelResponse(feedback=feedback, satisfied=satisfied, score=score)


__all__ = [
    "ReviewContext",
    "ScriptedRoundSpec",
    "UserModel",
    "UserModelResponse",
    "resolve_task_intent",
]
