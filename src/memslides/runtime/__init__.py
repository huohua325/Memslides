from __future__ import annotations

from typing import Any

__all__ = [
    "AgentLoop",
    "IntentResolutionResult",
    "ModifyExecutionPlan",
    "ModifyToolPolicyPlan",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from memslides.runtime.agent_loop import AgentLoop, IntentResolutionResult, ModifyExecutionPlan
        from memslides.runtime.support import ModifyToolPolicyPlan

        exports = {
            "AgentLoop": AgentLoop,
            "IntentResolutionResult": IntentResolutionResult,
            "ModifyExecutionPlan": ModifyExecutionPlan,
            "ModifyToolPolicyPlan": ModifyToolPolicyPlan,
        }
        return exports[name]
    raise AttributeError(name)
