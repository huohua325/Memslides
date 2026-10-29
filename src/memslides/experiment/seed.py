from __future__ import annotations

from typing import Any

from memslides.experiment.config import UserModelProfile


BUILTIN_PERSONAS: dict[str, dict[str, Any]] = {
    "academic": {
        "description": "Academic reviewer",
        "prompt": "You prefer clear structure, sober colors, and precise claims.",
        "strictness": 0.75,
        "focus": ["content", "layout", "color"],
        "style": "directive",
        "language": "zh",
    },
    "business": {
        "description": "Business reviewer",
        "prompt": "You prefer concise messaging, a clear conclusion, and action-oriented slides.",
        "strictness": 0.65,
        "focus": ["layout", "content", "visual"],
        "style": "directive",
        "language": "zh",
    },
    "creative": {
        "description": "Creative reviewer",
        "prompt": "You prefer energetic visuals, variation, and stronger page rhythm.",
        "strictness": 0.60,
        "focus": ["color", "layout", "visual"],
        "style": "suggestive",
        "language": "zh",
    },
    "scholar": {
        "description": "Scholar persona",
        "prompt": "You care about rigorous logic, evidence, and restrained styling.",
        "strictness": 0.75,
        "focus": ["content", "layout", "color"],
        "style": "directive",
        "language": "zh",
    },
    "executive": {
        "description": "Executive persona",
        "prompt": "You care about conclusions first, strong contrast, and practical decisions.",
        "strictness": 0.65,
        "focus": ["layout", "content", "visual"],
        "style": "directive",
        "language": "zh",
    },
    "designer": {
        "description": "Designer persona",
        "prompt": "You care about composition, rhythm, and a strong visual identity.",
        "strictness": 0.60,
        "focus": ["color", "layout", "visual"],
        "style": "suggestive",
        "language": "zh",
    },
    "postsecondary_teacher": {
        "description": "Postsecondary teacher persona",
        "prompt": "You care about teachable structure, accurate terms, and clear progression.",
        "strictness": 0.74,
        "focus": ["content", "layout", "visual"],
        "style": "directive",
        "language": "zh",
    },
    "software_developer": {
        "description": "Software developer persona",
        "prompt": "You care about system boundaries, implementation logic, and readable diagrams.",
        "strictness": 0.72,
        "focus": ["content", "layout", "visual"],
        "style": "directive",
        "language": "zh",
    },
    "management_analyst": {
        "description": "Management consultant persona",
        "prompt": "You care about diagnosis, frameworks, and actionable recommendations.",
        "strictness": 0.73,
        "focus": ["content", "layout", "visual"],
        "style": "directive",
        "language": "zh",
    },
    "marketing_manager": {
        "description": "Marketing persona",
        "prompt": "You care about audience fit, brand feel, and narrative momentum.",
        "strictness": 0.66,
        "focus": ["visual", "content", "layout"],
        "style": "mixed",
        "language": "zh",
    },
    "graphic_designer": {
        "description": "Graphic designer persona",
        "prompt": "You care about visual hierarchy, spacing, and polished page composition.",
        "strictness": 0.61,
        "focus": ["color", "layout", "visual"],
        "style": "suggestive",
        "language": "zh",
    },
}


def get_user_model_profile(name: str) -> UserModelProfile:
    payload = dict(BUILTIN_PERSONAS.get(name, BUILTIN_PERSONAS["academic"]))
    return UserModelProfile(
        strictness=float(payload.get("strictness", 0.7)),
        focus=list(payload.get("focus", ["layout", "content", "visual"])),
        style=str(payload.get("style", "directive")),
        language=str(payload.get("language", "zh")),
        satisfaction_threshold=float(payload.get("satisfaction_threshold", 0.85) or 0.85),
        persona=name,
    )


__all__ = ["BUILTIN_PERSONAS", "get_user_model_profile"]
