from __future__ import annotations

import os
import re

_UNRESOLVED_ENV_PATTERN = re.compile(
    r"^\$(?:\{)?([A-Z][A-Z0-9_]*)(?::-[^}]*)?(?:\})?$"
)


def is_unresolved_env_placeholder(value: str | None) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    return _UNRESOLVED_ENV_PATTERN.fullmatch(raw) is not None


def getenv_optional(name: str, default: str = "") -> str:
    value = str(os.getenv(name, "") or "").strip()
    if not value:
        return default
    if is_unresolved_env_placeholder(value):
        return default
    return value
