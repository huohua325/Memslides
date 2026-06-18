from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memslides.templates.semantic_access import canonical_layout_names


TEMPLATE_RUNTIME_STATE_FILE = ".template_runtime_state.json"


@dataclass
class TemplateRuntimeState:
    """Canonical runtime switch for template-driven tool exposure."""

    active: bool = False
    mode: str = "disabled"
    selected_template_id: str = ""
    selected_template_name: str = ""
    profile_path: str = ""
    quality_path: str = ""
    structure_score: float | None = None
    visual_safety_score: float | None = None
    canonical_layout_names: list[str] = field(default_factory=list)
    selection_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["canonical_layout_names"] = _unique_nonempty(self.canonical_layout_names)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TemplateRuntimeState":
        payload = payload or {}
        return cls(
            active=bool(payload.get("active", False)),
            mode=str(payload.get("mode", "") or "disabled"),
            selected_template_id=str(payload.get("selected_template_id", "") or ""),
            selected_template_name=str(payload.get("selected_template_name", "") or ""),
            profile_path=str(payload.get("profile_path", "") or ""),
            quality_path=str(payload.get("quality_path", "") or ""),
            structure_score=payload.get("structure_score"),
            visual_safety_score=payload.get("visual_safety_score"),
            canonical_layout_names=_unique_nonempty(
                payload.get("canonical_layout_names", []) or []
            ),
            selection_source=str(payload.get("selection_source", "") or ""),
        )


def _unique_nonempty(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def template_runtime_workspace(workspace: Path | str | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).resolve()
    env_workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
    if env_workspace:
        return Path(env_workspace).resolve()
    return Path.cwd().resolve()


def template_runtime_state_path(workspace: Path | str | None = None) -> Path:
    return template_runtime_workspace(workspace) / TEMPLATE_RUNTIME_STATE_FILE


def load_template_runtime_state(
    workspace: Path | str | None = None,
) -> TemplateRuntimeState | None:
    path = template_runtime_state_path(workspace)
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return TemplateRuntimeState.from_dict(payload)
    except Exception:
        return None


def save_template_runtime_state(
    workspace: Path | str,
    state: TemplateRuntimeState,
) -> Path:
    path = template_runtime_state_path(workspace)
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def clear_template_runtime_state(workspace: Path | str | None = None) -> None:
    path = template_runtime_state_path(workspace)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def canonical_layout_names_from_profile(template_profile: Any) -> list[str]:
    return _unique_nonempty(canonical_layout_names(template_profile))
