from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from memslides.memory.core.template_models import TemplateProfile, TemplateSemanticModel


def materialize_template_shell_assets(
    template_profile: TemplateProfile,
    workspace: Path | str,
) -> tuple[TemplateProfile, dict[str, Any]]:
    """Copy reusable shell assets into the active workspace and rewrite paths.

    Template profiles may live in the shared template store, but generated HTML
    must only reference current-workspace assets. This returns a cloned profile
    whose template_shell background/decorative asset names are workspace-local.
    """

    workspace_path = Path(workspace).resolve()
    source_dir = Path(getattr(template_profile, "template_dir", "") or "")
    source_images = source_dir / "images" if source_dir else Path()
    target_dir = workspace_path / "template_shell" / "assets"
    target_dir.mkdir(parents=True, exist_ok=True)

    profile_data = deepcopy(template_profile.to_dict())
    semantic_payload = deepcopy(profile_data.get("semantic_model", {}) or {})
    shell_model = semantic_payload.get("template_shell", {}) or {}
    layouts = shell_model.get("layouts", {}) if isinstance(shell_model, dict) else {}
    copied: list[dict[str, str]] = []
    missing: list[str] = []

    def materialize_asset(asset_name: str) -> str:
        name = str(asset_name or "").strip()
        if not name:
            return ""
        candidate = Path(name)
        if candidate.is_absolute() and candidate.exists():
            src = candidate
        else:
            src = source_images / name
        if not src.exists():
            missing.append(name)
            return name
        dest = target_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        rel = dest.relative_to(workspace_path).as_posix()
        copied.append({"source": str(src), "workspace_path": rel})
        return rel

    if isinstance(layouts, dict):
        for payload in layouts.values():
            if not isinstance(payload, dict):
                continue
            for key in ("background_assets", "decorative_layers"):
                for item in payload.get(key, []) or []:
                    if not isinstance(item, dict) or not item.get("asset_name"):
                        continue
                    item["asset_name"] = materialize_asset(str(item.get("asset_name", "")))

    semantic_payload["template_shell"] = shell_model
    profile_data["semantic_model"] = semantic_payload
    cloned = TemplateProfile.from_dict(profile_data)

    audit = {
        "workspace": str(workspace_path),
        "target_dir": str(target_dir),
        "copied": copied,
        "missing": sorted(set(missing)),
        "layout_count": len(layouts) if isinstance(layouts, dict) else 0,
    }
    audit_path = workspace_path / "template_shell" / "template_shell_assets.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return cloned, audit


__all__ = ["materialize_template_shell_assets"]
