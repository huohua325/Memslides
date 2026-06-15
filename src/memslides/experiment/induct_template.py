from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memslides.experiment.config import ExperimentConfig
from memslides.session import MemSlidesSession
from memslides.utils.constants import WORKSPACE_BASE
from memslides.utils.log import isolated_context_logger


@dataclass
class TemplateInductionResult:
    template_file: str
    profile_id: str = ""
    profile_name: str = ""
    narrative_style: str = ""
    stored: bool = False
    workspace: str = ""
    output_dir: str = ""
    profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def induct_template(
    template_file: Path | str,
    *,
    user_id: str = "default",
    narrative_style: str | None = None,
    auto_infer_style: bool = True,
    output_dir: Path | str | None = None,
    workspace: Path | str | None = None,
    config_file: Path | str | None = None,
    memory_db_dir: Path | str | None = None,
) -> TemplateInductionResult:
    template_path = Path(template_file).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    session_id = f"template_{uuid.uuid4().hex[:8]}"
    workspace_path = Path(workspace) if workspace else (WORKSPACE_BASE / "template_induct" / session_id)
    workspace_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) if output_dir else workspace_path / "template_induction"
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        experiment_id=session_id,
        user_id=user_id,
        instruction=f"template induction: {template_path.name}",
        attachments=[],
        memory_enabled=True,
        memory_mode="global",
        memory_db_dir=Path(memory_db_dir) if memory_db_dir else output_path / "memory_db",
        config_file=Path(config_file) if config_file else None,
        output_dir=output_path,
        user_model_mode="auto",
    )

    with isolated_context_logger():
        session = MemSlidesSession(options=cfg.build_session_options(
            workspace=workspace_path,
            memory_db_dir=cfg.resolve_memory_db_dir(workspace_path),
        ))
        try:
            session.config.memory.artifact_trace = True
            await session._ensure_memory_runtime()

            profile = await session.runtime._analyze_template(str(template_path))
            if profile is None:
                raise RuntimeError(f"Template analysis failed: {template_path}")

            if narrative_style:
                try:
                    if getattr(profile, "content_patterns", None):
                        profile.content_patterns.narrative_style = narrative_style
                    store = getattr(session.memory_runtime.system, "template_store", None) if session.memory_runtime else None
                    if store:
                        await store.add(profile)
                except Exception:
                    pass
            elif auto_infer_style:
                try:
                    store = getattr(session.memory_runtime.system, "template_store", None) if session.memory_runtime else None
                    if store:
                        await store.add(profile)
                except Exception:
                    pass

            result = TemplateInductionResult(
                template_file=str(template_path),
                profile_id=getattr(profile, "id", "") or "",
                profile_name=getattr(profile, "name", "") or template_path.stem,
                narrative_style=str(getattr(getattr(profile, "content_patterns", None), "narrative_style", "") or narrative_style or ""),
                stored=True,
                workspace=str(workspace_path),
                output_dir=str(output_path),
                profile=profile.model_dump() if hasattr(profile, "model_dump") else profile.to_dict() if hasattr(profile, "to_dict") else None,
            )
            (output_path / "template_induction.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            template_dir = Path(getattr(profile, "template_dir", "") or "")
            if template_dir.exists() and template_dir.resolve() != output_path.resolve():
                _mirror_template_artifacts(template_dir, output_path)
            return result
        finally:
            await session.close()


def _mirror_template_artifacts(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("source.pptx", "analysis.json", "slide_induction.json", "semantic_model.json", "template_audit.json", "image_stats.json", "shape_geometry.json", "description.txt"):
        source_file = source_dir / name
        if source_file.exists():
            shutil.copy2(source_file, target_dir / name)
    images_dir = source_dir / "images"
    if images_dir.exists():
        target_images = target_dir / "images"
        if target_images.exists():
            shutil.rmtree(target_images, ignore_errors=True)
        shutil.copytree(images_dir, target_images)


__all__ = ["TemplateInductionResult", "induct_template"]
