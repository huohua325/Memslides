from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from memslides.contracts import DeckRequest, MemoryOptions, RevisionRequest, SessionOptions, TemplateOptions
from memslides.utils.constants import DEFAULT_CACHE_BASE
from memslides.utils.typings import ConvertType


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "memslides").is_dir():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "PPT").is_dir():
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "MemSlides").is_dir():
            return candidate
    return start.parent


def _resolve_path(
    value: str | Path,
    *,
    suite_dir: Path,
    project_root: Path,
    prefer_existing: bool = True,
) -> Path:
    raw_value = os.path.expandvars(str(value)).strip()
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path

    suite_candidate = (suite_dir / path).resolve()
    project_candidate = (project_root / path).resolve()
    workspace_candidate = (project_root.parent / path).resolve()
    if prefer_existing:
        if suite_candidate.exists():
            return suite_candidate
        if project_candidate.exists():
            return project_candidate
        if workspace_candidate.exists():
            return workspace_candidate
    return project_candidate


def _resolve_suite_path(suite_path: Path | str) -> Path:
    candidate = Path(suite_path)
    if candidate.exists():
        return candidate

    builtin_dir = Path(__file__).resolve().parent / "suites"
    names: list[str] = [candidate.name]
    if not candidate.suffix:
        names.extend([f"{candidate.name}.yaml", f"{candidate.name}.yml"])

    for name in names:
        builtin_candidate = builtin_dir / name
        if builtin_candidate.exists():
            return builtin_candidate

    raise FileNotFoundError(candidate)


class UserModelProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    strictness: float = Field(default=0.7, ge=0.0, le=1.0)
    focus: list[str] = Field(default_factory=lambda: ["layout", "content", "visual"])
    style: str = "directive"
    language: Literal["zh", "en"] = "zh"
    satisfaction_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    persona: str = ""

    @field_validator("style")
    @classmethod
    def validate_style(cls, value: str) -> str:
        allowed = {"directive", "suggestive", "mixed"}
        if value not in allowed:
            raise ValueError(f"style must be one of {sorted(allowed)}")
        return value


class SeedTemplateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: Path
    narrative_style: str | None = None
    auto_infer_style: bool = True


TOOL_MEMORY_ONLY_MODULE_OVERRIDES: dict[str, bool] = {
    "profile_injection": False,
    "wm_preference_injection": False,
    "wm_experience_injection": False,
    "wm_round_history_injection": False,
    "wm_task_history_injection": False,
    "ltm_tool_experience_injection": True,
    "experience_preload": False,
}

NO_INJECTION_MODULE_OVERRIDES: dict[str, bool] = {
    **TOOL_MEMORY_ONLY_MODULE_OVERRIDES,
    "ltm_tool_experience_injection": False,
}


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    user_id: str = "default"
    instruction: str = ""
    attachments: list[Path] = Field(default_factory=list)
    num_pages: int | str | None = None
    max_rounds: int = 1
    round_timeout: int | None = None
    memory_enabled: bool = True
    memory_mode: str = "global"
    memory_artifact_trace: bool = False
    memory_db_dir: Path | None = None
    reset_memory_db: bool = False
    user_persona: str = ""
    force_all_rounds: bool = False
    require_pptx_export: bool = False
    user_model_mode: str = "auto"
    user_model_profile: UserModelProfile = Field(default_factory=UserModelProfile)
    user_model_strategy_config: dict[str, Any] = Field(default_factory=dict)
    seed_templates: list[SeedTemplateConfig] = Field(default_factory=list)
    template_file: Path | None = None
    template_id: str | None = None
    template_as_reference: bool | None = None
    template_narrative_style: str | None = None
    template_auto_infer_style: bool = True
    memory_intent: str = ""
    language: Literal["zh", "en"] = "zh"
    check_llms: bool = False
    config_file: Path | None = None
    output_dir: Path | None = None
    powerpoint_type: str | None = None
    extra_info: dict[str, Any] = Field(default_factory=dict)

    @field_validator("attachments", mode="before")
    @classmethod
    def _coerce_attachments(cls, value: Any) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) if not isinstance(item, Path) else item for item in value]

    @field_validator("seed_templates", mode="before")
    @classmethod
    def _coerce_seed_templates(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        return list(value)

    def resolved_user_model_profile(self) -> UserModelProfile:
        return self.user_model_profile

    def resolve_output_dir(self, suite_output_dir: Path) -> Path:
        return Path(self.output_dir or suite_output_dir / self.experiment_id).expanduser().resolve()

    def resolve_memory_db_dir(self, output_dir: Path) -> Path:
        if self.memory_db_dir:
            return Path(self.memory_db_dir)
        return (output_dir / "memory_db").resolve()

    def normalized_extra_info(self) -> dict[str, Any]:
        payload = dict(self.extra_info)
        payload.setdefault("experiment_id", self.experiment_id)
        payload.setdefault("user_id", self.user_id)
        payload.setdefault("memory_mode", self.memory_mode)
        module_overrides = dict(payload.get("memory_module_overrides") or {})
        module_overrides.update(self.memory_module_overrides())
        if module_overrides:
            payload["memory_module_overrides"] = module_overrides
        if self.user_persona:
            payload.setdefault("user_persona", self.user_persona)
            payload.setdefault("core_persona", self.user_persona)
        if self.memory_intent:
            payload.setdefault("memory_intent", self.memory_intent)
        return payload

    def memory_module_overrides(self) -> dict[str, bool]:
        mode = str(self.memory_mode or "").strip().lower().replace("-", "_")
        if mode == "tool_only":
            return dict(TOOL_MEMORY_ONLY_MODULE_OVERRIDES)
        if mode == "no_injection":
            return dict(NO_INJECTION_MODULE_OVERRIDES)
        return {}

    def build_session_options(self, *, workspace: Path, memory_db_dir: Path) -> SessionOptions:
        template_as_reference = self.template_as_reference
        if template_as_reference is None:
            template_as_reference = bool(self.template_file)

        return SessionOptions(
            config_file=Path(self.config_file) if self.config_file else None,
            workspace=workspace,
            session_id=self.experiment_id,
            language=self.language,
            memory=MemoryOptions(
                enabled=bool(self.memory_enabled),
                user_id=self.user_id,
                global_db_dir=memory_db_dir,
            ),
            template=TemplateOptions(
                template=Path(self.template_file) if self.template_file else None,
                template_id=self.template_id,
                template_as_reference=bool(template_as_reference),
            ),
            check_llms=bool(self.check_llms),
        )

    def build_deck_request(self) -> DeckRequest:
        template_as_reference = self.template_as_reference
        if template_as_reference is None:
            template_as_reference = bool(self.template_file)

        return DeckRequest(
            instruction=self.instruction,
            attachments=[Path(path) for path in self.attachments],
            num_pages=self.num_pages,
            language=self.language,
            memory_intent=self.memory_intent,
            template=Path(self.template_file) if self.template_file else None,
            template_id=self.template_id,
            template_as_reference=bool(template_as_reference),
            convert_type=ConvertType.MEMSLIDES,
            extra_info=self.normalized_extra_info(),
        )

    def build_revision_request(self, feedback: str) -> RevisionRequest:
        return RevisionRequest(
            feedback=feedback,
            memory_intent=self.memory_intent,
            extra_info=self.normalized_extra_info(),
        )


class ExperimentSuite(BaseModel):
    model_config = ConfigDict(extra="allow")

    suite_id: str
    output_base: Path = DEFAULT_CACHE_BASE
    max_parallel: int = 1
    experiments: list[ExperimentConfig] = Field(default_factory=list)
    _suite_output_dir: Path | None = PrivateAttr(default=None)
    _suite_run_stamp: str | None = PrivateAttr(default=None)

    @classmethod
    def from_yaml(cls, suite_path: Path | str) -> "ExperimentSuite":
        suite_path = _resolve_suite_path(suite_path)
        with suite_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Suite YAML must contain a mapping: {suite_path}")

        project_root = _find_project_root(suite_path.resolve())
        suite_dir = suite_path.resolve().parent

        resolved = dict(raw)
        output_base = resolved.get("output_base")
        if isinstance(output_base, str) and output_base:
            resolved["output_base"] = _resolve_path(
                output_base,
                suite_dir=suite_dir,
                project_root=project_root,
                prefer_existing=False,
            )

        experiments = resolved.get("experiments") or []
        if isinstance(experiments, list):
            resolved_experiments: list[dict[str, Any]] = []
            for item in experiments:
                if not isinstance(item, dict):
                    continue
                resolved_item = dict(item)

                for key in ("template_file", "config_file", "output_dir", "memory_db_dir"):
                    raw_value = resolved_item.get(key)
                    if isinstance(raw_value, str) and raw_value:
                        resolved_item[key] = _resolve_path(
                            raw_value,
                            suite_dir=suite_dir,
                            project_root=project_root,
                        )

                raw_attachments = resolved_item.get("attachments")
                if isinstance(raw_attachments, list):
                    resolved_item["attachments"] = [
                        _resolve_path(path, suite_dir=suite_dir, project_root=project_root)
                        if isinstance(path, (str, Path))
                        else path
                        for path in raw_attachments
                    ]

                raw_seed_templates = resolved_item.get("seed_templates")
                if isinstance(raw_seed_templates, list):
                    normalized_seed_templates: list[dict[str, Any]] = []
                    for seed in raw_seed_templates:
                        if not isinstance(seed, dict):
                            continue
                        normalized_seed = dict(seed)
                        seed_path = normalized_seed.get("path")
                        if isinstance(seed_path, str) and seed_path:
                            normalized_seed["path"] = _resolve_path(
                                seed_path,
                                suite_dir=suite_dir,
                                project_root=project_root,
                            )
                        normalized_seed_templates.append(normalized_seed)
                    resolved_item["seed_templates"] = normalized_seed_templates

                raw_profile = resolved_item.get("user_model_profile")
                if isinstance(raw_profile, dict):
                    resolved_item["user_model_profile"] = raw_profile

                resolved_experiments.append(resolved_item)
            resolved["experiments"] = resolved_experiments

        return cls.model_validate(resolved)

    def _build_suite_output_dir(self) -> Path:
        base = Path(self.output_base).expanduser().resolve()
        run_stamp = self._suite_run_stamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        if self._suite_run_stamp is None:
            self._suite_run_stamp = run_stamp
        date_part, time_part = run_stamp.split("_", 1)
        candidate = base / date_part / f"{self.suite_id}_{time_part}"
        suffix = 1
        while candidate.exists():
            candidate = base / date_part / f"{self.suite_id}_{time_part}_{suffix:02d}"
            suffix += 1
        return candidate

    def suite_output_dir(self, *, reuse_existing: bool = False) -> Path:
        if self._suite_output_dir is not None:
            return self._suite_output_dir

        if reuse_existing:
            existing = self.latest_suite_output_dir()
            if existing is not None:
                self._suite_output_dir = existing
                return existing

        self._suite_output_dir = self._build_suite_output_dir()
        return self._suite_output_dir

    def latest_suite_output_dir(self) -> Path | None:
        base = Path(self.output_base).expanduser().resolve()
        if not base.exists():
            return None
        matches: list[Path] = []
        for date_dir in base.iterdir():
            if not date_dir.is_dir():
                continue
            matches.extend(
                path
                for path in date_dir.glob(f"{self.suite_id}_*")
                if path.is_dir()
            )
        if not matches:
            return None
        matches.sort(key=lambda path: (path.stat().st_mtime, path.as_posix()))
        return matches[-1]


__all__ = [
    "ExperimentConfig",
    "ExperimentSuite",
    "NO_INJECTION_MODULE_OVERRIDES",
    "SeedTemplateConfig",
    "TOOL_MEMORY_ONLY_MODULE_OVERRIDES",
    "UserModelProfile",
]
