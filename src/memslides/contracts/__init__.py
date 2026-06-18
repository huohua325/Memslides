from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from memslides.utils.typings import ConvertType, PowerPointType


class MemoryOptions(BaseModel):
    enabled: bool = True
    user_id: str = "default"
    global_db_dir: Path | None = None


class TemplateOptions(BaseModel):
    template: Path | None = None
    template_id: str | None = None
    template_as_reference: bool = False


class SessionOptions(BaseModel):
    config_file: Path | None = None
    workspace: Path | None = None
    session_id: str | None = None
    language: Literal["zh", "en"] = "en"
    memory: MemoryOptions = Field(default_factory=MemoryOptions)
    template: TemplateOptions = Field(default_factory=TemplateOptions)
    check_llms: bool = False
    api_profile_id: str = ""
    service_profile_id: str = ""
    runtime_llm_profile: dict[str, Any] = Field(default_factory=dict)
    runtime_service_profile: dict[str, Any] = Field(default_factory=dict)


class DeckRequest(BaseModel):
    instruction: str
    attachments: list[Path] = Field(default_factory=list)
    num_pages: int | str | None = None
    language: Literal["zh", "en"] | None = None
    memory_intent: str = ""
    template: Path | None = None
    template_id: str | None = None
    template_as_reference: bool | None = None
    powerpoint_type: PowerPointType = PowerPointType.WIDE_SCREEN
    convert_type: ConvertType = ConvertType.MEMSLIDES
    extra_info: dict[str, Any] = Field(default_factory=dict)


class RevisionRequest(BaseModel):
    feedback: str
    memory_intent: str = ""
    extra_info: dict[str, Any] = Field(default_factory=dict)


class DeckResult(BaseModel):
    session_id: str
    workspace: Path
    final_path: Path | None = None
    pptx_path: Path | None = None
    pdf_path: Path | None = None
    slide_html_dir: Path | None = None
    manuscript_path: Path | None = None
    intermediate: dict[str, str] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)


__all__ = [
    "DeckRequest",
    "RevisionRequest",
    "DeckResult",
    "SessionOptions",
    "MemoryOptions",
    "TemplateOptions",
]
