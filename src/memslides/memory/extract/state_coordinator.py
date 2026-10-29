"""State Coordinator: atomic update orchestration across components.

Provides:
- StateCoordinator: before/after modification flow with parameter extraction,
  versioned storage, and diff computation.

Modification flow:
    1. Extract pre-modification params (PPTStateExtractor)
    2. Save pre-modification snapshot (SlideParamsStore)
    3. Execute modification (Agent/Tool — external)
    4. Extract post-modification params (PPTStateExtractor)
    5. Save post-modification snapshot (SlideParamsStore)
    6. Return diff for downstream processing
"""

from __future__ import annotations

import uuid

from ..store.params_store import SlideParamsStore
from typing import Any
from .state_extractor import ExtractionResult, PPTStateExtractor


class StateCoordinator:
    """Implementation details."""

    def __init__(
        self,
        extractor: PPTStateExtractor,
        params_store: SlideParamsStore,
        factual_memory: Any = None,  # Unused, kept for API compatibility
    ):
        self.extractor = extractor
        self.params_store = params_store
        self.factual_memory = None  # Always None - kept for API compatibility

    async def before_modification(
        self, slide_source, slide_type: str = "html"
    ) -> ExtractionResult:
        """Implementation details."""
        result = self.extractor.extract(slide_source, slide_type)
        self.params_store.save_snapshot(result.slide_id, result)
        return result

    async def after_modification(
        self,
        slide_source,
        slide_type: str = "html",
        session_id: str = "",
        modification_id: str = "",
        rule_ids_applied: list[str] | None = None,
    ) -> tuple[ExtractionResult, dict]:
        """Implementation details."""
        result = self.extractor.extract(slide_source, slide_type)
        version = self.params_store.save_snapshot(result.slide_id, result)


        diff = {}
        if version > 1:
            diff = self.params_store.diff(result.slide_id, version - 1, version)


        if self.factual_memory and diff:
            prev_data = self.params_store.get_version(result.slide_id, version - 1)
            before_params = prev_data.get("params", {}) if prev_data else {}
            await self.factual_memory.record_modification(
                session_id=session_id,
                modification_id=modification_id or str(uuid.uuid4()),
                slide_index=self._parse_slide_index(result.slide_id),
                before_params=before_params,
                after_params=result.params,
                rule_ids_applied=rule_ids_applied or [],
            )

        return result, diff

    async def full_cycle(
        self,
        slide_source_before,
        slide_source_after,
        slide_type: str = "html",
        session_id: str = "",
        modification_id: str = "",
        rule_ids_applied: list[str] | None = None,
    ) -> tuple[ExtractionResult, ExtractionResult, dict]:
        """Implementation details."""
        before_result = await self.before_modification(slide_source_before, slide_type)
        after_result, diff = await self.after_modification(
            slide_source_after,
            slide_type,
            session_id,
            modification_id,
            rule_ids_applied,
        )
        return before_result, after_result, diff

    @staticmethod
    def _parse_slide_index(slide_id: str) -> int:
        try:
            return int(slide_id.split("_")[-1])
        except (ValueError, IndexError):
            return 0
