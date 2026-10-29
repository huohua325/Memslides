"""Implementation details."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ..core.models import DesignEpisode
    from ..collect.artifact_writer import ExtractionArtifactWriter
except ImportError:
    import sys, pathlib
    _core = str(pathlib.Path(__file__).resolve().parent.parent / "core")
    _collect = str(pathlib.Path(__file__).resolve().parent.parent / "collect")
    for p in (_core, _collect):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import DesignEpisode  # type: ignore[no-redef]
    from artifact_writer import ExtractionArtifactWriter  # type: ignore[no-redef]


# ═══════════════════════════════════════════════

from ..prompts import EPISODE_EXTRACTION_PROMPT


# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

OBSERVABLE_PATTERNS = [
    r"font.?size.*\d+",
    r"color.*#[0-9a-f]",
    r"changed to \d+",
    r"set to \d+",
    r"background.*#[0-9a-f]",
    r"\d+px",
]


# ═══════════════════════════════════════════════
# EpisodeExtractor
# ═══════════════════════════════════════════════

class EpisodeExtractor:
    """Implementation details."""

    def __init__(
        self,
        llm: Any = None,
        episode_store: Any = None,
        event_bus: Any = None,
        debug_tracer: Any = None,
        embedding_func: Any = None,
        extractions_dir: Path | str | None = None,
        preference_extractor: Any = None,
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._store = episode_store
        self._event_bus = event_bus
        self._debug_tracer = debug_tracer
        self._embed = embedding_func

        self._stats: dict[str, int] = {
            "total_candidates": 0,
            "observable_rejected": 0,
            "unknown_reaction_rejected": 0,
            "empty_insight_rejected": 0,
            "kept": 0,
        }
        self._batch_counter: int = 0

        self._extraction_writer: ExtractionArtifactWriter | None = None
        if extractions_dir:
            try:
                self._extraction_writer = ExtractionArtifactWriter(Path(extractions_dir).parent)
            except Exception as e:
                logger.warning("Failed to init ExtractionArtifactWriter: %s", e)



    def _pre_filter(self, batch: list) -> list | None:
        """Implementation details."""
        filtered = [r for r in batch if self._should_extract_round(r)]
        return filtered if filtered else None

    @staticmethod
    def _should_extract_round(compact: Any) -> bool:
        """Implementation details."""

        if sum(1 for s in compact.segments if s.is_error) >= 2:
            return True

        if len(set(s.target_file for s in compact.segments if s.target_file)) >= 3:
            return True

        if len(compact.segments) <= 2:
            return False
        return True



    async def _do_extract(self, filtered_batch: list, **kwargs: Any) -> list[dict]:
        """Implementation details."""
        batch_text = "\n---\n".join(r.to_extraction_text() for r in filtered_batch)
        prompt = EPISODE_EXTRACTION_PROMPT.format(batch_text=batch_text)

        if not self._llm:
            logger.warning("EpisodeExtractor: no LLM configured, returning empty")
            return []

        try:
            response = await self._llm(prompt)
            parsed = self._parse_response(response)


            if self._extraction_writer:
                try:
                    round_ids = [getattr(r, 'round_id', 0) for r in filtered_batch]
                    self._extraction_writer.write_episode_extraction(
                        round_ids=round_ids,
                        prompt=prompt,
                        response=response,
                        stored_episodes=parsed,
                    )
                except Exception as e:
                    logger.debug("Failed to write extraction artifacts: %s", e)

            return parsed
        except Exception as e:
            logger.error("EpisodeExtractor LLM call failed: %s", e)
            return []

    @staticmethod
    def _parse_response(response: str) -> list[dict]:
        """Implementation details."""
        try:

            data = json.loads(response)
            if isinstance(data, dict) and "episodes" in data:
                return data["episodes"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass


        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "episodes" in data:
                    return data["episodes"]
            except json.JSONDecodeError:
                pass


        brace_match = re.search(r"\{[^{}]*\"episodes\"[^{}]*\[.*?\]\s*\}", response, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                return data.get("episodes", [])
            except json.JSONDecodeError:
                pass

        logger.warning("EpisodeExtractor: failed to parse LLM response")
        return []



    def _post_filter(self, episode: dict) -> bool:
        """Implementation details."""
        self._stats["total_candidates"] += 1


        if not episode.get("design_insight", "").strip():
            self._stats["empty_insight_rejected"] += 1
            return False


        insight = episode["design_insight"].lower()
        for pattern in OBSERVABLE_PATTERNS:
            if re.search(pattern, insight):
                self._stats["observable_rejected"] += 1
                return False

        self._stats["kept"] += 1
        return True

    @property
    def observable_reject_rate(self) -> float:
        """Implementation details."""
        total = self._stats["total_candidates"]
        if total == 0:
            return 0.0
        return self._stats["observable_rejected"] / total

    @property
    def stats(self) -> dict:
        """Implementation details."""
        s = dict(self._stats)
        s["observable_reject_rate"] = self.observable_reject_rate
        return s



    async def extract_and_store(
        self,
        batch: list,
        user_id: str = "",
        session_id: str = "",
        *,
        store: bool = True,
    ) -> list[DesignEpisode]:
        """Implementation details."""
        _bid = self._batch_counter

        # Pre-filter (deterministic)
        filtered = self._pre_filter(batch)

        if not filtered:
            self._batch_counter += 1
            return []

        # LLM extract
        raw_episodes = await self._do_extract(filtered)

        if not raw_episodes:
            self._batch_counter += 1
            return []

        # Post-filter (deterministic)
        kept_eps = [ep for ep in raw_episodes if self._post_filter(ep)]

        # Build DesignEpisode objects + store
        stored: list[DesignEpisode] = []
        for ep_dict in kept_eps:
            _confidence = 0.5 if ep_dict.get("_low_confidence") else 0.7
            episode = DesignEpisode(
                user_id=user_id,
                session_id=session_id,
                source_round_id=ep_dict.get("source_round_id", 0),
                user_intent=ep_dict.get("user_intent", ""),
                interpretation_gap=ep_dict.get("interpretation_gap", ""),
                action_outcome=ep_dict.get("action_outcome", ""),
                design_insight=ep_dict.get("design_insight", ""),
                category=ep_dict.get("category", ""),
                confidence=_confidence,
            )

            if store and self._store:
                try:
                    await self._store.add_episode(episode)
                except Exception as e:
                    logger.error("Failed to store episode %s: %s", episode.id, e)

            stored.append(episode)

            # Per-episode event (only when storing to LTM)
            if store and self._event_bus:
                try:
                    await self._event_bus.publish("episode_stored", {
                        "episode_id": episode.id,
                        "category": episode.category,
                    })
                except Exception:
                    pass

        # Batch-level event (only when storing to LTM)
        if store and stored and self._event_bus:
            try:
                await self._event_bus.publish("episode_extracted", {
                    "episodes": [e.to_dict() for e in stored],
                    "user_id": user_id,
                    "session_id": session_id,
                })
            except Exception as e:
                logger.warning("Failed to publish episode_extracted event: %s", e)

        self._batch_counter += 1
        logger.info("Extracted %d episodes from %d rounds (batch %d)",
                    len(stored), len(batch), _bid)




        return stored
