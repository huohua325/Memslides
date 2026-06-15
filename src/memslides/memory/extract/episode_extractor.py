"""
EpisodeExtractor — 情景记忆提取器 (Stage 2)

三阶段 Pipeline:
  预过滤 (确定性) → LLM 提取 → 后过滤 (确定性)

从 CompactRound batch 中提取 DesignEpisode。
"""

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
# Prompt 模板 — 迁移到 memory/prompts/extraction.py
from ..prompts import EPISODE_EXTRACTION_PROMPT


# ═══════════════════════════════════════════════
# 后过滤: 可观测状态正则模式
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
    """情景记忆提取器 — 从 CompactRound batch 中提取 DesignEpisode

    三阶段 Pipeline:
    1. _pre_filter: 移除无信息量的轮次 (0 LLM)
    2. _do_extract: LLM 提取因果链 Episode
    3. _post_filter: 移除含可观测状态的 Episode (0 LLM)
    """

    def __init__(
        self,
        llm: Any = None,
        episode_store: Any = None,
        event_bus: Any = None,
        debug_tracer: Any = None,
        embedding_func: Any = None,
        extractions_dir: Path | str | None = None,  # Stage 4 新增
        preference_extractor: Any = None,  # 已废弃，保留签名兼容性
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._store = episode_store
        self._event_bus = event_bus
        self._debug_tracer = debug_tracer
        self._embed = embedding_func
        # preference_extractor 已废弃，不再使用
        self._stats: dict[str, int] = {
            "total_candidates": 0,
            "observable_rejected": 0,
            "unknown_reaction_rejected": 0,
            "empty_insight_rejected": 0,
            "kept": 0,
        }
        self._batch_counter: int = 0
        # Stage 4 新增: 中间产物写入器
        self._extraction_writer: ExtractionArtifactWriter | None = None
        if extractions_dir:
            try:
                self._extraction_writer = ExtractionArtifactWriter(Path(extractions_dir).parent)
            except Exception as e:
                logger.warning("Failed to init ExtractionArtifactWriter: %s", e)

    # ── 预过滤 (确定性) ──

    def _pre_filter(self, batch: list) -> list | None:
        """确定性预过滤 — 移除无信息量的轮次"""
        filtered = [r for r in batch if self._should_extract_round(r)]
        return filtered if filtered else None

    @staticmethod
    def _should_extract_round(compact: Any) -> bool:
        """单轮预过滤条件"""
        # ≥2 次错误 → 保留
        if sum(1 for s in compact.segments if s.is_error) >= 2:
            return True
        # 涉及 ≥3 个不同文件 → 保留
        if len(set(s.target_file for s in compact.segments if s.target_file)) >= 3:
            return True
        # 仅 ≤2 个 segments → 跳过
        if len(compact.segments) <= 2:
            return False
        return True

    # ── LLM 提取 ──

    async def _do_extract(self, filtered_batch: list, **kwargs: Any) -> list[dict]:
        """LLM 提取 — 使用 EPISODE_EXTRACTION_PROMPT

        Stage 4 改造: 保存完整 LLM 输入/输出。
        """
        batch_text = "\n---\n".join(r.to_extraction_text() for r in filtered_batch)
        prompt = EPISODE_EXTRACTION_PROMPT.format(batch_text=batch_text)

        if not self._llm:
            logger.warning("EpisodeExtractor: no LLM configured, returning empty")
            return []

        try:
            response = await self._llm(prompt)
            parsed = self._parse_response(response)

            # Stage 4 新增: 保存完整 LLM I/O
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
        """解析 LLM JSON 响应"""
        try:
            # 尝试直接解析
            data = json.loads(response)
            if isinstance(data, dict) and "episodes" in data:
                return data["episodes"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取 JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "episodes" in data:
                    return data["episodes"]
            except json.JSONDecodeError:
                pass

        # 尝试寻找 JSON 对象
        brace_match = re.search(r"\{[^{}]*\"episodes\"[^{}]*\[.*?\]\s*\}", response, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                return data.get("episodes", [])
            except json.JSONDecodeError:
                pass

        logger.warning("EpisodeExtractor: failed to parse LLM response")
        return []

    # ── 后过滤 (确定性) ──

    def _post_filter(self, episode: dict) -> bool:
        """确定性后过滤 — 丢弃含可观测状态或低质量的 Episode"""
        self._stats["total_candidates"] += 1

        # 必须有 design_insight
        if not episode.get("design_insight", "").strip():
            self._stats["empty_insight_rejected"] += 1
            return False

        # 检查可观测状态模式
        insight = episode["design_insight"].lower()
        for pattern in OBSERVABLE_PATTERNS:
            if re.search(pattern, insight):
                self._stats["observable_rejected"] += 1
                return False

        self._stats["kept"] += 1
        return True

    @property
    def observable_reject_rate(self) -> float:
        """可观测状态拒绝率 — 持续 >50% 说明 LLM Prompt 需要调优"""
        total = self._stats["total_candidates"]
        if total == 0:
            return 0.0
        return self._stats["observable_rejected"] / total

    @property
    def stats(self) -> dict:
        """返回提取统计信息"""
        s = dict(self._stats)
        s["observable_reject_rate"] = self.observable_reject_rate
        return s

    # ── 端到端提取 + 存储 ──

    async def extract_and_store(
        self,
        batch: list,
        user_id: str = "",
        session_id: str = "",
        *,
        store: bool = True,
    ) -> list[DesignEpisode]:
        """端到端: 提取 → 构建 DesignEpisode → (可选)存储 → 发布事件

        Args:
            store: 为 False 时仅返回 DesignEpisode 列表，不写入 LTM。
                   memory_v2 路径下由调用方暂存到 WM。
        """
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

        # 注: 原 Stage 4 实时 AtomicPreference 提取路径已废弃。
        # 偏好由 Consolidator._consolidate_preferences() 在 Job 结束时统一写 UserProfile。

        return stored
