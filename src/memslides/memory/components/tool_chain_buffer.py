"""ToolChainBuffer — WM 中的工具链缓冲区（重构版）

变更要点：
- 移除阈值触发机制（纯累积）
- 新增 _ltm_cache / _ltm_queried_tools 支持 Operation 级实时匹配
- 新增 consolidation 专用接口（不含 LTM 缓存）

Requirements: 2.1, 2.2, 2.3, 3.1–3.5, 4.1
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from ..core.models import ChainExperience, ChainSegment, ToolChain

if TYPE_CHECKING:
    from ..store.chain_store import ChainStore

logger = logging.getLogger(__name__)


class ToolChainBuffer:
    """工具链缓冲区 — 纯累积 + LTM 缓存叠加。"""

    def __init__(self) -> None:
        # WM 新产生的数据
        self._chains: dict[str, list[ChainSegment]] = defaultdict(list)
        self._experiences: dict[str, list[ChainExperience]] = defaultdict(list)
        # LTM 缓存（按 tool_name 为键）
        self._ltm_cache: dict[str, list[ChainExperience]] = {}
        self._ltm_queried_tools: set[str] = set()

    # ── 写入 ──

    def add_chains(self, chains: list[ChainSegment]) -> None:
        """添加工具链，纯累积，不触发提取。"""
        for chain in chains:
            self._chains[chain.chain_name].append(chain)

    def add_experience(self, experience: ChainExperience) -> None:
        """添加提炼的工具链经验（WM 新产生）。"""
        self._experiences[experience.chain_name].append(experience)

    # ── LTM 缓存查询 ──

    async def query_or_cache_from_ltm(
        self,
        tool_name: str,
        chain_store: ChainStore,
        user_id: str,
    ) -> list[ChainExperience]:
        """查 WM 缓存 → 未命中则查 LTM → 写入缓存 → 返回经验。

        每个 tool_name 最多查 LTM 一次，后续全走 WM 缓存。
        """
        if tool_name in self._ltm_cache:
            return self._ltm_cache[tool_name]

        if tool_name in self._ltm_queried_tools:
            return []

        # 未查过 → 查 LTM
        self._ltm_queried_tools.add(tool_name)
        try:
            results = await chain_store.query_experiences_by_tool(
                tool_name, user_id,
            )
            if results:
                self._ltm_cache[tool_name] = results
            return results
        except Exception:
            logger.exception("Failed to query LTM for tool=%s", tool_name)
            return []

    # ── 读取 ──

    def get_experiences_for_tool(self, tool_name: str) -> list[ChainExperience]:
        """合并 WM 新经验 + LTM 缓存，WM 优先。"""
        wm_exps: list[ChainExperience] = []
        for _sig, exps in self._experiences.items():
            for exp in exps:
                if tool_name in exp.tool_pipeline:
                    wm_exps.append(exp)

        ltm_exps = self._ltm_cache.get(tool_name, [])
        return wm_exps + ltm_exps

    def get_new_chains_for_consolidation(self) -> dict[str, list[ChainSegment]]:
        """只返回 WM 新产生的链（不含 LTM 缓存），供 Consolidator 归档。"""
        return dict(self._chains)

    def get_new_experiences_for_consolidation(self) -> dict[str, list[ChainExperience]]:
        """只返回 WM 新产生的经验（不含 LTM 缓存），供 Consolidator 归档。"""
        return dict(self._experiences)

    # ── 兼容旧接口（供 job_manager / consolidator 快照使用）──

    def get_all_chains(self) -> dict[str, list[ChainSegment]]:
        """获取所有 WM 工具链（等同 get_new_chains_for_consolidation）。"""
        return dict(self._chains)

    def get_all_experiences(self) -> dict[str, list[ChainExperience]]:
        """获取所有 WM 经验（等同 get_new_experiences_for_consolidation）。"""
        return dict(self._experiences)

    def search_experiences(
        self, query: str, tool_name: str = "", top_k: int = 20,
    ) -> list[ChainExperience]:
        """搜索工具链经验（兼容旧接口）。"""
        if tool_name:
            return self.get_experiences_for_tool(tool_name)[:top_k]

        results: list[ChainExperience] = []
        if query:
            query_lower = query.lower()
            for exps in self._experiences.values():
                for exp in exps:
                    if (
                        query_lower in exp.lesson.lower()
                        or query_lower in exp.applicable_when.lower()
                    ):
                        results.append(exp)
        return results[:top_k]

    # ── 生命周期 ──

    def release(self) -> None:
        """释放所有数据（含 LTM 缓存）。"""
        self._chains.clear()
        self._experiences.clear()
        self._ltm_cache.clear()
        self._ltm_queried_tools.clear()
