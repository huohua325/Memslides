"""Implementation details."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from ..core.models import ChainExperience, ChainSegment

if TYPE_CHECKING:
    from ..store.chain_store import ChainStore

logger = logging.getLogger(__name__)


class ToolChainBuffer:
    """Implementation details."""

    def __init__(self) -> None:

        self._chains: dict[str, list[ChainSegment]] = defaultdict(list)
        self._experiences: dict[str, list[ChainExperience]] = defaultdict(list)

        self._ltm_cache: dict[str, list[ChainExperience]] = {}
        self._ltm_queried_tools: set[str] = set()



    def add_chains(self, chains: list[ChainSegment]) -> None:
        """Implementation details."""
        for chain in chains:
            self._chains[chain.chain_name].append(chain)

    def add_experience(self, experience: ChainExperience) -> None:
        """Implementation details."""
        self._experiences[experience.chain_name].append(experience)



    async def query_or_cache_from_ltm(
        self,
        tool_name: str,
        chain_store: ChainStore,
        user_id: str,
    ) -> list[ChainExperience]:
        """Implementation details."""
        if tool_name in self._ltm_cache:
            return self._ltm_cache[tool_name]

        if tool_name in self._ltm_queried_tools:
            return []


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



    def get_experiences_for_tool(self, tool_name: str) -> list[ChainExperience]:
        """Implementation details."""
        wm_exps: list[ChainExperience] = []
        for _sig, exps in self._experiences.items():
            for exp in exps:
                if tool_name in exp.tool_pipeline:
                    wm_exps.append(exp)

        ltm_exps = self._ltm_cache.get(tool_name, [])
        return wm_exps + ltm_exps

    def get_new_chains_for_consolidation(self) -> dict[str, list[ChainSegment]]:
        """Implementation details."""
        return dict(self._chains)

    def get_new_experiences_for_consolidation(self) -> dict[str, list[ChainExperience]]:
        """Implementation details."""
        return dict(self._experiences)



    def get_all_chains(self) -> dict[str, list[ChainSegment]]:
        """Implementation details."""
        return dict(self._chains)

    def get_all_experiences(self) -> dict[str, list[ChainExperience]]:
        """Implementation details."""
        return dict(self._experiences)

    def search_experiences(
        self, query: str, tool_name: str = "", top_k: int = 20,
    ) -> list[ChainExperience]:
        """Implementation details."""
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



    def release(self) -> None:
        """Implementation details."""
        self._chains.clear()
        self._experiences.clear()
        self._ltm_cache.clear()
        self._ltm_queried_tools.clear()
