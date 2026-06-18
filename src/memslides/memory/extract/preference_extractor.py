"""
PreferenceExtractor — 偏好提取器

入口:
- extract_from_round(): 从 CompactRound 交互数据中提取偏好
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..core.models import AtomicPreference
from ..prompts import (
    ATOMIC_PREFERENCE_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


class PreferenceExtractor:
    """偏好提取器

    从交互历史 (round_data) 提取偏好 — 原 AtomicPreferenceExtractor
    """

    def __init__(
        self,
        llm: Any = None,
        store: Any = None,
        profile_evidence_pool: Any = None,
        user_id: str = "",
        artifact_writer: Any = None,
    ):
        self.llm = llm
        self.store = store  # AtomicPreferenceStore
        self._profile_pool = profile_evidence_pool
        self._user_id = user_id
        self._artifact_writer = artifact_writer
        self._turn_counter = 0

    async def extract_from_round(
        self,
        round_data: Any,
        user_id: str,
        session_id: str = "",
    ) -> list[AtomicPreference]:
        """从单轮交互数据中提取偏好并存储"""
        if not self.llm:
            logger.warning("PreferenceExtractor: no LLM, skipping round extraction")
            return []

        try:
            # Build episode_content blob from round_data fields
            user_msg = getattr(round_data, 'user_message', '')
            agent_resp = getattr(round_data, 'agent_response', '')
            tool_summary = self._summarize_tool_calls(getattr(round_data, 'tool_calls', []))
            episode_content = (
                f"用户请求: {user_msg}\n"
                f"Agent响应: {agent_resp[:500]}\n"
                f"工具调用: {tool_summary}"
            )
            prompt = ATOMIC_PREFERENCE_EXTRACTION_PROMPT.format(
                episode_content=episode_content,
            )
            # llm is a _make_llm_callable()-wrapped async callable(prompt: str) -> str
            if callable(self.llm) and not hasattr(self.llm, 'agenerate'):
                response_text = await self.llm(prompt)
            else:
                response = await self.llm.agenerate([{"role": "user", "content": prompt}])
                response_text = response.generations[0][0].text
            preferences = self._parse_preferences(response_text, user_id, session_id=session_id)
            await self._store_preferences(preferences)

            if self._artifact_writer and preferences:
                try:
                    await self._artifact_writer.write_preferences(
                        user_id=user_id,
                        preferences=[p.to_dict() for p in preferences],
                    )
                except Exception as e:
                    logger.debug(f"Artifact write failed: {e}")

            return preferences
        except Exception as e:
            logger.warning(f"Preference extraction from round failed: {e}")
            return []

    # ── 内部方法 ──

    async def _store_preferences(self, preferences: list[AtomicPreference]):
        """批量存储偏好"""
        if self.store and preferences:
            for pref in preferences:
                try:
                    await self.store.add(pref)
                except Exception as e:
                    logger.warning(f"Failed to store preference: {e}")

    @staticmethod
    def _summarize_tool_calls(tool_calls: list) -> str:
        if not tool_calls:
            return "无工具调用"
        summaries = []
        for tc in tool_calls:
            tool_name = getattr(tc, 'tool_name', 'unknown')
            summary = getattr(tc, 'action_summary', '')
            summaries.append(f"- {tool_name}: {summary}")
        return "\n".join(summaries)

    @staticmethod
    def _parse_preferences(response_text: str, user_id: str, session_id: str = "") -> list[AtomicPreference]:
        """解析 LLM 输出为 AtomicPreference 列表"""
        preferences = []
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    pref = AtomicPreference(
                        user_id=user_id,
                        preference_type=item.get('preference_type', 'value'),
                        trigger=item.get('trigger', ''),
                        preference=item.get('preference', ''),
                        rationale=item.get('rationale', ''),
                        scope=item.get('scope', 'global'),
                        scope_value=item.get('scope_value', ''),
                        confidence=item.get('confidence', 0.5),
                        source_session_id=session_id,
                    )
                    preferences.append(pref)
        except Exception as e:
            logger.debug(f"JSON parse failed, trying line-by-line: {e}")

        if not preferences:
            for line in response_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    pref = AtomicPreference(
                        user_id=user_id,
                        preference=line,
                        rationale="从反馈中提取",
                        source_session_id=session_id,
                    )
                    preferences.append(pref)
        return preferences


    async def extract_and_store(
        self,
        episodes: list,
        user_id: str,
        session_id: str = "",
    ) -> list[AtomicPreference]:
        """从 DesignEpisode 列表中批量提取偏好并存储。

        将每个 DesignEpisode 适配为 round_data 结构后调用 extract_from_round()。
        """
        all_prefs: list[AtomicPreference] = []
        for ep in episodes:
            try:
                round_data = _EpisodeAsRound(ep)
                prefs = await self.extract_from_round(round_data, user_id, session_id=session_id)
                all_prefs.extend(prefs)
            except Exception as e:
                logger.warning("extract_and_store: episode extraction failed: %s", e)
        return all_prefs


class _EpisodeAsRound:
    """将 DesignEpisode 适配为 extract_from_round() 所需的 round_data 接口。"""

    def __init__(self, episode: Any) -> None:
        # DesignEpisode fields: user_intent, design_insight, action_outcome
        # ExperienceTrace fields: task_description, lessons_learned (fallback)
        user_intent = getattr(episode, 'user_intent', '') or getattr(episode, 'task_description', '') or ''
        insight = getattr(episode, 'design_insight', '') or getattr(episode, 'lessons_learned', '') or ''
        outcome = getattr(episode, 'action_outcome', '') or ''
        # Compose a richer user_message / agent_response so episode_content is meaningful
        self.user_message = user_intent
        self.agent_response = f"{insight} {outcome}".strip()
        self.tool_calls = []


# ── 向后兼容别名 ──
AtomicPreferenceExtractor = PreferenceExtractor
