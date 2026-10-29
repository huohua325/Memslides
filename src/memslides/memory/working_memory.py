"""Working Memory — Job 级临时存储 (Stage 12 + Stage 14)

WM + 双 LTM 架构的核心组件。
存储 TempPreference、RoundExperience、DesignEpisode、ToolChainBuffer、JobHistory。

Stage 14 重构:
- RoundExperience 替代 TempExperienceTrace，统一两种经验来源
- 通过 source 字段区分 agent_lesson / auto_extract / preload
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from memslides.memory.core.models import TempPreference, RoundExperience, DesignEpisode
from memslides.memory.components import ToolChainBuffer, JobHistory

logger = logging.getLogger(__name__)

_CONFLICT_CHECK_PROMPT = """你是PPT设计偏好冲突检测器。用户新表达了一条偏好，请判断它与同维度的每条已有偏好是否冲突。

## 新偏好
{new_content}

## 同维度已有偏好（编号从 0 开始）
{existing_items}

## 判断规则
- "conflict": 新偏好与该已有偏好在同一方面表达了不同/矛盾的意图（如颜色、字体、风格等），新偏好应替代旧偏好
- "coexist": 新偏好与该已有偏好描述的是不同方面，可以共存

输出 JSON 数组，每个元素格式：{{"index": 0, "verdict": "conflict"}} 或 {{"index": 0, "verdict": "coexist"}}

只输出 JSON 数组，无需解释。"""

_GENERAL_UPDATE_PROMPT = """你是PPT设计助手的 general 偏好整合器。请把新的 general 偏好整合到当前仍生效的 general 偏好集合中。

## 当前仍生效的 general 偏好（编号从 0 开始）
{existing_items}

## 新偏好
{new_content}

## 处理规则
- "conflict": 新偏好与该已有偏好冲突，旧偏好应被替换
- "merge": 新偏好与该已有偏好表达的是同一条全局规则，但新偏好提供了补充/修正，应合并成一条更完整的规则
- "duplicate": 新偏好与该已有偏好等价，无需新增新条目
- "coexist": 两条 general 偏好不冲突，应同时保留
- 新偏好如果只是在已有规则基础上做更明确的表述，优先用更清晰、更稳定、更适合长期注入的表达
- 不要因为新偏好更新了某一条 general 规则，就丢掉其他不冲突的 general 规则

输出 JSON：
{{
  "judgements": [{{"index": 0, "verdict": "merge"}}],
  "normalized_preference": "合并或替换后的 general 偏好；如果 duplicate 且无需新增则留空",
  "skip_add": false
}}

只输出 JSON，无需解释。"""


class WorkingMemory:
    """纯内存 WorkingMemory — Job 级临时存储。"""

    def __init__(self, job_id: str, chain_extract_threshold: int = 3,
                 embedding_func=None):
        self._job_id = job_id
        self._temp_preferences: list[TempPreference] = []
        self._temp_experiences: list[RoundExperience] = []
        self._temp_episodes: list[DesignEpisode] = []
        self._chain_buffer = ToolChainBuffer()
        self._job_history = JobHistory()
        self._embedder = embedding_func  # EmbeddingFunc: async (list[str]) -> np.ndarray
        self._exp_embeddings_cache: dict[str, "np.ndarray"] = {}  # id → embedding vector

    async def add_preference(
        self, pref: TempPreference, llm: Callable | None = None,
    ) -> None:
        """添加临时偏好，LLM 判断同维度冲突后精准 supersede。

        对同维度已有的 active 条目，用 LLM 逐条判断是否与新偏好冲突：
        - conflict → supersede（标记为废弃）
        - coexist  → 保留（两条共存）

        无 LLM 或无维度时，对普通维度降级为全量 supersede；
        general 维度走保守追加/去重，避免误删其他全局规则。

        Args:
            pref: 要添加的临时偏好
            llm: LLM callable，用于冲突检测/merge。如果为 None，普通维度降级到全量
                 supersede，general 维度则保守保留旧规则并只做去重追加。
        """
        if pref.dimension == "general":
            await self._add_general_preference(pref, llm=llm)
            return

        # 降级场景：无维度或无 LLM，同维度全量 supersede
        if not pref.dimension or not llm:
            if pref.dimension:
                for existing in self._temp_preferences:
                    if existing.dimension == pref.dimension and not existing.superseded:
                        existing.superseded = True
            self._temp_preferences.append(pref)
            return

        # 找到同维度 active 条目
        same_dim_active = [
            (i, p) for i, p in enumerate(self._temp_preferences)
            if p.dimension == pref.dimension and not p.superseded
        ]

        if not same_dim_active:
            # 无同维度条目，直接追加
            self._temp_preferences.append(pref)
            return

        # LLM 冲突检测
        conflict_indices = await self._llm_conflict_check(
            pref.content, same_dim_active, llm,
        )

        # 只 supersede 冲突的条目
        for global_idx in conflict_indices:
            self._temp_preferences[global_idx].superseded = True

        self._temp_preferences.append(pref)
        logger.debug(
            "add_preference [%s]: %d existing, %d conflict, %d coexist",
            pref.dimension, len(same_dim_active),
            len(conflict_indices), len(same_dim_active) - len(conflict_indices),
        )

    async def _add_general_preference(
        self,
        pref: TempPreference,
        llm: Callable | None = None,
    ) -> None:
        """general 偏好单独处理。

        general 更接近“全局规则集合”，不适合普通维度的一对一 conflict/coexist。
        策略：
        1. 先把历史遗留的多行 general 块拆成原子条目
        2. 有 LLM 时，让模型判断 conflict / merge / duplicate / coexist
        3. 无 LLM 时走保守降级：不覆盖旧规则，只做去重后追加
        """
        self._normalize_active_general_preferences()

        same_dim_active = [
            (i, p) for i, p in enumerate(self._temp_preferences)
            if p.dimension == "general" and not p.superseded
        ]

        if not same_dim_active:
            self._temp_preferences.append(pref)
            return

        merge_result = None
        if llm is not None:
            merge_result = await self._llm_general_update_check(
                pref.content,
                same_dim_active,
                llm,
            )

        if merge_result is None:
            if self._has_equivalent_general_preference(pref.content, same_dim_active):
                logger.debug("add_preference [general]: duplicate detected in fallback, skip append")
                return
            self._temp_preferences.append(pref)
            logger.debug(
                "add_preference [general]: fallback append, keep %d existing active rules",
                len(same_dim_active),
            )
            return

        conflict_indices, skip_add, normalized_content = merge_result
        for global_idx in conflict_indices:
            self._temp_preferences[global_idx].superseded = True

        if skip_add:
            logger.debug(
                "add_preference [general]: %d existing, %d superseded, duplicate/keep-existing",
                len(same_dim_active),
                len(conflict_indices),
            )
            return

        final_content = normalized_content or pref.content.strip()
        if not final_content:
            logger.debug("add_preference [general]: normalized content empty, skip append")
            return

        if final_content == pref.content.strip():
            new_pref = pref
        else:
            new_pref = TempPreference(
                content=final_content,
                dimension="general",
                preference_type=pref.preference_type,
                source_task_id=pref.source_task_id,
                scope=pref.scope,
                scope_value=pref.scope_value,
                structured_data=pref.structured_data,
            )

        self._temp_preferences.append(new_pref)
        logger.debug(
            "add_preference [general]: %d existing, %d superseded, appended='%s'",
            len(same_dim_active),
            len(conflict_indices),
            final_content[:80],
        )

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = " ".join(str(item or "").split()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @classmethod
    def extract_general_preference_items(cls, content: str) -> list[str]:
        """将 general 文本拆成原子规则。

        兼容以下两类输入：
        - 单条 general 偏好文本
        - 历史遗留的 "## General Preferences\\n- ...\\n- ..." 块
        """
        raw = str(content or "").strip()
        if not raw:
            return []

        items: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("## "):
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            elif stripped.startswith("* "):
                stripped = stripped[2:].strip()
            if stripped and not stripped.startswith("</"):
                items.append(stripped)

        if items:
            return cls._dedupe_preserve_order(items)

        normalized = " ".join(line.strip() for line in raw.splitlines() if line.strip())
        return [normalized] if normalized else []

    def _normalize_active_general_preferences(self) -> None:
        """将旧格式的 general 大块文本拆成原子规则，保留原来源元数据。"""
        active_items = {
            p.content.strip()
            for p in self._temp_preferences
            if p.dimension == "general" and not p.superseded and p.content.strip()
        }
        appended: list[TempPreference] = []

        for existing in self._temp_preferences:
            if existing.superseded or existing.dimension != "general":
                continue

            extracted_items = self.extract_general_preference_items(existing.content)
            normalized_existing = existing.content.strip()
            if not extracted_items:
                continue
            if len(extracted_items) == 1 and extracted_items[0] == normalized_existing:
                continue

            existing.superseded = True
            for item in extracted_items:
                if item in active_items:
                    continue
                active_items.add(item)
                appended.append(
                    TempPreference(
                        content=item,
                        dimension="general",
                        preference_type=existing.preference_type,
                        source_task_id=existing.source_task_id,
                        scope=existing.scope,
                        scope_value=existing.scope_value,
                        structured_data=existing.structured_data,
                        timestamp=existing.timestamp,
                    )
                )

        if appended:
            self._temp_preferences.extend(appended)
            logger.debug(
                "_normalize_active_general_preferences: split %d atomic general rules",
                len(appended),
            )

    def _has_equivalent_general_preference(
        self,
        new_content: str,
        same_dim_active: list[tuple[int, TempPreference]],
    ) -> bool:
        """判断新 general 偏好是否与当前 active 集合等价。"""
        active_items: list[str] = []
        for _, pref in same_dim_active:
            active_items.extend(self.extract_general_preference_items(pref.content))
        active_set = set(self._dedupe_preserve_order(active_items))
        new_items = self.extract_general_preference_items(new_content)
        return bool(new_items) and all(item in active_set for item in new_items)

    async def _llm_general_update_check(
        self,
        new_content: str,
        same_dim_active: list[tuple[int, TempPreference]],
        llm: Callable,
    ) -> tuple[list[int], bool, str] | None:
        """LLM 判断 general 偏好的冲突、合并与重复关系。"""
        existing_items = "\n".join(
            f"[{seq}] {pref.content}" for seq, (_, pref) in enumerate(same_dim_active)
        )
        prompt = _GENERAL_UPDATE_PROMPT.format(
            existing_items=existing_items,
            new_content=new_content[:500],
        )

        try:
            try:
                response = await llm(prompt)
                text = str(response).strip() if response else ""
            except Exception:
                response = await llm.run(
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.choices[0].message.content or "").strip()

            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                logger.warning("_llm_general_update_check: no JSON object, fallback to append")
                return None

            data = json.loads(text[start:end + 1])
            judgements = data.get("judgements", [])
            normalized_content = " ".join(
                str(data.get("normalized_preference", "") or "").split()
            ).strip()
            skip_add = bool(data.get("skip_add", False))

            conflict_indices: list[int] = []
            duplicate_found = False
            for item in judgements:
                if not isinstance(item, dict):
                    continue
                seq = item.get("index", -1)
                verdict = str(item.get("verdict", "")).strip().lower()
                if verdict in {"conflict", "merge"} and 0 <= seq < len(same_dim_active):
                    global_idx = same_dim_active[seq][0]
                    if global_idx not in conflict_indices:
                        conflict_indices.append(global_idx)
                elif verdict == "duplicate":
                    duplicate_found = True

            if duplicate_found and not normalized_content:
                skip_add = True

            return conflict_indices, skip_add, normalized_content

        except Exception as e:
            logger.warning("_llm_general_update_check failed: %s, fallback to append", e)
            return None

    def add_preference_sync(self, pref: TempPreference) -> None:
        """同步包装器，仅用于测试和 Job 恢复。

        生产代码应使用 async add_preference(pref, llm=...)。
        此方法使用 llm=None，降级到全量 supersede。
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在 async 上下文中，不能使用 run_until_complete
                raise RuntimeError("Cannot use sync wrapper in async context. Use await add_preference() instead.")
            loop.run_until_complete(self.add_preference(pref, llm=None))
        except RuntimeError as e:
            if "no running event loop" in str(e) or "no current event loop" in str(e):
                # 创建新的事件循环
                asyncio.run(self.add_preference(pref, llm=None))
            else:
                raise

    async def _llm_conflict_check(
        self,
        new_content: str,
        same_dim_active: list[tuple[int, TempPreference]],
        llm: Callable,
    ) -> list[int]:
        """LLM 判断新偏好与同维度已有偏好的冲突关系。

        Returns:
            冲突条目的全局索引列表（应被 supersede）。
        """
        existing_items = "\n".join(
            f"[{seq}] {p.content}" for seq, (_, p) in enumerate(same_dim_active)
        )
        prompt = _CONFLICT_CHECK_PROMPT.format(
            new_content=new_content[:500],
            existing_items=existing_items,
        )

        try:
            try:
                response = await llm(prompt)
                text = str(response).strip() if response else ""
            except Exception:
                response = await llm.run(
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.choices[0].message.content or "").strip()

            logger.debug(f"LLM response text: {text}")

            start = text.find("[")
            end = text.rfind("]")
            if start < 0 or end <= start:
                logger.warning("_llm_conflict_check: no JSON array, fallback to full supersede")
                return [idx for idx, _ in same_dim_active]

            items = json.loads(text[start:end + 1])
            logger.debug(f"Parsed items: {items}")

            conflict_global_indices: list[int] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                seq = item.get("index", -1)
                verdict = item.get("verdict", "")
                logger.debug(f"Processing item: seq={seq}, verdict={verdict}, same_dim_active[{seq}]={same_dim_active[seq] if 0 <= seq < len(same_dim_active) else 'OUT_OF_RANGE'}")
                if verdict == "conflict" and 0 <= seq < len(same_dim_active):
                    global_idx = same_dim_active[seq][0]
                    conflict_global_indices.append(global_idx)
                    logger.debug(f"  -> Adding global index {global_idx} to conflict list")

            logger.debug(f"Final conflict_global_indices: {conflict_global_indices}")
            return conflict_global_indices

        except Exception as e:
            logger.warning("_llm_conflict_check failed: %s, fallback to full supersede", e)
            return [idx for idx, _ in same_dim_active]

    # ── Stage 14: RoundExperience 管理 ──

    def add_experience(self, exp: RoundExperience) -> None:
        """添加经验（remember_lesson / 自动提取 / LTM 预加载）"""
        exp.source_job_id = self._job_id
        self._temp_experiences.append(exp)

    def get_experiences(self) -> list[RoundExperience]:
        """获取所有临时经验"""
        return self._temp_experiences.copy()

    async def search_experiences_by_keywords(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RoundExperience]:
        """语义检索临时经验（向量优先，关键词降级）

        Args:
            query: 检索查询（通常是 user_message 或工具名）
            top_k: 返回数量上限

        Returns:
            按相关性排序的 RoundExperience 列表
        """
        if not self._temp_experiences:
            return []

        # 优先使用向量语义检索
        if self._embedder is not None:
            try:
                return await self._semantic_search(query, top_k)
            except Exception:
                pass  # 降级到关键词匹配

        # 降级：关键词重叠度匹配
        return self._keyword_search(query, top_k)

    async def _semantic_search(
        self, query: str, top_k: int,
    ) -> list[RoundExperience]:
        """向量语义检索：embed query + 经验文本，cosine 排序取 top_k"""
        from memslides.memory.core.embedding import batch_cosine_similarity

        # 构建经验文本（keywords + content 拼接）
        exp_texts = []
        for exp in self._temp_experiences:
            kw_part = " ".join(exp.keywords) if exp.keywords else ""
            text = f"{kw_part} {exp.content}".strip()
            exp_texts.append(text)

        # 批量 embed（利用缓存避免重复计算）
        uncached_texts = []
        uncached_ids = []
        for i, exp in enumerate(self._temp_experiences):
            if exp.id not in self._exp_embeddings_cache:
                uncached_texts.append(exp_texts[i])
                uncached_ids.append(exp.id)

        if uncached_texts:
            vecs = await self._embedder(uncached_texts)
            for j, eid in enumerate(uncached_ids):
                self._exp_embeddings_cache[eid] = vecs[j]

        # embed query
        query_vec = (await self._embedder([query]))[0]

        # 构建矩阵并计算相似度
        import numpy as np
        matrix = np.array(
            [self._exp_embeddings_cache[exp.id] for exp in self._temp_experiences],
            dtype=np.float32,
        )
        scores = batch_cosine_similarity(query_vec, matrix)

        # 按分数排序取 top_k（不设阈值，全量返回 top_k）
        indices = np.argsort(scores)[::-1][:top_k]
        return [self._temp_experiences[i] for i in indices]

    def _keyword_search(self, query: str, top_k: int) -> list[RoundExperience]:
        """关键词重叠度匹配（降级策略）"""
        query_tokens = set(self._tokenize(query.lower()))
        scored = []
        for exp in self._temp_experiences:
            exp_tokens = set()
            for kw in exp.keywords:
                exp_tokens.update(self._tokenize(kw.lower()))
            exp_tokens.update(self._tokenize(exp.content.lower()))
            if exp.tool_name:
                exp_tokens.add(exp.tool_name.lower())

            overlap = len(query_tokens & exp_tokens)
            if overlap > 0:
                scored.append((exp, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored[:top_k]]

    def get_experiences_prompt(self) -> str:
        """生成 WM 经验的注入 prompt 文本"""
        if not self._temp_experiences:
            return ""
        lines = ['<working_memory_experiences priority="high" note="当前会话经验，优先级高于历史经验">']
        for exp in self._temp_experiences:
            lines.append(exp.to_prompt_text())
        lines.append("</working_memory_experiences>")
        return "\n".join(lines)

    # ── Episode 管理 ──

    def add_episode(self, episode: DesignEpisode) -> None:
        """添加临时 Episode"""
        self._temp_episodes.append(episode)

    def get_episodes(self) -> list[DesignEpisode]:
        """获取所有临时 Episode"""
        return self._temp_episodes.copy()

    async def search_preferences_by_dimension(
        self,
        matched_dims: set[str] | None,
        context: dict,
        top_k: int = 10,
    ) -> list[TempPreference]:
        """基于 DimensionMatcher 结果筛选 WM 偏好（推荐使用）。

        与 LTM UserProfile 使用相同的 DimensionMatcher 维度筛选逻辑，
        确保 WM 和 LTM 的偏好检索口径一致。

        Args:
            matched_dims: DimensionMatcher 匹配到的维度集合。
                          None 表示全量返回（无法判断维度时的兜底）。
            context: scope 过滤上下文 {"slide_type": ..., "element_type": ...}
            top_k: 返回数量上限

        Returns:
            按维度匹配筛选后的 TempPreference 列表
        """
        candidates = [p for p in self._temp_preferences
                      if not p.superseded and p.matches_context(context)]
        if not candidates:
            return []

        if matched_dims is None:
            general = []
            specific = []
            for pref in candidates:
                dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
                if dim_prefix in {"", "general"}:
                    general.append(pref)
                else:
                    specific.append(pref)
            if len(general) >= top_k:
                return general
            return general + specific[: max(top_k - len(general), 0)]

        # 按维度前缀匹配：matched_dims 是顶级维度名（如 "theme"），
        # TempPreference.dimension 是带子字段的（如 "theme.primary_colors"）
        matched = []
        general = []
        for pref in candidates:
            dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
            if dim_prefix in matched_dims:
                matched.append(pref)
            elif dim_prefix in {"", "general"}:
                # general 偏好始终包含，不受 matched_dims / top_k specific 限制
                general.append(pref)

        if len(general) >= top_k:
            return general
        remaining = top_k - len(general)
        return general + matched[:remaining]

    def search_preferences(self, query: str, context: dict, top_k: int = 10) -> list[TempPreference]:
        """搜索临时偏好（向后兼容）：只返回当前生效的（superseded=False），scope 过滤 → token 重叠度排序"""
        candidates = [p for p in self._temp_preferences
                      if not p.superseded and p.matches_context(context)]
        if not candidates:
            return []

        if not query:
            return candidates[:top_k]

        query_tokens = set(self._tokenize(query.lower()))
        scored = []
        for pref in candidates:
            content_tokens = set(self._tokenize(pref.content.lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap > 0:
                scored.append((pref, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """简单双语分词"""
        import re
        tokens = re.findall(r'[\w]+', text)
        return tokens

    def search_tool_experiences(self, query: str, tool_name: str = "", top_k: int = 20):
        """委托给 ToolChainBuffer（存储 ChainSegment 和 ChainExperience）"""
        if tool_name:
            return self._chain_buffer.get_experiences_for_tool(tool_name)[:top_k]
        return self._chain_buffer.search_experiences(query, top_k=top_k)

    @property
    def chain_buffer(self) -> ToolChainBuffer:
        return self._chain_buffer

    @property
    def job_history(self) -> JobHistory:
        return self._job_history

    def release(self) -> None:
        """释放所有数据"""
        self._temp_preferences.clear()
        self._temp_experiences.clear()  # Stage 14
        self._temp_episodes.clear()
        self._chain_buffer.release()
        self._job_history.release()
