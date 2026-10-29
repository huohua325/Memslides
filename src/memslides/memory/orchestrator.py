"""MemoryOrchestrator — 编排层（Design Doc §7）

协调 WM / LTM / RoundCache / Distributor 的统一接口。
提供给 AgentLoop 的 on_job_start/end, on_round_start/end, on_operation_complete 等方法。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .components.operation_distributor import OperationDistributor
from .components.round_cache import RoundCache
from .core.models import (
    TempPreference,
    RoundExperience,
    default_outcome_for_experience_type,
    infer_experience_type,
    infer_task_experience_category,
)
from .extract.round_summarizer import RoundSummarizer
from .extract.tool_reasoning import normalize_tool_reason
from memslides.runtime.support import persist_live_wm_snapshot_from_orchestrator

if TYPE_CHECKING:
    from .collect.artifact_dumper import ArtifactDumper
    from .collect.collector import MemoryCollector
    from .extract.chain_experience_extractor import ChainExperienceExtractor
    from .extract.chain_segmenter import ChainSegmenter
    from .extract.episode_extractor import EpisodeExtractor
    from .extract.preference_extractor import PreferenceExtractor
    from .session.job_manager import JobManager
    from .store.atomic_preference_store import AtomicPreferenceStore
    from .store.chain_store import ChainStore

logger = logging.getLogger(__name__)

PRELOAD_CANDIDATE_LIMIT = 100
PRELOAD_CATEGORY_QUOTAS: dict[str, int] = {
    "tool_error": 2,
    "tool_misuse": 2,
    "tool_limitation": 2,
    "pattern": 4,
}
PRELOAD_CATEGORY_MAX: dict[str, int] = {
    "pattern": 6,
}
PRELOAD_TOTAL_LIMIT = 14
PRELOAD_FILL_CATEGORIES = {
    "tool_error",
    "tool_misuse",
    "tool_limitation",
    "pattern",
    "generic",
}

TASK_EXPERIENCE_CANDIDATE_POOL = 20
TASK_EXPERIENCE_SECONDARY_POOL = 10
TASK_INJECTION_CATEGORY_LIMITS: dict[str, int] = {
    "tool_error": 1,
    "tool_misuse": 1,
    "tool_limitation": 1,
    "pattern": 2,
}
TASK_INJECTION_TOTAL_LIMIT = sum(TASK_INJECTION_CATEGORY_LIMITS.values())
TASK_INJECTION_FILL_CATEGORIES = {
    "tool_error",
    "tool_misuse",
    "tool_limitation",
    "pattern",
    "generic",
}
_STALE_STRUCTURAL_TOOL_LIMITATION_PATTERNS = (
    r"缺乏.{0,40}(插入|新增|新建).{0,40}(幻灯片|页面|页)",
    r"无\s*`?insert_slide",
    r"no\s+insert_slide",
    r"without\s+insert_slide",
    r"缺乏.{0,40}write_new_slide_file",
    r"write_new_slide_file\s*/\s*reorder",
)
DESIGN_TASK_EXPERIENCE_ALLOWED_TOOLS = frozenset({
    "write_html_file",
    "inspect_slide",
    "query_slide_layout",
    "query_layout_geometry",
    "query_image_info",
    "write_html",
    "render_slide",
    "write_ppt",
})
DESIGN_TASK_EXPERIENCE_BLOCKED_TOOLS = frozenset({
    "convert_to_markdown",
    "write_markdown_file",
    "inspect_manuscript",
})
DESIGN_TASK_EXPERIENCE_ALLOWED_TERMS = (
    "html",
    "css",
    "slide",
    "slides",
    "幻灯片",
    "layout",
    "layouts",
    "版式",
    "排版",
    "template",
    "模板",
    "inspect_slide",
    "write_html",
    "render",
    "渲染",
    "aspect ratio",
    "图片比例",
)
DESIGN_TASK_EXPERIENCE_BLOCKED_TERMS = (
    "convert_to_markdown",
    "write_markdown",
    "inspect_manuscript",
    "markdown",
    ".md",
    "markdown_file",
    "页数/语言/资源警告",
    "image asset validation",
    "按章节锚点",
    "内容抽取与资源列举",
    "仅完成内容抽取",
    "只完成内容抽取",
    "extract content",
    "resource warning",
    "language warning",
)

AUTO_EXTRACT_LTM_MIN_CONFIDENCE = 0.8
CONSOLIDATION_CLUSTER_SIM_THRESHOLD = 0.58
CONSOLIDATION_NEIGHBOR_QUERY_LIMIT = 25
CONSOLIDATION_NEIGHBOR_RETURN_LIMIT = 3
CONSOLIDATION_HEURISTIC_MERGE_THRESHOLD = 0.72
CONSOLIDATION_HEURISTIC_RELATED_THRESHOLD = 0.38


class MemoryOrchestrator:
    """编排层 — 协调所有记忆组件的统一接口。"""

    def __init__(
        self,
        db: Any = None,
        job_manager: JobManager | None = None,
        tool_memory_retriever: Any = None,
        preference_store: AtomicPreferenceStore | None = None,
        profile_store: Any = None,
        user_state_resolver: Any = None,
        collector: MemoryCollector | None = None,
        preference_extractor: PreferenceExtractor | None = None,
        chain_segmenter: ChainSegmenter | None = None,
        episode_extractor: EpisodeExtractor | None = None,
        round_summarizer: RoundSummarizer | None = None,
        llm: Any = None,
        profile_injection_llm: Any = None,
        embedding_func: Any = None,
        chain_store: ChainStore | None = None,
        artifact_dumper: ArtifactDumper | None = None,
        enable_profile_injection: bool = True,
        enable_wm_preference_collection: bool = True,
        enable_wm_preference_injection: bool = True,
        enable_wm_experience_collection: bool = True,
        enable_wm_experience_injection: bool = True,
        enable_chain_experience_collection: bool = True,
        enable_wm_round_history_injection: bool = False,
        enable_wm_task_history_injection: bool | None = None,
        enable_ltm_tool_experience_injection: bool = True,
        enable_experience_preload: bool = True,
        enable_round_experience_writeback: bool = True,
        enable_task_experience_writeback: bool | None = None,
    ):
        self._db = db
        self._job_mgr = job_manager
        # Stage 14: _exp_writer 即 tool_memory_retriever（ExperienceTraceWriter 实例）
        # 用于 on_round_end 自动提取经验 → WM
        self._exp_writer = tool_memory_retriever
        # preference_store 不再使用，保留参数以兼容调用方
        del preference_store
        self._profile_store = profile_store
        self._user_state_resolver = user_state_resolver
        self._collector = collector
        self._pref_extractor = preference_extractor
        self._chain_segmenter = chain_segmenter
        self._episode_extractor = episode_extractor
        self._round_summarizer = round_summarizer or RoundSummarizer(llm=llm)
        self._llm = llm
        self._profile_injection_llm = profile_injection_llm or llm
        self._chain_store = chain_store
        self._round_cache: RoundCache | None = None
        self._distributor: OperationDistributor | None = None
        self._current_round_tool_log: list[dict] = []
        self._artifact_dumper = artifact_dumper
        self._round_start_time: str = ""
        self._op_counter: int = 0
        self._current_round_index: int = 0  # 统一的 round_index，避免 injection_traces 和 rounds 不对齐
        # Stage 12: DimensionMatcher 惰性创建，供 RoundCache 维度筛选
        self._dim_matcher = None
        self._embedding_func = embedding_func
        self._enable_profile_injection = enable_profile_injection
        self._enable_wm_preference_collection = enable_wm_preference_collection
        self._enable_wm_preference_injection = enable_wm_preference_injection
        self._enable_wm_experience_collection = enable_wm_experience_collection
        self._enable_wm_experience_injection = enable_wm_experience_injection
        self._enable_chain_experience_collection = enable_chain_experience_collection
        if enable_wm_task_history_injection is not None:
            enable_wm_round_history_injection = enable_wm_task_history_injection
        self._enable_wm_round_history_injection = enable_wm_round_history_injection
        self._enable_ltm_tool_experience_injection = enable_ltm_tool_experience_injection
        self._enable_experience_preload = enable_experience_preload
        if enable_task_experience_writeback is not None:
            enable_round_experience_writeback = enable_task_experience_writeback
        self._enable_round_experience_writeback = enable_round_experience_writeback
        self._profile_execution_contract: dict[str, Any] | None = None
        self._profile_execution_plan: dict[str, Any] | None = None
        self._profile_realization_report: dict[str, Any] | None = None

    def _dump_profile_routing_error(
        self,
        *,
        error: Exception,
        user_id: str,
        task_intent: str,
        read_intent: str,
        write_intent: str,
    ) -> None:
        dumper = self._artifact_dumper
        base_dir = getattr(dumper, "base_dir", None) or getattr(dumper, "artifact_dir", None)
        if not base_dir:
            return
        try:
            debug_dir = base_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "profile_routing_error.json").write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(error),
                        "user_id": user_id,
                        "task_intent": task_intent,
                        "read_intent": read_intent,
                        "write_intent": write_intent,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to write profile routing error artifact", exc_info=True)

    @staticmethod
    def _compose_profile_from_snapshot(snapshot: Any) -> Any:
        """Merge core persona profile into the current intent profile for runtime read."""
        from .core.models import UserProfile

        intent_profile = deepcopy(
            getattr(snapshot, "intent_profile", None)
            or UserProfile(user_id=getattr(snapshot, "user_id", ""))
        )
        core_profile = getattr(snapshot, "core_profile", None)
        if core_profile is None:
            return intent_profile

        for dim_name in ("theme", "visual", "layout", "content"):
            intent_dim = getattr(intent_profile, dim_name, None)
            core_dim = getattr(core_profile, dim_name, None)
            if intent_dim is None or core_dim is None:
                continue
            for field_name in getattr(intent_dim, "__dataclass_fields__", {}):
                if field_name in {"confidence", "keywords"}:
                    continue
                intent_value = deepcopy(getattr(intent_dim, field_name))
                if intent_value not in ("", [], None):
                    continue
                core_value = deepcopy(getattr(core_dim, field_name))
                if core_value in ("", [], None):
                    continue
                setattr(intent_dim, field_name, core_value)

        merged_general: list[str] = []
        for pref in [
            *list(getattr(core_profile.general, "preferences", []) or []),
            *list(getattr(intent_profile.general, "preferences", []) or []),
        ]:
            text = str(pref or "").strip()
            if text and text not in merged_general:
                merged_general.append(text)
        intent_profile.general.preferences = merged_general
        return intent_profile

    @staticmethod
    def _profile_has_meaningful_content(profile: Any) -> bool:
        if profile is None:
            return False
        try:
            data = profile.to_dict() if hasattr(profile, "to_dict") else {}
        except Exception:
            return False

        def _has_value(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    _has_value(child)
                    for key, child in value.items()
                    if key not in {"confidence", "keywords"}
                )
            if isinstance(value, list):
                return any(_has_value(item) for item in value)
            return value not in ("", None)

        return any(
            _has_value(value)
            for key, value in data.items()
            if key not in {"user_id", "version", "last_updated"}
        )

        # 注：指纹+多模态分析已移至 Consolidation 阶段，仅针对最终 PPT。
        # VLM 由 Consolidator 直接持有，MemoryOrchestrator 不再引用。

    # ── Job 生命周期 ──

    async def on_job_start(
        self,
        user_id: str,
        project_id: str,
        user_prompt: str = "",
        intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
    ) -> None:
        """Job 开始：创建 Job + ProfileInjectionRouter 路由 + 初始化 WM + 从 LTM 预加载经验。

        Args:
            user_id: 用户 ID
            project_id: 项目 ID
            user_prompt: 用户第一轮 prompt（用于 ProfileInjectionRouter 路由）
            intent: 当前任务 intent（如 "academic"/"business"），空字符串时自动推断
            read_intent: 画像读取 intent
            write_intent: 画像写回 intent
            core_persona: 用户稳定人格
        """
        if not intent and user_prompt:
            from .core.models import classify_intent_by_keywords
            intent = classify_intent_by_keywords(user_prompt)
        if not read_intent:
            read_intent = intent
        if not write_intent:
            write_intent = intent or read_intent

        if self._job_mgr:
            await self._job_mgr.start_job(
                user_id,
                project_id,
                intent=intent,
                read_intent=read_intent,
                write_intent=write_intent,
                core_persona=core_persona,
            )
        self._profile_execution_contract = None
        self._profile_execution_plan = None
        self._profile_realization_report = None

        # Stage 15: ProfileInjectionRouter 智能路由（LTM UserProfile → WM TempPreference）
        if self._enable_profile_injection and self._profile_store and user_prompt:
            await self._route_and_inject_profile(
                user_id,
                user_prompt,
                task_intent=intent,
                read_intent=read_intent,
                write_intent=write_intent,
                core_persona=core_persona,
            )

        # Stage 13: 从 LTM 预加载相关经验到 WM
        if self._enable_experience_preload:
            await self._preload_experiences_from_ltm(user_id, project_id, user_prompt=user_prompt)

    async def on_job_end(
        self,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Job 结束：合并 WM 经验到 LTM + 触发 Consolidation + 释放 WM。

        Args:
            slide_html_dir: 最终 PPT 的 HTML 文件目录路径。
                           指纹+多模态分析仅在 Consolidation 时对该目录下的最终 HTML 执行。
            freeze_preference_writeback: 是否冻结偏好写回，仅保留工具/episode 等非偏好归档。
        """
        from .progress import emit_memory_save_progress

        preference_written_first = False
        result = {
            "status": "nothing_to_save",
            "saved": False,
            "consolidation_failed": False,
            "pending_consolidation_backup": "",
        }

        if self._job_mgr and hasattr(self._job_mgr, "write_preferences_first"):
            pref_result = await self._job_mgr.write_preferences_first(
                slide_html_dir=slide_html_dir,
                freeze_preference_writeback=freeze_preference_writeback,
                progress_callback=progress_callback,
            )
            preference_written_first = str(pref_result.get("status") or "") == "written"

        # Stage 13: 合并 WM 临时经验到 LTM。放在画像写回之后，
        # 避免大量经验聚类/LLM 合并阻塞用户偏好持久化。
        await emit_memory_save_progress(
            progress_callback,
            stage="round_experience_writeback",
            progress=32,
            message="Consolidating round-level working-memory experiences.",
        )
        await self._consolidate_experiences_to_ltm(
            progress_callback=progress_callback,
            progress_start=34,
            progress_end=52,
        )

        if self._job_mgr:
            await emit_memory_save_progress(
                progress_callback,
                stage="job_consolidation",
                progress=54,
                message="Archiving tool-chain memory, episodes, and traces.",
            )
            result = await self._job_mgr.end_job(
                slide_html_dir=slide_html_dir,
                freeze_preference_writeback=freeze_preference_writeback or preference_written_first,
                progress_callback=progress_callback,
            )
            if preference_written_first and result.get("status") == "saved":
                result["preference_profile_written_first"] = True
        self._round_cache = None
        self._distributor = None
        self._profile_execution_contract = None
        self._profile_execution_plan = None
        self._profile_realization_report = None
        return result

    # ── Round 生命周期 ──

    async def on_round_start(
        self,
        user_message: str,
        user_id: str,
        context: dict | None = None,
        session_id: str = "",
        memory_injection: str = "",
        memory_injection_full: str = "",
        agent_name: str = "",
        retrieval_stats: dict | None = None,
        intent: str = "",
    ) -> None:
        """Round 开始：创建 Round + 检索记忆 + 构建 RoundCache + 偏好提取。

        Args:
            memory_injection: 压缩版记忆注入文本 (≤500字符)
            memory_injection_full: 完整记忆注入文本
            agent_name: Agent 名称 (research/design/modify)
            retrieval_stats: 检索统计信息 (ap_value, et_tool_error 等)
            intent: 用户意图（如 "academic"），空字符串时由 RoundCache 自动推断
        """
        if self._job_mgr:
            await self._job_mgr.start_round(user_message)

        if not intent and self._job_mgr and self._job_mgr._active_job:
            intent = self._job_mgr._active_job.intent or ""

        # 确定当前 Round 的 1-based index（在 start_round 之后，此时 Round 已添加到 Job）
        self._current_round_index = self._job_mgr.round_count() if self._job_mgr else 1

        self._current_round_tool_log = []
        self._op_counter = 0

        from datetime import datetime
        self._round_start_time = datetime.now().isoformat()

        wm = self._job_mgr.working_memory if self._job_mgr else None

        # 惰性创建 DimensionMatcher（首次 Round 时初始化，后续复用）
        if self._dim_matcher is None and self._embedding_func is not None:
            try:
                from .core.dimension_matcher import DimensionMatcher
                self._dim_matcher = DimensionMatcher(self._embedding_func)
            except Exception as e:
                logger.debug("DimensionMatcher creation failed (non-fatal): %s", e)

        # 构建 RoundCache 并加载（使用统一的 round_index）
        round_index = self._current_round_index
        self._round_cache = RoundCache(
            profile_store=self._profile_store,
            working_memory=wm,
            dimension_matcher=self._dim_matcher,
        )
        # 注入 ArtifactDumper 和 round_index 用于中间产物记录
        self._round_cache._artifact_dumper = self._artifact_dumper
        self._round_cache._round_index = round_index
        try:
            await self._round_cache.load(user_message, user_id, context, session_id, intent=intent)
        except Exception as e:
            logger.warning(f"RoundCache load failed (non-fatal): {e}")

        self._distributor = OperationDistributor(
            self._round_cache,
            chain_store=self._chain_store,
            working_memory=wm,
            user_id=user_id,
            embedding_func=self._embedding_func,
            user_message=user_message,
            enable_ltm_tool_experience_injection=self._enable_ltm_tool_experience_injection,
        )

        # MemoryCollector begin_round
        if self._collector:
            try:
                round_count = self._job_mgr.round_count() if self._job_mgr else 1
                self._collector.begin_round(
                    round_id=round_count,
                    user_message=user_message,
                    agent_name=agent_name,
                    memory_injection=memory_injection,
                    memory_injection_full=memory_injection_full,
                    session_id=session_id,
                    user_id=user_id,
                )
            except Exception as e:
                logger.warning(f"Collector begin_round failed: {e}")

        # (dump_memory_injection 已移除，由 dump_injection_trace 替代)

    async def on_round_end(self, agent_response: str = "", compact_round: Any = None) -> None:
        """Round 结束：Episode 提取 + RoundSummary + 工具链分割 + 经验提取。"""
        wm = self._job_mgr.working_memory if self._job_mgr else None

        # Stage 12: 更新 Round 数据库记录（SQLite 表名暂保留 tasks）
        if self._job_mgr:
            try:
                round_id = self._job_mgr.current_round_id()
                if round_id:
                    await self._job_mgr.end_round(round_id, agent_response)
            except Exception as e:
                logger.warning(f"JobManager end_round failed (non-fatal): {e}")

        # Collector end_round
        if self._collector and compact_round is None:
            try:
                self._collector.add_agent_response(agent_response)
                compact_round = self._collector.end_round()
            except Exception as e:
                logger.warning(f"Collector end_round failed: {e}")

        # 1. Episode 提取
        design_episode = None
        if compact_round and self._episode_extractor and wm:
            try:
                if hasattr(self._episode_extractor, "extract_and_store"):
                    round_user_id = getattr(compact_round, "user_id", "") or (
                        self._job_mgr._active_job.user_id
                        if self._job_mgr and self._job_mgr._active_job else ""
                    )
                    round_session_id = getattr(compact_round, "session_id", "") or (
                        self._job_mgr._active_job.project_id
                        if self._job_mgr and self._job_mgr._active_job else ""
                    )
                    episodes = await self._episode_extractor.extract_and_store(
                        [compact_round],
                        user_id=round_user_id,
                        session_id=round_session_id,
                        store=False,
                    )
                    round_id = getattr(compact_round, "round_id", 0)
                    if (
                        round_id > 0 and self._collector
                        and hasattr(self._collector, "acknowledge_rounds")
                    ):
                        self._collector.acknowledge_rounds([round_id])
                    if episodes:
                        design_episode = episodes[0]
                        wm.add_episode(design_episode)
            except Exception as e:
                logger.warning(f"Episode extraction failed (non-fatal): {e}")

        # 2. 生成 RoundSummary
        round_summary = None
        if wm and self._job_mgr and self._job_mgr._active_job:
            try:
                rounds = self._job_mgr._active_job.rounds
                round_summary = await self._round_summarizer.summarize(
                    round_index=len(rounds),
                    user_message=rounds[-1].user_message if rounds else "",
                    tool_calls_log=self._current_round_tool_log,
                    agent_response=agent_response,
                    round_id=rounds[-1].id if rounds else "",
                )
                round_summary.design_episode = design_episode
                wm.job_history.add_summary(round_summary)
            except Exception as e:
                logger.warning(f"RoundSummary generation failed (non-fatal): {e}")

        # ArtifactDumper: RoundSummary 产物
        if self._artifact_dumper and round_summary:
            try:
                round_index = (len(self._job_mgr._active_job.rounds)
                              if self._job_mgr and self._job_mgr._active_job else 0)
                self._artifact_dumper.dump_round_summary(round_index, summary=round_summary)
            except Exception as e:
                logger.warning(f"ArtifactDumper dump_round_summary failed (non-fatal): {e}")

        # 3. 工具链分割（纯累积，不触发经验提取）
        #    使用完整的 rich traces（reasoning + tool_call + observation）
        _segmented_chains = None
        _rich_traces = None
        if compact_round and self._chain_segmenter and wm and self._enable_chain_experience_collection:
            try:
                from .extract.chain_segmenter import build_rich_trace

                round_id = self._job_mgr.current_round_id() if self._job_mgr else ""

                # 从 _current_round_tool_log 构建 rich traces
                _rich_traces = []
                for tl in self._current_round_tool_log:
                    _rich_traces.append(build_rich_trace(
                        tool_name=tl.get("name", ""),
                        args=tl.get("args", ""),
                        result=tl.get("result_preview", ""),
                        is_error=tl.get("is_error", False),
                        reasoning=tl.get("reasoning", ""),
                    ))

                _segmented_chains = await self._chain_segmenter.segment(
                    compact_round, round_id, rich_traces=_rich_traces,
                )
                wm.chain_buffer.add_chains(_segmented_chains)
            except Exception as e:
                logger.warning(f"Chain segmentation failed (non-fatal): {e}")

        # ArtifactDumper: 工具链分割产物 + rich_traces
        if self._artifact_dumper:
            try:
                round_index = (len(self._job_mgr._active_job.rounds)
                              if self._job_mgr and self._job_mgr._active_job else 0)
                if _segmented_chains is not None:
                    self._artifact_dumper.dump_round_chain_segmentation(
                        round_index, chains=_segmented_chains,
                    )
                if _rich_traces is not None:
                    self._artifact_dumper.dump_round_rich_traces(
                        round_index, traces=_rich_traces,
                    )
            except Exception as e:
                logger.warning(f"ArtifactDumper dump_chain/traces failed (non-fatal): {e}")

        # 4. 注：指纹+多模态分析已移至 Consolidation 阶段，仅针对最终 PPT。
        # mid-job 不再构建快照 / 采集视觉指纹。

        # 5. ArtifactDumper — 转储 Round 级中间产物
        if self._artifact_dumper:
            try:
                from datetime import datetime
                round_index = (len(self._job_mgr._active_job.rounds)
                              if self._job_mgr and self._job_mgr._active_job else 0)
                current_round = (
                    self._job_mgr._active_job.rounds[-1]
                    if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.rounds
                    else None
                )
                self._artifact_dumper.dump_round_meta(
                    round_index,
                    round_id=current_round.id if current_round else "",
                    user_message=current_round.user_message if current_round else "",
                    agent_response=agent_response[:500],
                    tool_call_count=len(self._current_round_tool_log),
                    total_duration_ms=sum(
                        int(t.get("duration_ms", 0) or 0)
                        for t in self._current_round_tool_log
                    ),
                    start_time=self._round_start_time,
                    end_time=datetime.now().isoformat(),
                )
                self._artifact_dumper.dump_wm_snapshot(
                    round_index,
                    chain_buffer=wm.chain_buffer if wm else None,
                    temp_preferences=wm._temp_preferences if wm else None,
                    temp_experiences=wm.get_experiences() if wm else None,
                    temp_episodes=wm.get_episodes() if wm else None,
                )
            except Exception as e:
                logger.warning(f"ArtifactDumper on_round_end failed (non-fatal): {e}")

        # 6. Stage 14/15: 自动提取经验 → 仅写入 WM，LTM 统一在 job_end consolidation
        if self._enable_wm_experience_collection and self._exp_writer and wm:
            try:
                session_id = ""
                if self._job_mgr and self._job_mgr._active_job:
                    session_id = self._job_mgr._active_job.id
                has_errors = any(t.get("is_error") for t in self._current_round_tool_log)
                _outcome = "partial" if has_errors else "success"

                if compact_round:
                    # 路径 A: 有 compact_round (from_agent_round)
                    round_experiences = await self._exp_writer.extract_to_round_experiences(
                        agent_round=compact_round,
                        session_id=session_id,
                        outcome=_outcome,
                    )
                elif self._current_round_tool_log:
                    # 路径 B: 无 compact_round，用 tool_calls_log
                    current_round = (
                        self._job_mgr._active_job.rounds[-1]
                        if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.rounds
                        else None
                    )
                    round_experiences = await self._exp_writer.extract_to_round_experiences(
                        tool_calls_log=self._current_round_tool_log,
                        user_task=current_round.user_message if current_round else "",
                        session_id=session_id,
                        outcome=_outcome,
                    )
                else:
                    round_experiences = []

                if round_experiences:
                    added = self.add_round_experiences(round_experiences)
                    logger.info(
                        f"[Stage 15] Auto-extracted {added}/{len(round_experiences)} experiences → WM"
                    )
            except Exception as e:
                logger.warning(f"Auto-extract experiences to WM failed (non-fatal): {e}")
        persist_live_wm_snapshot_from_orchestrator(self, reason="round_end")

    async def _extract_chain_experiences(
        self, chain_name: str, batch: list,
    ) -> list:
        """从同名链批次中提取经验。"""
        if not self._llm:
            return []
        try:
            from .extract.chain_experience_extractor import ChainExperienceExtractor
            extractor = ChainExperienceExtractor(llm=self._llm)
            return await extractor.extract(chain_name, batch)
        except Exception as e:
            logger.warning(f"Chain experience extraction failed: {e}")
            return []

    # ── Operation 级 ──

    async def get_memory_for_operation(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        context: dict | None = None,
    ) -> str:
        """为 Operation 获取记忆注入文本。"""
        if self._distributor is None:
            return ""
        try:
            structured = await self._distributor.distribute_structured(
                tool_name, tool_args, context,
            )
            result = structured.get("combined", "")
            # ArtifactDumper — 转储 Operation 级中间产物（统一使用 dump_injection_trace）
            if self._artifact_dumper:
                try:
                    round_index = (len(self._job_mgr._active_job.rounds)
                                  if self._job_mgr and self._job_mgr._active_job else 0)
                    _ltm_exp = structured.get("ltm_tool_experiences", "")
                    _agent_phase = (context or {}).get("agent_phase", "")
                    self._artifact_dumper.dump_injection_trace(
                        round_index,
                        level="operation",
                        agent_role=_agent_phase,
                        turn=0,
                        op_index=self._op_counter,
                        trigger_tools=[tool_name],
                        ltm_tool_experiences=_ltm_exp,
                        final_injected_text=result,
                        injection_target="USER <memory_hint>" if result else "",
                        skipped_reason="" if result else "no memory available",
                    )
                    self._op_counter += 1
                except Exception as e:
                    logger.warning(f"ArtifactDumper dump_injection_trace failed (non-fatal): {e}")
            return result
        except Exception as e:
            logger.warning(f"Memory distribution failed: {e}")
            return ""

    def on_operation_complete(
        self,
        tool_name: str,
        args: dict | str,
        result: str,
        is_error: bool = False,
        duration_ms: int = 0,
        reasoning: str = "",
        reason_source: str = "",
    ) -> None:
        """Operation 完成：记录到 Round + Collector + 本地日志。

        当 is_error=True 时，额外将错误封装为 ChainExperience 写入 WM chain_buffer，
        使同一 Round 内后续 Operation 能通过 OperationDistributor 立即感知到此错误。

        Args:
            reasoning: Agent 在调用此工具前的推理内容（reasoning_content），完整保留。
        """
        args_dict = args if isinstance(args, dict) else {}
        current_round = (
            self._job_mgr._active_job.rounds[-1]
            if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.rounds
            else None
        )
        normalized_reason, normalized_source, reason_quality = normalize_tool_reason(
            tool_name=tool_name,
            arguments=args,
            raw_reason=reasoning,
            user_message=current_round.user_message if current_round else "",
            observation=result,
            reason_source=reason_source,
            language="auto",
        )
        self._current_round_tool_log.append({
            "name": tool_name,
            "args": str(args),
            "result_preview": result[:5000],
            "is_error": is_error,
            "duration_ms": duration_ms,
            "reasoning": normalized_reason,
            "tool_reason": normalized_reason,
            "reason_source": normalized_source,
            "reason_quality": reason_quality,
        })
        if self._job_mgr:
            self._job_mgr.record_operation_to_current_round(
                tool_name, args_dict, result, is_error,
            )
        if self._collector:
            try:
                self._collector.add_tool_call(
                    name=tool_name, args=str(args),
                    result=result[:2000], is_error=is_error, duration_ms=duration_ms,
                )
                # collector 保留真实的模型可观测理由；合成理由仅进入 rich traces
                if normalized_reason and normalized_source != "synthesized_from_tool_context":
                    self._collector.add_reasoning(normalized_reason)
            except Exception:
                pass

        # 即时注入：工具错误 → WM chain_buffer，供本 Round 后续 Operation 查询
        if is_error and self._enable_chain_experience_collection:
            wm = self._job_mgr.working_memory if self._job_mgr else None
            if wm:
                try:
                    from .core.models import ChainExperience, make_chain_signature
                    sig = make_chain_signature([tool_name])
                    exp = ChainExperience(
                        chain_name=sig,
                        tool_pipeline=[tool_name],
                        lesson=f"工具 {tool_name} 调用失败: {result[:200]}",
                        applicable_when=f"调用 {tool_name} 时",
                        anti_pattern=result[:200],
                        confidence=0.6,
                        keywords=[tool_name],
                    )
                    wm.chain_buffer.add_experience(exp)
                except Exception as _e:
                    logger.debug("on_operation_complete: WM chain_buffer write failed: %s", _e)

    async def on_user_preference(
        self,
        preference_content: str,
        dimension: str = "",
        scope: str = "global",
        structured_data: dict[str, Any] | None = None,
    ) -> None:
        """用户显式表达偏好时写入 WM。

        如果调用方未提供 dimension，自动推断：
        1. DimensionMatcher 向量匹配（如果可用）
        2. UserProfile.match_dimension() 关键词降级
        3. 都匹配不到 → dimension 留空，归档时落入 general
        """
        if not self._enable_wm_preference_collection:
            return

        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm:
            return

        if not dimension:
            dimension = await self._infer_dimension(preference_content)

        await wm.add_preference(
            TempPreference(
                content=preference_content,
                dimension=dimension,
                source_task_id=self._current_round_id(),
                scope=scope,
                structured_data=deepcopy(structured_data) if structured_data else None,
            ),
            llm=self._llm,
        )
        persist_live_wm_snapshot_from_orchestrator(self, reason="user_preference")

        # 刷新 RoundCache 的 WM 偏好缓存，消除 1 轮注入延迟
        if self._round_cache is not None:
            self._round_cache.refresh_wm_preferences()

    def on_remember_lesson(
        self,
        content: str,
        tool_name: str = "",
        keywords: list[str] | None = None,
        category: str = "tool_error",
    ) -> None:
        """remember_lesson 工具调用时写入 WM (Stage 14)。

        Agent 调用 remember_lesson MCP 工具后，由 _tool_result_callback 触发此方法，
        将经验写入 WM 的 RoundExperience，供后续 Round 检索注入。
        不再双写 LTM，LTM 持久化统一由 Job 结束时 _consolidate_experiences_to_ltm() 处理。

        Args:
            content: 经验内容
            tool_name: 相关工具名
            keywords: 检索关键词（用于向量匹配）
            category: 分类 - "tool_error" | "tool_misuse" | "tool_limitation" | "pattern"
        """
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm:
            logger.debug("on_remember_lesson: no WM available, skipping")
            return
        if not self._enable_wm_experience_collection:
            logger.debug("on_remember_lesson: WM experience collection disabled, skipping")
            return

        # 自动生成 keywords（如果未提供）
        if not keywords:
            keywords = self._auto_generate_keywords(content, tool_name)

        exp = RoundExperience(
            content=content,
            tool_name=tool_name,
            keywords=keywords,
            category=category,
            source="agent_lesson",
            source_task_id=self._current_round_id(),
        )
        self.add_round_experiences([exp])
        logger.info(f"[MemoryV2] remember_lesson → WM: {content[:50]}... (keywords={keywords[:3]})")

    def on_tool_error(
        self,
        tool_name: str,
        args: str = "",
        error_msg: str = "",
    ) -> None:
        """工具失败时即时写入 WM RoundExperience。"""
        if not self._enable_wm_experience_collection:
            return
        if not tool_name:
            return
        keywords = self._auto_generate_keywords(error_msg, tool_name)
        keywords.extend(["tool_error", f"tool_{tool_name}"])
        exp = RoundExperience(
            content=f"{tool_name} 调用失败: {error_msg[:220]}",
            tool_name=tool_name,
            keywords=list(dict.fromkeys(k for k in keywords if k)),
            category="tool_error",
            source="tool_error",
            source_task_id=self._current_round_id(),
            confidence=0.95,
            tools_used=[tool_name],
            outcome="failed",
        )
        added = self.add_round_experiences([exp])
        if added:
            logger.info("[MemoryV2] on_tool_error → WM: %s", exp.content[:80])

    def add_round_experiences(self, experiences: list[RoundExperience]) -> int:
        """统一将外部生成的 RoundExperience 写入 WM，并做轻量去重。"""
        if not self._enable_wm_experience_collection:
            return 0
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm:
            return 0

        added = 0
        for exp in experiences or []:
            if not exp or not getattr(exp, "content", "").strip():
                continue
            if not exp.source_task_id:
                exp.source_task_id = self._current_round_id()
            if self._is_duplicate_round_experience(exp):
                continue
            wm.add_experience(exp)
            added += 1
        return added

    def add_task_experiences(self, experiences: list[RoundExperience]) -> int:
        """Deprecated compatibility wrapper for ``add_round_experiences``."""
        return self.add_round_experiences(experiences)

    def _current_round_id(self) -> str:
        if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.rounds:
            return self._job_mgr._active_job.rounds[-1].id
        return ""

    def _is_duplicate_round_experience(self, candidate: RoundExperience) -> bool:
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_experiences:
            return False

        candidate_content = " ".join(candidate.content.split())
        for existing in reversed(wm._temp_experiences[-12:]):
            existing_content = " ".join((existing.content or "").split())
            if (
                existing.source == candidate.source
                and existing.category == candidate.category
                and existing.tool_name == candidate.tool_name
                and existing.source_task_id == candidate.source_task_id
                and existing_content == candidate_content
            ):
                return True
        return False

    def _is_duplicate_task_experience(self, candidate: RoundExperience) -> bool:
        """Deprecated compatibility wrapper for ``_is_duplicate_round_experience``."""
        return self._is_duplicate_round_experience(candidate)

    @staticmethod
    def _auto_generate_keywords(content: str, tool_name: str) -> list[str]:
        """从 content 和 tool_name 自动提取关键词"""
        keywords = []
        if tool_name:
            keywords.append(tool_name)

        # 常见模式词
        patterns = [
            "overflow", "layout", "spacing", "font", "color", "image",
            "text", "title", "card", "grid", "flex", "margin", "padding",
            "error", "failed", "limit", "constraint", "avoid", "must",
            "溢出", "布局", "间距", "字体", "颜色", "图片", "文本", "标题",
        ]
        content_lower = content.lower()
        for p in patterns:
            if p in content_lower:
                keywords.append(p)

        return keywords[:10]  # 限制数量

    async def _infer_dimension(self, text: str) -> str:
        """推断偏好文本的维度：向量匹配 → 关键词降级 → 空字符串。"""
        # 1. DimensionMatcher 向量匹配
        if self._dim_matcher is not None:
            try:
                dim = await self._dim_matcher.match(text)
                if dim != "general":
                    return dim
            except Exception as e:
                logger.debug("DimensionMatcher failed in _infer_dimension: %s", e)

        # 2. 关键词降级：用 UserProfile 的静态关键词
        if self._profile_store:
            try:
                from .core.models import UserProfile
                profile = UserProfile()
                dim = profile.match_dimension(text)
                if dim != "general":
                    return dim
            except Exception:
                pass

        return ""

    def get_wm_general_rules_text(self) -> str:
        """高优先级输出当前 WM 中的 general 规则。"""
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_preferences:
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for pref in wm._temp_preferences:
            if pref.superseded:
                continue
            dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
            if dim_prefix not in {"", "general"}:
                continue

            candidate_texts: list[str] = []
            if isinstance(pref.structured_data, dict):
                normalized = str(pref.structured_data.get("normalized_sentence", "") or "").strip()
                if normalized:
                    candidate_texts.append(normalized)

            if dim_prefix == "general":
                flattened = wm.extract_general_preference_items(pref.content)
                candidate_texts.extend(flattened or [pref.content])
            elif pref.content:
                candidate_texts.append(pref.content)

            for text in candidate_texts:
                normalized_text = " ".join(str(text or "").split()).strip()
                if not normalized_text or normalized_text in seen:
                    continue
                seen.add(normalized_text)
                lines.append(f"- {normalized_text}")

        if not lines:
            return ""
        return "\n".join(
            [
                '<working_memory_general_rules priority="highest" note="当前会话全局偏好/约束；后续轮次与新增页应优先遵守">',
                *lines,
                "</working_memory_general_rules>",
            ]
        )

    def get_wm_structured_rules_text(self) -> str:
        """输出 current WM 中可机读的结构化规则 JSONL。"""
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_preferences:
            return ""

        json_lines: list[str] = []
        seen_rule_ids: set[str] = set()
        fallback_index = 0
        for pref in wm._temp_preferences:
            if pref.superseded or not isinstance(pref.structured_data, dict):
                continue

            data = pref.structured_data
            rule_spec = data.get("rule_spec") if isinstance(data.get("rule_spec"), dict) else None
            spec = deepcopy(rule_spec) if rule_spec else deepcopy(data)
            if not isinstance(spec, dict):
                continue
            if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
                continue

            fallback_index += 1
            spec.setdefault("schema_version", "wm_rule_v1")
            spec.setdefault("dimension", pref.dimension or "general")
            spec.setdefault("scope", pref.scope or "global")
            spec.setdefault("content", pref.content)
            spec.setdefault(
                "normalized_sentence",
                str(data.get("normalized_sentence", "") or pref.content or "").strip(),
            )
            spec.setdefault(
                "retention_scope",
                str(data.get("retention_scope", "") or "").strip() or "job_local",
            )
            rule_id = str(spec.get("rule_id", "") or "").strip() or f"wm_rule_{fallback_index:03d}"
            if rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)
            spec["rule_id"] = rule_id
            json_lines.append(json.dumps(spec, ensure_ascii=False))

        if not json_lines:
            return ""
        return "\n".join(
            [
                '<working_memory_rule_specs format="jsonl" priority="highest" note="当前会话结构化全局偏好/约束（可机读子集）">',
                *json_lines,
                "</working_memory_rule_specs>",
            ]
        )

    def get_wm_preferences_text(
        self,
        relevant_dims: set[str] | None = None,
        *,
        include_general: bool = False,
    ) -> str:
        """获取 WM 中当前生效的临时偏好文本（每轮注入用）。

        WM 偏好优先级高于 LTM UserProfile：反映当前 Job 内用户的最新意图。

        Args:
            relevant_dims: 维度筛选集合（来自 RoundCache 的 DimensionMatcher 结果）。
                           None 表示全量注入（WM 偏好数量有限，全量注入是合理的兜底）。
                           非 None 时只注入匹配维度 + 无维度标签的偏好。
            include_general: 是否把 general 一并混入偏好文本；默认关闭，由独立 general block 承载。
        """
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_preferences:
            return ""
        active = [p for p in wm._temp_preferences if not p.superseded]
        if not active:
            return ""

        # 维度筛选（与 RoundCache.search_preferences_by_dimension 同口径）
        if relevant_dims is not None:
            filtered = []
            for p in active:
                dim_prefix = p.dimension.split(".")[0] if p.dimension else ""
                if dim_prefix in relevant_dims or dim_prefix == "":
                    filtered.append(p)
                elif include_general and dim_prefix == "general":
                    filtered.append(p)
            active = filtered
            if not active:
                return ""

        # 按一级维度分组，结构化展示
        from collections import defaultdict
        dim_groups: dict[str, list] = defaultdict(list)
        for p in active:
            if not include_general and (p.dimension.split(".")[0] if p.dimension else "") in {"", "general"}:
                continue
            # 提取一级维度：theme.primary_colors → Theme
            full_dim = p.dimension if p.dimension else "general"
            top_level = full_dim.split(".")[0].capitalize() if "." in full_dim else full_dim.capitalize()
            if full_dim == "general":
                flattened = wm.extract_general_preference_items(p.content)
                items = flattened or [p.content]
                for item in items:
                    dim_groups[top_level].append((item, full_dim))
            else:
                dim_groups[top_level].append((p.content, full_dim))

        # 按固定顺序排列维度
        if not dim_groups:
            return ""
        dim_order = ["Theme", "Visual", "Layout", "Content", "Template", "General"]
        lines = ['<working_memory_preferences priority="high" note="当前会话偏好，优先级高于历史偏好">']
        for dim_key in dim_order:
            if dim_key not in dim_groups:
                continue
            prefs = dim_groups[dim_key]
            lines.append(f"  [{dim_key}]")
            for content, full_dim in prefs:
                # 显示完整维度路径作为标注
                dim_suffix = f" ({full_dim})" if full_dim != dim_key.lower() else ""
                lines.append(f"    - {content}{dim_suffix}")
        # 处理未知维度
        for dim_key in sorted(dim_groups.keys()):
            if dim_key not in dim_order:
                prefs = dim_groups[dim_key]
                lines.append(f"  [{dim_key}]")
                for content, full_dim in prefs:
                    lines.append(f"    - {content} ({full_dim})")
        lines.append("</working_memory_preferences>")
        return "\n".join(lines)

    def get_round_history_prompt(self) -> str:
        """获取 Round 历史摘要 prompt。"""
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if wm:
            return wm.job_history.build_context_prompt()
        return ""

    def get_task_history_prompt(self) -> str:
        """Deprecated compatibility wrapper for ``get_round_history_prompt``."""
        return self.get_round_history_prompt()

    def _get_active_job_intent(self) -> str:
        if self._job_mgr and self._job_mgr._active_job:
            return str(self._job_mgr._active_job.intent or "").strip().lower()
        return ""

    def _build_persona_task_contract(self) -> str:
        if not self._job_mgr or not self._job_mgr._active_job:
            return ""

        active_job = self._job_mgr._active_job
        task_intent = str(active_job.intent or "").strip().lower()
        read_intent = str(active_job.read_intent or task_intent or "").strip().lower()
        write_intent = str(active_job.write_intent or read_intent or task_intent or "").strip().lower()
        core_persona = str(active_job.core_persona or "").strip()
        if not any((task_intent, read_intent, write_intent, core_persona)):
            return ""

        persona_label = core_persona or "unspecified"
        task_label = task_intent or "unspecified"
        read_label = read_intent or "unspecified"
        write_label = write_intent or "unspecified"

        lines = ["<persona_task_contract>"]
        if core_persona and task_intent:
            if core_persona.lower() == task_intent:
                lines.append(f"- 用户稳定 persona 与当前任务场景都按「{persona_label}」理解。")
            else:
                lines.append(f"- 把用户理解为「{persona_label}」persona，此次正在做「{task_label}」场景的 PPT。")
        elif core_persona:
            lines.append(f"- 用户稳定 persona: 「{persona_label}」。")
        else:
            lines.append(f"- 当前任务场景: 「{task_label}」。")

        lines.append(f"- 读取画像桶: 「{read_label}」；新的稳定偏好优先写回「{write_label}」。")
        lines.append("- 先满足当前任务目标，再尽量保留稳定 persona 的表达习惯与审美底色。")
        lines.append("- memory_read_intent 只是参考来源，不等于要把当前任务改写成那个 intent。")
        lines.append("- 只有跨轮稳定、不是一次性执行约束的新偏好，才适合沉淀到 write bucket。")
        lines.append("</persona_task_contract>")
        return "\n".join(lines)

    def _build_memory_context_meta(self) -> str:
        if not self._job_mgr or not self._job_mgr._active_job:
            return ""
        active_job = self._job_mgr._active_job
        task_intent = str(active_job.intent or "").strip().lower()
        read_intent = str(active_job.read_intent or task_intent or "").strip().lower()
        write_intent = str(active_job.write_intent or read_intent or task_intent or "").strip().lower()
        core_persona = str(active_job.core_persona or "").strip()
        if not any((task_intent, read_intent, write_intent, core_persona)):
            return ""

        persona_line = f"- core_persona: {core_persona or 'unspecified'}"
        task_line = f"- current_task_intent: {task_intent or 'unspecified'}"
        read_line = f"- memory_read_intent: {read_intent or 'unspecified'}"
        write_line = f"- memory_write_intent: {write_intent or 'unspecified'}"

        return (
            "<memory_context_meta>\n"
            f"{persona_line}\n"
            f"{task_line}\n"
            f"{read_line}\n"
            f"{write_line}\n"
            "- Default policy: 优先满足 current_task_intent，同时把 active memory 当作当前 deck 的默认风格/约束。\n"
            "- Latest explicit user instruction only overrides the conflicting part; unrelated active preferences remain valid.\n"
            "- New slides should also inherit active deck-level defaults unless the user explicitly asks for an exception.\n"
            "</memory_context_meta>"
        )

    def _build_memory_execution_brief(self) -> str:
        """把当前 active WM 偏好压缩成更易执行的前置摘要。"""
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_preferences:
            return ""

        general_items: list[str] = []
        dim_groups: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()

        def _normalize_item(text: str) -> str:
            normalized = re.sub(r"\s+", " ", str(text or "")).strip()
            normalized = re.sub(r"^#+\s*", "", normalized)
            normalized = normalized.lstrip("- ").strip()
            return normalized[:180]

        for pref in wm._temp_preferences:
            if pref.superseded:
                continue
            dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
            if dim_prefix in {"", "general"}:
                raw_items = wm.extract_general_preference_items(pref.content) or [pref.content]
            else:
                raw_items = [pref.content]

            for raw_item in raw_items:
                item = _normalize_item(raw_item)
                if not item or item in seen:
                    continue
                seen.add(item)
                if dim_prefix in {"", "general"}:
                    general_items.append(item)
                else:
                    dim_groups[dim_prefix].append(item)

        if not general_items and not dim_groups:
            return ""

        dim_labels = {
            "theme": "Theme",
            "visual": "Visual",
            "layout": "Layout",
            "content": "Content",
            "template": "Template",
        }
        ordered_dims = ["theme", "visual", "layout", "content", "template"]

        lines = [
            '<working_memory_execution_brief priority="highest" note="优先把这些 active 画像/偏好当作默认执行方向；只有本轮用户明确冲突时才局部覆盖">',
        ]
        if general_items:
            lines.append(f"- General defaults: {'; '.join(general_items[:3])}")
        for dim in ordered_dims:
            items = dim_groups.get(dim, [])
            if items:
                lines.append(f"- {dim_labels[dim]}: {'; '.join(items[:3])}")
        lines.append("</working_memory_execution_brief>")
        return "\n".join(lines)

    async def compile_and_register_profile_execution_contract(
        self,
        *,
        instruction: str,
        resolved_intent_artifact: dict[str, Any] | None = None,
        source_evidence_summary: str = "",
        core_persona: str = "",
        task_intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
    ) -> dict[str, Any] | None:
        """Compile runtime profile contract for design-stage page planning."""
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not getattr(wm, "_temp_preferences", None):
            self._profile_execution_contract = None
            return None

        from .inject.profile_execution_contract import ProfileExecutionContractCompiler

        compiler = ProfileExecutionContractCompiler(self._profile_injection_llm or self._llm)
        contract = await compiler.compile(
            wm_preferences=list(getattr(wm, "_temp_preferences", []) or []),
            instruction=instruction,
            resolved_intent_artifact=resolved_intent_artifact,
            source_evidence_summary=source_evidence_summary,
            core_persona=core_persona,
            task_intent=task_intent,
            read_intent=read_intent,
            write_intent=write_intent,
        )
        self._profile_execution_contract = contract or None

        if self._artifact_dumper and contract:
            try:
                self._artifact_dumper.dump_profile_execution_contract(
                    round_index=0,
                    contract=contract,
                )
            except Exception as exc:
                logger.warning(
                    "ArtifactDumper dump_profile_execution_contract failed (non-fatal): %s",
                    exc,
                )

        return self._profile_execution_contract

    async def compile_and_register_profile_execution_plan(
        self,
        *,
        instruction: str,
        source_evidence_summary: str = "",
        target_page_count: int = 0,
        template_mode: bool = False,
    ) -> dict[str, Any] | None:
        """Compile runtime page-level persona execution plan for design-stage execution."""
        contract = self._profile_execution_contract
        if not contract:
            self._profile_execution_plan = None
            return None

        from .inject.profile_execution_contract import ProfileExecutionPlanCompiler

        compiler = ProfileExecutionPlanCompiler(self._profile_injection_llm or self._llm)
        plan = await compiler.compile(
            contract=contract,
            instruction=instruction,
            source_evidence_summary=source_evidence_summary,
            target_page_count=target_page_count,
            template_mode=template_mode,
        )
        self._profile_execution_plan = plan or None

        if self._artifact_dumper and plan:
            try:
                self._artifact_dumper.dump_profile_execution_plan(
                    round_index=0,
                    plan=plan,
                )
            except Exception as exc:
                logger.warning(
                    "ArtifactDumper dump_profile_execution_plan failed (non-fatal): %s",
                    exc,
                )

        return self._profile_execution_plan

    def get_profile_execution_contract_payload(self) -> dict[str, Any] | None:
        return deepcopy(self._profile_execution_contract) if self._profile_execution_contract else None

    def get_profile_execution_plan_payload(self) -> dict[str, Any] | None:
        return deepcopy(self._profile_execution_plan) if self._profile_execution_plan else None

    def set_profile_realization_report(self, report: dict[str, Any] | None) -> None:
        self._profile_realization_report = deepcopy(report) if report else None
        if self._artifact_dumper and report:
            try:
                self._artifact_dumper.dump_profile_realization_report(
                    round_index=0,
                    report=report,
                )
            except Exception as exc:
                logger.warning(
                    "ArtifactDumper dump_profile_realization_report failed (non-fatal): %s",
                    exc,
                )

    def get_profile_execution_contract_text(self, agent_name: str = "") -> str:
        if agent_name != "design" or not self._profile_execution_contract:
            return ""
        from .inject.profile_execution_contract import format_profile_execution_contract

        return format_profile_execution_contract(self._profile_execution_contract)

    def get_profile_execution_plan_text(self, agent_name: str = "") -> str:
        if agent_name != "design" or not self._profile_execution_plan:
            return ""
        from .inject.profile_execution_contract import format_profile_execution_plan

        return format_profile_execution_plan(self._profile_execution_plan)

    async def get_memory_components(
        self,
        tool_name: str = "*",
        tool_args: dict | None = None,
        context: dict | None = None,
        user_message: str = "",
        agent_name: str = "",
    ) -> dict[str, str]:
        """返回各记忆组件的独立文本（供 injection trace 完整记录）。

        5 个组件（按优先级排序）：
        - wm_preferences:       WM 临时偏好（ProfileInjectionRouter 注入的画像 + 当前 Job 偏好）
        - profile_execution_contract: 仅 design 阶段可见的画像执行合同（页级义务）
        - profile_execution_plan: 仅 design 阶段可见的逐页 persona 执行计划
        - wm_experiences:       WM 临时经验（remember_lesson 写入，当前 Job 内即时生效）
        - ltm_tool_experiences: 工具链经验文本（LTM）— 仅在 Operation 级别按工具名匹配
        - wm_round_history:     轮次历史摘要（RoundSummary via JobHistory）

        注意：
        - ProfileInjectionRouter 在 Job 开始时将 LTM UserProfile 按维度写入 WM，
          因此所有阶段都通过 wm_preferences 统一获取偏好。
        - AtomicPreference 不注入 prompt，仅用于更新 UserProfile。
        - ltm_tool_experiences 在 System 级别为空（tool_name="*" 时无法匹配），
          仅在 Operation 级别按具体工具名实时匹配。

        Returns:
            {
                "wm_preferences": ...,
                "wm_experiences": ...,
                "ltm_tool_experiences": ...,
                "wm_round_history": ...,
                "combined": ...,
            }
        """
        result: dict[str, str] = {
            "wm_general_rules": "",
            "wm_rule_specs": "",
            "wm_preferences": "",
            "profile_execution_contract": "",
            "profile_execution_plan": "",
            "wm_experiences": "",
            "ltm_tool_experiences": "",
            "wm_round_history": "",
            "combined": "",
        }

        # WM 偏好 — 统一注入入口
        # ProfileInjectionRouter 已将画像按维度写入 WM（标记为 ltm_profile_inject / profile_override），
        # 因此所有阶段（Research/Design/Modify）都通过 wm_preferences 获取偏好。
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if self._enable_wm_preference_injection:
            result["wm_general_rules"] = self.get_wm_general_rules_text()
            result["wm_rule_specs"] = self.get_wm_structured_rules_text()
            result["wm_preferences"] = self.get_wm_preferences_text(
                relevant_dims=None,
                include_general=False,
            )
            result["profile_execution_contract"] = self.get_profile_execution_contract_text(agent_name)
            result["profile_execution_plan"] = self.get_profile_execution_plan_text(agent_name)

        # WM 经验（remember_lesson / auto_extract / preload）
        if self._enable_wm_experience_injection and wm and wm._temp_experiences:
            candidate_pool_size = min(len(wm._temp_experiences), TASK_EXPERIENCE_CANDIDATE_POOL)

            # 先取更大的相关候选池，再按类别平衡选择实际注入条目
            if user_message:
                matched_exps = await wm.search_experiences_by_keywords(
                    user_message,
                    top_k=max(candidate_pool_size, TASK_INJECTION_TOTAL_LIMIT),
                )
            else:
                matched_exps = wm.get_experiences()[:candidate_pool_size]

            matched_ids = {id(e) for e in matched_exps}
            unmatched_lessons = [
                e for e in wm._temp_experiences
                if e.source == "agent_lesson" and id(e) not in matched_ids
            ]
            if unmatched_lessons:
                tool_terms = []
                for e in unmatched_lessons:
                    if e.tool_name:
                        tool_terms.append(e.tool_name)
                    tool_terms.extend(e.keywords[:2])
                enhanced_query = " ".join(dict.fromkeys(tool_terms))  # 去重保序
                if enhanced_query:
                    secondary = await wm.search_experiences_by_keywords(
                        enhanced_query,
                        top_k=TASK_EXPERIENCE_SECONDARY_POOL,
                    )
                    for exp in secondary:
                        if id(exp) not in matched_ids:
                            matched_exps.append(exp)
                            matched_ids.add(id(exp))

            if matched_exps:
                filtered_exps = self._filter_round_injection_experiences(
                    agent_name,
                    matched_exps,
                )
                selected_exps = self._select_round_injection_experiences(filtered_exps)
                if selected_exps:
                    result["wm_experiences"] = self._format_wm_experiences(selected_exps)

        # ltm_tool_experiences: System 级别不注入（tool_name="*" 无法匹配具体工具）
        # 仅在 Operation 级别通过 OperationDistributor 按工具名实时匹配

        # WM 轮次历史
        if self._enable_wm_round_history_injection:
            result["wm_round_history"] = self.get_round_history_prompt()

        # 拼接。即使当前还没有可注入的偏好/经验，也显式告诉主模型当前使用的记忆桶。
        memory_execution_brief = self._build_memory_execution_brief()
        persona_task_contract = self._build_persona_task_contract()
        memory_context_meta = self._build_memory_context_meta()
        parts = [
            memory_execution_brief,
            result["wm_general_rules"],
            result["wm_rule_specs"],
            result["wm_preferences"],
            result["profile_execution_contract"],
            result["profile_execution_plan"],
            persona_task_contract,
            memory_context_meta,
            result["wm_experiences"],
            result["wm_round_history"],
        ]
        result["combined"] = "\n\n".join(filter(None, parts))
        return result

    def _format_wm_experiences(self, experiences: list[RoundExperience]) -> str:
        """格式化 WM 经验为注入 prompt 文本"""
        if not experiences:
            return ""
        lines = ['<working_memory_experiences priority="high" note="当前会话经验，优先级高于历史经验">']
        for exp in experiences:
            lines.append(exp.to_prompt_text())
        lines.append("</working_memory_experiences>")
        return "\n".join(lines)

    @staticmethod
    def _normalize_experience_text(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _is_design_relevant_round_experience(cls, exp: RoundExperience) -> bool:
        tool_names = {
            cls._normalize_experience_text(exp.tool_name),
            *(cls._normalize_experience_text(name) for name in exp.tools_used),
        }
        tool_names.discard("")

        if tool_names & DESIGN_TASK_EXPERIENCE_BLOCKED_TOOLS:
            return False
        if tool_names & DESIGN_TASK_EXPERIENCE_ALLOWED_TOOLS:
            return True

        haystack_parts = [
            exp.content,
            exp.tool_name,
            *exp.keywords,
            *exp.tools_used,
        ]
        haystack = " ".join(
            cls._normalize_experience_text(part) for part in haystack_parts if part
        )
        if not haystack:
            return False
        if any(term in haystack for term in DESIGN_TASK_EXPERIENCE_BLOCKED_TERMS):
            return False
        return any(term in haystack for term in DESIGN_TASK_EXPERIENCE_ALLOWED_TERMS)

    def _filter_round_injection_experiences(
        self,
        agent_name: str,
        ranked_experiences: list[RoundExperience],
    ) -> list[RoundExperience]:
        normalized_agent = self._normalize_experience_text(agent_name)
        if normalized_agent != "design":
            return ranked_experiences

        filtered = [
            exp for exp in ranked_experiences
            if self._is_design_relevant_round_experience(exp)
        ]
        removed = len(ranked_experiences) - len(filtered)
        if removed > 0:
            logger.info(
                "[MemoryV2] Filtered %d/%d WM experiences for design prompt; kept %d design-relevant items",
                removed,
                len(ranked_experiences),
                len(filtered),
            )
        return filtered

    @staticmethod
    def _select_ranked_items_by_category(
        items: list[Any],
        category_limits: dict[str, int],
        total_limit: int,
        get_category: Any,
        get_key: Any,
        fill_categories: set[str] | None = None,
        max_per_category: dict[str, int] | None = None,
    ) -> list[Any]:
        """按类别配额从已排序候选中做平衡选择。"""
        if not items or total_limit <= 0:
            return []

        selected: list[Any] = []
        selected_keys: set[str] = set()
        counts: dict[str, int] = {}

        def _can_add(category: str) -> bool:
            if max_per_category is None:
                return True
            if category not in max_per_category:
                return True
            return counts.get(category, 0) < max_per_category[category]

        def _try_add(item: Any) -> bool:
            key = str(get_key(item))
            if key in selected_keys:
                return False
            category = str(get_category(item))
            if not _can_add(category):
                return False
            selected.append(item)
            selected_keys.add(key)
            counts[category] = counts.get(category, 0) + 1
            return True

        for category, limit in category_limits.items():
            taken = 0
            for item in items:
                if taken >= limit or len(selected) >= total_limit:
                    break
                if str(get_category(item)) != category:
                    continue
                if _try_add(item):
                    taken += 1
            if len(selected) >= total_limit:
                return selected

        allowed_fill = fill_categories if fill_categories is not None else set(category_limits)
        for item in items:
            if len(selected) >= total_limit:
                break
            category = str(get_category(item))
            if category not in allowed_fill:
                continue
            _try_add(item)

        return selected

    @staticmethod
    def _row_experience_type(row: dict[str, Any]) -> str:
        return infer_experience_type(
            row.get("experience_type", ""),
            row.get("applicable_scenarios", "[]"),
            row.get("final_outcome", ""),
        )

    def _select_preload_rows(self, ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从重排后的 LTM 候选中选择平衡的 preload 集合。"""
        eligible_rows = [
            row for row in ranked_rows
            if self._row_experience_type(row) not in {"chain", "agent_run"}
            and not self._is_stale_structural_tool_limitation_text(
                "\n".join(
                    str(row.get(key, "") or "")
                    for key in ("task_description", "lessons_learned", "anti_pattern", "applicable_when")
                )
            )
        ]
        return self._select_ranked_items_by_category(
            items=eligible_rows,
            category_limits=PRELOAD_CATEGORY_QUOTAS,
            total_limit=PRELOAD_TOTAL_LIMIT,
            get_category=lambda row: infer_task_experience_category(
                row.get("experience_type", ""),
                row.get("applicable_scenarios", "[]"),
                row.get("final_outcome", ""),
            ),
            get_key=lambda row: row.get("id", ""),
            fill_categories=PRELOAD_FILL_CATEGORIES,
            max_per_category=PRELOAD_CATEGORY_MAX,
        )

    def _select_round_injection_experiences(
        self,
        ranked_experiences: list[RoundExperience],
    ) -> list[RoundExperience]:
        """从已排序 WM 经验中按类别平衡选择注入条目。"""
        ranked_experiences = [
            exp for exp in ranked_experiences
            if not self._is_stale_structural_tool_limitation_text(
                "\n".join(
                    str(part or "")
                    for part in (
                        getattr(exp, "content", ""),
                        getattr(exp, "tool_name", ""),
                        " ".join(getattr(exp, "keywords", []) or []),
                    )
                )
            )
        ]
        return self._select_ranked_items_by_category(
            items=ranked_experiences,
            category_limits=TASK_INJECTION_CATEGORY_LIMITS,
            total_limit=TASK_INJECTION_TOTAL_LIMIT,
            get_category=lambda exp: infer_task_experience_category(
                exp.category,
                exp.keywords,
                exp.outcome,
            ),
            get_key=lambda exp: exp.id,
            fill_categories=TASK_INJECTION_FILL_CATEGORIES,
            max_per_category=TASK_INJECTION_CATEGORY_LIMITS,
        )

    @staticmethod
    def _is_stale_structural_tool_limitation_text(text: str) -> bool:
        source = str(text or "").lower()
        if not source:
            return False
        if not any(signal in source for signal in ("insert_slide", "write_new_slide_file", "插入", "新增", "新建")):
            return False
        return any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in _STALE_STRUCTURAL_TOOL_LIMITATION_PATTERNS)

    def _filter_task_injection_experiences(
        self,
        agent_name: str,
        ranked_experiences: list[RoundExperience],
    ) -> list[RoundExperience]:
        """Deprecated compatibility wrapper for round experience filtering."""
        return self._filter_round_injection_experiences(agent_name, ranked_experiences)

    def _select_task_injection_experiences(
        self,
        ranked_experiences: list[RoundExperience],
    ) -> list[RoundExperience]:
        """Deprecated compatibility wrapper for round experience selection."""
        return self._select_round_injection_experiences(ranked_experiences)

    # ── Stage 14: WM 经验 ↔ LTM 同步 ──

    async def _consolidate_experiences_to_ltm(
        self,
        progress_callback: Any = None,
        progress_start: int = 10,
        progress_end: int = 30,
    ) -> None:
        """Job 结束时统一归档 WM RoundExperience → LTM。"""
        from .progress import emit_memory_save_progress

        if not self._enable_round_experience_writeback:
            logger.info("_consolidate_experiences_to_ltm: round experience writeback disabled")
            await emit_memory_save_progress(
                progress_callback,
                stage="round_experience_writeback",
                progress=progress_end,
                message="Round experience writeback is disabled.",
            )
            return
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not wm._temp_experiences:
            logger.debug("_consolidate_experiences_to_ltm: no WM experiences to consolidate")
            await emit_memory_save_progress(
                progress_callback,
                stage="round_experience_writeback",
                progress=progress_end,
                message="No round experiences need consolidation.",
            )
            return

        all_experiences = wm.get_experiences()
        experiences = [e for e in all_experiences if self._should_consolidate_experience(e)]
        if not experiences:
            logger.debug(
                "_consolidate_experiences_to_ltm: %d total, 0 eligible experiences → skip",
                len(all_experiences),
            )
            await emit_memory_save_progress(
                progress_callback,
                stage="round_experience_writeback",
                progress=progress_end,
                message=f"{len(all_experiences)} round experiences found; none are eligible for writeback.",
            )
            return

        grouped: dict[str, list[RoundExperience]] = {}
        for exp in experiences:
            cat = infer_task_experience_category(exp.category, exp.keywords, exp.outcome)
            grouped.setdefault(cat, []).append(exp)

        cluster_counts = {
            category: len(self._cluster_round_experiences(bucket))
            for category, bucket in grouped.items()
        }
        total_clusters = max(1, sum(cluster_counts.values()))
        completed_clusters = 0

        logger.info(
            "[MemoryV2] Consolidating %d eligible experiences in %d categories",
            len(experiences), len(grouped),
        )
        await emit_memory_save_progress(
            progress_callback,
            stage="round_experience_writeback",
            progress=progress_start,
            message=(
                f"Consolidating {len(experiences)} eligible round experiences "
                f"across {len(grouped)} categories."
            ),
            total_experiences=len(experiences),
            total_clusters=total_clusters,
        )

        session_id = ""
        if self._job_mgr and self._job_mgr._active_job:
            active_job = self._job_mgr._active_job
            # ExperienceTrace.session_id 应绑定到真实 session/project，而不是瞬时 job UUID。
            session_id = getattr(active_job, "project_id", "") or getattr(active_job, "id", "")

        for category, bucket in grouped.items():
            clusters = self._cluster_round_experiences(bucket)
            cluster_total = len(clusters)
            for cluster_index, cluster in enumerate(clusters, start=1):
                try:
                    await emit_memory_save_progress(
                        progress_callback,
                        stage="round_experience_writeback",
                        progress=progress_start + int(
                            max(1, progress_end - progress_start)
                            * completed_clusters
                            / total_clusters
                        ),
                        message=(
                            f"Writing round experience cluster {cluster_index}/{cluster_total} "
                            f"for {category}."
                        ),
                        category=category,
                        completed_clusters=completed_clusters,
                        total_clusters=total_clusters,
                    )
                    merged_trace = await self._merge_and_write_category(
                        category=category,
                        experiences=cluster,
                        session_id=session_id,
                        cluster_index=cluster_index,
                        cluster_total=cluster_total,
                    )
                    if merged_trace:
                        logger.info(
                            "[MemoryV2] Consolidated %d %s experiences → LTM (%s)",
                            len(cluster), category, merged_trace.id[:8],
                        )
                except Exception as e:
                    logger.warning(f"_consolidate_experiences_to_ltm failed for {category}: {e}")
                finally:
                    completed_clusters += 1

        await emit_memory_save_progress(
            progress_callback,
            stage="round_experience_writeback",
            progress=progress_end,
            message="Round experience writeback completed.",
            completed_clusters=completed_clusters,
            total_clusters=total_clusters,
        )

    async def _merge_and_write_category(
        self,
        category: str,
        experiences: list[RoundExperience],
        session_id: str,
        cluster_index: int = 0,
        cluster_total: int = 0,
    ) -> "ExperienceTrace | None":
        """将单个 cluster 与已有 LTM 近邻对齐后写入。"""
        from .core.models import ExperienceTrace

        if not experiences:
            return None

        trace = await self._build_trace_from_cluster(category, experiences, session_id)
        neighbors = await self._retrieve_category_neighbors(category, experiences)
        decision = await self._decide_consolidation_relation(category, trace, neighbors)

        target_row = None
        if decision.get("target_id"):
            target_row = next(
                (row for row in neighbors if row.get("id") == decision["target_id"]),
                None,
            )

        if target_row:
            existing_keywords = self._coerce_json_list(target_row.get("applicable_scenarios", "[]"))
            existing_tools = self._coerce_json_list(target_row.get("tools_used", "[]"))
            trace.applicable_scenarios = list(
                dict.fromkeys(self._coerce_json_list(trace.applicable_scenarios) + existing_keywords)
            )[:15]
            trace.tools_used = list(
                dict.fromkeys(self._coerce_json_list(trace.tools_used) + existing_tools)
            )[:10]
            lineage_ids = list(
                dict.fromkeys(
                    [*self._coerce_json_list(target_row.get("merged_from_ids", "[]")), target_row.get("id", "")]
                    + [exp.id for exp in experiences]
                )
            )
            trace.merged_from_ids = [item for item in lineage_ids if item]

            source_types = list(
                dict.fromkeys(
                    self._coerce_json_list(target_row.get("source_types", "[]"))
                    + [exp.source for exp in experiences]
                )
            )
            trace.source_types = [item for item in source_types if item]
            trace.confidence = max(float(target_row.get("confidence", 0.7)), trace.confidence)

        relation = decision.get("relation", "unrelated")
        superseded_target_id = ""
        ltm_write_succeeded: bool | None = None
        if relation == "merge" and target_row:
            merged_lesson = decision.get("merged_lesson", "").strip()
            if not merged_lesson:
                merged_lesson = await self._merge_new_and_existing_content(trace, target_row)
            if merged_lesson:
                trace.lessons_learned = merged_lesson
            trace.source_types = list(dict.fromkeys(self._coerce_json_list(trace.source_types) + ["ltm_merge"]))
            superseded_target_id = target_row.get("id", "")
            ltm_write_succeeded = await self._write_experience_to_ltm(trace)
            if ltm_write_succeeded:
                await self._mark_trace_superseded(superseded_target_id, trace.id)
            self._dump_round_experience_consolidation_artifact(
                category=category,
                cluster_index=cluster_index,
                cluster_total=cluster_total,
                session_id=session_id,
                experiences=experiences,
                neighbors=neighbors,
                decision=decision,
                target_row=target_row,
                output_trace=trace,
                superseded_target_id=superseded_target_id,
                ltm_write_succeeded=ltm_write_succeeded,
            )
            return trace

        if relation == "supersede" and target_row:
            trace.source_types = list(dict.fromkeys(self._coerce_json_list(trace.source_types) + ["ltm_supersede"]))
            superseded_target_id = target_row.get("id", "")
            ltm_write_succeeded = await self._write_experience_to_ltm(trace)
            if ltm_write_succeeded:
                await self._mark_trace_superseded(superseded_target_id, trace.id)
            self._dump_round_experience_consolidation_artifact(
                category=category,
                cluster_index=cluster_index,
                cluster_total=cluster_total,
                session_id=session_id,
                experiences=experiences,
                neighbors=neighbors,
                decision=decision,
                target_row=target_row,
                output_trace=trace,
                superseded_target_id=superseded_target_id,
                ltm_write_succeeded=ltm_write_succeeded,
            )
            return trace

        ltm_write_succeeded = await self._write_experience_to_ltm(trace)
        self._dump_round_experience_consolidation_artifact(
            category=category,
            cluster_index=cluster_index,
            cluster_total=cluster_total,
            session_id=session_id,
            experiences=experiences,
            neighbors=neighbors,
            decision=decision,
            target_row=target_row,
            output_trace=trace,
            superseded_target_id=superseded_target_id,
            ltm_write_succeeded=ltm_write_succeeded,
        )
        return trace

    def _dump_round_experience_consolidation_artifact(
        self,
        *,
        category: str,
        cluster_index: int = 0,
        cluster_total: int = 0,
        session_id: str,
        experiences: list[RoundExperience],
        neighbors: list[dict[str, Any]],
        decision: dict[str, Any],
        target_row: dict[str, Any] | None,
        output_trace: "ExperienceTrace",
        superseded_target_id: str = "",
        ltm_write_succeeded: bool | None = None,
    ) -> None:
        if not self._artifact_dumper:
            return
        try:
            self._artifact_dumper.dump_round_experience_consolidation(
                category=category,
                trace_id=getattr(output_trace, "id", ""),
                cluster_index=cluster_index,
                cluster_total=cluster_total,
                session_id=session_id,
                input_experiences=experiences,
                neighbors=neighbors,
                decision=decision,
                selected_target_before=target_row,
                output_trace=output_trace,
                superseded_target_id=superseded_target_id,
                ltm_write_succeeded=ltm_write_succeeded,
            )
        except Exception as e:
            logger.warning("RoundExperience consolidation artifact dump failed (non-fatal): %s", e)

    def _dump_task_experience_consolidation_artifact(self, **kwargs: Any) -> None:
        """Deprecated compatibility wrapper for round experience artifacts."""
        self._dump_round_experience_consolidation_artifact(**kwargs)

    def _should_consolidate_experience(self, exp: RoundExperience) -> bool:
        source = getattr(exp, "source", "")
        if source == "preload":
            return False

        content = (exp.content or "").strip()
        if len(content) < 8:
            return False

        category = infer_task_experience_category(exp.category, exp.keywords, exp.outcome)
        if source in {"agent_lesson", "tool_error"}:
            return True
        if category == "tool_error":
            return True
        if source != "auto_extract":
            return False
        if float(exp.confidence or 0.0) >= AUTO_EXTRACT_LTM_MIN_CONFIDENCE:
            return True
        if category != "generic" and (exp.tool_name or exp.keywords):
            return True
        return bool(exp.tool_name and len(exp.content) >= 20)

    def _cluster_round_experiences(
        self,
        experiences: list[RoundExperience],
    ) -> list[list[RoundExperience]]:
        """同 category 内做轻量聚类，避免整桶强行合并。"""
        if not experiences:
            return []

        ranked = sorted(
            experiences,
            key=lambda exp: (
                exp.tool_name == "",
                -float(exp.confidence or 0.0),
                exp.timestamp,
            ),
        )
        clusters: list[list[RoundExperience]] = []
        for exp in ranked:
            placed = False
            for cluster in clusters:
                score = max(
                    self._pairwise_round_experience_similarity(exp, existing)
                    for existing in cluster
                )
                if score >= CONSOLIDATION_CLUSTER_SIM_THRESHOLD:
                    cluster.append(exp)
                    placed = True
                    break
            if not placed:
                clusters.append([exp])
        return clusters

    def _cluster_task_experiences(
        self,
        experiences: list[RoundExperience],
    ) -> list[list[RoundExperience]]:
        """Deprecated compatibility wrapper for round experience clustering."""
        return self._cluster_round_experiences(experiences)

    def _pairwise_round_experience_similarity(
        self,
        left: RoundExperience,
        right: RoundExperience,
    ) -> float:
        left_tools = {left.tool_name, *left.tools_used}
        right_tools = {right.tool_name, *right.tools_used}
        left_tools.discard("")
        right_tools.discard("")
        left_keywords = {str(k).strip() for k in left.keywords if str(k).strip()}
        right_keywords = {str(k).strip() for k in right.keywords if str(k).strip()}
        left_tokens = self._tokenize_text(left.content)
        right_tokens = self._tokenize_text(right.content)

        def _jaccard(a: set[str], b: set[str]) -> float:
            union = a | b
            if not union:
                return 0.0
            return len(a & b) / len(union)

        return min(
            1.0,
            0.45 * _jaccard(left_tools, right_tools)
            + 0.35 * _jaccard(left_keywords, right_keywords)
            + 0.20 * _jaccard(left_tokens, right_tokens),
        )

    def _pairwise_task_experience_similarity(
        self,
        left: RoundExperience,
        right: RoundExperience,
    ) -> float:
        """Deprecated compatibility wrapper for round experience similarity."""
        return self._pairwise_round_experience_similarity(left, right)

    async def _retrieve_category_neighbors(
        self,
        category: str,
        experiences: list[RoundExperience],
    ) -> list[dict[str, Any]]:
        if not self._db:
            return []
        rows = await self._db.query(
            """
            SELECT * FROM experience_traces
            WHERE experience_type = ?
              AND COALESCE(status, 'active') = 'active'
            ORDER BY reuse_count DESC, confidence DESC, created_at DESC
            LIMIT ?
            """,
            (category, CONSOLIDATION_NEIGHBOR_QUERY_LIMIT),
        )
        scored_rows: list[dict[str, Any]] = []
        for row in rows:
            score = self._score_cluster_against_trace(experiences, row)
            if score <= 0:
                continue
            item = dict(row)
            item["_similarity_score"] = score
            scored_rows.append(item)
        scored_rows.sort(key=lambda row: row.get("_similarity_score", 0.0), reverse=True)
        return scored_rows[:CONSOLIDATION_NEIGHBOR_RETURN_LIMIT]

    def _score_cluster_against_trace(
        self,
        experiences: list[RoundExperience],
        row: dict[str, Any],
    ) -> float:
        cluster_tools: set[str] = set()
        cluster_keywords: set[str] = set()
        cluster_tokens: set[str] = set()
        for exp in experiences:
            cluster_tools.update({exp.tool_name, *exp.tools_used})
            cluster_keywords.update(str(k).strip() for k in exp.keywords if str(k).strip())
            cluster_tokens.update(self._tokenize_text(exp.content))
        cluster_tools.discard("")

        row_tools = set(self._coerce_json_list(row.get("tools_used", "[]")))
        row_keywords = set(self._coerce_json_list(row.get("applicable_scenarios", "[]")))
        row_tokens = self._tokenize_text(
            f"{row.get('task_description', '')}\n{row.get('lessons_learned', '')}"
        )

        def _jaccard(a: set[str], b: set[str]) -> float:
            union = a | b
            if not union:
                return 0.0
            return len(a & b) / len(union)

        return min(
            1.0,
            0.45 * _jaccard(cluster_tools, row_tools)
            + 0.35 * _jaccard(cluster_keywords, row_keywords)
            + 0.20 * _jaccard(cluster_tokens, row_tokens),
        )

    async def _build_trace_from_cluster(
        self,
        category: str,
        experiences: list[RoundExperience],
        session_id: str,
    ) -> "ExperienceTrace":
        from .core.models import ExperienceTrace

        normalized_category = infer_task_experience_category(category)
        merged_content = (
            experiences[0].content
            if len(experiences) == 1
            else await self._llm_merge_experiences(category, experiences)
        )
        if not merged_content:
            merged_content = "\n".join(
                f"- {exp.content.strip()}" for exp in experiences if exp.content.strip()
            )[:1000]

        keywords = [normalized_category]
        tools: list[str] = []
        for exp in experiences:
            keywords.extend(exp.keywords)
            tools.extend([exp.tool_name, *exp.tools_used])
        unique_keywords = list(dict.fromkeys(k for k in keywords if k))[:15]
        unique_tools = list(dict.fromkeys(t for t in tools if t))[:10]
        outcomes = [str(exp.outcome or "").strip().lower() for exp in experiences if str(exp.outcome or "").strip()]
        if "failed" in outcomes:
            outcome = "failed"
        elif "partial" in outcomes:
            outcome = "partial"
        else:
            outcome = default_outcome_for_experience_type(normalized_category, outcomes[0] if outcomes else "")
        confidence_values = [float(exp.confidence or 0.7) for exp in experiences]
        confidence = max(confidence_values) if len(experiences) == 1 else min(0.95, sum(confidence_values) / len(confidence_values) + 0.05)

        source_types = list(dict.fromkeys(exp.source for exp in experiences if exp.source))
        merged_from_ids = [exp.id for exp in experiences if exp.id]

        summary = experiences[0].content[:140] if len(experiences) == 1 else f"cluster of {len(experiences)} experiences"
        return ExperienceTrace(
            session_id=session_id or "default",
            task_description=f"[{normalized_category}] {summary}",
            tools_used=unique_tools,
            final_outcome=outcome,
            lessons_learned=merged_content[:1000],
            applicable_scenarios=unique_keywords,
            confidence=confidence,
            status="active",
            merged_from_ids=merged_from_ids,
            source_types=source_types,
            consolidation_version="stage15",
            experience_type=normalized_category,
        )

    async def _decide_consolidation_relation(
        self,
        category: str,
        new_trace: "ExperienceTrace",
        neighbors: list[dict[str, Any]],
    ) -> dict[str, str]:
        if not neighbors:
            return {"relation": "unrelated", "target_id": "", "merged_lesson": ""}

        if not self._llm:
            return self._heuristic_relation(new_trace, neighbors)

        neighbor_payload = [
            {
                "id": row.get("id", ""),
                "score": round(float(row.get("_similarity_score", 0.0)), 4),
                "tools": self._coerce_json_list(row.get("tools_used", "[]")),
                "keywords": self._coerce_json_list(row.get("applicable_scenarios", "[]")),
                "lesson": row.get("lessons_learned", "")[:220],
            }
            for row in neighbors
        ]
        prompt = f"""你是经验归档判定器。请判断新经验与已有 LTM 候选的关系。

类别: {category}

新经验:
{json.dumps({
    "task_description": new_trace.task_description,
    "lesson": new_trace.lessons_learned,
    "tools": self._coerce_json_list(new_trace.tools_used),
    "keywords": self._coerce_json_list(new_trace.applicable_scenarios),
}, ensure_ascii=False, indent=2)}

候选经验:
{json.dumps(neighbor_payload, ensure_ascii=False, indent=2)}

判定规则:
- merge: 讲的是同一个问题，结论兼容或互补，应融合
- supersede: 讲的是同一场景，但新经验明确修正或替代旧经验
- unrelated: 虽然 category 一样，但其实是不同问题

返回 JSON:
{{"relation":"merge|supersede|unrelated","target_id":"候选id或空","merged_lesson":"仅 merge 时填写，不超过220字"}}

只输出 JSON，不要解释。"""
        try:
            try:
                response = await self._llm(prompt)
                text = str(response).strip() if response else ""
            except Exception:
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.choices[0].message.content or "").strip()
            data = self._extract_json_object(text)
            relation = str(data.get("relation", "unrelated")).strip().lower()
            target_id = str(data.get("target_id", "")).strip()
            if relation not in {"merge", "supersede", "unrelated"}:
                return self._heuristic_relation(new_trace, neighbors)
            if relation != "unrelated" and not any(row.get("id") == target_id for row in neighbors):
                return self._heuristic_relation(new_trace, neighbors)
            return {
                "relation": relation,
                "target_id": target_id if relation != "unrelated" else "",
                "merged_lesson": str(data.get("merged_lesson", "")).strip(),
            }
        except Exception as e:
            logger.warning("_decide_consolidation_relation failed: %s", e)
            return self._heuristic_relation(new_trace, neighbors)

    def _heuristic_relation(
        self,
        new_trace: "ExperienceTrace",
        neighbors: list[dict[str, Any]],
    ) -> dict[str, str]:
        best = neighbors[0]
        score = float(best.get("_similarity_score", 0.0))
        if score >= CONSOLIDATION_HEURISTIC_MERGE_THRESHOLD:
            return {"relation": "merge", "target_id": best.get("id", ""), "merged_lesson": ""}
        if score >= CONSOLIDATION_HEURISTIC_RELATED_THRESHOLD and self._has_conflict_signal(
            new_trace.lessons_learned,
            str(best.get("lessons_learned", "")),
        ):
            return {"relation": "supersede", "target_id": best.get("id", ""), "merged_lesson": ""}
        return {"relation": "unrelated", "target_id": "", "merged_lesson": ""}

    async def _merge_new_and_existing_content(
        self,
        new_trace: "ExperienceTrace",
        existing_row: dict[str, Any],
    ) -> str:
        existing_text = str(existing_row.get("lessons_learned", "")).strip()
        if not self._llm:
            merged_lines = [line for line in [existing_text, new_trace.lessons_learned] if line]
            return "\n".join(dict.fromkeys(merged_lines))[:1000]

        prompt = f"""请将下面两条经验融合为一条简洁且可执行的经验，不超过220字。

旧经验:
{existing_text}

新经验:
{new_trace.lessons_learned}

直接输出融合后的经验文本，不要 JSON。"""
        try:
            try:
                response = await self._llm(prompt)
                text = str(response).strip() if response else ""
            except Exception:
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.choices[0].message.content or "").strip()
            return text[:1000] if text else new_trace.lessons_learned
        except Exception:
            merged_lines = [line for line in [existing_text, new_trace.lessons_learned] if line]
            return "\n".join(dict.fromkeys(merged_lines))[:1000]

    async def _mark_trace_superseded(self, trace_id: str, superseded_by: str) -> None:
        if not self._db or not trace_id or not superseded_by:
            return
        from datetime import datetime

        try:
            await self._db.update(
                "experience_traces",
                {
                    "status": "superseded",
                    "superseded_by": superseded_by,
                    "superseded_at": datetime.now().isoformat(),
                },
                {"id": trace_id},
            )
        except Exception as e:
            logger.warning("_mark_trace_superseded failed: %s", e)

    @staticmethod
    def _coerce_json_list(value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [stripped]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", str(text or "").lower()))

    @staticmethod
    def _has_conflict_signal(new_text: str, old_text: str) -> bool:
        negative_terms = ("不要", "不能", "避免", "失败", "不支持", "unsupported", "avoid", "never", "failed")
        positive_terms = ("可以", "推荐", "应该", "优先", "使用", "should", "use", "prefer", "must")
        new_lower = str(new_text or "").lower()
        old_lower = str(old_text or "").lower()
        new_negative = any(term in new_lower for term in negative_terms)
        old_negative = any(term in old_lower for term in negative_terms)
        new_positive = any(term in new_lower for term in positive_terms)
        old_positive = any(term in old_lower for term in positive_terms)
        return (new_negative and old_positive) or (new_positive and old_negative)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start:end])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def _llm_merge_experiences(
        self,
        category: str,
        experiences: list[RoundExperience],
    ) -> str:
        """使用 LLM 合并同类经验。

        Args:
            category: 经验类别
            experiences: 待合并的 RoundExperience 列表

        Returns:
            合并后的经验文本，LLM 失败返回空字符串
        """
        if not self._llm:
            return ""

        # 构建 prompt
        exp_texts = "\n".join(f"- [{e.tool_name or 'general'}] {e.content}" for e in experiences)
        prompt = f"""你是经验合并助手。以下是同一类别（{category}）的多条经验记录，请合并成简洁的教训。

类别说明:
- tool_error: 工具硬错误或直接失败
- tool_misuse: 工具角色误用、流程误用、软失败
- tool_limitation: 工具能力边界、观测盲区、规避方案
- pattern: 有效工作模式、pipeline、workflow

经验记录:
{exp_texts}

请输出合并后的教训（不超过 200 字），保留最有价值的信息，去除重复内容。
直接输出文本，不要 JSON 或其他格式。"""

        try:
            try:
                response = await self._llm(prompt)
                text = str(response).strip() if response else ""
            except Exception:
                # 兼容 llm.run() 接口
                response = await self._llm.run(
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.choices[0].message.content or "").strip()
            return text[:500]
        except Exception as e:
            logger.warning(f"_llm_merge_experiences failed: {e}")
            return ""

    async def _write_experience_to_ltm(self, trace: "ExperienceTrace") -> bool:
        """写入 ExperienceTrace 到 SQLite experience_traces 表。"""
        if not self._db:
            logger.debug("_write_experience_to_ltm: no DB available")
            return False

        try:
            # 确保 session 存在（FK 约束）
            existing = await self._db.query_one(
                "SELECT id FROM sessions WHERE id = ?", (trace.session_id,)
            )
            if not existing:
                fallback_user_id = "default"
                fallback_project_id = trace.session_id
                if self._job_mgr and self._job_mgr._active_job:
                    active_job = self._job_mgr._active_job
                    active_project_id = getattr(active_job, "project_id", "") or ""
                    if trace.session_id == active_project_id:
                        fallback_user_id = getattr(active_job, "user_id", "") or fallback_user_id
                        fallback_project_id = active_project_id or fallback_project_id
                await self._db.insert("sessions", {
                    "id": trace.session_id,
                    "user_id": fallback_user_id,
                    "project_id": fallback_project_id,
                    "status": "active",
                    "created_at": trace.created_at,
                })

            await self._db.insert("experience_traces", trace.to_dict())
            logger.debug(f"Wrote ExperienceTrace to LTM: {trace.id}")
            return True
        except Exception as e:
            logger.warning(f"_write_experience_to_ltm failed: {e}")
            return False

    async def _route_and_inject_profile(
        self,
        user_id: str,
        user_prompt: str,
        task_intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
    ) -> None:
        """使用 ProfileInjectionRouter 将 LTM UserProfile 智能注入到 WM。

        Stage 15: 实现架构图中的 Job Start 智能路由流程：
        1. 从 LTM 加载 UserProfile (intent-aware)
        2. ProfileInjectionRouter.route() 判断每个维度的注入策略
           - INJECT: 用户未提及，LTM 值直接沿用
           - OVERRIDE: 用户明确冲突，只用新值
           - ENRICH: 用户补充细化，LTM + 新值共存
           - SKIP: 无数据且未提及，不注入
        3. apply() 将决策结果写入 WM TempPreference
        4. 保存中间产物（routing decisions）

        Args:
            user_id: 用户 ID
            user_prompt: 用户第一轮 prompt
        """
        from .inject.profile_injection_router import ProfileInjectionRouter, ProfileInjectionRoutingError
        from .core.models import classify_intent_by_keywords

        logger.info(
            "_route_and_inject_profile called with user_id='%s', task_intent='%s', "
            "read_intent='%s', write_intent='%s', core_persona='%s'",
            user_id,
            task_intent,
            read_intent,
            write_intent,
            core_persona,
        )

        # 推断 intent（如果未提供）
        if not task_intent:
            if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.intent:
                task_intent = self._job_mgr._active_job.intent
            else:
                task_intent = classify_intent_by_keywords(user_prompt)
        if not read_intent:
            if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.read_intent:
                read_intent = self._job_mgr._active_job.read_intent
            else:
                read_intent = task_intent
        if not write_intent:
            if self._job_mgr and self._job_mgr._active_job and self._job_mgr._active_job.write_intent:
                write_intent = self._job_mgr._active_job.write_intent
            else:
                write_intent = task_intent or read_intent

        try:
            _old_debug_path = os.getenv("MEMSLIDES_PROFILE_ROUTING_DEBUG_PATH")
            if self._artifact_dumper:
                try:
                    debug_path = (
                        self._artifact_dumper._memory_dir
                        / "rounds"
                        / "round_001"
                        / "profile_routing"
                        / "profile_routing_llm_response_debug.json"
                    )
                    os.environ["MEMSLIDES_PROFILE_ROUTING_DEBUG_PATH"] = str(debug_path)
                except Exception:
                    logger.debug("Failed to configure profile routing debug artifact path", exc_info=True)
            # 1. 从共享用户状态加载当前可读画像（core persona + current intent）
            profile = None
            if self._user_state_resolver is not None:
                snapshot = await self._user_state_resolver.build_snapshot(
                    user_id=user_id,
                    task_intent=task_intent,
                    read_intent=read_intent,
                    write_intent=write_intent,
                    core_persona=core_persona,
                    working_memory=None,
                    include_cross_intent=False,
                )
                profile = self._compose_profile_from_snapshot(snapshot)
                if not core_persona:
                    core_persona = getattr(snapshot, "core_persona", "") or core_persona

            if profile is None:
                profile = await self._profile_store.get(user_id, intent=read_intent)
            profile_has_content = self._profile_has_meaningful_content(profile)
            logger.info(
                "ProfileInjectionRouter: loaded %s profile for %s (read_intent=%s, task_intent=%s, write_intent=%s)",
                "populated" if profile_has_content else "EMPTY",
                user_id,
                read_intent,
                task_intent,
                write_intent,
            )

            # 2. 路由决策
            router = ProfileInjectionRouter(
                llm=self._profile_injection_llm or self._llm
            )
            decisions = await router.route(
                user_prompt,
                profile,
                target_intent=read_intent,
                task_intent=task_intent,
                write_intent=write_intent,
                core_persona=core_persona,
            )

            logger.info(
                f"ProfileInjectionRouter decisions:\n{ProfileInjectionRouter.format_decisions_for_log(decisions)}"
            )

            # 3. 应用到 WM
            wm = self._job_mgr.working_memory if self._job_mgr else None
            if wm:
                result_summary = await router.apply(decisions, wm, profile, llm=self._llm)
                logger.info(f"ProfileInjectionRouter applied: {result_summary}")
                persist_live_wm_snapshot_from_orchestrator(self, reason="profile_routing")

            # 4. 保存中间产物
            if self._artifact_dumper and wm:
                self._artifact_dumper.dump_profile_routing(
                    round_index=0,
                    user_prompt=user_prompt,
                    intent=read_intent,
                    ltm_profile=profile,
                    profile_load_status={
                        "user_id": user_id,
                        "read_intent": read_intent,
                        "task_intent": task_intent,
                        "write_intent": write_intent,
                        "has_content": profile_has_content,
                        "source": "user_state_resolver" if self._user_state_resolver else "profile_store_direct",
                    },
                    decisions=decisions,
                    wm_prefs_after=wm._temp_preferences,
                    summary=result_summary,
                )

        except Exception as e:
            self._dump_profile_routing_error(
                error=e,
                user_id=user_id,
                task_intent=task_intent,
                read_intent=read_intent,
                write_intent=write_intent,
            )
            logger.error("ProfileInjectionRouter failed: %s", e, exc_info=True)
            if isinstance(e, ProfileInjectionRoutingError):
                raise
            raise ProfileInjectionRoutingError(str(e)) from e
        finally:
            if "_old_debug_path" in locals():
                if _old_debug_path is None:
                    os.environ.pop("MEMSLIDES_PROFILE_ROUTING_DEBUG_PATH", None)
                else:
                    os.environ["MEMSLIDES_PROFILE_ROUTING_DEBUG_PATH"] = _old_debug_path

    async def _preload_experiences_from_ltm(
        self,
        user_id: str,
        project_id: str,
        user_prompt: str = "",
    ) -> None:
        """Job 开始时：从 LTM 预加载相关经验到 WM。

        检索策略:
        1. 查询全局共享的较大高置信度候选池
        2. 若有 user_prompt 且 embedding_func 可用，则做相似度重排
        3. 跳过 chain / agent_run 等独立路径
        4. 按类别配额选择 preload 集合
        5. 转换为 RoundExperience(source="preload") 加入 WM

        Args:
            user_id: 用户 ID（保留签名兼容，不再用于 preload 过滤）
            project_id: 项目 ID
        """
        wm = self._job_mgr.working_memory if self._job_mgr else None
        if not wm or not self._db:
            logger.debug("_preload_experiences_from_ltm: no WM or DB available")
            return

        try:
            # 查询全局共享的高置信度经验候选池（tool_error 在 SQL 层仍有轻微优先级）
            rows = await self._db.query(
                """
                SELECT et.* FROM experience_traces et
                WHERE COALESCE(et.status, 'active') = 'active'
                AND et.confidence >= 0.7
                ORDER BY
                    CASE WHEN et.experience_type = 'tool_error' THEN 0 ELSE 1 END,
                    et.reuse_count DESC,
                    et.created_at DESC
                LIMIT ?
                """,
                (PRELOAD_CANDIDATE_LIMIT,),
            )

            if not rows:
                logger.debug("_preload_experiences_from_ltm: no eligible global experiences found")
                return

            ranked_rows = list(rows)
            if user_prompt and self._embedding_func and len(rows) > 1:
                try:
                    import json
                    from .core.embedding import batch_cosine_similarity

                    def _as_list(value: Any) -> list[str]:
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                            except Exception:
                                return [value] if value else []
                            return parsed if isinstance(parsed, list) else [str(parsed)]
                        return value if isinstance(value, list) else []

                    candidate_texts = []
                    for row in rows:
                        tools = _as_list(row.get("tools_used", "[]"))
                        scenarios = _as_list(row.get("applicable_scenarios", "[]"))
                        candidate_texts.append(
                            "\n".join(
                                part for part in [
                                    row.get("task_description", ""),
                                    row.get("lessons_learned", ""),
                                    " ".join(tools),
                                    " ".join(scenarios),
                                ]
                                if part
                            )
                        )

                    embeddings = await self._embedding_func([user_prompt, *candidate_texts])
                    if len(embeddings) == len(candidate_texts) + 1:
                        query_vec = embeddings[0]
                        candidate_matrix = embeddings[1:]
                        similarities = batch_cosine_similarity(query_vec, candidate_matrix)
                        ranked_rows = [
                            row for _, row in sorted(
                                zip(similarities.tolist(), rows, strict=False),
                                key=lambda item: item[0],
                                reverse=True,
                            )
                        ]
                except Exception as e:
                    logger.debug("preload similarity rerank failed, fallback to DB ordering: %s", e)

            selected_rows = self._select_preload_rows(ranked_rows)

            # 转换为 RoundExperience(source="preload") 并加入 WM
            import json
            for row in selected_rows:
                try:
                    # 解析 applicable_scenarios 作为 keywords
                    scenarios = row.get("applicable_scenarios", "[]")
                    if isinstance(scenarios, str):
                        keywords = json.loads(scenarios)
                    else:
                        keywords = scenarios if isinstance(scenarios, list) else []

                    # 解析 tools_used
                    tools_used_raw = row.get("tools_used", "[]")
                    if isinstance(tools_used_raw, str):
                        tools_used = json.loads(tools_used_raw)
                    else:
                        tools_used = tools_used_raw if isinstance(tools_used_raw, list) else []

                    preloaded_exp = RoundExperience(
                        id=row["id"],
                        content=row.get("lessons_learned", ""),
                        tool_name=tools_used[0] if tools_used else "",
                        keywords=keywords if isinstance(keywords, list) else [],
                        category=infer_task_experience_category(
                            row.get("experience_type", ""),
                            keywords,
                            row.get("final_outcome", ""),
                        ),
                        source="preload",
                        source_task_id=row.get("session_id", ""),
                        confidence=row.get("confidence", 0.7),
                        tools_used=tools_used if isinstance(tools_used, list) else [],
                        outcome=row.get("final_outcome", ""),
                    )
                    wm.add_experience(preloaded_exp)
                except Exception as e:
                    logger.debug(f"Failed to convert LTM row to RoundExperience: {e}")
                    continue

            logger.info(f"[MemoryV2] Preloaded {len(selected_rows)} experiences from LTM for user {user_id}")

        except Exception as e:
            logger.warning(f"_preload_experiences_from_ltm failed: {e}")
