"""ArtifactDumper — 记忆管线中间产物转储器

通过 config 开关控制启停，将 Round/Operation/Job 各阶段的关键中间数据
序列化为 JSON 文件写入 .memory/ 目录，供人工审查和自动化测试断言使用。

目录结构:
  .memory/
  ├── rounds/round_{index}/
  │   ├── round_meta.json
  │   ├── chain_segmentation.json
  │   ├── preference_extraction.json
  │   ├── round_summary.json
  │   ├── wm_snapshot.json
  │   ├── agent_round_{index}.json
  │   └── profile_routing/                       ← ProfileInjectionRouter 决策产物（按 round 分目录）
  │       ├── ltm_profile_snapshot.json           ← 从 LTM 加载的 UserProfile 快照
  │       ├── routing_decisions.json              ← 路由决策（inject/skip/override/enrich）
  │       └── wm_after_routing.json               ← 路由后 WM 中的偏好快照
  ├── injection_traces/round_{index}/
  │   ├── system_{agent_role}_turn_{turn}.json   ← Round 级注入 (4 组件)
  │   └── op_{op_index}_{tool_name}.json         ← Operation 级注入 (4 组件)
  │         两者使用统一的 dump_injection_trace 格式，包含:
  │           wm_preferences        — ProfileInjectionRouter 注入的画像 + 当前 Job 偏好
  │           wm_experiences        — remember_lesson 写入的临时经验
  │           ltm_tool_experiences  — 工具链经验（仅 Operation 级别）
  │           wm_round_history       — RoundSummary via JobHistory
  │         Operation 级别通常只有 ltm_tool_experiences 非空，
  │         其余组件已在 System 级别注入。
  └── consolidation/
      ├── tool_chains/merge_{signature}.json
      ├── round_experiences/{category}/cluster_{index}_{trace_id}.json
      ├── preferences/
      │   ├── input_sources.json          ← 偏好提取输入（intent, slide_html_dir, raw_messages, slide_analysis）
      │   ├── dimension_assignment.json
      │   ├── profile_before.json
      │   ├── profile_after.json
      │   └── llm_merges/merge_{dimension}.json
      ├── episodes/archived_episodes.json
      └── consolidation_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...memory.core.models import RoundSummary, ToolChain

logger = logging.getLogger(__name__)


class ArtifactDumper:
    """记忆管线中间产物转储器。

    所有写入操作均 try/except 包裹，异常仅记录 WARNING 日志，
    不向调用方传播。enabled=False 时跳过所有 I/O。
    """

    def __init__(self, memory_dir: Path, enabled: bool = False) -> None:
        self._memory_dir = Path(memory_dir)
        self._enabled = enabled
        # 扫描已有 round 目录，计算 offset 避免重启 session 后覆盖
        self._round_offset = 0
        if enabled:
            try:
                existing: list[int] = []
                for subdir in ("rounds", "injection_traces"):
                    scan_dir = self._memory_dir / subdir
                    if scan_dir.exists():
                        for d in scan_dir.iterdir():
                            if d.is_dir() and (d.name.startswith("round_") or d.name.startswith("task_")):
                                parts = d.name.split("_", 1)
                                if len(parts) == 2 and parts[1].isdigit():
                                    existing.append(int(parts[1]))
                if existing:
                    self._round_offset = max(existing)
            except Exception:
                pass


    # ── 内部工具 ──

    def _resolve_round_index(self, round_index: int) -> int:
        """将 Job 内的 round_index 转换为全局唯一编号（避免跨 session 覆盖）。"""
        return round_index + self._round_offset

    def _resolve_task_index(self, task_index: int) -> int:
        """Deprecated compatibility wrapper for ``_resolve_round_index``."""
        return self._resolve_round_index(task_index)

    def _write_json(self, path: Path, data: dict) -> None:
        """安全写入 JSON 文件。异常仅记录日志。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = self._safe_serialize(data)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(serialized, ensure_ascii=False, indent=4))
        except Exception as e:
            logger.warning("ArtifactDumper: failed to write %s: %s", path, e)

    @staticmethod
    def _safe_serialize(obj: Any) -> Any:
        """递归安全序列化，不可序列化对象降级为 str(obj)。"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [ArtifactDumper._safe_serialize(item) for item in obj]
        if isinstance(obj, set):
            return [ArtifactDumper._safe_serialize(item) for item in sorted(obj, key=str)]
        if isinstance(obj, dict):
            return {str(k): ArtifactDumper._safe_serialize(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            try:
                return ArtifactDumper._safe_serialize(obj.to_dict())
            except Exception:
                pass
        if hasattr(obj, "model_dump"):
            try:
                return ArtifactDumper._safe_serialize(obj.model_dump())
            except Exception:
                pass
        if callable(obj) and not isinstance(obj, type):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                str(k): ArtifactDumper._safe_serialize(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        return str(obj)

    # ── Round 级 ──

    def dump_round_meta(
        self,
        round_index: int,
        *,
        round_id: str = "",
        user_message: str = "",
        agent_response: str = "",
        tool_call_count: int = 0,
        total_duration_ms: int = 0,
        start_time: str = "",
        end_time: str = "",
        memory_pipeline_stats: dict | None = None,
    ) -> None:
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            data = {
                "round_id": round_id,
                "round_index": round_index,
                "user_message": user_message,
                "agent_response": agent_response,
                "tool_call_count": tool_call_count,
                "total_duration_ms": total_duration_ms,
                "start_time": start_time,
                "end_time": end_time,
                "memory_pipeline_stats": memory_pipeline_stats or {},
            }
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "round_meta.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump round_meta: %s", e)

    def dump_task_meta(
        self,
        task_index: int,
        *,
        task_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Deprecated compatibility wrapper for ``dump_round_meta``."""
        self.dump_round_meta(task_index, round_id=task_id, **kwargs)

    def dump_round_chain_segmentation(
        self,
        round_index: int,
        *,
        chains: list[ToolChain] | list[dict] | None = None,
        fallback: bool = False,
    ) -> None:
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            chains_data = []
            for c in (chains or []):
                if hasattr(c, "to_dict"):
                    chains_data.append(self._safe_serialize(c.to_dict()))
                else:
                    chains_data.append(self._safe_serialize(c))
            data = {
                "fallback": fallback,
                "chains": chains_data,
            }
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "chain_segmentation.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump chain_segmentation: %s", e)

    def dump_chain_segmentation(self, task_index: int, **kwargs: Any) -> None:
        """Deprecated compatibility wrapper for ``dump_round_chain_segmentation``."""
        self.dump_round_chain_segmentation(task_index, **kwargs)

    def dump_round_rich_traces(
        self,
        round_index: int,
        *,
        traces: list[dict] | None = None,
    ) -> None:
        """转储完整执行轨迹（reasoning + tool_call + observation）。

        替代旧版 compressed_segments.json，保留更多语义信息。
        """
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            data = {
                "round_index": round_index,
                "traces": traces or [],
                "total_traces": len(traces) if traces else 0,
            }
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "rich_traces.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump rich_traces: %s", e)

    def dump_rich_traces(self, task_index: int, **kwargs: Any) -> None:
        """Deprecated compatibility wrapper for ``dump_round_rich_traces``."""
        self.dump_round_rich_traces(task_index, **kwargs)

    def dump_preference_extraction(
        self,
        round_index: int,
        *,
        text_extraction: dict | None = None,
        wm_state_after: list[dict] | None = None,
    ) -> None:
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            data = {
                "text_extraction": text_extraction or {"raw_output": "", "parsed_preferences": []},
                "wm_state_after": wm_state_after or [],
            }
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "preference_extraction.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump preference_extraction: %s", e)

    def dump_round_summary(
        self,
        round_index: int,
        *,
        summary: RoundSummary | dict | None = None,
    ) -> None:
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            if summary is None:
                data = {
                    "round_id": "",
                    "round_index": round_index,
                    "summary": "",
                    "slides_modified": [],
                    "key_actions": [],
                    "outcome": "",
                    "unresolved_issues": [],
                }
            elif hasattr(summary, "to_dict"):
                data = self._safe_serialize(summary.to_dict())
            else:
                data = self._safe_serialize(summary)
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "round_summary.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump round_summary: %s", e)

    def dump_task_summary(self, task_index: int, **kwargs: Any) -> None:
        """Deprecated compatibility wrapper for ``dump_round_summary``."""
        self.dump_round_summary(task_index, **kwargs)


    def dump_wm_snapshot(
        self,
        round_index: int,
        *,
        chain_buffer: Any = None,
        temp_preferences: list | None = None,
        temp_experiences: list | None = None,
        temp_episodes: list | None = None,
    ) -> None:
        if not self._enabled:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            # Serialize chain_buffer internals
            cb_data: dict[str, Any] = {"chains": {}, "experiences": {}}
            ltm_cache: dict[str, Any] = {}
            ltm_queried: list[str] = []

            if chain_buffer is not None:
                # chains
                if hasattr(chain_buffer, "get_all_chains"):
                    for sig, chains in chain_buffer.get_all_chains().items():
                        cb_data["chains"][sig] = [self._safe_serialize(c) for c in chains]
                # experiences
                if hasattr(chain_buffer, "get_all_experiences"):
                    for sig, exps in chain_buffer.get_all_experiences().items():
                        cb_data["experiences"][sig] = [self._safe_serialize(e) for e in exps]
                # ltm cache
                if hasattr(chain_buffer, "_ltm_cache"):
                    for tool, exps in chain_buffer._ltm_cache.items():
                        ltm_cache[tool] = [self._safe_serialize(e) for e in exps]
                # ltm queried tools
                if hasattr(chain_buffer, "_ltm_queried_tools"):
                    ltm_queried = sorted(chain_buffer._ltm_queried_tools)

            data = {
                "chain_buffer": cb_data,
                "temp_preferences": [self._safe_serialize(p) for p in (temp_preferences or [])],
                "temp_experiences": [self._safe_serialize(e) for e in (temp_experiences or [])],
                "temp_episodes": [self._safe_serialize(e) for e in (temp_episodes or [])],
                "ltm_cache": ltm_cache,
                "ltm_queried_tools": ltm_queried,
            }
            path = self._memory_dir / f"rounds/round_{round_index:03d}" / "wm_snapshot.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump wm_snapshot: %s", e)


    # ── Job 级 ──

    def dump_chain_merge(
        self,
        signature: str,
        *,
        raw_chains_count: int = 0,
        raw_chains_evicted: int = 0,
        experiences_before: int = 0,
        experiences_after: int = 0,
        new_experiences: list[dict] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            data = {
                "signature": signature,
                "raw_chains_count": raw_chains_count,
                "raw_chains_evicted": raw_chains_evicted,
                "experiences_before": experiences_before,
                "experiences_after": experiences_after,
                "new_experiences": new_experiences or [],
            }
            safe_sig = signature.replace("/", "_").replace("\\", "_")
            path = self._memory_dir / f"consolidation/tool_chains/merge_{safe_sig}.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump chain_merge: %s", e)

    def dump_round_experience_consolidation(
        self,
        category: str,
        trace_id: str,
        *,
        cluster_index: int = 0,
        cluster_total: int = 0,
        session_id: str = "",
        input_experiences: list[Any] | None = None,
        neighbors: list[dict] | None = None,
        decision: dict | None = None,
        selected_target_before: dict | None = None,
        output_trace: Any = None,
        superseded_target_id: str = "",
        ltm_write_succeeded: bool | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            safe_category = (category or "generic").replace("/", "_").replace("\\", "_")
            safe_trace_id = (trace_id or "pending").replace("/", "_").replace("\\", "_")
            safe_trace_id = safe_trace_id[:32]
            relation = str((decision or {}).get("relation", "")).strip().lower()
            data = {
                "category": category,
                "trace_id": trace_id,
                "cluster_index": cluster_index,
                "cluster_total": cluster_total,
                "session_id": session_id,
                "cluster_size": len(input_experiences or []),
                "neighbor_count": len(neighbors or []),
                "relation": relation,
                "decision": self._safe_serialize(decision or {}),
                "input_experiences": [
                    self._safe_serialize(exp) for exp in (input_experiences or [])
                ],
                "neighbors": [
                    self._safe_serialize(row) for row in (neighbors or [])
                ],
                "selected_target_before": self._safe_serialize(selected_target_before)
                if selected_target_before else None,
                "output_trace": self._safe_serialize(output_trace),
                "superseded_target_id": superseded_target_id,
                "ltm_write_succeeded": ltm_write_succeeded,
            }
            path = (
                self._memory_dir
                / "consolidation"
                / "round_experiences"
                / safe_category
                / f"cluster_{cluster_index:03d}_{safe_trace_id}.json"
            )
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump round_experience_consolidation: %s", e)

    def dump_task_experience_consolidation(self, *args: Any, **kwargs: Any) -> None:
        """Deprecated compatibility wrapper for ``dump_round_experience_consolidation``."""
        self.dump_round_experience_consolidation(*args, **kwargs)

    def dump_preference_consolidation(
        self,
        *,
        dimension_assignment: list[dict] | None = None,
        profile_before: dict | None = None,
        profile_after: dict | None = None,
        llm_merges: dict[str, dict] | None = None,
        intent: str = "",
        slide_html_dir: str = "",
        raw_messages: str = "",
        slide_analysis: str = "",
    ) -> None:
        if not self._enabled:
            return
        try:
            pref_dir = self._memory_dir / "consolidation" / "preferences"

            # input_sources.json — 记录偏好提取的输入数据
            self._write_json(
                pref_dir / "input_sources.json",
                {
                    "intent": intent,
                    "slide_html_dir": slide_html_dir,
                    "raw_messages": raw_messages or "",
                    "slide_analysis": slide_analysis[:3000] if slide_analysis else "",
                },
            )
            # dimension_assignment.json
            self._write_json(
                pref_dir / "dimension_assignment.json",
                {"assignments": dimension_assignment or []},
            )
            # profile_before.json
            self._write_json(
                pref_dir / "profile_before.json",
                profile_before or {},
            )
            # profile_after.json
            self._write_json(
                pref_dir / "profile_after.json",
                profile_after or {},
            )
            # llm_merges per dimension
            for dim, merge_data in (llm_merges or {}).items():
                safe_dim = dim.replace("/", "_").replace("\\", "_")
                self._write_json(
                    pref_dir / "llm_merges" / f"merge_{safe_dim}.json",
                    merge_data,
                )
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump preference_consolidation: %s", e)

    def dump_episode_archive(
        self,
        *,
        episodes: list[dict] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            data = {"episodes": episodes or []}
            path = self._memory_dir / "consolidation" / "episodes" / "archived_episodes.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump episode_archive: %s", e)

    def dump_consolidation_summary(
        self,
        *,
        job_id: str = "",
        round_count: int = 0,
        task_count: int | None = None,
        chains_processed: int = 0,
        preferences_archived: int = 0,
        episodes_archived: int = 0,
        total_duration_ms: int = 0,
    ) -> None:
        if not self._enabled:
            return
        try:
            if task_count is not None and not round_count:
                round_count = task_count
            data = {
                "job_id": job_id,
                "round_count": round_count,
                "chains_processed": chains_processed,
                "preferences_archived": preferences_archived,
                "episodes_archived": episodes_archived,
                "total_duration_ms": total_duration_ms,
            }
            path = self._memory_dir / "consolidation" / "consolidation_summary.json"
            self._write_json(path, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump consolidation_summary: %s", e)

    # ── ProfileInjectionRouter 产物 ──

    def dump_profile_routing(
        self,
        round_index: int | None = None,
        *,
        task_index: int | None = None,
        user_prompt: str = "",
        intent: str = "",
        ltm_profile: Any = None,
        decisions: list[Any] | None = None,
        wm_prefs_after: list[Any] | None = None,
        summary: dict[str, str] | None = None,
        profile_load_status: dict[str, Any] | None = None,
    ) -> None:
        """转储 ProfileInjectionRouter 的完整决策过程。

        产物目录: .memory/rounds/round_NNN/profile_routing/
        - ltm_profile_snapshot.json: 从 LTM 加载的 UserProfile 快照
        - profile_load_status.json: 当前 read_intent 命中的画像是否为空/有内容
        - routing_decisions.json: 路由决策详情（每个维度的 action/reason）
        - wm_after_routing.json: 路由后 WM 中的偏好快照

        Args:
            round_index: 当前 round 序号（用于按轮次分目录）
            user_prompt: 用户原始指令
            intent: 意图分类结果
            ltm_profile: 从 LTM 加载的 UserProfile 对象
            decisions: ProfileInjectionRouter 的决策列表
            wm_prefs_after: 路由后 WM 中的 TempPreference 列表
            summary: 路由摘要统计
        """
        if not self._enabled:
            return
        try:
            from datetime import datetime
            if round_index is None:
                round_index = task_index or 0
            round_index = self._resolve_round_index(round_index)
            routing_dir = self._memory_dir / f"rounds/round_{round_index:03d}" / "profile_routing"
            routing_dir.mkdir(parents=True, exist_ok=True)

            # 1. LTM Profile 快照
            if ltm_profile is not None:
                profile_data = {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": getattr(ltm_profile, "user_id", ""),
                    "version": getattr(ltm_profile, "version", 1),
                    "last_updated": getattr(ltm_profile, "last_updated", ""),
                    "dimensions": {},
                }
                for dim_name in ["theme", "visual", "layout", "content", "template", "general"]:
                    dim_obj = getattr(ltm_profile, dim_name, None)
                    if dim_obj:
                        # 提取维度的所有字段
                        dim_dict = {}
                        for field_name in dir(dim_obj):
                            if not field_name.startswith("_") and field_name not in ("keywords", "notes"):
                                val = getattr(dim_obj, field_name, None)
                                if val is not None and not callable(val):
                                    dim_dict[field_name] = val
                        if dim_dict:
                            profile_data["dimensions"][dim_name] = dim_dict
                self._write_json(routing_dir / "ltm_profile_snapshot.json", profile_data)

            if profile_load_status is not None:
                self._write_json(routing_dir / "profile_load_status.json", profile_load_status)

            # 2. 路由决策
            if decisions is not None:
                decisions_data = {
                    "timestamp": datetime.now().isoformat(),
                    "user_prompt": user_prompt[:500] if user_prompt else "",
                    "intent": intent,
                    "decisions": [],
                    "summary": summary or {},
                }
                for d in decisions:
                    decisions_data["decisions"].append({
                        "dimension": getattr(d, "dimension", ""),
                        "action": getattr(d, "action", ""),
                        "reason": getattr(d, "reason", ""),
                        "override_content": getattr(d, "override_content", ""),
                    })
                # 按 action 分组统计
                action_counts: dict[str, int] = {}
                for d in decisions:
                    action = getattr(d, "action", "unknown")
                    action_counts[action] = action_counts.get(action, 0) + 1
                decisions_data["action_counts"] = action_counts
                self._write_json(routing_dir / "routing_decisions.json", decisions_data)

            # 3. 路由后 WM 偏好快照
            if wm_prefs_after is not None:
                wm_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_prefs": len(wm_prefs_after),
                    "by_dimension": {},
                    "preferences": [],
                }
                # 按一级维度分组
                for p in wm_prefs_after:
                    dim = getattr(p, "dimension", "") or "general"
                    top_level = dim.split(".")[0].capitalize() if "." in dim else dim.capitalize()
                    if top_level not in wm_data["by_dimension"]:
                        wm_data["by_dimension"][top_level] = []
                    wm_data["by_dimension"][top_level].append({
                        "content": getattr(p, "content", ""),
                        "dimension": dim,
                        "source_round_id": getattr(p, "source_round_id", "")
                        or getattr(p, "source_task_id", ""),
                        "superseded": getattr(p, "superseded", False),
                    })
                    wm_data["preferences"].append({
                        "content": getattr(p, "content", ""),
                        "dimension": dim,
                        "source_round_id": getattr(p, "source_round_id", "")
                        or getattr(p, "source_task_id", ""),
                    })
                self._write_json(routing_dir / "wm_after_routing.json", wm_data)

        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump profile_routing: %s", e)

    def dump_dimension_matching(
        self,
        round_index: int | None = None,
        *,
        task_index: int | None = None,
        user_message: str = "",
        matched_dims: set[str] | list[str] | None = None,
        match_method: str = "",
        profile_dims_with_data: list[str] | None = None,
        fallback_reason: str = "",
    ) -> None:
        """转储维度匹配过程（DimensionMatcher）。

        产物路径: .memory/rounds/round_{index}/dimension_matching.json

        Args:
            round_index: Round 索引
            user_message: 用户消息（用于匹配）
            matched_dims: 匹配到的维度集合
            match_method: 匹配方法（"vector_top_k" / "vector_single" / "keyword" / "fallback_all"）
            profile_dims_with_data: Profile 中有数据的维度列表
            fallback_reason: 降级原因（如果有）
        """
        if not self._enabled:
            return
        if round_index is None:
            round_index = task_index or 0
        round_index = self._resolve_round_index(round_index)
        try:
            from datetime import datetime
            round_dir = self._memory_dir / f"rounds/round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message or "",
                "match_method": match_method,
                "matched_dimensions": list(matched_dims) if matched_dims else None,
                "is_full_injection": matched_dims is None,
                "profile_dims_with_data": profile_dims_with_data or [],
                "fallback_reason": fallback_reason,
            }
            self._write_json(round_dir / "dimension_matching.json", data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump dimension_matching: %s", e)

    def dump_profile_execution_contract(
        self,
        round_index: int | None = None,
        *,
        task_index: int | None = None,
        contract: dict[str, Any] | None = None,
    ) -> None:
        """转储 runtime profile execution contract。

        产物路径:
        .memory/rounds/round_NNN/profile_routing/profile_execution_contract.json
        """
        if not self._enabled or not contract:
            return
        try:
            if round_index is None:
                round_index = task_index or 0
            round_index = self._resolve_round_index(round_index)
            routing_dir = self._memory_dir / f"rounds/round_{round_index:03d}" / "profile_routing"
            routing_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(routing_dir / "profile_execution_contract.json", contract)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump profile_execution_contract: %s", e)

    def dump_profile_execution_plan(
        self,
        round_index: int | None = None,
        *,
        task_index: int | None = None,
        plan: dict[str, Any] | None = None,
    ) -> None:
        """转储 runtime profile execution plan。

        产物路径:
        .memory/rounds/round_NNN/profile_routing/profile_execution_plan.json
        """
        if not self._enabled or not plan:
            return
        try:
            if round_index is None:
                round_index = task_index or 0
            round_index = self._resolve_round_index(round_index)
            routing_dir = self._memory_dir / f"rounds/round_{round_index:03d}" / "profile_routing"
            routing_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(routing_dir / "profile_execution_plan.json", plan)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump profile_execution_plan: %s", e)

    def dump_profile_realization_report(
        self,
        round_index: int | None = None,
        *,
        task_index: int | None = None,
        report: dict[str, Any] | None = None,
    ) -> None:
        """转储 persona realization report。

        产物路径:
        .memory/rounds/round_NNN/profile_routing/profile_realization_report.json
        """
        if not self._enabled or not report:
            return
        try:
            if round_index is None:
                round_index = task_index or 0
            round_index = self._resolve_round_index(round_index)
            routing_dir = self._memory_dir / f"rounds/round_{round_index:03d}" / "profile_routing"
            routing_dir.mkdir(parents=True, exist_ok=True)
            report_payload = dict(report)
            audit = report_payload.pop("profile_realization_audit", None)
            audit_markdown = str(report_payload.pop("profile_realization_audit_markdown", "") or "")
            self._write_json(routing_dir / "profile_realization_report.json", report_payload)
            if isinstance(audit, dict) and audit:
                self._write_json(routing_dir / "profile_realization_audit.json", audit)
                if audit_markdown:
                    try:
                        (routing_dir / "profile_realization_audit.md").write_text(
                            audit_markdown,
                            encoding="utf-8",
                        )
                    except Exception as write_exc:
                        logger.warning(
                            "ArtifactDumper: failed to write profile_realization_audit.md: %s",
                            write_exc,
                        )
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump profile_realization_report: %s", e)

    # ── 记忆注入产物（替代旧的 _save_memory_injection） ──

    # dump_memory_injection 已删除，由 dump_injection_trace 替代

    # ── AgentRound 产物（合并自 RoundArtifactWriter） ──

    def dump_agent_round(
        self,
        round_index: int,
        *,
        agent_round: Any = None,
        raw_tool_calls: list[dict] | None = None,
    ) -> None:
        """转储 AgentRound 完整数据（合并自 RoundArtifactWriter）。

        将原 round_XXX/ 目录的产物统一写入 round_XXX/ 目录：
        - agent_round.json
        - raw_tool_calls.json
        - compressed_segments.json
        - memory_injection.txt
        - agent_reasoning.txt
        """
        if not self._enabled or agent_round is None:
            return
        round_index = self._resolve_round_index(round_index)
        try:
            from datetime import datetime

            round_dir = self._memory_dir / f"rounds/round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            segments = getattr(agent_round, 'segments', [])
            memory_injection_full = getattr(agent_round, 'memory_injection_full', '')
            agent_reasoning = getattr(agent_round, 'agent_reasoning', '')

            # 1. agent_round.json
            agent_round_data = {
                "round_id": getattr(agent_round, 'round_id', 0),
                "round_index": round_index,
                "agent_name": getattr(agent_round, 'agent_name', ''),
                "session_id": getattr(agent_round, 'session_id', ''),
                "user_id": getattr(agent_round, 'user_id', ''),
                "timestamp": getattr(agent_round, 'timestamp', datetime.now().isoformat()),
                "input": {
                    "user_message": getattr(agent_round, 'user_message', ''),
                    "memory_injection_tokens": getattr(agent_round, 'memory_injection_tokens', 0),
                },
                "execution": {
                    "raw_tool_calls_count": len(raw_tool_calls) if raw_tool_calls else 0,
                    "compressed_segments_count": len(segments),
                    "duration_seconds": getattr(agent_round, 'duration_seconds', 0.0),
                },
                "output": {
                    "error_count": getattr(agent_round, 'error_count', 0),
                },
                "metadata": {
                    "total_tool_calls": getattr(agent_round, 'total_tool_calls', len(segments)),
                    "tools_used": list({getattr(s, 'tool_name', '') for s in segments}),
                    "tools_failed": [
                        getattr(s, 'tool_name', '')
                        for s in segments
                        if getattr(s, 'is_error', False)
                    ],
                },
            }
            self._write_json(round_dir / "agent_round.json", agent_round_data)

            # 2. raw_tool_calls.json
            if raw_tool_calls:
                total_chars = sum(len(str(tc.get("result", ""))) for tc in raw_tool_calls)
                raw_data = {
                    "round_index": round_index,
                    "tool_calls": raw_tool_calls,
                    "total_tool_calls": len(raw_tool_calls),
                    "total_result_chars": total_chars,
                    "estimated_tokens": total_chars // 4,
                }
                self._write_json(round_dir / "raw_tool_calls.json", raw_data)

            # 3. compressed_segments.json
            segments_data = [
                {
                    "tool_name": getattr(s, 'tool_name', ''),
                    "target_file": getattr(s, 'target_file', ''),
                    "action_summary": getattr(s, 'action_summary', ''),
                    "is_error": getattr(s, 'is_error', False),
                    "error_msg": getattr(s, 'error_msg', ''),
                }
                for s in segments
            ]
            compressed_data = {
                "round_index": round_index,
                "segments": segments_data,
                "total_segments": len(segments_data),
            }
            self._write_json(round_dir / "compressed_segments.json", compressed_data)

            # 4. memory_injection.txt (文本格式保留，便于阅读)
            if memory_injection_full:
                tokens = getattr(agent_round, 'memory_injection_tokens', 0)
                text = f"""=== Memory Context Injected at Round {round_index} ===
Timestamp: {datetime.now().isoformat()}
Tokens: {tokens}

{memory_injection_full}
"""
                try:
                    (round_dir / "memory_injection.txt").write_text(text, encoding="utf-8")
                except Exception as e:
                    logger.warning("ArtifactDumper: failed to write memory_injection.txt: %s", e)

            # 5. agent_reasoning.txt
            if agent_reasoning:
                agent_name = getattr(agent_round, 'agent_name', '')
                text = f"""=== Agent Reasoning at Round {round_index} ===
Agent: {agent_name}
Timestamp: {datetime.now().isoformat()}

{agent_reasoning}
"""
                try:
                    (round_dir / "agent_reasoning.txt").write_text(text, encoding="utf-8")
                except Exception as e:
                    logger.warning("ArtifactDumper: failed to write agent_reasoning.txt: %s", e)

            logger.debug("ArtifactDumper: wrote agent_round for round %d", round_index)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump agent_round: %s", e)

    def dump_injection_trace(
        self,
        round_index: int,
        *,
        level: str = "system",
        agent_role: str = "",
        turn: int = 0,
        # ── 4 个独立组件（完整内容，不截断） ──
        wm_preferences: str = "",
        profile_execution_contract: str = "",
        profile_execution_plan: str = "",
        wm_experiences: str = "",
        ltm_tool_experiences: str = "",
        wm_round_history: str = "",
        wm_task_history: str = "",
        # ── 最终注入 ──
        final_injected_text: str = "",
        injection_target: str = "SYSTEM",
        skipped_reason: str = "",
        template_prompt: str = "",
        # ── Operation 级别额外字段 ──
        op_index: int = 0,
        trigger_tools: list | None = None,
    ) -> None:
        """统一转储记忆注入 trace（完整记录所有组件）。

        6 个组件（AtomicPreference 不注入，仅用于更新 UserProfile）：
        - wm_preferences:       WM 临时偏好（ProfileInjectionRouter 注入的画像 + 当前 Job 偏好）
        - profile_execution_contract: 仅 Design 阶段注入的页级画像执行合同
        - profile_execution_plan: 仅 Design 阶段注入的逐页 persona 执行计划
        - wm_experiences:       WM 临时经验（remember_lesson 写入）
        - ltm_tool_experiences: LTM 工具链经验（仅 Operation 级别）
        - wm_round_history:      WM 任务历史摘要（RoundSummary via JobHistory）

        每个组件独立记录完整内容 + 字符数 + token 估算，
        便于追溯哪些记忆被注入、哪些为空。

        Args:
            round_index: Round 索引
            level: "system" (SYSTEM prompt 注入) 或 "operation"
            agent_role: Agent 角色 (research/design/modify/template_analysis)
            turn: 修改轮次（Modify 多轮场景）
            wm_preferences: WM 临时偏好（ProfileInjectionRouter 注入的画像 + 当前 Job 偏好）
            profile_execution_contract: 仅 Design 阶段注入的页级画像执行合同
            profile_execution_plan: 仅 Design 阶段注入的逐页 persona 执行计划
            wm_experiences: WM 临时经验（remember_lesson 写入）
            ltm_tool_experiences: LTM 工具链经验（仅 Operation 级别）
            wm_round_history: WM 任务历史摘要（RoundSummary via JobHistory）
            final_injected_text: 最终注入的完整文本
            injection_target: 注入位置 ("SYSTEM" 或 "USER <memory_hint>")
            skipped_reason: 跳过注入的原因
            template_prompt: 模板附加注入（Research/Design 的 template prompt）
            op_index: Operation 索引（仅 level="operation" 时使用）
            trigger_tools: 触发工具列表（仅 level="operation" 时使用）
        """
        if not self._enabled:
            return
        if wm_task_history and not wm_round_history:
            wm_round_history = wm_task_history
        round_index = self._resolve_round_index(round_index)
        try:
            from datetime import datetime

            def _component(text: str) -> dict:
                return {
                    "chars": len(text),
                    "tokens_estimated": len(text) // 4,
                    "content": text,
                }

            data = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "agent_role": agent_role,
                "turn": turn,
                "injection_target": injection_target,
                "components": {
                    "wm_preferences": _component(wm_preferences),
                    "profile_execution_contract": _component(profile_execution_contract),
                    "profile_execution_plan": _component(profile_execution_plan),
                    "wm_experiences": _component(wm_experiences),
                    "ltm_tool_experiences": _component(ltm_tool_experiences),
                    "wm_round_history": _component(wm_round_history),
                },
                "summary": {
                    "total_components": sum(1 for t in [
                        wm_preferences, profile_execution_contract, profile_execution_plan, wm_experiences,
                        ltm_tool_experiences, wm_round_history,
                    ] if t),
                    "template_injected": bool(template_prompt),
                    "empty_components": [
                        name for name, text in [
                            ("wm_preferences", wm_preferences),
                            ("profile_execution_contract", profile_execution_contract),
                            ("profile_execution_plan", profile_execution_plan),
                            ("wm_experiences", wm_experiences),
                            ("ltm_tool_experiences", ltm_tool_experiences),
                            ("wm_round_history", wm_round_history),
                        ] if not text
                    ],
                },
                "final_injected": {
                    "chars": len(final_injected_text),
                    "tokens_estimated": len(final_injected_text) // 4,
                    "content": final_injected_text,
                },
                "template_prompt": _component(template_prompt),
                "skipped_reason": skipped_reason,
            }

            # Operation 级别额外字段
            if level == "operation":
                data["op_index"] = op_index
                data["trigger_tools"] = trigger_tools or []

            trace_dir = self._memory_dir / f"injection_traces/round_{round_index:03d}"
            if level == "operation":
                filename = f"op_{op_index:03d}_{('_'.join(trigger_tools[:2]) if trigger_tools else 'unknown')}.json"
            else:
                filename = f"system_{agent_role}_turn_{turn:03d}.json"
            self._write_json(trace_dir / filename, data)
        except Exception as e:
            logger.warning("ArtifactDumper: failed to dump injection_trace: %s", e)
