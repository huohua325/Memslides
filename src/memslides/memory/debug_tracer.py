"""DebugTracer — 记忆系统数据流可观测性基础设施

Stage 4 之后，活跃的主链路中间产物主要落在 `.memory/rounds/`
与 ArtifactDumper 输出中。`.memory/extractions/` 仅保留给历史
调试/兼容场景；本文件仅维护 `debug/` 下的事件流、dashboard
与诊断快照，不再维护旧版 `debug/extractions/` 目录。

输出文件结构:
    .memory/
    ├── rounds/                   # 主链路 Round 产物
    ├── extractions/              # 历史调试产物（可选）
    └── debug/
        ├── events.jsonl          # 事件流
        ├── dashboard.json        # 系统状态
        ├── episodes_snapshot.json
        └── memory_evolution.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_serialize(obj: Any) -> Any:
    """将任意对象转为可 JSON 序列化的形式"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {str(k): _safe_serialize(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


class DebugTracer:
    """记忆系统数据流追踪器

    使用方式:
        tracer = DebugTracer(workspace / ".memory" / "debug")
        tracer.log_event("session_created", {"session_id": "abc"})
        tracer.log_step(turn=1, step_name="intent", data=intent_result)
        tracer.update_dashboard("session", {"phase": "interactive"})
    """

    def __init__(self, debug_dir: Path | str):
        self.debug_dir = Path(debug_dir)
        self.events_file = self.debug_dir / "events.jsonl"
        self.dashboard_file = self.debug_dir / "dashboard.json"
        self.episodes_snapshot_file = self.debug_dir / "episodes_snapshot.json"
        self.memory_evolution_file = self.debug_dir / "memory_evolution.json"

        self._dashboard: dict[str, Any] = {}
        self._memory_evolution: list[dict] = []

        # 确保目录存在
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ── 事件流 ──

    def log_event(self, event_type: str, data: dict | None = None) -> None:
        """追加一条事件到 events.jsonl"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": _safe_serialize(data) if data else {},
        }
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"DebugTracer.log_event failed: {e}")

    # ── Pipeline 步骤 ──

    def log_step(self, turn: int, step_name: str, data: Any) -> None:
        """记录 pipeline 步骤到 events.jsonl

        Stage 4 重构: 不再写入单独文件，改为追加到 events.jsonl 统一事件流。
        """
        self.log_event(f"step:{step_name}", {
            "turn": turn,
            "step": step_name,
            "data": data,
        })

    def get_step_data(self, turn: int, step_name: str) -> Any | None:
        """Read back a logged pipeline step from events.jsonl.

        Returns the *data* payload for the given (turn, step_name), or None.
        Scans from the end so the most recent write wins.
        """
        if not self.events_file.exists():
            return None
        try:
            target_event = f"step:{step_name}"
            # Read all lines and scan in reverse for the latest match
            lines = self.events_file.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("event") == target_event:
                    d = ev.get("data", {})
                    if d.get("turn") == turn:
                        return d.get("data")
        except Exception as e:
            logger.warning(f"DebugTracer.get_step_data failed: {e}")
        return None


    # ── Dashboard 快照 ──

    def update_dashboard(self, section: str, data: dict) -> None:
        """更新 dashboard.json 的某个区域

        Args:
            section: 区域名（如 session / memory / rules / agent）
            data: 该区域的最新数据
        """
        self._dashboard[section] = _safe_serialize(data)
        self._dashboard["last_updated"] = datetime.now().isoformat()
        try:
            self.dashboard_file.write_text(
                json.dumps(self._dashboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"DebugTracer.update_dashboard failed: {e}")

    # ── Stage 2: Per-turn artifacts (Stage 4: 已迁移到 RoundArtifactWriter) ──

    def log_collected_round(self, turn: int, compact_round: Any) -> None:
        """记录本轮 MemoryCollector 采集结果 → turn_NNN_collected.json

        Stage 4 注意: 此方法已废弃，完整数据已迁移到 rounds/round_NNN/agent_round.json
        保留此方法以兼容旧代码。
        """
        try:
            segments_summary = []
            for seg in getattr(compact_round, "segments", []):
                name = getattr(seg, "name", getattr(seg, "tool_name", "?"))
                result_preview = str(getattr(seg, "result", ""))[:80]
                segments_summary.append(f"{name} → {result_preview}")

            data = {
                "round_id": getattr(compact_round, "round_id", None),
                "user_message": str(getattr(compact_round, "user_message", ""))[:200],
                "segment_count": len(getattr(compact_round, "segments", [])),
                "segments_summary": segments_summary[:10],
                "estimated_tokens": (
                    compact_round.estimate_tokens()
                    if hasattr(compact_round, "estimate_tokens") else 0
                ),
            }
            self.log_step(turn, "collected", data)
            self.log_event("collector_round_end", {
                "turn": turn,
                "round_id": data["round_id"],
                "segment_count": data["segment_count"],
                "tokens": data["estimated_tokens"],
            })
        except Exception as e:
            logger.warning("DebugTracer.log_collected_round failed: %s", e)

    def log_retrieval(self, turn: int, query: str, results: list) -> None:
        """记录 HybridRetriever 检索详情 → turn_NNN_retrieval.json"""
        try:
            merged_items = []
            for r in (results or []):
                merged_items.append({
                    "id": getattr(r, "id", str(r)[:40]),
                    "score": getattr(r, "score", 0.0),
                    "source": getattr(r, "source", ""),
                    "snippet": str(
                        getattr(r, "design_insight", getattr(r, "content", ""))
                    )[:120],
                })

            data = {
                "query": query[:200],
                "rrf_merged": merged_items[:10],
                "total_candidates": len(results) if results else 0,
                "final_count": len(merged_items),
            }
            self.log_step(turn, "retrieval", data)
            self.log_event("cognitive_retrieval", {
                "turn": turn,
                "query_len": len(query),
                "result_count": data["final_count"],
            })
        except Exception as e:
            logger.warning("DebugTracer.log_retrieval failed: %s", e)

    def log_cognitive_injection(self, turn: int, injection_detail: dict) -> None:
        """记录认知记忆注入详情 → turn_NNN_cognitive_injection.json

        Stage 4 注意: 此方法已废弃，完整数据已迁移到 rounds/round_NNN/memory_injection.txt
        保留此方法以兼容旧代码。
        """
        try:
            self.log_step(turn, "cognitive_injection", injection_detail)
            self.log_event("cognitive_injection", {
                "turn": turn,
                "types_injected": injection_detail.get("injected_types", []),
                "total_tokens": injection_detail.get("estimated_tokens", 0),
            })
        except Exception as e:
            logger.warning("DebugTracer.log_cognitive_injection failed: %s", e)

    def log_extraction_trigger(self, turn: int, triggered: bool, reason: dict) -> None:
        """记录是否触发了 Episode 提取 → turn_NNN_extraction_trigger.json"""
        try:
            data = {"triggered": triggered, **reason}
            self.log_step(turn, "extraction_trigger", data)
            if triggered:
                self.log_event("extraction_triggered", {
                    "turn": turn,
                    "reason": reason.get("reason", ""),
                    "batch_size": reason.get("batch_size", 0),
                })
        except Exception as e:
            logger.warning("DebugTracer.log_extraction_trigger failed: %s", e)

    # ── Stage 2: Extraction batch trace (Stage 4: 已迁移到 ExtractionArtifactWriter) ──

    def log_extraction_batch(self, batch_id: int, stage: str, data: Any) -> None:
        """兼容旧调用，不再写入 `debug/extractions/` 文件。"""
        if stage == "stored":
            stored_count = data.get("stored_count", 0) if isinstance(data, dict) else 0
            self.log_event("extraction_completed", {
                "batch_id": batch_id,
                "stored_count": stored_count,
            })

    # ── Stage 2: Episode snapshot ──

    async def dump_episodes_snapshot(self, episode_store: Any, user_id: str = "default") -> None:
        """导出当前 EpisodeStore 到 episodes_snapshot.json"""
        try:
            episodes_data = []
            by_status: dict[str, int] = {}
            by_category: dict[str, int] = {}

            if hasattr(episode_store, "get_active"):
                active = await episode_store.get_active(user_id)
                for ep in active:
                    d = ep.to_dict() if hasattr(ep, "to_dict") else _safe_serialize(ep)
                    episodes_data.append({
                        "id": d.get("id", ""),
                        "created_at": d.get("created_at", ""),
                        "design_insight": str(d.get("design_insight", ""))[:200],
                        "category": d.get("category", ""),
                        "status": d.get("status", "active"),
                    })
                    status = d.get("status", "active")
                    by_status[status] = by_status.get(status, 0) + 1
                    cat = d.get("category", "unknown")
                    by_category[cat] = by_category.get(cat, 0) + 1

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "total_count": len(episodes_data),
                "by_status": by_status,
                "by_category": by_category,
                "episodes": episodes_data,
            }
            self.episodes_snapshot_file.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("DebugTracer.dump_episodes_snapshot failed: %s", e)

    # ── 工具方法 ──

    def get_recent_events(self, n: int = 20) -> list[dict]:
        """读取最近 n 条事件"""
        if not self.events_file.exists():
            return []
        try:
            lines = self.events_file.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(line) for line in lines[-n:] if line.strip()]
        except Exception:
            return []

    def get_dashboard(self) -> dict:
        """读取当前 dashboard 数据"""
        if not self.dashboard_file.exists():
            return {}
        try:
            return json.loads(self.dashboard_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ── Layer 3: Memory Evolution cumulative view ──

    def append_memory_evolution(self, snapshot: dict) -> None:
        """追加一轮的 memory 快照到 memory_evolution.json

        这是一个累积视图，展示从 Turn 1 到当前的所有轮次的 memory 演变。
        每次调用会追加一个新的 turn snapshot 并重写整个文件。

        Args:
            snapshot: 包含 turn、timestamp、user_message、memory_snapshot 的字典
        """
        try:
            # 首次调用：尝试从文件加载已有数据
            if not self._memory_evolution and self.memory_evolution_file.exists():
                try:
                    existing = json.loads(
                        self.memory_evolution_file.read_text(encoding="utf-8")
                    )
                    if isinstance(existing, list):
                        self._memory_evolution = existing
                except Exception:
                    pass

            # 追加新快照
            self._memory_evolution.append(_safe_serialize(snapshot))

            # 写回文件
            self.memory_evolution_file.write_text(
                json.dumps(self._memory_evolution, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 记录事件
            self.log_event("memory_evolution_updated", {
                "turn": snapshot.get("turn", 0),
                "total_turns": len(self._memory_evolution),
            })
        except Exception as e:
            logger.warning(f"DebugTracer.append_memory_evolution failed: {e}")
