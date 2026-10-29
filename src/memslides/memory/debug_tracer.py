"""Implementation details."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_serialize(obj: Any) -> Any:
    """Implementation details."""
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
    """Implementation details."""

    def __init__(self, debug_dir: Path | str):
        self.debug_dir = Path(debug_dir)
        self.events_file = self.debug_dir / "events.jsonl"
        self.dashboard_file = self.debug_dir / "dashboard.json"
        self.episodes_snapshot_file = self.debug_dir / "episodes_snapshot.json"
        self.memory_evolution_file = self.debug_dir / "memory_evolution.json"

        self._dashboard: dict[str, Any] = {}
        self._memory_evolution: list[dict] = []


        self.debug_dir.mkdir(parents=True, exist_ok=True)



    def log_event(self, event_type: str, data: dict | None = None) -> None:
        """Implementation details."""
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



    def log_step(self, turn: int, step_name: str, data: Any) -> None:
        """Implementation details."""
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




    def update_dashboard(self, section: str, data: dict) -> None:
        """Implementation details."""
        self._dashboard[section] = _safe_serialize(data)
        self._dashboard["last_updated"] = datetime.now().isoformat()
        try:
            self.dashboard_file.write_text(
                json.dumps(self._dashboard, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"DebugTracer.update_dashboard failed: {e}")



    def log_collected_round(self, turn: int, compact_round: Any) -> None:
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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



    def log_extraction_batch(self, batch_id: int, stage: str, data: Any) -> None:
        """Implementation details."""
        if stage == "stored":
            stored_count = data.get("stored_count", 0) if isinstance(data, dict) else 0
            self.log_event("extraction_completed", {
                "batch_id": batch_id,
                "stored_count": stored_count,
            })

    # ── Stage 2: Episode snapshot ──

    async def dump_episodes_snapshot(self, episode_store: Any, user_id: str = "default") -> None:
        """Implementation details."""
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



    def get_recent_events(self, n: int = 20) -> list[dict]:
        """Implementation details."""
        if not self.events_file.exists():
            return []
        try:
            lines = self.events_file.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(line) for line in lines[-n:] if line.strip()]
        except Exception:
            return []

    def get_dashboard(self) -> dict:
        """Implementation details."""
        if not self.dashboard_file.exists():
            return {}
        try:
            return json.loads(self.dashboard_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ── Layer 3: Memory Evolution cumulative view ──

    def append_memory_evolution(self, snapshot: dict) -> None:
        """Implementation details."""
        try:

            if not self._memory_evolution and self.memory_evolution_file.exists():
                try:
                    existing = json.loads(
                        self.memory_evolution_file.read_text(encoding="utf-8")
                    )
                    if isinstance(existing, list):
                        self._memory_evolution = existing
                except Exception:
                    pass


            self._memory_evolution.append(_safe_serialize(snapshot))


            self.memory_evolution_file.write_text(
                json.dumps(self._memory_evolution, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


            self.log_event("memory_evolution_updated", {
                "turn": snapshot.get("turn", 0),
                "total_turns": len(self._memory_evolution),
            })
        except Exception as e:
            logger.warning(f"DebugTracer.append_memory_evolution failed: {e}")
