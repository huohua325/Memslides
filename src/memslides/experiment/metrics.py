from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InjectionStats:
    ltm_experience_chars: int = 0
    ltm_experience_tokens: int = 0
    wm_preference_chars: int = 0
    wm_preference_tokens: int = 0
    wm_experience_chars: int = 0
    wm_experience_tokens: int = 0
    wm_round_history_chars: int = 0
    ops_with_injection: int = 0
    total_ops: int = 0


@dataclass
class RoundMetrics:
    round_index: int
    user_feedback: str = ""
    user_model_score: float = 0.0
    satisfied: bool = False
    duration_sec: float = 0.0
    pptx_snapshot: str = ""
    injection: InjectionStats = field(default_factory=InjectionStats)


@dataclass
class MemoryDiff:
    preferences_before: int = 0
    preferences_after: int = 0
    preferences_added: int = 0
    experiences_before: int = 0
    experiences_after: int = 0
    experiences_added: int = 0
    chain_experiences_before: int = 0
    chain_experiences_after: int = 0
    chain_experiences_added: int = 0
    profile_rows_before: int = 0
    profile_rows_after: int = 0
    profile_version_before: int = 0
    profile_version_after: int = 0


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count_table(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0] or 0) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def _parse_injection_traces(workspace: Path, round_index: int) -> InjectionStats:
    stats = InjectionStats()

    trace_dirs = [
        workspace / ".memory" / "injection_traces" / f"round_{round_index + 1:03d}",
        workspace / ".memory" / "injection_traces" / f"round_{round_index:03d}",
    ]
    trace_dir = next((path for path in trace_dirs if path.exists()), None)
    if trace_dir is None:
        return stats

    total_ops = 0
    ops_with_injection = 0
    for path in sorted(trace_dir.glob("*.json")):
        total_ops += 1
        payload = _safe_load_json(path)
        if not payload:
            continue
        final_injected = payload.get("final_injected", {})
        if isinstance(final_injected, dict):
            injected_text = str(final_injected.get("content", "") or "")
            injected_chars = int(final_injected.get("chars", 0) or 0)
            injected_tokens = int(final_injected.get("tokens_estimated", 0) or 0)
        else:
            injected_text = ""
            injected_chars = 0
            injected_tokens = 0
        if injected_text or injected_chars > 0:
            ops_with_injection += 1
        components = payload.get("components", {})
        if not isinstance(components, dict):
            components = {}

        def _component_chars(name: str, *legacy_keys: str) -> int:
            component = components.get(name, {})
            if isinstance(component, dict):
                return int(component.get("chars", 0) or 0)
            for legacy_key in legacy_keys:
                legacy_value = payload.get(legacy_key, "")
                if legacy_value:
                    return len(str(legacy_value))
            return 0

        def _component_tokens(name: str, *legacy_keys: str) -> int:
            component = components.get(name, {})
            if isinstance(component, dict):
                tokens = component.get("tokens_estimated", 0)
                if tokens is not None:
                    return int(tokens or 0)
            for legacy_key in legacy_keys:
                legacy_value = payload.get(legacy_key, "")
                if legacy_value:
                    return len(str(legacy_value)) // 4
            return 0

        stats.ltm_experience_chars += _component_chars("ltm_tool_experiences", "ltm_tool_experiences")
        stats.wm_preference_chars += _component_chars("wm_preferences", "wm_preferences")
        stats.wm_experience_chars += _component_chars("wm_experiences", "wm_experiences")
        stats.wm_round_history_chars += _component_chars("wm_round_history", "wm_round_history")
        stats.ltm_experience_tokens += _component_tokens("ltm_tool_experiences", "ltm_tool_experiences")
        stats.wm_preference_tokens += _component_tokens("wm_preferences", "wm_preferences")
        stats.wm_experience_tokens += _component_tokens("wm_experiences", "wm_experiences")
        if injected_tokens > 0 and stats.wm_experience_tokens == 0:
            # Preserve direct top-level injected token estimates when available.
            stats.wm_experience_tokens += injected_tokens

    stats.ops_with_injection = ops_with_injection
    stats.total_ops = total_ops
    return stats


class ExperimentMetrics:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds: list[RoundMetrics] = []
        self._round_start_time = 0.0
        self.resolved_intent_artifact: dict[str, Any] = _safe_load_json(
            self.output_dir / "resolved_intent.json"
        )
        self.template_match_artifact: dict[str, Any] = _safe_load_json(
            self.output_dir / "template_match.json"
        )
        self.runtime_validation: dict[str, Any] = _safe_load_json(
            self.output_dir / "runtime_validation.json"
        )

    def start_round(self) -> None:
        self._round_start_time = time.time()

    def set_runtime_artifacts(
        self,
        *,
        resolved_intent: dict[str, Any] | None = None,
        template_match: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> None:
        if resolved_intent:
            self.resolved_intent_artifact = dict(resolved_intent)
        if template_match:
            self.template_match_artifact = dict(template_match)
        if validation:
            self.runtime_validation = dict(validation)

    def record_round_from_artifacts(
        self,
        workspace: Path,
        round_index: int,
        user_feedback: str,
        user_model_score: float,
        satisfied: bool,
        pptx_snapshot: str = "",
    ) -> RoundMetrics:
        duration = time.time() - self._round_start_time if self._round_start_time else 0.0
        injection = _parse_injection_traces(Path(workspace), round_index)
        record = RoundMetrics(
            round_index=round_index,
            user_feedback=user_feedback,
            user_model_score=user_model_score,
            satisfied=satisfied,
            duration_sec=round(duration, 2),
            pptx_snapshot=pptx_snapshot,
            injection=injection,
        )
        self.rounds.append(record)
        self._save_metrics()
        return record

    def compute_memory_diff(self, before_db: Path, after_db: Path, user_id: str) -> MemoryDiff:
        diff = MemoryDiff()
        try:
            diff.preferences_before = _count_table(before_db, "atomic_preferences")
            diff.preferences_after = _count_table(after_db, "atomic_preferences")
            diff.preferences_added = diff.preferences_after - diff.preferences_before
            diff.experiences_before = _count_table(before_db, "experience_traces")
            diff.experiences_after = _count_table(after_db, "experience_traces")
            diff.experiences_added = diff.experiences_after - diff.experiences_before
            diff.chain_experiences_before = _count_table(before_db, "chain_experiences")
            diff.chain_experiences_after = _count_table(after_db, "chain_experiences")
            diff.chain_experiences_added = diff.chain_experiences_after - diff.chain_experiences_before
        except Exception as e:
            logger.warning("memory diff failed: %s", e)

        path = self.output_dir / "memory_diff.json"
        path.write_text(json.dumps(asdict(diff), ensure_ascii=False, indent=2), encoding="utf-8")
        return diff

    def _save_metrics(self) -> None:
        path = self.output_dir / "metrics.json"
        payload = {
            "rounds": [asdict(item) for item in self.rounds],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize(self) -> dict[str, Any]:
        total_rounds = len(self.rounds)
        final_score = self.rounds[-1].user_model_score if self.rounds else 0.0
        total_duration = sum(item.duration_sec for item in self.rounds)
        summary: dict[str, Any] = {
            "total_rounds": total_rounds,
            "final_score": final_score,
            "early_stopped": bool(self.rounds and self.rounds[-1].satisfied),
            "total_duration_sec": round(total_duration, 2),
            "memory_injection_totals": {
                "ltm_experience_chars": sum(item.injection.ltm_experience_chars for item in self.rounds),
                "wm_preference_chars": sum(item.injection.wm_preference_chars for item in self.rounds),
                "wm_experience_chars": sum(item.injection.wm_experience_chars for item in self.rounds),
                "wm_round_history_chars": sum(item.injection.wm_round_history_chars for item in self.rounds),
            },
            "resolved_memory_intent": str(self.resolved_intent_artifact.get("resolved_memory_intent", "") or ""),
            "resolved_scenario_intent": str(self.resolved_intent_artifact.get("resolved_scenario_intent", "") or ""),
            "resolved_memory_intent_source": str(self.resolved_intent_artifact.get("source", "") or ""),
            "template_intent": str(self.template_match_artifact.get("template_intent", "") or ""),
            "selected_template_name": str(self.template_match_artifact.get("selected_template_name", "") or ""),
            "template_selection_source": str(self.template_match_artifact.get("selection_source", "") or ""),
            "intent_hit": self.runtime_validation.get("intent_hit"),
            "template_hit": self.runtime_validation.get("template_hit"),
            "rounds": [asdict(item) for item in self.rounds],
        }
        (self.output_dir / "metrics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


__all__ = [
    "ExperimentMetrics",
    "InjectionStats",
    "MemoryDiff",
    "RoundMetrics",
]
