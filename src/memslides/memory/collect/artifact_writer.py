"""Implementation details."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RoundArtifactWriter:
    """Implementation details."""

    def __init__(self, memory_dir: Path | str) -> None:
        """Implementation details."""
        self.memory_dir = Path(memory_dir)
        self.rounds_dir = self.memory_dir / "rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)

    def write_round(
        self,
        agent_round: Any,
        raw_tool_calls: list[dict] | None = None,
    ) -> Path:
        """Implementation details."""
        round_id = getattr(agent_round, 'round_id', 0)
        round_dir = self.rounds_dir / f"round_{round_id:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        try:

            self._write_agent_round(round_dir, agent_round, raw_tool_calls)


            if raw_tool_calls:
                self._write_raw_tool_calls(round_dir, round_id, raw_tool_calls)


            self._write_compressed_segments(round_dir, agent_round)


            memory_injection_full = getattr(agent_round, 'memory_injection_full', '')
            if memory_injection_full:
                self._write_memory_injection(round_dir, agent_round)


            agent_reasoning = getattr(agent_round, 'agent_reasoning', '')
            if agent_reasoning:
                self._write_agent_reasoning(round_dir, agent_round)

            logger.debug("Wrote round %d artifacts to %s", round_id, round_dir)
        except Exception as e:
            logger.warning("Failed to write round artifacts: %s", e)

        return round_dir

    def _write_agent_round(
        self,
        round_dir: Path,
        agent_round: Any,
        raw_tool_calls: list[dict] | None,
    ) -> None:
        """Implementation details."""
        segments = getattr(agent_round, 'segments', [])
        memory_injection_full = getattr(agent_round, 'memory_injection_full', '')
        agent_reasoning = getattr(agent_round, 'agent_reasoning', '')
        agent_reply = getattr(agent_round, 'agent_reply', '')

        data = {
            "round_id": getattr(agent_round, 'round_id', 0),
            "agent_name": getattr(agent_round, 'agent_name', ''),
            "session_id": getattr(agent_round, 'session_id', ''),
            "user_id": getattr(agent_round, 'user_id', ''),
            "timestamp": getattr(agent_round, 'timestamp', datetime.now().isoformat()),

            "input": {
                "user_message": getattr(agent_round, 'user_message', ''),
                "memory_injection_tokens": getattr(agent_round, 'memory_injection_tokens', 0),
                "memory_injection_file": "memory_injection.txt" if memory_injection_full else None,
            },

            "execution": {
                "raw_tool_calls_count": len(raw_tool_calls) if raw_tool_calls else 0,
                "raw_tool_calls_file": "raw_tool_calls.json" if raw_tool_calls else None,
                "compressed_segments_count": len(segments),
                "compressed_segments_file": "compressed_segments.json",
                "compression_ratio": self._calc_compression_ratio(raw_tool_calls, segments),
                "agent_reasoning_file": "agent_reasoning.txt" if agent_reasoning else None,
                "duration_seconds": getattr(agent_round, 'duration_seconds', 0.0),
            },

            "output": {
                "agent_reply": agent_reply[:500] if agent_reply else '',
                "agent_reply_truncated": len(agent_reply) > 500 if agent_reply else False,
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
            }
        }

        with open(round_dir / "agent_round.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_raw_tool_calls(
        self,
        round_dir: Path,
        round_id: int,
        raw_tool_calls: list[dict],
    ) -> None:
        """Implementation details."""
        total_chars = sum(
            len(str(tc.get("result", ""))) for tc in raw_tool_calls
        )

        data = {
            "round_id": round_id,
            "tool_calls": raw_tool_calls,
            "total_tool_calls": len(raw_tool_calls),
            "total_result_chars": total_chars,
            "estimated_tokens": total_chars // 4,
        }

        with open(round_dir / "raw_tool_calls.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_compressed_segments(
        self,
        round_dir: Path,
        agent_round: Any,
    ) -> None:
        """Implementation details."""
        segments = getattr(agent_round, 'segments', [])

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

        total_chars = sum(len(json.dumps(s, ensure_ascii=False)) for s in segments_data)

        data = {
            "round_id": getattr(agent_round, 'round_id', 0),
            "segments": segments_data,
            "total_segments": len(segments_data),
            "total_chars": total_chars,
            "estimated_tokens": total_chars // 4,
        }

        with open(round_dir / "compressed_segments.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_memory_injection(
        self,
        round_dir: Path,
        agent_round: Any,
    ) -> None:
        """Implementation details."""
        round_id = getattr(agent_round, 'round_id', 0)
        timestamp = getattr(agent_round, 'timestamp', datetime.now().isoformat())
        tokens = getattr(agent_round, 'memory_injection_tokens', 0)
        content = getattr(agent_round, 'memory_injection_full', '')

        text = f"""=== Memory Context Injected at Round {round_id} ===
Timestamp: {timestamp}
Tokens: {tokens}

{content}
"""
        with open(round_dir / "memory_injection.txt", "w", encoding="utf-8") as f:
            f.write(text)

    def _write_agent_reasoning(
        self,
        round_dir: Path,
        agent_round: Any,
    ) -> None:
        """Implementation details."""
        round_id = getattr(agent_round, 'round_id', 0)
        agent_name = getattr(agent_round, 'agent_name', '')
        timestamp = getattr(agent_round, 'timestamp', datetime.now().isoformat())
        reasoning = getattr(agent_round, 'agent_reasoning', '')

        text = f"""=== Agent Reasoning at Round {round_id} ===
Agent: {agent_name}
Timestamp: {timestamp}

{reasoning}
"""
        with open(round_dir / "agent_reasoning.txt", "w", encoding="utf-8") as f:
            f.write(text)

    def _calc_compression_ratio(
        self,
        raw_tool_calls: list[dict] | None,
        segments: list,
    ) -> float:
        """Implementation details."""
        if not raw_tool_calls:
            return 0.0

        raw_chars = sum(len(str(tc.get("result", ""))) for tc in raw_tool_calls)
        compressed_chars = sum(
            len(getattr(s, 'action_summary', '')) for s in segments
        )

        if raw_chars == 0:
            return 0.0

        return round(compressed_chars / raw_chars, 4)


class ExtractionArtifactWriter:
    """Implementation details."""

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.extractions_dir = self.memory_dir / "extractions"
        self._batch_counters: dict[str, int] = {}

        self._init_batch_counters()

    def _init_batch_counters(self) -> None:
        """Implementation details."""
        for extraction_type in [
            "episode",
            "experience",
            "preference",
            "intent",
            "style_intent",
            "template_selection",
        ]:
            type_dir = self.extractions_dir / extraction_type
            if not type_dir.exists():
                continue

            max_id = -1
            for item in type_dir.iterdir():
                name = item.name


                if item.is_dir() and name.startswith("batch_"):
                    try:
                        batch_id = int(name.split("_")[1])
                        max_id = max(max_id, batch_id)
                    except (ValueError, IndexError):
                        pass


                elif item.is_file() and name.startswith("turn_"):
                    try:
                        turn_id = int(name.split("_")[1])
                        max_id = max(max_id, turn_id)
                    except (ValueError, IndexError):
                        pass

            if max_id >= 0:
                self._batch_counters[extraction_type] = max_id
                logger.debug("ExtractionArtifactWriter: %s counter initialized to %d", extraction_type, max_id)

    def write_episode_extraction(
        self,
        round_ids: list[int],
        prompt: str,
        response: str,
        pre_filter: list | None = None,
        post_filter: list | None = None,
        stored_episodes: list | None = None,
    ) -> Path:
        """Implementation details."""
        batch_id = self._next_batch_id("episode")
        batch_dir = self.extractions_dir / "episode" / f"batch_{batch_id:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # input_rounds.json
        with open(batch_dir / "input_rounds.json", "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": batch_id,
                "round_ids": round_ids,
                "total_rounds": len(round_ids),
            }, f, ensure_ascii=False, indent=2)

        # llm_prompt.txt
        with open(batch_dir / "llm_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        # llm_output.txt
        with open(batch_dir / "llm_output.txt", "w", encoding="utf-8") as f:
            f.write(response)


        if pre_filter is not None:
            with open(batch_dir / "pre_filter.json", "w", encoding="utf-8") as f:
                json.dump(pre_filter, f, ensure_ascii=False, indent=2)


        if post_filter is not None:
            with open(batch_dir / "post_filter.json", "w", encoding="utf-8") as f:
                json.dump(post_filter, f, ensure_ascii=False, indent=2)


        if stored_episodes is not None:
            with open(batch_dir / "stored_episodes.json", "w", encoding="utf-8") as f:
                json.dump(stored_episodes, f, ensure_ascii=False, indent=2)

        logger.debug("Wrote episode extraction batch %d to %s", batch_id, batch_dir)
        return batch_dir

    def write_preference_extraction(
        self,
        episode_id: str,
        episode_data: dict,
        prompt: str,
        response: str,
        preferences: list | None = None,
    ) -> Path:
        """Implementation details."""
        pref_dir = self.extractions_dir / "preference"
        pref_dir.mkdir(parents=True, exist_ok=True)

        # episode_xxx_input.json
        with open(pref_dir / f"episode_{episode_id}_input.json", "w", encoding="utf-8") as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)

        # episode_xxx_prompt.txt
        with open(pref_dir / f"episode_{episode_id}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        # episode_xxx_output.txt
        with open(pref_dir / f"episode_{episode_id}_output.txt", "w", encoding="utf-8") as f:
            f.write(response)

        # episode_xxx_preferences.json
        if preferences is not None:
            with open(pref_dir / f"episode_{episode_id}_preferences.json", "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)

        return pref_dir

    def write_experience_trace(
        self,
        turn_id: int,
        trace_data: dict,
    ) -> Path:
        """Implementation details."""
        exp_dir = self.extractions_dir / "experience"
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / f"turn_{turn_id:03d}_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)

        return exp_dir

    def write_intent_classification(
        self,
        turn_id: int,
        user_message: str,
        prompt: str,
        response: str,
        intent_result: dict | None = None,
    ) -> Path:
        """Implementation details."""
        intent_dir = self.extractions_dir / "intent"
        intent_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"turn_{turn_id:03d}"

        # turn_xxx_prompt.txt
        with open(intent_dir / f"{prefix}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        # turn_xxx_output.txt
        with open(intent_dir / f"{prefix}_output.txt", "w", encoding="utf-8") as f:
            f.write(response)

        # turn_xxx_intent.json
        if intent_result:
            with open(intent_dir / f"{prefix}_intent.json", "w", encoding="utf-8") as f:
                json.dump({
                    "turn_id": turn_id,
                    "user_message": user_message[:200],
                    "timestamp": datetime.now().isoformat(),
                    "intent": intent_result,
                }, f, ensure_ascii=False, indent=2)

        logger.debug("Wrote intent classification turn %d to %s", turn_id, intent_dir)
        return intent_dir

    def write_style_intent(
        self,
        user_message: str,
        prompt: str,
        response: str,
        result: dict | None = None,
        source: str = "",
    ) -> Path:
        """Implementation details."""
        turn_id = self._next_batch_id("style_intent")
        style_dir = self.extractions_dir / "style_intent"
        style_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"turn_{turn_id:03d}"
        with open(style_dir / f"{prefix}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(style_dir / f"{prefix}_output.txt", "w", encoding="utf-8") as f:
            f.write(response)

        if result is not None:
            with open(style_dir / f"{prefix}_style_intent.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "turn_id": turn_id,
                        "user_message": user_message[:500],
                        "timestamp": datetime.now().isoformat(),
                        "source": source,
                        "style_intent": result,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.debug("Wrote style intent turn %d to %s", turn_id, style_dir)
        return style_dir

    def write_template_selection(
        self,
        user_message: str,
        prompt: str,
        response: str,
        selection_result: dict | None = None,
        candidates: list[dict] | None = None,
        usage_history: list[dict] | None = None,
    ) -> Path:
        """Implementation details."""
        turn_id = self._next_batch_id("template_selection")
        select_dir = self.extractions_dir / "template_selection"
        select_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"turn_{turn_id:03d}"
        with open(select_dir / f"{prefix}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(select_dir / f"{prefix}_output.txt", "w", encoding="utf-8") as f:
            f.write(response)

        if candidates is not None:
            with open(select_dir / f"{prefix}_candidates.json", "w", encoding="utf-8") as f:
                json.dump(candidates, f, ensure_ascii=False, indent=2)

        if usage_history is not None:
            with open(select_dir / f"{prefix}_usage_history.json", "w", encoding="utf-8") as f:
                json.dump(usage_history, f, ensure_ascii=False, indent=2)

        if selection_result is not None:
            with open(select_dir / f"{prefix}_selection.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "turn_id": turn_id,
                        "user_message": user_message[:500],
                        "timestamp": datetime.now().isoformat(),
                        "selection": selection_result,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.debug("Wrote template selection turn %d to %s", turn_id, select_dir)
        return select_dir

    def _next_batch_id(self, extraction_type: str) -> int:
        """Implementation details."""
        current = self._batch_counters.get(extraction_type, -1)
        next_id = current + 1
        self._batch_counters[extraction_type] = next_id
        return next_id


class EvolutionTracker:
    """Implementation details."""

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        self.evolution_file = self.memory_dir / "evolution.json"
        self._data: dict = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Implementation details."""
        if self.evolution_file.exists():
            try:
                with open(self.evolution_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "session_id": "",
            "last_updated": datetime.now().isoformat(),
            "rounds_count": 0,
            "evolution": [],
        }

    def record_round(
        self,
        round_id: int,
        user_message: str,
        memory_snapshot: dict,
        extraction_triggered: bool = False,
        new_episodes_count: int = 0,
        new_preferences_count: int = 0,
    ) -> None:
        """Implementation details."""
        entry = {
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message[:100],
            "memory_snapshot": memory_snapshot,
            "extraction_triggered": extraction_triggered,
            "new_episodes_count": new_episodes_count,
            "new_preferences_count": new_preferences_count,
        }

        self._data["evolution"].append(entry)
        self._data["rounds_count"] = len(self._data["evolution"])
        self._data["last_updated"] = datetime.now().isoformat()

        self._save()

    def set_session_id(self, session_id: str) -> None:
        """Implementation details."""
        self._data["session_id"] = session_id
        self._save()

    def _save(self) -> None:
        """Implementation details."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.evolution_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
