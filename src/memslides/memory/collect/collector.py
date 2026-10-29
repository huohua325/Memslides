"""Implementation details."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .tool_segment import AgentRound, CompactRound, ModifyRound, ToolCallSegment
    from .artifact_writer import RoundArtifactWriter
except ImportError:
    from tool_segment import AgentRound, CompactRound, ModifyRound, ToolCallSegment  # type: ignore[no-redef]
    from artifact_writer import RoundArtifactWriter  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_PENDING_FILE = "pending_rounds.json"


def _compact_round_from_dict(d: dict) -> AgentRound:
    """Implementation details."""
    segments = [
        ToolCallSegment(
            tool_name=s.get("tool_name", "unknown"),
            target_file=s.get("target_file", ""),
            action_summary=s.get("action_summary", ""),
            observation_summary=s.get("observation_summary", ""),
            is_error=s.get("is_error", False),
            error_msg=s.get("error_msg", ""),
        )
        for s in d.get("segments", [])
    ]
    return AgentRound(

        round_id=d.get("round_id", 0),
        agent_name=d.get("agent_name", ""),
        session_id=d.get("session_id", ""),
        user_id=d.get("user_id", ""),

        user_message=d.get("user_message", ""),
        memory_injection=d.get("memory_injection", ""),
        memory_injection_full=d.get("memory_injection_full", ""),
        memory_injection_tokens=d.get("memory_injection_tokens", 0),

        segments=segments,
        agent_reasoning=d.get("agent_reasoning", ""),
        reasoning_traces=d.get("reasoning_traces", []),
        agent_reply=d.get("agent_reply", ""),

        timestamp=d.get("timestamp", ""),
        duration_seconds=d.get("duration_seconds", 0.0),
        total_tool_calls=d.get("total_tool_calls", 0),
        error_count=d.get("error_count", 0),
    )


class MemoryCollector:
    """Implementation details."""

    DEFAULT_ROUND_TRIGGER = 3
    DEFAULT_TOKEN_BUDGET = 2000
    # Timed fallback: consolidate after one idle hour.
    IDLE_CONSOLIDATION_INTERVAL = 3600  # seconds

    def __init__(
        self,
        round_trigger: int = DEFAULT_ROUND_TRIGGER,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        persist_dir: str | None = None,
        artifact_writer: RoundArtifactWriter | None = None,
        artifact_dumper: Any = None,
        on_extraction_needed: Any = None,
    ) -> None:
        self._current_round: ModifyRound | None = None
        self._pending_rounds: list[CompactRound] = []
        self._pending_token_count: int = 0
        self._rounds_since_extract: int = 0
        self._round_trigger = round_trigger
        self._token_budget = token_budget
        self._last_compact_round: CompactRound | None = None

        self._last_consolidation_time: float = time.time()

        self._persist_dir: Path | None = Path(persist_dir) if persist_dir else None

        self._on_extraction_needed = on_extraction_needed

        self._artifact_writer = artifact_writer

        self._artifact_dumper = artifact_dumper

        self._raw_tool_calls: list[dict] = []

        self._file_content_cache: dict[str, str] = {}

        self._task_index: int = 0

        self._current_round_started_at: float | None = None

        if self._persist_dir:
            self._restore_pending()



    def begin_round(
        self,
        round_id: int,
        user_message: str,
        agent_name: str = "",
        memory_injection: str = "",
        memory_injection_full: str = "",
        session_id: str = "",
        user_id: str = "",
    ) -> None:
        """Implementation details."""

        self._current_round = ModifyRound(
            round_id=round_id,
            user_message=user_message,
            timestamp=datetime.now().isoformat(),

            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            memory_injection=memory_injection[:500] if memory_injection else "",
            memory_injection_full=memory_injection_full or memory_injection,
        )
        self._current_round_started_at = time.perf_counter()

    def add_tool_call(
        self,
        name: str,
        args: dict | str,
        result: str = "",
        is_error: bool = False,
        duration_ms: int = 0,
        error_msg: str = "",
    ) -> None:
        """Implementation details."""
        if self._current_round:

            self._raw_tool_calls.append({
                "index": len(self._raw_tool_calls),
                "tool_name": name,
                "arguments": args,
                "result": result,
                "duration_ms": duration_ms,
                "is_error": is_error,
                "error_msg": error_msg,
            })


            prev_content = ""
            parsed_args = args if isinstance(args, dict) else {}
            if isinstance(args, str):
                try:
                    import json as _json
                    parsed_args = _json.loads(args)
                except Exception:
                    parsed_args = {}

            if name == "write_html_file":
                target_file = (
                    parsed_args.get("target_file", "")
                    or parsed_args.get("file_path", "")
                    or parsed_args.get("html_file", "")
                    or parsed_args.get("path", "")
                )
                if target_file:
                    prev_content = self._file_content_cache.get(target_file, "")

                    new_content = parsed_args.get("content", "") or parsed_args.get("html_content", "")
                    if new_content:
                        self._file_content_cache[target_file] = new_content


            elif name in ("read_file", "read_html_file"):
                target_file = (
                    parsed_args.get("target_file", "")
                    or parsed_args.get("file_path", "")
                    or parsed_args.get("path", "")
                )

                if target_file and result and not is_error:

                    if target_file.endswith(".html") or target_file.endswith(".htm"):
                        self._file_content_cache[target_file] = result[:50000]


            self._current_round.tool_calls_log.append({
                "name": name,
                "args": args,
                "result": result,
                "is_error": is_error,
                "prev_content": prev_content,
            })

    def add_agent_response(self, text: str) -> None:
        """Implementation details."""
        if self._current_round and text:
            self._current_round.agent_responses.append(text)

    def add_reasoning(self, reasoning: str) -> None:
        """Implementation details."""
        if self._current_round and reasoning:
            truncated = reasoning[:300]
            self._current_round.reasoning_traces.append(truncated)

            self._current_round.agent_reasoning = truncated

    def end_round(
        self,
        duration: float | None = None,
    ) -> CompactRound | None:
        """Implementation details."""
        if not self._current_round:
            return None
        if duration is None:
            if self._current_round_started_at is not None:
                duration = max(time.perf_counter() - self._current_round_started_at, 0.0)
            else:
                duration = 0.0
        self._current_round.duration_seconds = duration
        compact = self._current_round.to_compact()


        if self._artifact_dumper:
            try:

                round_index = getattr(compact, 'round_id', self._task_index)
                self._artifact_dumper.dump_agent_round(
                    round_index=round_index,
                    agent_round=compact,
                    raw_tool_calls=self._raw_tool_calls,
                )
            except Exception as e:
                logger.warning("ArtifactDumper.dump_agent_round failed: %s", e)

        elif self._artifact_writer:
            try:
                self._artifact_writer.write_round(
                    compact,
                    raw_tool_calls=self._raw_tool_calls,
                )
            except Exception as e:
                logger.warning("Failed to write round artifacts: %s", e)


        self._raw_tool_calls = []
        self._current_round_started_at = None

        self._current_round = None
        self._last_compact_round = compact
        self._pending_rounds.append(compact)

        self._pending_token_count += compact.estimate_tokens()
        self._rounds_since_extract += 1
        self._save_pending()
        return compact



    def should_extract(self) -> bool:
        """Return true when round count, token budget, session end, or idle timeout fires."""
        if not self._pending_rounds:
            return False
        if self._rounds_since_extract >= self._round_trigger:
            return True
        if self._pending_token_count >= self._token_budget:
            return True

        elapsed = time.time() - self._last_consolidation_time
        if elapsed > self.IDLE_CONSOLIDATION_INTERVAL:
            return True
        return False

    def pop_batch(self) -> list[CompactRound]:
        """Implementation details."""
        batch = list(self._pending_rounds)
        self._pending_rounds.clear()
        self._pending_token_count = 0
        self._rounds_since_extract = 0

        self._last_consolidation_time = time.time()

        self._save_pending()
        return batch

    def acknowledge_rounds(self, round_ids: list[int]) -> int:
        """Implementation details."""
        if not round_ids or not self._pending_rounds:
            return 0

        round_id_set = set(round_ids)
        before_count = len(self._pending_rounds)
        self._pending_rounds = [
            r for r in self._pending_rounds
            if getattr(r, "round_id", 0) not in round_id_set
        ]
        removed = before_count - len(self._pending_rounds)
        if removed <= 0:
            return 0

        self._pending_token_count = sum(r.estimate_tokens() for r in self._pending_rounds)
        self._rounds_since_extract = len(self._pending_rounds)
        if not self._pending_rounds:
            self._last_consolidation_time = time.time()
        self._save_pending()
        return removed

    def flush(self) -> list[CompactRound]:
        """Implementation details."""
        return self.pop_batch()

    def format_batch_for_extraction(self, batch: list[CompactRound]) -> str:
        """Implementation details."""
        return "\n---\n".join(c.to_extraction_text() for c in batch)



    @property
    def pending_count(self) -> int:
        return len(self._pending_rounds)

    @property
    def pending_tokens(self) -> int:
        return self._pending_token_count

    @property
    def has_active_round(self) -> bool:
        return self._current_round is not None

    @property
    def latest_round(self) -> CompactRound | None:
        return self._last_compact_round



    async def on_rollback(self, data: dict) -> None:
        """Implementation details."""
        rolled_back_turn = data.get("turn", 0) if data else 0


        if self._current_round:
            if self._current_round.round_id == rolled_back_turn or rolled_back_turn == 0:
                logger.debug("on_rollback: discarding current round (round_id=%d)",
                             self._current_round.round_id)
                self._current_round = None
                self._raw_tool_calls = []
                return


        if self._pending_rounds:
            before_count = len(self._pending_rounds)
            if rolled_back_turn > 0:

                self._pending_rounds = [
                    r for r in self._pending_rounds
                    if getattr(r, "round_id", 0) != rolled_back_turn
                ]
            else:

                self._pending_rounds = self._pending_rounds[:-1]

            after_count = len(self._pending_rounds)
            if before_count != after_count:

                self._pending_token_count = sum(r.estimate_tokens() for r in self._pending_rounds)
                self._rounds_since_extract = len(self._pending_rounds)
                self._save_pending()
                logger.debug("on_rollback: removed round (turn=%d), pending: %d → %d",
                             rolled_back_turn, before_count, after_count)


        if self._last_compact_round:
            if rolled_back_turn == 0 or getattr(self._last_compact_round, "round_id", 0) == rolled_back_turn:
                self._last_compact_round = None



    def _pending_path(self) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / _PENDING_FILE

    def _save_pending(self) -> None:
        """Implementation details."""
        path = self._pending_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [dataclasses.asdict(r) for r in self._pending_rounds]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to persist pending rounds: %s", e)

    def _restore_pending(self) -> None:
        """Implementation details."""
        path = self._pending_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rounds = [_compact_round_from_dict(d) for d in data]
            self._pending_rounds = rounds
            self._pending_token_count = sum(r.estimate_tokens() for r in rounds)
            self._rounds_since_extract = len(rounds)


            if rounds:
                self._last_consolidation_time = 0.0
            logger.info(
                "MemoryCollector: restored %d pending rounds from disk (%s), "
                "idle_trigger=%s",
                len(rounds), path, "immediate" if rounds else "normal",
            )
        except Exception as e:
            logger.warning("Failed to restore pending rounds from %s: %s", path, e)
            self._pending_rounds = []
            self._pending_token_count = 0
            self._rounds_since_extract = 0
