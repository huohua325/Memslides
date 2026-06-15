"""
collector.py — MemoryCollector 双层缓冲区采集器

记忆系统 collect 阶段的核心组件，是 Agent 交互数据进入记忆管线的唯一入口。

职责:
  在 Agent 每轮交互过程中实时收集原始数据（用户消息、工具调用、Agent 回复等），
  通过双层缓冲区机制压缩并积攒，在满足触发条件时输出 batch
  供下游 EpisodeExtractor 提取结构化记忆。

双层缓冲区:
  Layer 1 — 单轮采集:
    begin_round → add_tool_call × N → add_agent_response → end_round
    每轮结束时通过 ModifyRound.to_compact() 将原始 tool_calls 压缩为
    AgentRound（含 ToolCallSegment 摘要），同时通过 RoundArtifactWriter
    将原始数据和压缩产物落盘到 .memory/rounds/。

  Layer 2 — 批次触发:
    积攒多轮 AgentRound，当满足以下四个条件之一时 pop_batch() 输出整个 batch:
      A. 累积 N 轮未提取 (默认 N=3)
      B. Session 结束 (flush)
      C. 累积 token 超预算 (默认 2000)
      D. 空闲超时 (默认 1 小时)

附加能力:
  - Rollback 处理: 撤销操作时从 pending batch 中移除对应 round
  - 持久化恢复: 进程重启后从磁盘恢复 pending rounds，并立即触发 idle timeout
  - Event Bus 集成: 通过 on_rollback 回调响应外部事件
"""

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
    """从 JSON dict 反序列化 AgentRound（重启恢复用）

    Stage 4 改造: 支持 AgentRound 新增字段的反序列化。
    """
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
        # 基础信息
        round_id=d.get("round_id", 0),
        agent_name=d.get("agent_name", ""),
        session_id=d.get("session_id", ""),
        user_id=d.get("user_id", ""),
        # 输入上下文
        user_message=d.get("user_message", ""),
        memory_injection=d.get("memory_injection", ""),
        memory_injection_full=d.get("memory_injection_full", ""),
        memory_injection_tokens=d.get("memory_injection_tokens", 0),
        # 执行过程
        segments=segments,
        agent_reasoning=d.get("agent_reasoning", ""),
        reasoning_traces=d.get("reasoning_traces", []),
        agent_reply=d.get("agent_reply", ""),
        # 统计信息
        timestamp=d.get("timestamp", ""),
        duration_seconds=d.get("duration_seconds", 0.0),
        total_tool_calls=d.get("total_tool_calls", 0),
        error_count=d.get("error_count", 0),
    )


class MemoryCollector:
    """双层缓冲区记忆采集器

    Layer 1 — 单轮采集:
        begin_round → add_tool_call × N → add_agent_response → end_round → CompactRound

    Layer 2 — 批次触发 (四条件任一):
        A. 累积 N 轮未提取 (默认 N=3)
        B. Session 结束 (flush)
        C. 累积 token 超预算 (默认 2000)
        D. 空闲超时 (默认 1 小时)
    """

    DEFAULT_ROUND_TRIGGER = 3
    DEFAULT_TOKEN_BUDGET = 2000
    # Timed fallback: consolidate after one idle hour.
    IDLE_CONSOLIDATION_INTERVAL = 3600  # seconds

    def __init__(
        self,
        round_trigger: int = DEFAULT_ROUND_TRIGGER,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        persist_dir: str | None = None,
        artifact_writer: RoundArtifactWriter | None = None,  # Stage 4 新增 (v1)
        artifact_dumper: Any = None,  # Stage 12 新增: v2 ArtifactDumper
        on_extraction_needed: Any = None,  # Stage 4 新增: 满意度触发提取回调
    ) -> None:
        self._current_round: ModifyRound | None = None
        self._pending_rounds: list[CompactRound] = []
        self._pending_token_count: int = 0
        self._rounds_since_extract: int = 0
        self._round_trigger = round_trigger
        self._token_budget = token_budget
        self._last_compact_round: CompactRound | None = None
        # 定时兜底机制
        self._last_consolidation_time: float = time.time()
        # 持久化目录（可选）
        self._persist_dir: Path | None = Path(persist_dir) if persist_dir else None
        # Stage 4 新增: 满意度触发提取回调
        self._on_extraction_needed = on_extraction_needed
        # Stage 4 新增: 中间产物写入器 (v1)
        self._artifact_writer = artifact_writer
        # Stage 12 新增: v2 ArtifactDumper (优先使用)
        self._artifact_dumper = artifact_dumper
        # Stage 4 新增: 存储原始 tool_calls (压缩前)
        self._raw_tool_calls: list[dict] = []
        # Stage 5 新增: 文件内容缓存 (用于 before→after diff)
        self._file_content_cache: dict[str, str] = {}
        # Stage 12 新增: task_index 计数器
        self._task_index: int = 0
        # 使用 monotonic clock 统一计算 round duration，避免调用方手工传占位值
        self._current_round_started_at: float | None = None
        # 进程重启后恢复 pending rounds
        if self._persist_dir:
            self._restore_pending()

    # ── Layer 1: 单轮采集 ──

    def begin_round(
        self,
        round_id: int,
        user_message: str,
        agent_name: str = "",                    # Stage 4 新增
        memory_injection: str = "",              # Stage 4 新增
        memory_injection_full: str = "",         # Stage 4 新增 (可选)
        session_id: str = "",                    # Stage 4 新增
        user_id: str = "",                       # Stage 4 新增
    ) -> None:
        """开始一轮修改的采集

        Stage 4 改造: 新增 agent_name, memory_injection 等参数，
        支持生成包含完整上下文的 AgentRound。

        Args:
            round_id: 轮次 ID
            user_message: 用户原始消息
            agent_name: Agent 名称 ("design" / "research" / "modify")
            memory_injection: 注入的 memory prompt (压缩版 ≤500字符)
            memory_injection_full: 完整 memory prompt (用于调试)
            session_id: 会话 ID
            user_id: 用户 ID
        """

        self._current_round = ModifyRound(
            round_id=round_id,
            user_message=user_message,
            timestamp=datetime.now().isoformat(),
            # Stage 4 新增字段
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
        duration_ms: int = 0,  # Stage 4 新增
        error_msg: str = "",  # Stage 4 新增
    ) -> None:
        """记录一次 tool call

        Stage 4 改造: 同时保存原始数据和压缩版本。
        Stage 5 改造: 支持 before→after diff (通过 file_content_cache)。
        """
        if self._current_round:
            # 保存原始 tool_call (用于中间产物)
            self._raw_tool_calls.append({
                "index": len(self._raw_tool_calls),
                "tool_name": name,
                "arguments": args,
                "result": result,
                "duration_ms": duration_ms,
                "is_error": is_error,
                "error_msg": error_msg,
            })

            # Stage 5 新增: 获取 prev_content 用于 diff
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
                    # 更新缓存为新内容
                    new_content = parsed_args.get("content", "") or parsed_args.get("html_content", "")
                    if new_content:
                        self._file_content_cache[target_file] = new_content

            # Stage 5 新增: read_file/read_html_file 时缓存文件内容（用于后续 write 的 diff）
            elif name in ("read_file", "read_html_file"):
                target_file = (
                    parsed_args.get("target_file", "")
                    or parsed_args.get("file_path", "")
                    or parsed_args.get("path", "")
                )
                # result 包含文件内容，缓存它
                if target_file and result and not is_error:
                    # 只缓存 HTML 文件
                    if target_file.endswith(".html") or target_file.endswith(".htm"):
                        self._file_content_cache[target_file] = result[:50000]  # 限制缓存大小

            # 原有逻辑: 生成压缩的 ToolCallSegment (现在带 prev_content)
            self._current_round.tool_calls_log.append({
                "name": name,
                "args": args,
                "result": result,
                "is_error": is_error,
                "prev_content": prev_content,  # Stage 5 新增
            })

    def add_agent_response(self, text: str) -> None:
        """记录 agent 回复"""
        if self._current_round and text:
            self._current_round.agent_responses.append(text)

    def add_reasoning(self, reasoning: str) -> None:
        """记录一次 agent reasoning/thinking（每个 LLM turn 的推理内容）。

        reasoning 来自 LLM 响应的 reasoning_content 或 content（当伴随 tool_calls 时）。
        截断到 300 字符以控制 token 预算。
        同时更新 agent_reasoning（取最新一条，向后兼容旧代码）。
        """
        if self._current_round and reasoning:
            truncated = reasoning[:300]
            self._current_round.reasoning_traces.append(truncated)
            # 向后兼容：agent_reasoning 保留最新一条
            self._current_round.agent_reasoning = truncated

    def end_round(
        self,
        duration: float | None = None,
    ) -> CompactRound | None:
        """结束一轮采集，执行 Layer 1 压缩

        Stage 4 改造: 写入中间产物。
        Stage 12 改造: 优先使用 ArtifactDumper (v2)，fallback 到 ArtifactWriter (v1)。
        """
        if not self._current_round:
            return None
        if duration is None:
            if self._current_round_started_at is not None:
                duration = max(time.perf_counter() - self._current_round_started_at, 0.0)
            else:
                duration = 0.0
        self._current_round.duration_seconds = duration
        compact = self._current_round.to_compact()

        # Stage 12: 优先使用 ArtifactDumper (v2 统一目录 round_XXX/)
        if self._artifact_dumper:
            try:
                # 使用 round_id（来自 orchestrator 的 round_count）保持与 on_round_end 的 round_index 对齐
                round_index = getattr(compact, 'round_id', self._task_index)
                self._artifact_dumper.dump_agent_round(
                    round_index=round_index,
                    agent_round=compact,
                    raw_tool_calls=self._raw_tool_calls,
                )
            except Exception as e:
                logger.warning("ArtifactDumper.dump_agent_round failed: %s", e)
        # Stage 4: fallback 到 ArtifactWriter (v1 目录 round_XXX/)
        elif self._artifact_writer:
            try:
                self._artifact_writer.write_round(
                    compact,
                    raw_tool_calls=self._raw_tool_calls,
                )
            except Exception as e:
                logger.warning("Failed to write round artifacts: %s", e)

        # 清理原始 tool_calls
        self._raw_tool_calls = []
        self._current_round_started_at = None

        self._current_round = None
        self._last_compact_round = compact
        self._pending_rounds.append(compact)

        self._pending_token_count += compact.estimate_tokens()
        self._rounds_since_extract += 1
        self._save_pending()
        return compact

    # ── Layer 2: 批次触发 ──

    def should_extract(self) -> bool:
        """Return true when round count, token budget, session end, or idle timeout fires."""
        if not self._pending_rounds:
            return False
        if self._rounds_since_extract >= self._round_trigger:
            return True
        if self._pending_token_count >= self._token_budget:
            return True
        # 定时兜底检查：距上次整理超过 1 小时
        elapsed = time.time() - self._last_consolidation_time
        if elapsed > self.IDLE_CONSOLIDATION_INTERVAL:
            return True
        return False

    def pop_batch(self) -> list[CompactRound]:
        """弹出所有 pending rounds，重置状态"""
        batch = list(self._pending_rounds)
        self._pending_rounds.clear()
        self._pending_token_count = 0
        self._rounds_since_extract = 0
        # 更新最后整理时间
        self._last_consolidation_time = time.time()
        # 清空持久化缓冲区（已提取，无需再恢复）
        self._save_pending()
        return batch

    def acknowledge_rounds(self, round_ids: list[int]) -> int:
        """确认这些 round 已被上游处理，避免后续 flush/批量提取重复消费。

        该方法用于 orchestrator 已经对单轮做过 episode 提取的场景：
        round 仍然会暂存在 collector 的 pending 队列中，如果不清理，
        job 结束时 collector.flush() 会再次把同一 round 送去提取。
        """
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
        """Session 结束时 flush 所有 pending"""
        return self.pop_batch()

    def format_batch_for_extraction(self, batch: list[CompactRound]) -> str:
        """格式化 batch 为 EpisodeExtractor 的输入文本"""
        return "\n---\n".join(c.to_extraction_text() for c in batch)

    # ── 状态查询 ──

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

    # ── Event Bus 回调 ──

    async def on_rollback(self, data: dict) -> None:
        """rollback_executed 事件回调 → 从 pending 中移除被撤销的 round

        Stage 4 修复: 撤销操作应该从 batch 中移除被撤销的 round，
        而不仅仅是标记 undo。否则会导致 round_id 重复。
        """
        rolled_back_turn = data.get("turn", 0) if data else 0

        # 1. 如果当前正在收集的 round 被撤销，丢弃它
        if self._current_round:
            if self._current_round.round_id == rolled_back_turn or rolled_back_turn == 0:
                logger.debug("on_rollback: discarding current round (round_id=%d)",
                             self._current_round.round_id)
                self._current_round = None
                self._raw_tool_calls = []
                return

        # 2. 从 pending_rounds 中移除被撤销的 round
        if self._pending_rounds:
            before_count = len(self._pending_rounds)
            if rolled_back_turn > 0:
                # 移除指定 round_id 的 round
                self._pending_rounds = [
                    r for r in self._pending_rounds
                    if getattr(r, "round_id", 0) != rolled_back_turn
                ]
            else:
                # 没有指定 turn，移除最后一个 round
                self._pending_rounds = self._pending_rounds[:-1]

            after_count = len(self._pending_rounds)
            if before_count != after_count:
                # 重新计算 token 数
                self._pending_token_count = sum(r.estimate_tokens() for r in self._pending_rounds)
                self._rounds_since_extract = len(self._pending_rounds)
                self._save_pending()
                logger.debug("on_rollback: removed round (turn=%d), pending: %d → %d",
                             rolled_back_turn, before_count, after_count)

        # 3. 清除 last_compact_round 引用（如果它被撤销了）
        if self._last_compact_round:
            if rolled_back_turn == 0 or getattr(self._last_compact_round, "round_id", 0) == rolled_back_turn:
                self._last_compact_round = None

    # ── 持久化 ──

    def _pending_path(self) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / _PENDING_FILE

    def _save_pending(self) -> None:
        """将当前 pending rounds 序列化到磁盘（best-effort）"""
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
        """进程重启时从磁盘恢复 pending rounds

        关键修复: 如果有 pending rounds，将 _last_consolidation_time 设为 0，
        使 idle timeout 条件立即满足，确保 stale 的 pending rounds 在下一次
        should_extract() 调用时被触发提取。
        """
        path = self._pending_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rounds = [_compact_round_from_dict(d) for d in data]
            self._pending_rounds = rounds
            self._pending_token_count = sum(r.estimate_tokens() for r in rounds)
            self._rounds_since_extract = len(rounds)
            # 关键修复: 如果有恢复的 pending rounds，立即触发 idle timeout
            # 将 _last_consolidation_time 设为 0，使下次 should_extract() 为 True
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
