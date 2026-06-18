"""ExperienceTraceWriter: write, query, and format ExperienceTrace records.

Provides the complete lifecycle for experiential memory:
1. Write: from_agent_run(), from_modify(), from_tool_error()
2. Query: query_relevant(), query_tool_failures()
3. Format: format_for_prompt(), extract_lessons()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from .core.db import DatabaseBackend
from .core.models import ExperienceTrace, infer_task_experience_category

# Prompt 模板 — 迁移到 memory/prompts/experience.py
from .prompts import TURN_EXPERIENCE_PROMPT as _TURN_EXPERIENCE_PROMPT
from .prompts import MERGE_TOOL_ERROR_PROMPT as _MERGE_TOOL_ERROR_PROMPT

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 8: Role-Based Memory Injection 配置
# ══════════════════════════════════════════════════════════════════════════════

# 标签 → 分类 映射 (applicable_scenarios 字段值 → 语义类别)
SCENARIO_CATEGORIES = {
    # Stage 9: constraint 类别已删除，历史记录已迁移为 tool_error
    "tool_error": ["tool_error"],  # TKL 即时捕获 + remember_lesson
    "tool_misuse": ["soft_failure", "tool_misuse"],  # 工具软错误/误用
    "tool_limitation": ["tool_limitation"],  # 工具能力边界
    "pipeline": ["pipeline", "pattern"],  # 成功模式
    "research": ["agent_research"],  # Research Agent 专属
    "design": ["agent_design"],  # DeckDesigner 专属
    "modify": ["modification"],  # RevisionEditor 专属
}

# Role → 注入配置 (哪些类别注入到哪个 Agent)
ROLE_INJECTION_CONFIG = {
    "research": {
        # Stage 9: l1_constraint 已删除，历史记录已迁移为 tool_error
        "l1_tool_error": True,       # ET 工具硬错误 (TKL 即时 + remember_lesson)
        "l2_value": True,            # AP value
        "l2_strategy": False,        # AP strategy (Research 不注入)
        "l2_role": "research",       # ET 角色专属
        "l3_tool_limitation": False, # (Research 不注入)
        "l3_pipeline": False,        # (Research 不注入)
    },
    "design": {
        "l1_tool_error": True,
        "l2_value": True,
        "l2_strategy": True,
        "l2_role": "design",
        "l3_tool_limitation": True,
        "l3_pipeline": True,
    },
    "modify": {
        "l1_tool_error": True,
        "l2_value": True,
        "l2_strategy": True,
        "l2_role": "modify",
        "l3_tool_limitation": True,
        "l3_pipeline": True,
    },
}


def _parse_json_response(text: str) -> dict:
    """从 LLM 响应中提取 JSON（容错 markdown 代码块）"""
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    return json.loads(text[start:end])


class ExperienceTraceWriter:
    """ExperienceTrace 的完整生命周期: 写入 + 查询 + 格式化注入

    Stage 4 改造: 支持 UnifiedSearchEngine 向量检索（可选）
    - 如果传入 retriever，使用 Chroma+BM25+RRF 混合检索
    - 否则 fallback 到 FTS5 全文检索
    - 可选地写入 .memory/extractions/experience/ 目录供历史调试追溯
    """

    def __init__(
        self,
        db: DatabaseBackend,
        llm: Any = None,
        retriever: Any = None,  # UnifiedSearchEngine 实例（可选）
        extractions_dir: Any = None,  # 兼容保留：历史调试 dump 目录
    ):
        self.db = db
        self.llm = llm  # 可选，用于 LLM 总结 lessons_learned
        self.retriever = retriever  # Stage 4: 向量检索引擎
        self._extractions_dir = extractions_dir  # Stage 4: 文件写入目录
        self._turn_counter = 0  # 轮次计数器

    @staticmethod
    def _active_status_clause(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f"COALESCE({prefix}status, 'active') = 'active'"

    # ── 内部: FK 防御 ──

    async def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure a session row exists for the given session_id.

        experience_traces has FOREIGN KEY (session_id) REFERENCES sessions(id).
        If the session was created via InteractiveSession directly (bypassing
        SessionManager), the sessions row may be missing. This method creates
        it defensively to prevent silent FK failures.
        """
        try:
            existing = await self.db.query_one(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            )
            if not existing:
                await self.db.insert("sessions", {
                    "id": session_id,
                    "user_id": "default",
                    "project_id": session_id,
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                })
                logger.info("Auto-created session row for FK: %s", session_id)
        except Exception as e:
            logger.warning("_ensure_session_exists failed: %s", e)

    # ── 内部: 公共写入逻辑 ──

    async def _write_trace(self, trace: ExperienceTrace, agent_name: str = "", skip_ltm: bool = False) -> None:
        """统一的 trace 写入逻辑：DB + 向量索引 + 文件

        Args:
            skip_ltm: 为 True 时跳过 DB 写入和向量索引，仅保留文件输出（调试用）。
                      用于 extract_to_round_experiences 路径：提取结果只写 WM，LTM 由 consolidation 统一处理。
        """
        try:
            if not skip_ltm:
                await self._ensure_session_exists(trace.session_id)
                await self.db.insert("experience_traces", trace.to_dict())
                await self._index_to_vector(trace)
            if agent_name:
                self._write_to_file(trace, agent_name)
        except Exception as e:
            logger.warning("Failed to write ExperienceTrace (non-fatal): %s", e)

    # ── 内部: 向量索引 ──

    async def _index_to_vector(self, trace: ExperienceTrace) -> None:
        """Stage 4: 写入后同步索引到向量库（如果 retriever 可用）

        支持两种 retriever 类型:
        1. ExperienceTraceRetriever (新): 使用 add_trace() 接口
        2. UnifiedSearchEngine (旧): 使用 add_episode() 接口 (兼容)
        """
        if self.retriever is None:
            return
        try:
            metadata = {
                "trace_id": trace.id,
                "session_id": trace.session_id,
                "outcome": trace.outcome,
                "confidence": trace.confidence,
                "applicable_scenarios": trace.applicable_scenarios,
            }
            # 优先使用 ExperienceTraceRetriever 的 add_trace 接口
            if hasattr(self.retriever, "add_trace"):
                await self.retriever.add_trace(
                    trace_id=trace.id,
                    task_description=trace.task_description,
                    lessons_learned=trace.lessons_learned,
                    metadata=metadata,
                )
            else:
                # 兼容 UnifiedSearchEngine 的 add_episode 接口
                document = f"{trace.task_description}\n{trace.lessons_learned}"
                await self.retriever.add_episode(
                    user_id=trace.session_id or "default",
                    episode_id=trace.id,
                    document=document,
                    metadata=metadata,
                    mem_type="experiences",
                )
        except Exception as e:
            logger.debug("Vector indexing failed (non-fatal): %s", e)

    def _write_to_file(self, trace: ExperienceTrace, agent_name: str = "") -> None:
        """可选历史调试 dump：写入 .memory/extractions/experience/ 目录。"""
        if not self._extractions_dir:
            return
        try:
            from pathlib import Path
            exp_dir = Path(self._extractions_dir) / "experience"
            exp_dir.mkdir(parents=True, exist_ok=True)

            self._turn_counter += 1
            filename = f"turn_{self._turn_counter:03d}_trace.json"
            if agent_name:
                filename = f"{agent_name}_{filename}"

            trace_data = {
                "trace_id": trace.id,
                "session_id": trace.session_id,
                "agent_name": agent_name,
                "task_description": trace.task_description,
                "reasoning_steps": trace.reasoning_steps,
                "tools_used": trace.tools_used,
                "final_outcome": trace.final_outcome,
                "lessons_learned": trace.lessons_learned,
                "applicable_scenarios": trace.applicable_scenarios,
                "confidence": trace.confidence,
                "created_at": trace.created_at,
            }

            filepath = exp_dir / filename
            filepath.write_text(json.dumps(trace_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("Wrote ExperienceTrace to %s", filepath)
        except Exception as e:
            logger.debug("File write failed (non-fatal): %s", e)

    def _write_llm_io_to_file(self, turn: int, prompt: str, response: str, parsed_data: dict) -> None:
        """可选历史调试 dump：写入 Experience 提取的 LLM I/O 中间产物。"""
        if not self._extractions_dir:
            return
        try:
            from pathlib import Path
            turn_dir = Path(self._extractions_dir) / "experience" / f"turn_{turn:03d}"
            turn_dir.mkdir(parents=True, exist_ok=True)

            # 写入 prompt
            (turn_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

            # 写入 raw response
            (turn_dir / "llm_output.txt").write_text(response, encoding="utf-8")

            # 写入 parsed JSON
            (turn_dir / "parsed_experiences.json").write_text(
                json.dumps(parsed_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            logger.debug("Wrote ExperienceTrace LLM I/O to %s", turn_dir)
        except Exception as e:
            logger.debug("LLM I/O file write failed (non-fatal): %s", e)

    # ── 写入 ──

    # ── Stage 14: 统一提取接口 ──

    async def extract_to_round_experiences(
        self,
        agent_round: Any = None,
        tool_calls_log: list[dict] | None = None,
        user_task: str = "",
        session_id: str = "",
        outcome: str = "success",
    ) -> list:
        """从 AgentRound 或 tool_calls_log 提取经验，返回 RoundExperience 列表。

        提取结果只写 WM（skip_ltm=True），用于当前 Job 内的后续检索与注入。
        是否持久化到 LTM 由编排层当前策略决定。

        Args:
            agent_round: AgentRound 数据（优先使用）
            tool_calls_log: 工具调用日志（agent_round 不可用时的 fallback）
            user_task: 用户任务描述
            session_id: Session ID
            outcome: "success" | "partial" | "failed"

        Returns:
            list[RoundExperience]，每个元素 source="auto_extract"
        """
        from .core.models import RoundExperience

        traces: list[ExperienceTrace] = []

        try:
            if agent_round is not None:
                # 路径 A: 有 AgentRound — skip_ltm=True，只提取不写 LTM
                traces = await self.from_agent_round(agent_round, outcome=outcome, skip_ltm=True)
            elif tool_calls_log:
                # 路径 B: 有 tool_calls_log — skip_ltm=True，只提取不写 LTM
                traces = await self.from_modify_structured(
                    session_id=session_id or "default",
                    user_task=user_task,
                    tool_calls_log=tool_calls_log,
                    outcome=outcome,
                    skip_ltm=True,
                )
        except Exception as e:
            logger.warning(f"extract_to_round_experiences LLM extraction failed (non-fatal): {e}")
            return []

        # 仅转换为 RoundExperience 写入 WM，供当前 Job 后续 Round 检索使用
        round_experiences = []
        for trace in traces:
            try:
                # 解析 JSON 字段
                tools_used = json.loads(trace.tools_used) if isinstance(trace.tools_used, str) else trace.tools_used
                keywords = json.loads(trace.applicable_scenarios) if isinstance(trace.applicable_scenarios, str) else trace.applicable_scenarios

                te = RoundExperience(
                    content=trace.lessons_learned or trace.task_description,
                    tool_name=tools_used[0] if tools_used else "",
                    keywords=keywords if isinstance(keywords, list) else [],
                    category=infer_task_experience_category(
                        trace.experience_type,
                        keywords,
                        trace.final_outcome,
                    ),
                    source="auto_extract",
                    source_task_id=trace.session_id,
                    confidence=trace.confidence,
                    tools_used=tools_used if isinstance(tools_used, list) else [],
                    outcome=trace.final_outcome,
                )
                if te.content:
                    round_experiences.append(te)
            except Exception as e:
                logger.debug(f"Failed to convert ExperienceTrace to RoundExperience: {e}")
                continue

        return round_experiences

    async def extract_to_task_experiences(self, *args: Any, **kwargs: Any) -> list:
        """Deprecated compatibility wrapper for ``extract_to_round_experiences``."""
        return await self.extract_to_round_experiences(*args, **kwargs)

    async def from_agent_run(
        self,
        session_id: str,
        agent_name: str,
        chat_history: list,
        tool_history: list,
        task: str,
        outcome: str,
        template_id: str = "",
    ) -> ExperienceTrace:
        """从单个 Agent 的 run 过程生成经验 — 逐 Agent 记录

        Stage 3 改造 3.4: 废弃后台总结，改用同步结构化提取。
        将 tool_history 转换为 tool_calls_log 格式，复用 from_modify_structured 路径。

        Args:
            agent_name: "research" / "design" / "template_analysis"
            chat_history: 该 Agent 的 chat_history
            tool_history: 该 Agent 阶段的 tool_history 切片
            task: 任务描述
            outcome: "success" / "failed" / "partial"
        """
        # 将 tool_history (ToolCall, ChatMessage) 转换为 tool_calls_log 格式
        tool_calls_log = []
        for call, result in tool_history:
            tool_calls_log.append({
                "name": call.function.name if hasattr(call, "function") else str(call),
                "args": call.function.arguments if hasattr(call, "function") else "",
                "result_preview": (getattr(result, "text", "") or "")[:500],
                "is_error": getattr(result, "is_error", False),
                "duration_ms": int(getattr(result, "extra_info", {}).get("duration_ms", 0) or 0),
            })

        # 尝试结构化提取 (复用 from_modify_structured 路径)
        if self.llm and tool_calls_log:
            try:
                traces = await self.from_modify_structured(
                    session_id=session_id,
                    user_task=f"[{agent_name}] {task}",
                    tool_calls_log=tool_calls_log,
                    outcome=outcome,
                    template_id=template_id,
                )
                if traces:
                    return traces[0]
            except Exception as e:
                logger.warning("from_agent_run structured extraction failed: %s → fallback", e)

        # Fallback: 写入基本 trace (不调用后台总结)
        tool_errors = [
            (call, result) for call, result in tool_history
            if hasattr(result, "is_error") and result.is_error
        ]
        tools_used = self._extract_tools_used(tool_history)
        error_summary = self._summarize_tool_errors(tool_errors)

        trace = ExperienceTrace(
            session_id=session_id,
            task_description=f"[{agent_name}] {task[:400]}",
            reasoning_steps=json.dumps(
                self._extract_reasoning_steps(chat_history), ensure_ascii=False
            ),
            tools_used=json.dumps(tools_used, ensure_ascii=False),
            final_outcome=outcome,
            lessons_learned=error_summary,
            applicable_scenarios=json.dumps(
                ["initial_generation", f"agent_{agent_name}"]
            ),
            confidence=0.8 if outcome == "success" else 0.9,
            template_id=template_id,
            experience_type="agent_run",
        )
        await self._write_trace(trace, agent_name)
        return trace

    async def from_modify(
        self,
        session_id: str,
        user_msg: str,
        agent_response: str,
        tool_calls_log: list[dict],
        outcome: str,
        debug_tracer: Any = None,
        turn: int = 0,
        template_id: str = "",
    ) -> ExperienceTrace:
        """从 modify() 路径生成经验 — 降级路径，直接调用结构化版本"""
        return await self.from_modify_structured(
            session_id=session_id,
            user_msg=user_msg,
            agent_response=agent_response,
            tool_calls_log=tool_calls_log,
            outcome=outcome,
            debug_tracer=debug_tracer,
            turn=turn,
            template_id=template_id,
        )

    async def from_tool_error(
        self,
        session_id: str,
        tool_name: str,
        args: str,
        error_msg: str,
    ) -> ExperienceTrace:
        """工具错误 → 即时经验（如 "image_generation 不支持大尺寸图片"）

        相同工具的错误会被合并（LLM background merge），不重复插入新行。
        这些即时经验可以在同一 run() 中被后续 Agent 查询到。
        """
        new_lesson = f"{tool_name} failed: {error_msg}\nArgs: {str(args)[:300]}"
        try:
            existing = await self.db.query_one(
                "SELECT * FROM experience_traces "
                "WHERE task_description = ? AND final_outcome = 'failed' "
                f"AND {self._active_status_clause()} "
                "ORDER BY created_at DESC LIMIT 1",
                (f"Tool error: {tool_name}",),
            )
        except Exception:
            existing = None

        if existing:
            existing_trace = ExperienceTrace.from_dict(existing)
            if self.llm:
                asyncio.create_task(self._background_merge_tool_error(
                    trace_id=existing_trace.id,
                    tool_name=tool_name,
                    existing_lessons=existing_trace.lessons_learned or "",
                    new_error=error_msg,
                    new_args=str(args)[:300],
                ))
            else:
                merged = (existing_trace.lessons_learned or "") + "\n---\n" + new_lesson
                try:
                    await self.db.execute(
                        "UPDATE experience_traces SET lessons_learned = ?, created_at = ? WHERE id = ?",
                        (merged[:2000], datetime.now().isoformat(), existing_trace.id),
                    )
                except Exception as e:
                    logger.warning("Failed to merge tool error trace (non-fatal): %s", e)
            return existing_trace

        trace = ExperienceTrace(
            session_id=session_id,
            task_description=f"Tool error: {tool_name}",
            tools_used=json.dumps([tool_name]),
            final_outcome="failed",
            lessons_learned=new_lesson,
            applicable_scenarios=json.dumps(
                [f"tool_{tool_name}", "tool_error"]
            ),
            confidence=0.95,
            experience_type="tool_error",
        )
        await self._write_trace(trace)
        return trace

    # ── 结构化经验提取 (六类) ──

    def _format_agent_round_for_experience(self, agent_round: Any) -> str:
        """为 Experience Trace 生成精简的工具序列（不包含 memory injection）

        Experience Trace 分析的是工具调用本身的模式和错误，
        不需要 memory 上下文（那是 Episode 提取关注的）。

        格式：
        [Round N] User: <user_message>
        Tool chain: tool1 → tool2 → ...
        Changes: slide1: +prop1; slide2: -prop2
        """
        lines = []

        # 用户任务
        context = f" [{agent_round.agent_name}]" if agent_round.agent_name else ""
        lines.append(f"[Round {agent_round.round_id}]{context} User: {agent_round.user_message}")

        # 工具链（核心信息）
        if agent_round.segments:
            chain = " → ".join(s.to_text() for s in agent_round.segments)
            lines.append(f"Tool chain: {chain}")

        return "\n".join(lines)

    def _format_tool_sequence_compact(self, tool_calls_log: list[dict]) -> str:
        """复用 Layer 1 压缩：ToolCallSegment 对每种工具做语义摘要，
        消除 HTML/base64 等原始内容膨胀。
        """
        try:
            from .collect.tool_segment import ToolCallSegment
        except ImportError:
            lines = []
            for i, t in enumerate(tool_calls_log, 1):
                status = "[ERROR]" if t.get("is_error") else "[OK]"
                lines.append(f"{i}. {t.get('name', '?')} → {status}")
            return "\n".join(lines)

        segments = [
            ToolCallSegment.from_raw_tool_call(
                name=tc.get("name", "unknown"),
                args=tc.get("args", {}),
                result=tc.get("result_preview", ""),
                is_error=tc.get("is_error", False),
            )
            for tc in tool_calls_log
        ]
        return " → ".join(s.to_text() for s in segments)

    async def _write_typed_trace(
        self,
        session_id: str,
        task: str,
        tools: list[str],
        outcome: str,
        lesson: str,
        scenarios: list[str],
        experience_type: str,
        confidence: float = 0.8,
        template_id: str = "",
        skip_ltm: bool = False,
    ) -> ExperienceTrace | None:
        """写入单条带类型标签的 ExperienceTrace"""
        if not lesson.strip():
            return None
        trace = ExperienceTrace(
            session_id=session_id,
            task_description=task[:500],
            tools_used=json.dumps(tools, ensure_ascii=False),
            final_outcome=outcome,
            lessons_learned=lesson[:1000],
            applicable_scenarios=json.dumps(scenarios),
            confidence=confidence,
            template_id=template_id,
            experience_type=experience_type,
        )
        await self._write_trace(trace, skip_ltm=skip_ltm)
        return trace

    async def _extract_typed_traces(
        self,
        session_id: str,
        data: dict,
        user_task: str,
        tools_used: list[str],
        template_id: str = "",
        skip_ltm: bool = False,
    ) -> list[ExperienceTrace]:
        """从 LLM 返回的结构化数据中提取各类经验 trace"""
        traces: list[ExperienceTrace] = []

        # tool_misuse
        for tm in data.get("tool_misuse", []):
            t = await self._write_typed_trace(
                session_id=session_id,
                task=f"[{user_task[:80]}] Misuse: {tm.get('tool', '?')} — {tm.get('description', '')[:60]}",
                tools=[tm.get("tool", "?")] if tm.get("tool") else tools_used,
                outcome="partial",
                lesson=tm.get("lesson", ""),
                scenarios=["soft_failure", "tool_misuse", f"tool_{tm.get('tool', '')}"],
                experience_type="tool_misuse",
                confidence=0.85,
                template_id=template_id,
                skip_ltm=skip_ltm,
            )
            if t:
                traces.append(t)

        # tool_limitations
        for tl in data.get("tool_limitations", []):
            t = await self._write_typed_trace(
                session_id=session_id,
                task=f"Tool limitation: {tl.get('tool', '?')} — {tl.get('limitation', '')[:80]}",
                tools=[tl.get("tool", "?")],
                outcome="partial",
                lesson=f"{tl.get('limitation', '')} Workaround: {tl.get('workaround', '')}",
                scenarios=["tool_limitation", f"tool_{tl.get('tool', '')}"],
                experience_type="tool_limitation",
                confidence=0.95,
                template_id=template_id,
                skip_ltm=skip_ltm,
            )
            if t:
                traces.append(t)

        # effective_patterns
        for ep in data.get("effective_patterns", []):
            task_type = ep.get("task_type", "")
            t = await self._write_typed_trace(
                session_id=session_id,
                task=f"[{task_type or user_task[:40]}] Pipeline: {ep.get('applicable_when', '')[:60]}",
                tools=ep.get("pipeline", []),
                outcome="success",
                lesson=ep.get("lesson", ""),
                scenarios=["pipeline", "pattern", f"task_{task_type}" if task_type else "general"],
                experience_type="pattern",
                confidence=0.8,
                template_id=template_id,
                skip_ltm=skip_ltm,
            )
            if t:
                traces.append(t)

        return traces

    async def _extract_hard_error_traces(
        self,
        session_id: str,
        data: dict,
        user_task: str,
        template_id: str = "",
        skip_ltm: bool = False,
    ) -> list[ExperienceTrace]:
        """从 LLM 返回的 hard_errors 中提取 tool_error traces。"""
        traces: list[ExperienceTrace] = []
        for err in data.get("hard_errors", []):
            t = await self._write_typed_trace(
                session_id=session_id,
                task=f"[{user_task[:80]}] Tool error: {err.get('tool', '?')}",
                tools=[err.get("tool", "?")],
                outcome="failed",
                lesson=err.get("lesson", ""),
                scenarios=["tool_error", f"tool_{err.get('tool', '')}"],
                experience_type="tool_error",
                confidence=0.9,
                template_id=template_id,
                skip_ltm=skip_ltm,
            )
            if t:
                traces.append(t)
        return traces

    async def from_agent_round(
        self,
        agent_round: Any,
        outcome: str = "success",
        skip_ltm: bool = False,
    ) -> list[ExperienceTrace]:
        """从 AgentRound 创建 ExperienceTrace 列表 — 统一输入接口

        Stage 4 改造: 直接使用 AgentRound.to_extraction_text() 生成完整的
        工具序列信息，包含 memory_injection、agent_reasoning 等上下文。

        Args:
            agent_round: 完整的 AgentRound 数据 (from tool_segment.py)
            outcome: "success" | "partial" | "failed"

        Returns:
            写入数据库的 ExperienceTrace 列表，失败时返回空列表
        """
        if not self.llm or not agent_round.segments:
            return []

        # 为 Experience Trace 生成精简的工具序列（不包含 memory injection）
        # Experience Trace 分析的是工具调用本身，不需要 memory 上下文
        tool_sequence = self._format_agent_round_for_experience(agent_round)

        prompt = _TURN_EXPERIENCE_PROMPT.format(
            user_task=agent_round.user_message[:300],
            tool_sequence=tool_sequence[:3000],
            outcome=outcome,
        )

        try:
            response = await self.llm(prompt)
            data = _parse_json_response(response)
        except Exception as e:
            logger.warning("from_agent_round LLM/parse failed: %s → skip", e)
            return []

        # 写入 LLM prompt/output 中间产物
        self._write_llm_io_to_file(agent_round.round_id, prompt, response, data)

        # 提取工具列表用于 trace 记录
        tools_used = [seg.tool_name for seg in agent_round.segments]
        session_id = agent_round.session_id or "unknown"

        traces = await self._extract_hard_error_traces(
            session_id=session_id,
            data=data,
            user_task=agent_round.user_message,
            template_id="",
            skip_ltm=skip_ltm,
        )
        traces.extend(await self._extract_typed_traces(
            session_id=session_id,
            data=data,
            user_task=agent_round.user_message,
            tools_used=tools_used,
            template_id="",
            skip_ltm=skip_ltm,
        ))

        return traces

    async def from_modify_structured(
        self,
        session_id: str,
        user_task: str,
        tool_calls_log: list[dict],
        outcome: str,
        agent_response: str = "",
        debug_tracer: Any = None,
        turn: int = 0,
        template_id: str = "",
        skip_ltm: bool = False,
    ) -> list[ExperienceTrace]:
        """一次 LLM 调用分析整轮工具序列，提取五类结构化经验。

        Stage 3 改造 3.4: LLM 不可用 / tool_calls_log 为空 / LLM 失败
        → 返回空列表（不再 fallback 写入无价值的空 trace）。
        """
        if not self.llm or not tool_calls_log:
            return []

        tool_sequence = self._format_tool_sequence_compact(tool_calls_log)
        prompt = _TURN_EXPERIENCE_PROMPT.format(
            user_task=user_task[:300],
            tool_sequence=tool_sequence[:3000],
            outcome=outcome,
        )
        try:
            response = await self.llm(prompt)
            data = _parse_json_response(response)
        except Exception as e:
            logger.warning("from_modify_structured LLM/parse failed: %s → skip", e)
            return []

        # Stage 4 新增: 保存 LLM prompt/output 中间产物
        self._write_llm_io_to_file(turn, prompt, response, data)

        # 提取工具列表
        tools_used = [c["name"] for c in tool_calls_log]

        traces = await self._extract_hard_error_traces(
            session_id=session_id,
            data=data,
            user_task=user_task,
            template_id=template_id,
            skip_ltm=skip_ltm,
        )

        # 使用统一的提取方法处理其他类型
        traces.extend(await self._extract_typed_traces(
            session_id=session_id,
            data=data,
            user_task=user_task,
            tools_used=tools_used,
            template_id=template_id,
            skip_ltm=skip_ltm,
        ))

        return traces

    # ── 查询 ──

    async def query_relevant(
        self,
        user_id: str = "",
        task_description: str = "",
        scenarios: list[str] | None = None,
        outcome_filter: str = "",
        limit: int = 5,
    ) -> list[ExperienceTrace]:
        """查询相关经验 — 优先返回失败经验

        Args:
            outcome_filter: "failed" / "success" / "" (all)
        """
        where = []
        params: list = []

        where.append(self._active_status_clause())
        if outcome_filter:
            where.append("final_outcome = ?")
            params.append(outcome_filter)
        if scenarios:
            for s in scenarios:
                where.append("applicable_scenarios LIKE ?")
                params.append(f"%{s}%")
        if task_description:
            where.append("task_description LIKE ?")
            params.append(f"%{task_description[:50]}%")

        where_sql = " AND ".join(where) if where else "1=1"
        rows = await self.db.query(
            f"SELECT * FROM experience_traces WHERE {where_sql} "
            "ORDER BY confidence DESC, reuse_count DESC, created_at DESC LIMIT ?",
            tuple([*params, limit]),
        )
        return [ExperienceTrace.from_dict(r) for r in rows]

    async def query_tool_failures(
        self, tool_name: str = "", limit: int = 3
    ) -> list[ExperienceTrace]:
        """专门查询工具失败经验 — 用于注入 Agent prompt"""
        if tool_name:
            rows = await self.db.query(
                "SELECT * FROM experience_traces "
                "WHERE final_outcome = 'failed' AND applicable_scenarios LIKE ? "
                f"AND {self._active_status_clause()} "
                "ORDER BY confidence DESC LIMIT ?",
                (f"%tool_{tool_name}%", limit),
            )
        else:
            rows = await self.db.query(
                "SELECT * FROM experience_traces "
                "WHERE final_outcome = 'failed' AND applicable_scenarios LIKE '%tool_error%' "
                f"AND {self._active_status_clause()} "
                "ORDER BY confidence DESC LIMIT ?",
                (limit,),
            )
        return [ExperienceTrace.from_dict(r) for r in rows]

    async def query_failed_experiences(self, limit: int = 5) -> list[ExperienceTrace]:
        """查询所有失败经验（全局） — 工具错误/系统问题跨session共享"""
        rows = await self.db.query(
            "SELECT * FROM experience_traces WHERE final_outcome = 'failed' "
            f"AND {self._active_status_clause()} "
            "ORDER BY confidence DESC, created_at DESC LIMIT ?",
            (limit,),
        )
        return [ExperienceTrace.from_dict(r) for r in rows]

    async def query_project_experiences(
        self, session_id: str, limit: int = 5
    ) -> list[ExperienceTrace]:
        """查询同项目的成功经验（项目隔离）

        成功经验是项目特定的（用户偏好/风格规则），不应跨项目共享。
        通过session_id关联到同一project_id下的所有session。
        """
        rows = await self.db.query(
            "SELECT et.* FROM experience_traces et "
            "JOIN sessions s ON et.session_id = s.id "
            "WHERE et.final_outcome != 'failed' "
            f"AND {self._active_status_clause('et')} "
            "AND s.project_id = (SELECT project_id FROM sessions WHERE id = ? LIMIT 1) "
            "ORDER BY et.confidence DESC, et.reuse_count DESC, et.created_at DESC LIMIT ?",
            (session_id, limit),
        )
        return [ExperienceTrace.from_dict(r) for r in rows]

    async def query_for_task(
        self,
        user_task: str,
        session_id: str = "",
        limit: int = 8,
    ) -> list[ExperienceTrace]:
        """检索与当前任务相关的经验。

        Stage 4 改造: 优先使用向量检索（如果 retriever 可用）

        检索策略 (四路合并，去重):
        0. 向量检索 — Chroma+BM25+RRF 混合检索（如果 retriever 可用）
        1. FTS5 全文匹配 — task_description + lessons_learned 与 user_task 匹配
        2. 高置信度失败经验 — tool_error / tool_misuse / tool_limitation (全局共享)
        3. 同项目经验 — pipeline / domain_rule (项目隔离)

        返回去重后的 trace 列表，按 confidence 降序排列。
        """
        seen_ids: set[str] = set()
        results: list[ExperienceTrace] = []

        def _add(traces: list[ExperienceTrace]) -> None:
            for t in traces:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    results.append(t)

        # ── 路径 0: 向量检索（Stage 4 新增）──
        if self.retriever is not None:
            try:
                vector_results = await self.retriever.search(
                    query=user_task[:300],
                    user_id=session_id or "default",
                    top_k=limit,
                    method="hybrid",  # Chroma + BM25 + RRF
                )
                # 从向量结果中提取 trace_id，再从 DB 加载完整 trace
                for vr in vector_results:
                    trace_id = vr.metadata.get("trace_id") if vr.metadata else None
                    if trace_id and trace_id not in seen_ids:
                        try:
                            row = await self.db.query_one(
                                "SELECT * FROM experience_traces "
                                f"WHERE id = ? AND {self._active_status_clause()}",
                                (trace_id,),
                            )
                            if row:
                                _add([ExperienceTrace.from_dict(row)])
                        except Exception:
                            pass
                logger.debug("Vector retrieval returned %d results", len(vector_results))
            except Exception as e:
                logger.debug("Vector retrieval failed (fallback to FTS5): %s", e)

        # ── 路径 1: FTS5 全文匹配 ──
        try:
            fts_query = user_task[:200].replace('"', ' ')
            rows = await self.db.query(
                "SELECT t.*, f.rank FROM experience_traces t "
                "JOIN experience_traces_fts f ON t.rowid = f.rowid "
                "WHERE experience_traces_fts MATCH ? "
                f"AND {self._active_status_clause('t')} "
                "ORDER BY f.rank LIMIT ?",
                (f'"{fts_query}"', limit),
            )
            _add([ExperienceTrace.from_dict(r) for r in rows])
        except Exception as e:
            logger.debug("FTS5 query failed (fallback to SQL): %s", e)
            # FTS5 不可用时 fallback 到 LIKE 匹配
            try:
                rows = await self.db.query(
                    "SELECT * FROM experience_traces "
                    f"WHERE {self._active_status_clause()} "
                    "AND (task_description LIKE ? OR lessons_learned LIKE ?) "
                    "ORDER BY confidence DESC LIMIT ?",
                    (f"%{user_task[:50]}%", f"%{user_task[:50]}%", limit),
                )
                _add([ExperienceTrace.from_dict(r) for r in rows])
            except Exception:
                pass

        # ── 路径 2: 高置信度失败/误用经验 (全局共享) ──
        try:
            rows = await self.db.query(
                "SELECT * FROM experience_traces "
                "WHERE (applicable_scenarios LIKE '%tool_error%' "
                "   OR applicable_scenarios LIKE '%tool_misuse%' "
                "   OR applicable_scenarios LIKE '%tool_limitation%') "
                f"AND {self._active_status_clause()} "
                "AND confidence >= 0.8 "
                "ORDER BY confidence DESC, created_at DESC LIMIT ?",
                (5,),
            )
            _add([ExperienceTrace.from_dict(r) for r in rows])
        except Exception:
            pass

        # ── 路径 3: 同项目经验 (pipeline / domain_rule) ──
        if session_id:
            try:
                rows = await self.db.query(
                    "SELECT et.* FROM experience_traces et "
                    "JOIN sessions s ON et.session_id = s.id "
                    "WHERE (et.applicable_scenarios LIKE '%pipeline%' "
                    "   OR et.applicable_scenarios LIKE '%domain_rule%') "
                    f"AND {self._active_status_clause('et')} "
                    "AND s.project_id = (SELECT project_id FROM sessions WHERE id = ? LIMIT 1) "
                    "ORDER BY et.confidence DESC LIMIT ?",
                    (session_id, 5),
                )
                _add([ExperienceTrace.from_dict(r) for r in rows])
            except Exception:
                pass

        # 按 衰减分数 降序 (Stage 3 改造 3.6: 时间衰减)
        results.sort(key=lambda t: self._decayed_score(t), reverse=True)
        return results[:limit]

    @staticmethod
    def _decayed_score(t: ExperienceTrace) -> float:
        """计算时间衰减分数: confidence / (1 + age_days/30)

        30天前的经验权重衰减到50%, 90天前衰减到25%。
        reuse_count 每次+0.05分 奖励。
        """
        try:
            age = datetime.now() - datetime.fromisoformat(t.created_at)
            age_days = max(age.total_seconds() / 86400, 0)
        except Exception:
            age_days = 0
        decay = 1.0 / (1.0 + age_days / 30.0)
        reuse_bonus = t.reuse_count * 0.05
        return t.confidence * decay + reuse_bonus

    # ══════════════════════════════════════════════════════════════════════════════
    # Stage 8: Role-Based Query
    # ══════════════════════════════════════════════════════════════════════════════

    async def query_for_role(
        self,
        user_task: str,
        agent_role: str,  # "research" | "design" | "modify"
        session_id: str = "",
    ) -> dict[str, list[ExperienceTrace]]:
        """按 Agent Role 分层检索经验 (Stage 8/9)

        根据 ROLE_INJECTION_CONFIG 配置，按角色差异化返回各层级的经验。

        Returns:
            {
                "l1_tool_error": [...],   # ET 工具硬错误 (TKL 即时 + remember_lesson)
                "l2_role": [...],         # ET 角色专属
                "l3_tool_limitation": [...],  # ET 工具边界 (Design/Modify only)
                "l3_pipeline": [...],     # ET 成功模式 (Design/Modify only)
            }
        """
        config = ROLE_INJECTION_CONFIG.get(agent_role, ROLE_INJECTION_CONFIG["modify"])
        result: dict[str, list[ExperienceTrace]] = {
            "l1_tool_error": [],
            "l2_role": [],
            "l3_tool_limitation": [],
            "l3_pipeline": [],
        }

        # Stage 9: l1_constraint 已删除，历史记录已迁移为 tool_error

        # L1: 工具硬错误 (TKL 即时捕获 + remember_lesson，所有 Agent 都注入)
        if config.get("l1_tool_error"):
            result["l1_tool_error"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES["tool_error"], limit=10,
                exclude_other_roles=agent_role,
            )

        # L2: 角色专属经验 (只查当前角色的，全部检索)
        role_key = config.get("l2_role", agent_role)
        if role_key and role_key in SCENARIO_CATEGORIES:
            result["l2_role"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES[role_key], limit=5,
            )

        # L3: 工具能力边界 (Design/Modify only，全部检索，排除其他 Agent 专属)
        if config.get("l3_tool_limitation"):
            result["l3_tool_limitation"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES["tool_limitation"], limit=5,
                exclude_other_roles=agent_role,
            )

        # L3: 成功模式 (Design/Modify only，全部检索，排除其他 Agent 专属)
        if config.get("l3_pipeline"):
            result["l3_pipeline"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES["pipeline"], limit=5,
                exclude_other_roles=agent_role,
            )

        return result

    async def _query_by_scenarios(
        self,
        scenario_tags: list[str],
        limit: int = 1,
        exclude_other_roles: str = "",
    ) -> list[ExperienceTrace]:
        """按 applicable_scenarios 标签查询经验

        Args:
            scenario_tags: 要匹配的标签列表
            limit: 返回数量限制
            exclude_other_roles: 当前 agent_role，用于排除其他 Agent 专属标签
        """
        if not scenario_tags:
            return []

        # 构建 OR 条件: applicable_scenarios LIKE '%tag1%' OR ... '%tagN%'
        where_clauses = ["applicable_scenarios LIKE ?" for _ in scenario_tags]
        params = [f"%{tag}%" for tag in scenario_tags]

        # Stage 8: 排除其他 Agent 专属标签
        exclude_clauses = []
        if exclude_other_roles:
            other_role_tags = []
            if exclude_other_roles != "research":
                other_role_tags.extend(SCENARIO_CATEGORIES.get("research", []))
            if exclude_other_roles != "design":
                other_role_tags.extend(SCENARIO_CATEGORIES.get("design", []))
            if exclude_other_roles != "modify":
                other_role_tags.extend(SCENARIO_CATEGORIES.get("modify", []))
            for tag in other_role_tags:
                exclude_clauses.append("applicable_scenarios NOT LIKE ?")
                params.append(f"%{tag}%")

        exclude_sql = f" AND {' AND '.join(exclude_clauses)}" if exclude_clauses else ""

        try:
            rows = await self.db.query(
                f"SELECT * FROM experience_traces "
                f"WHERE {self._active_status_clause()} "
                f"AND ({' OR '.join(where_clauses)}){exclude_sql} "
                f"ORDER BY confidence DESC, created_at DESC LIMIT ?",
                tuple([*params, limit]),
            )
            return [ExperienceTrace.from_dict(r) for r in rows]
        except Exception as e:
            logger.warning("_query_by_scenarios failed: %s", e)
            return []

    def _has_tag(self, trace: ExperienceTrace, tag: str) -> bool:
        """检查 trace 是否包含指定标签"""
        try:
            scenarios = json.loads(trace.applicable_scenarios or "[]")
            return tag in scenarios
        except Exception:
            return False

    def _has_any_tag(self, trace: ExperienceTrace, tags: set[str]) -> bool:
        """检查 trace 是否包含指定标签集中的任一标签"""
        try:
            scenarios = json.loads(trace.applicable_scenarios or "[]")
            return bool(set(scenarios) & tags)
        except Exception:
            return False

    async def mark_reused(self, trace_ids: list[str]) -> None:
        """经验被注入 prompt 后调用，递增 reuse_count。

        Stage 3 改造 3.7: 改为 await 直接调用（<1ms 轻量 SQL），
        不再用 asyncio.create_task 以避免 GC 竞态。
        """
        if not trace_ids:
            return
        for tid in trace_ids:
            try:
                await self.db.execute(
                    "UPDATE experience_traces SET reuse_count = reuse_count + 1 WHERE id = ?",
                    (tid,),
                )
            except Exception as e:
                logger.warning("mark_reused failed for %s (non-fatal): %s", tid[:8], e)

    # ── GC (Stage 3 改造 3.6) ──

    async def gc_expired(self, max_age_days: int = 90) -> int:
        """删除超过 max_age_days 天且 reuse_count=0 的低价值经验。

        保留条件（任一满足则不删）:
        - reuse_count > 0 (曾被复用)
        - confidence >= 0.9 (高置信度，如 tool_error / domain_rule)
        - 创建时间 < max_age_days

        Returns: 删除的记录数
        """
        try:
            result = await self.db.execute(
                "DELETE FROM experience_traces "
                "WHERE julianday('now') - julianday(created_at) > ? "
                "AND reuse_count = 0 "
                "AND confidence < 0.9",
                (max_age_days,),
            )
            deleted = result if isinstance(result, int) else 0
            if deleted:
                logger.info("GC expired: deleted %d traces older than %d days", deleted, max_age_days)
            return deleted
        except Exception as e:
            logger.warning("gc_expired failed (non-fatal): %s", e)
            return 0

    async def gc_by_capacity(self, max_traces: int = 200) -> int:
        """超过容量限制时删除最低价值的经验。

        价值 = confidence * time_decay + reuse_count * 0.05
        删除底部 20% 的低价值经验。

        Returns: 删除的记录数
        """
        try:
            rows = await self.db.query(
                "SELECT COUNT(*) as cnt FROM experience_traces", ()
            )
            total = rows[0]["cnt"] if rows else 0
            if total <= max_traces:
                return 0

            # 计算要删除的数量 (20% of excess)
            excess = total - max_traces
            to_delete = max(excess, int(total * 0.2))

            # 按衰减分数排序，删除最低的
            rows = await self.db.query(
                "SELECT id, confidence, reuse_count, created_at, "
                "  confidence * (1.0 / (1 + (julianday('now') - julianday(created_at)) / 30.0)) "
                "  + reuse_count * 0.05 as value_score "
                "FROM experience_traces "
                "ORDER BY value_score ASC LIMIT ?",
                (to_delete,),
            )
            if not rows:
                return 0

            ids_to_delete = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids_to_delete))
            result = await self.db.execute(
                f"DELETE FROM experience_traces WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
            )
            deleted = result if isinstance(result, int) else len(ids_to_delete)
            logger.info("GC capacity: deleted %d traces (total was %d, max %d)", deleted, total, max_traces)
            return deleted
        except Exception as e:
            logger.warning("gc_by_capacity failed (non-fatal): %s", e)
            return 0

    async def run_gc(self, max_age_days: int = 90, max_traces: int = 200) -> dict:
        """运行完整 GC 流程。应在 session 结束时调用。

        Returns: {"expired": N, "capacity": N}
        """
        expired = await self.gc_expired(max_age_days)
        capacity = await self.gc_by_capacity(max_traces)
        return {"expired": expired, "capacity": capacity}

    # ── 迁移 (Stage 4) ──

    async def migrate_to_vector(self, batch_size: int = 50) -> dict:
        """将历史 experience_traces 迁移到向量库。

        遍历所有 traces，调用 _index_to_vector 索引到向量库。
        用于一次性迁移历史数据。

        Returns: {"total": N, "indexed": N, "failed": N}
        """
        if self.retriever is None:
            logger.warning("migrate_to_vector: retriever is None, skipping")
            return {"total": 0, "indexed": 0, "failed": 0, "error": "no retriever"}

        total = 0
        indexed = 0
        failed = 0

        try:
            # 获取所有 traces
            rows = await self.db.query(
                "SELECT * FROM experience_traces "
                f"WHERE {self._active_status_clause()} "
                "ORDER BY created_at DESC",
                (),
            )
            total = len(rows)
            logger.info("migrate_to_vector: found %d traces to migrate", total)

            for i, row in enumerate(rows):
                try:
                    trace = ExperienceTrace.from_dict(row)
                    await self._index_to_vector(trace)
                    indexed += 1
                except Exception as e:
                    failed += 1
                    logger.debug("migrate_to_vector: failed for trace %s: %s", row.get("id", "?")[:8], e)

                # 进度日志
                if (i + 1) % batch_size == 0:
                    logger.info("migrate_to_vector: progress %d/%d", i + 1, total)

            logger.info("migrate_to_vector: completed. indexed=%d, failed=%d", indexed, failed)
        except Exception as e:
            logger.warning("migrate_to_vector failed: %s", e)
            return {"total": total, "indexed": indexed, "failed": failed, "error": str(e)}

        return {"total": total, "indexed": indexed, "failed": failed}

    # ── 格式化 (注入 prompt) ──

    def format_for_prompt(self, traces: list[ExperienceTrace]) -> str:
        """将 ExperienceTrace 列表格式化为可注入 prompt 的文本

        按五个区块分类展示（兼容新旧两种 trace 格式）：
        1. 工具使用禁忌与能力边界 (hard_error + tool_limitation)
        2. 必须遵守的领域规则 (domain_rule)
        3. 工具角色与使用范围 (soft_failure / tool_misuse)
        4. 高效工具流水线 (pipeline/pattern)
        5. 可参考的成功经验 (其余 success)
        """
        if not traces:
            return ""

        hard = [t for t in traces if self._has_tag(t, "tool_error")]
        limits = [t for t in traces if self._has_tag(t, "tool_limitation")]
        domain = [t for t in traces if self._has_tag(t, "domain_rule")]
        soft = [t for t in traces if self._has_tag(t, "soft_failure")]
        pats = [t for t in traces if self._has_tag(t, "pipeline") or self._has_tag(t, "pattern")]

        _categorized_ids = {id(t) for t in hard + limits + domain + soft + pats}
        old_failed = [t for t in traces if t.final_outcome == "failed" and id(t) not in _categorized_ids]
        success = [t for t in traces if t.final_outcome == "success" and id(t) not in _categorized_ids]

        sections: list[str] = []

        if hard or limits or old_failed:
            sections.append("### 工具使用禁忌与能力边界")
            for t in (hard + limits + old_failed)[:4]:
                if t.lessons_learned:
                    sections.append(f"- {t.lessons_learned}")

        if domain:
            sections.append("### 必须遵守的领域规则")
            for t in domain[:3]:
                if t.lessons_learned:
                    sections.append(f"- {t.lessons_learned}")

        if soft:
            sections.append("### 工具角色与使用范围")
            for t in soft[:3]:
                if t.lessons_learned:
                    sections.append(f"- {t.lessons_learned}")

        if pats:
            sections.append("### 高效工具流水线")
            for t in pats[:3]:
                if not t.lessons_learned:
                    continue
                # 展示完整编排: [任务类型] tool1 → tool2 → tool3 + 关键要点
                try:
                    tools = json.loads(t.tools_used or "[]")
                except Exception:
                    tools = []
                if tools:
                    pipeline_str = " → ".join(tools)
                    sections.append(f"- [{t.task_description[:50]}] {pipeline_str}")
                    sections.append(f"  要点: {t.lessons_learned}")
                else:
                    sections.append(f"- {t.lessons_learned}")

        if success:
            sections.append("### 可参考的成功经验")
            for t in success[:2]:
                lesson = (t.lessons_learned or "").strip() or f"成功完成（复用{t.reuse_count}次）"
                sections.append(f"- [{t.task_description[:60]}] {lesson}")

        return "## 历史经验参考\n" + "\n".join(sections) if sections else ""

    def extract_lessons(self, trace: ExperienceTrace) -> str:
        """从单个 trace 中提取教训文本 — 用于 Agent 间传递"""
        if not trace.lessons_learned:
            return ""
        return f"来自 {trace.task_description} 的经验: {trace.lessons_learned}"

    # ── 内部方法 ──

    def _extract_reasoning_steps(self, chat_history: list) -> list[dict]:
        """从对话历史中提取 assistant 的推理步骤"""
        steps = []
        for msg in chat_history:
            role = getattr(msg, "role", None)
            text = getattr(msg, "text", None) or getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)

            if role and str(role).lower() in ("assistant",) and text:
                steps.append({"thought": str(text)[:300]})
            if tool_calls:
                for tc in tool_calls:
                    name = getattr(tc, "function", None)
                    if name:
                        name = getattr(name, "name", str(name))
                    steps.append({"tool_call": str(name)})
        return steps

    def _extract_tools_used(self, tool_history: list) -> list[str]:
        """从工具历史中提取使用的工具列表"""
        tools = set()
        for item in tool_history:
            if isinstance(item, tuple) and len(item) >= 1:
                call = item[0]
                func = getattr(call, "function", None)
                if func:
                    name = getattr(func, "name", str(func))
                    tools.add(str(name))
        return list(tools)

    def _summarize_tool_errors(self, tool_data: list) -> str:
        """总结工具错误 - 支持两种输入格式"""
        if not tool_data:
            return ""
        lines = []
        for item in tool_data:
            if isinstance(item, dict):
                # tool_calls_log 格式
                lines.append(f"{item['name']}: {item.get('result_preview', 'unknown error')[:200]}")
            else:
                # (call, result) tuple 格式
                call, result = item
                func = getattr(call, "function", None)
                name = getattr(func, "name", str(func)) if func else "unknown"
                text = getattr(result, "text", None)
                if text is None:
                    content = getattr(result, "content", None)
                    if isinstance(content, list) and content:
                        text = str(content[0])
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = str(result)
                lines.append(f"{name}: {str(text)[:200]}")
        return "Tool errors: " + "; ".join(lines)

    async def _background_merge_tool_error(
        self,
        trace_id: str,
        tool_name: str,
        existing_lessons: str,
        new_error: str,
        new_args: str,
    ) -> None:
        """Background task: use LLM to merge new tool error into existing trace.

        Runs as asyncio.create_task() — never blocks the main pipeline.
        Updates lessons_learned and refreshes created_at so this record
        stays at the top of ORDER BY created_at DESC queries.
        """
        try:
            prompt = _MERGE_TOOL_ERROR_PROMPT.format(
                tool_name=tool_name,
                existing_lessons=existing_lessons[:600],
                new_error=new_error[:300],
                new_args=new_args[:200],
            )
            merged = await self.llm(prompt)
            merged = merged.strip() if merged else ""
            if not merged or merged in ("", "无", "N/A"):
                merged = existing_lessons + "\n---\n" + f"{tool_name} failed: {new_error}\nArgs: {new_args}"
            await self.db.execute(
                "UPDATE experience_traces SET lessons_learned = ?, created_at = ? WHERE id = ?",
                (merged[:2000], datetime.now().isoformat(), trace_id),
            )
            logger.info("Merged tool error trace %s for %s", trace_id[:8], tool_name)
        except Exception as e:
            logger.warning("_background_merge_tool_error failed (non-fatal): %s", e)

    # _background_summarize_lessons — DEPRECATED (Stage 3, 改造 3.4)
    # 产出的教训 47% 同质化 ("先inspect再改", "减少重复调用").
    # reuse_count 全为 0 证实 asyncio.create_task 在 session 结束前被 GC.
    # 现已由 from_modify_structured() 的同步结构化提取完全替代。
