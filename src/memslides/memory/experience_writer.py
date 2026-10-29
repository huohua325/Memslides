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


from .prompts import TURN_EXPERIENCE_PROMPT as _TURN_EXPERIENCE_PROMPT
from .prompts import MERGE_TOOL_ERROR_PROMPT as _MERGE_TOOL_ERROR_PROMPT

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════


SCENARIO_CATEGORIES = {

    "tool_error": ["tool_error"],
    "tool_misuse": ["soft_failure", "tool_misuse"],
    "tool_limitation": ["tool_limitation"],
    "pipeline": ["pipeline", "pattern"],
    "research": ["agent_research"],
    "design": ["agent_design"],
    "modify": ["modification"],
}


ROLE_INJECTION_CONFIG = {
    "research": {

        "l1_tool_error": True,
        "l2_value": True,            # AP value
        "l2_strategy": False,
        "l2_role": "research",
        "l3_tool_limitation": False,
        "l3_pipeline": False,
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
    """Implementation details."""
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    return json.loads(text[start:end])


class ExperienceTraceWriter:
    """Implementation details."""

    def __init__(
        self,
        db: DatabaseBackend,
        llm: Any = None,
        retriever: Any = None,
        extractions_dir: Any = None,
    ):
        self.db = db
        self.llm = llm
        self.retriever = retriever
        self._extractions_dir = extractions_dir
        self._turn_counter = 0

    @staticmethod
    def _active_status_clause(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f"COALESCE({prefix}status, 'active') = 'active'"



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



    async def _write_trace(self, trace: ExperienceTrace, agent_name: str = "", skip_ltm: bool = False) -> None:
        """Implementation details."""
        try:
            if not skip_ltm:
                await self._ensure_session_exists(trace.session_id)
                await self.db.insert("experience_traces", trace.to_dict())
                await self._index_to_vector(trace)
            if agent_name:
                self._write_to_file(trace, agent_name)
        except Exception as e:
            logger.warning("Failed to write ExperienceTrace (non-fatal): %s", e)



    async def _index_to_vector(self, trace: ExperienceTrace) -> None:
        """Implementation details."""
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

            if hasattr(self.retriever, "add_trace"):
                await self.retriever.add_trace(
                    trace_id=trace.id,
                    task_description=trace.task_description,
                    lessons_learned=trace.lessons_learned,
                    metadata=metadata,
                )
            else:

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
        """Implementation details."""
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
        """Implementation details."""
        if not self._extractions_dir:
            return
        try:
            from pathlib import Path
            turn_dir = Path(self._extractions_dir) / "experience" / f"turn_{turn:03d}"
            turn_dir.mkdir(parents=True, exist_ok=True)


            (turn_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")


            (turn_dir / "llm_output.txt").write_text(response, encoding="utf-8")


            (turn_dir / "parsed_experiences.json").write_text(
                json.dumps(parsed_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            logger.debug("Wrote ExperienceTrace LLM I/O to %s", turn_dir)
        except Exception as e:
            logger.debug("LLM I/O file write failed (non-fatal): %s", e)





    async def extract_to_round_experiences(
        self,
        agent_round: Any = None,
        tool_calls_log: list[dict] | None = None,
        user_task: str = "",
        session_id: str = "",
        outcome: str = "success",
    ) -> list:
        """Implementation details."""
        from .core.models import RoundExperience

        traces: list[ExperienceTrace] = []

        try:
            if agent_round is not None:

                traces = await self.from_agent_round(agent_round, outcome=outcome, skip_ltm=True)
            elif tool_calls_log:

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


        round_experiences = []
        for trace in traces:
            try:

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
        """Implementation details."""

        tool_calls_log = []
        for call, result in tool_history:
            tool_calls_log.append({
                "name": call.function.name if hasattr(call, "function") else str(call),
                "args": call.function.arguments if hasattr(call, "function") else "",
                "result_preview": (getattr(result, "text", "") or "")[:500],
                "is_error": getattr(result, "is_error", False),
                "duration_ms": int(getattr(result, "extra_info", {}).get("duration_ms", 0) or 0),
            })


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
        """Implementation details."""
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
        """Implementation details."""
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



    def _format_agent_round_for_experience(self, agent_round: Any) -> str:
        """Implementation details."""
        lines = []


        context = f" [{agent_round.agent_name}]" if agent_round.agent_name else ""
        lines.append(f"[Round {agent_round.round_id}]{context} User: {agent_round.user_message}")


        if agent_round.segments:
            chain = " → ".join(s.to_text() for s in agent_round.segments)
            lines.append(f"Tool chain: {chain}")

        return "\n".join(lines)

    def _format_tool_sequence_compact(self, tool_calls_log: list[dict]) -> str:
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
        if not self.llm or not agent_round.segments:
            return []



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


        self._write_llm_io_to_file(agent_round.round_id, prompt, response, data)


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
        """Implementation details."""
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


        self._write_llm_io_to_file(turn, prompt, response, data)


        tools_used = [c["name"] for c in tool_calls_log]

        traces = await self._extract_hard_error_traces(
            session_id=session_id,
            data=data,
            user_task=user_task,
            template_id=template_id,
            skip_ltm=skip_ltm,
        )


        traces.extend(await self._extract_typed_traces(
            session_id=session_id,
            data=data,
            user_task=user_task,
            tools_used=tools_used,
            template_id=template_id,
            skip_ltm=skip_ltm,
        ))

        return traces



    async def query_relevant(
        self,
        user_id: str = "",
        task_description: str = "",
        scenarios: list[str] | None = None,
        outcome_filter: str = "",
        limit: int = 5,
    ) -> list[ExperienceTrace]:
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
        seen_ids: set[str] = set()
        results: list[ExperienceTrace] = []

        def _add(traces: list[ExperienceTrace]) -> None:
            for t in traces:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    results.append(t)


        if self.retriever is not None:
            try:
                vector_results = await self.retriever.search(
                    query=user_task[:300],
                    user_id=session_id or "default",
                    top_k=limit,
                    method="hybrid",  # Chroma + BM25 + RRF
                )

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


        results.sort(key=lambda t: self._decayed_score(t), reverse=True)
        return results[:limit]

    @staticmethod
    def _decayed_score(t: ExperienceTrace) -> float:
        """Implementation details."""
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
        """Implementation details."""
        config = ROLE_INJECTION_CONFIG.get(agent_role, ROLE_INJECTION_CONFIG["modify"])
        result: dict[str, list[ExperienceTrace]] = {
            "l1_tool_error": [],
            "l2_role": [],
            "l3_tool_limitation": [],
            "l3_pipeline": [],
        }




        if config.get("l1_tool_error"):
            result["l1_tool_error"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES["tool_error"], limit=10,
                exclude_other_roles=agent_role,
            )


        role_key = config.get("l2_role", agent_role)
        if role_key and role_key in SCENARIO_CATEGORIES:
            result["l2_role"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES[role_key], limit=5,
            )


        if config.get("l3_tool_limitation"):
            result["l3_tool_limitation"] = await self._query_by_scenarios(
                SCENARIO_CATEGORIES["tool_limitation"], limit=5,
                exclude_other_roles=agent_role,
            )


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
        """Implementation details."""
        if not scenario_tags:
            return []


        where_clauses = ["applicable_scenarios LIKE ?" for _ in scenario_tags]
        params = [f"%{tag}%" for tag in scenario_tags]


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
        """Implementation details."""
        try:
            scenarios = json.loads(trace.applicable_scenarios or "[]")
            return tag in scenarios
        except Exception:
            return False

    def _has_any_tag(self, trace: ExperienceTrace, tags: set[str]) -> bool:
        """Implementation details."""
        try:
            scenarios = json.loads(trace.applicable_scenarios or "[]")
            return bool(set(scenarios) & tags)
        except Exception:
            return False

    async def mark_reused(self, trace_ids: list[str]) -> None:
        """Implementation details."""
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



    async def gc_expired(self, max_age_days: int = 90) -> int:
        """Implementation details."""
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
        """Implementation details."""
        try:
            rows = await self.db.query(
                "SELECT COUNT(*) as cnt FROM experience_traces", ()
            )
            total = rows[0]["cnt"] if rows else 0
            if total <= max_traces:
                return 0


            excess = total - max_traces
            to_delete = max(excess, int(total * 0.2))


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
        """Implementation details."""
        expired = await self.gc_expired(max_age_days)
        capacity = await self.gc_by_capacity(max_traces)
        return {"expired": expired, "capacity": capacity}



    async def migrate_to_vector(self, batch_size: int = 50) -> dict:
        """Implementation details."""
        if self.retriever is None:
            logger.warning("migrate_to_vector: retriever is None, skipping")
            return {"total": 0, "indexed": 0, "failed": 0, "error": "no retriever"}

        total = 0
        indexed = 0
        failed = 0

        try:

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


                if (i + 1) % batch_size == 0:
                    logger.info("migrate_to_vector: progress %d/%d", i + 1, total)

            logger.info("migrate_to_vector: completed. indexed=%d, failed=%d", indexed, failed)
        except Exception as e:
            logger.warning("migrate_to_vector failed: %s", e)
            return {"total": total, "indexed": indexed, "failed": failed, "error": str(e)}

        return {"total": total, "indexed": indexed, "failed": failed}



    def format_for_prompt(self, traces: list[ExperienceTrace]) -> str:
        """Implementation details."""
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
        """Implementation details."""
        if not trace.lessons_learned:
            return ""
        return f"来自 {trace.task_description} 的经验: {trace.lessons_learned}"



    def _extract_reasoning_steps(self, chat_history: list) -> list[dict]:
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
        if not tool_data:
            return ""
        lines = []
        for item in tool_data:
            if isinstance(item, dict):

                lines.append(f"{item['name']}: {item.get('result_preview', 'unknown error')[:200]}")
            else:

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
