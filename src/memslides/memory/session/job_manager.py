"""JobManager — Job / Round 生命周期管理（替代 SessionManager）

Design Doc §5.4: 管理 Job 的创建/结束/WM 绑定/Consolidation 触发。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..core.models import Job, Operation, Round, RoundExperience
from ..working_memory import WorkingMemory

if TYPE_CHECKING:
    from ..core.db import DatabaseBackend

logger = logging.getLogger(__name__)
CONSOLIDATION_TIMEOUT_S = float(os.getenv("MEMSLIDES_CONSOLIDATION_TIMEOUT", "0"))


class JobManager:
    """Job 生命周期管理器。

    职责：
    - start_job(): 创建 Job + 初始化 WM + DB 持久化
    - start_round(): 创建 Round 追加到 Job
    - record_operation(): 记录 Operation 到 Round
    - end_job(): 触发 Consolidation + 释放 WM
    """

    def __init__(self, db: DatabaseBackend, consolidator: Any = None,
                 workspace: Path | None = None, embedding_func: Any = None):
        self.db = db
        self._consolidator = consolidator
        self._workspace = workspace or Path(".")
        self._embedding_func = embedding_func
        self._active_job: Job | None = None
        self._working_memory: WorkingMemory | None = None
        self._migrated = False  # 迁移标记

    async def _ensure_intent_column(self) -> None:
        """确保 jobs 表有 intent 列（向后兼容旧数据库）"""
        if self._migrated:
            return
        try:
            rows = await self.db.query("PRAGMA table_info(jobs)", ())
            col_names = {r["name"] for r in rows} if rows else set()
            if col_names and "intent" not in col_names:
                # 旧表：添加 intent 列
                logger.info("Migrating jobs table: adding intent column")
                await self.db.execute(
                    "ALTER TABLE jobs ADD COLUMN intent TEXT DEFAULT ''", ()
                )
                logger.info("jobs table migration complete")
        except Exception as e:
            logger.debug("Intent column migration check: %s", e)
        self._migrated = True

    async def start_job(
        self,
        user_id: str,
        project_id: str,
        intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
    ) -> Job:
        """创建 Job + 初始化 WM + DB 持久化。

        Args:
            user_id: 用户 ID
            project_id: 项目 ID
            intent: 用户意图（可选，为空时在第一次 ProfileRouter 调用时推断）
            read_intent: 画像读取 intent
            write_intent: 画像写回 intent
            core_persona: 用户稳定人格
        """
        # 确保 jobs 表有 intent 列（自动迁移）
        await self._ensure_intent_column()

        job = Job(
            id=str(uuid4()),
            user_id=user_id,
            project_id=project_id,
            intent=intent,
            read_intent=read_intent or intent,
            write_intent=write_intent or intent or read_intent,
            core_persona=core_persona,
        )
        try:
            metadata = json.dumps(
                {
                    "read_intent": job.read_intent,
                    "write_intent": job.write_intent,
                    "core_persona": job.core_persona,
                },
                ensure_ascii=False,
            )
            await self.db.insert("jobs", {
                "id": job.id, "user_id": job.user_id,
                "project_id": job.project_id, "status": job.status,
                "started_at": job.started_at, "ended_at": job.ended_at,
                "intent": job.intent,
                "metadata": metadata,
            })
        except Exception as e:
            logger.warning(f"Failed to persist job to DB (non-fatal): {e}")
        self._active_job = job
        self._working_memory = WorkingMemory(job_id=job.id, embedding_func=self._embedding_func)
        # 检查并重试 pending consolidation
        await self._retry_pending_consolidation(user_id)
        logger.info(f"Job started: {job.id} (user={user_id}, project={project_id})")
        return job

    async def start_round(self, user_message: str) -> Round:
        """创建 Round 追加到当前 Job + DB 持久化。

        TODO(memory-migration): SQLite schema 暂保留历史 tasks 表名；
        新运行时通过本 adapter 将该表映射为 Round。
        """
        assert self._active_job is not None, "No active job"
        round_obj = Round(
            id=str(uuid4()), job_id=self._active_job.id,
            user_message=user_message,
        )
        self._active_job.rounds.append(round_obj)
        try:
            await self.db.insert("tasks", {
                "id": round_obj.id,
                "job_id": round_obj.job_id,
                "user_message": round_obj.user_message,
                "agent_response": "",
                "started_at": round_obj.started_at,
                "ended_at": None,
            })
        except Exception as e:
            logger.warning(f"Failed to persist round to DB (non-fatal): {e}")
        return round_obj

    def record_operation(self, round_obj: Round, tool_name: str, args: dict,
                         result: str, is_error: bool) -> None:
        """记录 Operation 到 Round（仅原始数据，工具链分割在 Round 结束时处理）。"""
        op = Operation(
            tool_name=tool_name,
            args_summary=str(args)[:200],
            result_summary=result[:200],
            is_error=is_error,
        )
        round_obj.operations.append(op)

    def record_operation_to_current_round(self, tool_name: str, args: dict,
                                          result: str, is_error: bool) -> None:
        """便捷方法：记录到当前 Round。"""
        if self._active_job and self._active_job.rounds:
            self.record_operation(
                self._active_job.rounds[-1], tool_name, args, result, is_error)

    async def end_round(self, round_id: str, agent_response: str = "") -> None:
        """结束 Round：更新 ended_at + agent_response。"""
        from datetime import datetime
        try:
            await self.db.execute(
                "UPDATE tasks SET ended_at = ?, agent_response = ? WHERE id = ?",
                (datetime.now().isoformat(), agent_response[:2000], round_id),
            )
        except Exception as e:
            logger.warning(f"Failed to update round end (non-fatal): {e}")

    async def end_job(
        self,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """结束 Job：触发 Consolidation + 释放 WM。

        Args:
            slide_html_dir: 最终 PPT 的 HTML 文件目录路径。
                           指纹+多模态分析仅在 Consolidation 时对该目录下的最终 HTML 执行。
            freeze_preference_writeback: 是否冻结偏好写回，仅保留非偏好归档。
        """
        if self._active_job is None:
            return {
                "status": "nothing_to_save",
                "saved": False,
                "consolidation_failed": False,
                "pending_consolidation_backup": "",
            }
        result: dict[str, Any] = {
            "status": "saved",
            "saved": True,
            "consolidation_failed": False,
            "pending_consolidation_backup": "",
        }
        try:
            if self._consolidator and self._working_memory:
                from ..progress import emit_memory_save_progress

                await emit_memory_save_progress(
                    progress_callback,
                    stage="job_consolidation",
                    progress=54,
                    message=(
                        "Starting job consolidation without a hard timeout."
                        if CONSOLIDATION_TIMEOUT_S <= 0
                        else f"Starting job consolidation (timeout {CONSOLIDATION_TIMEOUT_S:.0f}s)."
                    ),
                    timeout_s=CONSOLIDATION_TIMEOUT_S if CONSOLIDATION_TIMEOUT_S > 0 else None,
                )
                await self._run_consolidation(
                    job=self._active_job,
                    working_memory=self._working_memory,
                    slide_html_dir=slide_html_dir,
                    freeze_preference_writeback=freeze_preference_writeback,
                    progress_callback=progress_callback,
                )
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                message = (
                    f"Consolidation timed out after {CONSOLIDATION_TIMEOUT_S:.0f}s; "
                    "pending backup was written."
                )
                logger.warning(
                    "Consolidation timed out after %.0fs; continuing shutdown",
                    CONSOLIDATION_TIMEOUT_S,
                )
            else:
                message = str(e) or e.__class__.__name__
                logger.error(f"Consolidation failed: {e}")
            backup_path = self._save_backup(
                self._active_job,
                self._working_memory,
                slide_html_dir=slide_html_dir,
                freeze_preference_writeback=freeze_preference_writeback,
            )
            result = {
                "status": "failed",
                "saved": False,
                "consolidation_failed": True,
                "pending_consolidation_backup": str(backup_path) if backup_path else "",
                "message": message,
                "stage": "failed",
            }
        finally:
            if self._working_memory:
                self._working_memory.release()
                self._working_memory = None
            self._active_job = None
        return result

    async def write_preferences_first(
        self,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Persist preference/profile data before slower experience/tool memory.

        This protects user-facing personalization from later best-effort
        tool-chain or episode archival work.  The full job consolidation can
        then run with preference writeback frozen to avoid double writes.
        """
        if (
            self._active_job is None
            or self._working_memory is None
            or self._consolidator is None
            or not hasattr(self._consolidator, "consolidate_preferences_only")
        ):
            return {"status": "unavailable", "pref_written": 0}
        try:
            pref_written = await self._consolidator.consolidate_preferences_only(
                job=self._active_job,
                working_memory=self._working_memory,
                slide_html_dir=slide_html_dir,
                freeze_preference_writeback=freeze_preference_writeback,
                progress_callback=progress_callback,
                progress_start=8,
                progress_end=30,
            )
            return {
                "status": "frozen" if freeze_preference_writeback else "written",
                "pref_written": int(pref_written or 0),
            }
        except Exception as exc:
            logger.warning("Preference-first writeback failed: %s", exc)
            return {"status": "failed", "pref_written": 0, "message": str(exc)}

    async def retry_pending_consolidation(
        self,
        user_id: str = "",
        pending_backup: str = "",
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Retry one or more pending consolidation backup files."""
        if not self._consolidator:
            return {
                "status": "no_memory",
                "saved": False,
                "message": "Memory consolidator is not available.",
            }

        pending_files = self._pending_backup_files(pending_backup)
        if not pending_files:
            return {
                "status": "nothing_to_save",
                "saved": False,
                "message": "There is no pending memory backup to retry.",
            }

        from ..progress import emit_memory_save_progress

        completed = 0
        last_job_id = ""
        last_round_count = 0
        for index, backup_file in enumerate(pending_files, start=1):
            await emit_memory_save_progress(
                progress_callback,
                stage="retry_from_backup",
                progress=4,
                message=f"Retrying pending memory backup {index}/{len(pending_files)}.",
                pending_consolidation_backup=str(backup_file),
            )
            data = json.loads(backup_file.read_text(encoding="utf-8"))
            job, wm, freeze_preference_writeback = await self._restore_backup_payload(data)
            last_job_id = job.id
            last_round_count = len(getattr(job, "rounds", []) or [])
            try:
                await self._run_consolidation(
                    job=job,
                    working_memory=wm,
                    slide_html_dir=str(data.get("slide_html_dir") or ""),
                    freeze_preference_writeback=freeze_preference_writeback,
                    progress_callback=progress_callback,
                )
                wm.release()
                backup_file.unlink()
                completed += 1
                logger.info("Pending consolidation retried and removed: %s", backup_file.name)
            except Exception as exc:
                wm.release()
                logger.warning("Retry of %s failed: %s", backup_file.name, exc)
                return {
                    "status": "failed",
                    "saved": False,
                    "consolidation_failed": True,
                    "pending_consolidation_backup": str(backup_file),
                    "message": str(exc) or exc.__class__.__name__,
                    "job_id": last_job_id,
                    "round_count": last_round_count,
                    "completed_backups": completed,
                    "total_backups": len(pending_files),
                }

        return {
            "status": "saved",
            "saved": True,
            "consolidation_failed": False,
            "pending_consolidation_backup": "",
            "message": "Pending memory backup was saved to long-term memory.",
            "job_id": last_job_id,
            "round_count": last_round_count,
            "completed_backups": completed,
            "total_backups": len(pending_files),
        }

    async def _run_consolidation(
        self,
        *,
        job: Job,
        working_memory: WorkingMemory,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
    ) -> dict[str, Any] | None:
        coro = self._consolidator.consolidate(
            job=job,
            working_memory=working_memory,
            slide_html_dir=slide_html_dir,
            freeze_preference_writeback=freeze_preference_writeback,
            progress_callback=progress_callback,
        )
        if CONSOLIDATION_TIMEOUT_S > 0:
            return await asyncio.wait_for(coro, timeout=CONSOLIDATION_TIMEOUT_S)
        return await coro

    # ── Properties ──

    @property
    def working_memory(self) -> WorkingMemory | None:
        return self._working_memory

    def round_count(self) -> int:
        return len(self._active_job.rounds) if self._active_job else 0

    def current_round_id(self) -> str:
        if self._active_job and self._active_job.rounds:
            return self._active_job.rounds[-1].id
        return ""

    def is_active(self) -> bool:
        return self._active_job is not None

    # ── Backup / Recovery ──

    def _save_backup(
        self,
        job: Job,
        wm: WorkingMemory | None,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
    ) -> Path | None:
        """Consolidation 失败时保存备份。"""
        try:
            backup_dir = self._workspace / ".memory"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"pending_consolidation_{job.id}.json"
            backup_data = {
                "job": job.to_dict(),
                "slide_html_dir": slide_html_dir,
                "freeze_preference_writeback": freeze_preference_writeback,
                "preferences": [p.__dict__ for p in (wm._temp_preferences if wm else [])],
                "episodes": [e.to_dict() for e in (wm._temp_episodes if wm else [])],
                "round_experiences": [e.to_dict() for e in (wm.get_experiences() if wm else [])],
                "chains": {
                    name: [c.to_dict() for c in chains]
                    for name, chains in (wm.chain_buffer.get_all_chains().items() if wm else {})
                },
                "experiences": {
                    name: [e.to_dict() for e in exps]
                    for name, exps in (wm.chain_buffer.get_all_experiences().items() if wm else {})
                },
            }
            backup_path.write_text(json.dumps(backup_data, ensure_ascii=False, indent=2))
            logger.info(f"Backup saved: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to save backup: {e}")
            return None

    def _pending_backup_files(self, pending_backup: str = "") -> list[Path]:
        if pending_backup:
            path = Path(pending_backup)
            return [path] if path.exists() and path.is_file() else []
        pending_dir = self._workspace / ".memory"
        if not pending_dir.exists():
            return []
        return sorted(pending_dir.glob("pending_consolidation_*.json"))

    async def _restore_backup_payload(
        self,
        data: dict[str, Any],
    ) -> tuple[Job, WorkingMemory, bool]:
        job = Job.from_dict(data["job"])
        wm = WorkingMemory(job_id=job.id, embedding_func=self._embedding_func)
        from ..core.models import (
            ChainExperience,
            DesignEpisode,
            TempPreference,
            ToolChain,
        )

        for p in data.get("preferences", []):
            try:
                await wm.add_preference(TempPreference(**p), llm=None)
            except Exception:
                logger.debug("Failed to restore pending TempPreference", exc_info=True)
        for e in data.get("episodes", []):
            try:
                wm.add_episode(DesignEpisode.from_dict(e))
            except Exception:
                logger.debug("Failed to restore pending DesignEpisode", exc_info=True)
        for _sig, chains in data.get("chains", {}).items():
            for c in chains:
                try:
                    wm.chain_buffer.add_chain(ToolChain.from_dict(c))
                except Exception:
                    logger.debug("Failed to restore pending ToolChain", exc_info=True)
        for _sig, exps in data.get("experiences", {}).items():
            for e in exps:
                try:
                    wm.chain_buffer.add_experience(ChainExperience.from_dict(e))
                except Exception:
                    logger.debug("Failed to restore pending ChainExperience", exc_info=True)
        round_experiences = data.get("round_experiences")
        if round_experiences is None:
            round_experiences = data.get("task_experiences", [])
        for e in round_experiences:
            try:
                wm.add_experience(RoundExperience.from_dict(e))
            except Exception:
                logger.debug("Failed to restore pending RoundExperience", exc_info=True)

        freeze_preference_writeback = data.get("freeze_preference_writeback", False)
        if isinstance(freeze_preference_writeback, str):
            freeze_preference_writeback = freeze_preference_writeback.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            freeze_preference_writeback = bool(freeze_preference_writeback)
        return job, wm, freeze_preference_writeback

    async def _retry_pending_consolidation(self, user_id: str) -> None:
        """检测 pending 文件并重试 consolidation。"""
        if not self._consolidator:
            return
        try:
            pending_dir = self._workspace / ".memory"
            if not pending_dir.exists():
                return
            for f in self._pending_backup_files():
                logger.info(f"Retrying pending consolidation: {f.name}")
                try:
                    data = json.loads(f.read_text())
                    job, wm, freeze_preference_writeback = await self._restore_backup_payload(data)
                    await self._run_consolidation(
                        job=job,
                        working_memory=wm,
                        slide_html_dir=str(data.get("slide_html_dir") or ""),
                        freeze_preference_writeback=freeze_preference_writeback,
                    )
                    wm.release()
                    f.unlink()
                    logger.info(f"Pending consolidation retried and removed: {f.name}")
                except Exception as e:
                    logger.warning(f"Retry of {f.name} failed (will try again next session): {e}")
        except Exception as e:
            logger.warning(f"Pending consolidation check failed: {e}")
