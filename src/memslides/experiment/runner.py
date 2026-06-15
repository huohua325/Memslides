from __future__ import annotations

import asyncio
import json
import logging
import shutil
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memslides.experiment.config import ExperimentConfig, ExperimentSuite
from memslides.experiment.induct_template import induct_template
from memslides.experiment.metrics import ExperimentMetrics
from memslides.experiment.user_model import ReviewContext, UserModel
from memslides.session import MemSlidesSession
from memslides.utils.log import reset_context_logger

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    experiment_id: str
    output_dir: Path
    success: bool
    final_path: Path | None = None
    rounds_completed: int = 0
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["final_path"] = str(self.final_path) if self.final_path else ""
        return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _append_log(log_lines: list[str], text: str) -> None:
    if text:
        log_lines.append(text.rstrip())


def _latest_pptx_path(result: Any) -> Path | None:
    for attr in ("pptx_path", "final_path"):
        value = getattr(result, attr, None)
        if value:
            path = Path(value)
            if path.suffix.lower() == ".pptx" and path.is_file():
                return path
    return None


def _latest_output_path(result: Any) -> Path | None:
    if result is None:
        return None
    for attr in ("final_path", "pptx_path", "pdf_path"):
        value = getattr(result, attr, None)
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def _safe_clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, output_dir: Path, suite_id: str = ""):
        self.config = config
        self.output_dir = Path(output_dir)
        self.suite_id = suite_id

    async def run(self, resume: bool = False) -> ExperimentResult:
        reset_context_logger()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_lines: list[str] = []
        _append_log(
            log_lines,
            f"[experiment] {self.config.experiment_id} suite={self.suite_id} output={self.output_dir}",
        )

        checkpoint_path = self.output_dir / "checkpoint.json"
        done_path = self.output_dir / "done.json"
        spec_path = self.output_dir / "experiment_spec.json"
        _write_json(spec_path, self.config.model_dump(mode="json"))

        memory_db_dir = self.config.resolve_memory_db_dir(self.output_dir)
        if self.config.reset_memory_db:
            try:
                if memory_db_dir.is_relative_to(self.output_dir) or bool(
                    int(str((self.config.extra_info or {}).get("allow_destructive_reset", "0")) or "0")
                ):
                    _safe_clear_directory(memory_db_dir)
            except Exception:
                pass

        session = MemSlidesSession(
            options=self.config.build_session_options(
                workspace=self.output_dir,
                memory_db_dir=memory_db_dir,
            )
        )
        session.config.memory.artifact_trace = bool(self.config.memory_artifact_trace)

        metrics = ExperimentMetrics(self.output_dir)
        user_model = UserModel.from_config(
            self.config.resolved_user_model_profile(),
            self.config.user_model_strategy_config,
        )
        feedback_history: list[str] = []
        rounds_completed = 0
        final_result: Any | None = None

        try:
            await self._seed_templates(memory_db_dir)

            deck_request = self.config.build_deck_request()
            _append_log(log_lines, f"[generate] {deck_request.instruction[:120]}")
            if self.config.round_timeout:
                final_result = await asyncio.wait_for(
                    session.generate(deck_request),
                    timeout=self.config.round_timeout,
                )
            else:
                final_result = await session.generate(deck_request)
            if self.config.require_pptx_export and not _latest_pptx_path(final_result):
                raise RuntimeError(
                    f"Experiment {self.config.experiment_id} requires a PPTX artifact "
                    "but generation returned none."
                )
            self._record_result_artifacts(
                output_dir=self.output_dir,
                result=final_result,
                round_label="round_00_initial",
                log_lines=log_lines,
                runtime=session.runtime,
            )
            metrics.set_runtime_artifacts(
                resolved_intent=_safe_load_json(self.output_dir / "resolved_intent.json"),
                template_match=_safe_load_json(self.output_dir / "template_match.json"),
                validation=_safe_load_json(self.output_dir / "runtime_validation.json"),
            )
            metrics.record_round_from_artifacts(
                self.output_dir,
                0,
                deck_request.instruction,
                0.0,
                False,
                pptx_snapshot=str(_latest_pptx_path(final_result) or ""),
            )
            _write_json(
                checkpoint_path,
                {
                    "experiment_id": self.config.experiment_id,
                    "rounds_completed": rounds_completed,
                    "feedback_history": feedback_history,
                    "final_path": str(getattr(final_result, "final_path", "") or ""),
                },
            )

            latest_pptx = _latest_pptx_path(final_result)

            for round_index in range(1, max(0, int(self.config.max_rounds)) + 1):
                review_context = self._build_review_context(round_index, feedback_history)
                review = await user_model.review(review_context)
                _append_log(
                    log_lines,
                    f"[review {round_index}] score={review.score:.2f} satisfied={review.satisfied} feedback={review.feedback}",
                )

                rounds_completed = round_index
                if review.satisfied and not self.config.force_all_rounds:
                    if latest_pptx:
                        metrics.start_round()
                        metrics.record_round_from_artifacts(
                            self.output_dir,
                            round_index,
                            review.feedback,
                            review.score,
                            review.satisfied,
                            pptx_snapshot=str(latest_pptx),
                        )
                    feedback_history.append(review.feedback)
                    break

                feedback_history.append(review.feedback)
                metrics.start_round()
                revise_request = self.config.build_revision_request(review.feedback)
                if self.config.round_timeout:
                    final_result = await asyncio.wait_for(
                        session.revise(revise_request),
                        timeout=self.config.round_timeout,
                    )
                else:
                    final_result = await session.revise(revise_request)
                if self.config.require_pptx_export and not _latest_pptx_path(final_result):
                    raise RuntimeError(
                        f"Experiment {self.config.experiment_id} requires a PPTX artifact "
                        f"but revision round {round_index} returned none."
                    )
                latest_pptx = _latest_pptx_path(final_result)
                self._record_result_artifacts(
                    output_dir=self.output_dir,
                    result=final_result,
                    round_label=f"round_{round_index:02d}",
                    log_lines=log_lines,
                    runtime=session.runtime,
                )
                metrics.record_round_from_artifacts(
                    self.output_dir,
                    round_index,
                    review.feedback,
                    review.score,
                    review.satisfied,
                    pptx_snapshot=str(latest_pptx) if latest_pptx else "",
                )
                _write_json(
                    checkpoint_path,
                    {
                        "experiment_id": self.config.experiment_id,
                        "rounds_completed": rounds_completed,
                        "feedback_history": feedback_history,
                        "final_path": str(getattr(final_result, "final_path", "") or ""),
                    },
                )

            await session.close()
            summary = metrics.finalize()
            final_path = _latest_pptx_path(final_result)
            if final_path is None:
                final_path = _latest_output_path(final_result)
            _write_json(
                done_path,
                {
                    "success": True,
                    "experiment_id": self.config.experiment_id,
                    "output_dir": str(self.output_dir),
                    "rounds_completed": rounds_completed,
                    "final_path": str(final_path) if final_path else "",
                    "metrics_summary": summary,
                },
            )
            _append_log(log_lines, "[done] success")
            (self.output_dir / "conversation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return ExperimentResult(
                experiment_id=self.config.experiment_id,
                output_dir=self.output_dir,
                success=True,
                final_path=final_path,
                rounds_completed=rounds_completed,
                metrics_summary=summary,
            )
        except Exception as exc:
            error_text = f"{exc}\n{traceback.format_exc()}"
            logger.warning("Experiment %s failed: %s", self.config.experiment_id, exc)
            _append_log(log_lines, "[error]")
            _append_log(log_lines, error_text)
            try:
                await session.close()
            except Exception:
                pass
            try:
                _write_json(
                    done_path,
                    {
                        "success": False,
                        "experiment_id": self.config.experiment_id,
                        "output_dir": str(self.output_dir),
                        "rounds_completed": rounds_completed,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
            (self.output_dir / "conversation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return ExperimentResult(
                experiment_id=self.config.experiment_id,
                output_dir=self.output_dir,
                success=False,
                final_path=_latest_pptx_path(final_result),
                rounds_completed=rounds_completed,
                metrics_summary={},
                error=error_text,
            )
        finally:
            reset_context_logger()

    async def _seed_templates(self, memory_db_dir: Path) -> None:
        if not self.config.seed_templates:
            return
        seed_root = self.output_dir / "seed_templates"
        for seed in self.config.seed_templates:
            try:
                await induct_template(
                    seed.path,
                    user_id=self.config.user_id,
                    narrative_style=seed.narrative_style,
                    auto_infer_style=seed.auto_infer_style,
                    output_dir=seed_root / Path(seed.path).stem,
                    workspace=self.output_dir / ".template_induct" / Path(seed.path).stem,
                    memory_db_dir=memory_db_dir,
                    config_file=self.config.config_file,
                )
            except Exception as exc:
                logger.warning("Seed template induction failed for %s: %s", seed.path, exc)

    def _build_review_context(self, round_index: int, feedback_history: list[str]) -> ReviewContext:
        extra = self.config.extra_info or {}
        profile = self.config.resolved_user_model_profile()
        return ReviewContext(
            slide_image_paths=self._slide_previews(),
            round_num=round_index,
            instruction=self.config.instruction,
            feedback_history=list(feedback_history),
            task_intent=str(extra.get("resolved_task_intent") or extra.get("task_intent") or extra.get("intent") or self.config.memory_intent or ""),
            role_intent_id=str(extra.get("role_intent_id", "") or ""),
            role_intent_label_zh=str(extra.get("role_intent_label_zh", "") or ""),
            role_intent_prompt=str(extra.get("role_intent_prompt", "") or ""),
            task_primary_anchor=str(extra.get("task_primary_anchor", "") or ""),
            task_secondary_anchor=str(extra.get("task_secondary_anchor", "") or ""),
            memory_bucket_id=str(extra.get("memory_bucket_id", "") or ""),
            core_persona=str(extra.get("core_persona") or self.config.user_persona or profile.persona),
            read_intent=str(extra.get("memory_read_intent") or self.config.memory_intent or ""),
            write_intent=str(extra.get("memory_write_intent") or self.config.memory_intent or ""),
            shared_user_state={},
            user_role=str(extra.get("user_role") or self.config.user_persona or ""),
            role_preference_schema_id=str(extra.get("role_preference_schema_id", "") or ""),
            role_preference_profile_id=str(extra.get("role_preference_profile_id", "") or ""),
            role_preference_profile=self.config.resolved_user_model_profile().model_dump(),
            condition=str(extra.get("condition", "") or ""),
            zone=str(extra.get("zone", "") or ""),
            phase=str(extra.get("phase", "") or ""),
        )

    def _slide_previews(self) -> list[str]:
        try:
            session_preview_dir = self.output_dir
            previews = list((session_preview_dir).glob(".slide_images-pdf-*"))
            if not previews:
                return []
            previews.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            images = sorted(previews[0].glob("slide_*.jpg"))
            return [str(path) for path in images]
        except Exception:
            return []

    def _record_result_artifacts(
        self,
        *,
        output_dir: Path,
        result: Any,
        round_label: str,
        log_lines: list[str],
        runtime: Any | None = None,
    ) -> None:
        pptx_path = _latest_pptx_path(result)
        if pptx_path and pptx_path.is_file():
            target = output_dir / f"{round_label}.pptx"
            if target.resolve() != pptx_path.resolve():
                shutil.copy2(pptx_path, target)
            else:
                target = pptx_path
            _append_log(log_lines, f"[artifact] {target.name}")

        messages = list(getattr(result, "messages", []) or [])
        if messages:
            _append_log(log_lines, f"[messages] {len(messages)}")

        if runtime is not None:
            try:
                resolved_intent = runtime.get_resolved_intent_artifact()
                if resolved_intent:
                    _write_json(output_dir / "resolved_intent.json", resolved_intent)
            except Exception:
                pass
            try:
                template_match = runtime.get_template_match_artifact()
                if template_match:
                    _write_json(output_dir / "template_match.json", template_match)
            except Exception:
                pass


class ExperimentSuiteRunner:
    def __init__(self, suite: ExperimentSuite):
        self.suite = suite

    async def run_all(self, resume: bool = False) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        suite_output_dir = self.suite.suite_output_dir(reuse_existing=resume)
        suite_output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            suite_output_dir / "suite_config.json",
            self.suite.model_dump(mode="json"),
        )
        _write_json(
            suite_output_dir / "run_manifest.json",
            {
                "suite_id": self.suite.suite_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "output_base": str(Path(self.suite.output_base).expanduser()),
                "suite_output_dir": str(suite_output_dir),
                "default_log_file": str(suite_output_dir / ".history" / f"{self.suite.suite_id}.log"),
                "experiments": [
                    {
                        "experiment_id": config.experiment_id,
                        "output_dir": str(config.resolve_output_dir(suite_output_dir)),
                        "memory_db_dir": str(
                            config.resolve_memory_db_dir(config.resolve_output_dir(suite_output_dir))
                        ),
                        "template_file": str(config.template_file) if config.template_file else "",
                    }
                    for config in self.suite.experiments
                ],
            },
        )
        for config in self.suite.experiments:
            output_dir = config.resolve_output_dir(suite_output_dir)
            done_path = output_dir / "done.json"
            if resume and done_path.exists():
                try:
                    done = json.loads(done_path.read_text(encoding="utf-8"))
                    if isinstance(done, dict) and done.get("success"):
                        continue
                except Exception:
                    pass
            runner = ExperimentRunner(config, output_dir, suite_id=self.suite.suite_id)
            results.append(await runner.run(resume=resume))
        return results


def summarize_suite_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "experiments": [],
    }
    for done_path in sorted(output_dir.rglob("done.json")):
        try:
            data = json.loads(done_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        summary["experiments"].append(
            {
                "experiment_id": data.get("experiment_id", ""),
                "success": bool(data.get("success", False)),
                "rounds_completed": data.get("rounds_completed", 0),
                "output_dir": str(done_path.parent),
                "final_path": data.get("final_path", ""),
            }
        )
    summary["total"] = len(summary["experiments"])
    summary["successful"] = sum(1 for item in summary["experiments"] if item["success"])
    return summary


__all__ = [
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSuiteRunner",
    "summarize_suite_output",
]
