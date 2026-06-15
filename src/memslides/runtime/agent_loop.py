import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import Counter
from collections.abc import AsyncGenerator
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from memslides.agents.deck_designer import DeckDesigner
from memslides.agents.env import AgentEnv
from memslides.agents.revision_editor import RevisionEditor
from memslides.agents.template_planner import TemplatePlanner
from memslides.agents.researcher import Researcher
from memslides.agents.agent import RoleToolContractError
from memslides.memory.session.session import InteractiveSession
from memslides.utils.config import GLOBAL_CONFIG, MemSlidesConfig
from memslides.utils.constants import (
    FORCE_FINALIZE_MSG,
    MAX_MODIFY_ITERATIONS,
    WORKSPACE_BASE,
)
from memslides.utils.log import debug, error, info, set_logger, timer, warning
from memslides.utils.typings import ChatMessage, ConvertType, InputRequest, Role
from memslides.utils.webview import (
    PlaywrightConverter,
    convert_html_to_pptx_with_retry,
)
from memslides.tools.deck_runtime import (
    initialize_design_plan_tracking,
    set_current_agent,
)
from memslides.pipelines.generation_support import (
    build_profile_realization_report as _build_profile_realization_report,
    build_non_template_design_plan_scaffold as _build_non_template_design_plan_scaffold,
    build_profile_execution_source_evidence_summary as _build_profile_execution_source_evidence_summary,
    find_existing_design_plan_rel as _find_existing_design_plan_rel,
    normalize_bool_flag as _normalize_bool_flag,
    normalize_memory_intent as _normalize_memory_intent,
    normalize_text_flag as _normalize_text_flag,
    render_design_plan_execution_plan as _render_design_plan_execution_plan,
    render_profile_execution_contract_markdown as _render_profile_execution_contract_markdown,
    render_profile_execution_page_obligations_markdown as _render_profile_execution_page_obligations_markdown,
    resolve_request_memory_intent as _resolve_request_memory_intent,
)
from . import support as _runtime_support
from .support import IntentResolutionResult, ModifyExecutionPlan

logger = logging.getLogger(__name__)


def _wm_payload_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _mtime_iso(path: Path) -> str:
    from datetime import datetime

    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except Exception:
        return ""


def _round_count(rounds_root: Path) -> int:
    try:
        return sum(1 for item in rounds_root.glob("round_*") if item.is_dir())
    except Exception:
        return 0


def _extract_memory_injection_section(text: str, tag: str) -> str:
    if not text or not tag:
        return ""
    pattern = re.compile(rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _bind_runtime_method(func):
    def _method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return _method


def _render_modify_execution_plan_bridge(self: "AgentLoop", plan: ModifyExecutionPlan) -> str:
    return _runtime_support.render_modify_execution_plan(plan, runtime=self)


def _render_modify_tool_policy_plan_bridge(self: "AgentLoop", policy: Any) -> str:
    return _runtime_support.render_modify_tool_policy_plan(policy, runtime=self)


class _AgentLoopRuntimeBindings:
    build_user_state_snapshot = _bind_runtime_method(_runtime_support.build_user_state_snapshot)
    _resolve_request_intent_runtime = _bind_runtime_method(_runtime_support.resolve_request_intent_runtime)
    _cache_resolved_request_intent = _bind_runtime_method(_runtime_support.cache_resolved_request_intent)
    _get_runtime_request_intent = _bind_runtime_method(_runtime_support.get_runtime_request_intent)
    get_resolved_intent_artifact = _bind_runtime_method(_runtime_support.get_resolved_intent_artifact)
    get_template_match_artifact = _bind_runtime_method(_runtime_support.get_template_match_artifact)
    _initialize_template_match_state = _bind_runtime_method(_runtime_support.initialize_template_match_state)
    _update_template_match_state = _bind_runtime_method(_runtime_support.update_template_match_state)

    _template_user_candidates = _bind_runtime_method(_runtime_support.template_user_candidates)
    _normalize_template_lookup_text = staticmethod(_runtime_support.normalize_template_lookup_text)
    _resolve_template_from_selection = staticmethod(_runtime_support.resolve_template_from_selection)
    _list_accessible_templates = _bind_runtime_method(_runtime_support.list_accessible_templates)
    _pick_template_by_name_hint = staticmethod(_runtime_support.pick_template_by_name_hint)
    _filter_template_preferences_for_matching = staticmethod(_runtime_support.filter_template_preferences_for_matching)
    _template_profile_context_from_profile = staticmethod(_runtime_support.template_profile_context_from_profile)
    _build_runtime_template_summary_from_usage_history = staticmethod(
        _runtime_support.build_runtime_template_summary_from_usage_history
    )
    _load_usage_history_for_template_memory = _bind_runtime_method(_runtime_support.load_usage_history_for_template_memory)
    _sync_template_profile_from_usage_history = _bind_runtime_method(_runtime_support.sync_template_profile_from_usage_history)
    _search_templates_by_style = _bind_runtime_method(_runtime_support.search_templates_by_style)
    _search_templates_by_query = _bind_runtime_method(_runtime_support.search_templates_by_query)
    _get_recent_template = _bind_runtime_method(_runtime_support.get_recent_template)
    _auto_match_template_profile = _bind_runtime_method(_runtime_support.auto_match_template_profile)
    _activate_template_profile = _bind_runtime_method(_runtime_support.activate_template_profile)
    _record_template_usage = _bind_runtime_method(_runtime_support.record_template_usage)
    _record_explicit_template_seed_usage = _bind_runtime_method(_runtime_support.record_explicit_template_seed_usage)

    _session_preference_has_general_signal = staticmethod(
        _runtime_support.session_preference_has_general_signal
    )
    _infer_session_preference_retention_scope = staticmethod(
        _runtime_support.infer_session_preference_retention_scope
    )
    _infer_session_preference_category = staticmethod(
        _runtime_support.infer_session_preference_category
    )
    _slugify_session_rule = staticmethod(_runtime_support.slugify_session_rule)
    _infer_session_rule_element_kind = staticmethod(_runtime_support.infer_session_rule_element_kind)
    _infer_session_rule_type = staticmethod(_runtime_support.infer_session_rule_type)
    _build_session_rule_spec = staticmethod(_runtime_support.build_session_rule_spec)
    _normalize_session_preference_payload = staticmethod(
        _runtime_support.normalize_session_preference_payload
    )
    _append_session_preference = _bind_runtime_method(_runtime_support.append_session_preference)
    _dump_current_round_wm_snapshot = _bind_runtime_method(_runtime_support.dump_current_round_wm_snapshot)
    _dump_manual_wm_snapshot = _bind_runtime_method(_runtime_support.dump_manual_wm_snapshot)
    record_review_feedback_preference = _bind_runtime_method(_runtime_support.record_review_feedback_preference)
    _hydrate_active_wm_from_latest_snapshot = _bind_runtime_method(
        _runtime_support.hydrate_active_wm_from_latest_snapshot
    )

    _ensure_modifyagent_loaded = _bind_runtime_method(_runtime_support.ensure_modifyagent_loaded)
    _resolve_export_repair_agent = _bind_runtime_method(_runtime_support.resolve_export_repair_agent)
    _extract_export_failure_slide = staticmethod(_runtime_support.extract_export_failure_slide)
    _build_export_repair_message = _bind_runtime_method(_runtime_support.build_export_repair_message)
    _run_slide_export_repair = _bind_runtime_method(_runtime_support.run_slide_export_repair)
    _export_slides_with_agent_repair = _bind_runtime_method(_runtime_support.export_slides_with_agent_repair)

    _build_modify_execution_plan = _bind_runtime_method(_runtime_support.build_modify_execution_plan)
    _render_modify_execution_plan = _render_modify_execution_plan_bridge
    _resolve_modify_coverage_path = _bind_runtime_method(_runtime_support.resolve_modify_coverage_path)
    _extract_modify_coverage_paths = _bind_runtime_method(_runtime_support.extract_modify_coverage_paths)
    _build_modify_plan_followup = _bind_runtime_method(_runtime_support.build_modify_plan_followup)
    _build_modify_no_mutation_followup = _bind_runtime_method(
        _runtime_support.build_modify_no_mutation_followup
    )
    _collect_pending_inserted_slides = _bind_runtime_method(
        _runtime_support.collect_pending_inserted_slides
    )
    _build_inserted_slide_completion_followup = _bind_runtime_method(
        _runtime_support.build_inserted_slide_completion_followup
    )
    _collect_future_slide_preference_failures = _bind_runtime_method(
        _runtime_support.collect_future_slide_preference_failures
    )
    _collect_future_slide_preference_failures_async = _bind_runtime_method(
        _runtime_support.collect_future_slide_preference_failures_async
    )
    _build_future_preference_followup = _bind_runtime_method(
        _runtime_support.build_future_preference_followup
    )
    _future_preference_verifiable_on_single_slide = staticmethod(
        _runtime_support.future_preference_verifiable_on_single_slide
    )
    _render_modify_tool_policy_plan = _render_modify_tool_policy_plan_bridge


class AgentLoop(_AgentLoopRuntimeBindings):
    def __init__(
        self,
        config: MemSlidesConfig = GLOBAL_CONFIG,
        session_id: str | None = None,
        workspace: Path = None,
        language: Literal["zh", "en"] = "en",
        user_id: str | None = None,
    ):
        self.config = config
        self.language = language
        # Store user_id for memory system (strip whitespace, handle empty strings)
        _user_id = (user_id or "").strip()
        self.user_id = _user_id if _user_id else "default"
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.workspace = workspace or WORKSPACE_BASE / session_id
        self.intermediate_output = {}
        self.agent = None
        self.agent_env: AgentEnv | None = None
        self._env_owned = False
        self._modify_turn_count = 0
        # Demo hook: when feedback is already written right after review,
        # skip exactly one pre-modify priming to avoid duplicate WM writes.
        self._skip_next_session_preference_priming = False
        # Job-local fallback for extracted session preferences when the
        # MemoryOrchestrator / WM path is unavailable in a resumed modify run.
        self._session_preference_fallback_cache: list[dict[str, Any]] = []
        self.designagent: DeckDesigner | None = None
        self.modifyagent: RevisionEditor | None = None
        self.research_agent: Researcher | None = None
        self.template_planner: TemplatePlanner | None = None
        self._last_request: InputRequest | None = None
        self.memory_system: Any | None = None
        self._session: InteractiveSession | None = None
        # Clarification flow state (yield-resume pattern)
        self._clarification_answer: str | None = None
        # Stage 7: 模板状态持久化（供 modify() 访问）
        self._template_profile: Any = None
        self._guide_builder: Any = None
        self._template_runtime_state: Any = None
        # Stage 9: 模板记忆关联
        self._current_template_id: str = ""  # 当前 run 使用的模板 ID
        self._template_selection_manager: Any = None  # TemplateSelectionManager
        # Scope-aware preference: 跨轮次保存上一轮的 intent context，供 SYSTEM 注入使用
        self._last_modify_context: dict = {}  # {"slide_type": ..., "element_type": ..., "template_id": ...}
        # MemoryOrchestrator instance (created lazily by _create_orchestrator)
        self._memory_orchestrator_instance: Any | None = None
        self._memory_job_needs_start: bool = False
        self._last_memory_job_start: dict[str, str] = {}
        self.memory_runtime: Any | None = None
        # Deferred on_job_end: slide_html_dir saved by run() for close_env() to finalize
        self._deferred_slide_html_dir: str = ""
        self._freeze_preference_writeback_current_job: bool = False
        # Cache the latest SYSTEM injection state so template prompt updates can refresh trace files.
        self._system_injection_trace_state: dict[tuple[str, int], dict[str, Any]] = {}
        self._resolved_request_intent: str = ""
        self._resolved_request_intent_scenario: str = ""
        self._resolved_request_intent_source: str = ""
        self._resolved_request_intent_confidence: float | None = None
        self._resolved_request_intent_raw_response: str = ""
        self._resolved_request_intent_payload: dict[str, Any] = {}
        self._template_match_state: dict[str, Any] = {}
        self._template_usage_seeded: bool = False
        set_logger(
            f"memslides-loop-{self.workspace.stem}",
            self.workspace / ".history" / "memslides-loop.log",
        )
        debug(f"Initialized AgentLoop with workspace={self.workspace}")
        debug(f"AgentLoop user_id: {self.user_id}")
        debug(
            "Config summary: language=%s, memory_enabled=%s, offline_mode=%s, "
            "mcp_config_file=%s",
            self.language,
            bool(getattr(getattr(self.config, "memory", None), "enabled", False)),
            bool(getattr(self.config, "offline_mode", False)),
            getattr(self.config, "mcp_config_file", ""),
        )

    async def _export_pdf_best_effort(
        self,
        html_files: list[Path],
        output_pdf: Path,
        aspect_ratio: str,
        context_label: str,
    ) -> bool:
        try:
            async with PlaywrightConverter() as pc:
                await pc.convert_to_pdf(
                    html_files,
                    output_pdf,
                    aspect_ratio=aspect_ratio,
                )
            return True
        except Exception as e:
            warning(f"PDF export skipped for {context_label} (non-fatal): {e}")
            return False

    def _workspace_relative_label(self, path: Path | str) -> str:
        return _runtime_support.workspace_relative_label(self, path)

    def _resolve_workspace_context_slide_dir(
        self,
        slide_dir: Path | str | None = None,
    ) -> Path | None:
        return _runtime_support.resolve_workspace_context_slide_dir(self, slide_dir)

    def _prime_design_slide_output_dir(
        self,
        manuscript_path: Path | str | None = None,
    ) -> Path:
        return _runtime_support.prime_design_slide_output_dir(self, manuscript_path)

    def _build_workspace_context_block(
        self,
        *,
        slide_dir: Path | str | None = None,
        manuscript_path: Path | str | None = None,
        include_active_slide_dir: bool = False,
    ) -> str:
        return _runtime_support.build_workspace_context_block(
            self,
            slide_dir=slide_dir,
            manuscript_path=manuscript_path,
            include_active_slide_dir=include_active_slide_dir,
        )

    @property
    def has_memory(self) -> bool:
        """Check if memory system with DB is available."""
        return bool(self.memory_system and getattr(self.memory_system, 'db', None))

    @property
    def _job_mgr(self):
        """Get JobManager from orchestrator (primary) or memory_system (fallback)."""
        if self._memory_orchestrator_instance:
            return self._memory_orchestrator_instance._job_mgr
        if self.memory_system:
            return getattr(self.memory_system, 'job_manager', None) or getattr(self.memory_system, '_job_mgr', None)
        return None

    def _make_exp_writer(self):
        """Create an ExperienceTraceWriter from the current memory system.

        Returns None if memory system is not available.
        """
        if not self.has_memory:
            return None
        from memslides.memory.experience_writer import ExperienceTraceWriter
        _retriever = getattr(self.memory_system, 'retriever', None) or getattr(self.memory_system, 'hybrid_retriever', None)
        return ExperienceTraceWriter(
            self.memory_system.db,
            llm=getattr(self.memory_system, 'llm', None),
            retriever=_retriever,
        )

    def _create_orchestrator(self):
        """Create MemoryOrchestrator from MemorySystem components.

        Returns the orchestrator instance, or None if creation fails.
        The instance is cached on self._memory_orchestrator_instance.
        """
        if self._memory_orchestrator_instance is not None:
            return self._memory_orchestrator_instance
        if not self.memory_system:
            return None
        try:
            from memslides.memory.orchestrator import MemoryOrchestrator
            from memslides.memory.evolution.consolidator import Consolidator
            from memslides.memory.session.job_manager import JobManager
            from memslides.memory.store.chain_store import ChainStore
            from memslides.memory.extract.chain_segmenter import ChainSegmenter

            _db = getattr(self.memory_system, 'db', None)
            _llm = getattr(self.memory_system, 'llm', None)
            _llm_objects = getattr(self.memory_system, 'llm_objects_by_task', None) or {}
            _embedding = getattr(self.memory_system, 'embedding_func', None)
            _offline_consolidator = getattr(self.memory_system, 'memory_consolidator', None)
            _mem_cfg = getattr(self.memory_system, 'config', None)
            _mods = getattr(_mem_cfg, 'modules', None)

            chain_store = ChainStore(_db) if _db else None
            chain_segmenter = ChainSegmenter(llm=_llm)
            runtime_consolidator = Consolidator(
                llm=_llm,
                vlm=None,
                experience_writer=self._make_exp_writer(),
                preference_store=getattr(self.memory_system, 'preference_store', None),
                profile_store=getattr(self.memory_system, 'profile_store', None),
                episode_store=getattr(self.memory_system, 'episode_store', None),
                offline_consolidator=_offline_consolidator,
                workspace=self.workspace,
                embedding_func=_embedding,
                chain_store=chain_store,
                artifact_dumper=getattr(self.memory_system, 'artifact_dumper', None),
                enable_atomic_preference_writeback=getattr(_mods, 'atomic_preference_writeback', True),
                enable_profile_writeback=getattr(_mods, 'profile_writeback', True),
                enable_chain_experience_writeback=getattr(_mods, 'chain_experience_writeback', True),
            )
            job_mgr = JobManager(
                db=_db,
                consolidator=runtime_consolidator,
                workspace=self.workspace,
                embedding_func=_embedding,
            )

            orch = MemoryOrchestrator(
                db=_db,
                job_manager=job_mgr,
                tool_memory_retriever=self._make_exp_writer(),
                profile_store=getattr(self.memory_system, 'profile_store', None),
                user_state_resolver=getattr(self.memory_system, 'user_state_resolver', None),
                collector=getattr(self.memory_system, 'collector', None),
                preference_extractor=getattr(self.memory_system, 'preference_extractor', None),
                chain_segmenter=chain_segmenter,
                episode_extractor=getattr(self.memory_system, 'episode_extractor', None),
                llm=_llm,
                profile_injection_llm=_llm_objects.get('profile_injection') or _llm,
                embedding_func=_embedding,
                chain_store=chain_store,
                artifact_dumper=getattr(self.memory_system, 'artifact_dumper', None),
                enable_profile_injection=getattr(_mods, 'profile_injection', True),
                enable_wm_preference_collection=getattr(_mods, 'wm_preference_collection', True),
                enable_wm_preference_injection=getattr(_mods, 'wm_preference_injection', True),
                enable_wm_experience_collection=getattr(_mods, 'wm_experience_collection', True),
                enable_wm_experience_injection=getattr(_mods, 'wm_experience_injection', True),
                enable_chain_experience_collection=getattr(_mods, 'chain_experience_collection', True),
                enable_wm_round_history_injection=getattr(
                    _mods,
                    'wm_round_history_injection',
                    getattr(_mods, 'wm_task_history_injection', False),
                ),
                enable_ltm_tool_experience_injection=getattr(_mods, 'ltm_tool_experience_injection', True),
                enable_experience_preload=getattr(_mods, 'experience_preload', True),
                enable_round_experience_writeback=getattr(
                    _mods,
                    'round_experience_writeback',
                    getattr(_mods, 'task_experience_writeback', True),
                ),
            )
            self._memory_orchestrator_instance = orch
            try:
                setattr(self.memory_system, "orchestrator", orch)
            except Exception:
                pass
            if self.memory_runtime is not None and hasattr(self.memory_runtime, "bind_orchestrator"):
                self.memory_runtime.bind_orchestrator(orch)
            logger.info("MemoryOrchestrator created successfully")
            return orch
        except Exception as e:
            logger.warning(f"Failed to create MemoryOrchestrator: {e}")
            return None

    def _memory_active_job_state(self) -> dict[str, Any]:
        orchestrator = self._memory_orchestrator_instance
        job_mgr = getattr(orchestrator, "_job_mgr", None) if orchestrator else None
        active_job = getattr(job_mgr, "_active_job", None) if job_mgr is not None else None
        working_memory = getattr(job_mgr, "working_memory", None) if job_mgr is not None else None
        return {
            "orchestrator": orchestrator,
            "job_manager": job_mgr,
            "active_job": active_job,
            "working_memory": working_memory,
            "has_active_job": active_job is not None and working_memory is not None,
        }

    def get_working_memory_snapshot(self) -> dict[str, Any]:
        """Return a read-only summary of the live job-local working memory."""
        state = self._memory_active_job_state()
        orchestrator = state.get("orchestrator")
        active_job = state.get("active_job")
        wm = state.get("working_memory")
        base: dict[str, Any] = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "active": bool(state.get("has_active_job")),
            "job": None,
            "counts": {
                "preferences": 0,
                "active_preferences": 0,
                "structured_rules": 0,
                "experiences": 0,
                "rounds": 0,
            },
            "preferences": [],
            "structured_rules": [],
            "experiences": [],
            "general_rules_text": "",
            "structured_rules_text": "",
            "preferences_text": "",
            "message": "",
        }
        if not state.get("has_active_job") or wm is None:
            saved = self._latest_saved_working_memory_snapshot(base)
            if saved is not None:
                return saved
            base["message"] = "No session working memory yet. Saved long-term memory lives in the user profile/LTM stores."
            return base

        rounds = list(getattr(active_job, "rounds", []) or [])
        raw_preferences = list(getattr(wm, "_temp_preferences", []) or [])
        active_preferences = [pref for pref in raw_preferences if not bool(getattr(pref, "superseded", False))]
        raw_experiences = list(wm.get_experiences() if hasattr(wm, "get_experiences") else getattr(wm, "_temp_experiences", []) or [])
        structured_rule_specs = _runtime_support.collect_active_structured_wm_rule_specs(self, orchestrator)
        structured_rules = structured_rule_specs or [
            rule
            for rule in (
                self._working_memory_rule_from_preference(pref, index)
                for index, pref in enumerate(active_preferences)
            )
            if rule is not None
        ]
        base.update(
            {
                "snapshot_source": "live",
                "job": {
                    "id": str(getattr(active_job, "id", "") or ""),
                    "intent": str(getattr(active_job, "intent", "") or ""),
                    "read_intent": str(getattr(active_job, "read_intent", "") or ""),
                    "write_intent": str(getattr(active_job, "write_intent", "") or ""),
                    "started_at": str(getattr(active_job, "started_at", "") or ""),
                    "round_count": len(rounds),
                },
                "counts": {
                    "preferences": len(raw_preferences),
                    "active_preferences": len(active_preferences),
                    "structured_rules": len(structured_rules),
                    "experiences": len(raw_experiences),
                    "rounds": len(rounds),
                },
                "preferences": [
                    self._working_memory_preference_payload(pref, index)
                    for index, pref in enumerate(raw_preferences)
                ],
                "structured_rules": structured_rules,
                "experiences": [
                    self._working_memory_experience_payload(exp)
                    for exp in raw_experiences[-12:]
                ],
                "message": "Live session working memory is available for upcoming rounds.",
            }
        )
        if orchestrator is not None:
            try:
                base["general_rules_text"] = str(orchestrator.get_wm_general_rules_text() or "")
                base["structured_rules_text"] = _runtime_support.format_rule_specs_text(
                    structured_rule_specs,
                    source="working_memory_snapshot",
                    note="Merged active structured WM rules for UI and next-round injection.",
                ) if structured_rule_specs else str(orchestrator.get_wm_structured_rules_text() or "")
                base["preferences_text"] = str(orchestrator.get_wm_preferences_text(relevant_dims=None, include_general=False) or "")
            except Exception:
                logger.debug("Failed to collect working-memory injection text", exc_info=True)
        _runtime_support.persist_live_wm_snapshot(self, reason="get_working_memory_snapshot")
        return base

    async def update_working_memory_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update editable session WM entries while preserving seeded and carryover items."""
        from memslides.memory.core.models import RoundExperience, TempPreference
        from memslides.memory.inject.profile_injection_router import LTM_INJECT_SOURCE

        if not self.memory_system:
            raise RuntimeError("Memory is not enabled for this session.")
        orchestrator = self._create_orchestrator()
        if orchestrator is None:
            raise RuntimeError("Working memory is not available for this session.")
        state = self._memory_active_job_state()
        if not state.get("has_active_job"):
            await self._ensure_memory_job_started(
                user_prompt="Manual working-memory edit",
                intent=self._resolved_request_intent or "",
                read_intent=self._resolved_request_intent or "",
                write_intent=self._resolved_request_intent or "",
                reason="manual_working_memory_edit",
            )
            state = self._memory_active_job_state()

        wm = state.get("working_memory")
        if wm is None:
            raise RuntimeError("Working memory could not be started for this session.")

        preferences_payload = payload.get("preferences")
        experiences_payload = payload.get("experiences")
        if not isinstance(preferences_payload, list):
            raise RuntimeError("Working memory preferences must be a list.")
        if experiences_payload is not None and not isinstance(experiences_payload, list):
            raise RuntimeError("Working memory experiences must be a list.")

        next_preferences: list[TempPreference] = []
        for item in preferences_payload[:200]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            scope = str(item.get("scope") or "global").strip()
            if scope not in {"global", "slide_type", "element_type"}:
                scope = "global"
            structured_data = item.get("structured_data")
            next_preferences.append(
                TempPreference(
                    content=content[:4000],
                    dimension=str(item.get("dimension") or "general").strip()[:120],
                    preference_type=str(item.get("preference_type") or "value").strip()[:40],
                    source_task_id=str(item.get("source_task_id") or "manual_wm_edit").strip()[:160],
                    scope=scope,
                    scope_value="" if scope == "global" else str(item.get("scope_value") or "").strip()[:160],
                    superseded=bool(item.get("superseded", False)),
                    structured_data=deepcopy(structured_data) if isinstance(structured_data, dict) else None,
                    timestamp=str(item.get("timestamp") or ""),
                )
            )

        next_experiences: list[RoundExperience] = []
        for item in (experiences_payload or [])[:100]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            try:
                next_experiences.append(
                    RoundExperience(
                        id=str(item.get("id") or uuid.uuid4()),
                        content=content[:4000],
                        tool_name=str(item.get("tool_name") or "").strip()[:120],
                        keywords=[str(value).strip()[:80] for value in (item.get("keywords") or []) if str(value).strip()][:20],
                        category=str(item.get("category") or "generic").strip()[:80],
                        source=str(item.get("source") or "manual_edit").strip()[:80],
                        source_task_id=str(item.get("source_task_id") or "manual_wm_edit").strip()[:160],
                        confidence=float(item.get("confidence") if item.get("confidence") not in ("", None) else 0.8),
                        timestamp=str(item.get("timestamp") or ""),
                        outcome=str(item.get("outcome") or "").strip()[:80],
                    )
                )
            except Exception:
                logger.debug("Skipping invalid manually edited WM experience", exc_info=True)

        preserved_preferences: list[TempPreference] = []
        for existing in list(getattr(wm, "_temp_preferences", []) or []):
            source_task_id = str(getattr(existing, "source_task_id", "") or "")
            if source_task_id == LTM_INJECT_SOURCE or source_task_id in {
                "saved_structured_rule_carryover",
                "live_snapshot_structured_rule_carryover",
            }:
                preserved_preferences.append(existing)

        wm._temp_preferences = [*preserved_preferences, *next_preferences]
        wm._temp_experiences = next_experiences
        if hasattr(wm, "_normalize_active_general_preferences"):
            wm._normalize_active_general_preferences()

        if getattr(orchestrator, "_round_cache", None) is not None:
            try:
                orchestrator._round_cache.refresh_wm_preferences()
            except Exception:
                logger.debug("Failed to refresh WM preference cache after manual edit", exc_info=True)

        dumped = self._dump_current_round_wm_snapshot() or self._dump_manual_wm_snapshot()
        snapshot = self.get_working_memory_snapshot()
        snapshot["message"] = "Working memory was updated manually and will be used by the next round."
        snapshot["manual_edit_saved"] = True
        snapshot["snapshot_dumped"] = dumped
        return snapshot

    def _latest_saved_working_memory_snapshot(self, base: dict[str, Any]) -> dict[str, Any] | None:
        memory_root = self.workspace / ".memory"
        rounds_root = memory_root / "rounds"
        candidates: list[Path] = []
        carryover_candidates: list[Path] = []
        live_snapshot = memory_root / _runtime_support.LIVE_WM_SNAPSHOT_FILENAME
        if live_snapshot.exists():
            snapshot_path = live_snapshot
            if rounds_root.exists():
                carryover_candidates = sorted(
                    rounds_root.glob("round_*/wm_snapshot.json"),
                    key=lambda item: item.stat().st_mtime if item.exists() else 0,
                    reverse=True,
                )
        elif rounds_root.exists():
            candidates.extend(rounds_root.glob("round_*/wm_snapshot.json"))
            if not candidates:
                return None
            candidates = sorted(
                candidates,
                key=lambda item: item.stat().st_mtime if item.exists() else 0,
                reverse=True,
            )
            snapshot_path = candidates[0]
            carryover_candidates = candidates[1:]
        else:
            return None
        if not candidates:
            candidates = [snapshot_path]
        try:
            raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read saved working-memory snapshot: %s", snapshot_path, exc_info=True)
            return None
        if not isinstance(raw, dict):
            return None

        raw_preferences = list(raw.get("temp_preferences") or [])
        round_dir = snapshot_path.parent
        is_live_session_snapshot = snapshot_path.name == _runtime_support.LIVE_WM_SNAPSHOT_FILENAME
        structured_rule_ids = {
            _runtime_support._structured_rule_identity(pref.get("structured_data"))
            for pref in raw_preferences
            if isinstance(pref, dict)
            and not bool(pref.get("superseded", False))
            and isinstance(pref.get("structured_data"), dict)
        }
        structured_rule_ids.discard("")
        if not is_live_session_snapshot and rounds_root.exists():
            for older_snapshot in carryover_candidates:
                older_raw = _read_json_dict(older_snapshot)
                for pref in older_raw.get("temp_preferences", []) or []:
                    if not isinstance(pref, dict) or bool(pref.get("superseded", False)):
                        continue
                    rule_key = _runtime_support._structured_rule_identity(pref.get("structured_data"))
                    if not rule_key or rule_key in structured_rule_ids:
                        continue
                    carried = deepcopy(pref)
                    structured_data = carried.get("structured_data")
                    if isinstance(structured_data, dict):
                        structured_data = deepcopy(structured_data)
                        structured_data["source"] = str(structured_data.get("source") or "saved_round_snapshot")
                        structured_data["restored_from_prior_snapshot"] = True
                        carried["structured_data"] = structured_data
                    raw_preferences.append(carried)
                    structured_rule_ids.add(rule_key)
        saved_specs = _runtime_support.collect_saved_structured_wm_rule_specs(self)
        if saved_specs:
            raw_preferences = _runtime_support.merge_structured_rule_specs_into_temp_preferences(
                raw_preferences,
                saved_specs,
                timestamp=str(raw.get("saved_at") or ""),
                source_task_id="saved_structured_rule_carryover",
            )
        active_preferences = [pref for pref in raw_preferences if not bool(_wm_payload_get(pref, "superseded", False))]
        raw_experiences = list(raw.get("temp_experiences") or [])
        structured_rules = [
            rule
            for rule in (
                self._working_memory_rule_from_preference(pref, index)
                for index, pref in enumerate(active_preferences)
            )
            if rule is not None
        ]
        if saved_specs:
            saved_text = _runtime_support.format_rule_specs_text(
                saved_specs,
                source="saved_working_memory_snapshot",
                note="Merged active structured WM rules restored from live and round snapshots.",
            )
        else:
            saved_text = ""
        round_summary = {} if is_live_session_snapshot else _read_json_dict(round_dir / "round_summary.json")
        round_meta = {} if is_live_session_snapshot else _read_json_dict(round_dir / "round_meta.json")
        memory_injection = "" if is_live_session_snapshot else _read_text_file(round_dir / "memory_injection.txt")
        raw_job = raw.get("job") if isinstance(raw.get("job"), dict) else {}
        raw_counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}
        saved_at = _mtime_iso(snapshot_path)
        round_count = int(
            raw_counts.get("rounds")
            or raw_job.get("round_count")
            or (_round_count(rounds_root) if rounds_root.exists() else 0)
        )
        payload = dict(base)
        payload.update(
            {
                "active": False,
                "snapshot_source": "live_session" if is_live_session_snapshot else "saved_round",
                "snapshot_path": str(snapshot_path),
                "saved_at": str(raw.get("saved_at") or saved_at),
                "job": {
                    "id": str(raw_job.get("id") or round_summary.get("round_id") or round_meta.get("round_id") or round_dir.name),
                    "intent": str(raw_job.get("intent") or round_meta.get("intent") or self._resolved_request_intent or ""),
                    "read_intent": str(raw_job.get("read_intent") or round_meta.get("read_intent") or ""),
                    "write_intent": str(raw_job.get("write_intent") or round_meta.get("write_intent") or ""),
                    "started_at": str(raw_job.get("started_at") or round_summary.get("timestamp") or saved_at),
                    "round_count": round_count,
                },
                "counts": {
                    "preferences": len(raw_preferences),
                    "active_preferences": len(active_preferences),
                    "structured_rules": len(structured_rules),
                    "experiences": len(raw_experiences),
                    "rounds": round_count,
                },
                "preferences": [
                    self._working_memory_preference_payload(pref, index)
                    for index, pref in enumerate(raw_preferences)
                ],
                "structured_rules": structured_rules,
                "experiences": [
                    self._working_memory_experience_payload(exp)
                    for exp in raw_experiences[-12:]
                ],
                "general_rules_text": str(raw.get("general_rules_text") or _extract_memory_injection_section(memory_injection, "working_memory_general_rules")),
                "structured_rules_text": saved_text or str(raw.get("structured_rules_text") or _extract_memory_injection_section(memory_injection, "working_memory_rule_specs")),
                "preferences_text": str(raw.get("preferences_text") or _extract_memory_injection_section(memory_injection, "working_memory_execution_brief")),
                "message": (
                    "Showing saved session working memory. It will survive navigation and service restart."
                    if is_live_session_snapshot
                    else "Showing the latest saved working-memory snapshot from the last completed round."
                ),
            }
        )
        return payload

    @staticmethod
    def _working_memory_preference_payload(pref: Any, index: int) -> dict[str, Any]:
        structured_data = deepcopy(_wm_payload_get(pref, "structured_data", None))
        return {
            "id": f"pref_{index + 1:03d}",
            "content": str(_wm_payload_get(pref, "content", "") or ""),
            "dimension": str(_wm_payload_get(pref, "dimension", "") or ""),
            "preference_type": str(_wm_payload_get(pref, "preference_type", "") or ""),
            "scope": str(_wm_payload_get(pref, "scope", "") or ""),
            "scope_value": str(_wm_payload_get(pref, "scope_value", "") or ""),
            "source_task_id": str(_wm_payload_get(pref, "source_task_id", "") or ""),
            "superseded": bool(_wm_payload_get(pref, "superseded", False)),
            "timestamp": str(_wm_payload_get(pref, "timestamp", "") or ""),
            "structured_data": structured_data if isinstance(structured_data, dict) else None,
        }

    @staticmethod
    def _working_memory_rule_from_preference(pref: Any, index: int) -> dict[str, Any] | None:
        if bool(_wm_payload_get(pref, "superseded", False)):
            return None
        structured_data = _wm_payload_get(pref, "structured_data", None)
        if isinstance(structured_data, dict):
            rule_spec = structured_data.get("rule_spec") if isinstance(structured_data.get("rule_spec"), dict) else None
            spec = deepcopy(rule_spec) if rule_spec else deepcopy(structured_data)
        else:
            spec = None
        if not isinstance(spec, dict):
            return None
        if not (spec.get("schema_version") or spec.get("action") or spec.get("propagation")):
            return None
        rule_id = str(spec.get("rule_id", "") or "").strip() or f"wm_rule_{index + 1:03d}"
        spec.setdefault("content", str(_wm_payload_get(pref, "content", "") or ""))
        spec.setdefault("dimension", str(_wm_payload_get(pref, "dimension", "") or "general"))
        spec["rule_id"] = rule_id
        return spec

    @staticmethod
    def _working_memory_experience_payload(exp: Any) -> dict[str, Any]:
        if hasattr(exp, "to_dict"):
            payload = exp.to_dict()
        elif isinstance(exp, dict):
            payload = dict(exp)
        else:
            payload = dict(getattr(exp, "__dict__", {}) or {})
        return {
            "id": str(payload.get("id") or ""),
            "content": str(payload.get("content") or ""),
            "tool_name": str(payload.get("tool_name") or ""),
            "keywords": list(payload.get("keywords") or []),
            "category": str(payload.get("category") or ""),
            "source": str(payload.get("source") or ""),
            "source_task_id": str(payload.get("source_task_id") or ""),
            "confidence": payload.get("confidence"),
            "timestamp": str(payload.get("timestamp") or ""),
            "outcome": str(payload.get("outcome") or ""),
        }

    def _append_memory_save_trace(self, payload: dict[str, Any]) -> None:
        try:
            from datetime import datetime

            history_dir = self.workspace / ".history"
            history_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "session_id": self.session_id,
                **payload,
            }
            with (history_dir / "memory_save_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to append memory save trace", exc_info=True)

    async def _ensure_memory_job_started(
        self,
        *,
        user_prompt: str,
        intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
        reason: str = "",
    ) -> bool:
        """Start a fresh memory job if manual save ended the previous one."""
        if not self.memory_system:
            return False
        orchestrator = self._create_orchestrator()
        if orchestrator is None:
            return False
        state = self._memory_active_job_state()
        if state["has_active_job"]:
            self._memory_job_needs_start = False
            return False

        await orchestrator.on_job_start(
            user_id=self.user_id,
            project_id=self.workspace.stem,
            user_prompt=user_prompt or "",
            intent=intent or self._resolved_request_intent or "",
            read_intent=read_intent or intent or self._resolved_request_intent or "",
            write_intent=write_intent or intent or self._resolved_request_intent or read_intent or "",
            core_persona=core_persona or "",
        )
        try:
            hydrate_result = await self._hydrate_active_wm_from_latest_snapshot(
                orchestrator,
                reason=reason or "fresh_job_start",
            )
            if hydrate_result.get("hydrated"):
                logger.info("MemoryOrchestrator: %s", hydrate_result)
        except Exception:
            logger.debug("Saved WM snapshot hydration failed (non-fatal)", exc_info=True)
        self._memory_job_needs_start = False
        self._last_memory_job_start = {
            "reason": reason,
            "user_prompt": (user_prompt or "")[:300],
            "intent": intent or self._resolved_request_intent or "",
            "read_intent": read_intent or intent or self._resolved_request_intent or "",
            "write_intent": write_intent or intent or self._resolved_request_intent or read_intent or "",
        }
        logger.info("MemoryOrchestrator: fresh job started after memory save (%s)", reason or "unspecified")
        return True

    async def _restore_memory_job_from_saved_snapshot(
        self,
        orchestrator: Any,
        *,
        reason: str = "session_snapshot_save",
    ) -> dict[str, Any]:
        """Recreate a transient active WM from the durable session snapshot.

        This lets manual save and the 03:00 autosave consolidate working memory
        even after the original worker/runtime has been released.
        """
        job_mgr = getattr(orchestrator, "_job_mgr", None)
        if job_mgr is None:
            return {"restored": False, "reason": "no_job_manager"}
        state = self._memory_active_job_state()
        if state["has_active_job"]:
            return {"restored": False, "reason": "already_active"}

        saved = self._latest_saved_working_memory_snapshot({"session_id": self.session_id, "user_id": self.user_id})
        if not saved:
            return {"restored": False, "reason": "no_saved_snapshot"}
        counts = saved.get("counts") if isinstance(saved.get("counts"), dict) else {}
        has_content = any(
            int(counts.get(key) or 0) > 0
            for key in ("preferences", "active_preferences", "experiences")
        )
        if not has_content:
            return {"restored": False, "reason": "empty_saved_snapshot"}

        job_payload = saved.get("job") if isinstance(saved.get("job"), dict) else {}
        intent = str(job_payload.get("intent") or self._resolved_request_intent or "")
        await orchestrator.on_job_start(
            user_id=self.user_id,
            project_id=self.workspace.stem,
            user_prompt="Saved session working memory",
            intent=intent,
            read_intent=str(job_payload.get("read_intent") or intent),
            write_intent=str(job_payload.get("write_intent") or intent),
            core_persona="",
        )
        hydrate_result = await self._hydrate_active_wm_from_latest_snapshot(
            orchestrator,
            reason=reason,
        )
        if not hydrate_result.get("hydrated"):
            # Avoid leaving an empty synthetic job around.
            next_job_mgr = getattr(orchestrator, "_job_mgr", None)
            if next_job_mgr is not None:
                try:
                    if getattr(next_job_mgr, "working_memory", None) is not None:
                        next_job_mgr.working_memory.release()
                except Exception:
                    logger.debug("Failed to release empty restored WM", exc_info=True)
                try:
                    next_job_mgr._working_memory = None
                    next_job_mgr._active_job = None
                except Exception:
                    logger.debug("Failed to clear empty restored job", exc_info=True)
            return {"restored": False, **hydrate_result}

        restored_state = self._memory_active_job_state()
        restored_job = restored_state.get("active_job")
        if restored_job is not None and not getattr(restored_job, "rounds", None):
            try:
                from memslides.memory.core.models import Round

                restored_job.rounds.append(
                    Round(
                        id=str(uuid.uuid4()),
                        job_id=str(getattr(restored_job, "id", "") or ""),
                        user_message="Saved session working memory",
                    )
                )
            except Exception:
                logger.debug("Failed to add synthetic round for restored WM save", exc_info=True)
        return {"restored": True, **hydrate_result}

    async def flush_memory(
        self,
        reason: str = "manual_web_save",
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Consolidate the current WM to LTM without closing the live runtime."""
        from datetime import datetime
        from memslides.memory.progress import emit_memory_save_progress

        saved_at = datetime.now().astimezone().isoformat()
        config_memory_dir = str(getattr(getattr(self.config, "memory", None), "global_db_dir", "") or "")
        system_memory_dir = str(getattr(getattr(self.memory_system, "config", None), "global_db_dir", "") or "")
        base = {
            "saved_at": saved_at,
            "session_id": self.session_id,
            "memory_db_dir": system_memory_dir or config_memory_dir,
            "reason": reason,
        }
        if not self.memory_system:
            await emit_memory_save_progress(
                progress_callback,
                stage="unavailable",
                progress=100,
                status="no_memory",
                message="Memory runtime is not available for this session.",
            )
            result = {
                **base,
                "status": "no_memory",
                "saved": False,
                "message": "Memory runtime is not available for this session.",
            }
            self._append_memory_save_trace(result)
            return result

        orchestrator = self._create_orchestrator()
        if orchestrator is None:
            await emit_memory_save_progress(
                progress_callback,
                stage="unavailable",
                progress=100,
                status="no_memory",
                message="Memory orchestrator is not available for this session.",
            )
            result = {
                **base,
                "status": "no_memory",
                "saved": False,
                "message": "Memory orchestrator is not available for this session.",
            }
            self._append_memory_save_trace(result)
            return result

        state = self._memory_active_job_state()
        if not state["has_active_job"]:
            pending_backup = self._find_pending_consolidation_backup()
            job_mgr = getattr(orchestrator, "_job_mgr", None)
            if pending_backup and job_mgr is not None and hasattr(job_mgr, "retry_pending_consolidation"):
                await emit_memory_save_progress(
                    progress_callback,
                    stage="retry_from_backup",
                    progress=3,
                    message="Retrying previous pending memory backup.",
                    pending_consolidation_backup=pending_backup,
                )
                try:
                    retry_result = await job_mgr.retry_pending_consolidation(
                        user_id=self.user_id,
                        pending_backup=pending_backup,
                        progress_callback=progress_callback,
                    )
                except Exception as exc:
                    retry_result = {
                        "status": "failed",
                        "saved": False,
                        "message": str(exc),
                        "pending_consolidation_backup": pending_backup,
                    }
                retry_result = retry_result if isinstance(retry_result, dict) else {}
                saved = bool(retry_result.get("saved"))
                result = {
                    **base,
                    "status": "saved" if saved else str(retry_result.get("status") or "failed"),
                    "saved": saved,
                    "message": str(
                        retry_result.get("message")
                        or (
                            "Pending memory backup was saved to long-term memory."
                            if saved
                            else "Pending memory backup retry failed."
                        )
                    ),
                    "job_id": str(retry_result.get("job_id") or ""),
                    "round_count": int(retry_result.get("round_count") or 0),
                "consolidation_failed": not saved,
                "pending_consolidation_backup": str(
                    retry_result.get("pending_consolidation_backup") or ("" if saved else pending_backup)
                    ),
                    "retried_pending_backup": True,
                    "stage": "complete" if saved else "retry_from_backup",
                }
                self._append_memory_save_trace(result)
                await emit_memory_save_progress(
                    progress_callback,
                    stage="complete" if saved else "failed",
                    progress=100,
                    status="saved" if saved else "failed",
                    message=result["message"],
                    job_id=result["job_id"],
                    round_count=result["round_count"],
                    pending_consolidation_backup=result["pending_consolidation_backup"],
                )
                return result

            restore_result = await self._restore_memory_job_from_saved_snapshot(
                orchestrator,
                reason="flush_memory_snapshot_restore",
            )
            if restore_result.get("restored"):
                logger.info("MemoryOrchestrator: restored WM for save: %s", restore_result)
                state = self._memory_active_job_state()
            else:
                logger.debug("No saved WM snapshot available for save: %s", restore_result)

        if not state["has_active_job"]:

            await emit_memory_save_progress(
                progress_callback,
                stage="nothing_to_save",
                progress=100,
                status="nothing_to_save",
                message="There is no active working-memory job to save.",
            )
            result = {
                **base,
                "status": "nothing_to_save",
                "saved": False,
                "message": "There is no active working-memory job to save.",
            }
            self._append_memory_save_trace(result)
            return result

        active_job = state["active_job"]
        job_id = str(getattr(active_job, "id", "") or "")
        round_count = len(getattr(active_job, "rounds", []) or [])
        _slide_dir = self._deferred_slide_html_dir or ""
        _final = self.intermediate_output.get("slide_html_dir", _slide_dir)
        freeze_preference_writeback = (
            self._freeze_preference_writeback_current_job
            or self._should_freeze_preference_writeback()
        )
        try:
            await emit_memory_save_progress(
                progress_callback,
                stage="job_end",
                progress=6,
                message=f"Preparing to consolidate {round_count} memory rounds.",
                job_id=job_id,
                round_count=round_count,
            )
            try:
                _profile_execution_plan = orchestrator.get_profile_execution_plan_payload()
                if _profile_execution_plan:
                    _realization_report = _build_profile_realization_report(
                        workspace=self.workspace,
                        slide_dir=str(_final) if _final else _slide_dir,
                        profile_execution_contract=orchestrator.get_profile_execution_contract_payload(),
                        profile_execution_plan=_profile_execution_plan,
                    )
                    if _realization_report:
                        orchestrator.set_profile_realization_report(_realization_report)
                        info(
                            "Built profile_realization_report before job end (alignment=%s, pages=%s)",
                            _realization_report.get("overall_alignment", ""),
                            len(_realization_report.get("coverage_by_page", []) or []),
                        )
            except Exception as exc:
                logger.warning("profile_realization_report build failed (non-fatal): %s", exc)
            end_result = await orchestrator.on_job_end(
                slide_html_dir=str(_final) if _final else _slide_dir,
                freeze_preference_writeback=freeze_preference_writeback,
                progress_callback=progress_callback,
            )
            self._memory_job_needs_start = True
            self._freeze_preference_writeback_current_job = False
            end_result = end_result if isinstance(end_result, dict) else {}
            consolidation_failed = bool(end_result.get("consolidation_failed"))
            result = {
                **base,
                "status": "failed" if consolidation_failed else "saved",
                "saved": not consolidation_failed,
                "message": (
                    str(end_result.get("message") or "Memory consolidation failed; pending backup was written.")
                    if consolidation_failed
                    else "Working memory was saved to long-term memory."
                ),
                "job_id": job_id,
                "round_count": round_count,
                "consolidation_failed": consolidation_failed,
                "pending_consolidation_backup": str(end_result.get("pending_consolidation_backup") or ""),
                "stage": str(end_result.get("stage") or ("failed" if consolidation_failed else "complete")),
            }
            self._append_memory_save_trace(result)
            if consolidation_failed:
                await emit_memory_save_progress(
                    progress_callback,
                    stage="failed",
                    progress=100,
                    status="failed",
                    message=result["message"],
                    job_id=job_id,
                    round_count=round_count,
                    pending_consolidation_backup=result["pending_consolidation_backup"],
                )
                logger.warning(
                    "Manual memory save failed during consolidation for job %s; pending backup=%s",
                    job_id,
                    result["pending_consolidation_backup"],
                )
            else:
                await emit_memory_save_progress(
                    progress_callback,
                    stage="complete",
                    progress=100,
                    status="saved",
                    message="Working memory was saved to long-term memory.",
                    job_id=job_id,
                    round_count=round_count,
                )
                logger.info("Manual memory save completed for job %s (%s rounds)", job_id, round_count)
            return result
        except Exception as exc:
            result = {
                **base,
                "status": "failed",
                "saved": False,
                "message": str(exc),
                "job_id": job_id,
                "round_count": round_count,
                "stage": "failed",
            }
            self._append_memory_save_trace(result)
            await emit_memory_save_progress(
                progress_callback,
                stage="failed",
                progress=100,
                status="failed",
                message=str(exc),
                job_id=job_id,
                round_count=round_count,
            )
            logger.warning("Manual memory save failed: %s", exc)
            return result

    def _find_pending_consolidation_backup(self) -> str:
        try:
            pending_dir = self.workspace / ".memory"
            if not pending_dir.exists():
                return ""
            files = sorted(
                pending_dir.glob("pending_consolidation_*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            return str(files[0]) if files else ""
        except Exception:
            return ""

    @staticmethod
    def _confidence_text(value: float | None) -> str:
        return _runtime_support.confidence_text(value)

    def _get_request_extra_info(self, request: InputRequest) -> dict[str, Any]:
        return _runtime_support.get_request_extra_info(request)

    def _get_request_core_persona(self, request: InputRequest) -> str:
        return _runtime_support.get_request_core_persona(request)

    def _get_request_task_intent(self, request: InputRequest) -> str:
        return _runtime_support.get_request_task_intent(self, request)

    def _get_request_memory_read_intent(self, request: InputRequest) -> str:
        return _runtime_support.get_request_memory_read_intent(self, request)

    def _get_request_memory_write_intent(self, request: InputRequest) -> str:
        return _runtime_support.get_request_memory_write_intent(self, request)

    async def build_user_state_snapshot(
        self,
        request: InputRequest | None = None,
        *,
        task_intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
        include_cross_intent: bool = True,
    ) -> dict[str, Any]:
        """Build a shared user-state snapshot for generation/review side reuse."""
        if not self.memory_system:
            return {}

        resolver = getattr(self.memory_system, "user_state_resolver", None)
        if resolver is None:
            return {}

        candidate = request or self._last_request
        resolved_task_intent = task_intent
        resolved_read_intent = read_intent
        resolved_write_intent = write_intent
        resolved_core_persona = core_persona

        if candidate is not None:
            resolved_task_intent = (
                resolved_task_intent or self._get_request_task_intent(candidate)
            )
            resolved_read_intent = (
                resolved_read_intent or self._get_request_memory_read_intent(candidate)
            )
            resolved_write_intent = (
                resolved_write_intent or self._get_request_memory_write_intent(candidate)
            )
            resolved_core_persona = (
                resolved_core_persona or self._get_request_core_persona(candidate)
            )

        job_mgr = self._job_mgr
        working_memory = getattr(job_mgr, "working_memory", None) if job_mgr else None

        snapshot = await resolver.build_snapshot(
            user_id=self.user_id,
            task_intent=resolved_task_intent,
            read_intent=resolved_read_intent,
            write_intent=resolved_write_intent,
            core_persona=resolved_core_persona,
            working_memory=working_memory,
            include_cross_intent=include_cross_intent,
        )
        payload = snapshot.to_dict()
        payload["prompt_blocks"] = resolver.build_prompt_blocks(snapshot)
        return payload

    def _should_freeze_preference_writeback(
        self,
        request: InputRequest | None = None,
    ) -> bool:
        return _runtime_support.should_freeze_preference_writeback(self, request)

    def _get_intent_resolution_mode(self, request: InputRequest) -> str:
        return _runtime_support.get_intent_resolution_mode(request)

    def _get_explicit_extra_info_intent(self, request: InputRequest) -> str:
        return _runtime_support.get_explicit_extra_info_intent(request)

    async def _call_runtime_intent_llm(self, llm_client: Any, prompt: str) -> str:
        from memslides.memory.core.models import call_intent_classifier_llm

        return await call_intent_classifier_llm(llm_client, prompt)

    async def _classify_request_intent_with_llm(
        self,
        request: InputRequest,
    ):
        from memslides.memory.core.models import (
            IntentClassificationResult,
            build_intent_classification_prompt,
            classify_intent_details_with_llm,
        )

        if not self.memory_system:
            return IntentClassificationResult(intent="", scenario="", raw_response="")

        llm_objects = getattr(self.memory_system, "llm_objects_by_task", {}) or {}
        llm_client = (
            llm_objects.get("intent_classify")
            or getattr(self.memory_system, "llm", None)
        )
        if llm_client is None:
            return IntentClassificationResult(intent="", scenario="", raw_response="")

        attachment_names = [Path(path).name for path in request.attachments if path]
        template_name = Path(request.template).stem if request.template else ""
        prompt = build_intent_classification_prompt(
            request.instruction,
            attachment_names=attachment_names,
            template_name=template_name,
            num_pages=request.num_pages,
        )
        result = await classify_intent_details_with_llm(
            request.instruction,
            llm_client=llm_client,
            fallback_to_keywords=False,
            attachment_names=attachment_names,
            template_name=template_name,
            num_pages=request.num_pages,
        )

        artifact_writer = getattr(self.memory_system, "artifact_writer", None)
        if artifact_writer and hasattr(artifact_writer, "write_intent_classification"):
            try:
                artifact_writer.write_intent_classification(
                    turn_id=0,
                    user_message=request.instruction,
                    prompt=prompt,
                    response=result.raw_response,
                    intent_result={
                        "intent": result.intent,
                        "scenario": result.scenario,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "source": "runtime_request",
                    },
                )
            except Exception as e:
                logger.debug("Runtime request intent artifact write failed: %s", e)

        return result

    async def _resolve_request_intent_runtime(
        self,
        request: InputRequest,
    ) -> IntentResolutionResult:
        from memslides.memory.core.models import (
            DEFAULT_INTENT,
            build_intent_signal_text,
            classify_intent_by_keywords,
            infer_intent_scenario,
        )

        if self._resolved_request_intent:
            return IntentResolutionResult(
                intent=self._resolved_request_intent,
                scenario=self._resolved_request_intent_scenario,
                source=self._resolved_request_intent_source,
                confidence=self._resolved_request_intent_confidence,
                raw_response=self._resolved_request_intent_raw_response,
            )

        explicit_request_intent = _normalize_memory_intent(request.memory_intent)
        explicit_extra_info_intent = self._get_explicit_extra_info_intent(request)
        resolution_mode = self._get_intent_resolution_mode(request)
        intent_signal_text = build_intent_signal_text(
            request.instruction,
            attachment_names=[Path(path).name for path in request.attachments if path],
            template_name=Path(request.template).stem if request.template else "",
            num_pages=request.num_pages,
        )

        if resolution_mode == "explicit_first":
            if explicit_request_intent:
                return IntentResolutionResult(
                    intent=explicit_request_intent,
                    scenario=infer_intent_scenario(intent_signal_text, explicit_request_intent),
                    source="explicit_request",
                    confidence=1.0,
                )
            if explicit_extra_info_intent:
                return IntentResolutionResult(
                    intent=explicit_extra_info_intent,
                    scenario=infer_intent_scenario(intent_signal_text, explicit_extra_info_intent),
                    source="explicit_extra_info",
                    confidence=1.0,
                )

        llm_result = None
        if resolution_mode != "keyword_only":
            llm_result = await self._classify_request_intent_with_llm(request)
            if llm_result.intent and llm_result.intent != DEFAULT_INTENT:
                return IntentResolutionResult(
                    intent=llm_result.intent,
                    scenario=llm_result.scenario,
                    source="llm",
                    confidence=llm_result.confidence,
                    raw_response=llm_result.raw_response,
                )

        keyword_intent = classify_intent_by_keywords(request.instruction)
        keyword_scenario = infer_intent_scenario(intent_signal_text, keyword_intent)
        if keyword_intent != DEFAULT_INTENT:
            return IntentResolutionResult(
                intent=keyword_intent,
                scenario=keyword_scenario,
                source="keyword",
                raw_response=llm_result.raw_response if llm_result else "",
            )

        if llm_result and llm_result.intent:
            return IntentResolutionResult(
                intent=llm_result.intent,
                scenario=llm_result.scenario,
                source="llm",
                confidence=llm_result.confidence,
                raw_response=llm_result.raw_response,
            )

        return IntentResolutionResult(
            intent=keyword_intent or DEFAULT_INTENT,
            scenario=keyword_scenario,
            source="keyword",
            raw_response=llm_result.raw_response if llm_result else "",
        )

    def _write_runtime_json(self, filename: str, payload: dict[str, Any]) -> None:
        try:
            with open(self.workspace / filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("Failed to write %s: %s", filename, e)

    def _cache_resolved_request_intent(
        self,
        request: InputRequest,
        result: IntentResolutionResult,
    ) -> None:
        extra_info = self._get_request_extra_info(request)
        self._resolved_request_intent = result.intent
        self._resolved_request_intent_scenario = result.scenario
        self._resolved_request_intent_source = result.source
        self._resolved_request_intent_confidence = result.confidence
        self._resolved_request_intent_raw_response = result.raw_response

        extra_info["resolved_memory_intent"] = result.intent
        extra_info["resolved_scenario_intent"] = result.scenario
        extra_info["resolved_memory_intent_source"] = result.source
        extra_info["resolved_task_intent"] = self._get_request_task_intent(request)
        extra_info["resolved_memory_read_intent"] = self._get_request_memory_read_intent(request)
        extra_info["resolved_memory_write_intent"] = self._get_request_memory_write_intent(request)
        core_persona = self._get_request_core_persona(request)
        if core_persona:
            extra_info["core_persona"] = core_persona
        if result.confidence is not None:
            extra_info["resolved_memory_intent_confidence"] = result.confidence
        elif "resolved_memory_intent_confidence" in extra_info:
            extra_info.pop("resolved_memory_intent_confidence", None)

        payload = {
            "resolved_memory_intent": result.intent,
            "resolved_scenario_intent": result.scenario,
            "resolved_task_intent": extra_info.get("resolved_task_intent", ""),
            "resolved_memory_read_intent": extra_info.get("resolved_memory_read_intent", ""),
            "resolved_memory_write_intent": extra_info.get("resolved_memory_write_intent", ""),
            "core_persona": extra_info.get("core_persona", ""),
            "source": result.source,
            "confidence": result.confidence,
            "raw_response": result.raw_response,
            "resolution_mode": self._get_intent_resolution_mode(request),
            "explicit_request_intent": _normalize_memory_intent(request.memory_intent),
            "explicit_extra_info_intent": self._get_explicit_extra_info_intent(request),
            "instruction_preview": request.instruction[:500],
        }
        self._resolved_request_intent_payload = payload
        self._write_runtime_json("resolved_intent.json", payload)

    def _get_runtime_request_intent(self, request: InputRequest) -> str:
        return self._resolved_request_intent or _resolve_request_memory_intent(request)

    def _initialize_template_match_state(self, request: InputRequest) -> None:
        self._template_match_state = {
            "resolved_memory_intent": self._get_runtime_request_intent(request),
            "resolved_scenario_intent": self._resolved_request_intent_scenario,
            "resolved_task_intent": self._get_request_task_intent(request),
            "resolved_memory_read_intent": self._get_request_memory_read_intent(request),
            "resolved_memory_write_intent": self._get_request_memory_write_intent(request),
            "core_persona": self._get_request_core_persona(request),
            "template_intent": "",
            "template_intent_confidence": None,
            "selected_template_id": "",
            "selected_template_name": "",
            "selection_source": "",
            "selection_confidence": None,
            "template_reasoning": "",
            "selection_reasoning": "",
            "style_memory_hint": "",
            "template_query": "",
            "matched_by_history": False,
            "matched_history_intent": "",
            "request_used_explicit_template": bool(
                request.template or request.template_id or request.template_as_reference
            ),
            "instruction_preview": request.instruction[:500],
        }
        self._write_runtime_json("template_match.json", self._template_match_state)

    def _update_template_match_state(
        self,
        request: InputRequest,
        **updates: Any,
    ) -> None:
        if not self._template_match_state:
            self._initialize_template_match_state(request)
        self._template_match_state.update(updates)
        self._template_match_state["resolved_memory_intent"] = self._get_runtime_request_intent(request)
        self._template_match_state["resolved_task_intent"] = self._get_request_task_intent(request)
        self._template_match_state["resolved_memory_read_intent"] = self._get_request_memory_read_intent(request)
        self._template_match_state["resolved_memory_write_intent"] = self._get_request_memory_write_intent(request)
        self._template_match_state["core_persona"] = self._get_request_core_persona(request)
        self._template_match_state["instruction_preview"] = request.instruction[:500]
        self._write_runtime_json("template_match.json", self._template_match_state)

    def get_resolved_intent_artifact(self) -> dict[str, Any]:
        return dict(self._resolved_request_intent_payload)

    def get_template_match_artifact(self) -> dict[str, Any]:
        return dict(self._template_match_state)

    async def _inject_system_memory(
        self,
        orchestrator,
        agent_role: str,
        user_message: str,
        agent,
        turn: int = 0,
    ) -> str:
        """Get structured memory components from orchestrator and inject into agent's SYSTEM prompt.

        Also dumps a system-level injection_trace via ArtifactDumper (4-component JSON).

        Args:
            orchestrator: MemoryOrchestrator instance
            agent_role: "research" | "design" | "modify"
            user_message: user instruction or modify message
            agent: Agent instance whose chat_history[0] will be updated
            turn: turn number (0 for run(), self._modify_turn_count for modify())

        Returns:
            The combined memory prompt text that was injected (empty string if nothing injected).
        """
        if not orchestrator:
            return ""
        try:
            components = await orchestrator.get_memory_components(
                user_message=user_message,
                agent_name=agent_role,
            )
            combined = components.get("combined", "")

            # Dump system-level injection trace (regardless of whether combined is empty)
            _dumper = getattr(self.memory_system, 'artifact_dumper', None) if self.memory_system else None
            if _dumper:
                round_index = orchestrator._job_mgr.round_count() if orchestrator._job_mgr else 1
                _dumper.dump_injection_trace(
                    round_index,
                    level="system",
                    agent_role=agent_role,
                    turn=turn,
                    wm_preferences=components.get("wm_preferences", ""),
                    profile_execution_contract=components.get("profile_execution_contract", ""),
                    profile_execution_plan=components.get("profile_execution_plan", ""),
                    wm_experiences=components.get("wm_experiences", ""),
                    ltm_tool_experiences=components.get("ltm_tool_experiences", ""),
                    wm_round_history=components.get("wm_round_history", ""),
                    final_injected_text=combined,
                    injection_target="SYSTEM" if combined else "",
                    skipped_reason="" if combined else "no memory available",
                )
            self._system_injection_trace_state[(agent_role, turn)] = {
                "round_index": orchestrator._job_mgr.round_count() if orchestrator._job_mgr else 1,
                "agent_role": agent_role,
                "turn": turn,
                "components": {
                    "wm_preferences": components.get("wm_preferences", ""),
                    "profile_execution_contract": components.get("profile_execution_contract", ""),
                    "profile_execution_plan": components.get("profile_execution_plan", ""),
                    "wm_experiences": components.get("wm_experiences", ""),
                    "ltm_tool_experiences": components.get("ltm_tool_experiences", ""),
                    "wm_round_history": components.get("wm_round_history", ""),
                },
                "final_injected_text": combined,
                "injection_target": "SYSTEM" if combined else "",
                "skipped_reason": "" if combined else "no memory available",
            }

            if not combined:
                return ""

            # Inject into agent's system prompt
            current_system = agent.chat_history[0].text if agent.chat_history else agent.system
            agent.chat_history[0] = ChatMessage(
                role=Role.SYSTEM,
                content=current_system + f"\n\n{combined}",
            )

            # Set orchestrator on agent for operation-level memory injection
            agent._memory_orchestrator = orchestrator

            info(f"Injected system memory for {agent_role} ({len(combined)} chars, "
                 f"components: {[k for k, v in components.items() if v and k != 'combined']})")
            return combined
        except Exception as e:
            logger.warning(f"_inject_system_memory({agent_role}) failed: {e}")
            return ""

    def _refresh_system_injection_trace_with_template(
        self,
        *,
        agent_role: str,
        turn: int,
        template_prompt: str,
    ) -> None:
        """Refresh a saved SYSTEM injection trace so it also records template prompt injection."""
        if not template_prompt or not self.memory_system:
            return

        state = self._system_injection_trace_state.get((agent_role, turn))
        dumper = getattr(self.memory_system, "artifact_dumper", None)
        if not state or not dumper:
            return

        final_injected_text = state.get("final_injected_text", "")
        if final_injected_text:
            final_injected_text = f"{final_injected_text}\n\n{template_prompt}"
        else:
            final_injected_text = template_prompt

        components = state.get("components", {})
        dumper.dump_injection_trace(
            state.get("round_index", 1),
            level="system",
            agent_role=agent_role,
            turn=turn,
            wm_preferences=components.get("wm_preferences", ""),
            profile_execution_contract=components.get("profile_execution_contract", ""),
            profile_execution_plan=components.get("profile_execution_plan", ""),
            wm_experiences=components.get("wm_experiences", ""),
            ltm_tool_experiences=components.get("ltm_tool_experiences", ""),
            wm_round_history=components.get("wm_round_history", ""),
            final_injected_text=final_injected_text,
            injection_target=state.get("injection_target", "SYSTEM"),
            skipped_reason=state.get("skipped_reason", ""),
            template_prompt=template_prompt,
        )
        state["final_injected_text"] = final_injected_text
        state["template_prompt"] = template_prompt

    def _resolve_all_slide_paths(self) -> list[Path]:
        return _runtime_support.resolve_all_slide_paths(self)

    def _candidate_slide_dirs(
        self,
        slide_dir: Path | str | None = None,
        *,
        include_workspace_root: bool = True,
    ) -> list[Path]:
        return _runtime_support.candidate_slide_dirs(
            self,
            slide_dir,
            include_workspace_root=include_workspace_root,
        )

    def _collect_exportable_slide_paths(self, slide_dir: Path) -> list[Path]:
        return _runtime_support.collect_exportable_slide_paths(slide_dir)

    @staticmethod
    def _exportable_slide_dir_score(
        slide_paths: list[Path],
        candidate_index: int,
    ) -> tuple[int, float, int, int]:
        return _runtime_support.exportable_slide_dir_score(slide_paths, candidate_index)

    def _resolve_exportable_slide_dir(self, slide_dir: Path | str | None = None) -> Path | None:
        return _runtime_support.resolve_exportable_slide_dir(self, slide_dir)

    def _resolve_exportable_slide_paths(self, slide_dir: Path | str | None = None) -> list[Path]:
        return _runtime_support.resolve_exportable_slide_paths(self, slide_dir)

    def _ensure_modifyagent_loaded(self, *, load_reason: str) -> RevisionEditor | None:
        if self.modifyagent is not None:
            if isinstance(self.modifyagent, RevisionEditor):
                return self.modifyagent
            raise RuntimeError(
                "RevisionEditor initialization failed because the cached modify agent "
                f"is not a RevisionEditor instance during {load_reason}: "
                f"{type(self.modifyagent).__name__}"
            )

        if self.agent_env is None:
            warning(
                "RevisionEditor is unavailable for %s because the agent environment is not initialized.",
                load_reason,
            )
            return None

        try:
            info(f"Loading RevisionEditor for {load_reason}...")
            self.modifyagent = RevisionEditor(
                self.config,
                self.agent_env,
                self.workspace,
                self.language,
            )
            info(f"RevisionEditor loaded with {len(self.modifyagent.tools)} tools")
        except RoleToolContractError as e:
            error(f"RevisionEditor protocol mismatch during {load_reason}: {e}")
            raise RuntimeError(
                "RevisionEditor initialization failed because the runtime tool schema "
                "does not match the required existing-slide patch protocol. "
                f"{e}"
            ) from e
        except Exception as e:
            logger.warning(f"Failed to load RevisionEditor for {load_reason}: {e}")
            raise RuntimeError(
                f"RevisionEditor initialization failed during {load_reason}: {e}"
            ) from e

        return self.modifyagent

    def _resolve_export_repair_agent(self, preferred_agent: Any | None = None) -> RevisionEditor | None:
        if isinstance(preferred_agent, RevisionEditor):
            return preferred_agent
        return self._ensure_modifyagent_loaded(load_reason="export repair")

    def _extract_export_failure_slide(self, error_text: str) -> Path | None:
        match = re.search(r"(/\S+\.html)", error_text)
        if not match:
            return None
        try:
            return Path(match.group(1)).resolve()
        except Exception:
            return Path(match.group(1))

    def _export_failure_kind(self, error_text: str) -> str:
        text = " ".join(str(error_text or "").lower().split())
        if "overflows body" in text or "clipped outside" in text:
            return "overflow"
        if "too close to bottom edge" in text:
            return "bottom_safe_zone"
        if "don't match presentation layout" in text or "expected body css" in text:
            return "canvas_dimensions"
        if "image not found" in text or "background image not found" in text:
            return "missing_image"
        return text[:120] or "export_failure"

    def _export_failure_signature(self, error_text: str, failing_slide: Path | None) -> str:
        slide_key = str(failing_slide or self._extract_export_failure_slide(error_text) or "unknown")
        return f"{slide_key}:{self._export_failure_kind(error_text)}"

    def _format_export_diagnostics(self, diagnostics: Any, *, limit: int = 4) -> str:
        if not isinstance(diagnostics, list) or not diagnostics:
            return ""
        lines: list[str] = []
        for item in diagnostics[:limit]:
            if not isinstance(item, dict):
                continue
            parts = [
                str(item.get("code") or item.get("severity") or "diagnostic"),
                str(item.get("message") or "").strip(),
            ]
            selector = str(item.get("target_selector_hint") or "").strip()
            dom_path = str(item.get("target_dom_path") or "").strip()
            repair_paths = item.get("repair_dom_paths")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if selector:
                parts.append(f"selector={selector}")
            if dom_path:
                parts.append(f"dom_path={dom_path}")
            if isinstance(repair_paths, list) and repair_paths:
                parts.append(f"repair_dom_paths={', '.join(str(path) for path in repair_paths[:3])}")
            if isinstance(metadata, dict):
                overflow = metadata.get("overflow")
                if overflow:
                    parts.append(f"overflow={overflow}")
                text_preview = metadata.get("text_preview") or item.get("text_preview")
                if text_preview:
                    parts.append(f"text={str(text_preview)[:120]}")
            line = " | ".join(part for part in parts if part)
            if line:
                lines.append(f"- {line}")
        return "\n".join(lines)

    def _build_export_repair_message(
        self,
        *,
        error_text: str,
        context_label: str,
        failing_slide: Path | None,
        slide_dir: Path | str | None,
        diagnostics: Any = None,
        failure_count: int = 1,
    ) -> str:
        cleaned_error = " ".join(str(error_text or "").split())
        if len(cleaned_error) > 1800:
            cleaned_error = cleaned_error[:1797] + "..."
        slide_label = (
            self._workspace_relative_label(failing_slide)
            if failing_slide is not None
            else "unknown"
        )
        workspace_context = self._build_workspace_context_block(
            slide_dir=slide_dir,
            include_active_slide_dir=True,
        )
        force_rewrite = failure_count >= 2
        repair_strategy = (
            "This same slide/export error has repeated. Do not attempt another tiny patch. "
            "Rewrite the complete failing slide with `write_html_file`, preserving the deck style but simplifying layout density, "
            "then inspect the rewritten slide before `finalize`."
            if force_rewrite
            else "First attempt may use `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch`; if layout targets are broad or ambiguous, rewrite the failing slide."
        )
        message = (
            f"SYSTEM: PPT export validation failed during {context_label}. "
            "Use the RevisionEditor-stage local repair protocol on the current on-disk HTML: "
            "inspect the failing slide and relevant current HTML first, run `plan_slide_patch` when available, "
            "prioritize `offending_targets`, `repair_context`, `recommended_patch_strategy`, `repair_candidates`, `rules`, and target selectors before patching, "
            "never call `apply_slide_patch` with empty or invalid `patch_ops`, reserve "
            "`write_html_file` for controlled regenerate only, run `inspect_slide` on "
            "every changed slide, and only call `finalize` after the current on-disk HTML "
            "passes inspection. `inspect_slide` is diagnostic only and never rewrites the "
            "disk HTML.\n"
            f"Failing slide: {slide_label}\n"
            f"Failure count for this slide/error kind: {failure_count}\n"
            f"Repair strategy: {repair_strategy}\n"
            f"Export error: {cleaned_error}\n"
            "Do not restate the plan in prose. Fix the slide now."
        )
        diagnostic_text = self._format_export_diagnostics(diagnostics)
        if diagnostic_text:
            message += f"\nStructured pptx_export diagnostics:\n{diagnostic_text}"
        if workspace_context:
            message += f"\n{workspace_context}"
        return message

    async def _run_slide_export_repair(
        self,
        *,
        current_slide_dir: Path | str,
        error: Exception | str,
        context_label: str,
        preferred_agent: Any | None = None,
        max_turns: int = 3,
        failure_count: int = 1,
        diagnostics: Any = None,
    ) -> Path | None:
        repair_agent = self._resolve_export_repair_agent(preferred_agent)
        if repair_agent is None:
            return None

        repaired_dir = self._resolve_exportable_slide_dir(current_slide_dir)
        if repaired_dir is None:
            repaired_dir = Path(current_slide_dir)
            if not repaired_dir.is_absolute():
                repaired_dir = self.workspace / repaired_dir
            repaired_dir = repaired_dir.resolve()

        error_text = str(error)
        failing_slide = self._extract_export_failure_slide(error_text)
        diagnostics = diagnostics if diagnostics is not None else getattr(error, "pptx_export_diagnostics", None)
        repair_agent.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=self._build_export_repair_message(
                    error_text=error_text,
                    context_label=context_label,
                    failing_slide=failing_slide,
                    slide_dir=repaired_dir,
                    diagnostics=diagnostics,
                    failure_count=failure_count,
                ),
            )
        )
        set_current_agent(
            repair_agent.name,
            workspace=self.workspace,
            model_ref=getattr(repair_agent, "model_ref", "modify_agent"),
        )

        try:
            for repair_turn in range(max_turns):
                agent_message = await repair_agent.action()
                if not agent_message.tool_calls:
                    warning(
                        "%s export-repair turn %s returned no tool calls; stopping repair loop.",
                        repair_agent.name,
                        repair_turn + 1,
                    )
                    break

                outcome = await repair_agent.execute(agent_message.tool_calls)

                candidate_dir = repaired_dir
                if isinstance(outcome, str):
                    candidate_dir = Path(outcome)
                    if not candidate_dir.is_absolute():
                        candidate_dir = self.workspace / candidate_dir
                    candidate_dir = (
                        self._resolve_exportable_slide_dir(candidate_dir)
                        or candidate_dir.resolve()
                    )
                elif repaired_dir is not None:
                    candidate_dir = (
                        self._resolve_exportable_slide_dir(repaired_dir)
                        or repaired_dir
                    )

                if candidate_dir is not None:
                    repaired_dir = candidate_dir
                    self.intermediate_output["slide_html_dir"] = str(repaired_dir)
                    self.save_results()

                if isinstance(outcome, str):
                    return repaired_dir
        finally:
            repair_agent.save_history()
            self.save_results()

        return repaired_dir

    async def _export_slides_with_agent_repair(
        self,
        slide_html_dir: Path | str,
        output_pptx: Path | str,
        *,
        aspect_ratio: str,
        context_label: str,
        preferred_agent: Any | None = None,
        max_repair_rounds: int = 2,
        max_agent_turns: int = 3,
    ) -> tuple[Path, list[Path]]:
        try:
            max_repair_rounds = max(0, int(os.getenv("MEMSLIDES_EXPORT_STRICT_REPAIR_ROUNDS", str(max_repair_rounds)) or max_repair_rounds))
        except ValueError:
            max_repair_rounds = max(0, max_repair_rounds)
        allow_relaxed_fallback = str(os.getenv("MEMSLIDES_EXPORT_ALLOW_RELAXED_FALLBACK", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        failure_counts: dict[str, int] = {}
        resolved_slide_dir = self._resolve_exportable_slide_dir(slide_html_dir)
        if resolved_slide_dir is None:
            searched_dirs = ", ".join(
                str(path) for path in self._candidate_slide_dirs(slide_html_dir)
            )
            raise RuntimeError(
                "DeckDesigner agent did not produce exportable slides. "
                f"Searched: {searched_dirs}"
            )

        repair_agent = preferred_agent if isinstance(preferred_agent, RevisionEditor) else None
        last_error: Exception | None = None

        for repair_round in range(max_repair_rounds + 1):
            resolved_slide_dir = (
                self._resolve_exportable_slide_dir(resolved_slide_dir)
                or resolved_slide_dir
            )
            self.intermediate_output["slide_html_dir"] = str(resolved_slide_dir)
            self.save_results()

            export_html_files = self._resolve_exportable_slide_paths(resolved_slide_dir)
            if not export_html_files:
                raise RuntimeError(
                    f"No exportable slides found in {resolved_slide_dir}; expected slide_XX.html files."
                )

            try:
                _exp_writer = self._make_exp_writer()
                await convert_html_to_pptx_with_retry(
                    export_html_files,
                    output_pptx,
                    aspect_ratio=aspect_ratio,
                    session_id=str(self.workspace.stem),
                    experience_writer=_exp_writer,
                    allow_skip_layout_validation_fallback=False,
                )
                self.intermediate_output["export_status"] = "strict"
                self.intermediate_output["export_warnings"] = []
                self.save_results()
                return resolved_slide_dir, export_html_files
            except Exception as e:
                last_error = e
                if repair_round >= max_repair_rounds:
                    break

                warning(
                    "PPT export failed during %s (repair round %s/%s): %s",
                    context_label,
                    repair_round + 1,
                    max_repair_rounds,
                    e,
                )
                error_text = str(e)
                failing_slide = self._extract_export_failure_slide(error_text)
                signature = self._export_failure_signature(error_text, failing_slide)
                failure_counts[signature] = failure_counts.get(signature, 0) + 1
                diagnostics = getattr(e, "pptx_export_diagnostics", None)
                repaired_dir = await self._run_slide_export_repair(
                    current_slide_dir=resolved_slide_dir,
                    error=e,
                    context_label=context_label,
                    preferred_agent=repair_agent,
                    max_turns=max_agent_turns,
                    failure_count=failure_counts[signature],
                    diagnostics=diagnostics,
                )
                if isinstance(self.modifyagent, RevisionEditor):
                    repair_agent = self.modifyagent
                if repaired_dir is None:
                    break
                resolved_slide_dir = repaired_dir

        export_html_files = self._resolve_exportable_slide_paths(resolved_slide_dir)
        if allow_relaxed_fallback and export_html_files:
            warning_text = f"PPTX strict export failed during {context_label}: {last_error}"
            try:
                _exp_writer = self._make_exp_writer()
                await convert_html_to_pptx_with_retry(
                    export_html_files,
                    output_pptx,
                    aspect_ratio=aspect_ratio,
                    session_id=str(self.workspace.stem),
                    experience_writer=_exp_writer,
                    allow_skip_layout_validation_fallback=True,
                )
                warning_text += " Relaxed PPTX fallback was used."
            except Exception as relaxed_error:
                warning_text += f" Relaxed PPTX fallback also failed: {relaxed_error}. HTML/PDF fallback remains available."
                warning(warning_text)
            warnings = list(self.intermediate_output.get("export_warnings") or [])
            if warning_text not in warnings:
                warnings.append(warning_text)
            self.intermediate_output["export_status"] = "partial"
            self.intermediate_output["export_warnings"] = warnings
            self.save_results()
            return resolved_slide_dir, export_html_files

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"PPT export failed during {context_label}")

    def _resolve_slide_path(self, target_slide: str) -> Path | None:
        """Resolve target_slide to an actual slide path in the current slide dir.

        Handles zero-padded filenames by trying the exact name first,
        then extracting the number and zero-padding it.
        """
        for slide_dir in self._candidate_slide_dirs():
            exact = slide_dir / f"{target_slide}.html"
            if exact.exists():
                return exact
            m = re.search(r'(\d+)$', target_slide)
            if m:
                num = int(m.group(1))
                padded = target_slide[:m.start()] + f"{num:02d}"
                padded_path = slide_dir / f"{padded}.html"
                if padded_path.exists():
                    return padded_path
        return None

    @staticmethod
    def _extract_tag_block(text: str, tag_name: str) -> str:
        """Extract the inner text of a simple XML-like prompt block."""
        source = str(text or "")
        if not source:
            return ""
        start = source.find(f"<{tag_name}")
        if start < 0:
            return ""
        open_end = source.find(">", start)
        if open_end < 0:
            return ""
        close = source.find(f"</{tag_name}>", open_end)
        if close < 0:
            return ""
        return source[open_end + 1 : close].strip()

    @classmethod
    def _parse_rule_specs_from_injection(cls, text: str) -> list[dict[str, Any]]:
        """Parse JSONL rule specs from <working_memory_rule_specs> injection text."""
        source = str(text or "")
        blocks = re.findall(
            r"<working_memory_rule_specs\b[^>]*>(.*?)</working_memory_rule_specs>",
            source,
            flags=re.DOTALL,
        )
        if not blocks:
            inner = cls._extract_tag_block(source, "working_memory_rule_specs")
            blocks = [inner] if inner else []
        if not blocks:
            return []
        specs: list[dict[str, Any]] = []
        for inner in blocks:
            for line in inner.splitlines():
                payload = line.strip()
                if not payload or not payload.startswith("{"):
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    specs.append(data)
        return specs

    @staticmethod
    def _selector_hints_for_element_kind(element_kind: str) -> list[str]:
        mapping = {
            "slide_title": ["h1", ".title", "[data-role='title']"],
            "subtitle": ["h2", ".subtitle", "[data-role='subtitle']"],
            "body_text": ["p", "li", ".body", ".content"],
            "footer": [".footer", "footer", "[data-role='footer']"],
        }
        return mapping.get(str(element_kind or "").strip().lower(), [])

    @staticmethod
    def _rule_spec_applies_to_existing_slides(spec: dict[str, Any]) -> bool:
        """Return whether a WM rule expects edits on already-rendered slides."""
        if not isinstance(spec, dict):
            return True

        propagation = spec.get("propagation")
        if isinstance(propagation, dict) and "apply_existing_slides" in propagation:
            return _normalize_bool_flag(propagation.get("apply_existing_slides"))

        target = spec.get("target")
        if isinstance(target, dict):
            slide_scope = str(target.get("slide_scope", "") or "").strip().lower()
            if slide_scope in {
                "future",
                "future_only",
                "new",
                "new_only",
                "new_slides",
                "inserted_only",
            }:
                return False

        return True

    @staticmethod
    def _rule_spec_applies_to_future_slides(spec: dict[str, Any]) -> bool:
        """Return whether a WM rule explicitly targets future/new slides."""
        if not isinstance(spec, dict):
            return False

        propagation = spec.get("propagation")
        if isinstance(propagation, dict) and "apply_future_slides" in propagation:
            return _normalize_bool_flag(propagation.get("apply_future_slides"))

        target = spec.get("target")
        if isinstance(target, dict):
            slide_scope = str(target.get("slide_scope", "") or "").strip().lower()
            if slide_scope in {
                "future",
                "future_only",
                "new",
                "new_only",
                "new_slides",
                "inserted_only",
            }:
                return True

        return False

    @staticmethod
    def _parse_style_declarations(style_text: str) -> dict[str, str]:
        declarations: dict[str, str] = {}
        for chunk in str(style_text or "").split(";"):
            if ":" not in chunk:
                continue
            name, value = chunk.split(":", 1)
            key = name.strip().lower()
            normalized_value = value.strip()
            if key and normalized_value:
                declarations[key] = normalized_value
        return declarations

    @classmethod
    def _resolve_css_var_value(
        cls,
        value: str,
        css_vars: dict[str, str],
        depth: int = 0,
    ) -> str:
        if depth > 4:
            return str(value or "").strip()

        text = str(value or "").strip()
        if not text:
            return ""

        match = re.fullmatch(
            r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+?)\s*)?\)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return text

        var_name = match.group(1).strip()
        fallback = (match.group(2) or "").strip()
        resolved = css_vars.get(var_name, fallback)
        if not resolved:
            return ""
        return cls._resolve_css_var_value(resolved, css_vars, depth + 1)

    @staticmethod
    def _extract_inserted_slide_path_from_result(result_text: str) -> str:
        text = str(result_text or "").strip()
        if not text:
            return ""

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for key in ("new_slide_path", "slide_path", "file_path", "write_target", "path"):
                candidate = str(payload.get(key, "") or "").strip()
                if candidate:
                    return candidate

        match = re.search(
            r"(?:New slide path|new_slide_path|slide_path|write_target)\s*[:=]\s*[\"']?(?P<path>[^\n\r\"']+)",
            text,
            re.IGNORECASE,
        )
        return match.group("path").strip() if match else ""

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start:end])
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _future_preference_judge_llm(self) -> Any | None:
        modify_agent = getattr(self, "modifyagent", None)
        llm = getattr(modify_agent, "llm", None) if modify_agent is not None else None
        if llm and hasattr(llm, "run"):
            return llm
        design_agent = getattr(self, "designagent", None)
        llm = getattr(design_agent, "llm", None) if design_agent is not None else None
        return llm if llm and hasattr(llm, "run") else None

    def _collect_future_slide_preference_failures(
        self,
        slide_paths: list[Path],
        rule_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _runtime_support.collect_future_slide_preference_failures(
            self,
            slide_paths,
            rule_specs,
        )

    async def _judge_future_slide_preferences_with_llm(
        self,
        slide_path: Path,
        rule_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return []

    async def _collect_future_slide_preference_failures_async(
        self,
        slide_paths: list[Path],
        rule_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return await _runtime_support.collect_future_slide_preference_failures_async(
            self,
            slide_paths,
            rule_specs,
        )

    def _build_future_preference_followup(
        self,
        failures: list[dict[str, Any]],
        *,
        include_header: bool = True,
    ) -> str:
        if not failures:
            return ""

        lines = []
        for item in failures[:8]:
            observed_text = str(item.get("observed_issue", "") or "").strip()
            if not observed_text:
                observed_text = "concrete CSS preference is not satisfied"
            lines.append(
                f"- {self._workspace_relative_label(item['slide_path'])}: "
                f"{observed_text}; active future preference says {item['requirement']}"
            )
        prefix = (
            "SYSTEM: A newly inserted slide is not yet following an active remembered future preference.\n"
            if include_header
            else ""
        )
        return (
            prefix
            + "\n".join(lines)
            + "\nRepair only the new slide(s) that fail this remembered preference. "
            "Do not batch-edit old slides, and do not call `finalize` until the inserted slide(s) comply."
        )

    @staticmethod
    def _intent_value_mentions_structural_slide_op(intent: dict[str, Any] | None) -> bool:
        if not isinstance(intent, dict):
            return False

        structural_keywords = (
            "insert",
            "insert_slide",
            "add_slide",
            "new_slide",
            "append_slide",
            "delete_slide",
            "remove_slide",
            "reorder_slide",
            "move_slide",
            "duplicate_slide",
        )
        for key in (
            "intent_type",
            "action",
            "op",
            "operation",
            "task_type",
            "modify_action",
            "edit_action",
        ):
            value = str(intent.get(key, "") or "").strip().lower()
            if value and any(keyword in value for keyword in structural_keywords):
                return True
        return False

    @classmethod
    def _is_structural_slide_operation(
        cls,
        user_message: str,
        intent: dict[str, Any] | None,
    ) -> bool:
        """Detect add/delete/reorder slide requests that should stay local."""
        if cls._intent_value_mentions_structural_slide_op(intent):
            return True

        normalized = " ".join(str(user_message or "").split()).lower()
        if not normalized:
            return False

        if re.search(r"(后续|以后|未来|默认).{0,10}(新增页|新增页面|新加页|新页面)", normalized):
            if not re.search(r"(在|于).{0,12}(前|后|之前|之后|结尾|最后|第\d+页)", normalized):
                return False

        structural_patterns = (
            r"(插入|新增一页|新增一张|添加一页|添加一张|增加一页|增加一张|补一页|加一页|append a slide|insert a slide|add a slide)",
            r"(在|于).{0,12}(前|后|之前|之后|结尾|最后|最后一页|结论页|第\d+页).{0,10}(插入|新增|添加|增加|补充|append|insert|add)",
            r"(删除|删掉|移除|去掉|remove|delete).{0,10}(第\d+页|该页|这页|这一页|页面|页|slide)",
            r"(移动|挪到|调到|放到|移到|重排|调整顺序|重新排序|move|reorder).{0,20}(第\d+页|前面|后面|最后|开头|页面|页|slide)",
        )
        return any(re.search(pattern, normalized) for pattern in structural_patterns)

    def _build_modify_execution_plan(
        self,
        *,
        user_message: str,
        intent: dict[str, Any] | None,
        wm_rule_specs_text: str,
    ) -> ModifyExecutionPlan | None:
        """Compile current-turn global intent into an explicit modify plan."""
        return _runtime_support.build_modify_execution_plan(
            self,
            user_message=user_message,
            intent=intent,
            wm_rule_specs_text=wm_rule_specs_text,
        )
        all_slide_paths = self._resolve_all_slide_paths()
        if not all_slide_paths:
            return None

        intent = intent or {}
        intent_target = str(intent.get("target_slide", "") or "").strip()
        specific_slide_path = None
        if intent_target and intent_target.lower() != "all":
            specific_slide_path = self._resolve_slide_path(intent_target)

        all_rule_specs = self._parse_rule_specs_from_injection(wm_rule_specs_text)
        current_turn_specs: list[dict[str, Any]] = []
        for spec in all_rule_specs:
            try:
                source_turn = int(spec.get("source_turn", -1))
            except (TypeError, ValueError):
                source_turn = -1
            if source_turn == self._modify_turn_count:
                current_turn_specs.append(spec)

        structural_slide_operation = self._is_structural_slide_operation(user_message, intent)
        user_has_global_signal = (
            self._session_preference_has_general_signal(user_message)
            or intent_target.lower() == "all"
        ) and not structural_slide_operation
        current_turn_global_specs = [
            spec
            for spec in current_turn_specs
            if str(spec.get("scope", "") or "").strip().lower() == "global"
        ]
        current_turn_existing_specs = [
            spec for spec in current_turn_global_specs
            if self._rule_spec_applies_to_existing_slides(spec)
        ]
        current_turn_future_only_specs = [
            spec for spec in current_turn_global_specs
            if not self._rule_spec_applies_to_existing_slides(spec)
        ]

        if structural_slide_operation and not current_turn_existing_specs:
            selector_hints: list[str] = []
            rule_ids: list[str] = []
            for spec in current_turn_future_only_specs:
                rule_id = str(spec.get("rule_id", "") or "").strip()
                if rule_id and rule_id not in rule_ids:
                    rule_ids.append(rule_id)
                target = spec.get("target")
                if isinstance(target, dict):
                    for hint in self._selector_hints_for_element_kind(
                        str(target.get("element_kind", "") or "").strip()
                    ):
                        if hint not in selector_hints:
                            selector_hints.append(hint)

            if specific_slide_path is not None:
                reason = (
                    "Structural slide operation detected; treat this as a deck-level page operation "
                    "anchored to the resolved slide, without requiring batch edits on existing slides."
                )
                if current_turn_future_only_specs:
                    reason += " Future-only global rules will apply to the inserted slide once it exists."
                return ModifyExecutionPlan(
                    scope="global",
                    reason=reason,
                    target_slide_paths=[specific_slide_path],
                    target_rule_ids=rule_ids,
                    selector_hints=selector_hints,
                    coverage_required=False,
                    operation_kind="structural",
                )

            if current_turn_future_only_specs:
                return ModifyExecutionPlan(
                    scope="global",
                    reason=(
                        "Structural slide operation detected with future-only global rules; "
                        "do not require deck-wide edits on existing slides."
                    ),
                    target_slide_paths=[],
                    target_rule_ids=rule_ids,
                    selector_hints=selector_hints,
                    coverage_required=False,
                    operation_kind="structural",
                )

            return ModifyExecutionPlan(
                scope="global",
                reason=(
                    "Structural slide operation detected; use deck-level insert/delete/new-slide tools "
                    "instead of narrowing this turn to existing-slide patch tools."
                ),
                target_slide_paths=[],
                target_rule_ids=rule_ids,
                selector_hints=selector_hints,
                coverage_required=False,
                operation_kind="structural",
            )

        if not user_has_global_signal and not current_turn_existing_specs:
            if specific_slide_path is not None:
                return ModifyExecutionPlan(
                    scope="local",
                    reason="Current request resolves to a specific slide without a new deck-level rule.",
                    target_slide_paths=[specific_slide_path],
                    target_rule_ids=[],
                    selector_hints=[],
                    coverage_required=False,
                    operation_kind="style",
                )
            return None

        scope: Literal["global", "hybrid"]
        scope = "hybrid" if specific_slide_path is not None else "global"
        target_slide_paths = list(all_slide_paths)

        selector_hints: list[str] = []
        rule_ids: list[str] = []
        active_specs = current_turn_existing_specs or all_rule_specs
        inferred_element_kind = self._infer_session_rule_element_kind(user_message)

        for spec in active_specs:
            rule_id = str(spec.get("rule_id", "") or "").strip()
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(rule_id)
            target = spec.get("target")
            if isinstance(target, dict):
                for hint in self._selector_hints_for_element_kind(
                    str(target.get("element_kind", "") or "").strip()
                ):
                    if hint not in selector_hints:
                        selector_hints.append(hint)

        if not selector_hints:
            for hint in self._selector_hints_for_element_kind(inferred_element_kind):
                if hint not in selector_hints:
                    selector_hints.append(hint)

        reason_parts: list[str] = []
        if user_has_global_signal:
            reason_parts.append("The current user message contains deck-level/global language.")
        if current_turn_existing_specs:
            reason_parts.append(
                f"{len(current_turn_existing_specs)} structured existing-slide rule(s) were extracted for this turn."
            )
        elif current_turn_future_only_specs:
            reason_parts.append(
                f"{len(current_turn_future_only_specs)} future-only rule(s) were extracted for this turn."
            )
        if scope == "hybrid" and specific_slide_path is not None:
            reason_parts.append(
                f"A specific target slide was also inferred ({specific_slide_path.name})."
            )

        return ModifyExecutionPlan(
            scope=scope,
            reason=" ".join(reason_parts) or "Deck-level modification plan generated.",
            target_slide_paths=target_slide_paths,
            target_rule_ids=rule_ids,
            selector_hints=selector_hints,
            coverage_required=True,
            operation_kind="style",
        )

    def _render_modify_execution_plan(self, plan: ModifyExecutionPlan) -> str:
        """Render a model-facing execution contract for modify turns."""
        return _runtime_support.render_modify_execution_plan(plan, runtime=self)

    def _resolve_modify_coverage_path(self, raw_path: str) -> Path | None:
        """Resolve a tool argument path back to the current workspace slide file."""
        text = str(raw_path or "").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = self.workspace / path
        if path.exists():
            return path.resolve()
        if path.suffix.lower() == ".html":
            alias = self._resolve_slide_path(path.stem)
            if alias is not None:
                return alias.resolve()
        return None

    def _extract_modify_coverage_paths(
        self,
        *,
        tool_name: str,
        arguments: str | dict[str, Any] | None,
        result_text: str,
    ) -> list[Path]:
        """Collect slide paths covered by a successful modify tool call."""
        parsed_args: dict[str, Any] = {}
        if isinstance(arguments, dict):
            parsed_args = arguments
        elif isinstance(arguments, str) and arguments.strip():
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = {}

        def _extract_path_candidates_from_text(text: str, field_names: tuple[str, ...]) -> list[str]:
            raw = str(text or "").strip()
            if not raw:
                return []

            candidates: list[str] = []

            def _add(candidate: Any) -> None:
                text_value = str(candidate or "").strip()
                if text_value and text_value not in candidates:
                    candidates.append(text_value)

            payload = self._extract_json_object(raw)
            if isinstance(payload, dict):
                for field_name in field_names:
                    _add(payload.get(field_name, ""))
                results = payload.get("results", [])
                if isinstance(results, list):
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        for field_name in field_names:
                            _add(item.get(field_name, ""))

            for field_name in field_names:
                key_pattern = r"[_\s-]*".join(re.escape(part) for part in field_name.split("_") if part)
                if not key_pattern:
                    continue
                pattern = re.compile(
                    rf'["\']?(?:{key_pattern})["\']?\s*[:=]\s*["\']?(?P<path>[^\n\r"\'`{{}}\[\],]+)',
                    re.IGNORECASE,
                )
                for match in pattern.finditer(raw):
                    text_value = match.group("path").strip().rstrip(".,;:)")
                    if text_value and text_value not in candidates:
                        candidates.append(text_value)

            return candidates

        paths: list[Path] = []
        if tool_name in {"write_html_file", "patch_semantic_inline_style", "apply_slide_patch"}:
            if tool_name in {"patch_semantic_inline_style", "apply_slide_patch"}:
                payload = self._extract_json_object(result_text or "")
                if isinstance(payload, dict) and payload.get("success") is False:
                    return []
                if not payload and re.search(
                    r'["\']?success["\']?\s*[:=]\s*false',
                    str(result_text or ""),
                    re.IGNORECASE,
                ):
                    return []

            field_names = ("slide_path", "file_path") if tool_name == "apply_slide_patch" else ("file_path", "path")
            for raw_path in _extract_path_candidates_from_text(result_text, field_names):
                candidate = self._resolve_modify_coverage_path(raw_path)
                if candidate is not None and candidate not in paths:
                    paths.append(candidate)
            if not paths and isinstance(parsed_args, dict):
                candidate = self._resolve_modify_coverage_path(parsed_args.get("file_path", ""))
                if candidate is not None:
                    paths.append(candidate)
            return paths

        if tool_name == "insert_slide":
            anchor_path = self._resolve_modify_coverage_path(
                parsed_args.get("target_slide", "") if isinstance(parsed_args, dict) else ""
            )
            if anchor_path is not None:
                paths.append(anchor_path)

            match = re.search(
                r"New slide path:\s*(?P<path>[^\n\r]+)",
                str(result_text or ""),
                re.IGNORECASE,
            )
            if match:
                new_path = self._resolve_modify_coverage_path(match.group("path").strip())
                if new_path is not None and new_path not in paths:
                    paths.append(new_path)
            return paths

        if tool_name in {"batch_update_css_rule", "batch_update_semantic_style"}:
            payload = self._extract_json_object(result_text or "")
            if isinstance(payload, dict) and payload.get("success") is False:
                return []
            if not payload and re.search(
                r'["\']?success["\']?\s*[:=]\s*false',
                str(result_text or ""),
                re.IGNORECASE,
            ):
                return []

            results = payload.get("results", []) if isinstance(payload, dict) else []
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status", "") or "").strip().lower()
                    if status == "error":
                        continue
                    candidate = self._resolve_modify_coverage_path(item.get("file_path", ""))
                    if candidate is not None and candidate not in paths:
                        paths.append(candidate)
            if not paths:
                for raw_path in _extract_path_candidates_from_text(result_text, ("file_path",)):
                    candidate = self._resolve_modify_coverage_path(raw_path)
                    if candidate is not None and candidate not in paths:
                        paths.append(candidate)
            if paths:
                return paths
            for raw_path in parsed_args.get("file_paths", []) if isinstance(parsed_args, dict) else []:
                candidate = self._resolve_modify_coverage_path(raw_path)
                if candidate is not None:
                    paths.append(candidate)
        return paths

    def _build_modify_plan_followup(self, uncovered_paths: list[Path]) -> str:
        labels = ", ".join(self._workspace_relative_label(path) for path in uncovered_paths[:12])
        return (
            "SYSTEM: The global modify plan is still incomplete. "
            f"Remaining target slides: {labels}. "
            "Use `scan_slide_index` if you still need a deck summary, then prefer "
            "`read_slide_snapshot` + `apply_slide_patch` for existing-slide content or local structure edits, "
            "prioritizing `repair_candidates`, `risk_flags`, exposed `rules`, `slide_canvas`, and `layout_container` for layout fixes, "
            "using `merge_css_rule` / `replace_css_rule` / `merge_style` before shrinking text, "
            "using `set_attr` / `remove_attr` only for safe attributes, "
            "and making sure `apply_slide_patch` includes a non-empty `patch_ops` list built from repair candidates, rules, or snapshot targets. "
            "If `apply_slide_patch` returns `STALE_SNAPSHOT`, use the fresh snapshot and `rebind_hints` to retry instead of switching straight to full rewrite. "
            "`batch_update_semantic_style` for semantic batch edits, "
            "`batch_update_css_rule` for explicit-selector edits, "
            "`patch_semantic_inline_style` for single-slide exception repair, "
            "and reserve `write_new_slide_file` for brand-new inserted slides while using `write_html_file` only for generation or corrupted-slide recovery. "
            "Do not call `finalize` until these slides are covered."
        )

    def _template_user_candidates(self) -> list[str]:
        """Template retrieval should only use the current user scope."""
        user_id = (self.user_id or "").strip()
        return [user_id] if user_id else []

    @staticmethod
    def _normalize_template_lookup_text(text: str) -> str:
        normalized = (text or "").lower()
        normalized = normalized.replace("模板", "")
        normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)
        return normalized

    def _resolve_template_from_selection(
        self,
        templates: list[Any],
        *,
        template_id: str = "",
        template_name: str = "",
    ) -> Any | None:
        template_id = (template_id or "").strip()
        template_name = (template_name or "").strip()

        if template_id:
            for template in templates:
                if getattr(template, "id", "") == template_id:
                    return template

        if template_name:
            return self._pick_template_by_name_hint(templates, template_name)

        return None

    async def _list_accessible_templates(self, store: Any, limit: int = 20) -> list[Any]:
        templates: list[Any] = []
        seen_names: set[str] = set()
        per_user_limit = max(limit, 1)
        for user_id in self._template_user_candidates():
            try:
                for template in await store.list_by_user(user_id, limit=per_user_limit):
                    key = self._normalize_template_lookup_text(getattr(template, "name", ""))
                    if key and key in seen_names:
                        continue
                    if key:
                        seen_names.add(key)
                    templates.append(template)
                    if len(templates) >= limit:
                        return templates
            except Exception as e:
                logger.debug("Failed to list templates for %s: %s", user_id, e)
        return templates

    def _pick_template_by_name_hint(
        self,
        templates: list[Any],
        template_query: str,
    ) -> Any | None:
        normalized_query = self._normalize_template_lookup_text(template_query)
        if not normalized_query:
            return None

        for template in templates:
            normalized_name = self._normalize_template_lookup_text(
                getattr(template, "name", "")
            )
            if normalized_name and normalized_name == normalized_query:
                return template

        for template in templates:
            normalized_name = self._normalize_template_lookup_text(
                getattr(template, "name", "")
            )
            if not normalized_name:
                continue
            if normalized_query in normalized_name or normalized_name in normalized_query:
                return template
        return None

    @staticmethod
    def _filter_template_preferences_for_matching(
        preferences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Template matching should ignore semantic template prefs and rely on usage history."""
        from memslides.memory.core.models import is_template_related_preference

        filtered: list[dict[str, Any]] = []
        for pref in preferences:
            if is_template_related_preference(
                dimension=str(pref.get("conflict_group", "") or ""),
                content=pref.get("preference", ""),
                trigger=pref.get("trigger", ""),
                rationale=pref.get("rationale", ""),
            ):
                continue
            filtered.append(pref)
        return filtered

    @staticmethod
    def _template_profile_context_from_profile(profile: Any | None) -> dict[str, Any]:
        if profile is None:
            return {}
        context: dict[str, Any] = {}

        template_pref = getattr(profile, "template", None)
        if template_pref is not None:
            if hasattr(template_pref, "to_usage_selection_context"):
                context.update(template_pref.to_usage_selection_context())
            elif hasattr(template_pref, "to_selection_context"):
                context.update(template_pref.to_selection_context())

        def _collect_texts(values: list[Any]) -> list[str]:
            texts: list[str] = []
            for value in values:
                text = str(value or "").strip()
                if text:
                    texts.append(text)
            return texts

        def _contains_any(source: str, keywords: tuple[str, ...]) -> bool:
            return any(keyword in source for keyword in keywords)

        theme = getattr(profile, "theme", None)
        visual = getattr(profile, "visual", None)
        layout = getattr(profile, "layout", None)
        content = getattr(profile, "content", None)
        general = getattr(profile, "general", None)

        raw_texts: list[str] = []
        if theme is not None:
            raw_texts.extend(_collect_texts(getattr(theme, "primary_colors", [])))
            raw_texts.extend(_collect_texts(getattr(theme, "accent_colors", [])))
            raw_texts.extend(_collect_texts([
                getattr(theme, "font_family", ""),
                getattr(theme, "background_style", ""),
            ]))
            raw_texts.extend(_collect_texts(getattr(theme, "notes", [])))
        if visual is not None:
            raw_texts.extend(_collect_texts([
                getattr(visual, "image_style", ""),
                getattr(visual, "icon_usage", ""),
                getattr(visual, "animation_preference", ""),
            ]))
            raw_texts.extend(_collect_texts(getattr(visual, "chart_type_priority", [])))
            raw_texts.extend(_collect_texts(getattr(visual, "notes", [])))
        if layout is not None:
            raw_texts.extend(_collect_texts([
                getattr(layout, "content_density", ""),
                getattr(layout, "alignment_style", ""),
                getattr(layout, "spacing_preference", ""),
                getattr(layout, "slide_structure", ""),
            ]))
            raw_texts.extend(_collect_texts(getattr(layout, "notes", [])))
        if content is not None:
            raw_texts.extend(_collect_texts([
                getattr(content, "text_density", ""),
                getattr(content, "language_style", ""),
                getattr(content, "bullet_point_style", ""),
                getattr(content, "title_length", ""),
            ]))
            raw_texts.extend(_collect_texts(getattr(content, "notes", [])))
        if general is not None:
            raw_texts.extend(_collect_texts(getattr(general, "preferences", [])))

        combined = " ".join(raw_texts).lower()
        style_defaults: dict[str, Any] = {}

        narrative_style_map = {
            "academic": ("academic", "学术", "论文", "research", "研究", "progress review"),
            "business": ("business", "商务", "商业", "管理", "executive", "汇报"),
            "technical": ("technical", "技术", "method", "workflow", "方法", "流程"),
            "educational": ("educational", "教育", "教学", "课堂", "course unit", "课程"),
            "creative": ("creative", "创意", "storytelling", "叙事", "品牌化"),
        }
        for label, keywords in narrative_style_map.items():
            if _contains_any(combined, keywords):
                style_defaults["narrative_style"] = label
                break

        density_source = " ".join(_collect_texts([
            getattr(layout, "content_density", "") if layout is not None else "",
            getattr(content, "text_density", "") if content is not None else "",
            getattr(content, "title_length", "") if content is not None else "",
        ])).lower()
        if _contains_any(density_source, ("low", "sparse", "minimal", "简洁", "简约", "留白", "短标题")):
            style_defaults["info_density"] = "low"
        elif _contains_any(density_source, ("high", "dense", "detailed", "紧凑", "详细", "丰富", "长标题")):
            style_defaults["info_density"] = "high"
        elif density_source:
            style_defaults["info_density"] = "medium"

        layout_notes = getattr(layout, "notes", []) if layout is not None else []
        layout_source = " ".join(_collect_texts([
            getattr(layout, "slide_structure", "") if layout is not None else "",
            getattr(layout, "alignment_style", "") if layout is not None else "",
            getattr(layout, "spacing_preference", "") if layout is not None else "",
            *layout_notes,
        ])).lower()
        if _contains_any(layout_source, ("minimal", "简约", "极简", "留白")):
            style_defaults["layout_preference"] = "minimal"
        elif _contains_any(layout_source, ("dense", "紧凑", "多栏", "grid", "网格")):
            style_defaults["layout_preference"] = "dense"
        elif layout_source:
            style_defaults["layout_preference"] = "balanced"

        theme_primary_colors = getattr(theme, "primary_colors", []) if theme is not None else []
        theme_accent_colors = getattr(theme, "accent_colors", []) if theme is not None else []
        theme_notes = getattr(theme, "notes", []) if theme is not None else []
        theme_source = " ".join(_collect_texts([
            getattr(theme, "background_style", "") if theme is not None else "",
            *theme_primary_colors,
            *theme_accent_colors,
            *theme_notes,
        ])).lower()
        if _contains_any(theme_source, ("dark", "深色", "黑", "navy", "墨")):
            style_defaults["color_tone"] = "dark"
        elif _contains_any(theme_source, ("warm", "暖", "橙", "红", "金")):
            style_defaults["color_tone"] = "warm"
        elif _contains_any(theme_source, ("cool", "冷", "蓝", "青", "绿")):
            style_defaults["color_tone"] = "cool"
        elif _contains_any(theme_source, ("vibrant", "鲜艳", "高饱和", "亮色")):
            style_defaults["color_tone"] = "vibrant"
        elif theme_source:
            style_defaults["color_tone"] = "neutral"

        reference_style = ""
        for text in raw_texts:
            lowered = text.lower()
            if "像" in text or "类似" in text or "like " in lowered or "similar to" in lowered:
                reference_style = text
                break
        if reference_style:
            style_defaults["reference_style"] = reference_style

        query_terms: list[str] = []
        for value in [
            style_defaults.get("narrative_style", ""),
            style_defaults.get("layout_preference", ""),
            style_defaults.get("color_tone", ""),
            getattr(layout, "slide_structure", "") if layout is not None else "",
            getattr(theme, "background_style", "") if theme is not None else "",
        ]:
            text = str(value or "").strip()
            if text and text not in query_terms:
                query_terms.append(text)

        if style_defaults:
            context["profile_style_defaults"] = style_defaults
            context["profile_style_signals"] = raw_texts[:8]
        if query_terms:
            context["profile_template_query_terms"] = query_terms[:5]

        return context

    @staticmethod
    def _build_runtime_template_summary_from_usage_history(
        usage_history: list[Any],
    ) -> Any | None:
        """Build a read-time template summary without mutating persisted profile state."""
        if not usage_history:
            return None
        from memslides.memory.core.models import TemplatePreference

        runtime_template_pref = TemplatePreference()
        runtime_template_pref.refresh_from_usage_records(
            usage_history,
            reset_semantic_fields=True,
        )
        return runtime_template_pref

    async def _load_usage_history_for_template_memory(
        self,
        store: Any,
        *,
        request_intent: str,
        aspect_ratio: str,
        limit: int = 20,
        success_only: bool = False,
        allow_unscoped_fallback: bool = True,
    ) -> list[Any]:
        """优先按 intent / aspect_ratio 取模板使用历史，旧数据不足时再回退。"""
        attempts: list[tuple[str, str]] = [
            (request_intent, aspect_ratio),
            (request_intent, ""),
        ]
        if allow_unscoped_fallback:
            attempts.extend([
                ("", aspect_ratio),
                ("", ""),
            ])

        deduped_attempts: list[tuple[str, str]] = []
        for candidate in attempts:
            if candidate not in deduped_attempts:
                deduped_attempts.append(candidate)

        for memory_intent, ratio in deduped_attempts:
            if (
                allow_unscoped_fallback
                and not memory_intent
                and request_intent == "default"
                and ratio == aspect_ratio
            ):
                continue
            try:
                history = await store.get_usage_history(
                    user_id=self.user_id,
                    limit=limit,
                    success_only=success_only,
                    memory_intent=memory_intent,
                    aspect_ratio=ratio,
                )
            except Exception as e:
                logger.debug(
                    "Failed to load template usage history (intent=%s, ratio=%s): %s",
                    memory_intent,
                    ratio,
                    e,
                )
                continue
            if history:
                return history
        return []

    async def _sync_template_profile_from_usage_history(
        self,
        request: InputRequest,
    ) -> None:
        """把 template_usage_history 派生为 intent 画像下的模板摘要。"""
        if not self.memory_system:
            return
        if self._freeze_preference_writeback_current_job:
            return

        store = getattr(self.memory_system, "template_store", None)
        profile_store = getattr(self.memory_system, "profile_store", None)
        if not store or not profile_store:
            return

        write_intent = self._get_request_memory_write_intent(request)
        aspect_ratio = getattr(request.powerpoint_type, "value", "") or ""
        usage_history = await self._load_usage_history_for_template_memory(
            store,
            request_intent=write_intent,
            aspect_ratio=aspect_ratio,
            limit=50,
            success_only=True,
            allow_unscoped_fallback=False,
        )
        if not usage_history:
            return

        try:
            profile = await profile_store.get(self.user_id, intent=write_intent)
            template_pref = getattr(profile, "template", None)
            if (
                template_pref is not None
                and hasattr(template_pref, "refresh_from_usage_records")
                and template_pref.refresh_from_usage_records(
                    usage_history,
                    reset_semantic_fields=True,
                )
            ):
                await profile_store.save(profile, intent=write_intent)
                info(
                    "[Stage 14] Refreshed template profile summary from usage history "
                    f"(write_intent={write_intent}, records={len(usage_history)})"
                )
        except Exception as e:
            logger.debug("template profile sync failed: %s", e)

    async def _search_templates_by_style(
        self,
        store: Any,
        *,
        narrative_style: str = "",
        info_density: str = "",
        color_tone: str = "",
        aspect_ratio: str | None = None,
        limit: int = 5,
    ) -> list[Any]:
        for user_id in self._template_user_candidates():
            try:
                results = await store.search_by_style(
                    narrative_style=narrative_style,
                    info_density=info_density,
                    color_tone=color_tone,
                    user_id=user_id,
                    aspect_ratio=aspect_ratio,
                    limit=limit,
                )
                if results:
                    return results
            except Exception as e:
                logger.debug("search_by_style failed for %s: %s", user_id, e)
        return []

    async def _search_templates_by_query(
        self,
        store: Any,
        query: str,
        *,
        aspect_ratio: str | None = None,
        limit: int = 5,
    ) -> list[Any]:
        for user_id in self._template_user_candidates():
            try:
                results = await store.search(
                    query=query,
                    user_id=user_id,
                    aspect_ratio=aspect_ratio,
                    limit=limit,
                )
                if results:
                    return results
            except Exception as e:
                logger.debug("template search failed for %s: %s", user_id, e)
        return []

    async def _get_recent_template(self, store: Any, limit: int = 1) -> list[Any]:
        for user_id in self._template_user_candidates():
            try:
                results = await store.get_recent(user_id=user_id, limit=limit)
                if results:
                    return results
            except Exception as e:
                logger.debug("get_recent failed for %s: %s", user_id, e)
        return []

    async def _auto_match_template_profile(
        self,
        request: InputRequest,
    ) -> tuple[Any | None, Any | None]:
        return await _runtime_support.auto_match_template_profile(self, request)
        if not self.memory_system:
            return None, None

        store = getattr(self.memory_system, "template_store", None)
        if not store:
            return None, None

        try:
            from memslides.memory.style_intent_classifier import (
                StyleIntentClassifier,
                TemplateIntent,
            )
        except ImportError as e:
            logger.warning("StyleIntentClassifier not available: %s", e)
            return None, None

        accessible_templates = await self._list_accessible_templates(store, limit=50)
        available_names = [
            t.name for t in accessible_templates if getattr(t, "name", "")
        ]
        read_intent = self._get_request_memory_read_intent(request)
        aspect_ratio = getattr(request.powerpoint_type, "value", "") or ""

        user_preferences: list[dict] = []
        pref_store = getattr(self.memory_system, "preference_store", None)
        if pref_store:
            try:
                prefs = await pref_store.get_active(self.user_id, limit=10)
                user_preferences = self._filter_template_preferences_for_matching(
                    [p.to_dict() for p in prefs]
                )
            except Exception as e:
                logger.debug("Failed to load preferences for style classify: %s", e)

        profile = None
        template_profile_context: dict[str, Any] = {}
        profile_store = getattr(self.memory_system, "profile_store", None)
        if profile_store:
            try:
                profile = await profile_store.get(self.user_id, intent=read_intent)
            except Exception as e:
                logger.debug(
                    "Failed to load user profile for template matching: %s", e
                )

        usage_history = await self._load_usage_history_for_template_memory(
            store,
            request_intent=read_intent,
            aspect_ratio=aspect_ratio,
            limit=12,
        )
        runtime_template_summary = self._build_runtime_template_summary_from_usage_history(
            usage_history
        )
        base_profile_context = self._template_profile_context_from_profile(profile)
        if runtime_template_summary is not None and hasattr(
            runtime_template_summary, "to_usage_selection_context"
        ):
            template_profile_context = {
                **base_profile_context,
                **runtime_template_summary.to_usage_selection_context(),
            }
        else:
            template_profile_context = base_profile_context

        llm_objects = getattr(self.memory_system, "llm_objects_by_task", {}) or {}
        style_llm = (
            llm_objects.get("style_classify")
            or llm_objects.get("intent_classify")
            or getattr(self.memory_system, "llm", None)
        )
        classifier = StyleIntentClassifier(
            llm=style_llm,
            artifact_writer=getattr(self.memory_system, "artifact_writer", None),
            use_llm=style_llm is not None,
        )
        style_intent_result = await classifier.classify(
            user_message=request.instruction,
            user_preferences=user_preferences,
            available_templates=available_names,
            template_store=store,
            user_id=self.user_id,
            embedding_func=getattr(self.memory_system, "embedding_func", None),
            memory_intent=read_intent,
            template_profile_context=template_profile_context,
        )
        info(
            f"[Stage 14] Style intent: {style_intent_result.template_intent.value} "
            f"(confidence={style_intent_result.confidence:.2f})"
        )
        self._update_template_match_state(
            request,
            template_intent=style_intent_result.template_intent.value,
            template_intent_confidence=style_intent_result.confidence,
            template_reasoning=style_intent_result.reasoning,
            style_memory_hint=style_intent_result.memory_hint,
            template_query=style_intent_result.template_query,
        )

        if style_intent_result.template_intent in (
            TemplateIntent.NONE,
            TemplateIntent.ANTI,
        ):
            self._update_template_match_state(
                request,
                selection_source=f"skip_{style_intent_result.template_intent.value}",
                selection_confidence=style_intent_result.confidence,
                selection_reasoning=style_intent_result.reasoning,
                matched_by_history=False,
                matched_history_intent="",
            )
            return None, style_intent_result

        selector_llm = (
            llm_objects.get("template_analyze")
            or llm_objects.get("style_classify")
            or llm_objects.get("intent_classify")
            or getattr(self.memory_system, "llm", None)
        )
        if accessible_templates and selector_llm is not None:
            try:
                from memslides.memory.template_selector import TemplateSelector

                selector = TemplateSelector(
                    llm=selector_llm,
                    artifact_writer=getattr(self.memory_system, "artifact_writer", None),
                )
                selection = await selector.select(
                    user_message=request.instruction,
                    style_intent=style_intent_result.to_dict(),
                    templates=accessible_templates,
                    usage_history=[record.to_dict() for record in usage_history],
                    user_preferences=user_preferences,
                    template_profile_context=template_profile_context,
                    user_id=self.user_id,
                    aspect_ratio=aspect_ratio,
                )
                matched = self._resolve_template_from_selection(
                    accessible_templates,
                    template_id=selection.selected_template_id,
                    template_name=selection.selected_template_name,
                )
                if matched is not None and selection.should_use_template:
                    matched_by_history = style_intent_result.memory_hint in {
                        "similar_scenario",
                        "last_template",
                        "previous_template",
                        "that_template",
                    }
                    self._update_template_match_state(
                        request,
                        selected_template_id=getattr(matched, "id", "") or getattr(matched, "template_id", ""),
                        selected_template_name=getattr(matched, "name", ""),
                        selection_source="llm_selector",
                        selection_confidence=selection.confidence,
                        selection_reasoning=selection.reasoning,
                        matched_by_history=matched_by_history,
                        matched_history_intent=read_intent if matched_by_history else "",
                    )
                    info(
                        f"[Stage 14] LLM-selected template: {matched.name} "
                        f"(confidence={selection.confidence:.2f})"
                    )
                    return matched, style_intent_result
                if selection.should_use_template:
                    logger.warning(
                        "LLM template selector returned an unknown template "
                        "(id=%s, name=%s); falling back to heuristic matching",
                        selection.selected_template_id,
                        selection.selected_template_name,
                    )
                else:
                    self._update_template_match_state(
                        request,
                        selection_source="llm_selector_skip",
                        selection_confidence=selection.confidence,
                        selection_reasoning=selection.reasoning,
                        selected_template_id="",
                        selected_template_name="",
                        matched_by_history=False,
                        matched_history_intent="",
                    )
                    info(
                        "[Stage 14] LLM template selector chose to skip template "
                        f"(reason={selection.reasoning or 'no suitable template'})"
                    )
                    return None, style_intent_result
            except Exception as e:
                logger.warning(
                    "LLM template selection failed, falling back to heuristic matching: %s",
                    e,
                )

        matched_templates: list[Any] = []
        selection_source = ""

        if style_intent_result.memory_hint in {
            "last_template",
            "previous_template",
            "that_template",
        }:
            matched_templates = await self._get_recent_template(store, limit=1)
            if matched_templates:
                selection_source = "recent_history"

        if not matched_templates and style_intent_result.template_query:
            exact_match = self._pick_template_by_name_hint(
                accessible_templates, style_intent_result.template_query
            )
            if exact_match is not None:
                matched_templates = [exact_match]
                selection_source = "name_hint"

        if (
            not matched_templates
            and not style_intent_result.style_attributes.is_empty()
        ):
            matched_templates = await self._search_templates_by_style(
                store,
                narrative_style=style_intent_result.style_attributes.narrative_style,
                info_density=style_intent_result.style_attributes.info_density,
                color_tone=style_intent_result.style_attributes.color_tone,
                aspect_ratio=request.powerpoint_type.value,
                limit=3,
            )
            if matched_templates:
                selection_source = "style_search"

        if not matched_templates and style_intent_result.template_query:
            matched_templates = await self._search_templates_by_query(
                store,
                style_intent_result.template_query,
                aspect_ratio=request.powerpoint_type.value,
                limit=3,
            )
            if matched_templates:
                selection_source = "query_search"

        if matched_templates:
            matched = matched_templates[0]
            matched_by_history = selection_source == "recent_history" or style_intent_result.memory_hint == "similar_scenario"
            self._update_template_match_state(
                request,
                selected_template_id=getattr(matched, "id", "") or getattr(matched, "template_id", ""),
                selected_template_name=getattr(matched, "name", ""),
                selection_source=selection_source or "heuristic",
                selection_confidence=style_intent_result.confidence,
                selection_reasoning=style_intent_result.reasoning,
                matched_by_history=matched_by_history,
                matched_history_intent=read_intent if matched_by_history else "",
            )
            info(
                f"[Stage 14] Auto-matched template: {matched.name} "
                f"(user={getattr(matched, 'user_id', '') or 'default'})"
            )
            return matched, style_intent_result

        self._update_template_match_state(
            request,
            selection_source="freeform_fallback",
            selection_confidence=style_intent_result.confidence,
            selection_reasoning=style_intent_result.reasoning,
            selected_template_id="",
            selected_template_name="",
            matched_by_history=False,
            matched_history_intent="",
        )
        info("[Stage 14] No template matched, falling back to freeform generation")
        return None, style_intent_result

    def _activate_template_profile(
        self,
        request: InputRequest,
        template_profile: Any,
    ) -> tuple[Any, list[dict]]:
        return _runtime_support.activate_template_profile(self, request, template_profile)
        from memslides.memory.inject.template_guide_builder import TemplateGuideBuilder

        guide_builder = TemplateGuideBuilder(template_profile)
        self._template_profile = template_profile
        self._guide_builder = guide_builder
        self._current_template_id = (
            getattr(template_profile, "id", "")
            or getattr(template_profile, "template_id", "")
            or ""
        )

        if self._current_template_id and not request.template_id:
            request.template_id = self._current_template_id

        template_source = getattr(template_profile, "template_source", "") or ""
        if template_source and Path(template_source).exists():
            request.template = template_source
            request.template_as_reference = True

        template_conflicts = guide_builder.detect_conflicts(request.instruction)
        return guide_builder, template_conflicts

    async def _record_template_usage(
        self,
        request: InputRequest,
        template_profile: Any,
        style_intent_result: Any = None,
        success: bool = True,
    ) -> None:
        if not self.memory_system or not template_profile:
            return

        store = getattr(self.memory_system, "template_store", None)
        if not store:
            return

        template_id = getattr(template_profile, "id", "") or getattr(
            template_profile, "template_id", ""
        )
        if not template_id:
            return

        if success:
            try:
                await store.mark_used(template_id)
            except Exception as e:
                logger.debug("mark_used failed for %s: %s", template_id, e)

        try:
            from memslides.memory.core.models import TemplateUsageRecord
        except ImportError:
            return

        message_embedding = None
        embedding_func = getattr(self.memory_system, "embedding_func", None)
        if embedding_func:
            try:
                embeddings = await embedding_func([request.instruction[:500]])
                if len(embeddings) > 0:
                    first = embeddings[0]
                    message_embedding = (
                        first.tolist() if hasattr(first, "tolist") else list(first)
                    )
            except Exception as e:
                logger.debug("template usage embedding failed: %s", e)

        attachment_type = ""
        if request.attachments:
            suffix = Path(request.attachments[0]).suffix.lower().lstrip(".")
            if suffix in {"ppt", "pptx"}:
                attachment_type = "pptx"
            elif suffix in {"pdf"}:
                attachment_type = "pdf"
            elif suffix in {"doc", "docx", "md", "txt"}:
                attachment_type = "doc"
            elif suffix in {"png", "jpg", "jpeg", "webp"}:
                attachment_type = "image"

        intent = "explicit"
        if style_intent_result is not None and hasattr(style_intent_result, "template_intent"):
            intent = style_intent_result.template_intent.value

        record = TemplateUsageRecord(
            user_id=self.user_id,
            template_id=template_id,
            template_name=getattr(template_profile, "name", ""),
            user_message=request.instruction[:500],
            user_message_embedding=message_embedding,
            intent=intent,
            memory_intent=self._get_request_memory_write_intent(request),
            aspect_ratio=getattr(request.powerpoint_type, "value", "") or "",
            has_attachment=bool(request.attachments),
            attachment_type=attachment_type,
            success=success,
        )
        try:
            await store.add_usage_record(record)
        except Exception as e:
            logger.debug("add_usage_record failed: %s", e)
            return

        await self._sync_template_profile_from_usage_history(request)

    async def _record_explicit_template_seed_usage(
        self,
        request: InputRequest,
        template_profile: Any,
    ) -> bool:
        if self._template_usage_seeded or not request.template_as_reference or not template_profile:
            return False

        await self._record_template_usage(
            request=request,
            template_profile=template_profile,
            style_intent_result=None,
            success=True,
        )
        self._template_usage_seeded = True
        info(
            "[Stage 14] Seeded explicit template usage history "
            f"(template={getattr(template_profile, 'name', '') or getattr(template_profile, 'id', '')}, "
            f"write_intent={self._get_request_memory_write_intent(request)})"
        )
        return True

    async def _run_generation_impl(
        self,
        request: InputRequest,
        check_llms: bool = False,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in self.run(request, check_llms=check_llms):
            yield item

    @timer("MemSlides Loop")
    async def run(
        self,
        request: InputRequest,
        check_llms: bool = False,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in self._run_generation_flow(request, check_llms=check_llms):
            yield item

    async def _run_generation_flow(
        self,
        request: InputRequest,
        check_llms: bool = False,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """Compatibility wrapper; generation orchestration lives in pipelines.generation."""
        from memslides.pipelines.generation import run_generation_flow

        async for item in run_generation_flow(self, request, check_llms=check_llms):
            yield item

    def save_results(self):
        with open(self.workspace / "intermediate_output.json", "w") as f:
            json.dump(
                {k: str(v) for k, v in self.intermediate_output.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _runtime_session_key(self) -> str:
        return str(self.session_id or self.workspace.stem)

    async def _ensure_env(self):
        """Create and open environment if not already open.

        When memory_system is available, routes through SessionManager to create
        InteractiveSession(AgentEnv) — ensuring proper session tracking, DB row
        creation, and cleanup/resume support.
        Otherwise, falls back to plain AgentEnv.
        """
        if self.agent_env is None:
            # Propagate memory DB location so MCP server references resolve cleanly.
            import os as _os
            mem_cfg = getattr(self.config, 'memory', None)
            if mem_cfg and getattr(mem_cfg, 'global_db_dir', ''):
                _os.environ["MEMSLIDES_MEMORY_GLOBAL_DB_DIR"] = str(
                    Path(mem_cfg.global_db_dir).expanduser()
                )
            if self.has_memory:
                # Route through SessionManager for proper session lifecycle
                session_mgr = self.memory_system.session_manager
                session_key = self._runtime_session_key()
                if session_mgr is not None:
                    session = await session_mgr.get_or_create_session(
                        user_id=self.user_id,
                        project_id=session_key,
                        workspace=self.workspace,
                        session_id=session_key,
                        config=self.config,
                        memory_system=self.memory_system,
                    )
                else:
                    # Fallback: direct creation if SessionManager unavailable
                    session = InteractiveSession(
                        session_id=session_key,
                        user_id=self.user_id,
                        project_id=session_key,
                        workspace=self.workspace,
                        db=self.memory_system.db,
                        config=self.config,
                        memory_system=self.memory_system,
                    )
                    # Ensure session row exists in DB (required by FK)
                    try:
                        from datetime import datetime as _dt
                        existing = await self.memory_system.db.query_one(
                            "SELECT id FROM sessions WHERE id = ?",
                            (session_key,),
                        )
                        if not existing:
                            await self.memory_system.db.insert("sessions", {
                                "id": session_key,
                                "user_id": self.user_id,
                                "project_id": session_key,
                                "status": "active",
                                "created_at": _dt.now().isoformat(),
                            })
                    except Exception as e:
                        logger.warning(f"Failed to insert session row (non-fatal): {e}")

                await session.__aenter__()
                self.agent_env = session  # IS-A AgentEnv
                self._session = session
                debug("Created InteractiveSession via SessionManager (AgentEnv + memory)")
            else:
                env = AgentEnv(self.workspace, self.config)
                await env.__aenter__()
                self.agent_env = env
                debug("Created plain AgentEnv (no memory)")
            self._env_owned = True

    async def resume(self, session_id: str | None = None) -> str | None:
        """Resume a previous session from disk without re-running generation.

        Restores:
        - intermediate_output from intermediate_output.json
        - DeckDesigner agent with chat_history from .history/DeckDesigner-00-history.jsonl
        - AgentEnv (MCP connections)
        - Original request from .input_request.json

        Args:
            session_id: Session ID to resume. If None, uses self.workspace.stem.

        Returns:
            Path to the final output file, or None if resume failed.
        """
        import jsonlines as _jl

        workspace = self.workspace
        if session_id:
            workspace = WORKSPACE_BASE / session_id
            self.workspace = workspace

        info(f"Attempting to resume session from {workspace}")

        # 1. Check workspace exists
        if not workspace.exists():
            warning(f"Workspace not found: {workspace}")
            return None

        # 2. Load intermediate_output.json
        output_file = workspace / "intermediate_output.json"
        if not output_file.exists():
            warning(f"intermediate_output.json not found in {workspace}")
            return None
        with open(output_file) as f:
            self.intermediate_output = json.load(f)
        info(f"Loaded intermediate_output: {list(self.intermediate_output.keys())}")

        # 3. Load original request
        request_file = workspace / ".input_request.json"
        if request_file.exists():
            with open(request_file) as f:
                self._last_request = InputRequest(**json.load(f))
            self._freeze_preference_writeback_current_job = (
                self._should_freeze_preference_writeback(self._last_request)
            )
            info(f"Loaded original request: {self._last_request.instruction[:80]}")

            resolved_payload: dict[str, Any] = {}
            resolved_file = workspace / "resolved_intent.json"
            if resolved_file.exists():
                try:
                    with open(resolved_file, encoding="utf-8") as f:
                        loaded_payload = json.load(f)
                    if isinstance(loaded_payload, dict):
                        resolved_payload = loaded_payload
                except Exception as e:
                    logger.debug("Failed to load resolved_intent.json during resume: %s", e)

            restored_intent = _normalize_memory_intent(resolved_payload) or _resolve_request_memory_intent(
                self._last_request
            )
            if restored_intent:
                self._resolved_request_intent = restored_intent
                self._resolved_request_intent_scenario = str(
                    resolved_payload.get("resolved_scenario_intent")
                    or resolved_payload.get("scenario")
                    or restored_intent
                )
                self._resolved_request_intent_source = str(
                    resolved_payload.get("source") or "resume"
                )
                confidence = resolved_payload.get("confidence")
                try:
                    self._resolved_request_intent_confidence = (
                        float(confidence) if confidence is not None else None
                    )
                except (TypeError, ValueError):
                    self._resolved_request_intent_confidence = None
                self._resolved_request_intent_raw_response = str(
                    resolved_payload.get("raw_response") or ""
                )
                if resolved_payload:
                    self._resolved_request_intent_payload = dict(resolved_payload)
                else:
                    self._resolved_request_intent_payload = {
                        "resolved_memory_intent": restored_intent,
                        "resolved_scenario_intent": self._resolved_request_intent_scenario,
                        "resolved_task_intent": restored_intent,
                        "resolved_memory_read_intent": restored_intent,
                        "resolved_memory_write_intent": restored_intent,
                        "source": "resume_input_request",
                        "confidence": None,
                        "raw_response": "",
                        "explicit_request_intent": _normalize_memory_intent(
                            self._last_request.memory_intent
                        ),
                    }
                info(
                    "Restored resolved memory intent during resume: %s (source=%s)",
                    restored_intent,
                    self._resolved_request_intent_source or "resume",
                )

        # 4. Ensure AgentEnv is ready — use resume_session() for snapshot recovery
        # Propagate memory DB location before MCP server connections.
        import os as _os
        _mem_cfg = getattr(self.config, 'memory', None)
        if _mem_cfg and getattr(_mem_cfg, 'global_db_dir', ''):
            _os.environ["MEMSLIDES_MEMORY_GLOBAL_DB_DIR"] = str(
                Path(_mem_cfg.global_db_dir).expanduser()
            )
        if (
            self.has_memory
            and self.memory_system.session_manager is not None
        ):
            session_mgr = self.memory_system.session_manager
            session_key = self._runtime_session_key()
            session = await session_mgr.resume_session(
                user_id=self.user_id,
                project_id=session_key,
                workspace=workspace,
                session_id=session_key,
                config=self.config,
                memory_system=self.memory_system,
            )
            await session.__aenter__()
            self.agent_env = session
            self._session = session
            self._env_owned = True
            # Log resume context if snapshot was found
            if hasattr(session, "resume_context") and session.resume_context:
                info(f"Session resumed with snapshot context: {session.resume_context[:200]}")
            debug("Resumed InteractiveSession via SessionManager")
        else:
            await self._ensure_env()
        agent_env = self.agent_env

        # 5. Recreate DeckDesigner agent and load chat_history
        hist_dir = workspace / ".history"
        design_history_file = hist_dir / "DeckDesigner-00-history.jsonl"
        if not design_history_file.exists():
            warning(f"DeckDesigner history not found: {design_history_file}")
            return None

        self.designagent = DeckDesigner(
            self.config,
            agent_env,
            self.workspace,
            self.language,
        )
        self.agent = self.designagent

        # Load saved chat_history
        loaded_history = []
        with _jl.open(design_history_file, mode="r") as reader:
            for msg_dict in reader:
                loaded_history.append(ChatMessage(**msg_dict))
        self.designagent.chat_history = loaded_history
        info(f"Restored DeckDesigner agent with {len(loaded_history)} messages in chat_history")

        # 5b. Restore RevisionEditor if its history exists
        modify_history_file = hist_dir / "RevisionEditor-00-history.jsonl"
        if modify_history_file.exists():
            try:
                self.modifyagent = RevisionEditor(
                    self.config,
                    agent_env,
                    self.workspace,
                    self.language,
                )
                modify_history = []
                with _jl.open(modify_history_file, mode="r") as reader:
                    for msg_dict in reader:
                        modify_history.append(ChatMessage(**msg_dict))
                self.modifyagent.chat_history = modify_history
                info(f"Restored RevisionEditor with {len(modify_history)} messages in chat_history")
            except Exception as e:
                logger.warning(f"Failed to restore RevisionEditor (non-fatal): {e}")
                self.modifyagent = None

        # 6. Count existing modify turns from durable artifacts. HTML workspaces
        # reopened by Web Studio may not have modification_N exports yet, so use
        # round artifacts and RevisionEditor history as additional signals.
        self._modify_turn_count = self._infer_resumed_modify_turn_count(workspace)
        info(f"Detected {self._modify_turn_count} previous modify turns (from workspace files)")

        # Load rollback checkpoints from disk (survives page refresh)
        if self._modify_turn_count > 0:
            mem = self.memory_system
            if mem is not None and hasattr(mem, "rollback_manager") and mem.rollback_manager:
                try:
                    loaded = mem.rollback_manager.load_from_disk(self.workspace)
                    if loaded > 0:
                        info(f"Loaded {loaded} rollback checkpoints from disk")
                except Exception as e:
                    logger.warning(f"Rollback checkpoint load failed (non-fatal): {e}")

        # 7. Return final output path (auto-convert if interrupted before PPTX)
        final = self.intermediate_output.get("final")
        if final and Path(final).exists():
            info(f"Session resumed successfully, final output: {final}")
            return str(final)

        # If slides exist but PPTX was never generated (e.g. interrupted session),
        # auto-convert now.
        slide_html_dir = self._resolve_exportable_slide_dir(
            self.intermediate_output.get("slide_html_dir")
        )
        if slide_html_dir:
            self.intermediate_output["slide_html_dir"] = str(slide_html_dir)
            html_files = self._resolve_exportable_slide_paths(slide_html_dir)
            if html_files:
                info(f"No final PPTX but {len(html_files)} slides found — auto-converting...")
                aspect = "16:9"
                if self._last_request:
                    aspect = self._last_request.powerpoint_type
                manuscript = self.intermediate_output.get("manuscript", "")
                stem = Path(manuscript).stem if manuscript else "presentation"
                pptx_path = workspace / f"{stem}.pptx"
                try:
                    slide_html_dir, html_files = await self._export_slides_with_agent_repair(
                        slide_html_dir,
                        pptx_path,
                        aspect_ratio=aspect,
                        context_label="session resume",
                    )
                    await self._export_pdf_best_effort(
                        html_files,
                        pptx_path.with_suffix(".pdf"),
                        aspect_ratio=aspect,
                        context_label="session resume",
                    )
                    self.intermediate_output["final"] = str(pptx_path)
                    self.save_results()
                    info(f"Auto-converted interrupted session, final output: {pptx_path}")
                    return str(pptx_path)
                except Exception as e:
                    warning(f"Auto-conversion failed: {e}")

        return None

    def _infer_resumed_modify_turn_count(self, workspace: Path) -> int:
        max_turn = 0
        for mod_file in workspace.glob("modification_*.*"):
            try:
                max_turn = max(max_turn, int(mod_file.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue

        rounds_dir = workspace / ".memory" / "rounds"
        if rounds_dir.exists():
            for round_dir in rounds_dir.glob("round_*"):
                suffix = round_dir.name.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    max_turn = max(max_turn, int(suffix))

        history_file = workspace / ".history" / "RevisionEditor-00-history.jsonl"
        if history_file.exists() and history_file.stat().st_size > 0:
            max_turn = max(max_turn, 1)

        return max_turn

    async def close_env(self):
        """Close environment and release resources.

        InteractiveSession: routes through SessionManager.end_session() if available,
        which removes session from tracking + calls session.end_session() (GC + MCP + DB).
        Plain AgentEnv: calls __aexit__() directly.
        """
        # MemoryOrchestrator: deferred on_job_end (was in run(), deferred so modify() can add tasks)
        if self._memory_orchestrator_instance:
            try:
                state = self._memory_active_job_state()
                if state["has_active_job"]:
                    _slide_dir = self._deferred_slide_html_dir or ""
                    # Update slide_html_dir if modify() produced newer output
                    _final = self.intermediate_output.get("slide_html_dir", _slide_dir)
                    freeze_preference_writeback = (
                        self._freeze_preference_writeback_current_job
                        or self._should_freeze_preference_writeback()
                    )
                    await self._memory_orchestrator_instance.on_job_end(
                        slide_html_dir=str(_final) if _final else _slide_dir,
                        freeze_preference_writeback=freeze_preference_writeback,
                    )
                    logger.info("MemoryOrchestrator: job ended at close_env(), WM consolidated to LTM")
                else:
                    logger.info("MemoryOrchestrator: no active job at close_env(); skipping consolidation")
            except Exception as e:
                logger.warning(f"Orchestrator on_job_end at close_env() failed: {e}")
            finally:
                self._freeze_preference_writeback_current_job = False

        # Stage 7: 保存注入日志到 workspace/.history 目录
        if self._guide_builder and self.workspace:
            try:
                history_dir = self.workspace / ".history"
                history_dir.mkdir(parents=True, exist_ok=True)
                saved_dir = self._guide_builder.save_injection_logs(history_dir)
                info(f"Saved injection logs to {saved_dir}")
            except Exception as e:
                logger.warning(f"Failed to save injection logs (non-fatal): {e}")

        # Stage 7: 清除 MCP Tool 模板上下文
        try:
            from memslides.tools.template_tools import clear_template_context
            clear_template_context()
        except Exception:
            pass

        # Stage 3 改造 3.6: Experience GC — 在 session 结束时清理过期/低价值经验
        if self.has_memory:
            try:
                _gc_writer = self._make_exp_writer()
                gc_result = await _gc_writer.run_gc()
                if gc_result.get("expired") or gc_result.get("capacity"):
                    logger.info("Experience GC: expired=%d, capacity=%d",
                                gc_result.get("expired", 0), gc_result.get("capacity", 0))
            except Exception as e:
                logger.debug("Experience GC failed (non-fatal): %s", e)

        if self.agent_env is not None and self._env_owned:
            try:
                if isinstance(self.agent_env, InteractiveSession):
                    # Prefer SessionManager for proper cleanup + tracking removal
                    session_mgr = (
                        self.memory_system.session_manager
                        if self.memory_system and hasattr(self.memory_system, "session_manager")
                        else None
                    )
                    if session_mgr is not None:
                        await session_mgr.end_session(self.agent_env)
                    else:
                        await self.agent_env.end_session()
                else:
                    await self.agent_env.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
            finally:
                self.agent_env = None
                self._session = None
                self._env_owned = False

    @timer("MemSlides RevisionEditor")
    async def modify(
        self,
        user_message: str,
        memory: Any = None,
        debug_tracer: Any = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in self._run_revision_flow(
            user_message,
            memory=memory,
            debug_tracer=debug_tracer,
        ):
            yield item

    async def _run_revision_flow(
        self,
        user_message: str,
        memory: Any = None,
        debug_tracer: Any = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """Compatibility wrapper; revision orchestration lives in pipelines.revision."""
        from memslides.pipelines.revision import run_revision_flow

        async for item in run_revision_flow(
            self,
            user_message,
            memory=memory,
            debug_tracer=debug_tracer,
        ):
            yield item

    async def _run_revision_impl(
        self,
        user_message: str,
        memory: Any = None,
        debug_tracer: Any = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in self.modify(user_message, memory=memory, debug_tracer=debug_tracer):
            yield item

    async def _emergency_memory_extraction(
        self,
        memory: Any,
        message: str,
        agent_response: str,
        tool_calls_log: list[dict],
        satisfaction: str = None,
    ):
        """急救通道：用户不满意时立即提取教训（Stage 5）

        Args:
            memory: MemorySystem 实例
            message: 用户消息
            agent_response: Agent 响应
            tool_calls_log: 工具调用日志
            satisfaction: 满意度信号 ("thumbs_down", "undo")
        """
        info(f"Emergency memory extraction triggered: satisfaction={satisfaction}")

        # 1. 从负面反馈直接提取 AtomicPreference（无需完整 Episode）
        # EpisodeExtractor 只支持 extract_and_store(batch, user_id, session_id)，
        # 需要 CompactRound 批量输入，不适用于单条反馈。
        # 改用 preference_extractor.extract_from_feedback() 直接提取偏好。
        if hasattr(memory, 'preference_extractor') and memory.preference_extractor:
            try:
                prefs = await memory.preference_extractor.extract_from_feedback(
                    user_message=message,
                    agent_response=agent_response or "",
                    user_id=self.user_id,
                    satisfaction=satisfaction or "negative",
                )
                if prefs:
                    info(f"Emergency: extracted {len(prefs)} preferences from feedback")
                    for pref in prefs:
                        info(f"  preference: '{getattr(pref, 'preference', str(pref))[:50]}'")
            except Exception as e:
                logger.warning(f"Emergency preference extraction failed: {e}")

        # 3. 失败经验统一交给 round experience / job-end consolidation；
        # 无 orchestrator 时才回退到旧的 direct-to-LTM 路径。
        if hasattr(memory, 'db') and memory.db and self._memory_orchestrator_instance is None:
            try:
                exp_writer = self._make_exp_writer()
                if exp_writer is None:
                    return

                traces = await exp_writer.from_modify_structured(
                    session_id=self.workspace.stem,
                    user_task=message,
                    tool_calls_log=tool_calls_log,
                    outcome="failure",  # 标记为失败
                    agent_response=agent_response,
                    turn=self._modify_turn_count,
                    template_id=self._current_template_id,
                )
                if traces:
                    info(f"Emergency: wrote {len(traces)} experience traces")
            except Exception as e:
                logger.warning(f"Emergency experience write failed: {e}")

    # ── Stage 8 Phase 2: Session Preference 提取与追加 ──

    _PREFERENCE_CATEGORIES = {"style", "color", "layout", "typography", "content", "general"}
    _PREFERENCE_RETENTION_SCOPES = {"job_local", "intent_profile", "default_profile"}
    _FUTURE_PREFERENCE_JUDGE_PROMPT = """你是一个 working memory 合规检查器，负责判断“新插入的单页”是否满足当前记住的 future structured preferences。

要求：
1. 只根据给定的 structured preferences 和这张新增页 HTML 判断。
2. 如果明显满足，输出 `pass`。
3. 如果明显不满足，输出 `fail`。
4. 如果信息不足、偏好太抽象、或无法仅凭这张 HTML 稳定判断，输出 `uncertain`，不要轻易判 fail。
5. 只返回 JSON，不要输出额外说明。

返回格式：
{{
  "judgements": [
    {{
      "rule_id": "规则ID",
      "verdict": "pass|fail|uncertain",
      "reason": "一句简短原因"
    }}
  ]
}}

Remembered future structured preferences:
{rule_specs_json}

Inserted slide: {slide_name}

Inserted slide HTML:
```html
{slide_html}
```"""

    _PREFERENCE_EXTRACTION_PROMPT = """分析用户本次的修改请求，提炼其中蕴含的抽象偏好，以及是否应该升级成后续轮次持续遵守的全局偏好/约束。

规则：
1. preference 只保留一句高层偏好或约束，不要复述整段任务。
2. dimension 尽量填最具体的规范维度：整体配色/主题主色才用 theme.primary_colors；字体/文字/标题/正文颜色用 typography.text_color，不要误写成背景、surface 或主题主色；字体族用 theme.font_family，版式优先 layout.slide_structure，内容表达优先 content.language_style，来源/页数/硬约束放 general；拿不准可留空。
3. 只有当用户表达了 deck-level 统一要求、后续轮次继续生效、或新增页也要继承的偏好/约束时，write_general 才设为 true。
4. 如果只是单页修改或一次性动作，write_general=false，general_sentence 留空，structured_rule=null。
5. retention_scope 默认保守使用 job_local；只有明确表达“以后默认/长期/总是/每次都这样”才可提升为 intent_profile；只有明确跨任务跨场景通用要求才可设为 default_profile。
6. general_sentence 只在 write_general=true 时填写，格式尽量像“全局偏好：……；默认影响后续轮次，必要时也约束新增页，除非用户显式覆盖。”
7. structured_rule 只在 write_general=true 且该偏好足够明确、适合后续机读校验时填写；更宽泛的风格偏好可以保留 structured_rule=null。
8. 若无明显偏好，返回：{{"preference": "", "category": "general", "dimension": "", "write_general": false, "general_sentence": "", "retention_scope": "job_local", "structured_rule": null}}

返回 JSON：
{{
  "preference": "偏好描述",
  "category": "style|color|layout|typography|content|general",
  "dimension": "theme.primary_colors|typography.text_color|theme.font_family|layout.slide_structure|content.language_style|general|",
  "write_general": true,
  "general_sentence": "全局偏好：……",
  "retention_scope": "job_local|intent_profile|default_profile",
  "structured_rule": {{
    "schema_version": "wm_rule_v1",
    "rule_type": "style|layout|content|constraint",
    "scope": "global",
    "target": {{"slide_scope": "all", "element_kind": "slide_title|body_text|image|deck"}},
    "action": {{"op": "apply_preference", "description": "……"}},
    "propagation": {{"apply_existing_slides": true, "apply_future_slides": true}}
  }}
}}

User Request: {user_message}"""

    @classmethod
    def _session_preference_has_general_signal(cls, user_message: str) -> bool:
        return _runtime_support.session_preference_has_general_signal(user_message)
        normalized = " ".join(str(user_message or "").split()).lower()
        if not normalized:
            return False
        patterns = (
            r"(所有|全部|整套|全局|统一).{0,12}(页|页面|标题|配色|颜色|字体|版式|风格|正文)",
            r"(以后|未来|后续|后面|新增页|新加页).{0,16}(继续|沿用|保持|都用|都要|都保持|继承|默认)",
            r"(每次|总是|一直|默认).{0,16}(都|用|保持|遵守)",
            r"(来源|页数|格式|约束).{0,16}(必须|只能|不要|禁止|严格)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _infer_session_preference_retention_scope(
        cls,
        user_message: str,
        raw_scope: str,
        *,
        write_general: bool,
    ) -> str:
        normalized_scope = str(raw_scope or "").strip().lower()
        if normalized_scope in cls._PREFERENCE_RETENTION_SCOPES:
            return normalized_scope
        if not write_general:
            return "job_local"

        normalized = " ".join(str(user_message or "").split()).lower()
        if re.search(r"(所有任务|任何场景|任何ppt|所有ppt|通用要求|跨场景|无论什么任务)", normalized):
            return "default_profile"
        if re.search(r"(以后|未来|后续).{0,8}(默认|都|一直|总是|优先|沿用)", normalized):
            return "intent_profile"
        if re.search(r"(长期|以后默认|默认情况下|每次都|一直都|总是这样)", normalized):
            return "intent_profile"
        return "job_local"

    @classmethod
    def _infer_session_preference_category(cls, dimension: str, fallback: str = "general") -> str:
        dim = str(dimension or "").strip().lower()
        if dim.startswith("layout"):
            return "layout"
        if dim.startswith("content"):
            return "content"
        if dim.startswith("theme.primary_colors") or dim in {"color", "typography.text_color", "theme.text_color"}:
            return "color"
        if dim.startswith("theme.font") or dim == "typography" or dim.startswith("typography."):
            return "typography"
        if dim.startswith("theme") or dim.startswith("visual"):
            return "style"
        if dim == "general":
            return "general"
        normalized_fallback = str(fallback or "").strip().lower()
        return normalized_fallback if normalized_fallback in cls._PREFERENCE_CATEGORIES else "general"

    @staticmethod
    def _slugify_session_rule(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower())
        return normalized.strip("_")[:40] or "rule"

    @staticmethod
    def _infer_session_rule_element_kind(*texts: str) -> str:
        haystack = " ".join(str(text or "") for text in texts).lower()
        if any(keyword in haystack for keyword in ("标题", "title", "h1", "heading")):
            return "slide_title"
        if any(keyword in haystack for keyword in ("正文", "bullet", "body", "段落", "要点")):
            return "body_text"
        if any(keyword in haystack for keyword in ("页脚", "footer")):
            return "footer"
        if any(keyword in haystack for keyword in ("图片", "image", "图像")):
            return "image"
        return "deck"

    @staticmethod
    def _infer_session_rule_type(dimension: str, *texts: str) -> str:
        dim = str(dimension or "").strip().lower()
        haystack = " ".join(str(text or "") for text in texts).lower()
        if dim.startswith("layout"):
            return "layout"
        if dim.startswith("content"):
            return "content"
        if dim == "general" and any(keyword in haystack for keyword in ("必须", "只能", "不要", "禁止", "strict", "only")):
            return "constraint"
        return "style"

    @classmethod
    def _build_session_rule_spec(
        cls,
        *,
        user_message: str,
        preference: str,
        dimension: str,
        general_sentence: str,
        retention_scope: str,
        turn: int,
        structured_rule: dict[str, Any] | None,
    ) -> dict[str, Any]:
        spec = deepcopy(structured_rule) if isinstance(structured_rule, dict) else {}
        element_kind = cls._infer_session_rule_element_kind(user_message, preference, general_sentence)
        normalized_dim = str(spec.get("dimension") or dimension or "general").strip()
        spec.setdefault("schema_version", "wm_rule_v1")
        spec.setdefault(
            "rule_id",
            f"wmr_session_turn_{turn:03d}_{cls._slugify_session_rule(preference or general_sentence)}",
        )
        spec["source"] = "session_preference"
        spec["source_turn"] = turn
        spec.setdefault("natural_language", str(user_message or "").strip()[:200] or preference)
        spec["normalized_sentence"] = general_sentence
        spec["dimension"] = normalized_dim
        spec.setdefault(
            "rule_type",
            cls._infer_session_rule_type(normalized_dim, user_message, preference, general_sentence),
        )
        spec["scope"] = "global"

        target = spec.get("target")
        if not isinstance(target, dict):
            target = {}
        target.setdefault("slide_scope", "all")
        target.setdefault("element_kind", element_kind)
        spec["target"] = target

        action = spec.get("action")
        if not isinstance(action, dict):
            action = {}
        action.setdefault("op", "apply_preference")
        action.setdefault("description", preference or general_sentence)
        if normalized_dim.lower() in {"typography.text_color", "theme.text_color"}:
            action["css_properties"] = ["color"]
            action["preserve_css_properties"] = [
                "background",
                "background-color",
                "background-image",
            ]
            action["preserve_semantic_roles"] = ["background", "surface"]
        spec["action"] = action

        propagation = spec.get("propagation")
        if not isinstance(propagation, dict):
            propagation = {}
        propagation.setdefault("apply_existing_slides", True)
        propagation.setdefault("apply_future_slides", True)
        spec["propagation"] = propagation

        verification = spec.get("verification")
        if not isinstance(verification, dict):
            verification = {}
        verification.setdefault("must_cover_all_targets", True)
        spec["verification"] = verification

        spec["retention_scope"] = retention_scope
        try:
            spec["confidence"] = float(spec.get("confidence", 0.8))
        except (TypeError, ValueError):
            spec["confidence"] = 0.8
        return spec

    @classmethod
    def _normalize_session_preference_payload(
        cls,
        raw_payload: Any,
        *,
        user_message: str,
        turn: int,
    ) -> dict[str, Any]:
        return _runtime_support.normalize_session_preference_payload(
            raw_payload,
            user_message=user_message,
            turn=turn,
        )
        preference = str(raw_payload.get("preference", "") or "").strip()
        category = str(raw_payload.get("category", "general") or "general").strip().lower()
        dimension = str(raw_payload.get("dimension", "") or "").strip()

        text_color_request = _runtime_support.request_mentions_text_color_without_background(user_message)
        if not dimension and category in {"color", "layout", "typography", "content", "general"}:
            dimension = category
        if text_color_request:
            if dimension.lower() in {"", "color", "theme.primary_colors"}:
                dimension = "typography.text_color"

        structured_rule = raw_payload.get("structured_rule")
        if not isinstance(structured_rule, dict):
            structured_rule = None
        elif text_color_request:
            structured_dim = str(structured_rule.get("dimension", "") or "").strip().lower()
            if structured_dim in {"", "color", "theme.primary_colors"}:
                structured_rule["dimension"] = "typography.text_color"

        propagation = structured_rule.get("propagation", {}) if structured_rule else {}
        write_general = _normalize_bool_flag(raw_payload.get("write_general"))
        if not write_general and isinstance(propagation, dict):
            write_general = (
                _normalize_bool_flag(propagation.get("apply_future_slides"))
                or (
                    str(structured_rule.get("scope", "") or "").strip().lower() == "global"
                    and _normalize_bool_flag(propagation.get("apply_existing_slides"))
                )
            )
        if not write_general:
            write_general = cls._session_preference_has_general_signal(user_message)
        if str(dimension or "").strip().lower() == "general":
            write_general = True

        general_sentence = str(raw_payload.get("general_sentence", "") or "").strip()
        if write_general and not general_sentence and preference:
            general_sentence = f"全局规则：{preference}"

        retention_scope = cls._infer_session_preference_retention_scope(
            user_message,
            str(raw_payload.get("retention_scope", "") or ""),
            write_general=write_general,
        )
        category = cls._infer_session_preference_category(dimension, fallback=category)

        if write_general and general_sentence:
            structured_rule = cls._build_session_rule_spec(
                user_message=user_message,
                preference=preference,
                dimension=dimension or "general",
                general_sentence=general_sentence,
                retention_scope=retention_scope,
                turn=turn,
                structured_rule=structured_rule,
            )

        return {
            "preference": preference,
            "category": category,
            "dimension": dimension,
            "write_general": write_general,
            "general_sentence": general_sentence,
            "retention_scope": retention_scope,
            "structured_rule": structured_rule,
        }

    async def _append_session_preference(
        self,
        user_message: str,
        agent_response: str,
        append_to_history: bool = True,
        intent: dict[str, Any] | None = None,
    ) -> bool:
        return await _runtime_support.append_session_preference(
            self,
            user_message,
            agent_response,
            append_to_history=append_to_history,
            intent=intent,
        )

    def _dump_current_round_wm_snapshot(self) -> bool:
        """将当前 Round 的 WM 快照重写到 .memory/rounds/round_xxx/wm_snapshot.json。"""
        live_dumped = _runtime_support.persist_live_wm_snapshot(self, reason="current_round")
        try:
            orchestrator = getattr(self, "_memory_orchestrator_instance", None)
            if orchestrator is None:
                return live_dumped
            job_mgr = getattr(orchestrator, "_job_mgr", None)
            if job_mgr is None:
                return live_dumped
            wm = getattr(job_mgr, "working_memory", None)
            if wm is None:
                return live_dumped

            dumper = getattr(self.memory_system, "artifact_dumper", None) if self.memory_system else None
            if dumper is None:
                return live_dumped

            round_index = job_mgr.round_count()
            if round_index <= 0:
                return live_dumped

            dumper.dump_wm_snapshot(
                round_index,
                chain_buffer=wm.chain_buffer,
                temp_preferences=wm._temp_preferences,
                temp_experiences=wm.get_experiences(),
                temp_episodes=wm.get_episodes(),
            )
            return True
        except Exception as e:
            logger.debug("Post-review wm_snapshot dump failed (non-fatal): %s", e)
            return False

    async def record_review_feedback_preference(self, feedback: str) -> bool:
        """在评审结束后立即写入本轮反馈偏好，并刷新当前 Round 的 WM 快照。"""
        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            return False

        wrote = await self._append_session_preference(
            feedback_text,
            "",
            append_to_history=False,
        )
        if not wrote:
            return False

        # 下一次 modify 入口跳过一次 priming，避免同条偏好重复写入。
        self._skip_next_session_preference_priming = True
        dumped = self._dump_current_round_wm_snapshot()
        info(
            "Post-review session preference recorded "
            f"(feedback_len={len(feedback_text)}, snapshot_dumped={dumped})."
        )
        return True

    async def _analyze_template(self, template_path: str) -> Any:
        """分析模板并提取技能。"""
        import shutil

        try:
            from memslides.memory.extract.template_analyzer import TemplateAnalyzer
            from memslides.memory.extract.template_extractors import ContentPatternExtractor, StyleExtractor
            from memslides.memory.core.template_models import TemplateProfile, TemplateSemanticModel
        except ImportError as e:
            logger.warning(f"Template analysis modules not available: {e}")
            return None

        try:
            from memslides.utils.constants import WORKSPACE_BASE

            templates_base = WORKSPACE_BASE / "templates"
            templates_base.mkdir(parents=True, exist_ok=True)

            template_id = str(uuid.uuid4())[:8]
            template_dir = templates_base / template_id
            template_dir.mkdir(parents=True, exist_ok=True)

            source_pptx = Path(template_path)
            original_name = source_pptx.stem
            dest_pptx = template_dir / "source.pptx"
            shutil.copy2(source_pptx, dest_pptx)
            logger.info(f"[TEMPLATE] Copied source.pptx to {template_dir} (original: {original_name})")

            template_llm_obj = None
            vision_llm_obj = None
            if self.memory_system:
                llm_objects = getattr(self.memory_system, "llm_objects_by_task", None)
                if llm_objects:
                    template_llm_obj = llm_objects.get("template_analyze") or llm_objects.get("template")
                    vision_llm_obj = llm_objects.get("vision") or template_llm_obj

            analyzer = TemplateAnalyzer(
                workspace=self.workspace,
                language_model=template_llm_obj,
                vision_model=vision_llm_obj,
            )
            analysis = await analyzer.analyze(str(dest_pptx), output_dir=str(template_dir))

            slide_induction = self._clean_template_induction(analysis.layout_induction or {})
            slide_induction_path = template_dir / "slide_induction.json"
            slide_induction_path.write_text(
                json.dumps(slide_induction, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"[TEMPLATE] Saved slide_induction.json: {len(slide_induction)} layouts")

            image_stats = analysis.image_stats or {}
            if image_stats:
                (template_dir / "image_stats.json").write_text(
                    json.dumps(image_stats, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            shape_geometry = analysis.shape_geometry or {}
            if shape_geometry:
                (template_dir / "shape_geometry.json").write_text(
                    json.dumps(shape_geometry, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            style_extractor = StyleExtractor()
            _template_llm = None
            if self.memory_system:
                llm_objects = getattr(self.memory_system, "llm_objects_by_task", None)
                if llm_objects:
                    _template_llm = llm_objects.get("template_analyze") or llm_objects.get("template")
                elif hasattr(self.memory_system, "llm"):
                    _template_llm = self.memory_system.llm
            content_extractor = ContentPatternExtractor(llm=_template_llm)

            design_constraints = await style_extractor.extract(
                analysis.slides_html,
                pptx_path=dest_pptx,
            )

            font_names = analysis.font_names or []
            if font_names and design_constraints:
                font_counts = Counter(font_names)
                common_fonts = font_counts.most_common(2)
                if common_fonts and not design_constraints.typography.title_font:
                    design_constraints.typography.title_font = common_fonts[0][0]
                    logger.info(f"[TEMPLATE] Set title_font from analysis: {common_fonts[0][0]}")
                if len(common_fonts) > 1 and not design_constraints.typography.body_font:
                    design_constraints.typography.body_font = common_fonts[1][0]
                    logger.info(f"[TEMPLATE] Set body_font from analysis: {common_fonts[1][0]}")
                elif common_fonts and not design_constraints.typography.body_font:
                    design_constraints.typography.body_font = common_fonts[0][0]

            content_patterns = await content_extractor.extract(
                analysis.slides_html,
                analysis.slides_text,
                analysis.layout_induction,
                pptx_filename=original_name,
            )

            layout_count = len([k for k in slide_induction.keys() if k not in ("functional_keys", "language", "layout_capabilities")])
            profile = TemplateProfile(
                name=original_name,
                description=f"从 {original_name} 提取的 {layout_count} 种布局模式",
                template_source=str(dest_pptx),
                slide_count=analysis.slide_count,
                aspect_ratio=analysis.aspect_ratio,
                slide_induction=slide_induction,
                semantic_model=TemplateSemanticModel.from_dict(analysis.semantic_model or {}),
                template_dir=str(template_dir),
                image_stats=image_stats,
                design_constraints=design_constraints,
                content_patterns=content_patterns,
                confidence=0.8,
                user_id=self.user_id,
            )

            description_path = template_dir / "description.txt"
            description_text = f"{profile.description}\nAspect ratio: {profile.aspect_ratio}, {profile.slide_count} slides."
            description_path.write_text(description_text, encoding="utf-8")
            logger.info("[TEMPLATE] Saved description.txt")

            if self.memory_system and hasattr(self.memory_system, "template_store"):
                store = self.memory_system.template_store
                if store:
                    try:
                        await store.add(profile)
                        info(f"Stored template profile: {profile.name} (id={profile.id})")
                    except Exception as e:
                        logger.warning(f"Failed to store template profile: {e}")

            return profile
        except Exception as e:
            logger.error(f"Template analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _clean_template_induction(self, slide_induction: dict[str, Any]) -> dict[str, Any]:
        def _clean_key(value: str) -> str:
            cleaned = re.sub(r"<think>.*?</think>", "", str(value), flags=re.DOTALL).strip().strip("\n")
            return cleaned or str(value).strip()

        cleaned: dict[str, Any] = {}
        for key, value in slide_induction.items():
            cleaned[_clean_key(key)] = value
        return cleaned

    # [DEPRECATED] _slide_induction_to_patterns 已废弃
    # LayoutPattern / LayoutRegion 类已移除，布局信息直接由 slide_induction 承载
    # def _slide_induction_to_patterns(self, slide_induction: dict, shape_geometry: dict = None) -> list:
    #     """从 slide_induction 生成 LayoutPattern 列表 [DEPRECATED]"""
    #     pass

    def _compute_position_label(self, left_pct: float, top_pct: float, width_pct: float, height_pct: float) -> str:
        """根据坐标百分比计算位置标签（如 top-left, center, bottom-right）"""
        # 水平位置
        center_x = left_pct + width_pct / 2
        if center_x < 33:
            h_pos = "left"
        elif center_x > 67:
            h_pos = "right"
        else:
            h_pos = "center"

        # 垂直位置
        center_y = top_pct + height_pct / 2
        if center_y < 33:
            v_pos = "top"
        elif center_y > 67:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # 组合
        if h_pos == "center" and v_pos == "middle":
            return "center"
        elif v_pos == "middle":
            return h_pos
        elif h_pos == "center":
            return v_pos
        else:
            return f"{v_pos}-{h_pos}"

    async def _extract_qa_preference(self, question: str, answer: str) -> Any:
        """从 QA 对话中提取用户偏好并存储

        Stage 5: ask_user_clarification 即时记忆提取
        """
        if not self.memory_system:
            return None

        try:
            extractor = getattr(self.memory_system, 'preference_extractor', None)
            if not extractor:
                return None

            # 检查是否有 extract_from_qa 方法
            if hasattr(extractor, 'extract_from_qa'):
                pref = await extractor.extract_from_qa(question, answer)
            else:
                # 降级：使用通用提取方法
                combined_text = f"用户问答记录：\n问: {question}\n答: {answer}"
                pref = await extractor.extract(combined_text, user_id=self.user_id)

            if pref:
                store = getattr(self.memory_system, 'preference_store', None)
                if store and hasattr(store, 'add'):
                    await store.add(pref)
                    info(f"Stored QA preference: {pref.preference[:50] if hasattr(pref, 'preference') else str(pref)[:50]}")

            return pref
        except Exception as e:
            logger.warning(f"QA preference extraction failed: {e}")
            return None

    async def _post_modify_memory(
        self,
        memory: Any,
        message: str,
        agent_response: str,
        tool_calls_log: list[dict],
        intent: Any,
        tracer: Any,
        turn: int,
        before_params: Any = None,  # G2 Phase 1
        after_params: Any = None,   # G2 Phase 1
        target_slide: str = None,   # G2 Phase 1
        agent_round: Any = None,    # Stage 4: AgentRound 统一输入
        force_extract: bool = False,  # Stage 5: 急救通道
        satisfaction: str = None,  # Stage 5: 满意度信号
    ):
        """Post-modification memory processing (runs in background).

        Stage 4 改造: 优先使用 AgentRound 统一输入，fallback 到旧方法。
        Stage 5 增强: force_extract=True 时同步提取并优先提取"避坑指南"。
        """
        # Stage 5: 急救通道 — 用户不满意时立即提取教训
        if force_extract and memory:
            try:
                await self._emergency_memory_extraction(
                    memory, message, agent_response, tool_calls_log, satisfaction
                )
            except Exception as e:
                logger.warning(f"Emergency memory extraction failed: {e}")
        try:
            # ── 1. Rule extraction — DEPRECATED (Stage 2.5) ──
            # IntelligentRuleExtractor → SimpleRuleStore 写入的规则从未被读取
            # ProceduralRule 改由 EpisodeExtractor 快速规则路径 + RuleInductor 归纳产生
            # 原代码见 git history

            # 2. Experience extraction
            # Stage 15: 有 orchestrator 时，不再中途 direct-to-LTM，由 on_round_end/job_end 统一归档。
            if hasattr(memory, "db") and memory.db and self._memory_orchestrator_instance is None:
                try:
                    exp_writer = self._make_exp_writer()
                    outcome = "success"
                    if any(t.get("is_error") for t in tool_calls_log):
                        outcome = "partial"

                    # Stage 4: 优先使用 from_agent_round() 统一输入
                    traces = []
                    primary_trace = None
                    _traces_count = 0
                    if agent_round is not None:
                        traces = await exp_writer.from_agent_round(agent_round, outcome=outcome)
                        primary_trace = traces[0] if traces else None
                        _traces_count = len(traces)

                    # Fallback: 使用旧方法 (向后兼容)
                    if not traces:
                        traces = await exp_writer.from_modify_structured(
                            session_id=self.workspace.stem,
                            user_task=message,
                            tool_calls_log=tool_calls_log,
                            outcome=outcome,
                            agent_response=agent_response,
                            debug_tracer=tracer,
                            turn=turn,
                            template_id=self._current_template_id,
                        )
                        primary_trace = traces[0] if traces else None
                        _traces_count = len(traces) if traces else 0
                    if tracer:
                        # Count total experiences in DB for evolution tracking
                        _total_exp = 0
                        try:
                            _rows = await memory.db.query(
                                "SELECT COUNT(*) as cnt FROM experience_traces", ()
                            )
                            _total_exp = _rows[0]["cnt"] if _rows else 0
                        except Exception:
                            pass
                        tracer.log_step(turn, "experience_written", {
                            "outcome": outcome,
                            "traces_count": _traces_count,
                            "task": primary_trace.task_description[:200] if primary_trace else "",
                            "tools_used": [t["name"] for t in tool_calls_log],
                            "tools_failed": [t["name"] for t in tool_calls_log if t.get("is_error")],
                            "lessons_learned": primary_trace.lessons_learned[:500] if primary_trace and primary_trace.lessons_learned else "(pending LLM summarization)",
                            "applicable_scenarios": primary_trace.applicable_scenarios if primary_trace else "[]",
                            "confidence": primary_trace.confidence if primary_trace else 0,
                            "trace_id": primary_trace.id[:12] if primary_trace else "",
                            "total_experiences_in_db": _total_exp,
                        })
                except Exception as e:
                    logger.warning(f"ExperienceTrace write failed (non-fatal): {e}")
            elif tracer:
                tracer.log_step(turn, "experience_queued_for_consolidation", {
                    "tools_used": [t["name"] for t in tool_calls_log],
                    "tools_failed": [t["name"] for t in tool_calls_log if t.get("is_error")],
                    "orchestrator_enabled": True,
                })

            # ── 4. FactualMemory recording — DEPRECATED (Stage 2.5) ──
            # ToolCallSegment 的 _extract_html_changes/_extract_html_diff 已包含相同的 before/after 参数变化
            # FactualRecord 写入的数据从未被任何读取路径消费

            # 8. Stage 2: Cognitive Episode Extraction
            # Stage 15: orchestrator 模式下由 on_round_end + job_end 统一管理 Episode，
            # 避免这里 direct-to-LTM 与 WM/consolidation 双写同一 round。
            if (
                hasattr(memory, 'collector') and memory.collector
                and self._memory_orchestrator_instance is None
            ):
                try:
                    _should_extract = memory.collector.should_extract()
                    _pending_rounds = memory.collector.pending_count
                    _pending_tokens = memory.collector.pending_tokens

                    # Artifact #4: Extraction trigger trace
                    if tracer:
                        if _should_extract:
                            if _pending_tokens >= memory.collector._token_budget:
                                _reason = "token_budget"
                            elif (time.time() - memory.collector._last_consolidation_time) > memory.collector.IDLE_CONSOLIDATION_INTERVAL:
                                _reason = "idle_timeout"
                            else:
                                _reason = "round_threshold"
                        else:
                            _reason = "conditions_not_met"
                        tracer.log_extraction_trigger(turn, _should_extract, {
                            "reason": _reason,
                            "pending_rounds": _pending_rounds,
                            "pending_tokens": _pending_tokens,
                            "batch_size": _pending_rounds if _should_extract else 0,
                        })

                    if _should_extract:
                        batch = memory.collector.pop_batch()
                        if batch and hasattr(memory, 'episode_extractor') and memory.episode_extractor:
                            episodes = await memory.episode_extractor.extract_and_store(
                                batch, user_id=self.user_id, session_id=self.workspace.stem,
                            )
                            if tracer and episodes:
                                # Count total episodes in DB
                                _total_eps = 0
                                try:
                                    if hasattr(memory, 'episode_store') and memory.episode_store:
                                        _all_eps = await memory.episode_store.list_episodes(
                                            user_id=self.user_id, limit=999
                                        )
                                        _total_eps = len(_all_eps) if _all_eps else 0
                                except Exception:
                                    pass
                                tracer.log_step(turn, "episodes_extracted", {
                                    "count": len(episodes),
                                    "episodes": [
                                        {
                                            "id": ep.id[:12],
                                            "design_insight": ep.design_insight[:200] if ep.design_insight else "",
                                            "category": ep.category,
                                            "source_round_id": ep.source_round_id,
                                            "user_intent": ep.user_intent[:100] if ep.user_intent else "",
                                            "confidence": ep.confidence,
                                        }
                                        for ep in episodes
                                    ],
                                    "source_batch_rounds": [
                                        getattr(r, "round_id", None) for r in batch
                                    ] if batch else [],
                                    "total_episodes_in_db": _total_eps,
                                })
                            info(f"Extracted {len(episodes)} cognitive episodes")

                            # Episode snapshot after extraction
                            if tracer and hasattr(memory, 'episode_store') and memory.episode_store:
                                try:
                                    await tracer.dump_episodes_snapshot(memory.episode_store)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Cognitive episode extraction failed (non-fatal): {e}")
            elif tracer and hasattr(memory, 'collector') and memory.collector:
                tracer.log_extraction_trigger(turn, False, {
                    "reason": "orchestrator_managed",
                    "pending_rounds": memory.collector.pending_count,
                    "pending_tokens": memory.collector.pending_tokens,
                    "batch_size": 0,
                })

            # 9. Memory evolution cumulative view (Layer 3 debug artifact)
            if tracer and hasattr(memory, "db") and memory.db:
                try:
                    from datetime import datetime
                    # Query all memory component counts
                    _exp_rows = await memory.db.query("SELECT COUNT(*) as cnt, SUM(CASE WHEN final_outcome='failed' THEN 1 ELSE 0 END) as failed FROM experience_traces", ())
                    _exp_total = _exp_rows[0]["cnt"] if _exp_rows else 0
                    _exp_failed = _exp_rows[0]["failed"] if _exp_rows else 0

                    _eps_rows = await memory.db.query("SELECT COUNT(*) as cnt FROM design_episodes", ())
                    _eps_total = _eps_rows[0]["cnt"] if _eps_rows else 0

                    _rules_rows = await memory.db.query("SELECT status, COUNT(*) as cnt FROM procedural_rules GROUP BY status", ())
                    _rules_stats = {r["status"]: r["cnt"] for r in _rules_rows} if _rules_rows else {}

                    # Check if lessons were summarized (at least one experience has non-empty lessons_learned)
                    _lessons_rows = await memory.db.query(
                        "SELECT COUNT(*) as cnt FROM experience_traces WHERE lessons_learned IS NOT NULL AND lessons_learned != ''",
                        ()
                    )
                    _lessons_count = _lessons_rows[0]["cnt"] if _lessons_rows else 0

                    tracer.append_memory_evolution({
                        "turn": turn,
                        "timestamp": datetime.now().isoformat()[:19],
                        "user_message": message[:150] if message else "",
                        "memory_snapshot": {
                            "experiences": {
                                "total": _exp_total,
                                "failed": _exp_failed,
                                "success": _exp_total - _exp_failed,
                                "with_lessons": _lessons_count,
                            },
                            "episodes": {"total": _eps_total},
                            "rules": {
                                "confirmed": _rules_stats.get("confirmed", 0),
                                "pending": _rules_stats.get("pending", 0),
                                "deprecated": _rules_stats.get("deprecated", 0),
                            },
                        },
                    })
                except Exception as _e:
                    logger.warning(f"Memory evolution update failed (non-fatal): {_e}")

        except Exception as e:
            logger.warning(f"Post-modify memory processing failed: {e}")

    async def rollback(self, debug_tracer: Any = None) -> bool:
        """Rollback last modification.

        Restores:
        1. HTML slide content from RollbackManager checkpoints
        2. RevisionEditor chat_history trimmed to pre-modification length
        3. Output files (modification_N.pptx/.pdf) deleted
        4. intermediate_output["final"] reverted to previous value
        5. Preview PDF + images regenerated from restored HTML
        6. _modify_turn_count decremented

        Returns True if rollback succeeded, False otherwise.
        """
        mem = self.memory_system
        turn = self._modify_turn_count

        if turn <= 0:
            logger.warning("Nothing to rollback (turn count is 0)")
            return False

        # ── 1. Restore HTML from checkpoints ──
        slide_html_dir_str = self.intermediate_output.get("slide_html_dir")
        if not slide_html_dir_str:
            logger.warning("slide_html_dir not found in intermediate_output")
            return False
        slide_html_dir = Path(slide_html_dir_str)

        rolled_back = []
        if mem is not None and hasattr(mem, "rollback_manager") and mem.rollback_manager:
            rm = mem.rollback_manager
            checkpointed = [
                sid for sid in list(rm._html_checkpoints)
                if rm.has_checkpoint(sid)
            ]
            if not checkpointed:
                logger.warning(f"No rollback checkpoints available for turn {turn}")
                if debug_tracer:
                    debug_tracer.log_event("rollback", {
                        "turn": turn, "success": False, "reason": "no_checkpoints",
                    })
                return False

            for slide_id in checkpointed:
                slide_path = slide_html_dir / f"{slide_id}.html"
                ok = await rm.rollback(slide_id, slide_path)
                if ok:
                    rolled_back.append(slide_id)

            if not rolled_back:
                logger.warning(f"Rollback failed for turn {turn}: no slides restored")
                if debug_tracer:
                    debug_tracer.log_event("rollback", {"turn": turn, "success": False})
                return False
        else:
            logger.warning("RollbackManager not available")
            return False

        # ── 2. Trim RevisionEditor chat_history ──
        try:
            agent = self.modifyagent or self.designagent
            checkpoint_len = getattr(self, "_history_checkpoint", None)
            if agent and checkpoint_len is not None and checkpoint_len < len(agent.chat_history):
                trimmed = len(agent.chat_history) - checkpoint_len
                agent.chat_history = agent.chat_history[:checkpoint_len]
                info(f"Rollback: trimmed {trimmed} messages from agent chat_history")
                agent.save_history()
        except Exception as e:
            logger.warning(f"Rollback: chat_history trim failed (non-fatal): {e}")

        # ── 3. Delete modification_N output files ──
        try:
            for ext in (".pptx", ".pdf"):
                out_file = self.workspace / f"modification_{turn}{ext}"
                if out_file.exists():
                    out_file.unlink()
                    info(f"Rollback: deleted {out_file.name}")
            # Also remove preview image directory
            preview_dir = self.workspace / f".slide_images-pdf-modification_{turn}"
            if preview_dir.exists():
                import shutil
                shutil.rmtree(preview_dir, ignore_errors=True)
                info(f"Rollback: deleted preview dir {preview_dir.name}")
        except Exception as e:
            logger.warning(f"Rollback: output file cleanup failed (non-fatal): {e}")

        # ── 4. Revert intermediate_output["final"] ──
        previous_final = getattr(self, "_previous_final", None)
        if previous_final:
            self.intermediate_output["final"] = previous_final
        else:
            # Fallback: find the most recent remaining modification or original
            prev_turn = turn - 1
            if prev_turn > 0:
                for ext in (".pptx", ".pdf"):
                    candidate = self.workspace / f"modification_{prev_turn}{ext}"
                    if candidate.exists():
                        self.intermediate_output["final"] = str(candidate)
                        break
            else:
                # Revert to original pptx/pdf
                for f in sorted(self.workspace.glob("*.pptx")):
                    if not f.name.startswith("modification_"):
                        self.intermediate_output["final"] = str(f)
                        break
        self.save_results()

        # ── 5. Re-render preview (PDF + slide images) ──
        try:
            request = self._last_request
            aspect_ratio = request.powerpoint_type if request else "16:9"
            final_path = Path(self.intermediate_output.get("final", ""))
            pdf_path = final_path.with_suffix(".pdf") if final_path.suffix == ".pptx" else final_path
            if not pdf_path.suffix:
                pdf_path = self.workspace / "rollback_preview.pdf"
            export_html_files = self._resolve_exportable_slide_paths(slide_html_dir)
            if not export_html_files:
                raise RuntimeError(
                    f"No exportable slides found in {slide_html_dir}; expected slide_XX.html files."
                )
            exported = await self._export_pdf_best_effort(
                export_html_files,
                pdf_path,
                aspect_ratio=aspect_ratio,
                context_label="rollback preview",
            )
            if exported:
                info(f"Rollback: regenerated preview at {pdf_path}")
        except Exception as e:
            logger.warning(f"Rollback: preview regeneration failed (non-fatal): {e}")

        # ── 6. Decrement turn count ──
        self._modify_turn_count = max(0, self._modify_turn_count - 1)
        info(f"Rolled back modification turn {turn} → now at turn {self._modify_turn_count}: {rolled_back}")

        # ── 7. Notify collector to remove rolled-back round from pending batch ──
        if mem is not None and hasattr(mem, "collector") and mem.collector:
            try:
                await mem.collector.on_rollback({"turn": turn})
            except Exception as e:
                logger.warning(f"Rollback: collector notification failed (non-fatal): {e}")

        if debug_tracer:
            debug_tracer.log_event("rollback", {
                "turn": turn, "success": True, "slides": rolled_back,
            })
        return True

    async def get_slide_previews(self) -> list[str]:
        """Get preview image paths for all slides.

        Returns:
            List of paths to slide preview images (sorted by slide number).
        """
        slide_html_dir = self.intermediate_output.get("slide_html_dir")
        if not slide_html_dir:
            return []

        # Look for preview images generated by PlaywrightConverter
        # They are stored in .slide_images-pdf-<stem>/ directory
        # Return the NEWEST preview directory (by modification time)
        preview_dirs = list(self.workspace.glob(".slide_images-pdf-*"))
        if not preview_dirs:
            return []

        # Sort by modification time (newest first)
        preview_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        # Return images from the newest preview directory
        for preview_dir in preview_dirs:
            images = sorted(preview_dir.glob("slide_*.jpg"))
            if images:
                return [str(img) for img in images]
        return []


__all__ = ["AgentLoop", "IntentResolutionResult", "ModifyExecutionPlan"]
