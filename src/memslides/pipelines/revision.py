from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import traceback
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from memslides.contracts import DeckResult, RevisionRequest
from memslides.runtime import support as runtime_support
from memslides.tools.deck_runtime import set_current_agent, set_current_modify_context
from memslides.pipelines.generation_support import normalize_memory_intent
from memslides.utils.constants import FORCE_FINALIZE_MSG, MAX_MODIFY_ITERATIONS
from memslides.utils.log import debug, error, info, warning
from memslides.utils.typings import ChatMessage, Role

logger = logging.getLogger(__name__)

append_memory_flow_trace = runtime_support.append_memory_flow_trace
append_modify_tool_policy_trace = runtime_support.append_modify_tool_policy_trace
apply_modify_tool_policy = runtime_support.apply_modify_tool_policy
align_modify_execution_plan_to_tool_policy = runtime_support.align_modify_execution_plan_to_tool_policy
build_inserted_slide_completion_followup = runtime_support.build_inserted_slide_completion_followup
build_new_element_rule_applications = runtime_support.build_new_element_rule_applications
build_new_element_preference_followup = runtime_support.build_new_element_preference_followup
build_modify_tool_policy_plan = runtime_support.build_modify_tool_policy_plan
build_controlled_rewrite_recovery_followup = runtime_support.build_controlled_rewrite_recovery_followup
build_controlled_rewrite_recovery_policy = runtime_support.build_controlled_rewrite_recovery_policy
collect_soft_preference_evaluations = runtime_support.collect_soft_preference_evaluations
collect_new_element_preference_failures_async = runtime_support.collect_new_element_preference_failures_async
collect_pending_inserted_slides = runtime_support.collect_pending_inserted_slides
get_session_preference_fallback_prompt = runtime_support.get_session_preference_fallback_prompt
merge_wm_rule_specs_text = runtime_support.merge_wm_rule_specs_text
PREFERENCE_UPDATE_MUTATION_TOOLS = runtime_support.PREFERENCE_UPDATE_MUTATION_TOOLS
is_preference_update_plan = runtime_support.is_preference_update_plan
parse_modify_recovery_request_from_tool_result = runtime_support.parse_modify_recovery_request_from_tool_result
plan_has_future_only_rules = runtime_support.plan_has_future_only_rules
render_current_tool_authority = runtime_support.render_current_tool_authority
restore_agent_base_tools = runtime_support.restore_agent_base_tools
validate_modify_tool_capability_contract = runtime_support.validate_modify_tool_capability_contract
enforce_hard_title_prefix_preferences = getattr(
    runtime_support,
    "enforce_hard_title_prefix_preferences",
    lambda *args, **kwargs: [],
)


def _file_sha256(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _operation_has_fresh_strict_export(workspace: Path) -> bool:
    try:
        from memslides.web.session_store import (
            infer_export_status,
            operation_has_fresh_export_since_start,
            read_operation_state,
        )

        operation = read_operation_state(workspace)
        export_status, _warnings = infer_export_status(workspace, operation=operation)
        return bool(export_status == "strict" and operation_has_fresh_export_since_start(workspace, operation))
    except Exception:
        return False


def _set_current_modify_context_compat(**kwargs: Any) -> None:
    """Call set_current_modify_context while tolerating older runtime signatures."""
    try:
        supported = {
            key: value
            for key, value in kwargs.items()
            if key in inspect.signature(set_current_modify_context).parameters
        }
    except Exception:
        supported = dict(kwargs)
    try:
        set_current_modify_context(**supported)
    except TypeError:
        minimal = {
            key: kwargs[key]
            for key in (
                "workspace",
                "target_slide_paths",
                "operation_kind",
                "coverage_required",
                "raw_user_message",
                "user_preference_rule_specs",
                "user_preference_colors",
            )
            if key in kwargs
        }
        set_current_modify_context(**minimal)


READ_ONLY_INFORMATION_TOOLS = {
    "list_files",
    "read_file",
    "read_slide_snapshot",
    "scan_slide_index",
    "inspect_slide",
    "thinking",
    "remember_lesson",
    "plan_slide_patch",
    "list_document_figures",
    "explore_workspace_images",
    "image_caption",
    "search_web",
    "fetch_url",
    "search_images",
    "download_file",
    "image_generation",
    "document_summary",
    "search_experiences",
    "search_episodes",
}

MUTATION_TOOLS = {
    "apply_slide_patch",
    "write_html_file",
    "write_new_slide_file",
    "insert_slide",
    "delete_slide",
    "batch_update_css_rule",
    "batch_update_semantic_style",
    "patch_semantic_inline_style",
    "render_chart_asset",
    "render_table_asset",
    "render_flowchart_asset",
}

ALWAYS_PROGRESS_READ_ONLY_TOOLS = {
    "plan_slide_patch",
    "list_document_figures",
    "explore_workspace_images",
    "image_caption",
    "search_web",
    "fetch_url",
    "search_images",
    "download_file",
    "image_generation",
    "document_summary",
    "search_experiences",
    "search_episodes",
    "scan_slide_index",
    "list_files",
    "thinking",
    "remember_lesson",
}


def _safe_json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw or "{}"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_read_only_progress_path(
    raw_path: Any,
    *,
    workspace: Path,
    resolve_modify_path: Any = None,
) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    if text.lower() in {"all", "*"}:
        return None
    if callable(resolve_modify_path):
        try:
            resolved = resolve_modify_path(text)
            if resolved is not None:
                return Path(resolved)
        except Exception:
            pass
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        return candidate.resolve()
    except Exception:
        return candidate


def classify_read_only_call_progress(
    call: Any,
    result_msg: Any,
    *,
    workspace: Path,
    seen_keys: set[str],
    resolve_modify_path: Any = None,
) -> dict[str, Any]:
    tool_name = str(getattr(call.function, "name", "") or "").strip()
    args = _safe_json_args(getattr(call.function, "arguments", "{}"))
    result_text = str(getattr(result_msg, "text", "") or "")
    if tool_name not in READ_ONLY_INFORMATION_TOOLS:
        return {
            "tool": tool_name,
            "status": "ignored",
            "new_information": False,
            "resource_key": "",
        }
    if getattr(result_msg, "is_error", False):
        key = f"{tool_name}:error:{hashlib.sha256(result_text[:500].encode('utf-8', errors='ignore')).hexdigest()}"
        duplicate = key in seen_keys
        seen_keys.add(key)
        return {
            "tool": tool_name,
            "status": "duplicate" if duplicate else "progress",
            "new_information": not duplicate,
            "resource_key": key,
            "duplicate_reason": "duplicate_error_result" if duplicate else "",
            "new_information_reason": "new_error_information" if not duplicate else "",
        }
    if tool_name == "read_slide_snapshot":
        slide_path = _resolve_read_only_progress_path(
            args.get("slide_path"),
            workspace=workspace,
            resolve_modify_path=resolve_modify_path,
        )
        content_hash = ""
        try:
            payload = json.loads(result_text)
            if isinstance(payload, dict):
                content_hash = str(payload.get("content_hash", "") or "")
                if slide_path is None:
                    slide_path = _resolve_read_only_progress_path(
                        payload.get("slide_path"),
                        workspace=workspace,
                        resolve_modify_path=resolve_modify_path,
                    )
        except Exception:
            content_hash = ""
        if slide_path is not None and not content_hash:
            content_hash = _file_sha256(slide_path)
        key = f"read_slide_snapshot:{slide_path or args.get('slide_path', '')}:{content_hash}:full"
        duplicate = key in seen_keys
        seen_keys.add(key)
        return {
            "tool": tool_name,
            "status": "duplicate" if duplicate else "progress",
            "new_information": not duplicate,
            "resource_key": key,
            "content_hash": content_hash,
            "scope_signature": "full",
            "duplicate_reason": "duplicate_snapshot_same_hash" if duplicate else "",
            "new_information_reason": "new_snapshot_hash_or_slide" if not duplicate else "",
        }
    if tool_name == "read_file":
        file_path = _resolve_read_only_progress_path(
            args.get("file_path"),
            workspace=workspace,
            resolve_modify_path=resolve_modify_path,
        )
        file_hash = _file_sha256(file_path) if file_path is not None else ""
        offset = str(args.get("offset", 0) or 0)
        limit = str(args.get("limit", 500) or 500)
        key = f"read_file:{file_path or args.get('file_path', '')}:{file_hash}:{offset}:{limit}"
        duplicate = key in seen_keys
        seen_keys.add(key)
        return {
            "tool": tool_name,
            "status": "duplicate" if duplicate else "progress",
            "new_information": not duplicate,
            "resource_key": key,
            "content_hash": file_hash,
            "scope_signature": f"offset={offset};limit={limit}",
            "duplicate_reason": "duplicate_file_window" if duplicate else "",
            "new_information_reason": "new_file_window_or_hash" if not duplicate else "",
        }
    if tool_name == "inspect_slide":
        html_path = _resolve_read_only_progress_path(
            args.get("html_file"),
            workspace=workspace,
            resolve_modify_path=resolve_modify_path,
        )
        content_hash = _file_sha256(html_path) if html_path is not None else ""
        unchanged = "UNCHANGED" in result_text
        key = f"inspect_slide:{html_path or args.get('html_file', '')}:{content_hash}:{'unchanged' if unchanged else 'changed_or_first'}"
        duplicate = unchanged and key in seen_keys
        seen_keys.add(key)
        return {
            "tool": tool_name,
            "status": "duplicate" if duplicate else "progress",
            "new_information": not duplicate,
            "resource_key": key,
            "content_hash": content_hash,
            "scope_signature": "inspect",
            "duplicate_reason": "repeated_inspect_without_change" if duplicate else "",
            "new_information_reason": "inspect_verdict_or_changed_hash" if not duplicate else "",
        }
    args_hash = hashlib.sha256(
        json.dumps(args, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
    ).hexdigest()
    key = f"{tool_name}:{args_hash}"
    if tool_name in ALWAYS_PROGRESS_READ_ONLY_TOOLS:
        seen_keys.add(key)
        return {
            "tool": tool_name,
            "status": "progress",
            "new_information": True,
            "resource_key": key,
            "new_information_reason": "information_or_asset_discovery_tool",
        }
    duplicate = key in seen_keys
    seen_keys.add(key)
    return {
        "tool": tool_name,
        "status": "duplicate" if duplicate else "progress",
        "new_information": not duplicate,
        "resource_key": key,
        "duplicate_reason": "duplicate_read_only_call" if duplicate else "",
        "new_information_reason": "new_read_only_call" if not duplicate else "",
    }


def _parse_finalize_pending_inspect_slides(message: str) -> list[dict[str, str]]:
    text = " ".join(str(message or "").split())
    if "Pending slides:" not in text or "inspect_slide" not in text:
        return []
    match = re.search(r"Pending slides:\s*(.*?)(?:\.\s*Re-run|\.\s*Fix|$)", text)
    if not match:
        return []
    items: list[dict[str, str]] = []
    for raw_item in match.group(1).split(";"):
        item = raw_item.strip()
        if not item or item.startswith("and "):
            continue
        item_match = re.match(r"(?P<slide>[^()]+?\.html)\s*(?:\((?P<reason>.*?)\))?$", item)
        if not item_match:
            continue
        reason = " ".join(str(item_match.group("reason") or "").split())
        lowered = reason.lower()
        if "never passed" in lowered:
            next_action = "inspect_slide"
        elif "last inspect failed" in lowered or "visual qa failed" in lowered or "diagram qa failed" in lowered:
            next_action = "repair_then_inspect"
        else:
            next_action = "inspect_or_repair_then_inspect"
        items.append(
            {
                "slide_name": item_match.group("slide").strip(),
                "reason": reason,
                "next_action": next_action,
            }
        )
    return items[:12]


def _build_finalize_pending_inspect_followup(pending_slides: list[dict[str, str]]) -> str:
    if not pending_slides:
        return ""
    lines = [
        "SYSTEM: `finalize` is blocked because some modified slide HTML files have not passed current `inspect_slide` validation.",
        "Do not call `finalize` again yet. Work through these slides first:",
    ]
    for item in pending_slides:
        slide = item.get("slide_name", "target slide")
        reason = item.get("reason", "") or "current HTML has not passed inspect_slide"
        action = item.get("next_action", "inspect_or_repair_then_inspect")
        if action == "inspect_slide":
            instruction = f"run `inspect_slide` on `{slide}`"
        else:
            instruction = f"repair `{slide}` if needed, then run `inspect_slide`"
        lines.append(f"- {slide}: {reason}; next action: {instruction}.")
    lines.append("Only call `finalize` after the current on-disk HTML for every listed slide passes `inspect_slide`.")
    return "\n".join(lines)


def classify_read_only_batch_progress(
    new_results: list[tuple[Any, Any]],
    *,
    workspace: Path,
    seen_keys: set[str],
    resolve_modify_path: Any = None,
) -> dict[str, Any]:
    entries = [
        classify_read_only_call_progress(
            call,
            result_msg,
            workspace=workspace,
            seen_keys=seen_keys,
            resolve_modify_path=resolve_modify_path,
        )
        for call, result_msg in new_results
        if str(getattr(call.function, "name", "") or "").strip() in READ_ONLY_INFORMATION_TOOLS
    ]
    entries = [entry for entry in entries if entry.get("status") != "ignored"]
    if not entries:
        return {"has_read_only": False, "has_progress": False, "entries": []}
    progress_entries = [entry for entry in entries if entry.get("new_information")]
    duplicate_entries = [entry for entry in entries if not entry.get("new_information")]
    status = "progress" if progress_entries else "duplicate"
    reason = ""
    if progress_entries:
        reason = str(progress_entries[-1].get("new_information_reason", "") or "")
    elif duplicate_entries:
        reason = str(duplicate_entries[-1].get("duplicate_reason", "") or "")
    return {
        "has_read_only": True,
        "has_progress": bool(progress_entries),
        "status": status,
        "reason": reason,
        "entries": entries,
        "last_entry": entries[-1],
    }


async def run_revision_flow(
    runtime: Any,
    user_message: str,
    memory: Any = None,
    debug_tracer: Any = None,
    memory_intent: str = "",
    request_extra_info: dict[str, Any] | None = None,
) -> AsyncGenerator[str | ChatMessage, None]:
    """Multi-round modification entry point — uses dedicated RevisionEditor.

    Args:
        user_message: User's modification request text.
        memory: MemorySystem instance (optional).
                Falls back to self.memory_system if available.
        debug_tracer: DebugTracer instance (optional, for observability).

    Yields:
        ChatMessage or str: Messages or updated output path.
    """
    self = runtime
    request_memory_intent = normalize_memory_intent(memory_intent)
    if not request_memory_intent and isinstance(request_extra_info, dict):
        request_memory_intent = normalize_memory_intent(request_extra_info)
    if request_memory_intent:
        self._resolved_request_intent = request_memory_intent
        if not getattr(self, "_resolved_request_intent_scenario", ""):
            self._resolved_request_intent_scenario = request_memory_intent
        self._resolved_request_intent_source = "revision_request"
        self._resolved_request_intent_confidence = 1.0
        self._resolved_request_intent_raw_response = ""
        self._resolved_request_intent_payload = {
            "resolved_memory_intent": request_memory_intent,
            "resolved_scenario_intent": self._resolved_request_intent_scenario,
            "resolved_task_intent": request_memory_intent,
            "resolved_memory_read_intent": request_memory_intent,
            "resolved_memory_write_intent": request_memory_intent,
            "core_persona": "",
            "source": "revision_request",
            "confidence": 1.0,
            "raw_response": "",
            "explicit_request_intent": request_memory_intent,
        }
    if self.designagent is None:
        yield ChatMessage(
            role=Role.SYSTEM,
            content="No DeckDesigner agent available. Please run initial generation first.",
        )
        return

    if self.agent_env is None:
        yield ChatMessage(
            role=Role.SYSTEM,
            content="Agent environment not available. Please run initial generation first.",
        )
        return
    
    try:
        agent_to_use = self._ensure_modifyagent_loaded(
            load_reason="multi-turn modifications"
        )
    except RuntimeError as e:
        yield ChatMessage(
            role=Role.SYSTEM,
            content=str(e),
        )
        return

    if agent_to_use is None:
        yield ChatMessage(
            role=Role.SYSTEM,
            content="RevisionEditor is unavailable because the agent environment is not initialized.",
        )
        return

    set_current_agent(
        "RevisionEditor",
        workspace=self.workspace,
        model_ref=getattr(agent_to_use, "model_ref", "modify_agent"),
    )  # For finalize() behavior

    # Stage 7: 确保模板 MCP Tool 上下文有效（modify 入口恢复）
    if self._guide_builder:
        try:
            from memslides.tools.template_tools import (
                set_template_context, get_template_context
            )
            if get_template_context() is None:
                set_template_context(self._guide_builder)
                info("Restored template MCP tool context for modify()")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Template MCP context restore failed (non-fatal): {e}")

    # Use self.memory_system if available, fall back to memory param
    mem = self.memory_system or memory
    if self.memory_system is None and mem is not None:
        self.memory_system = mem

    self._modify_turn_count += 1
    turn = self._modify_turn_count
    info(f"Modification round {turn}: {user_message[:100]}")

    # Treat each modify turn as a fresh round to avoid previous-round
    # reasoning/tool traces from biasing the current request.
    if hasattr(agent_to_use, "reset_for_new_round"):
        try:
            agent_to_use.reset_for_new_round()
        except Exception as e:
            logger.warning(f"Failed to reset agent chat history for new modify round: {e}")

    # Stage 4: Track memory injection for later use in begin_round()
    _memory_injection_text = ""
    _memory_injection_full = ""
    wm_rule_specs_text = ""

    # MemoryOrchestrator: Round lifecycle + SYSTEM-level injection for RevisionEditor Agent.
    # A manual Web memory save ends the current WM job while keeping the runtime alive;
    # restart a fresh job before the next revision round starts collecting memory.
    if mem is not None:
        try:
            active_intent = request_memory_intent or getattr(self, "_resolved_request_intent", "")
            await self._ensure_memory_job_started(
                user_prompt=user_message,
                intent=active_intent,
                read_intent=active_intent,
                write_intent=active_intent,
                profile_override=request_extra_info if isinstance(request_extra_info, dict) else None,
                reason="revision_after_memory_save",
            )
        except Exception as e:
            logger.warning(f"MemoryOrchestrator restart before modify failed (non-fatal): {e}")
    _orchestrator = self._memory_orchestrator_instance

    # Always call on_round_start for EVERY modify turn (round lifecycle tracking)
    if _orchestrator:
        try:
            _modify_context = dict(self._last_modify_context)
            if self._current_template_id and not _modify_context.get("template_id"):
                _modify_context["template_id"] = self._current_template_id
            await _orchestrator.on_round_start(
                user_message=user_message[:300],
                user_id=self.user_id,
                context=_modify_context,
                session_id=self.workspace.stem,
                agent_name="modify",
            )
        except Exception as e:
            logger.warning(f"Orchestrator on_round_start(modify round {turn}) failed (non-fatal): {e}")

    # Re-inject SYSTEM memory on every modify turn so latest WM updates are always reflected.
    _should_reinject_system = True
    _preclassified_intent = None
    if mem is not None:
        try:
            if hasattr(mem, "classifier") and mem.classifier:
                if hasattr(mem.classifier, 'set_turn_id'):
                    mem.classifier.set_turn_id(turn)
                _preclassified_intent = await mem.classifier.classify(user_message)
                if debug_tracer:
                    debug_tracer.log_step(turn, "intent", _preclassified_intent)
                info(f"Intent classified: {_preclassified_intent}")
                _inferred_ctx: dict = {}
                if isinstance(_preclassified_intent, dict):
                    _ts = _preclassified_intent.get("target_slide", "")
                    if _ts and str(_ts).lower() not in ("all", ""):
                        _is_named_type = not bool(re.match(r'^slide_?\d+$', str(_ts).lower()))
                        if _is_named_type:
                            _inferred_ctx["slide_type"] = _ts
                    _elem = _preclassified_intent.get("element_type", "")
                    if _elem:
                        _inferred_ctx["element_type"] = _elem
                if self._current_template_id:
                    _inferred_ctx["template_id"] = self._current_template_id
                self._last_modify_context = _inferred_ctx
        except Exception as e:
            logger.warning(f"Intent preclassification before preference compilation failed (non-fatal): {e}")

    # Stage 8 Phase 2: 在当前 modify turn 开始前提取 session-level preference。
    # 当前有效链路：
    # _append_session_preference()
    #   -> MemoryOrchestrator.on_user_preference()
    #   -> WorkingMemory.add_preference()
    #
    # 说明：
    # - 历史上的 user_message_analyzer / constraint-vs-preference 二分写法
    #   已不再作为 modify 阶段的主写入链路，相关残留分支已移除。
    # - 会重建 SYSTEM memory 的轮次：仅写入 WM，后续统一进入本轮 system prompt
    # - 不会重建 SYSTEM memory 的轮次：立即追加到 chat_history，保证本轮也可见
    _skip_session_preference_priming = bool(
        getattr(self, "_skip_next_session_preference_priming", False)
    )
    if _skip_session_preference_priming:
        self._skip_next_session_preference_priming = False
        info("Session preference priming skipped once (already written post-review).")

    try:
        if user_message and not _skip_session_preference_priming:
            await self._append_session_preference(
                user_message,
                "",
                append_to_history=not _should_reinject_system,
                intent=_preclassified_intent if isinstance(_preclassified_intent, dict) else None,
            )
    except Exception as _sp_e:
        logger.warning("Session preference priming failed (non-fatal): %s", _sp_e)
        append_memory_flow_trace(
            self,
            event="preference_write_failed",
            user_message=user_message,
            intent=_preclassified_intent if isinstance(_preclassified_intent, dict) else None,
            extra={
                "stage": "session_preference_priming",
                "error_type": type(_sp_e).__name__,
                "error": str(_sp_e),
            },
        )

    if _orchestrator is None and self._memory_orchestrator_instance is not None:
        _orchestrator = self._memory_orchestrator_instance
        try:
            _modify_context = dict(self._last_modify_context)
            if self._current_template_id and not _modify_context.get("template_id"):
                _modify_context["template_id"] = self._current_template_id
            await _orchestrator.on_round_start(
                user_message=user_message[:300],
                user_id=self.user_id,
                context=_modify_context,
                session_id=self.workspace.stem,
                agent_name="modify",
            )
        except Exception as e:
            logger.warning(f"Orchestrator on_round_start(modify round {turn}) after priming failed (non-fatal): {e}")

    if _should_reinject_system and _orchestrator:
        try:
            # Get structured memory components
            components = await _orchestrator.get_memory_components(
                user_message=user_message[:300],
                agent_name="modify",
            )
            _modify_memory_prompt = components.get("combined", "")
            wm_rule_specs_text = components.get("wm_rule_specs", "") or ""
            _fallback_memory_prompt = get_session_preference_fallback_prompt(self)
            if _fallback_memory_prompt:
                _modify_memory_prompt = "\n\n".join(
                    filter(None, [_modify_memory_prompt, _fallback_memory_prompt])
                )
                wm_rule_specs_text = "\n\n".join(
                    filter(None, [wm_rule_specs_text, _fallback_memory_prompt])
                )
            wm_rule_specs_text, _control_rule_trace = merge_wm_rule_specs_text(
                self,
                wm_rule_specs_text,
                orchestrator=_orchestrator,
                source="control_plane_active_wm",
            )
            if wm_rule_specs_text and wm_rule_specs_text not in _modify_memory_prompt:
                _modify_memory_prompt = "\n\n".join(
                    filter(None, [_modify_memory_prompt, wm_rule_specs_text])
                )
            append_memory_flow_trace(
                self,
                event="active_structured_rules_injected",
                user_message=user_message,
                intent=_preclassified_intent if isinstance(_preclassified_intent, dict) else None,
                extra=_control_rule_trace,
            )

            # Dump system-level injection trace
            _dumper = getattr(self.memory_system, 'artifact_dumper', None) if self.memory_system else None
            if _dumper:
                round_index = _orchestrator._job_mgr.round_count() if _orchestrator._job_mgr else turn
                _dumper.dump_injection_trace(
                    round_index,
                    level="system",
                    agent_role="modify",
                    turn=turn,
                    wm_preferences=components.get("wm_preferences", ""),
                    profile_execution_contract=components.get("profile_execution_contract", ""),
                    profile_execution_plan=components.get("profile_execution_plan", ""),
                    wm_experiences=components.get("wm_experiences", ""),
                    ltm_tool_experiences=components.get("ltm_tool_experiences", ""),
                    wm_round_history=components.get("wm_round_history", ""),
                    final_injected_text=_modify_memory_prompt,
                    injection_target="SYSTEM" if _modify_memory_prompt else "",
                    skipped_reason="" if _modify_memory_prompt else "no memory available",
                )

            if _modify_memory_prompt:
                _memory_injection_full = _modify_memory_prompt
                _memory_injection_text = _modify_memory_prompt[:500]

                # Inject into SYSTEM message with markers for periodic replacement
                current_system = agent_to_use.chat_history[0].text if agent_to_use.chat_history else agent_to_use.system

                _MEMORY_START_MARKER = "\n\n<!-- MEMORY_INJECTION_START -->"
                _MEMORY_END_MARKER = "<!-- MEMORY_INJECTION_END -->"
                if _MEMORY_START_MARKER in current_system:
                    start_idx = current_system.find(_MEMORY_START_MARKER)
                    end_idx = current_system.find(_MEMORY_END_MARKER)
                    if start_idx >= 0 and end_idx > start_idx:
                        current_system = current_system[:start_idx] + current_system[end_idx + len(_MEMORY_END_MARKER):]

                new_system = (
                    current_system.rstrip() +
                    f"{_MEMORY_START_MARKER}\n{_modify_memory_prompt}\n{_MEMORY_END_MARKER}"
                )
                agent_to_use.chat_history[0] = ChatMessage(
                    role=Role.SYSTEM,
                    content=new_system,
                )

                # Set orchestrator on agent for operation-level injection
                agent_to_use._memory_orchestrator = _orchestrator

                info(f"Injected memory to RevisionEditor SYSTEM (turn {turn}, {len(_modify_memory_prompt)} chars, "
                     f"components: {[k for k, v in components.items() if v and k != 'combined']})")

                if debug_tracer:
                    debug_tracer.log_step(turn, "system_memory_injection", {
                        "reinject_turn": turn,
                        "tokens": len(_modify_memory_prompt) // 4,
                    })
        except Exception as e:
            logger.warning(f"Orchestrator modify injection failed (non-fatal): {e}")

    if _should_reinject_system and not _orchestrator:
        try:
            _fallback_memory_prompt = get_session_preference_fallback_prompt(self)
            if _fallback_memory_prompt:
                _memory_injection_full = _fallback_memory_prompt
                _memory_injection_text = _fallback_memory_prompt[:500]
                wm_rule_specs_text = _fallback_memory_prompt
                wm_rule_specs_text, _control_rule_trace = merge_wm_rule_specs_text(
                    self,
                    wm_rule_specs_text,
                    orchestrator=None,
                    source="control_plane_fallback_active_wm",
                )
                _fallback_memory_prompt = wm_rule_specs_text
                _memory_injection_full = _fallback_memory_prompt
                _memory_injection_text = _fallback_memory_prompt[:500]
                append_memory_flow_trace(
                    self,
                    event="active_structured_rules_injected",
                    user_message=user_message,
                    intent=_preclassified_intent if isinstance(_preclassified_intent, dict) else None,
                    extra=_control_rule_trace,
                )

                current_system = agent_to_use.chat_history[0].text if agent_to_use.chat_history else agent_to_use.system
                _MEMORY_START_MARKER = "\n\n<!-- MEMORY_INJECTION_START -->"
                _MEMORY_END_MARKER = "<!-- MEMORY_INJECTION_END -->"
                if _MEMORY_START_MARKER in current_system:
                    start_idx = current_system.find(_MEMORY_START_MARKER)
                    end_idx = current_system.find(_MEMORY_END_MARKER)
                    if start_idx >= 0 and end_idx > start_idx:
                        current_system = current_system[:start_idx] + current_system[end_idx + len(_MEMORY_END_MARKER):]

                agent_to_use.chat_history[0] = ChatMessage(
                    role=Role.SYSTEM,
                    content=(
                        current_system.rstrip()
                        + f"{_MEMORY_START_MARKER}\n{_fallback_memory_prompt}\n{_MEMORY_END_MARKER}"
                    ),
                )
                info(
                    f"Injected session preference fallback to RevisionEditor SYSTEM "
                    f"(turn {turn}, {len(_fallback_memory_prompt)} chars)"
                )
        except Exception as e:
            logger.warning(f"Session preference fallback injection failed (non-fatal): {e}")

    if debug_tracer:
        debug_tracer.log_event("modify_start", {
            "turn": turn, "message": user_message[:200],
        })

    # Stage 15: Unified tool callback for modify path
    # All calls → on_operation_complete (populates tool log for auto_extract)
    # Errors  → on_tool_error (writes round experience to WM)
    if _orchestrator:
        try:
            def _make_modify_cb(orch):
                async def _cb(tool_name, arguments, result, is_error, duration_ms=0, reasoning="", reason_source=""):
                    try:
                        orch.on_operation_complete(
                            tool_name=tool_name,
                            args=str(arguments),
                            result=str(result)[:5000],
                            is_error=is_error,
                            duration_ms=duration_ms,
                            reasoning=reasoning or "",
                            reason_source=reason_source or "",
                        )
                    except Exception as _e:
                        logger.debug("on_operation_complete failed (non-fatal): %s", _e)
                    if is_error:
                        try:
                            orch.on_tool_error(
                                tool_name=tool_name,
                                args=str(arguments),
                                error_msg=str(result)[:500],
                            )
                        except Exception as _e:
                            logger.debug("on_tool_error failed (non-fatal): %s", _e)
                return _cb
            agent_to_use._tool_result_callback = _make_modify_cb(_orchestrator)
        except Exception:
            pass

    # ── Memory Pre-Steps [before action] ──
    enhanced_message = user_message
    intent = None
    _experience_text = ""  # Tracked for debug logging
    
    # Debug: log memory system state for experience injection diagnosis
    debug(f"[Experience] mem={mem is not None}, _session={self._session is not None}")
    if mem is not None:
        debug(f"[Experience] mem.db={hasattr(mem, 'db') and mem.db is not None}")
    
    # Memory Pre-Steps: unified inline path (Pipeline removed - was DEPRECATED)
    if mem is not None:
        try:
            # Step 1: Intent classification
            if isinstance(_preclassified_intent, dict):
                intent = _preclassified_intent
            elif hasattr(mem, "classifier") and mem.classifier:
                # Stage 4: 设置轮次 ID 用于 artifact 文件命名
                if hasattr(mem.classifier, 'set_turn_id'):
                    mem.classifier.set_turn_id(turn)
                intent = await mem.classifier.classify(user_message)
                if debug_tracer:
                    debug_tracer.log_step(turn, "intent", intent)
                info(f"Intent classified: {intent}")
            # Scope-aware preference: 从 intent 推导 context 供下一轮 SYSTEM 注入
            # target_slide 格式如 "slide_3" 或 "title"/"content" 等
            _inferred_ctx: dict = {}
            if isinstance(intent, dict):
                _ts = intent.get("target_slide", "")
                if _ts and str(_ts).lower() not in ("all", ""):
                    # 若 target_slide 是页面类型名（非纯数字编号），用作 slide_type
                    _is_named_type = not bool(re.match(r'^slide_?\d+$', str(_ts).lower()))
                    if _is_named_type:
                        _inferred_ctx["slide_type"] = _ts
                _elem = intent.get("element_type", "")
                if _elem:
                    _inferred_ctx["element_type"] = _elem
            if self._current_template_id:
                _inferred_ctx["template_id"] = self._current_template_id
            if _inferred_ctx:
                self._last_modify_context = _inferred_ctx

            # Step 2: Query relevant ExperienceTrace
            experience_text = ""
            _experience_text = ""  # also track in outer scope for debug log
            debug(f"[Experience] Inline path: checking mem.db = {hasattr(mem, 'db') and mem.db is not None}")
            if hasattr(mem, "db") and mem.db:
                try:
                    exp_writer = self._make_exp_writer()

                    all_exps = await exp_writer.query_for_task(
                        user_task=user_message[:300],
                        session_id=self.workspace.stem, limit=8,
                    )
                    debug(f"[Experience] Query returned {len(all_exps)} traces")
                    if all_exps:
                        experience_text = exp_writer.format_for_prompt(all_exps)
                        _experience_text = experience_text
                        _memory_injection_full = "\n\n".join(
                            filter(None, [_memory_injection_full, experience_text])
                        )  # Stage 4: Track full injection
                        _memory_injection_text = _memory_injection_full[:500]  # Stage 4: Compressed version
                        await exp_writer.mark_reused([t.id for t in all_exps])
                        info(f"Found {len(all_exps)} relevant experiences")
                    if debug_tracer:
                        debug_tracer.log_step(turn, "experience_queried", {
                            "total_retrieved": len(all_exps),
                            "experiences": [
                                {
                                    "id": t.id[:12],
                                    "description": t.task_description[:150],
                                    "lessons": t.lessons_learned[:200] if t.lessons_learned else "",
                                    "outcome": t.final_outcome,
                                    "confidence": t.confidence,
                                }
                                for t in all_exps[:5]
                            ],
                            "injection_preview": experience_text[:500] if experience_text else "(empty)",
                        })
                except Exception as e:
                    logger.warning(f"ExperienceTrace query failed (non-fatal): {e}")

            # Step 4: Build enhanced message
            context_parts = []
            if experience_text.strip():
                context_parts.append(
                    f"<experience_context>\n"
                    f"{experience_text}\n"
                    f"</experience_context>"
                )
            if context_parts:
                enhanced_message += "\n\n" + "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"Memory pre-steps failed (non-fatal): {e}")

    modify_plan = self._build_modify_execution_plan(
        user_message=user_message,
        intent=intent if isinstance(intent, dict) else None,
        wm_rule_specs_text=wm_rule_specs_text,
    )
    active_structured_preferences = self._parse_rule_specs_from_injection(wm_rule_specs_text)
    active_future_structured_preferences: list[dict[str, Any]] = []
    for spec in active_structured_preferences:
        if self._future_preference_verifiable_on_single_slide(spec):
            active_future_structured_preferences.append(spec)
    if modify_plan is not None:
        info(
            "RevisionEditor execution plan: scope=%s, operation_kind=%s, targets=%s, rules=%s",
            modify_plan.scope,
            getattr(modify_plan, "operation_kind", "style"),
            len(modify_plan.target_slide_paths),
            len(modify_plan.target_rule_ids),
        )
    _set_current_modify_context_compat(
        workspace=self.workspace,
        target_slide_paths=[
            self._workspace_relative_label(path)
            for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
        ],
        operation_kind=getattr(modify_plan, "operation_kind", "") if modify_plan is not None else "",
        coverage_required=bool(getattr(modify_plan, "coverage_required", False)) if modify_plan is not None else False,
        raw_user_message=user_message,
        user_preference_rule_specs=active_structured_preferences,
        diagram_contract=getattr(modify_plan, "diagram_contract", None) if modify_plan is not None else None,
        rewrite_decision=getattr(modify_plan, "rewrite_decision", None) if modify_plan is not None else None,
        strict_visual_preference_eval=bool(
            getattr(self, "strict_visual_preference_eval", False)
            or os.environ.get("MEMSLIDES_STRICT_VISUAL_PREFERENCE_EVAL", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    )
    new_element_rule_applications: list[dict[str, Any]] = []
    if modify_plan is not None and not is_preference_update_plan(modify_plan):
        try:
            new_element_rule_applications = await build_new_element_rule_applications(
                self,
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                plan=modify_plan,
                rule_specs=self._parse_rule_specs_from_injection(wm_rule_specs_text),
                llm=getattr(agent_to_use, "llm", None),
            )
            if new_element_rule_applications:
                modify_plan.new_element_rule_applications = new_element_rule_applications
                existing_applicable = list(getattr(modify_plan, "applicable_rule_ids", []) or [])
                for app in new_element_rule_applications:
                    rule_id = str(app.get("rule_id", "") or "").strip()
                    if rule_id and rule_id not in existing_applicable:
                        existing_applicable.append(rule_id)
                modify_plan.applicable_rule_ids = existing_applicable
                info(
                    "New-element WM rule applications: %s",
                    ",".join(existing_applicable),
                )
        except Exception as exc:
            logger.warning("New-element WM rule applicability failed (fail-closed for applications): %s", exc)
            append_memory_flow_trace(
                self,
                event="new_element_rule_applicability_failed",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                plan=modify_plan,
                extra={"error": str(exc)},
            )
    _preference_update_turn = is_preference_update_plan(modify_plan)
    if _preference_update_turn:
        info(
            "Memory-flow classification: preference_update memory-only turn "
            "(future_rules=%s, existing_targets=0)",
            len(active_future_structured_preferences),
        )
    elif active_future_structured_preferences:
        info(
            "Memory-flow classification: current edit/structural turn with %s active future preference rule(s)",
            len(active_future_structured_preferences),
        )

    # ── Stage 4: Cognitive Memory Collector — begin round (after memory injection built) ──
    if self.memory_system and hasattr(self.memory_system, 'collector') and self.memory_system.collector:
        try:
            # Use orchestrator's round_count for round_id to avoid overlap with run() indices
            _collector_round_id = (
                _orchestrator._job_mgr.round_count()
                if _orchestrator and _orchestrator._job_mgr
                else turn
            )
            self.memory_system.collector.begin_round(
                round_id=_collector_round_id,
                user_message=user_message,
                agent_name="modify",                     # Fix: was "design"
                memory_injection=_memory_injection_text,  # Stage 4: Compressed memory
                memory_injection_full=_memory_injection_full,  # Stage 4: Full memory
                session_id=self.workspace.stem,         # Stage 4: Session ID
                user_id=self.user_id,                      # Stage 4: User ID
            )
        except Exception as e:
            logger.warning(f"MemoryCollector begin_round failed (non-fatal): {e}")

    # Step 5: Extract before_params via StateCoordinator (G2 Phase 1)
    target_slide = None
    before_params = None
    if mem is not None and intent:
        try:
            forced_target_slide = None
            if modify_plan is not None and modify_plan.target_slide_paths:
                if len(modify_plan.target_slide_paths) > 1:
                    forced_target_slide = "all"
                else:
                    forced_target_slide = modify_plan.target_slide_paths[0].stem

            target_slide = forced_target_slide or intent.get("target_slide", "")
            if target_slide and target_slide.lower() == "all":
                # Batch baseline extraction for all slides
                coordinator = getattr(mem, "state_coordinator", None)
                all_paths = self._resolve_all_slide_paths()
                if all_paths and coordinator is not None:
                    before_params = {}
                    for sp in all_paths:
                        try:
                            bp = await coordinator.before_modification(sp, slide_type="html")
                            if bp:
                                before_params[sp.stem] = bp
                        except Exception:
                            pass
                    if before_params and debug_tracer:
                        debug_tracer.log_step(turn, "params_before", {
                            sid: bp.to_dict() if hasattr(bp, 'to_dict') else {}
                            for sid, bp in before_params.items()
                        })
                    info(f"Batch extracted before_params for {len(before_params)}/{len(all_paths)} slides")
            elif target_slide:
                slide_html_path = self._resolve_slide_path(target_slide)
                if slide_html_path:
                    # Prefer StateCoordinator (versioned snapshots + params store)
                    coordinator = getattr(mem, "state_coordinator", None)
                    if coordinator is not None:
                        before_params = await coordinator.before_modification(
                            slide_html_path, slide_type="html",
                        )
                    elif hasattr(mem, "state_extractor") and mem.state_extractor:
                        before_params = mem.state_extractor.extract_from_html(slide_html_path)
                    if before_params and debug_tracer:
                        debug_tracer.log_step(turn, "params_before", before_params.to_dict() if hasattr(before_params, 'to_dict') and before_params else {})
                    info(f"Extracted before_params for {target_slide}")
        except Exception as e:
            logger.warning(f"Param extraction (before) failed (non-fatal): {e}")

    # ── Stage 2/10: Cognitive Memory Observability (five-class) ──
    # Stage 10: 新版 SYSTEM 注入已在上方完成，此处仅做检索观测日志
    _cognitive_retrieval_results = []
    if self.memory_system and hasattr(self.memory_system, 'retriever'):
        try:
            _retriever = getattr(self.memory_system, 'retriever', None)

            # Retrieve episodes for observability logging
            if _retriever and hasattr(_retriever, 'retrieve'):
                try:
                    _cognitive_retrieval_results = await _retriever.retrieve(
                        query=user_message, user_id=self.user_id, top_k=5,
                    )
                except Exception:
                    pass

            # Artifact #2: Retrieval trace
            if debug_tracer:
                debug_tracer.log_retrieval(turn, user_message, _cognitive_retrieval_results)

            # Artifact #3: Cognitive injection trace
            if debug_tracer:
                _prompt_lower = (_memory_injection_full or "").lower()
                _type_markers = {
                    "constraint": "user constraints",
                    "profile": "user design profile",
                    "episode": "relevant past experiences",
                    "rule": "learned strategies",
                    "foresight": "accepted pattern suggestions",
                }
                debug_tracer.log_cognitive_injection(turn, {
                    "injected_types": [t for t, m in _type_markers.items() if m in _prompt_lower],
                    "total_injected_chars": len(_memory_injection_full) if _memory_injection_full else 0,
                    "estimated_tokens": len(_memory_injection_full) // 4 if _memory_injection_full else 0,
                    "episode_count": len(_cognitive_retrieval_results),
                })
        except Exception as e:
            logger.warning(f"Cognitive memory injection failed (non-fatal): {e}")

    # Stage 4: 更新 collector 的 memory_injection（保留完整 SYSTEM/fallback/experience 注入）
    if self.memory_system and hasattr(self.memory_system, 'collector') and self.memory_system.collector:
        try:
            collector = self.memory_system.collector
            if collector._current_round:
                full_injection = _memory_injection_full.strip()
                collector._current_round.memory_injection = full_injection[:500]
                collector._current_round.memory_injection_full = full_injection
        except Exception as e:
            logger.debug(f"Failed to update collector memory_injection (non-fatal): {e}")

    if debug_tracer:
        _debug_prompt = (
            f"=== User Message ===\n{user_message}\n\n"
            "=== Experience Context ===\n"
            + (f"<experience_context>\n{_experience_text}\n</experience_context>" if _experience_text.strip() else "(empty)")
        )
        debug_tracer.log_step(turn, "prompt_injected", _debug_prompt)

    # ── Rollback: create checkpoint before modification ──
    # Save chat history length so rollback can trim back
    agent_to_use_history_len = len(agent_to_use.chat_history)
    self._history_checkpoint = agent_to_use_history_len
    self._previous_final = self.intermediate_output.get("final")

    if mem is not None and hasattr(mem, "rollback_manager") and mem.rollback_manager:
        try:
            slide_html_dir = self.intermediate_output.get("slide_html_dir")
            if slide_html_dir:
                slide_html_dir = Path(slide_html_dir)
                # Set disk persistence dir before creating checkpoints
                mem.rollback_manager.set_checkpoint_dir(self.workspace)
                for html_file in slide_html_dir.glob("*.html"):
                    slide_id = html_file.stem
                    mem.rollback_manager.create_checkpoint(
                        slide_id, html_file.read_text(encoding="utf-8")
                    )
                debug(f"Created rollback checkpoints for {slide_html_dir}")
        except Exception as e:
            logger.warning(f"Rollback checkpoint creation failed (non-fatal): {e}")

    # ── Tool expansion: only needed if using DeckDesigner as fallback ──
    # RevisionEditor has complete toolset via RevisionEditor.yaml configuration
    if agent_to_use is self.designagent and not getattr(self, "_modify_tools_expanded", False):
        try:
            info("Using DeckDesigner for modify (RevisionEditor unavailable), expanding toolset...")
            # Add asset tool server tools (image_caption, image_generation, document_summary)
            if "asset_tools" in self.agent_env._server_tools:
                for tool_name in self.agent_env._server_tools["asset_tools"]:
                    tool_spec = self.agent_env._tools_dict.get(tool_name)
                    if tool_spec and tool_spec not in self.designagent.tools:
                        self.designagent.tools.append(tool_spec)
                        info(f"RevisionEditor-mode: added tool '{tool_name}' to DeckDesigner Agent")

            # Add additional tools needed for modify workflow fallback.
            for extra_tool in (
                "list_files",
                "read_file",
                "insert_slide",
                "delete_slide",
                "scan_slide_index",
                "batch_update_css_rule",
                "batch_update_semantic_style",
                "patch_semantic_inline_style",
            ):
                tool_spec = self.agent_env._tools_dict.get(extra_tool)
                if tool_spec and tool_spec not in self.designagent.tools:
                    self.designagent.tools.append(tool_spec)
                    info(f"RevisionEditor-mode: added tool '{extra_tool}' to DeckDesigner Agent")

            self.designagent._base_tool_names = [
                str(tool.get("function", {}).get("name", "") or "").strip()
                for tool in self.designagent.tools
                if isinstance(tool, dict)
                and isinstance(tool.get("function"), dict)
                and str(tool.get("function", {}).get("name", "") or "").strip()
            ]
            self._modify_tools_expanded = True
        except Exception as e:
            logger.warning(f"RevisionEditor-mode tool expansion failed (non-fatal): {e}")

    tool_scope_contract = None
    tool_capability_validation: dict[str, Any] = {}
    try:
        restore_agent_base_tools(agent_to_use)
        tool_policy_plan = await build_modify_tool_policy_plan(
            self,
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            execution_plan=modify_plan,
            wm_rule_specs_text=wm_rule_specs_text,
            agent=agent_to_use,
        )
        aligned_modify_plan = align_modify_execution_plan_to_tool_policy(
            modify_plan,
            tool_policy_plan,
        )
        if aligned_modify_plan is not modify_plan:
            modify_plan = aligned_modify_plan
            _set_current_modify_context_compat(
                workspace=self.workspace,
                target_slide_paths=[
                    self._workspace_relative_label(path)
                    for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
                ],
                operation_kind=getattr(modify_plan, "operation_kind", "") if modify_plan is not None else "",
                coverage_required=bool(getattr(modify_plan, "coverage_required", False)) if modify_plan is not None else False,
                raw_user_message=user_message,
                user_preference_rule_specs=active_structured_preferences,
                diagram_contract=getattr(modify_plan, "diagram_contract", None) if modify_plan is not None else None,
                rewrite_decision=getattr(modify_plan, "rewrite_decision", None) if modify_plan is not None else getattr(tool_policy_plan, "rewrite_decision", None),
                strict_visual_preference_eval=bool(
                    getattr(self, "strict_visual_preference_eval", False)
                    or os.environ.get("MEMSLIDES_STRICT_VISUAL_PREFERENCE_EVAL", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                ),
            )
            append_modify_tool_policy_trace(
                self,
                event="execution_plan_aligned_to_policy",
                plan=modify_plan,
                policy=tool_policy_plan,
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                note=(
                    "Aligned model-facing execution plan to the final tool policy so the "
                    "RevisionEditor instructions match structural or controlled-rewrite authority."
                ),
            )
        elif tool_policy_plan is not None and getattr(tool_policy_plan, "operation_kind", "") == "controlled_rewrite":
            _set_current_modify_context_compat(
                workspace=self.workspace,
                target_slide_paths=[
                    self._workspace_relative_label(path)
                    for path in (tool_policy_plan.target_slide_paths or [])
                ],
                operation_kind="controlled_rewrite",
                coverage_required=True,
                raw_user_message=user_message,
                user_preference_rule_specs=active_structured_preferences,
                diagram_contract=getattr(modify_plan, "diagram_contract", None) if modify_plan is not None else None,
                rewrite_decision=getattr(tool_policy_plan, "rewrite_decision", None),
                strict_visual_preference_eval=bool(
                    getattr(self, "strict_visual_preference_eval", False)
                    or os.environ.get("MEMSLIDES_STRICT_VISUAL_PREFERENCE_EVAL", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                ),
            )
        tool_scope_contract = apply_modify_tool_policy(
            self,
            agent_to_use,
            modify_plan,
            tool_policy_plan,
            event="initial_policy",
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
        )
        info(
            "Modify tool policy applied: scope=%s, operation_kind=%s, groups=%s, source=%s, allowed=%s, removed=%s",
            tool_scope_contract.scope,
            getattr(tool_scope_contract, "operation_kind", "style"),
            ",".join(getattr(tool_scope_contract, "tool_groups", []) or []),
            getattr(tool_scope_contract, "policy_source", ""),
            len(tool_scope_contract.allowed_tools),
            len(tool_scope_contract.removed_tools),
        )
        tool_capability_validation = validate_modify_tool_capability_contract(
            tool_policy_plan,
            getattr(tool_scope_contract, "allowed_tools", []) or [],
            plan=modify_plan,
        )
        append_modify_tool_policy_trace(
            self,
            event="capability_preflight",
            plan=modify_plan,
            policy=tool_policy_plan,
            allowed_tools=getattr(tool_scope_contract, "allowed_tools", []) or [],
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            note="Revision capability contract preflight.",
            extra=tool_capability_validation,
        )
        if not tool_capability_validation.get("ok", False):
            missing = ", ".join(tool_capability_validation.get("missing_tools", []) or [])
            capability = str(tool_capability_validation.get("capability") or "modify")
            warning(
                "Modify capability preflight failed: capability=%s missing=%s",
                capability,
                missing,
            )
            yield ChatMessage(
                role=Role.SYSTEM,
                content=(
                    "This revision cannot start because the required editing tools are not available "
                    f"for the `{capability}` capability. Missing: {missing or 'required edit tools'}. "
                    "No files were changed; please retry after the worker toolset is refreshed."
                ),
            )
            return
    except Exception as e:
        tool_policy_plan = None
        logger.warning(f"Modify tool policy application failed (non-fatal): {e}")

    # Inject workspace file context so the model knows available resources
    # Stage 7: 注入模板约束到 RevisionEditor Agent（使用 TemplateGuideBuilder.build_for_modify）
    if self._template_profile and self._guide_builder:
        try:
            template_modify_prompt = self._guide_builder.build_for_modify()
            if template_modify_prompt:
                enhanced_message += (
                    f"\n\n<template_constraints priority=\"highest\">\n"
                    f"{template_modify_prompt}\n"
                    f"</template_constraints>"
                )
                info(f"Injected template constraints to RevisionEditor Agent ({len(template_modify_prompt)} chars)")
                
                # 保存注入日志到 .history 目录
                _modify_prompt_file = self.workspace / ".history" / f"template_prompt_modify_{self._guide_builder._modify_count:02d}.md"
                _modify_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                _modify_prompt_file.write_text(template_modify_prompt, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Template constraint injection to RevisionEditor failed (non-fatal): {e}")

    # IMPORTANT: All paths must be relative to workspace root, since MCP tools
    # resolve paths via os.chdir(workspace).
    try:
        workspace_context = self._build_workspace_context_block()
        if workspace_context:
            enhanced_message += "\n\n" + workspace_context
    except Exception as e:
        logger.warning(f"Workspace context injection failed (non-fatal): {e}")

    if modify_plan is not None:
        enhanced_message += "\n\n" + self._render_modify_execution_plan(modify_plan)
    if tool_policy_plan is not None:
        enhanced_message += "\n\n" + self._render_modify_tool_policy_plan(tool_policy_plan)
    if tool_scope_contract is not None:
        enhanced_message += "\n\n" + render_current_tool_authority(
            tool_scope_contract,
            tool_policy_plan,
            tool_capability_validation,
        )

    # ── Agent execution (collect full response + tool results) ──
    agent_to_use.chat_history.append(
        ChatMessage(role=Role.USER, content=enhanced_message)
    )

    yield ChatMessage(
        role=Role.SYSTEM,
        content=f"Processing modification round {turn}...",
    )

    agent_responses = []
    tool_calls_log = []
    _initial_tool_history_len = len(getattr(self.agent_env, "tool_history", []) or [])
    _consecutive_unchanged = 0  # Track consecutive UNCHANGED inspect results
    _UNCHANGED_THRESHOLD = 3    # Force finalize after N consecutive UNCHANGED
    _modify_plan_target_keys = {
        str(path.resolve())
        for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
    }
    _modify_plan_covered_keys: set[str] = set()
    _modify_plan_initial_hashes = {
        str(path.resolve()): _file_sha256(path)
        for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
    }
    _new_element_initial_hashes = {
        str(path.resolve()): _file_sha256(path)
        for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
    } if new_element_rule_applications else {}
    _new_element_before_html_by_path: dict[str, str] = {}
    if _new_element_initial_hashes:
        for raw_path in _new_element_initial_hashes:
            slide_path = Path(raw_path)
            try:
                _new_element_before_html_by_path[raw_path] = slide_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                _new_element_before_html_by_path[raw_path] = ""
    _preference_update_initial_hashes = {
        str(path.resolve()): _file_sha256(path)
        for path in self._resolve_all_slide_paths()
    } if _preference_update_turn else {}
    _inserted_slide_paths: dict[str, Path] = {}
    _initial_slide_count = len(self._resolve_all_slide_paths())
    _expected_slide_delta = int(getattr(tool_policy_plan, "expected_slide_delta", 0) or 0)
    _modify_plan_idle_rounds = 0
    _redundant_read_only_rounds = 0
    _read_only_seen_keys: set[str] = set()
    _last_read_only_progress: dict[str, Any] = {}
    _active_recovery_request: dict[str, Any] | None = None
    _controlled_rewrite_turn = (
        tool_policy_plan is not None
        and getattr(tool_policy_plan, "operation_kind", "") == "controlled_rewrite"
        and _expected_slide_delta == 0
    )
    _future_only_guard_turn = (
        plan_has_future_only_rules(modify_plan)
        and not _preference_update_turn
        and _expected_slide_delta == 0
        and getattr(modify_plan, "operation_kind", "") != "structural"
        and getattr(modify_plan, "operation_kind", "") != "diagram_layout"
        and getattr(modify_plan, "operation_kind", "") != "controlled_rewrite"
        and (
            tool_policy_plan is None
            or getattr(tool_policy_plan, "operation_kind", "") != "structural"
        )
        and (
            tool_policy_plan is None
            or getattr(tool_policy_plan, "operation_kind", "") != "diagram_layout"
        )
        and (
            tool_policy_plan is None
            or getattr(tool_policy_plan, "operation_kind", "") != "controlled_rewrite"
        )
    )
    _all_existing_initial_hashes = {
        str(path.resolve()): _file_sha256(path)
        for path in self._resolve_all_slide_paths()
    } if _future_only_guard_turn else {}
    _finalize_succeeded = False
    _pending_finalize_gate_inspect: list[dict[str, str]] = []
    _blocked_tool_batches = 0

    def _current_slide_delta() -> int:
        return len(self._resolve_all_slide_paths()) - _initial_slide_count

    def _slide_delta_followup() -> str:
        current_delta = _current_slide_delta()
        if _expected_slide_delta > 0:
            return (
                "SYSTEM: The user requested a real new slide, but the deck page count has not increased yet "
                f"(expected +{_expected_slide_delta}, current {current_delta:+d}). "
                "Call `scan_slide_index`, then `insert_slide`, fill the returned target with `write_new_slide_file`, "
                "inspect the completed slide, and only then finalize."
            )
        if _expected_slide_delta < 0:
            return (
                "SYSTEM: The user requested a real slide deletion, but the deck page count has not decreased as requested "
                f"(expected {_expected_slide_delta:+d}, current {current_delta:+d}). "
                "Call `scan_slide_index`, then `delete_slide(..., renumber=true)`, inspect the neighboring slide order, "
                "and only then finalize."
            )
        return ""

    def _slide_delta_unmet() -> bool:
        if _expected_slide_delta > 0:
            return _current_slide_delta() < _expected_slide_delta
        if _expected_slide_delta < 0:
            return _current_slide_delta() > _expected_slide_delta
        return False

    def _block_preference_update_mutations(tool_calls: list[Any], *, event: str) -> tuple[list[Any], list[str]]:
        if not _preference_update_turn or not tool_calls:
            return tool_calls, []
        blocked_names = [
            tc.function.name
            for tc in tool_calls
            if tc.function.name in PREFERENCE_UPDATE_MUTATION_TOOLS
        ]
        if not blocked_names:
            return tool_calls, []
        warning(
            "Blocked mutation tool(s) during preference_update turn: %s",
            ", ".join(blocked_names),
        )
        append_modify_tool_policy_trace(
            self,
            event=event,
            plan=modify_plan,
            policy=tool_policy_plan,
            allowed_tools=[
                str(tool.get("function", {}).get("name", "") or "").strip()
                for tool in getattr(agent_to_use, "tools", [])
                if isinstance(tool, dict)
                and str(tool.get("function", {}).get("name", "") or "").strip()
            ],
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            note="Blocked mutation tools because this is a memory-only preference_update turn.",
            extra={"blocked_tools": blocked_names},
        )
        return [
            tc for tc in tool_calls
            if tc.function.name not in PREFERENCE_UPDATE_MUTATION_TOOLS
        ], blocked_names

    def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                payload = json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
        return {}

    def _future_only_call_scope_violation(tool_call: Any) -> str:
        if not _future_only_guard_turn:
            return ""
        tool_name = str(getattr(tool_call.function, "name", "") or "").strip()
        if _active_recovery_request and tool_name == "write_html_file":
            return ""
        if tool_name in {"batch_update_css_rule", "batch_update_semantic_style"}:
            return f"{tool_name} is a deck/batch mutation tool, but this turn only allows the explicit current slide edit plus a future-only rule."
        if tool_name in {"insert_slide", "delete_slide", "write_new_slide_file", "write_html_file"}:
            return f"{tool_name} is not allowed for a current-slide edit plus future-only preference turn."
        if tool_name == "patch_semantic_inline_style":
            args = _parse_tool_arguments(getattr(tool_call.function, "arguments", None))
            candidate = self._resolve_modify_coverage_path(str(args.get("file_path", "") or ""))
            if candidate is None:
                return "patch_semantic_inline_style must target the explicit current slide in this future-only preference turn."
            if str(candidate.resolve()) not in _modify_plan_target_keys:
                return f"patch_semantic_inline_style targets {candidate}, outside the explicit current-slide plan."
        return ""

    def _block_future_only_scope_mutations(tool_calls: list[Any], *, event: str) -> tuple[list[Any], list[dict[str, str]]]:
        if not _future_only_guard_turn or not tool_calls:
            return tool_calls, []
        blocked: list[dict[str, str]] = []
        kept: list[Any] = []
        for tc in tool_calls:
            reason = _future_only_call_scope_violation(tc)
            if reason:
                blocked.append(
                    {
                        "tool": str(getattr(tc.function, "name", "") or ""),
                        "reason": reason,
                    }
                )
            else:
                kept.append(tc)
        if not blocked:
            return tool_calls, []
        warning(
            "Blocked %s out-of-scope mutation tool(s) during current-edit + future-only preference turn.",
            len(blocked),
        )
        append_modify_tool_policy_trace(
            self,
            event=event,
            plan=modify_plan,
            policy=tool_policy_plan,
            allowed_tools=[
                str(tool.get("function", {}).get("name", "") or "").strip()
                for tool in getattr(agent_to_use, "tools", [])
                if isinstance(tool, dict)
                and str(tool.get("function", {}).get("name", "") or "").strip()
            ],
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            note=(
                "Blocked mutation tools because LLM semantic compilation marked the remembered "
                "rule as future-only; current execution is limited to explicit target slide(s)."
            ),
            extra={"blocked_tools": blocked},
        )
        return kept, blocked

    def _recovery_call_scope_violation(tool_call: Any) -> str:
        if not _active_recovery_request:
            return ""
        tool_name = str(getattr(tool_call.function, "name", "") or "").strip()
        if tool_name in {"insert_slide", "delete_slide", "write_new_slide_file"}:
            return f"{tool_name} is not allowed during controlled rewrite recovery; the slide count must stay unchanged."
        if tool_name != "write_html_file":
            return ""
        args = _parse_tool_arguments(getattr(tool_call.function, "arguments", None))
        if not bool(args.get("force_regenerate", False)):
            return "write_html_file recovery must use force_regenerate=true."
        if not str(args.get("expected_hash", "") or "").strip():
            return "write_html_file recovery must include expected_hash from read_slide_snapshot."
        target = self._resolve_modify_coverage_path(str(args.get("file_path", "") or ""))
        recovery_target = str(_active_recovery_request.get("target_slide_path", "") or "").strip()
        if recovery_target and target is not None and str(target.resolve()) != str(Path(recovery_target).resolve()):
            return f"write_html_file targets {target}, outside the controlled recovery target."
        return ""

    def _controlled_rewrite_call_scope_violation(tool_call: Any) -> str:
        if not _controlled_rewrite_turn:
            return ""
        tool_name = str(getattr(tool_call.function, "name", "") or "").strip()
        if tool_name in {
            "insert_slide",
            "delete_slide",
            "write_new_slide_file",
            "batch_update_css_rule",
            "batch_update_semantic_style",
            "patch_semantic_inline_style",
            "render_flowchart_asset",
        }:
            return f"{tool_name} is not allowed during a controlled single-slide rewrite."
        if tool_name != "write_html_file":
            return ""
        args = _parse_tool_arguments(getattr(tool_call.function, "arguments", None))
        if not bool(args.get("force_regenerate", False)):
            return "write_html_file must use force_regenerate=true for controlled rewrite."
        if not str(args.get("expected_hash", "") or "").strip():
            return "write_html_file must include expected_hash from read_slide_snapshot."
        target = self._resolve_modify_coverage_path(str(args.get("file_path", "") or ""))
        if target is None:
            return "write_html_file must target the listed controlled rewrite slide."
        policy_targets = {
            str(path.resolve())
            for path in (tool_policy_plan.target_slide_paths if tool_policy_plan is not None else [])
        }
        if policy_targets and str(target.resolve()) not in policy_targets:
            return f"write_html_file targets {target}, outside the controlled rewrite target."
        return ""

    def _block_controlled_rewrite_scope_mutations(tool_calls: list[Any], *, event: str) -> tuple[list[Any], list[dict[str, str]]]:
        if not _controlled_rewrite_turn or not tool_calls:
            return tool_calls, []
        blocked: list[dict[str, str]] = []
        kept: list[Any] = []
        for tc in tool_calls:
            reason = _controlled_rewrite_call_scope_violation(tc)
            if reason:
                blocked.append(
                    {
                        "tool": str(getattr(tc.function, "name", "") or ""),
                        "reason": reason,
                    }
                )
            else:
                kept.append(tc)
        if blocked:
            append_modify_tool_policy_trace(
                self,
                event=event,
                plan=modify_plan,
                policy=tool_policy_plan,
                allowed_tools=[
                    str(tool.get("function", {}).get("name", "") or "").strip()
                    for tool in getattr(agent_to_use, "tools", [])
                    if isinstance(tool, dict)
                    and str(tool.get("function", {}).get("name", "") or "").strip()
                ],
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                note="Blocked tools outside controlled rewrite protocol.",
                extra={"blocked_tools": blocked},
            )
        return kept, blocked

    def _block_recovery_scope_mutations(tool_calls: list[Any], *, event: str) -> tuple[list[Any], list[dict[str, str]]]:
        if not _active_recovery_request or not tool_calls:
            return tool_calls, []
        blocked: list[dict[str, str]] = []
        kept: list[Any] = []
        for tc in tool_calls:
            reason = _recovery_call_scope_violation(tc)
            if reason:
                blocked.append(
                    {
                        "tool": str(getattr(tc.function, "name", "") or ""),
                        "reason": reason,
                    }
                )
            else:
                kept.append(tc)
        if blocked:
            append_modify_tool_policy_trace(
                self,
                event=event,
                plan=modify_plan,
                policy=tool_policy_plan,
                allowed_tools=[
                    str(tool.get("function", {}).get("name", "") or "").strip()
                    for tool in getattr(agent_to_use, "tools", [])
                    if isinstance(tool, dict)
                    and str(tool.get("function", {}).get("name", "") or "").strip()
                ],
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                note="Blocked mutation tools outside controlled rewrite recovery protocol.",
                extra={"blocked_tools": blocked, "recovery_request": _active_recovery_request},
            )
        return kept, blocked

    def _append_recovery_scope_guard(blocked: list[dict[str, str]]) -> None:
        blocked_names = ", ".join(item.get("tool", "") for item in blocked if item.get("tool")) or "mutation tool"
        agent_to_use.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    "SYSTEM: Controlled rewrite recovery is active. "
                    f"Blocked out-of-protocol tool(s): {blocked_names}. "
                    "Use the recovery protocol exactly: `read_slide_snapshot` on the target, then "
                    "`write_html_file(..., force_regenerate=true, expected_hash=<content_hash>)`, then `inspect_slide`. "
                    "Do not change slide count or write a different slide."
                ),
            )
        )

    def _append_controlled_rewrite_scope_guard(blocked: list[dict[str, str]]) -> None:
        blocked_names = ", ".join(item.get("tool", "") for item in blocked if item.get("tool")) or "mutation tool"
        target_labels = ", ".join(
            self._workspace_relative_label(path)
            for path in (tool_policy_plan.target_slide_paths if tool_policy_plan is not None else [])
        ) or "(target slide)"
        agent_to_use.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    "SYSTEM: Controlled rewrite is active for a single existing slide. "
                    f"Target: {target_labels}. Blocked out-of-protocol tool(s): {blocked_names}. "
                    "Use `read_slide_snapshot`, then `write_html_file(..., force_regenerate=true, expected_hash=<content_hash>)` "
                    "on the same slide, then `inspect_slide`. Do not add/delete slides or batch-edit the deck."
                ),
            )
        )

    def _block_unexpected_slide_count_mutations(tool_calls: list[Any], *, event: str) -> tuple[list[Any], list[str]]:
        if _expected_slide_delta != 0 or not tool_calls:
            return tool_calls, []
        blocked_names = [
            str(getattr(tc.function, "name", "") or "").strip()
            for tc in tool_calls
            if str(getattr(tc.function, "name", "") or "").strip() in {"insert_slide", "delete_slide"}
        ]
        if not blocked_names:
            return tool_calls, []
        warning(
            "Blocked slide-count mutation(s) during zero-delta RevisionEditor turn: %s",
            ", ".join(blocked_names),
        )
        append_modify_tool_policy_trace(
            self,
            event=event,
            plan=modify_plan,
            policy=tool_policy_plan,
            allowed_tools=[
                str(tool.get("function", {}).get("name", "") or "").strip()
                for tool in getattr(agent_to_use, "tools", [])
                if isinstance(tool, dict)
                and str(tool.get("function", {}).get("name", "") or "").strip()
            ],
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            note="Blocked insert/delete because the current modify policy expects no slide-count change.",
            extra={
                "blocked_tools": blocked_names,
                "expected_slide_delta": _expected_slide_delta,
                "current_slide_delta": _current_slide_delta(),
            },
        )
        kept = [
            tc for tc in tool_calls
            if str(getattr(tc.function, "name", "") or "").strip() not in {"insert_slide", "delete_slide"}
        ]
        return kept, blocked_names

    def _append_future_only_scope_guard(blocked: list[dict[str, str]]) -> None:
        blocked_names = ", ".join(item.get("tool", "") for item in blocked if item.get("tool")) or "mutation tool"
        target_labels = ", ".join(
            self._workspace_relative_label(path)
            for path in (modify_plan.target_slide_paths if modify_plan is not None else [])
        ) or "(no current target)"
        agent_to_use.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    "SYSTEM: The remembered rule in this turn is future-only. "
                    f"Blocked out-of-scope tool(s): {blocked_names}. "
                    f"Only modify the explicit current target slide(s): {target_labels}. "
                    "Do not batch-edit or rewrite old slides outside those target(s); then inspect and finalize."
                ),
            )
        )

    def _append_unexpected_slide_count_guard(blocked_names: list[str]) -> None:
        blocked_text = ", ".join(name for name in blocked_names if name) or "insert_slide/delete_slide"
        agent_to_use.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    "SYSTEM: This RevisionEditor turn does not allow any slide-count change "
                    f"(expected slide delta 0). Blocked tool(s): {blocked_text}. "
                    "Do not add or delete slides in this round. Use `plan_slide_patch` / "
                    "`read_slide_snapshot` / `apply_slide_patch` for the requested edit, "
                    "or call `finalize` if the deck already satisfies the feedback."
                ),
            )
        )

    def _has_allowed_current_mutation(tool_calls: list[Any]) -> bool:
        return any(
            str(getattr(tc.function, "name", "") or "").strip()
            in {
                "apply_slide_patch",
                "patch_semantic_inline_style",
            }
            for tc in tool_calls
        )

    def _append_preference_update_finalize_guard() -> None:
        agent_to_use.chat_history.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    "SYSTEM: This turn only records a future/new-slide preference. "
                    "Do not modify existing slide files, insert slides, or call batch/patch/write tools. "
                    "If the preference has been recorded, call `finalize` now."
                ),
            )
        )

    def _read_only_batch_progress(new_results: list[tuple[Any, Any]]) -> dict[str, Any]:
        return classify_read_only_batch_progress(
            new_results,
            workspace=self.workspace,
            seen_keys=_read_only_seen_keys,
            resolve_modify_path=self._resolve_modify_coverage_path,
        )

    def _refresh_modify_plan_file_coverage() -> int:
        """Use on-disk slide changes as a fallback coverage signal."""
        if modify_plan is None or not modify_plan.coverage_required:
            return 0
        covered_now = 0
        for path in modify_plan.target_slide_paths:
            key = str(path.resolve())
            if key in _modify_plan_covered_keys:
                continue
            before_hash = _modify_plan_initial_hashes.get(key, "")
            after_hash = _file_sha256(path)
            if after_hash and after_hash != before_hash:
                _modify_plan_covered_keys.add(key)
                covered_now += 1
        if covered_now:
            info(
                "RevisionEditor file-state coverage progress: %s/%s",
                len(_modify_plan_covered_keys),
                len(_modify_plan_target_keys),
            )
        return covered_now

    try:
        modify_iter = 0

        while True:
            modify_iter += 1

            def _uncovered_plan_paths() -> list[Path]:
                if modify_plan is None or not modify_plan.coverage_required:
                    return []
                _refresh_modify_plan_file_coverage()
                return [
                    path
                    for path in modify_plan.target_slide_paths
                    if str(path.resolve()) not in _modify_plan_covered_keys
                ]

            async def _future_preference_failures() -> list[dict[str, Any]]:
                if not active_future_structured_preferences or not _inserted_slide_paths:
                    return []
                return await self._collect_future_slide_preference_failures_async(
                    list(_inserted_slide_paths.values()),
                    active_future_structured_preferences,
                )

            def _changed_existing_new_element_paths() -> list[Path]:
                changed: list[Path] = []
                for raw_path, before_hash in _new_element_initial_hashes.items():
                    slide_path = Path(raw_path)
                    after_hash = _file_sha256(slide_path)
                    if after_hash and before_hash and after_hash != before_hash:
                        changed.append(slide_path)
                return changed

            async def _new_element_preference_failures() -> list[dict[str, Any]]:
                if not new_element_rule_applications:
                    return []
                changed_paths = _changed_existing_new_element_paths()
                if not changed_paths:
                    return []
                return await collect_new_element_preference_failures_async(
                    self,
                    before_html_by_path=_new_element_before_html_by_path,
                    changed_slide_paths=changed_paths,
                    applications=new_element_rule_applications,
                    user_message=user_message,
                    llm=getattr(agent_to_use, "llm", None),
                )

            def _pending_inserted_slide_completions() -> list[dict[str, Any]]:
                if not _inserted_slide_paths:
                    return []
                return collect_pending_inserted_slides(
                    self,
                    list(_inserted_slide_paths.values()),
                )

            def _inserted_slide_completion_followup(pending: list[dict[str, Any]]) -> str:
                return build_inserted_slide_completion_followup(self, pending)

            if (
                modify_plan is not None
                and modify_plan.coverage_required
                and _modify_plan_idle_rounds >= 2
            ):
                uncovered_paths = _uncovered_plan_paths()
                if uncovered_paths:
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=self._build_modify_plan_followup(uncovered_paths),
                        )
                    )
                _modify_plan_idle_rounds = 0

            # ── Guard: consecutive UNCHANGED detection ──
            # If agent keeps inspecting without making changes, force finalize early
            if _consecutive_unchanged >= _UNCHANGED_THRESHOLD:
                pending_inserted_slides = _pending_inserted_slide_completions()
                if pending_inserted_slides:
                    warning(
                        "modify() detected repeated unchanged inspections, but %s inserted slide(s) are still unfinished; requesting full new-slide write instead of finalize.",
                        len(pending_inserted_slides),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=_inserted_slide_completion_followup(pending_inserted_slides),
                        )
                    )
                    _consecutive_unchanged = 0
                    _modify_plan_idle_rounds += 1
                    continue
                if _slide_delta_unmet():
                    warning(
                        "modify() detected repeated unchanged inspections, but expected slide delta is unmet (expected=%s current=%s); requesting structural operation instead of finalize.",
                        _expected_slide_delta,
                        _current_slide_delta(),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(role=Role.USER, content=_slide_delta_followup())
                    )
                    _consecutive_unchanged = 0
                    _modify_plan_idle_rounds += 1
                    continue
                warning(
                    f"modify() detected {_consecutive_unchanged} consecutive UNCHANGED inspects, "
                    "forcing finalization to prevent loop"
                )
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                            content=(
                                "SYSTEM: The slide has been inspected multiple times without changes. "
                                "The modification round appears complete. "
                                "Call `finalize` NOW to complete this work."
                            ),
                        )
                    )
                agent_message = await agent_to_use.action()
                if agent_message.text:
                    agent_responses.append(agent_message.text)
                yield agent_message
                if agent_message.tool_calls:
                    agent_message.tool_calls, blocked_names = _block_preference_update_mutations(
                        agent_message.tool_calls,
                        event="preference_update_mutation_blocked_unchanged_finalize",
                    )
                    if blocked_names:
                        _append_preference_update_finalize_guard()
                    agent_message.tool_calls, blocked_slide_tools = _block_unexpected_slide_count_mutations(
                        agent_message.tool_calls,
                        event="unexpected_slide_count_mutation_blocked_unchanged_finalize",
                    )
                    if blocked_slide_tools:
                        _append_unexpected_slide_count_guard(blocked_slide_tools)
                    agent_message.tool_calls, blocked_rewrite = _block_controlled_rewrite_scope_mutations(
                        agent_message.tool_calls,
                        event="controlled_rewrite_scope_guard_unchanged_finalize",
                    )
                    if blocked_rewrite:
                        _append_controlled_rewrite_scope_guard(blocked_rewrite)
                    if not agent_message.tool_calls:
                        break
                    await agent_to_use.execute(agent_message.tool_calls)
                break

            # ── Guard: max iteration limit ──
            if modify_iter > MAX_MODIFY_ITERATIONS:
                pending_inserted_slides = _pending_inserted_slide_completions()
                delta_unmet = _slide_delta_unmet()
                if pending_inserted_slides or delta_unmet:
                    warning(
                        "modify() exceeded max iterations (%s) with unfinished structural work; inserted_pending=%s expected_delta=%s current_delta=%s. Blocking forced finalize.",
                        MAX_MODIFY_ITERATIONS,
                        len(pending_inserted_slides),
                        _expected_slide_delta,
                        _current_slide_delta(),
                    )
                else:
                    warning(
                        f"modify() exceeded max iterations ({MAX_MODIFY_ITERATIONS}), "
                        "forcing finalization"
                    )
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            _inserted_slide_completion_followup(pending_inserted_slides)
                            + "\n\nSYSTEM: The iteration budget is exhausted, but `finalize` is blocked until the inserted slide is complete. "
                            "Call `write_new_slide_file` now if you can still make one corrective tool call."
                            if pending_inserted_slides
                            else _slide_delta_followup()
                            if delta_unmet
                            else FORCE_FINALIZE_MSG["text"]
                        ),
                    )
                )
                # One last chance to finalize
                agent_message = await agent_to_use.action()
                if agent_message.text:
                    agent_responses.append(agent_message.text)
                yield agent_message
                if agent_message.tool_calls:
                    if pending_inserted_slides or delta_unmet:
                        agent_message.tool_calls = [
                            tc for tc in agent_message.tool_calls
                            if tc.function.name != "finalize"
                        ]
                    agent_message.tool_calls, blocked_names = _block_preference_update_mutations(
                        agent_message.tool_calls,
                        event="preference_update_mutation_blocked_iteration_limit",
                    )
                    if blocked_names:
                        _append_preference_update_finalize_guard()
                    agent_message.tool_calls, blocked_slide_tools = _block_unexpected_slide_count_mutations(
                        agent_message.tool_calls,
                        event="unexpected_slide_count_mutation_blocked_iteration_limit",
                    )
                    if blocked_slide_tools:
                        _append_unexpected_slide_count_guard(blocked_slide_tools)
                    agent_message.tool_calls, blocked_rewrite = _block_controlled_rewrite_scope_mutations(
                        agent_message.tool_calls,
                        event="controlled_rewrite_scope_guard_iteration_limit",
                    )
                    if blocked_rewrite:
                        _append_controlled_rewrite_scope_guard(blocked_rewrite)
                    if not agent_message.tool_calls:
                        break
                    outcome = await agent_to_use.execute(
                        agent_message.tool_calls
                    )
                break

            agent_message = await agent_to_use.action()
            # Collect agent text response
            if agent_message.text:
                agent_responses.append(agent_message.text)
            yield agent_message

            if _preference_update_turn and agent_message.tool_calls:
                agent_message.tool_calls, blocked_names = _block_preference_update_mutations(
                    agent_message.tool_calls,
                    event="preference_update_mutation_blocked",
                )
                if blocked_names:
                    _append_preference_update_finalize_guard()
                    if not agent_message.tool_calls:
                        _redundant_read_only_rounds = 0
                        continue
            if _future_only_guard_turn and agent_message.tool_calls:
                agent_message.tool_calls, blocked_scope = _block_future_only_scope_mutations(
                    agent_message.tool_calls,
                    event="future_only_existing_scope_guard",
                )
                if blocked_scope:
                    _blocked_tool_batches += 1
                    if not _has_allowed_current_mutation(agent_message.tool_calls):
                        agent_message.tool_calls = [
                            tc
                            for tc in agent_message.tool_calls
                            if tc.function.name != "finalize"
                        ]
                    _append_future_only_scope_guard(blocked_scope)
                    if not agent_message.tool_calls:
                        _redundant_read_only_rounds = 0
                        continue
            if _active_recovery_request and agent_message.tool_calls:
                agent_message.tool_calls, blocked_recovery = _block_recovery_scope_mutations(
                    agent_message.tool_calls,
                    event="controlled_rewrite_scope_guard",
                )
                if blocked_recovery:
                    _blocked_tool_batches += 1
                    _append_recovery_scope_guard(blocked_recovery)
                    if not agent_message.tool_calls:
                        _redundant_read_only_rounds = 0
                        continue
            if _controlled_rewrite_turn and agent_message.tool_calls:
                agent_message.tool_calls, blocked_rewrite = _block_controlled_rewrite_scope_mutations(
                    agent_message.tool_calls,
                    event="controlled_rewrite_scope_guard",
                )
                if blocked_rewrite:
                    _blocked_tool_batches += 1
                    _append_controlled_rewrite_scope_guard(blocked_rewrite)
                    if not agent_message.tool_calls:
                        _redundant_read_only_rounds = 0
                        continue
            if agent_message.tool_calls:
                agent_message.tool_calls, blocked_slide_tools = _block_unexpected_slide_count_mutations(
                    agent_message.tool_calls,
                    event="unexpected_slide_count_mutation_blocked",
                )
                if blocked_slide_tools:
                    _blocked_tool_batches += 1
                    _append_unexpected_slide_count_guard(blocked_slide_tools)
                    if not agent_message.tool_calls:
                        _redundant_read_only_rounds = 0
                        continue

            if not agent_message.tool_calls:
                uncovered_paths = _uncovered_plan_paths()
                if uncovered_paths:
                    warning(
                        "RevisionEditor plan still has %s uncovered target slides; continuing instead of exiting.",
                        len(uncovered_paths),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=self._build_modify_plan_followup(uncovered_paths),
                        )
                    )
                    _modify_plan_idle_rounds += 1
                    continue
                future_failures = await _future_preference_failures()
                if future_failures:
                    warning(
                        "Inserted slide still violates %s active future preference(s); continuing.",
                        len(future_failures),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=self._build_future_preference_followup(future_failures),
                        )
                    )
                    _modify_plan_idle_rounds += 1
                    continue
                new_element_failures = await _new_element_preference_failures()
                if new_element_failures:
                    warning(
                        "New elements still violate %s applicable WM preference(s); continuing.",
                        len(new_element_failures),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=build_new_element_preference_followup(
                                self,
                                new_element_failures,
                            ),
                        )
                    )
                    _modify_plan_idle_rounds += 1
                    continue
                pending_inserted_slides = _pending_inserted_slide_completions()
                if pending_inserted_slides:
                    warning(
                        "RevisionEditor stopped without tools while %s inserted slide(s) are still unfinished; continuing.",
                        len(pending_inserted_slides),
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=_inserted_slide_completion_followup(pending_inserted_slides),
                        )
                    )
                    _modify_plan_idle_rounds += 1
                    continue
                info("DeckDesigner agent responded without tool calls, ending modify loop")
                break

            if _pending_finalize_gate_inspect:
                has_finalize = any(tc.function.name == "finalize" for tc in agent_message.tool_calls)
                has_inspect = any(tc.function.name == "inspect_slide" for tc in agent_message.tool_calls)
                if has_finalize and not has_inspect:
                    warning(
                        "Blocking repeated finalize while %s slide(s) are pending inspect.",
                        len(_pending_finalize_gate_inspect),
                    )
                    agent_message.tool_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name != "finalize"
                    ]
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=_build_finalize_pending_inspect_followup(_pending_finalize_gate_inspect),
                        )
                    )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            # ── Guard: batch limit for full-file slide writes in modify path ──
            # Prevent context explosion by limiting concurrent slide writes
            _MAX_WRITE_PER_TURN = 3
            _full_file_write_tool_names = {"write_html_file", "write_new_slide_file"}
            _write_calls = [
                tc for tc in agent_message.tool_calls
                if tc.function.name in _full_file_write_tool_names
            ]
            _other_calls = [
                tc for tc in agent_message.tool_calls
                if tc.function.name not in _full_file_write_tool_names
            ]
            _batch_inject_msg = None

            if len(_write_calls) > _MAX_WRITE_PER_TURN:
                _kept = _write_calls[:_MAX_WRITE_PER_TURN]
                _deferred = _write_calls[_MAX_WRITE_PER_TURN:]
                _deferred_names = []
                for _dtc in _deferred:
                    try:
                        _deferred_names.append(json.loads(_dtc.function.arguments).get("file_path", "?"))
                    except Exception:
                        _deferred_names.append("?")
                warning(
                    f"Batch limiting full-file slide writes: executing {len(_kept)}, "
                    f"deferring {len(_deferred)} ({', '.join(_deferred_names)})"
                )
                agent_message.tool_calls = _other_calls + _kept
                # Inject deferred notice so agent continues in next turn
                _batch_inject_msg = (
                    f"SYSTEM: Only {_MAX_WRITE_PER_TURN} full-file slide write calls executed per turn "
                    f"to avoid context overflow. {len(_deferred)} slides deferred: "
                    f"{', '.join(_deferred_names)}. "
                    f"Continue writing the remaining slides in subsequent turns."
                )

            if _expected_slide_delta > 0:
                current_slide_delta = len(self._resolve_all_slide_paths()) - _initial_slide_count
                if current_slide_delta >= _expected_slide_delta:
                    redundant_insert_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name == "insert_slide"
                    ]
                    if redundant_insert_calls:
                        warning(
                            "Blocking redundant insert_slide: expected slide delta %s already reached.",
                            _expected_slide_delta,
                        )
                        append_modify_tool_policy_trace(
                            self,
                            event="redundant_insert_blocked",
                            plan=modify_plan,
                            policy=tool_policy_plan,
                            allowed_tools=[
                                str(tool.get("function", {}).get("name", "") or "").strip()
                                for tool in getattr(agent_to_use, "tools", [])
                                if isinstance(tool, dict)
                                and str(tool.get("function", {}).get("name", "") or "").strip()
                            ],
                            user_message=user_message,
                            intent=intent if isinstance(intent, dict) else None,
                            note="Blocked insert_slide because expected slide delta was already reached.",
                            extra={
                                "expected_slide_delta": _expected_slide_delta,
                                "current_slide_delta": current_slide_delta,
                                "blocked_call_count": len(redundant_insert_calls),
                            },
                        )
                        agent_message.tool_calls = [
                            tc for tc in agent_message.tool_calls
                            if tc.function.name != "insert_slide"
                        ]
                        agent_to_use.chat_history.append(
                            ChatMessage(
                                role=Role.USER,
                                content=(
                                    "SYSTEM: The requested new-slide count has already been reached. "
                                    "Do not insert another slide. If the inserted slide still has issues, "
                                    "complete placeholder or underfilled new slides with `write_new_slide_file` first. "
                                    "Use `plan_slide_patch` / `read_slide_snapshot` / `apply_slide_patch` only for "
                                    "small repairs after the full new-slide HTML exists, then inspect and finalize."
                                ),
                            )
                        )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            insert_and_finalize_calls = [
                tc for tc in agent_message.tool_calls
                if tc.function.name in {"insert_slide", "finalize"}
            ]
            if (
                any(tc.function.name == "insert_slide" for tc in insert_and_finalize_calls)
                and any(tc.function.name == "finalize" for tc in insert_and_finalize_calls)
            ):
                warning("Blocking same-turn finalize after insert_slide; inserted slide must be completed and inspected first.")
                agent_message.tool_calls = [
                    tc for tc in agent_message.tool_calls
                    if tc.function.name != "finalize"
                ]
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            "SYSTEM: `insert_slide` changes the canonical deck structure and may create only a placeholder. "
                            "Do not finalize in the same tool batch. After insertion, if the tool reports "
                            "`PLACEHOLDER_CREATED`, use `write_new_slide_file` on the returned `write_target` to write "
                            "complete slide HTML, then inspect the new slide before finalizing."
                        ),
                    )
                )

            if modify_plan is not None and modify_plan.coverage_required:
                uncovered_paths = _uncovered_plan_paths()
                if uncovered_paths:
                    finalize_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name == "finalize"
                    ]
                    if finalize_calls:
                        warning(
                            "Blocking premature finalize: %s uncovered target slides remain.",
                            len(uncovered_paths),
                        )
                        agent_message.tool_calls = [
                            tc for tc in agent_message.tool_calls
                            if tc.function.name != "finalize"
                        ]
                        agent_to_use.chat_history.append(
                            ChatMessage(
                                role=Role.USER,
                                content=self._build_modify_plan_followup(uncovered_paths),
                            )
                        )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            future_failures = await _future_preference_failures()
            if future_failures:
                finalize_calls = [
                    tc for tc in agent_message.tool_calls
                    if tc.function.name == "finalize"
                ]
                if finalize_calls:
                    warning(
                        "Blocking finalize because inserted slides still violate %s future preference(s).",
                        len(future_failures),
                    )
                    agent_message.tool_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name != "finalize"
                    ]
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=self._build_future_preference_followup(future_failures),
                        )
                    )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            new_element_failures = await _new_element_preference_failures()
            if new_element_failures:
                finalize_calls = [
                    tc for tc in agent_message.tool_calls
                    if tc.function.name == "finalize"
                ]
                if finalize_calls:
                    warning(
                        "Blocking finalize because changed existing slide(s) have %s new-element WM preference violation(s).",
                        len(new_element_failures),
                    )
                    agent_message.tool_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name != "finalize"
                    ]
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=build_new_element_preference_followup(
                                self,
                                new_element_failures,
                            ),
                        )
                    )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            pending_inserted_slides = _pending_inserted_slide_completions()
            if pending_inserted_slides:
                finalize_calls = [
                    tc for tc in agent_message.tool_calls
                    if tc.function.name == "finalize"
                ]
                if finalize_calls:
                    warning(
                        "Blocking finalize because %s inserted slide(s) are still placeholders or underfilled.",
                        len(pending_inserted_slides),
                    )
                    agent_message.tool_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name != "finalize"
                    ]
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=_inserted_slide_completion_followup(pending_inserted_slides),
                        )
                    )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            if _expected_slide_delta != 0:
                finalize_calls = [
                    tc for tc in agent_message.tool_calls
                    if tc.function.name == "finalize"
                ]
                current_slide_delta = _current_slide_delta()
                if finalize_calls and current_slide_delta != _expected_slide_delta:
                    warning(
                        "Blocking finalize because expected slide delta is not satisfied: expected=%s current=%s.",
                        _expected_slide_delta,
                        current_slide_delta,
                    )
                    append_modify_tool_policy_trace(
                        self,
                        event="slide_delta_finalize_blocked",
                        plan=modify_plan,
                        policy=tool_policy_plan,
                        user_message=user_message,
                        intent=intent if isinstance(intent, dict) else None,
                        note="Finalize was blocked because the requested slide-count change had not happened.",
                        extra={
                            "expected_slide_delta": _expected_slide_delta,
                            "current_slide_delta": current_slide_delta,
                        },
                    )
                    agent_message.tool_calls = [
                        tc for tc in agent_message.tool_calls
                        if tc.function.name != "finalize"
                    ]
                    agent_to_use.chat_history.append(
                        ChatMessage(role=Role.USER, content=_slide_delta_followup())
                    )
                    if not agent_message.tool_calls:
                        _modify_plan_idle_rounds += 1
                        continue

            # ── Stage 5: ask_user_clarification 检测与处理 ──
            # 检测是否为澄清工具调用，如果是则挂起执行流，等待用户回答
            from memslides.tools.ask_clarification import (
                is_clarification_tool_call, parse_clarification_arguments,
                ClarificationRequest, format_tool_response, ClarificationResponse,
            )
            
            _clarification_calls = [tc for tc in agent_message.tool_calls if is_clarification_tool_call(tc)]
            _other_tool_calls = [tc for tc in agent_message.tool_calls if not is_clarification_tool_call(tc)]
            
            if _clarification_calls:
                # 处理澄清请求（仅取第一个，避免多个澄清同时进行）
                _clar_tc = _clarification_calls[0]
                _clar_args = parse_clarification_arguments(_clar_tc)
                _clar_request = ClarificationRequest.from_tool_call(
                    _clar_args, agent_type="modify", context=user_message[:200]
                )
                
                info(f"Agent requesting clarification: {_clar_request.question}")
                
                # yield-resume 模式：yield 后 generator 暂停，
                # WebUI 设置 _clarification_answer 后再 resume generator
                self._clarification_answer = None  # 清空，等待 WebUI 设置
                yield {
                    "type": "clarification_required",
                    "data": _clar_request.to_dict(),
                    "tool_call_id": _clar_tc.id,
                }
                # Generator 在此恢复时，WebUI 已设置 _clarification_answer
                _user_answer = self._clarification_answer or ""
                self._clarification_answer = None
                
                info(f"Clarification answered: {_user_answer[:100]}")
                
                # 解析用户回答，如果是选项选择则获取完整选项文本
                _selected_option_idx = None
                _full_answer = _user_answer
                _answer_stripped = _user_answer.strip().upper()
                
                # 检测选项选择：A/B/C/D 或 1/2/3/4 或 "A." 等格式
                if _clar_request.options:
                    _option_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
                    # 提取首字符作为选项标识
                    _first_char = _answer_stripped[0] if _answer_stripped else ""
                    if _first_char in _option_map:
                        _idx = _option_map[_first_char]
                        if _idx < len(_clar_request.options):
                            _selected_option_idx = _idx
                            # 使用完整的选项文本作为答案
                            _full_answer = _clar_request.options[_idx]
                            info(f"User selected option {_idx + 1}: {_full_answer[:100]}")
                
                # 构造工具响应
                _clar_response = ClarificationResponse(
                    request_id=_clar_request.id,
                    answer=_full_answer,  # 使用完整选项文本
                    selected_option=_selected_option_idx,
                )
                _tool_response_text = format_tool_response(_clar_response, _clar_request)
                
                # 即时记忆提取：从 QA 对话中提取偏好
                if self.memory_system and hasattr(self.memory_system, 'preference_extractor'):
                    try:
                        pref = await self._extract_qa_preference(
                            _clar_request.question, _clar_response.answer
                        )
                        if pref:
                            info(f"Extracted preference from QA: {pref.preference[:100]}")
                    except Exception as _e:
                        logger.warning(f"QA preference extraction failed (non-fatal): {_e}")
                
                # 将工具响应添加到历史
                agent_to_use.chat_history.append(ChatMessage(
                    role=Role.TOOL,
                    content=_tool_response_text,
                    tool_call_id=_clar_tc.id,
                ))
                
                # 注入系统级指令，禁止再次提问确认
                agent_to_use.chat_history.append(ChatMessage(
                    role=Role.USER,
                    content=(
                        "SYSTEM: 用户已明确回答，请立即执行对应操作。"
                        "禁止再次使用 ask_user_clarification 工具，禁止重复确认。"
                        "直接调用工具完成用户选择的方案。"
                    ),
                ))
                
                # 如果还有其他工具调用，继续执行
                if _other_tool_calls:
                    agent_message.tool_calls = _other_tool_calls
                else:
                    # 没有其他工具调用，继续下一轮循环让 Agent 基于用户回答做决策
                    continue
            
            # Record tool call details with results
            pre_tool_count = len(self.agent_env.tool_history)
            outcome = await agent_to_use.execute(agent_message.tool_calls)

            # Inject batch continuation notice after tool results
            if _batch_inject_msg and isinstance(outcome, list) and outcome:
                outcome[-1].content.insert(0, {"type": "text", "text": _batch_inject_msg})

            # Extract this round's tool results from tool_history
            new_results = self.agent_env.tool_history[pre_tool_count:]
            if _future_only_guard_turn and _all_existing_initial_hashes:
                changed_target_paths: list[Path] = []
                changed_outside_scope: list[Path] = []
                for raw_path, before_hash in _all_existing_initial_hashes.items():
                    slide_path = Path(raw_path)
                    after_hash = _file_sha256(slide_path)
                    if after_hash and before_hash and after_hash != before_hash:
                        if raw_path in _modify_plan_target_keys:
                            changed_target_paths.append(slide_path)
                            continue
                        changed_outside_scope.append(slide_path)
                for changed_target in changed_target_paths:
                    _all_existing_initial_hashes[str(changed_target.resolve())] = _file_sha256(changed_target)
                if changed_outside_scope:
                    warning(
                        "current-edit + future-only turn changed %s out-of-scope existing slide file(s); restoring checkpoints.",
                        len(changed_outside_scope),
                    )
                    restored_paths: list[str] = []
                    rm = getattr(mem, "rollback_manager", None) if mem is not None else None
                    for slide_path in changed_outside_scope:
                        restored = False
                        if rm is not None:
                            try:
                                restored = bool(await rm.rollback(slide_path.stem, slide_path))
                            except Exception as rollback_e:
                                logger.warning(
                                    "Future-only scope rollback failed for %s: %s",
                                    slide_path,
                                    rollback_e,
                                )
                        if not restored:
                            try:
                                checkpoint_path = self.workspace / ".rollback" / f"{slide_path.stem}.html"
                                if checkpoint_path.exists():
                                    slide_path.write_text(checkpoint_path.read_text(encoding="utf-8"), encoding="utf-8")
                                    restored = True
                            except Exception as fallback_e:
                                logger.warning(
                                    "Future-only scope fallback restore failed for %s: %s",
                                    slide_path,
                                    fallback_e,
                                )
                        if restored:
                            restored_paths.append(str(slide_path))
                            _all_existing_initial_hashes[str(slide_path.resolve())] = _file_sha256(slide_path)
                    append_modify_tool_policy_trace(
                        self,
                        event="future_only_hash_guard",
                        plan=modify_plan,
                        policy=tool_policy_plan,
                        user_message=user_message,
                        intent=intent if isinstance(intent, dict) else None,
                        note="Out-of-scope old slide files changed during a current-edit + future-only rule turn and were restored.",
                        extra={
                            "changed_outside_scope_paths": [str(path) for path in changed_outside_scope],
                            "restored_paths": restored_paths,
                        },
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                "SYSTEM: The previous tool result changed old slide files outside the explicit current target. "
                                "Those files have been restored. Continue only on the current target slide(s), then inspect and finalize."
                            ),
                        )
                    )
                    _redundant_read_only_rounds = 0
                    continue
            for call, result_msg in new_results:
                if call.function.name == "finalize" and not result_msg.is_error:
                    result_text = str(result_msg.text or "").strip()
                    pending_finalize_gate = _parse_finalize_pending_inspect_slides(result_text)
                    if (
                        result_text
                        and not pending_finalize_gate
                        and not result_text.startswith("Error")
                        and not result_text.startswith("Outcome ")
                    ):
                        _finalize_succeeded = True
                if call.function.name == "inspect_slide" and _pending_finalize_gate_inspect:
                    _pending_finalize_gate_inspect = []
                tool_calls_log.append({
                    "name": call.function.name,
                    "args": call.function.arguments,
                    "result_preview": (result_msg.text or "")[:500],
                    "is_error": result_msg.is_error,
                    "duration_ms": int(result_msg.extra_info.get("duration_ms", 0) or 0),
                })
            finalize_gate_recovered = False
            for call, result_msg in new_results:
                if call.function.name != "finalize":
                    continue
                pending = _parse_finalize_pending_inspect_slides(result_msg.text or "")
                if not pending:
                    continue
                _pending_finalize_gate_inspect = pending
                warning(
                    "RevisionEditor finalize gate returned %s pending inspect slide(s); continuing repair loop.",
                    len(pending),
                )
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                        content=_build_finalize_pending_inspect_followup(pending),
                    )
                )
                _modify_plan_idle_rounds += 1
                finalize_gate_recovered = True
                break
            if finalize_gate_recovered:
                continue

            recovery_request: dict[str, Any] | None = None
            for call, result_msg in new_results:
                parsed_recovery = parse_modify_recovery_request_from_tool_result(
                    tool_name=call.function.name,
                    arguments=call.function.arguments,
                    result_text=result_msg.text or "",
                    is_error=bool(result_msg.is_error),
                    runtime=self,
                )
                if parsed_recovery:
                    recovery_request = parsed_recovery
                    break
            if recovery_request:
                _active_recovery_request = recovery_request
                try:
                    restore_agent_base_tools(agent_to_use)
                    recovery_policy = build_controlled_rewrite_recovery_policy(
                        self,
                        tool_policy_plan,
                        recovery_request,
                    )
                    tool_policy_plan = recovery_policy
                    tool_scope_contract = apply_modify_tool_policy(
                        self,
                        agent_to_use,
                        modify_plan,
                        recovery_policy,
                        event="tool_requested_recovery_policy",
                        user_message=user_message,
                        intent=intent if isinstance(intent, dict) else None,
                        note=(
                            "A tool returned a structured recovery request; switching from the "
                            "current modify policy to controlled rewrite recovery."
                        ),
                        extra={"recovery_request": recovery_request},
                    )
                    tool_capability_validation = validate_modify_tool_capability_contract(
                        recovery_policy,
                        getattr(tool_scope_contract, "allowed_tools", []) or [],
                        plan=modify_plan,
                    )
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                build_controlled_rewrite_recovery_followup(
                                    self,
                                    recovery_request,
                                    recovery_policy,
                                )
                                + "\n\n"
                                + self._render_modify_tool_policy_plan(recovery_policy)
                                + "\n\n"
                                + render_current_tool_authority(
                                    tool_scope_contract,
                                    recovery_policy,
                                    tool_capability_validation,
                                )
                            ),
                        )
                    )
                    _redundant_read_only_rounds = 0
                    continue
                except Exception as recovery_e:
                    logger.warning("Controlled rewrite recovery policy switch failed (non-fatal): %s", recovery_e)

            newly_inserted_paths: list[Path] = []
            for call, result_msg in new_results:
                if call.function.name != "insert_slide" or result_msg.is_error:
                    continue
                inserted_path_text = self._extract_inserted_slide_path_from_result(result_msg.text or "")
                if not inserted_path_text:
                    continue
                inserted_path = self._resolve_modify_coverage_path(inserted_path_text)
                if inserted_path is None:
                    candidate = Path(inserted_path_text)
                    if not candidate.is_absolute():
                        candidate = self.workspace / candidate
                    if candidate.suffix.lower() == ".html":
                        inserted_path = candidate
                if inserted_path is None:
                    continue
                inserted_key = str(inserted_path)
                _inserted_slide_paths[inserted_key] = inserted_path
                newly_inserted_paths.append(inserted_path)

            if active_future_structured_preferences and newly_inserted_paths:
                inserted_failures = await self._collect_future_slide_preference_failures_async(
                    newly_inserted_paths,
                    active_future_structured_preferences,
                )
                if inserted_failures:
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=self._build_future_preference_followup(inserted_failures),
                        )
                    )

            new_element_failures_after_tools = await _new_element_preference_failures()
            if new_element_failures_after_tools:
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                        content=build_new_element_preference_followup(
                            self,
                            new_element_failures_after_tools,
                        ),
                    )
                )
                _modify_plan_idle_rounds += 1
                continue

            if newly_inserted_paths:
                pending_inserted_slides = collect_pending_inserted_slides(
                    self,
                    newly_inserted_paths,
                )
                if pending_inserted_slides:
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=_inserted_slide_completion_followup(pending_inserted_slides),
                        )
                    )

            if modify_plan is not None and modify_plan.coverage_required:
                covered_now = 0
                used_modify_tool = False
                for call, result_msg in new_results:
                    tool_name = call.function.name
                    if tool_name in {
                        "apply_slide_patch",
                        "write_html_file",
                        "write_new_slide_file",
                        "batch_update_css_rule",
                        "batch_update_semantic_style",
                        "patch_semantic_inline_style",
                    } and not result_msg.is_error:
                        used_modify_tool = True
                    if result_msg.is_error:
                        continue
                    for covered_path in self._extract_modify_coverage_paths(
                        tool_name=tool_name,
                        arguments=call.function.arguments,
                        result_text=result_msg.text or "",
                    ):
                        key = str(covered_path.resolve())
                        if key in _modify_plan_target_keys and key not in _modify_plan_covered_keys:
                            _modify_plan_covered_keys.add(key)
                            covered_now += 1
                covered_now += _refresh_modify_plan_file_coverage()
                if covered_now > 0:
                    info(
                        "RevisionEditor execution plan coverage progress: %s/%s",
                        len(_modify_plan_covered_keys),
                        len(_modify_plan_target_keys),
                    )
                    _modify_plan_idle_rounds = 0
                elif _uncovered_plan_paths():
                    _modify_plan_idle_rounds = _modify_plan_idle_rounds + 1 if not used_modify_tool else 0
                else:
                    _modify_plan_idle_rounds = 0

            # ── Guard: detect repeated inspection and redundant read-only loops ──
            _has_inspect = False
            _has_unchanged = False
            _has_change_tool = False
            for call, result_msg in new_results:
                if call.function.name == "inspect_slide":
                    _has_inspect = True
                    _result_text = result_msg.text or ""
                    if "UNCHANGED" in _result_text:
                        _has_unchanged = True
                    elif "CHANGED" in _result_text:
                        _consecutive_unchanged = 0
                elif call.function.name in MUTATION_TOOLS:
                    _has_change_tool = True
                    _consecutive_unchanged = 0

            if _has_inspect and _has_unchanged and not _has_change_tool:
                _consecutive_unchanged += 1
            elif _has_change_tool:
                _consecutive_unchanged = 0
            if _preference_update_turn and _has_change_tool:
                warning("Mutation detected after execution in preference_update turn; forcing follow-up guard.")
                agent_to_use.chat_history.append(
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            "SYSTEM: A mutation tool ran during a memory-only preference update. "
                            "Do not make further slide changes. Call `finalize`; the runtime will restore any changed existing slide files."
                        ),
                    )
                )
                _redundant_read_only_rounds = 0
                continue
            if _has_change_tool:
                _redundant_read_only_rounds = 0
            else:
                read_only_progress = _read_only_batch_progress(new_results)
                if read_only_progress.get("has_read_only"):
                    _last_read_only_progress = read_only_progress
                    if read_only_progress.get("has_progress"):
                        _redundant_read_only_rounds = 0
                    else:
                        _redundant_read_only_rounds += 1
                if read_only_progress.get("has_read_only") and _redundant_read_only_rounds >= 2:
                    if _preference_update_turn:
                        warning(
                            "preference_update turn had %s redundant read-only rounds; requesting finalize without replanning.",
                            _redundant_read_only_rounds,
                        )
                        agent_to_use.chat_history.append(
                            ChatMessage(
                                role=Role.USER,
                                content=self._build_modify_no_mutation_followup(modify_plan),
                            )
                        )
                        _redundant_read_only_rounds = 0
                        continue
                    warning(
                        "modify() detected %s redundant read-only tool rounds without new information (%s); replanning tool policy.",
                        _redundant_read_only_rounds,
                        read_only_progress.get("reason") or "unknown",
                    )
                    append_modify_tool_policy_trace(
                        self,
                        event="redundant_read_only_detected",
                        plan=modify_plan,
                        policy=tool_policy_plan,
                        user_message=user_message,
                        intent=intent if isinstance(intent, dict) else None,
                        note="Consecutive read-only tool calls repeated the same resource/hash/scope without adding new information.",
                        extra={
                            "redundant_read_only_rounds": _redundant_read_only_rounds,
                            "read_only_progress_status": read_only_progress.get("status"),
                            "duplicate_reason": read_only_progress.get("reason"),
                            "last_entry": read_only_progress.get("last_entry", {}),
                            "entries": read_only_progress.get("entries", []),
                            "suggested_next_action": "Use a new information scope, an asset/search tool if needed, or the appropriate mutation tool.",
                        },
                    )
                    try:
                        restore_agent_base_tools(agent_to_use)
                        replanned_policy = await build_modify_tool_policy_plan(
                            self,
                            user_message=(
                                user_message
                                + "\n\nRuntime note: recent turns repeated the same read-only resource/hash/scope and added no new information. "
                                "Replan with either a genuinely new information/asset-discovery step or the appropriate edit tool."
                            ),
                            intent=intent if isinstance(intent, dict) else None,
                            execution_plan=modify_plan,
                            wm_rule_specs_text=wm_rule_specs_text,
                            agent=agent_to_use,
                        )
                        tool_policy_plan = replanned_policy
                        apply_modify_tool_policy(
                            self,
                            agent_to_use,
                            modify_plan,
                            replanned_policy,
                            event="replan_no_mutation",
                            user_message=user_message,
                            intent=intent if isinstance(intent, dict) else None,
                            note="Consecutive redundant read-only rounds added no new information.",
                            extra={
                                "redundant_read_only_rounds": _redundant_read_only_rounds,
                                "read_only_progress_status": read_only_progress.get("status"),
                                "duplicate_reason": read_only_progress.get("reason"),
                                "last_entry": read_only_progress.get("last_entry", {}),
                            },
                        )
                    except Exception as e:
                        logger.warning("Modify tool policy replan failed (non-fatal): %s", e)
                    agent_to_use.chat_history.append(
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                self._build_modify_no_mutation_followup(modify_plan)
                                + (
                                    "\n\n"
                                    + self._render_modify_tool_policy_plan(tool_policy_plan)
                                    if tool_policy_plan is not None
                                    else ""
                                )
                            ),
                        )
                    )
                    _redundant_read_only_rounds = 0

            # ── Stage 2: Feed tool calls to Collector ──
            if (not _orchestrator and self.memory_system
                    and hasattr(self.memory_system, 'collector')
                    and self.memory_system.collector):
                for call, result_msg in new_results:
                    try:
                        self.memory_system.collector.add_tool_call(
                            name=call.function.name,
                            args=call.function.arguments,
                            result=(result_msg.text or "")[:2000],
                            is_error=result_msg.is_error,
                            duration_ms=int(result_msg.extra_info.get("duration_ms", 0) or 0),
                        )
                    except Exception as _e:
                        logger.warning(f"MemoryCollector add_tool_call failed (non-fatal): {_e}")

            if isinstance(outcome, list):
                for item in outcome:
                    yield item
            else:
                break

        if _preference_update_turn and _preference_update_initial_hashes:
            changed_existing_paths: list[Path] = []
            for raw_path, before_hash in _preference_update_initial_hashes.items():
                slide_path = Path(raw_path)
                after_hash = _file_sha256(slide_path)
                if after_hash and before_hash and after_hash != before_hash:
                    changed_existing_paths.append(slide_path)
            if changed_existing_paths:
                warning(
                    "preference_update turn changed %s existing slide file(s); restoring checkpoints.",
                    len(changed_existing_paths),
                )
                restored_paths: list[str] = []
                rm = getattr(mem, "rollback_manager", None) if mem is not None else None
                for slide_path in changed_existing_paths:
                    restored = False
                    if rm is not None:
                        try:
                            restored = bool(await rm.rollback(slide_path.stem, slide_path))
                        except Exception as rollback_e:
                            logger.warning(
                                "Preference-update rollback failed for %s: %s",
                                slide_path,
                                rollback_e,
                            )
                    if not restored:
                        try:
                            checkpoint_path = self.workspace / ".rollback" / f"{slide_path.stem}.html"
                            if checkpoint_path.exists():
                                slide_path.write_text(checkpoint_path.read_text(encoding="utf-8"), encoding="utf-8")
                                restored = True
                        except Exception as fallback_e:
                            logger.warning(
                                "Preference-update fallback restore failed for %s: %s",
                                slide_path,
                                fallback_e,
                            )
                    if restored:
                        restored_paths.append(str(slide_path))
                append_modify_tool_policy_trace(
                    self,
                    event="preference_update_hash_guard",
                    plan=modify_plan,
                    policy=tool_policy_plan,
                    user_message=user_message,
                    intent=intent if isinstance(intent, dict) else None,
                    note="Existing slide files changed during preference_update and were restored.",
                    extra={
                        "changed_existing_paths": [str(path) for path in changed_existing_paths],
                        "restored_paths": restored_paths,
                    },
                )

        if debug_tracer:
            debug_tracer.log_step(turn, "agent_tools", tool_calls_log)

    except Exception as e:
        error_msg = f"DeckDesigner agent modification failed: {e}\n{traceback.format_exc()}"
        error(error_msg)
        yield ChatMessage(role=Role.SYSTEM, content=error_msg)
        return
    finally:
        self.designagent.save_history()
        if self.modifyagent is not None and self.modifyagent is not self.designagent:
            self.modifyagent.save_history()
        self.save_results()

    if not _finalize_succeeded:
        finalized_in_history = False
        for call, result_msg in getattr(self.agent_env, "tool_history", [])[_initial_tool_history_len:]:
            if getattr(getattr(call, "function", None), "name", "") != "finalize":
                continue
            if getattr(result_msg, "is_error", False):
                continue
            result_text = str(getattr(result_msg, "text", "") or "").strip()
            if result_text and not result_text.startswith("Error") and not result_text.startswith("Outcome "):
                finalized_in_history = True
                break
        _finalize_succeeded = finalized_in_history

    final_slide_delta = _current_slide_delta()
    if _finalize_succeeded and final_slide_delta != _expected_slide_delta:
        warning(
            "RevisionEditor finalize was called, but slide-count validation failed: expected=%s current=%s. Skipping export.",
            _expected_slide_delta,
            final_slide_delta,
        )
        append_modify_tool_policy_trace(
            self,
            event="slide_delta_validation_failed",
            plan=modify_plan,
            policy=tool_policy_plan,
            user_message=user_message,
            intent=intent if isinstance(intent, dict) else None,
            note="Final slide-count validation failed after finalize; export skipped.",
            extra={
                "expected_slide_delta": _expected_slide_delta,
                "final_slide_delta": final_slide_delta,
                "initial_slide_count": _initial_slide_count,
                "final_slide_count": len(self._resolve_all_slide_paths()),
            },
        )
        yield ChatMessage(
            role=Role.SYSTEM,
            content=(
                "Modification did not complete because the requested slide-count change was not applied "
                f"(expected {_expected_slide_delta:+d}, got {final_slide_delta:+d}). "
                "Export was skipped so the deck is not marked as successfully revised."
            ),
        )
        return

    if _finalize_succeeded and modify_plan is not None and getattr(modify_plan, "operation_kind", "") == "diagram_layout":
        try:
            from memslides.tools.deck_runtime import validate_diagram_layout_static
        except Exception:
            validate_diagram_layout_static = None
        diagram_contract = getattr(modify_plan, "diagram_contract", None) or {}
        diagram_failures: list[dict[str, Any]] = []
        if validate_diagram_layout_static is not None and diagram_contract:
            for slide_path in getattr(modify_plan, "target_slide_paths", []) or []:
                try:
                    html_text = Path(slide_path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    html_text = ""
                for diagnostic in validate_diagram_layout_static(html_text, diagram_contract):
                    if diagnostic.get("severity") == "error":
                        diagram_failures.append(
                            {
                                "slide_path": str(slide_path),
                                "code": diagnostic.get("code", ""),
                                "message": diagnostic.get("message", ""),
                            }
                        )
        if diagram_failures:
            warning(
                "RevisionEditor finalize was called, but diagram layout validation still has %s failure(s); skipping export.",
                len(diagram_failures),
            )
            append_modify_tool_policy_trace(
                self,
                event="diagram_layout_validation_failed",
                plan=modify_plan,
                policy=tool_policy_plan,
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                note="Final diagram layout validation failed after finalize; export skipped.",
                extra={"diagram_failures": diagram_failures[:12]},
            )
            yield ChatMessage(
                role=Role.SYSTEM,
                content=(
                    "Modification did not finalize successfully because the target slide still fails "
                    "the flowchart/pipeline diagram contract. Export was skipped.\n"
                    + "\n".join(
                        f"- {item['slide_path']}: {item['code']} - {item['message']}"
                        for item in diagram_failures[:6]
                    )
                ),
            )
            return

    if _finalize_succeeded and new_element_rule_applications:
        changed_paths = []
        for raw_path, before_hash in _new_element_initial_hashes.items():
            slide_path = Path(raw_path)
            after_hash = _file_sha256(slide_path)
            if after_hash and before_hash and after_hash != before_hash:
                changed_paths.append(slide_path)
        final_new_element_failures = []
        if changed_paths:
            final_new_element_failures = await collect_new_element_preference_failures_async(
                self,
                before_html_by_path=_new_element_before_html_by_path,
                changed_slide_paths=changed_paths,
                applications=new_element_rule_applications,
                user_message=user_message,
                llm=getattr(agent_to_use, "llm", None),
            )
        if final_new_element_failures:
            warning(
                "RevisionEditor finalize was called, but new elements still violate %s applicable WM preference(s); skipping export.",
                len(final_new_element_failures),
            )
            yield ChatMessage(
                role=Role.SYSTEM,
                content=build_new_element_preference_followup(
                    self,
                    final_new_element_failures,
                    include_header=True,
                )
                + "\n\nModification did not finalize successfully, so export was skipped.",
            )
            return

    if _finalize_succeeded and active_structured_preferences:
        hard_repairs = enforce_hard_title_prefix_preferences(
            self,
            self._resolve_all_slide_paths(),
            active_structured_preferences,
            user_message=user_message,
        )
        if hard_repairs:
            info(
                "Applied hard title-prefix preference repairs before export: %s",
                len(hard_repairs),
            )
            append_memory_flow_trace(
                self,
                event="hard_preference_repairs_applied",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                plan=modify_plan,
                extra={"repairs": hard_repairs[:20]},
            )

        soft_preference_evaluations = collect_soft_preference_evaluations(
            self,
            self._resolve_all_slide_paths(),
            active_structured_preferences,
        )
        if soft_preference_evaluations:
            append_memory_flow_trace(
                self,
                event="soft_preference_evaluation",
                user_message=user_message,
                intent=intent if isinstance(intent, dict) else None,
                plan=modify_plan,
                extra={"evaluations": soft_preference_evaluations[:20]},
            )

    if not _finalize_succeeded:
        warning("RevisionEditor modify round ended without a successful finalize; skipping export.")
        yield ChatMessage(
            role=Role.SYSTEM,
            content=(
                "Modification did not finalize successfully, so export was skipped. "
                "Please inspect the latest tool results and retry the modify round."
            ),
        )
        return

    # ── Convert modified HTML → PDF ──
    slide_html_dir = self.intermediate_output.get("slide_html_dir")
    if slide_html_dir:
        slide_html_dir = Path(slide_html_dir)
        try:
            request = self._last_request
            aspect_ratio = request.powerpoint_type if request else "16:9"

            pptx_path = self.workspace / f"modification_{turn}.pptx"
            slide_html_dir, export_html_files = await self._export_slides_with_agent_repair(
                slide_html_dir,
                pptx_path,
                aspect_ratio=aspect_ratio,
                context_label=f"modification_{turn}",
            )
            await self._export_pdf_best_effort(
                export_html_files,
                pptx_path.with_suffix(".pdf"),
                aspect_ratio=aspect_ratio,
                context_label=f"modification_{turn}",
            )

            final_artifact = pptx_path if pptx_path.exists() else slide_html_dir
            self.intermediate_output["final"] = str(final_artifact)
            self.save_results()
            info(f"Modification {turn} complete, output at: {final_artifact}")
            yield final_artifact
        except Exception as e:
            error(f"Modified slide export failed: {e}")
            yield ChatMessage(
                role=Role.SYSTEM,
                content=f"Slides modified but export failed: {e}",
            )

    # G2 Phase 1 Step 1.2: Extract after_params and compute diff via StateCoordinator
    after_params = None
    params_diff = None
    if mem is not None and before_params and target_slide:
        try:
            if target_slide.lower() == "all" and isinstance(before_params, dict):
                # Batch after_modification for all slides
                coordinator = getattr(mem, "state_coordinator", None)
                all_paths = self._resolve_all_slide_paths()
                if all_paths and coordinator is not None:
                    after_params = {}
                    params_diff = {}
                    for sp in all_paths:
                        try:
                            ap, diff = await coordinator.after_modification(
                                sp, slide_type="html",
                                session_id=self.workspace.stem,
                                modification_id=f"modify_{turn}",
                            )
                            if ap:
                                after_params[sp.stem] = ap
                            if diff:
                                params_diff[sp.stem] = diff
                        except Exception:
                            pass
                    if debug_tracer:
                        debug_tracer.log_step(turn, "params_after", {
                            sid: ap.to_dict() if hasattr(ap, 'to_dict') else {}
                            for sid, ap in after_params.items()
                        })
                        debug_tracer.log_step(turn, "params_diff", params_diff or {})
                    total_changes = sum(len(d) for d in params_diff.values())
                    info(f"Batch after_modification: {len(after_params)}/{len(all_paths)} slides, {total_changes} total diff changes")
            else:
                slide_html_path = self._resolve_slide_path(target_slide)
                if slide_html_path:
                    coordinator = getattr(mem, "state_coordinator", None)
                    if coordinator is not None:
                        after_params, params_diff = await coordinator.after_modification(
                            slide_html_path,
                            slide_type="html",
                            session_id=self.workspace.stem,
                            modification_id=f"modify_{turn}",
                        )
                        if debug_tracer:
                            debug_tracer.log_step(turn, "params_after", after_params.to_dict() if hasattr(after_params, 'to_dict') else {})
                            debug_tracer.log_step(turn, "params_diff", params_diff or {})
                        info(f"StateCoordinator: after_modification for {target_slide}, diff={len(params_diff) if params_diff else 0} changes")
                    elif hasattr(mem, "state_extractor") and mem.state_extractor:
                        after_params = mem.state_extractor.extract_from_html(slide_html_path)
                        if debug_tracer:
                            debug_tracer.log_step(turn, "params_after", after_params.to_dict() if hasattr(after_params, 'to_dict') and after_params else {})
                        if after_params and hasattr(after_params, 'diff'):
                            params_diff = after_params.diff(before_params)
                            if debug_tracer:
                                debug_tracer.log_step(turn, "params_diff", params_diff)
                            info(f"Computed params diff for {target_slide}: {len(params_diff) if params_diff else 0} changes")
        except Exception as e:
            logger.warning(f"Param extraction (after) or diff failed (non-fatal): {e}")

    # G10: Populate InteractiveSession context_buffer for EditSegment tracking
    full_response = "\n".join(agent_responses)

    # ── Stage 2: Cognitive Memory Collector — end round ──
    _compact_round = None
    if self.memory_system and hasattr(self.memory_system, 'collector') and self.memory_system.collector:
        try:
            self.memory_system.collector.add_agent_response(full_response)
            _compact_round = self.memory_system.collector.end_round()
            # Artifact #1: Collected round trace
            if debug_tracer and _compact_round:
                debug_tracer.log_collected_round(turn, _compact_round)
        except Exception as e:
            logger.warning(f"MemoryCollector end_round failed (non-fatal): {e}")

    # Session context update (Pipeline removed - inline path only)
    if self._session is not None:
        try:
            from memslides.memory.core.models import Message
            self._session.add_message(Message(role="user", content=user_message))
            if full_response:
                self._session.add_message(Message(role="assistant", content=full_response[:2000]))
            # BoundaryDetect: update focus context
            if hasattr(self._session, 'update_focus'):
                self._session.update_focus(intent or {})
        except Exception as e:
            logger.warning(f"Session context update failed (non-fatal): {e}")

    # ── Memory Post-Steps [after action] ──
    if mem is not None:
        _post_args = (
            mem, user_message, full_response, tool_calls_log,
            intent, debug_tracer, turn,
            before_params, after_params, target_slide,
            _compact_round,  # Stage 4: 传递 AgentRound 用于统一输入
        )
        # AsyncRuleExtractor scheduling — DEPRECATED (Stage 2.5)
        # Run _post_modify_memory and await completion to ensure memory is persisted.
        # Previously used asyncio.create_task() which could be cancelled on process exit.
        try:
            post_timeout = float(os.environ.get("MEMSLIDES_POST_MODIFY_MEMORY_TIMEOUT_SEC", "20") or "20")
            await asyncio.wait_for(self._post_modify_memory(*_post_args), timeout=max(1.0, post_timeout))
        except asyncio.TimeoutError:
            logger.warning("_post_modify_memory timed out after %.1fs (non-fatal)", post_timeout)
        except Exception as e:
            logger.warning(f"_post_modify_memory failed (non-fatal): {e}")

    # MemoryOrchestrator: Round End for this modify turn
    # Pass _compact_round so orchestrator skips collector.end_round() (already called above)
    if _orchestrator:
        try:
            round_end_timeout = float(os.environ.get("MEMSLIDES_ORCHESTRATOR_ROUND_END_TIMEOUT_SEC", "30") or "30")
            await asyncio.wait_for(
                _orchestrator.on_round_end(
                    agent_response=full_response[:2000] if full_response else "",
                    compact_round=_compact_round,
                ),
                timeout=max(1.0, round_end_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Orchestrator on_round_end(modify round %s) timed out after %.1fs (non-fatal)",
                turn,
                round_end_timeout,
            )
        except Exception as e:
            logger.warning(f"Orchestrator on_round_end(modify round {turn}) failed: {e}")

    if debug_tracer:
        debug_tracer.log_event("modify_complete", {"turn": turn})
        debug_tracer.update_dashboard("last_modify", {
            "turn": turn,
            "message": user_message[:100],
            "tools_used": [t["name"] for t in tool_calls_log] if tool_calls_log else [],
        })



class RevisionPipeline:
    """Multi-turn deck revision pipeline."""

    def __init__(self, runtime):
        self.runtime = runtime

    async def run(self, request: RevisionRequest) -> DeckResult:
        messages: list[str] = []
        async for item in self.stream(request):
            if isinstance(item, ChatMessage):
                if item.text:
                    messages.append(item.text)
            else:
                messages.append(str(item))

        from memslides.pipelines.generation import GenerationPipeline

        return GenerationPipeline(self.runtime)._build_result(messages)

    async def stream(
        self,
        request: RevisionRequest,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        async for item in run_revision_flow(
            self.runtime,
            request.feedback,
            memory_intent=request.memory_intent,
            request_extra_info=request.extra_info,
        ):
            yield item
