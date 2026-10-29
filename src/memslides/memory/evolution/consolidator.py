"""Implementation details."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.models import (
    ChainExperience,
    TempPreference,
    UserProfile,
    DEFAULT_INTENT,
    is_template_related_preference,
)
from ..prompts.chain_prompts import (
    PROFILE_CONSOLIDATION_RAW_PROMPT,
    PROFILE_UPDATE_PROMPT,
)

if TYPE_CHECKING:
    from ..core.models import Job
    from ..store.chain_store import ChainStore
    from ..working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class Consolidator:
    """Implementation details."""

    def __init__(
        self,
        llm: Any = None,
        vlm: Any = None,
        experience_writer: Any = None,
        preference_store: Any = None,
        profile_store: Any = None,
        episode_store: Any = None,
        offline_consolidator: Any = None,
        workspace: Path | None = None,
        embedding_func: Any = None,
        chain_store: ChainStore | None = None,
        artifact_dumper: Any = None,
        enable_atomic_preference_writeback: bool = True,
        enable_profile_writeback: bool = True,
        enable_chain_experience_writeback: bool = True,
    ):
        self._llm = llm
        self._vlm = vlm
        self._exp_writer = experience_writer
        self._pref_store = preference_store
        self._profile_store = profile_store
        self._episode_store = episode_store
        self._offline = offline_consolidator
        self._workspace = workspace or Path(".")
        self._embedding_func = embedding_func
        self._chain_store = chain_store
        self._artifact_dumper = artifact_dumper
        self._enable_atomic_preference_writeback = enable_atomic_preference_writeback
        self._enable_profile_writeback = enable_profile_writeback
        self._enable_chain_experience_writeback = enable_chain_experience_writeback

        self._dim_matcher = None

    async def consolidate(
        self, job: Job, working_memory: WorkingMemory,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
    ) -> dict:
        """Implementation details."""
        from ..progress import emit_memory_save_progress

        stats = {"tool_written": 0, "pref_written": 0, "episode_written": 0, "deduplicated": 0}
        chain_buffer = working_memory.chain_buffer
        new_chains_preview = chain_buffer.get_new_chains_for_consolidation()
        total_chain_groups = len(new_chains_preview)
        total_chain_segments = sum(len(segments) for segments in new_chains_preview.values())


        # Orchestrator may already have written it before slower round/tool
        # writeback; in that case freeze_preference_writeback avoids duplicates.
        if freeze_preference_writeback:
            logger.info(
                "Consolidator: preference/profile writeback already handled or frozen for job %s",
                job.id,
            )
            stats["pref_written"] = 0
        else:
            stats["pref_written"] = await self.consolidate_preferences_only(
                job,
                working_memory,
                slide_html_dir=slide_html_dir,
                freeze_preference_writeback=False,
                progress_callback=progress_callback,
                progress_start=8,
                progress_end=30,
            )


        if not self._enable_chain_experience_writeback:
            logger.info(
                "Consolidator: chain experience writeback disabled, "
                "skipping chain/tool archival for job %s",
                job.id,
            )
            await emit_memory_save_progress(
                progress_callback,
                stage="tool_chain_writeback",
                progress=74,
                message="Tool-chain writeback is disabled.",
            )
        elif self._chain_store:
            new_chains = new_chains_preview
            if new_chains:
                await emit_memory_save_progress(
                    progress_callback,
                    stage="tool_chain_writeback",
                    progress=56,
                    message=(
                        f"Distilling {total_chain_segments} tool-chain segments "
                        f"across {total_chain_groups} pipelines."
                    ),
                    total_chain_groups=total_chain_groups,
                    total_chain_segments=total_chain_segments,
                )
            completed_chain_groups = 0
            for sig, segments in new_chains.items():
                try:
                    await emit_memory_save_progress(
                        progress_callback,
                        stage="tool_chain_writeback",
                        progress=56 + int(18 * completed_chain_groups / max(1, total_chain_groups)),
                        message=(
                            f"Writing tool-chain memory {completed_chain_groups + 1}/"
                            f"{total_chain_groups}: {sig[:72]}"
                        ),
                        chain_signature=sig,
                        completed_chain_groups=completed_chain_groups,
                        total_chain_groups=total_chain_groups,
                    )

                    existing_exps = await self._chain_store.get_experience(sig)
                    experiences_before = len(existing_exps)
                    new_experiences_list: list = []
                    if self._llm and segments:
                        from ..extract.chain_experience_extractor import ChainExperienceExtractor
                        extractor = ChainExperienceExtractor(llm=self._llm)

                        for segment in segments:
                            try:
                                updated_exps = await extractor.distill_and_merge(
                                    sig, segment, existing_exps,
                                )
                                for exp in updated_exps:

                                    if self._embedding_func and exp.keywords:
                                        try:
                                            kw_text = " ".join(exp.keywords)
                                            emb = await self._embedding_func([kw_text])
                                            if hasattr(emb, '__getitem__') and len(emb) > 0:
                                                vec = emb[0]
                                                exp.keyword_embedding = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                                        except Exception as emb_e:
                                            logger.warning(f"keyword_embedding generation failed for {sig}: {emb_e}")
                                    await self._chain_store.save_experience(
                                        sig, exp, job.user_id, job.id,
                                    )
                                    stats["tool_written"] += 1
                                    new_experiences_list.append(exp)

                                existing_exps = await self._chain_store.get_experience(sig)
                            except Exception as seg_e:
                                logger.warning(f"Segment distill failed for {sig}: {seg_e}")
                                continue

                    await self._chain_store.save_raw_chains(
                        sig, segments, job.user_id, job.id,
                    )

                    if self._artifact_dumper:
                        try:
                            final_exps = await self._chain_store.get_experience(sig)
                            from ..collect.artifact_dumper import ArtifactDumper
                            self._artifact_dumper.dump_chain_merge(
                                sig,
                                raw_chains_count=len(segments),
                                experiences_before=experiences_before,
                                experiences_after=len(final_exps),
                                new_experiences=[
                                    ArtifactDumper._safe_serialize(e)
                                    for e in new_experiences_list
                                ],
                            )
                        except Exception as e:
                            logger.warning(f"ArtifactDumper dump_chain_merge failed (non-fatal): {e}")
                except Exception as e:
                    logger.warning(f"Chain consolidation failed for {sig}: {e}")
                finally:
                    completed_chain_groups += 1

            await emit_memory_save_progress(
                progress_callback,
                stage="tool_chain_writeback",
                progress=74,
                message="Tool-chain writeback completed.",
                completed_chain_groups=completed_chain_groups,
                total_chain_groups=total_chain_groups,
            )


        elif self._exp_writer:
            await emit_memory_save_progress(
                progress_callback,
                stage="tool_chain_writeback",
                progress=56,
                message="Writing legacy tool-chain experiences.",
            )
            for chain_name, exps in chain_buffer.get_all_experiences().items():
                for exp in exps:
                    try:
                        trace = await self._write_chain_experience(chain_name, exp, job.id)
                        if trace:
                            stats["tool_written"] += 1
                    except Exception as e:
                        logger.warning(f"Chain experience archival failed for {chain_name}: {e}")

            for chain_name, chains in chain_buffer.get_all_chains().items():
                if len(chains) >= 2:
                    try:
                        count = await self._extract_and_archive_chains(chain_name, chains, job)
                        stats["tool_written"] += count
                    except Exception as e:
                        logger.warning(f"Remaining chain archival failed for {chain_name}: {e}")


        if self._episode_store:
            episodes = working_memory.get_episodes()
            await emit_memory_save_progress(
                progress_callback,
                stage="episode_writeback",
                progress=80,
                message=f"Archiving {len(episodes)} design episodes.",
                total_episodes=len(episodes),
            )
            for episode in episodes:
                try:
                    await self._episode_store.add_episode(episode)
                    stats["episode_written"] += 1
                except Exception as e:
                    logger.warning(f"Episode archival failed: {e}")
        await emit_memory_save_progress(
            progress_callback,
            stage="episode_writeback",
            progress=84,
            message="Episode archival completed.",
            episode_written=stats["episode_written"],
        )


        if self._offline:
            try:
                await emit_memory_save_progress(
                    progress_callback,
                    stage="deduplication",
                    progress=88,
                    message="Running post-consolidation deduplication.",
                )
                dedup_stats = await self._offline.consolidate(user_id=job.user_id)
                stats["deduplicated"] = getattr(dedup_stats, "exact_duplicates_removed", 0)
            except Exception as e:
                logger.warning(f"Post-consolidation dedup failed: {e}")
        await emit_memory_save_progress(
            progress_callback,
            stage="deduplication",
            progress=92,
            message="Post-consolidation deduplication completed.",
            deduplicated=stats["deduplicated"],
        )


        if self._artifact_dumper:
            try:
                await emit_memory_save_progress(
                    progress_callback,
                    stage="artifacts",
                    progress=96,
                    message="Writing consolidation trace artifacts.",
                )

                episodes_data = []
                for ep in working_memory.get_episodes():
                    if hasattr(ep, "to_dict"):
                        episodes_data.append(ep.to_dict())
                self._artifact_dumper.dump_episode_archive(episodes=episodes_data)


                round_count = len(getattr(job, "rounds", []))
                self._artifact_dumper.dump_consolidation_summary(
                    job_id=job.id,
                    round_count=round_count,
                    chains_processed=stats["tool_written"],
                    preferences_archived=stats["pref_written"],
                    episodes_archived=stats["episode_written"],
                )
            except Exception as e:
                logger.warning(f"ArtifactDumper consolidation failed (non-fatal): {e}")

        await emit_memory_save_progress(
            progress_callback,
            stage="job_consolidation",
            progress=98,
            message="Job consolidation completed.",
            stats=stats,
        )
        return stats

    async def consolidate_preferences_only(
        self,
        job: Job,
        working_memory: WorkingMemory,
        slide_html_dir: str = "",
        freeze_preference_writeback: bool = False,
        progress_callback: Any = None,
        progress_start: int = 8,
        progress_end: int = 30,
    ) -> int:
        """Write user preferences/profile before slower archival stages."""
        from ..progress import emit_memory_save_progress

        if freeze_preference_writeback:
            logger.info(
                "Consolidator: freeze_preference_writeback=true, "
                "skipping AtomicPreference/UserProfile archival for job %s",
                job.id,
            )
            await emit_memory_save_progress(
                progress_callback,
                stage="preference_profile_writeback",
                progress=progress_end,
                message="Preference/profile writeback is frozen for this job.",
                pref_written=0,
            )
            return 0

        await emit_memory_save_progress(
            progress_callback,
            stage="preference_profile_writeback",
            progress=progress_start,
            message="Writing preferences and user profile first.",
        )
        pref_count = await self._consolidate_preferences(
            job, working_memory, slide_html_dir=slide_html_dir,
        )
        await emit_memory_save_progress(
            progress_callback,
            stage="preference_profile_writeback",
            progress=progress_end,
            message=f"Preference/profile writeback completed ({pref_count} records).",
            pref_written=pref_count,
        )
        return pref_count

    async def _write_chain_experience(
        self, chain_name: str, exp: ChainExperience, session_id: str,
    ) -> Any:
        """Implementation details."""
        if not self._exp_writer:
            return None

        if hasattr(self._exp_writer, "from_chain_experience"):
            return await self._exp_writer.from_chain_experience(
                chain_name=chain_name, experience=exp, session_id=session_id,
            )

        from ..core.models import ExperienceTrace
        trace = ExperienceTrace(
            session_id=session_id,
            task_description=f"工具链经验: {chain_name}",
            lessons_learned=exp.lesson,
            tools_used=json.dumps(exp.tool_pipeline),
            applicable_scenarios=json.dumps(["pipeline"]),
            confidence=exp.confidence,

            experience_type="chain",
            chain_name=chain_name,
            tool_sequence=json.dumps(exp.tool_pipeline),
            anti_pattern=exp.anti_pattern,
            applicable_when=exp.applicable_when,
            source_chain_ids=json.dumps(exp.source_chain_ids),
        )
        await self._exp_writer._write_trace(trace, agent_name="consolidator")
        return trace

    async def _extract_and_archive_chains(
        self, chain_name: str, chains: list, job: Job,
    ) -> int:
        """Implementation details."""
        if not self._llm:
            return 0
        from ..extract.chain_experience_extractor import ChainExperienceExtractor
        extractor = ChainExperienceExtractor(llm=self._llm)
        experiences = await extractor.extract(chain_name, chains)
        count = 0
        for exp in experiences:
            try:
                trace = await self._write_chain_experience(chain_name, exp, job.id)
                if trace:
                    count += 1
            except Exception as e:
                logger.warning(f"Chain experience write failed: {e}")
        return count

    async def _consolidate_preferences(
        self, job: Job, wm: WorkingMemory,
        slide_html_dir: str = "",
    ) -> int:
        """Implementation details."""


        from ..inject.profile_injection_router import LTM_INJECT_SOURCE
        wm_prefs = [p for p in wm._temp_preferences
                     if p.source_task_id != LTM_INJECT_SOURCE]

        logger.info(
            f"Consolidator: filtered {len(wm._temp_preferences)} → {len(wm_prefs)} preferences "
            f"(removed {len(wm._temp_preferences) - len(wm_prefs)} LTM inject entries)"
        )


        raw_messages = self._collect_raw_user_messages(job)




        archived_wm_prefs: list[TempPreference] = []
        skipped_job_local_prefs = 0
        skipped_template_prefs = 0
        for pref in wm_prefs:
            structured = pref.structured_data if isinstance(pref.structured_data, dict) else {}
            retention_scope = str(structured.get("retention_scope", "") or "").strip().lower()
            if retention_scope == "job_local":
                skipped_job_local_prefs += 1
                continue
            if is_template_related_preference(
                dimension=pref.dimension,
                content=pref.content,
            ):
                skipped_template_prefs += 1
                continue
            archived_wm_prefs.append(pref)

        if skipped_job_local_prefs:
            logger.info(
                "Consolidator: skipped %d job_local WM preferences for LTM archival",
                skipped_job_local_prefs,
            )

        if skipped_template_prefs:
            logger.info(
                "Consolidator: skipped %d template prefs for profile/AP archival; "
                "template memory is usage-history driven",
                skipped_template_prefs,
            )



        # slide_analysis = self._extract_fingerprints_from_final_ppt(slide_html_dir)
        # vlm_supplement = await self._vlm_analyze_final_ppt(
        #     slide_html_dir, stat_analysis=slide_analysis,
        # )
        # if vlm_supplement:
        #     slide_analysis += vlm_supplement
        slide_analysis = ""

        if not archived_wm_prefs and not raw_messages:
            return 0

        pref_write_count = 0
        profile_updated = False


        if self._enable_atomic_preference_writeback and self._pref_store and archived_wm_prefs:
            for pref in archived_wm_prefs:
                try:
                    from ..core.models import AtomicPreference
                    ap = AtomicPreference(
                        user_id=job.user_id,
                        preference=pref.content,
                        preference_type=pref.preference_type,
                        scope=pref.scope,
                        scope_value=pref.scope_value,
                        conflict_group=pref.dimension,
                        source_job_id=job.id,
                        source_session_id=job.project_id,
                    )
                    await self._pref_store.add(ap, auto_supersede=True)
                    pref_write_count += 1
                except Exception as e:
                    logger.warning(f"Preference archival to AP failed: {e}")
        elif archived_wm_prefs and not self._enable_atomic_preference_writeback:
            logger.info(
                "Consolidator: atomic preference writeback disabled, "
                "skipping %d WM preferences",
                len(archived_wm_prefs),
            )


        if self._enable_profile_writeback and self._profile_store:
            try:

                write_intent = getattr(job, "write_intent", "") or getattr(job, "intent", "") or ""
                task_intent = getattr(job, "intent", "") or ""
                core_persona = getattr(job, "core_persona", "") or ""
                if not write_intent:

                    write_intent = await self._infer_job_intent(job)
                    task_intent = task_intent or write_intent
                    logger.info("Consolidator: fallback inferred write_intent='%s' for job %s", write_intent, job.id)
                else:
                    logger.info(
                        "Consolidator: using job intents task='%s', read='%s', write='%s' (job %s)",
                        task_intent,
                        getattr(job, "read_intent", "") or "",
                        write_intent,
                        job.id,
                    )

                profile = await self._profile_store.get(job.user_id, intent=write_intent)


                _profile_before_snapshot = None
                if self._artifact_dumper:
                    try:
                        from ..collect.artifact_dumper import ArtifactDumper as _AD
                        _profile_before_snapshot = _AD._safe_serialize(
                            asdict(profile) if hasattr(profile, "__dataclass_fields__") else profile
                        )
                    except Exception:
                        pass


                _update_method = "none"
                if self._llm and (raw_messages or slide_analysis):
                    try:
                        await self._llm_raw_profile_update(
                            profile,
                            raw_messages,
                            slide_analysis,
                            target_intent=write_intent or DEFAULT_INTENT,
                            task_intent=task_intent or write_intent or DEFAULT_INTENT,
                            core_persona=core_persona,
                        )
                        _update_method = "llm_raw"
                        logger.info(
                            "Consolidator: raw profile update done "
                            "(messages=%d chars, slide_analysis=%d chars)",
                            len(raw_messages), len(slide_analysis),
                        )
                    except Exception as e:
                        logger.warning(
                            "LLM raw profile update failed, "
                            "falling back to legacy per-dimension merge: %s", e,
                        )

                        if archived_wm_prefs:
                            self._legacy_merge_preferences(profile, archived_wm_prefs)
                            _update_method = "legacy_fallback"
                elif archived_wm_prefs:

                    self._legacy_merge_preferences(profile, archived_wm_prefs)
                    _update_method = "legacy_rule_based"

                await self._profile_store.save(profile, intent=write_intent)
                profile_updated = True

                # Do not mirror explicit intent updates into the legacy default bucket.
                # Once a request resolves to a concrete write_intent, we keep the
                # writeback isolated to that bucket to avoid duplicate profiles and
                # cross-intent contamination.


                if self._artifact_dumper:
                    try:
                        from ..collect.artifact_dumper import ArtifactDumper as _AD
                        _profile_after_snapshot = _AD._safe_serialize(
                            asdict(profile) if hasattr(profile, "__dataclass_fields__") else profile
                        )
                        self._artifact_dumper.dump_preference_consolidation(
                            dimension_assignment=[{"method": _update_method}],
                            profile_before=_profile_before_snapshot,
                            profile_after=_profile_after_snapshot,
                            llm_merges={"method": _update_method},
                            intent=write_intent,
                            slide_html_dir=slide_html_dir,
                            raw_messages=raw_messages,
                            slide_analysis=slide_analysis,
                        )
                    except Exception as e:
                        logger.warning(
                            f"ArtifactDumper dump_preference_consolidation failed (non-fatal): {e}"
                        )
            except Exception as e:
                logger.warning(f"UserProfile update failed: {e}")
        elif (archived_wm_prefs or raw_messages) and not self._enable_profile_writeback:
            logger.info(
                "Consolidator: profile writeback disabled, "
                "skipping UserProfile update for job %s",
                job.id,
            )

        return pref_write_count + (1 if profile_updated else 0)



    @staticmethod
    def _collect_raw_user_messages(job: Job) -> str:
        """Implementation details."""
        rounds = getattr(job, "rounds", [])
        if not rounds:
            return ""
        parts: list[str] = []
        for i, round_obj in enumerate(rounds):
            msg = getattr(round_obj, "user_message", "")
            if msg:
                parts.append(f"[第{i + 1}轮] {msg}")
        return "\n".join(parts)

    @staticmethod
    def _extract_fingerprints_from_final_ppt(slide_html_dir: str) -> str:
        """Implementation details."""
        if not slide_html_dir:
            return "无幻灯片视觉数据"

        html_dir = Path(slide_html_dir)
        if not html_dir.exists() or not html_dir.is_dir():
            return "无幻灯片视觉数据"

        html_files = sorted(html_dir.glob("*.html"))
        if not html_files:
            return "无幻灯片视觉数据"


        from ..extract.visual_fingerprint import SlideVisualFingerprintExtractor
        fp_extractor = SlideVisualFingerprintExtractor()
        all_fps = []
        for html_file in html_files:
            try:
                fp = fp_extractor.extract(html_file)
                all_fps.append(fp)
            except Exception as e:
                logger.debug(f"Fingerprint extraction failed for {html_file.name}: {e}")

        if not all_fps:
            return "无幻灯片视觉指纹数据"

        lines: list[str] = [f"共 {len(all_fps)} 张最终幻灯片的视觉指纹分析："]


        from collections import Counter
        color_freq: Counter[str] = Counter()
        for fp in all_fps:
            for c in fp.colors_used:
                color_freq[c] += 1
        if color_freq:
            top_colors = color_freq.most_common(8)
            lines.append(
                f"- 常用颜色: {', '.join(f'{c}({n}次)' for c, n in top_colors)}"
            )


        font_freq: Counter[str] = Counter()
        for fp in all_fps:
            for fname, _ in fp.fonts_used:
                font_freq[fname] += 1
        if font_freq:
            top_fonts = font_freq.most_common(5)
            lines.append(
                f"- 常用字体: {', '.join(f'{f}({n}次)' for f, n in top_fonts)}"
            )


        title_sizes = [fp.title_size for fp in all_fps if fp.title_size > 0]
        body_sizes = [fp.body_size for fp in all_fps if fp.body_size > 0]
        if title_sizes:
            avg_title = sum(title_sizes) / len(title_sizes)
            lines.append(f"- 标题字号: 平均 {avg_title:.0f}pt")
        if body_sizes:
            avg_body = sum(body_sizes) / len(body_sizes)
            lines.append(f"- 正文字号: 平均 {avg_body:.0f}pt")


        bg_types = Counter(fp.background_type for fp in all_fps if fp.background_type)
        if bg_types:
            lines.append(
                f"- 背景类型: {', '.join(f'{t}({n}次)' for t, n in bg_types.most_common(3))}"
            )


        avg_elements = sum(
            sum(fp.element_counts.values()) for fp in all_fps
        ) / max(len(all_fps), 1)
        lines.append(f"- 平均元素数: {avg_elements:.1f}")


        avg_title_chars = sum(fp.title_char_count for fp in all_fps) / max(len(all_fps), 1)
        avg_body_chars = sum(fp.body_char_count for fp in all_fps) / max(len(all_fps), 1)
        lines.append(
            f"- 文本量: 标题平均 {avg_title_chars:.0f} 字, 正文平均 {avg_body_chars:.0f} 字"
        )


        ratios = [fp.image_text_ratio for fp in all_fps if fp.image_text_ratio > 0]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            lines.append(f"- 图文面积比: {avg_ratio:.2f}:1")


        last_fp = all_fps[-1]
        lines.append(f"\n最终幻灯片详情 (layout={last_fp.layout_name or 'unknown'}):")
        if last_fp.colors_used:
            lines.append(f"  颜色: {', '.join(last_fp.colors_used[:6])}")
        if last_fp.fonts_used:
            lines.append(
                f"  字体: {', '.join(f'{n}({s}pt)' for n, s in last_fp.fonts_used[:4])}"
            )
        lines.append(
            f"  背景: {last_fp.background_type} {last_fp.background_color}"
        )

        return "\n".join(lines)



    async def _vlm_analyze_final_ppt(
        self,
        slide_html_dir: str,
        stat_analysis: str = "",
    ) -> str:
        """Implementation details."""
        if not self._vlm or not slide_html_dir:
            return ""

        html_dir = Path(slide_html_dir)
        if not html_dir.exists() or not html_dir.is_dir():
            return ""

        html_files = sorted(html_dir.glob("*.html"))
        if not html_files:
            return ""

        import base64

        screenshots: list[str] = []




        workspace_dir = html_dir.parent
        existing_dirs = sorted(
            workspace_dir.glob(".slide_images-pdf-*"),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for img_dir in existing_dirs:
            imgs = sorted(img_dir.glob("slide_*.jpg")) + sorted(img_dir.glob("slide_*.png"))
            if imgs:
                for img_path in imgs[:6]:
                    try:
                        raw = img_path.read_bytes()
                        suffix = img_path.suffix.lstrip(".")
                        mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
                        b64 = base64.b64encode(raw).decode("ascii")
                        screenshots.append(f"data:image/{mime};base64,{b64}")
                    except Exception:
                        pass
                if screenshots:
                    logger.info(
                        "VLM: reusing %d existing screenshots from %s",
                        len(screenshots), img_dir.name,
                    )
                    break


        if not screenshots:
            try:
                from ..utils.webview import PlaywrightConverter
                async with PlaywrightConverter() as pc:
                    page = await pc.context.new_page()
                    try:
                        for html_file in html_files[:6]:
                            try:
                                await page.goto(
                                    html_file.resolve().as_uri(),
                                    wait_until="networkidle",
                                )
                                png_bytes = await page.screenshot(type="png")
                                if png_bytes:
                                    b64 = base64.b64encode(png_bytes).decode("ascii")
                                    screenshots.append(f"data:image/png;base64,{b64}")
                            except Exception:
                                pass
                    finally:
                        await page.close()
            except Exception as e:
                logger.debug(f"Playwright screenshot for VLM failed (non-fatal): {e}")

        if not screenshots:
            return ""

        from ..extract.visual_fingerprint import VLM_VISUAL_PREFERENCE_PROMPT

        prompt = VLM_VISUAL_PREFERENCE_PROMPT.format(
            n=len(html_files),
            statistical_summary=stat_analysis or "无确定性偏好检测到",
        )

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            *[{"type": "image_url", "image_url": {"url": s}} for s in screenshots[:6]],
        ]}]

        try:
            logger.info(
                "VLM: sending %d screenshots to %s for visual preference analysis",
                len(screenshots),
                getattr(self._vlm, "model_name", "unknown"),
            )
            if hasattr(self._vlm, "run"):
                response = await self._vlm.run(messages)
                response_text = response.choices[0].message.content
            else:
                response = await self._vlm.agenerate(messages)
                response_text = response.generations[0][0].text

            if response_text:
                logger.info("VLM: visual preference analysis succeeded (%d chars)", len(response_text))
                return f"\n\n## VLM 视觉分析补充\n{response_text}"
            else:
                logger.warning("VLM: visual preference analysis returned empty response")
        except Exception as e:
            logger.warning(f"VLM analysis at consolidation failed (non-fatal): {e}", exc_info=True)

        return ""



    async def _llm_raw_profile_update(
        self,
        profile: UserProfile,
        raw_messages: str,
        slide_analysis: str,
        *,
        target_intent: str = DEFAULT_INTENT,
        task_intent: str = DEFAULT_INTENT,
        core_persona: str = "",
    ) -> None:
        """Implementation details."""
        current_profile_text = json.dumps(
            asdict(profile), ensure_ascii=False, indent=2,
        )

        prompt = PROFILE_CONSOLIDATION_RAW_PROMPT.format(
            target_intent=target_intent or DEFAULT_INTENT,
            task_intent=task_intent or target_intent or DEFAULT_INTENT,
            core_persona=core_persona or "unspecified",
            all_user_messages=raw_messages or "（本次会话无用户消息）",
            slide_visual_analysis=slide_analysis,
            current_profile=current_profile_text,
        )

        try:
            response = await self._llm(prompt)
            text = str(response).strip() if response else ""
        except Exception:
            response = await self._llm.run(
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""


        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            logger.info("Consolidator: LLM raw update returned no updates")
            return

        try:
            updates = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            logger.warning("Consolidator: LLM raw profile update JSON parse failed")
            return

        if not updates:
            return


        dim_map = profile.get_dimension_map()
        from dataclasses import fields as dc_fields
        system_managed_template_fields = {
            "history_preferred_templates",
            "history_preferred_template_ids",
            "history_reuse_scenarios",
            "history_supported_aspect_ratios",
            "last_successful_template_id",
            "last_successful_template_name",
            "last_successful_at",
            "history_supporting_usage_count",
        }
        for dim_name, dim_updates in updates.items():
            if not isinstance(dim_updates, dict):
                continue
            if dim_name == "template":
                logger.info(
                    "Consolidator: skipped template profile update from raw messages; "
                    "template memory is usage-history driven",
                )
                continue
            sub_pref = dim_map.get(dim_name)
            if sub_pref is None:
                continue
            valid_fields = {f.name for f in dc_fields(type(sub_pref))}
            for k, v in dim_updates.items():
                if k == "keywords":
                    continue
                if dim_name == "template" and k in system_managed_template_fields:
                    continue
                if k in valid_fields:
                    current_val = getattr(sub_pref, k)
                    if dim_name == "general" and k in {"preferences", "notes"}:
                        if isinstance(current_val, list):
                            current_val = [
                                item
                                for item in current_val
                                if not is_template_related_preference(content=item)
                            ]
                        if isinstance(v, list):
                            v = [
                                item
                                for item in v
                                if not is_template_related_preference(content=item)
                            ]
                        if not v and not current_val:
                            continue

                    if isinstance(current_val, list) and isinstance(v, list):
                        merged = list(current_val)
                        for item in v:
                            if item not in merged:
                                merged.append(item)
                        setattr(sub_pref, k, merged)
                    else:
                        setattr(sub_pref, k, v)

        profile.last_updated = datetime.now().isoformat()
        logger.info(
            "Consolidator: raw profile update applied dims=%s",
            list(updates.keys()),
        )

    # ── Legacy fallback ──

    def _legacy_merge_preferences(
        self,
        profile: UserProfile,
        wm_prefs: list[TempPreference],
    ) -> None:
        """Implementation details."""
        for pref in wm_prefs:
            dim = profile.match_dimension(pref.content)
            self._rule_merge_dimension(profile, dim, [pref])

    async def _infer_job_intent(self, job: Job) -> str:
        """Implementation details."""

        user_messages = []
        rounds = getattr(job, "rounds", [])
        if rounds:
            for round_obj in rounds[:3]:
                if hasattr(round_obj, "user_message") and round_obj.user_message:
                    user_messages.append(round_obj.user_message)
        combined = " ".join(user_messages)[:500] if user_messages else ""

        if not combined:
            return DEFAULT_INTENT


        from ..core.models import classify_intent_with_llm
        return await classify_intent_with_llm(
            combined,
            llm_client=self._llm,
            fallback_to_keywords=True,
        )

    async def _llm_merge_dimension(
        self,
        profile: UserProfile,
        dim_name: str,
        new_prefs: list[TempPreference],
        *,
        target_intent: str = DEFAULT_INTENT,
        task_intent: str = DEFAULT_INTENT,
        core_persona: str = "",
    ) -> None:
        """Implementation details."""
        dim_map = profile.get_dimension_map()
        sub_pref = dim_map.get(dim_name)
        if sub_pref is None:
            return


        current_state = json.dumps(asdict(sub_pref), ensure_ascii=False, indent=2)


        lines = []
        for p in new_prefs:
            line = f"- [{p.preference_type}] {p.content}"
            line += ' (已被后续偏好取代)' if p.superseded else ' (当前生效)'
            if p.structured_data:
                line += f"\n  参考参数: {json.dumps(p.structured_data, ensure_ascii=False)}"
            lines.append(line)
        new_items = "\n".join(lines)

        prompt = PROFILE_UPDATE_PROMPT.format(
            target_intent=target_intent or DEFAULT_INTENT,
            task_intent=task_intent or target_intent or DEFAULT_INTENT,
            core_persona=core_persona or "unspecified",
            dimension_name=dim_name,
            current_state=current_state,
            new_preferences=new_items,
        )

        try:
            response = await self._llm(prompt)
            text = str(response).strip() if response else ""
        except Exception:

            response = await self._llm.run(
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""


        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                updated = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.warning(f"LLM response JSON parse failed for dim '{dim_name}'")
                return

            sub_cls = type(sub_pref)
            valid_fields = {f.name for f in sub_cls.__dataclass_fields__.values()}
            for k, v in updated.items():
                if k in valid_fields:
                    setattr(sub_pref, k, v)

    @staticmethod
    def _rule_merge_dimension(
        profile: UserProfile,
        dim_name: str,
        prefs: list[TempPreference],
    ) -> None:
        """Implementation details."""
        dim_map = profile.get_dimension_map()
        sub_pref = dim_map.get(dim_name)
        if sub_pref is None:
            return

        if dim_name == "general":
            for p in prefs:
                if p.content not in profile.general.preferences:
                    profile.general.preferences.append(p.content)
            return

        valid_fields = {f.name for f in type(sub_pref).__dataclass_fields__.values()}
        for p in prefs:

            if p.structured_data:
                for k, v in p.structured_data.items():
                    if k not in valid_fields:
                        continue
                    current = getattr(sub_pref, k)
                    if isinstance(current, list) and isinstance(v, list):
                        merged = list(current)
                        for item in v:
                            if item not in merged:
                                merged.append(item)
                        setattr(sub_pref, k, merged)
                    else:
                        setattr(sub_pref, k, v)

            notes = getattr(sub_pref, "notes", None)
            if notes is not None and p.content not in notes:
                notes.append(p.content)

    def save_backup(self, job: Job, wm: WorkingMemory) -> None:
        """Implementation details."""
        backup_dir = self._workspace / ".memory"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / f"pending_consolidation_{job.id}.json"
        try:
            data = {
                "job_id": job.id,
                "user_id": job.user_id,
                "timestamp": datetime.now().isoformat(),
                "temp_preferences": [
                    {"content": p.content, "dimension": p.dimension,
                     "scope": p.scope, "superseded": p.superseded}
                    for p in wm._temp_preferences
                ],
                "task_experiences": [e.to_dict() for e in wm.get_experiences()],
                "chain_experiences": {
                    name: [e.to_dict() for e in exps]
                    for name, exps in wm.chain_buffer.get_all_experiences().items()
                },
                "chains": {
                    name: [c.to_dict() for c in chains]
                    for name, chains in wm.chain_buffer.get_all_chains().items()
                },
            }
            backup_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info(f"Consolidation backup saved to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save consolidation backup: {e}")
