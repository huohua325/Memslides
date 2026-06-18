"""Factory to initialize memory system components from MemSlidesConfig.

Usage:
    from memslides.utils.config import GLOBAL_CONFIG
    from memslides.memory.config_helper import MemorySystem

    mem = await MemorySystem.from_config(GLOBAL_CONFIG)
    # Access components:
    #   mem.db, mem.llm, mem.embedding_func
    #   mem.classifier, mem.session_manager, mem.state_coordinator
    #   mem.rollback_manager, mem.debug_tracer
    #   mem.collector, mem.episode_store, mem.retriever
    #   mem.preference_store, mem.preference_extractor
    #   mem.memory_consolidator
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemorySystem:
    """Holds all initialized memory components."""

    # Core
    db: Any = None

    # LLM & Embedding
    llm: Any = None
    llm_by_task: dict = field(default_factory=dict)  # 分级LLM: {task_type: callable}
    llm_objects_by_task: dict = field(default_factory=dict)  # Stage 5: 原始LLM对象 {task_type: AsyncLLM}
    embedding_func: Any = None

    # Extractor
    classifier: Any = None

    # Session
    session_manager: Any = None

    # S6: State
    state_extractor: Any = None
    params_store: Any = None
    state_coordinator: Any = None

    # S9: Evolution
    rollback_manager: Any = None

    # Debug
    debug_tracer: Any = None

    # Stage 2: Cognitive Memory Pipeline
    collector: Any = None
    episode_extractor: Any = None
    episode_store: Any = None
    retriever: Any = None
    event_bus: Any = None

    # Stage 2 Phase 2/3: Advanced components
    experience_retriever: Any = None  # ChromaDB + BM25 混合检索

    # Stage 4: AtomicPreference (替代 UserDesignProfile)
    preference_store: Any = None
    preference_extractor: Any = None
    memory_consolidator: Any = None  # Stage 4: 离线记忆整合器

    # Stage 12: UserProfile (LTM 偏好检索的唯一源)
    profile_store: Any = None
    core_profile_store: Any = None
    user_state_resolver: Any = None

    # Stage 5: Template Profile
    template_store: Any = None

    # Config reference
    config: Any = None

    # Artifact dumper
    artifact_dumper: Any = None
    artifact_writer: Any = None

    async def _close_component(self, component: Any, label: str) -> None:
        if component is None:
            return
        close = getattr(component, "close", None) or getattr(component, "aclose", None)
        if not close:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Failed to close memory component %s: %s", label, exc)

    @classmethod
    async def from_config(
        cls,
        global_config: Any,
        project_dir: str | Path | None = None,
    ) -> MemorySystem:
        """Initialize all memory components from MemSlidesConfig.

        Args:
            global_config: MemSlidesConfig instance (from GLOBAL_CONFIG)
            project_dir: Project working directory for resolving relative paths.
                         Defaults to current directory.
        """
        mem_cfg = global_config.memory
        if not mem_cfg.enabled:
            logger.info("Memory system is disabled in config")
            return cls(config=mem_cfg)

        base_dir = Path(project_dir) if project_dir else Path(".")

        # ── Stage 2.5 module switches (defaults to all enabled) ──
        _modules = getattr(mem_cfg, 'modules', None)
        _mod = _modules if _modules else type('M', (), {})()  # fallback empty
        def _mod_enabled(name: str, default: bool = True) -> bool:
            return getattr(_mod, name, default)

        # ── Resolve paths ──
        if mem_cfg.global_db_dir:
            global_dir = Path(mem_cfg.global_db_dir).expanduser()
            db_name = "global_memory_v2.db" if mem_cfg.memory_v2 else "global_memory.db"
            db_path = global_dir / db_name
            logger.info(f"Using global memory DB: {db_path} (memory_v2={mem_cfg.memory_v2})")
        else:
            db_path = base_dir / mem_cfg.db_path

        # File artifacts always per-session
        params_store_dir = base_dir / mem_cfg.params_store_dir

        debug_dir = base_dir / ".memory" / "debug"

        # Ensure directories exist
        for p in [db_path.parent, params_store_dir, debug_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # ── Debug Tracer ──
        from .debug_tracer import DebugTracer

        debug_tracer = DebugTracer(debug_dir)
        debug_tracer.log_event("memory_system_init_start", {
            "db_path": str(db_path),
            "llm_ref": mem_cfg.llm_ref,
            "project_dir": str(base_dir),
        })

        # ── Artifact Dumper ──
        from .collect.artifact_dumper import ArtifactDumper

        _artifact_trace = getattr(mem_cfg, 'artifact_trace', False)
        artifact_dumper = ArtifactDumper(
            memory_dir=base_dir / ".memory",
            enabled=_artifact_trace,
        )
        if _artifact_trace:
            logger.info("ArtifactDumper enabled — writing intermediate artifacts to .memory/")

        # ── Resolve LLM ──
        default_llm_ref = mem_cfg.llm_ref
        default_llm_obj = getattr(global_config, default_llm_ref, None)
        if default_llm_obj is None:
            raise ValueError(
                f"memory.llm_ref='{default_llm_ref}' does not match any LLM in config. "
                f"Available: research_agent, design_agent, long_context_model, vision_model"
            )
        logger.info(f"Memory system default LLM: {default_llm_ref}")

        # ── 分级 LLM 配置 ──
        # memory.llm: { task_type: model_ref, ... }
        llm_config = getattr(mem_cfg, 'llm', None) or {}

        def _resolve_llm_obj(task_type: str) -> Any:
            """Resolve LLM object by task type, fallback to default.

            Uses ``global_config[ref]`` which checks both declared fields
            (research_agent, design_agent, …) and extra_llms (fast_model, balanced_model, …).
            """
            ref = llm_config.get(task_type, default_llm_ref)
            try:
                obj = global_config[ref]
            except (KeyError, AttributeError):
                obj = None
            if obj is None:
                logger.warning(
                    "memory.llm.%s='%s' not found, fallback to '%s'",
                    task_type, ref, default_llm_ref,
                )
                return default_llm_obj
            return obj

        def _make_llm_callable(llm_obj: Any):
            """Wrap LLM object as async callable(prompt: str) -> str."""
            async def _call(prompt: str) -> str:
                response = await llm_obj.run(prompt)
                return response.choices[0].message.content
            return _call

        # 默认 LLM callable (向后兼容)
        llm = _make_llm_callable(default_llm_obj)

        # 分级 LLM callable dict
        llm_by_task = {
            'intent_classify': _make_llm_callable(_resolve_llm_obj('intent_classify')),
            'user_message_analyze': _make_llm_callable(_resolve_llm_obj('user_message_analyze')),
            'episode_extract': _make_llm_callable(_resolve_llm_obj('episode_extract')),
            'multi_query': _make_llm_callable(_resolve_llm_obj('multi_query')),
            'memory_consolidation': _make_llm_callable(_resolve_llm_obj('memory_consolidation')),
            'template_analyze': _make_llm_callable(_resolve_llm_obj('template_analyze')),
        }
        logger.info("Memory LLM routing: %s", {k: llm_config.get(k, default_llm_ref) for k in llm_by_task})

        # Stage 5: 原始 LLM 对象字典（某些组件如 SlideInducter 需要 AsyncLLM 对象而不是 callable）
        llm_objects_by_task = {
            'template_analyze': _resolve_llm_obj('template_analyze'),
            'vision': getattr(global_config, 'design_agent', None),  # 用于 layout 聚类，使用 design_agent
            # main.py 使用: intent classification, profile injection, style classification
            'intent_classify': _resolve_llm_obj('intent_classify'),
            'profile_injection': _resolve_llm_obj('profile_injection'),
            'style_classify': _resolve_llm_obj('style_classify'),
        }

        # ── Embedding function ──
        embedding_func = None
        embedding_dim = 1024
        if mem_cfg.embedding:
            from .core.embedding import get_embedding_func

            _emb = mem_cfg.embedding
            embedding_func = get_embedding_func(
                provider=getattr(_emb, "provider", "bge-m3"),
                cache=getattr(_emb, "cache_enabled", True),
                cache_size=getattr(_emb, "cache_size", 512),
                model=_emb.model if hasattr(_emb, "model") else "BAAI/bge-m3",
                model_name=_emb.model if hasattr(_emb, "model") else "BAAI/bge-m3",
                device=getattr(_emb, "device", "auto"),
                batch_size=getattr(_emb, "batch_size", 32),
                max_length=getattr(_emb, "max_length", 512),
                api_model=getattr(_emb, "api_model", "") or getattr(_emb, "model", "BAAI/bge-m3"),
                api_base_url=getattr(_emb, "api_base_url", "") or getattr(_emb, "base_url", ""),
                api_key=getattr(_emb, "api_key", ""),
                base_url=getattr(_emb, "base_url", ""),
                api_fallback_model=getattr(_emb, "api_fallback_model", ""),
                api_fallback_base_url=getattr(_emb, "api_fallback_base_url", ""),
                api_fallback_api_key=getattr(_emb, "api_fallback_api_key", ""),
            )
            embedding_dim = _emb.dim
            logger.info(
                "Embedding: provider=%s, model=%s, dim=%d",
                getattr(_emb, "provider", "bge-m3"), _emb.model, embedding_dim,
            )

        # ── Database ──
        from .core.db import SQLiteBackend

        db = SQLiteBackend(db_path)
        await db.connect()
        await db.init_schema()
        logger.info(f"Memory DB initialized at {db_path}")

        # ── Intent Classifier ──
        from .intent_classifier import IntentClassifier

        classifier = IntentClassifier(
            llm_by_task['intent_classify'],
        )
        # IntelligentRuleExtractor: DELETED (Stage 1 legacy)

        # ── Session Manager ──
        from .session.session_manager import SessionManager

        session_manager = SessionManager(db)

        # ── State Extractor + Params Store ──
        from .store.params_store import SlideParamsStore
        from .extract.state_extractor import PPTStateExtractor

        state_extractor = PPTStateExtractor()
        params_store = SlideParamsStore(params_store_dir)

        # ── State Coordinator ──
        from .extract.state_coordinator import StateCoordinator

        state_coordinator = StateCoordinator(
            extractor=state_extractor,
            params_store=params_store,
        )

        # ── Rollback ──
        from .evolution.rollback import RollbackManager

        rollback_manager = RollbackManager(params_store)

        # ── Stage 2: Cognitive Memory Pipeline ──
        collector = None
        episode_store_obj = None
        episode_extractor_obj = None
        hybrid_retriever = None
        event_bus_obj = None
        unified_search_obj = None
        extraction_artifact_writer = None

        try:
            # Event Bus
            from .core.event_bus import MemoryEventBus
            event_bus_obj = MemoryEventBus()

            # Stage 4: 初始化中间产物写入器 (v1 fallback)
            from .collect.artifact_writer import ExtractionArtifactWriter, RoundArtifactWriter
            memory_dir = base_dir / ".memory"
            round_artifact_writer = RoundArtifactWriter(memory_dir)
            extraction_artifact_writer = ExtractionArtifactWriter(memory_dir)
            logger.info("RoundArtifactWriter initialized at %s", memory_dir)
            logger.info("ExtractionArtifactWriter initialized at %s", memory_dir)

            # Collector — persist_dir 使 pending rounds 在进程重启后可恢复
            # Stage 12: 传入 artifact_dumper (v2)，优先使用
            from .collect.collector import MemoryCollector
            collector_persist_dir = base_dir / ".memory"
            collector = MemoryCollector(
                persist_dir=str(collector_persist_dir),
                artifact_writer=round_artifact_writer,  # Stage 4 (v1 fallback)
                artifact_dumper=artifact_dumper,  # Stage 12 (v2 优先)
            )
            # Wire EventBus callbacks
            from .core.event_bus import MemoryEvent
            event_bus_obj.subscribe(MemoryEvent.ROLLBACK_EXECUTED, collector.on_rollback)

            # ── Stage 3: ChromaDB + BM25 + UnifiedSearchEngine ──
            chroma_search = None
            bm25_search = None

            if embedding_func:
                try:
                    from .retrieve.chroma_search import ChromaVectorSearch
                    chroma_dir = str(db_path.parent / "chroma_db")
                    chroma_search = ChromaVectorSearch(
                        persist_dir=chroma_dir,
                        embedding_func=embedding_func,
                        collection_prefix="dp",
                    )
                    logger.info("ChromaVectorSearch initialized at %s", chroma_dir)
                except Exception as e:
                    logger.warning("ChromaVectorSearch init failed (non-fatal): %s", e)

            try:
                from .retrieve.bm25_search import BM25SearchStrategy
                bm25_dir = str(db_path.parent / "bm25_indices")
                bm25_search = BM25SearchStrategy(persist_dir=bm25_dir)
                logger.info("BM25SearchStrategy initialized at %s", bm25_dir)
            except Exception as e:
                logger.warning("BM25SearchStrategy init failed (non-fatal): %s", e)

            if chroma_search or bm25_search:
                from .retrieve.unified_search import UnifiedSearchEngine
                unified_search_obj = UnifiedSearchEngine(
                    chroma_search=chroma_search,
                    bm25_search=bm25_search,
                    rrf_k=60,
                )
                logger.info(
                    "UnifiedSearchEngine initialized (chroma=%s, bm25=%s)",
                    chroma_search is not None, bm25_search is not None,
                )

            # Episode Store
            from .store.episode_store import EpisodeStore
            episode_store_obj = EpisodeStore(
                db=db,
                unified_search=unified_search_obj,
            )

            # Episode Extractor
            from .extract.episode_extractor import EpisodeExtractor
            episode_extractor_obj = EpisodeExtractor(
                llm=llm_by_task['episode_extract'],
                episode_store=episode_store_obj,
                event_bus=event_bus_obj,
                debug_tracer=debug_tracer,
            )

            # Stage 4 新增: 注入满意度触发提取回调
            if collector:
                async def _on_extraction_needed(batch):
                    """满意度信号触发的立即提取"""
                    if episode_extractor_obj and batch:
                        await episode_extractor_obj.extract_and_store(
                            batch, user_id="default", session_id=base_dir.stem,
                        )
                collector._on_extraction_needed = _on_extraction_needed
                logger.info("Injected on_extraction_needed callback into collector")

            # Hybrid Retriever — use UnifiedSearchEngine or fallback to old HybridRetriever
            if unified_search_obj:
                hybrid_retriever = unified_search_obj
            else:
                from .retrieve.strategies import (
                    FTS5SearchStrategy,
                    HybridRetriever,
                )
                strategies = []
                strategies.append(FTS5SearchStrategy(db, "design_episodes", "design_episodes_fts"))
                hybrid_retriever = HybridRetriever(strategies=strategies)
                logger.info("Fallback: HybridRetriever with %d strategies", len(strategies))

            logger.info(
                "Stage 2 cognitive pipeline initialized: collector, episode_store, "
                "episode_extractor, search_engine=%s, event_bus",
                type(hybrid_retriever).__name__,
            )
        except Exception as e:
            logger.warning("Stage 2 cognitive pipeline init failed (non-fatal): %s", e)

        # ── Stage 2 Phase 2/3: Advanced stores + components ──
        preference_store_obj = None
        preference_extractor_obj = None
        experience_retriever_obj = None
        memory_consolidator_obj = None

        try:
            if _mod_enabled('preference_extraction'):
                try:
                    from .store.atomic_preference_store import AtomicPreferenceStore
                    preference_store_obj = AtomicPreferenceStore(db)
                    logger.info("AtomicPreferenceStore initialized (Stage 4)")
                    # 注: AtomicPreferenceExtractor (Episode→AtomicPreference 路径) 已废弃。
                    # 偏好现通过 Consolidator 在 Job 结束时直接写 UserProfile。
                    # AtomicPreferenceStore 保留供 Consolidator 双写暂存记录使用。
                except Exception as e:
                    logger.warning("AtomicPreferenceStore init failed: %s", e)
            else:
                logger.info("AtomicPreference disabled by config (preference_extraction=false)")

            # Experience Trace Retriever (ChromaDB + BM25 混合检索)
            if db and embedding_func:
                try:
                    from .retrieve.experience_retriever import ExperienceTraceRetriever
                    experience_retriever_obj = ExperienceTraceRetriever(
                        db=db,
                        embedding_func=embedding_func,
                        persist_dir=str(db_path.parent),
                    )
                    # 重建索引以确保已有 traces 被索引
                    import asyncio
                    asyncio.create_task(experience_retriever_obj.rebuild_index())
                    logger.info("ExperienceTraceRetriever initialized (ChromaDB + BM25)")
                except Exception as e:
                    logger.warning("ExperienceTraceRetriever init failed: %s", e)

            # Agentic Retriever (Task 15) — wraps HybridRetriever
            if hybrid_retriever:
                from .retrieve.agentic_retriever import AgenticMemoryRetriever
                agentic_retriever = AgenticMemoryRetriever(
                    hybrid=hybrid_retriever,
                    llm=llm_by_task['multi_query'],
                    enable_agentic=getattr(mem_cfg, 'enable_agentic_retrieval', False),
                )
                hybrid_retriever = agentic_retriever  # replace

            from .evolution.memory_consolidator import OfflineMemoryConsolidator

            # Stage 4: 离线记忆整合器
            memory_consolidator_obj = OfflineMemoryConsolidator(
                db=db,
                llm=llm_by_task.get('memory_consolidation', llm),
                embedding_func=embedding_func,
                experience_vector_store=chroma_search,
                bm25_store=bm25_search,
            )

            logger.info(
                "Stage 2 Phase 2/3 initialized: "
                "preference_store, memory_consolidator"
            )
        except Exception as e:
            logger.warning("Stage 2 Phase 2/3 init failed (non-fatal): %s", e)

        # Stage 15: ToolKnowledgeLearner 已废弃，tool_error 统一走 orchestrator.on_tool_error() → WM

        # Stage 12: UserProfileStore
        profile_store_obj = None
        try:
            from .store.user_profile_store import UserProfileStore
            profile_store_obj = UserProfileStore(db=db)
            await profile_store_obj.ensure_schema()
            logger.info("UserProfileStore initialized (Stage 12)")
        except Exception as e:
            logger.warning("UserProfileStore init failed (non-fatal): %s", e)

        # Stage 16 foundation: stable core profile + shared user state resolver
        core_profile_store_obj = None
        user_state_resolver_obj = None
        try:
            from .store.user_core_profile_store import UserCoreProfileStore
            from .user_state_resolver import UserStateResolver

            core_profile_store_obj = UserCoreProfileStore(db=db)
            await core_profile_store_obj.ensure_schema()
            user_state_resolver_obj = UserStateResolver(
                core_profile_store=core_profile_store_obj,
                intent_profile_store=profile_store_obj,
            )
            logger.info("UserCoreProfileStore initialized (Stage 16 foundation)")
            logger.info("UserStateResolver initialized (Stage 16 foundation)")
        except Exception as e:
            logger.warning(
                "User core profile foundation init failed (non-fatal): %s",
                e,
            )

        # Stage 5: TemplateStore
        template_store_obj = None
        try:
            from .store.template_store import TemplateStore
            template_store_obj = TemplateStore(db=db, embedding_func=embedding_func)
            await template_store_obj.ensure_schema()
            logger.info("TemplateStore initialized (Stage 5)")
        except Exception as e:
            logger.warning("TemplateStore init failed (non-fatal): %s", e)

        logger.info("Memory system fully initialized")
        debug_tracer.log_event("memory_system_init_complete", {
            "components": [
                "db", "classifier",
                "session_manager", "state_coordinator",
                "rollback_manager",
                "collector", "episode_store", "episode_extractor", "hybrid_retriever",
                "event_bus", "preference_store",
            ]
        })
        debug_tracer.update_dashboard("system", {
            "status": "initialized",
            "db_path": str(db_path),
            "llm_ref": mem_cfg.llm_ref,
            "embedding_enabled": embedding_func is not None,
            "stage2_enabled": collector is not None,
        })

        return cls(
            db=db,
            llm=llm,
            llm_by_task=llm_by_task,
            llm_objects_by_task=llm_objects_by_task,
            embedding_func=embedding_func,
            classifier=classifier,
            session_manager=session_manager,
            state_extractor=state_extractor,
            params_store=params_store,
            state_coordinator=state_coordinator,
            rollback_manager=rollback_manager,
            debug_tracer=debug_tracer,
            collector=collector,
            episode_extractor=episode_extractor_obj,
            episode_store=episode_store_obj,
            retriever=hybrid_retriever,
            event_bus=event_bus_obj,
            experience_retriever=experience_retriever_obj,
            preference_store=preference_store_obj,
            memory_consolidator=memory_consolidator_obj,
            template_store=template_store_obj,  # Stage 5 新增
            profile_store=profile_store_obj,  # Stage 12 新增
            core_profile_store=core_profile_store_obj,
            user_state_resolver=user_state_resolver_obj,
            config=mem_cfg,
            artifact_dumper=artifact_dumper,
            artifact_writer=extraction_artifact_writer,
        )

    async def close(self):
        """Cleanup resources."""
        await self._close_component(self.retriever, "retriever")
        await self._close_component(self.experience_retriever, "experience_retriever")

        seen_llm_ids: set[int] = set()
        for label, llm_obj in [
            ("llm", getattr(self, "llm", None)),
            *[(f"llm_objects_by_task.{name}", obj) for name, obj in (self.llm_objects_by_task or {}).items()],
        ]:
            if llm_obj is None or id(llm_obj) in seen_llm_ids:
                continue
            seen_llm_ids.add(id(llm_obj))
            await self._close_component(llm_obj, label)

        await self._close_component(self.db, "db")
        logger.info("Memory system closed")
