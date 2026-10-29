"""RoundCache -- Round 级记忆缓存。

每个 Round 开始时从 LTM + WM 检索记忆，缓存供该 Round 内所有 Operation 使用。

Stage 12 重构:
- LTM 偏好检索源从 AtomicPreferenceStore 切换为 UserProfileStore
- UserProfile 按维度选择性注入：用 DimensionMatcher（向量匹配 → 关键词降级）
  筛选与 user_message 相关的维度，只生成相关维度的 prompt
- general.preferences 始终注入（无论是否在筛选集合中）

WM 优先级改造:
- WM 偏好检索改用 DimensionMatcher（与 LTM 同口径），替代旧的 token 重叠度搜索
- WM 偏好优先级高于 LTM：WM 反映当前 Job 内的最新意图，LTM 是跨 Session 的历史沉淀
- 注入顺序：WM 偏好 → LTM UserProfile，冲突时以 WM 为准
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..core.models import ChainExperience, OperationMemory, TempPreference, UserProfile, classify_intent_by_keywords

if TYPE_CHECKING:
    from ..collect.artifact_dumper import ArtifactDumper
    from ..core.dimension_matcher import DimensionMatcher
    from ..store.atomic_preference_store import AtomicPreferenceStore
    from ..store.user_profile_store import UserProfileStore
    from ..working_memory import WorkingMemory

logger = logging.getLogger(__name__)


def _profile_has_meaningful_content(profile: UserProfile | None) -> bool:
    if profile is None:
        return False

    def _has_value(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                _has_value(child)
                for key, child in value.items()
                if key not in {"confidence", "keywords"}
            )
        if isinstance(value, list):
            return any(_has_value(item) for item in value)
        return value not in ("", None)

    try:
        data = profile.to_dict()
    except Exception:
        return False

    return any(
        _has_value(value)
        for key, value in data.items()
        if key not in {"user_id", "version", "last_updated"}
    )


class RoundCache:
    """Round 级记忆缓存。

    三路检索：
    1. Tool Memory LTM -- ExperienceTraceWriter.query_for_role()
    2. Preference Memory LTM -- UserProfileStore.get() → DimensionMatcher 筛选维度
    3. WM -- search_preferences() + search_tool_experiences()
    """

    def __init__(
        self,
        tool_memory_retriever: Any = None,  # legacy, kept for API compatibility
        preference_store: AtomicPreferenceStore | None = None,  # legacy, kept for API compatibility
        profile_store: UserProfileStore | None = None,
        working_memory: WorkingMemory | None = None,
        dimension_matcher: DimensionMatcher | None = None,
    ):
        # legacy 参数保留以兼容调用方，但不再存储
        del tool_memory_retriever, preference_store
        self._profile_store = profile_store
        self._wm = working_memory
        self._dim_matcher = dimension_matcher
        self._artifact_dumper: ArtifactDumper | None = None
        self._round_index: int = 0
        self._cached_profile: UserProfile | None = None
        self._cached_profile_prompt: str = ""
        self._cached_relevant_dims: set[str] | None = None
        self._cached_wm_preferences: list[TempPreference] = []
        self._cached_wm_chain_exps: list[ChainExperience] = []
        self._is_loaded = False

    async def load(
        self,
        user_message: str,
        user_id: str,
        context: dict | None = None,
        session_id: str = "",
        intent: str = "",
    ) -> None:
        """从 LTM + WM 检索记忆并缓存。

        agent_phase 从 context["agent_phase"] 获取：
        - "research" / "design": 全量注入所有维度（初始创建需要完整画像）
        - "modify": 按 user_message 语义筛选 top-3 相关维度（减少 prompt 噪音）
        - 其他/缺失: 保持全量兜底

        intent: 用户意图类别（如 "academic"/"business"），空字符串时自动推断
        """
        context = context or {}
        agent_phase = context.get("agent_phase", "")

        # 推断 intent（如果未提供）
        if not intent:
            intent = classify_intent_by_keywords(user_message)

        # 1. Tool Memory LTM — 已移至 Operation 级实时匹配（ToolChainBuffer.query_or_cache_from_ltm）

        # 2. Preference Memory LTM -- 从 UserProfile 检索 + 维度筛选（intent-aware）
        if self._profile_store:
            try:
                self._cached_profile = await self._profile_store.get(user_id, intent=intent)
                if _profile_has_meaningful_content(self._cached_profile):
                    logger.info(
                        "RoundCache: loaded populated profile for %s (intent=%s), font=%s, density=%s",
                        user_id, intent,
                        getattr(self._cached_profile.theme, 'font_family', ''),
                        getattr(self._cached_profile.layout, 'content_density', ''),
                    )
                else:
                    logger.warning(
                        "RoundCache: loaded EMPTY profile shell for %s (intent=%s); no persisted profile content found",
                        user_id,
                        intent,
                    )
                # Research/Design 阶段：全量注入（初始创建需要完整用户画像）
                # Modify 阶段：按 user_message 筛选 top-3 相关维度
                if agent_phase in ("research", "design"):
                    relevant_dims = None  # 全量注入
                    logger.info(
                        "RoundCache: agent_phase='%s', injecting full profile (all dimensions)",
                        agent_phase,
                    )
                else:
                    relevant_dims = await self._select_relevant_dimensions(
                        user_message, self._cached_profile,
                    )
                self._cached_relevant_dims = relevant_dims
                self._cached_profile_prompt = self._cached_profile.to_prompt_text(
                    dimensions=relevant_dims,
                    include_general=True,
                )
                logger.info(
                    "RoundCache: profile_prompt len=%d, dims=%s",
                    len(self._cached_profile_prompt), relevant_dims,
                )
                if relevant_dims is not None:
                    logger.info(
                        "RoundCache: profile dims selected=%s for message='%s'",
                        relevant_dims, user_message[:80],
                    )
            except Exception as e:
                logger.warning(f"UserProfile LTM retrieval failed: {e}")

        # 3. WM 检索 — 使用与 LTM 相同的 DimensionMatcher 维度筛选
        if self._wm:
            try:
                self._cached_wm_preferences = await self._wm.search_preferences_by_dimension(
                    matched_dims=self._cached_relevant_dims,
                    context=context,
                    top_k=10,
                )
            except Exception:
                # 降级：旧的 token 重叠度搜索
                try:
                    self._cached_wm_preferences = self._wm.search_preferences(
                        query=user_message, context=context, top_k=10,
                    )
                except Exception:
                    pass
            try:
                self._cached_wm_chain_exps = self._wm.search_tool_experiences(
                    query=user_message, top_k=20,
                )
            except Exception:
                pass

        self._is_loaded = True

    async def _select_relevant_dimensions(
        self, user_message: str, profile: UserProfile,
    ) -> set[str] | None:
        """根据 user_message 选择 UserProfile 中相关的维度子集（top-3）。

        仅在 Modify 阶段调用（Research/Design 阶段直接全量注入）。

        策略（按优先级降级）：
        1. DimensionMatcher.match_top_k（向量余弦相似度）：取 score > threshold 的 top-3 维度
        2. DimensionMatcher.match（单维度降级）：向量匹配只返回 1 个
        3. 关键词匹配（profile.match_dimension）：无 embedding 时降级
        4. 全量兜底：如果匹配不到任何维度，返回 None（表示全量注入）

        返回 None 表示全量注入，返回 set 表示只注入指定维度。
        """
        # 收集 profile 中有数据的维度（用于中间产物记录）
        profile_dims_with_data = [
            d for d in ["theme", "visual", "layout", "content", "template", "general"]
            if profile.to_prompt_text(dimensions={d}, include_general=(d == "general"))
        ]

        match_method = ""
        matched_dims: set[str] | None = None
        fallback_reason = ""

        # 策略 1: 向量 top-3 匹配
        if self._dim_matcher is not None:
            try:
                dims = await self._dim_matcher.match_top_k(
                    user_message, top_k=3, profile=profile,
                )
                if dims:
                    # 验证至少有一个维度在 profile 中有数据
                    valid_dims = {
                        d for d in dims
                        if profile.to_prompt_text(dimensions={d}, include_general=False)
                    }
                    if valid_dims:
                        match_method = "vector_top_k"
                        matched_dims = valid_dims
                        self._dump_dimension_matching(
                            user_message, matched_dims, match_method,
                            profile_dims_with_data, fallback_reason,
                        )
                        return valid_dims
                    fallback_reason = f"vector matched {dims} but profile has no data"
                    logger.info(
                        "DimensionMatcher top-3 matched %s but profile has no data, fallback to all",
                        dims,
                    )
            except Exception as e:
                fallback_reason = f"vector match failed: {e}"
                logger.debug("DimensionMatcher.match_top_k failed, fallback to keywords: %s", e)

        # 策略 2: 关键词匹配（单维度）
        dim = profile.match_dimension(user_message)
        if dim != "general":
            if profile.to_prompt_text(dimensions={dim}, include_general=False):
                match_method = "keyword"
                matched_dims = {dim}
                self._dump_dimension_matching(
                    user_message, matched_dims, match_method,
                    profile_dims_with_data, fallback_reason,
                )
                return {dim}
            fallback_reason = f"keyword matched '{dim}' but profile has no data"
            logger.info("Keyword matched '%s' but profile has no data for it, fallback to all", dim)

        # 策略 3: 全量兜底（无法判断哪个维度相关时，全量注入）
        match_method = "fallback_all"
        matched_dims = None
        if not fallback_reason:
            fallback_reason = "no dimension matched"
        self._dump_dimension_matching(
            user_message, matched_dims, match_method,
            profile_dims_with_data, fallback_reason,
        )
        return None

    def _dump_dimension_matching(
        self,
        user_message: str,
        matched_dims: set[str] | None,
        match_method: str,
        profile_dims_with_data: list[str],
        fallback_reason: str,
    ) -> None:
        """转储维度匹配中间产物（如果 ArtifactDumper 可用）。"""
        if self._artifact_dumper is not None:
            try:
                self._artifact_dumper.dump_dimension_matching(
                    self._round_index,
                    user_message=user_message,
                    matched_dims=matched_dims,
                    match_method=match_method,
                    profile_dims_with_data=profile_dims_with_data,
                    fallback_reason=fallback_reason,
                )
            except Exception:
                pass

    def refresh_wm_preferences(self, context: dict | None = None) -> None:
        """刷新 WM 偏好缓存（偏好写入 WM 后调用，消除 1 轮延迟）。"""
        if self._wm:
            try:
                context = context or {}
                candidates = [
                    p for p in self._wm._temp_preferences
                    if not p.superseded and p.matches_context(context)
                ]
                if self._cached_relevant_dims is None:
                    general = []
                    specific = []
                    for pref in candidates:
                        dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
                        if dim_prefix in {"", "general"}:
                            general.append(pref)
                        else:
                            specific.append(pref)
                    if len(general) >= 10:
                        self._cached_wm_preferences = general
                    else:
                        self._cached_wm_preferences = general + specific[: 10 - len(general)]
                else:
                    general = []
                    matched = []
                    for pref in candidates:
                        dim_prefix = pref.dimension.split(".")[0] if pref.dimension else ""
                        if dim_prefix in {"", "general"}:
                            general.append(pref)
                        elif dim_prefix in self._cached_relevant_dims:
                            matched.append(pref)
                    if len(general) >= 10:
                        self._cached_wm_preferences = general
                    else:
                        self._cached_wm_preferences = general + matched[: 10 - len(general)]
            except Exception:
                pass

    def get_profile_prompt(self) -> str:
        """获取 LTM UserProfile 生成的偏好 prompt 文本。"""
        return self._cached_profile_prompt

    def get_for_operation(
        self, tool_name: str, context: dict | None = None,
    ) -> OperationMemory:
        """为单个 Operation 筛选相关记忆。

        工具经验来源已改为 ToolChainBuffer（WM 新经验 + LTM 缓存），
        由 OperationDistributor 在调用前触发 query_or_cache_from_ltm。
        """
        context = context or {}

        # 工具链经验：从 ToolChainBuffer 获取（WM 新经验 + LTM 缓存，WM 优先）
        tool_exps: list[ChainExperience] = []
        if self._wm:
            tool_exps = self._wm.chain_buffer.get_experiences_for_tool(tool_name)

        # WM 临时偏好
        wm_prefs = [
            p for p in self._cached_wm_preferences
            if p.matches_context(context)
        ]

        return OperationMemory(
            tool_experiences=tool_exps[:5],
            preferences=wm_prefs[:10],
        )
