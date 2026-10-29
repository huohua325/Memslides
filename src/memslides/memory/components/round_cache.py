"""Implementation details."""

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
    """Implementation details."""

    def __init__(
        self,
        tool_memory_retriever: Any = None,  # legacy, kept for API compatibility
        preference_store: AtomicPreferenceStore | None = None,  # legacy, kept for API compatibility
        profile_store: UserProfileStore | None = None,
        working_memory: WorkingMemory | None = None,
        dimension_matcher: DimensionMatcher | None = None,
    ):

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
        """Implementation details."""
        context = context or {}
        agent_phase = context.get("agent_phase", "")


        if not intent:
            intent = classify_intent_by_keywords(user_message)




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


                if agent_phase in ("research", "design"):
                    relevant_dims = None
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


        if self._wm:
            try:
                self._cached_wm_preferences = await self._wm.search_preferences_by_dimension(
                    matched_dims=self._cached_relevant_dims,
                    context=context,
                    top_k=10,
                )
            except Exception:

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
        """Implementation details."""

        profile_dims_with_data = [
            d for d in ["theme", "visual", "layout", "content", "template", "general"]
            if profile.to_prompt_text(dimensions={d}, include_general=(d == "general"))
        ]

        match_method = ""
        matched_dims: set[str] | None = None
        fallback_reason = ""


        if self._dim_matcher is not None:
            try:
                dims = await self._dim_matcher.match_top_k(
                    user_message, top_k=3, profile=profile,
                )
                if dims:

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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
        return self._cached_profile_prompt

    def get_for_operation(
        self, tool_name: str, context: dict | None = None,
    ) -> OperationMemory:
        """Implementation details."""
        context = context or {}


        tool_exps: list[ChainExperience] = []
        if self._wm:
            tool_exps = self._wm.chain_buffer.get_experiences_for_tool(tool_name)


        wm_prefs = [
            p for p in self._cached_wm_preferences
            if p.matches_context(context)
        ]

        return OperationMemory(
            tool_experiences=tool_exps[:5],
            preferences=wm_prefs[:10],
        )
