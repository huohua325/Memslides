"""Shared user-state resolver for core persona + intent profile + WM session prefs."""

from __future__ import annotations

from typing import Any

from .core.models import (
    DEFAULT_INTENT,
    UserCoreProfile,
    UserProfile,
    UserStateSnapshot,
    normalize_intent_label,
)


class UserStateResolver:
    """Builds a unified runtime snapshot for generation/review side sharing."""

    def __init__(self, core_profile_store: Any = None, intent_profile_store: Any = None):
        self.core_profile_store = core_profile_store
        self.intent_profile_store = intent_profile_store

    async def build_snapshot(
        self,
        user_id: str,
        task_intent: str = "",
        read_intent: str = "",
        write_intent: str = "",
        core_persona: str = "",
        working_memory: Any = None,
        include_cross_intent: bool = False,
        max_cross_intent_hints: int = 2,
    ) -> UserStateSnapshot:
        resolved_task_intent = normalize_intent_label(task_intent) or DEFAULT_INTENT
        resolved_read_intent = (
            normalize_intent_label(read_intent)
            or resolved_task_intent
            or DEFAULT_INTENT
        )
        resolved_write_intent = (
            normalize_intent_label(write_intent)
            or resolved_task_intent
            or resolved_read_intent
            or DEFAULT_INTENT
        )

        core_profile = await self._load_core_profile(user_id)
        intent_profile = await self._load_intent_profile(user_id, resolved_read_intent)

        resolved_persona = str(core_persona or core_profile.core_persona or "").strip()
        if resolved_persona and core_profile.core_persona != resolved_persona:
            core_profile.core_persona = resolved_persona

        session_preferences = self._collect_session_preferences(working_memory)
        cross_intent_hints: list[str] = []
        if include_cross_intent:
            cross_intent_hints = await self._collect_cross_intent_hints(
                user_id=user_id,
                current_intent=resolved_read_intent,
                max_hints=max_cross_intent_hints,
            )

        return UserStateSnapshot(
            user_id=user_id,
            core_persona=resolved_persona,
            task_intent=resolved_task_intent,
            read_intent=resolved_read_intent,
            write_intent=resolved_write_intent,
            core_profile=core_profile,
            intent_profile=intent_profile,
            session_preferences=session_preferences,
            cross_intent_hints=cross_intent_hints,
            meta={
                "has_core_profile": self._profile_has_content(core_profile),
                "has_intent_profile": self._profile_has_content(intent_profile),
                "session_preference_count": len(session_preferences),
                "cross_intent_hint_count": len(cross_intent_hints),
            },
        )

    def build_prompt_blocks(self, snapshot: UserStateSnapshot) -> dict[str, str]:
        blocks: dict[str, str] = {}

        if snapshot.core_persona:
            blocks["persona_block"] = "## Stable Persona\n- " + snapshot.core_persona

        core_profile_block = snapshot.core_profile.to_prompt_text(include_persona=False)
        if core_profile_block:
            blocks["core_profile_block"] = core_profile_block

        intent_profile_block = snapshot.intent_profile.to_prompt_text()
        if intent_profile_block:
            blocks["intent_profile_block"] = (
                f"## Current Intent Profile ({snapshot.read_intent or DEFAULT_INTENT})\n\n"
                f"{intent_profile_block}"
            )

        if snapshot.session_preferences:
            blocks["session_preferences_block"] = "\n".join(
                ["## Session Preferences"]
                + [f"- {pref}" for pref in snapshot.session_preferences]
            )

        if snapshot.cross_intent_hints:
            blocks["cross_intent_hints_block"] = "\n".join(
                ["## Cross-Intent Hints"]
                + [f"- {hint}" for hint in snapshot.cross_intent_hints]
            )

        ordered_blocks = [
            blocks[key]
            for key in (
                "persona_block",
                "core_profile_block",
                "intent_profile_block",
                "session_preferences_block",
                "cross_intent_hints_block",
            )
            if key in blocks
        ]
        if ordered_blocks:
            blocks["user_state_block"] = "\n\n".join(ordered_blocks)

        return blocks

    async def _load_core_profile(self, user_id: str) -> UserCoreProfile:
        if self.core_profile_store is None:
            return UserCoreProfile(user_id=user_id)
        profile = await self.core_profile_store.get(user_id)
        if not profile.user_id:
            profile.user_id = user_id
        return profile

    async def _load_intent_profile(self, user_id: str, intent: str) -> UserProfile:
        if self.intent_profile_store is None:
            return UserProfile(user_id=user_id)
        profile = await self.intent_profile_store.get(user_id, intent)
        if not profile.user_id:
            profile.user_id = user_id
        return profile

    async def _collect_cross_intent_hints(
        self,
        user_id: str,
        current_intent: str,
        max_hints: int,
    ) -> list[str]:
        if (
            self.intent_profile_store is None
            or not hasattr(self.intent_profile_store, "get_all_intents")
        ):
            return []

        intent_profile = await self.intent_profile_store.get_all_intents(user_id)
        hints: list[str] = []
        for intent in intent_profile.list_intents():
            if intent == current_intent:
                continue
            profile = intent_profile.profiles.get(intent)
            hint = self._summarize_intent_profile(intent, profile)
            if not hint:
                continue
            hints.append(hint)
            if len(hints) >= max_hints:
                break
        return hints

    @staticmethod
    def _collect_session_preferences(working_memory: Any) -> list[str]:
        if working_memory is None:
            return []

        raw_preferences = getattr(working_memory, "_temp_preferences", None) or []
        seen: set[str] = set()
        collected: list[str] = []
        for pref in raw_preferences:
            if getattr(pref, "superseded", False):
                continue
            content = str(getattr(pref, "content", "") or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            collected.append(content)
        return collected

    @staticmethod
    def _profile_has_content(profile: Any) -> bool:
        if profile is None:
            return False
        data = profile.to_dict() if hasattr(profile, "to_dict") else {}

        def _has_meaningful_value(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    _has_meaningful_value(child)
                    for key, child in value.items()
                    if key not in {"confidence", "keywords"}
                )
            if isinstance(value, list):
                return any(_has_meaningful_value(item) for item in value)
            return value not in ("", None)

        return any(
            _has_meaningful_value(value)
            for key, value in data.items()
            if key not in {"user_id", "core_persona", "version", "last_updated"}
        )

    @staticmethod
    def _summarize_intent_profile(intent: str, profile: UserProfile | None) -> str:
        if profile is None:
            return ""

        parts: list[str] = []
        if profile.theme.primary_colors:
            parts.append(f"颜色 {profile.theme.primary_colors}")
        if profile.visual.image_style:
            parts.append(f"视觉 {profile.visual.image_style}")
        if profile.layout.content_density:
            parts.append(f"布局密度 {profile.layout.content_density}")
        if profile.content.language_style:
            parts.append(f"语言风格 {profile.content.language_style}")
        for pref in profile.general.preferences:
            if pref:
                parts.append(pref)
                break

        if not parts:
            return ""
        return f"[{intent}] " + "; ".join(parts[:2])
