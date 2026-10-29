"""Implementation details."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

MERGE_EXPERIENCES_PROMPT = """你是一个记忆整合专家。请将以下相似的工具使用经验合并为一条更抽象、更通用的经验总结。

## 待合并的经验

{experiences_text}

## 要求

1. 保留核心洞察，去除冗余细节
2. 使用更通用的表述，使经验适用于更多场景
3. 如果有多个 workaround，合并为最佳实践
4. 保持简洁，单条经验不超过 100 字

## 输出格式

```json
{{
  "merged_lesson": "合并后的经验总结",
  "applicable_scenarios": ["适用场景1", "适用场景2"],
  "confidence": 0.8
}}
```
"""


@dataclass
class ConsolidationStats:
    """Implementation details."""
    total_before: int = 0
    duplicates_removed: int = 0
    clusters_found: int = 0
    merged_count: int = 0
    total_after: int = 0
    errors: list[str] = None

    episodes_before: int = 0
    episodes_duplicates: int = 0
    episodes_merged: int = 0
    episodes_after: int = 0

    preferences_before: int = 0
    preferences_duplicates: int = 0
    preferences_merged: int = 0
    preferences_after: int = 0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict:
        return {
            "total_before": self.total_before,
            "duplicates_removed": self.duplicates_removed,
            "clusters_found": self.clusters_found,
            "merged_count": self.merged_count,
            "total_after": self.total_after,
            "episodes": {
                "before": self.episodes_before,
                "duplicates": self.episodes_duplicates,
                "merged": self.episodes_merged,
                "after": self.episodes_after,
            },
            "preferences": {
                "before": self.preferences_before,
                "duplicates": self.preferences_duplicates,
                "merged": self.preferences_merged,
                "after": self.preferences_after,
            },
            "errors": self.errors,
        }


class OfflineMemoryConsolidator:
    """Implementation details."""


    SIMILARITY_THRESHOLD = 0.8
    MIN_CLUSTER_SIZE = 2
    MAX_CLUSTER_SIZE = 5

    def __init__(
        self,
        db: Any = None,
        llm: Callable[[str], Any] | None = None,
        embedding_func: Callable[[list[str]], Any] | None = None,
        experience_vector_store: Any = None,
        bm25_store: Any = None,
    ) -> None:
        self._db = db
        self._llm = llm
        self._embed = embedding_func
        self._vector_store = experience_vector_store
        self._bm25_store = bm25_store

    async def _ensure_db_healthy(self) -> bool:
        """Implementation details."""
        if not self._db:
            return False

        try:

            result = await self._db.query("PRAGMA integrity_check")
            if not result or result[0].get("integrity_check") != "ok":
                logger.error("Database integrity check failed")
                return False


            try:
                await self._db.query_one("SELECT COUNT(*) AS cnt FROM experience_traces_fts")
            except Exception as fts_error:
                if self._is_malformed_error(fts_error):
                    logger.warning(
                        "experience_traces_fts appears malformed, attempting repair: %s",
                        fts_error,
                    )
                    if not await self._repair_experience_fts():
                        return False
                else:
                    raise


            await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            logger.debug("Database health check passed")
            return True
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return False

    @staticmethod
    def _is_malformed_error(error: Exception) -> bool:
        return "database disk image is malformed" in str(error).lower()

    async def _repair_experience_fts(self) -> bool:
        """Implementation details."""
        if not self._db:
            return False

        try:
            await self._db.execute("DROP TRIGGER IF EXISTS experience_traces_fts_ai")
            await self._db.execute("DROP TRIGGER IF EXISTS experience_traces_fts_ad")
            await self._db.execute("DROP TRIGGER IF EXISTS experience_traces_fts_au")
            await self._db.execute("DROP TABLE IF EXISTS experience_traces_fts")


            await self._db.init_schema()


            await self._db.execute(
                "INSERT INTO experience_traces_fts(experience_traces_fts) VALUES('rebuild')"
            )
            logger.info("Rebuilt experience_traces_fts successfully")
            return True
        except Exception as e:
            logger.error("Failed to repair experience_traces_fts: %s", e)
            return False

    async def consolidate(self, user_id: str = "default") -> ConsolidationStats:
        """Implementation details."""
        stats = ConsolidationStats()


        if not await self._ensure_db_healthy():
            stats.errors.append("database_unhealthy")
            return stats


        experiences = await self._load_experiences(user_id)
        stats.total_before = len(experiences)

        if len(experiences) >= 2:
            unique_experiences, removed, ids_to_delete, reuse_updates = self._deduplicate_exact(experiences)
            stats.duplicates_removed = removed
            logger.info("Consolidator: removed %d exact duplicates (experiences)", removed)


            for exp_id, total_reuse in reuse_updates.items():
                try:
                    await self._db.execute(
                        "UPDATE experience_traces SET reuse_count = ? WHERE id = ?",
                        (total_reuse, exp_id),
                    )
                except Exception as e:
                    logger.warning("Failed to update reuse_count for %s: %s", exp_id, e)


            if ids_to_delete:
                try:
                    await self._delete_experiences(ids_to_delete)
                except Exception as e:
                    logger.warning("Failed to delete experience duplicates: %s", e)


            if self._embed and len(unique_experiences) >= 2:
                try:
                    clusters = await self._cluster_by_similarity(unique_experiences)
                    stats.clusters_found = len([c for c in clusters if len(c) >= self.MIN_CLUSTER_SIZE])

                    if self._llm and stats.clusters_found > 0:
                        merged = await self._merge_clusters(clusters)
                        stats.merged_count = merged
                        logger.info("Consolidator: merged %d clusters (experiences)", merged)
                except Exception as e:
                    logger.warning("Consolidator: experience clustering failed: %s", e)
                    stats.errors.append(f"exp_clustering: {e}")

        final_experiences = await self._load_experiences(user_id)
        stats.total_after = len(final_experiences)


        try:
            ep_stats = await self._consolidate_episodes(user_id)
            stats.episodes_before = ep_stats.get("before", 0)
            stats.episodes_duplicates = ep_stats.get("duplicates", 0)
            stats.episodes_merged = ep_stats.get("merged", 0)
            stats.episodes_after = ep_stats.get("after", 0)
            logger.info("Consolidator: episodes %d → %d", stats.episodes_before, stats.episodes_after)
        except Exception as e:
            logger.warning("Consolidator: episode consolidation failed: %s", e)
            stats.errors.append(f"episodes: {e}")


        try:
            pref_stats = await self._consolidate_preferences(user_id)
            stats.preferences_before = pref_stats.get("before", 0)
            stats.preferences_duplicates = pref_stats.get("duplicates", 0)
            stats.preferences_merged = pref_stats.get("merged", 0)
            stats.preferences_after = pref_stats.get("after", 0)
            logger.info("Consolidator: preferences %d → %d", stats.preferences_before, stats.preferences_after)
        except Exception as e:
            logger.warning("Consolidator: preference consolidation failed: %s", e)
            stats.errors.append(f"preferences: {e}")


        try:
            await self._rebuild_indexes(user_id)
        except Exception as e:
            logger.warning("Consolidator: index rebuild failed: %s", e)
            stats.errors.append(f"rebuild: {e}")

        return stats

    async def _load_experiences(self, user_id: str) -> list[dict]:
        """Implementation details."""
        if not self._db:
            return []

        try:
            rows = await self._db.query(
                """SELECT et.id, et.session_id, et.task_description, et.tools_used,
                          et.final_outcome, et.lessons_learned, et.applicable_scenarios,
                          et.confidence, et.reuse_count, et.created_at
                   FROM experience_traces et
                   JOIN sessions s ON et.session_id = s.id
                   WHERE s.user_id = ?
                   ORDER BY et.created_at DESC""",
                (user_id,)
            )
            return rows if rows else []
        except Exception as e:
            logger.error("Failed to load experiences: %s", e)
            return []

    def _deduplicate_exact(self, experiences: list[dict]) -> tuple[list[dict], int, list[str], dict[str, int]]:
        """Implementation details."""

        groups: dict[str, list[dict]] = {}
        for exp in experiences:
            key = (exp.get("lessons_learned", "") or "").strip()
            if not key:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)

        unique = []
        removed = 0
        ids_to_delete = []

        reuse_updates: dict[str, int] = {}  # id → merged reuse_count to write back

        for key, group in groups.items():
            if len(group) == 1:
                unique.append(group[0])
            else:

                sorted_group = sorted(
                    group,
                    key=lambda x: (x.get("reuse_count", 0), x.get("created_at", "")),
                    reverse=True
                )
                best = sorted_group[0]


                total_reuse = sum(g.get("reuse_count", 0) for g in group)
                best["reuse_count"] = total_reuse
                reuse_updates[best["id"]] = total_reuse

                unique.append(best)
                ids_to_delete.extend([g["id"] for g in sorted_group[1:]])
                removed += len(sorted_group) - 1

        return unique, removed, ids_to_delete, reuse_updates

    async def _delete_experiences(self, ids: list[str]) -> bool:
        """Implementation details."""
        if not self._db or not ids:
            return True

        try:
            placeholders = ",".join("?" * len(ids))
            await self._db.execute(
                f"DELETE FROM experience_traces WHERE id IN ({placeholders})",
                tuple(ids),
            )
            logger.debug("Deleted %d experiences", len(ids))
            return True
        except Exception as e:
            if self._is_malformed_error(e):
                logger.warning("Delete experiences hit malformed DB, trying FTS repair")
                if await self._repair_experience_fts():
                    try:
                        placeholders = ",".join("?" * len(ids))
                        await self._db.execute(
                            f"DELETE FROM experience_traces WHERE id IN ({placeholders})",
                            tuple(ids),
                        )
                        logger.debug("Deleted %d experiences after FTS repair", len(ids))
                        return True
                    except Exception as retry_error:
                        logger.error("Retry delete experiences failed: %s", retry_error)
            logger.error("Failed to delete experiences: %s", e)
            return False

    async def _cluster_by_similarity(self, experiences: list[dict]) -> list[list[dict]]:
        """Implementation details."""
        if not self._embed or len(experiences) < 2:
            return [[exp] for exp in experiences]

        lessons = [exp.get("lessons_learned", "") or "" for exp in experiences]
        try:
            vectors = await self._embed(lessons)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return [[exp] for exp in experiences]

        n = len(experiences)
        used = [False] * n
        clusters: list[list[dict]] = []

        for i in range(n):
            if used[i]:
                continue

            cluster = [experiences[i]]
            cluster_vecs = [vectors[i]]
            used[i] = True

            for j in range(i + 1, n):
                if used[j] or len(cluster) >= self.MAX_CLUSTER_SIZE:
                    continue

                if all(
                    self._cosine_similarity(cluster_vecs[k], vectors[j]) >= self.SIMILARITY_THRESHOLD
                    for k in range(len(cluster_vecs))
                ):
                    cluster.append(experiences[j])
                    cluster_vecs.append(vectors[j])
                    used[j] = True

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        """Implementation details."""
        import numpy as np


        a1 = np.array(v1).flatten() if hasattr(v1, '__iter__') else np.array([v1])
        a2 = np.array(v2).flatten() if hasattr(v2, '__iter__') else np.array([v2])

        if len(a1) == 0 or len(a2) == 0 or len(a1) != len(a2):
            return 0.0

        norm1 = np.linalg.norm(a1)
        norm2 = np.linalg.norm(a2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(a1, a2) / (norm1 * norm2))

    async def _merge_clusters(self, clusters: list[list[dict]]) -> int:
        """Implementation details."""
        if not self._llm:
            return 0

        merged_count = 0

        for cluster in clusters:
            if len(cluster) < self.MIN_CLUSTER_SIZE:
                continue

            try:

                experiences_text = "\n\n".join([
                    f"### 经验 {i+1}\n"
                    f"- 任务: {exp.get('task_description', '')[:100]}\n"
                    f"- 教训: {exp.get('lessons_learned', '')}\n"
                    f"- 适用场景: {exp.get('applicable_scenarios', '[]')}"
                    for i, exp in enumerate(cluster)
                ])

                prompt = MERGE_EXPERIENCES_PROMPT.format(experiences_text=experiences_text)
                response = await self._llm(prompt)


                merged = self._parse_merge_response(response)
                if not merged:
                    continue


                updated = await self._create_merged_experience(cluster, merged)
                if not updated:
                    continue


                old_ids = [exp["id"] for exp in cluster[1:]]
                if old_ids and not await self._delete_experiences(old_ids):
                    continue

                merged_count += 1

            except Exception as e:
                logger.warning("Failed to merge cluster: %s", e)

        return merged_count

    def _parse_merge_response(self, response: str) -> dict | None:
        """Implementation details."""
        import re


        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass


        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        return None

    async def _create_merged_experience(
        self,
        cluster: list[dict],
        merged: dict,
    ) -> bool:
        """Implementation details."""
        if not self._db or not cluster:
            return False

        primary = cluster[0]
        merged_lesson = merged.get("merged_lesson", "")
        if not merged_lesson:
            return False

        try:
            await self._db.execute(
                "UPDATE experience_traces SET lessons_learned = ? WHERE id = ?",
                (merged_lesson, primary["id"]),
            )
            logger.debug("Updated merged experience: %s", primary["id"])
            return True
        except Exception as e:
            if self._is_malformed_error(e):
                logger.warning("Update merged experience hit malformed DB, trying FTS repair")
                if await self._repair_experience_fts():
                    try:
                        await self._db.execute(
                            "UPDATE experience_traces SET lessons_learned = ? WHERE id = ?",
                            (merged_lesson, primary["id"]),
                        )
                        logger.debug("Updated merged experience after FTS repair: %s", primary["id"])
                        return True
                    except Exception as retry_error:
                        logger.error("Retry update merged experience failed: %s", retry_error)
            logger.error("Failed to update merged experience: %s", e)
            return False

    async def _rebuild_indexes(self, user_id: str) -> None:
        """Implementation details."""
        experiences = await self._load_experiences(user_id)
        if self._vector_store and self._embed:
            try:

                if hasattr(self._vector_store, 'delete_collection'):
                    await self._vector_store.delete_collection("experiences")


                for exp in experiences:
                    lesson = exp.get("lessons_learned", "")
                    if lesson:
                        vector = (await self._embed([lesson]))[0]
                        if hasattr(self._vector_store, 'add'):
                            await self._vector_store.add(
                                collection="experiences",
                                id=exp["id"],
                                vector=vector,
                                metadata={"lesson": lesson[:500]},
                            )
                logger.info("Rebuilt vector index with %d experiences", len(experiences))
            except Exception as e:
                logger.warning("Vector index rebuild failed: %s", e)

        if self._bm25_store:
            try:
                if hasattr(self._bm25_store, 'rebuild'):
                    documents = [
                        {"id": exp["id"], "text": exp.get("lessons_learned", "")}
                        for exp in experiences
                        if exp.get("lessons_learned")
                    ]
                    await self._bm25_store.rebuild(documents)
                    logger.info("Rebuilt BM25 index with %d experiences", len(documents))
            except Exception as e:
                logger.warning("BM25 index rebuild failed: %s", e)

    # ═══════════════════════════════════════════════

    # ═══════════════════════════════════════════════

    async def _consolidate_episodes(self, user_id: str) -> dict:
        """Implementation details."""
        stats = {"before": 0, "duplicates": 0, "merged": 0, "after": 0}

        if not self._db:
            return stats


        try:
            rows = await self._db.query(
                """SELECT id, user_id, session_id, source_round_id, user_intent,
                          interpretation_gap, action_outcome, design_insight, category, confidence, created_at
                   FROM design_episodes
                   WHERE user_id = ? AND status = 'active'
                   ORDER BY created_at DESC""",
                (user_id,)
            )
            episodes = rows if rows else []
        except Exception as e:
            logger.warning("Failed to load episodes: %s", e)
            return stats

        stats["before"] = len(episodes)
        if len(episodes) < 2:
            stats["after"] = len(episodes)
            return stats


        groups: dict[str, list[dict]] = {}
        for ep in episodes:
            key = (ep.get("design_insight", "") or "").strip()
            if not key:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(ep)

        ids_to_delete = []
        for key, group in groups.items():
            if len(group) > 1:

                sorted_group = sorted(group, key=lambda x: x.get("created_at", ""), reverse=True)
                ids_to_delete.extend([g["id"] for g in sorted_group[1:]])
                stats["duplicates"] += len(sorted_group) - 1


        deleted_ids: set[str] = set()
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            try:
                await self._db.execute(
                    f"DELETE FROM design_episodes WHERE id IN ({placeholders})",
                    tuple(ids_to_delete)
                )
                deleted_ids = set(ids_to_delete)
            except Exception as e:
                logger.warning("Failed to delete duplicate episodes: %s", e)


        if self._embed and self._llm:
            remaining_episodes = [ep for ep in episodes if ep["id"] not in deleted_ids]
            if len(remaining_episodes) >= 2:
                try:
                    merged = await self._merge_similar_episodes(remaining_episodes, user_id)
                    stats["merged"] = merged
                except Exception as e:
                    logger.warning("Episode merging failed: %s", e)


        try:
            result = await self._db.query_one(
                "SELECT COUNT(*) as cnt FROM design_episodes WHERE user_id = ? AND status = 'active'",
                (user_id,)
            )
            stats["after"] = result["cnt"] if result else 0
        except Exception:
            pass

        return stats

    async def _merge_similar_episodes(self, episodes: list[dict], user_id: str) -> int:
        """Implementation details."""
        if not self._embed or len(episodes) < 2:
            return 0

        insights = [ep.get("design_insight", "") or "" for ep in episodes]
        try:
            vectors = await self._embed(insights)
        except Exception:
            return 0

        n = len(episodes)
        used = [False] * n
        clusters: list[list[dict]] = []

        for i in range(n):
            if used[i]:
                continue
            cluster = [episodes[i]]
            cluster_vecs = [vectors[i]]
            used[i] = True
            for j in range(i + 1, n):
                if used[j] or len(cluster) >= self.MAX_CLUSTER_SIZE:
                    continue
                if all(
                    self._cosine_similarity(cluster_vecs[k], vectors[j]) >= self.SIMILARITY_THRESHOLD
                    for k in range(len(cluster_vecs))
                ):
                    cluster.append(episodes[j])
                    cluster_vecs.append(vectors[j])
                    used[j] = True
            clusters.append(cluster)


        merged_count = 0
        for cluster in clusters:
            if len(cluster) < self.MIN_CLUSTER_SIZE:
                continue

            try:

                primary = cluster[0]
                insights_text = "\n".join([
                    f"- {ep.get('design_insight', '')}" for ep in cluster
                ])

                prompt = f"""请将以下相似的设计洞察合并为一条更抽象的总结：

{insights_text}

要求：保留核心洞察，去除冗余，不超过 150 字。直接输出合并后的文本，无需格式。"""

                merged_insight = await self._llm(prompt)
                merged_insight = merged_insight.strip()[:300]


                await self._db.execute(
                    "UPDATE design_episodes SET design_insight = ? WHERE id = ?",
                    (merged_insight, primary["id"])
                )


                other_ids = [ep["id"] for ep in cluster[1:]]
                if other_ids:
                    placeholders = ",".join("?" * len(other_ids))
                    await self._db.execute(
                        f"DELETE FROM design_episodes WHERE id IN ({placeholders})",
                        tuple(other_ids)
                    )

                merged_count += 1
            except Exception as e:
                logger.warning("Failed to merge episode cluster: %s", e)

        return merged_count

    # ═══════════════════════════════════════════════

    # ═══════════════════════════════════════════════

    async def _consolidate_preferences(self, user_id: str) -> dict:
        """Implementation details."""
        stats = {"before": 0, "duplicates": 0, "merged": 0, "after": 0}

        if not self._db:
            return stats


        try:
            rows = await self._db.query(
                """SELECT id, user_id, preference_type, preference, confidence,
                          scope, scope_value, evidence_episode_ids, created_at
                   FROM atomic_preferences
                   WHERE user_id = ? AND status = 'active'
                   ORDER BY created_at DESC""",
                (user_id,)
            )
            preferences = rows if rows else []
        except Exception as e:
            logger.warning("Failed to load preferences: %s", e)
            return stats

        stats["before"] = len(preferences)
        if len(preferences) < 2:
            stats["after"] = len(preferences)
            return stats


        groups: dict[tuple, list[dict]] = {}
        for pref in preferences:
            pref_text = (pref.get("preference", "") or "").strip()
            if not pref_text:
                continue
            key = (
                pref_text,
                pref.get("scope", "global"),
                pref.get("scope_value", ""),
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(pref)

        ids_to_delete = []
        for key, group in groups.items():
            if len(group) > 1:

                sorted_group = sorted(group, key=lambda x: x.get("confidence", 0), reverse=True)
                ids_to_delete.extend([g["id"] for g in sorted_group[1:]])
                stats["duplicates"] += len(sorted_group) - 1


        deleted_ids: set[str] = set()
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            try:
                await self._db.execute(
                    f"DELETE FROM atomic_preferences WHERE id IN ({placeholders})",
                    tuple(ids_to_delete)
                )
                deleted_ids = set(ids_to_delete)
            except Exception as e:
                logger.warning("Failed to delete duplicate preferences: %s", e)


        if self._embed and self._llm:
            remaining = [p for p in preferences if p["id"] not in deleted_ids]
            if len(remaining) >= 2:
                try:
                    merged = await self._merge_similar_preferences(remaining, user_id)
                    stats["merged"] = merged
                except Exception as e:
                    logger.warning("Preference merging failed: %s", e)


        try:
            result = await self._db.query_one(
                "SELECT COUNT(*) as cnt FROM atomic_preferences WHERE user_id = ?",
                (user_id,)
            )
            stats["after"] = result["cnt"] if result else 0
        except Exception:
            pass

        return stats

    async def _merge_similar_preferences(self, preferences: list[dict], user_id: str) -> int:
        """Implementation details."""
        if not self._embed or len(preferences) < 2:
            return 0



        by_type: dict[tuple, list[dict]] = {}
        for pref in preferences:
            group_key = (
                pref.get("preference_type", "value"),
                pref.get("scope", "global"),
                pref.get("scope_value", ""),
            )
            if group_key not in by_type:
                by_type[group_key] = []
            by_type[group_key].append(pref)

        merged_count = 0

        for ptype, type_prefs in by_type.items():
            if len(type_prefs) < 2:
                continue


            texts = [p.get("preference", "") or "" for p in type_prefs]
            try:
                vectors = await self._embed(texts)
            except Exception:
                continue


            n = len(type_prefs)
            used = [False] * n
            clusters: list[list[dict]] = []

            for i in range(n):
                if used[i]:
                    continue
                cluster = [type_prefs[i]]
                cluster_vecs = [vectors[i]]
                used[i] = True
                for j in range(i + 1, n):
                    if used[j] or len(cluster) >= self.MAX_CLUSTER_SIZE:
                        continue
                    if all(
                        self._cosine_similarity(cluster_vecs[k], vectors[j]) >= self.SIMILARITY_THRESHOLD
                        for k in range(len(cluster_vecs))
                    ):
                        cluster.append(type_prefs[j])
                        cluster_vecs.append(vectors[j])
                        used[j] = True
                clusters.append(cluster)


            for cluster in clusters:
                if len(cluster) < self.MIN_CLUSTER_SIZE:
                    continue

                try:
                    primary = cluster[0]
                    prefs_text = "\n".join([f"- {p.get('preference', '')}" for p in cluster])

                    prompt = f"""请将以下相似的用户偏好合并为一条：

{prefs_text}

要求：保留核心偏好，去除冗余，不超过 100 字。直接输出合并后的文本。"""

                    merged_text = await self._llm(prompt)
                    merged_text = merged_text.strip()[:200]


                    max_conf = max(p.get("confidence", 0.5) for p in cluster)
                    await self._db.execute(
                        "UPDATE atomic_preferences SET preference = ?, confidence = ? WHERE id = ?",
                        (merged_text, min(max_conf + 0.1, 1.0), primary["id"])
                    )


                    other_ids = [p["id"] for p in cluster[1:]]
                    if other_ids:
                        placeholders = ",".join("?" * len(other_ids))
                        await self._db.execute(
                            f"DELETE FROM atomic_preferences WHERE id IN ({placeholders})",
                            tuple(other_ids)
                        )

                    merged_count += 1
                except Exception as e:
                    logger.warning("Failed to merge preference cluster: %s", e)

        return merged_count
