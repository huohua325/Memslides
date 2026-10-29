"""
TemplateStore — 模板档案 SQLite 存储 (Stage 5)

功能:
- add: 添加/更新模板档案（同源模板自动更新）
- get: 按 ID 获取
- get_by_source: 按模板源文件获取
- list_by_user: 获取用户所有模板档案
- search: FTS5 + LIKE 降级搜索
- delete: 软删除
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..core.template_models import (
    TemplateProfile,
    DesignConstraints,
    ContentPatterns,
    TemplateSemanticModel,
)


class TemplateStore:
    """模板档案 SQLite 存储

    复用 design_skills 表，通过 skill_type='template_layout' 区分。
    """

    TABLE = "design_skills"
    FTS_TABLE = "design_skills_fts"

    def __init__(self, db: Any, embedding_func: Any = None) -> None:
        """
        Args:
            db: SQLiteBackend 实例
            embedding_func: 嵌入函数（可选，用于语义搜索）
        """
        self.db = db
        self.embedding_func = embedding_func
        self._fingerprint_cache: dict[str, tuple[int, int, str]] = {}

    async def ensure_schema(self) -> None:
        """运行迁移：为旧数据库添加新列和表（表/索引/FTS 由 schema.sql 统一管理）"""
        # ══════════════════════════════════════════════════════════════════════════
        # Sub-task 2.1: Create template_usage_history table
        # ══════════════════════════════════════════════════════════════════════════
        try:
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS template_usage_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    user_message_embedding TEXT,
                    intent TEXT NOT NULL,
                    memory_intent TEXT DEFAULT '',
                    aspect_ratio TEXT DEFAULT '',
                    has_attachment INTEGER NOT NULL DEFAULT 0,
                    attachment_type TEXT DEFAULT '',
                    success INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)
            logger.info("[MIGRATION] Created template_usage_history table")
        except Exception as e:
            logger.debug(f"[MIGRATION] template_usage_history table already exists: {e}")

        try:
            usage_columns = await self.db.query(
                "PRAGMA table_info(template_usage_history)"
            )
            existing_usage_columns = {
                row["name"] for row in usage_columns
            } if usage_columns else set()
            for col_name, col_def in (
                ("memory_intent", "TEXT DEFAULT ''"),
                ("aspect_ratio", "TEXT DEFAULT ''"),
            ):
                if col_name in existing_usage_columns:
                    continue
                await self.db.execute(
                    f"ALTER TABLE template_usage_history ADD COLUMN {col_name} {col_def}"
                )
                logger.info(
                    "[MIGRATION] Added column %s to template_usage_history",
                    col_name,
                )
        except Exception as e:
            logger.warning("[MIGRATION] template_usage_history column repair failed: %s", e)

        # Create indexes for template_usage_history
        try:
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_template_usage_user_created
                ON template_usage_history(user_id, created_at DESC)
            """)
            logger.info("[MIGRATION] Created idx_template_usage_user_created index")
        except Exception as e:
            logger.debug(f"[MIGRATION] idx_template_usage_user_created already exists: {e}")

        try:
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_template_usage_template
                ON template_usage_history(template_id)
            """)
            logger.info("[MIGRATION] Created idx_template_usage_template index")
        except Exception as e:
            logger.debug(f"[MIGRATION] idx_template_usage_template already exists: {e}")

        try:
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_template_usage_user_intent_created
                ON template_usage_history(user_id, memory_intent, created_at DESC)
            """)
            logger.info("[MIGRATION] Created idx_template_usage_user_intent_created index")
        except Exception as e:
            logger.debug(
                f"[MIGRATION] idx_template_usage_user_intent_created already exists: {e}"
            )

        # ══════════════════════════════════════════════════════════════════════════
        # Sub-task 2.2: Add last_used_at column to design_skills table
        # ══════════════════════════════════════════════════════════════════════════
        migration_columns = [
            ("slide_induction", "TEXT DEFAULT '{}'"),
            ("semantic_model", "TEXT DEFAULT '{}'"),
            ("template_dir", "TEXT DEFAULT ''"),
            ("image_stats", "TEXT DEFAULT '{}'"),
            ("last_used_at", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in migration_columns:
            try:
                await self.db.execute(f"ALTER TABLE design_skills ADD COLUMN {col_name} {col_def}")
                logger.info(f"[MIGRATION] Added column {col_name} to design_skills")
            except Exception:
                pass  # 列已存在，忽略

        # Create index for last_used_at
        try:
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_design_skills_last_used
                ON design_skills(user_id, last_used_at DESC)
            """)
            logger.info("[MIGRATION] Created idx_design_skills_last_used index")
        except Exception as e:
            logger.debug(f"[MIGRATION] idx_design_skills_last_used already exists: {e}")

    async def add(self, profile: TemplateProfile) -> str:
        """添加模板档案

        如果同源模板已存在，更新现有记录。

        Args:
            profile: TemplateProfile 对象

        Returns:
            档案 ID
        """
        now = datetime.now().isoformat()

        # 提取关键词用于搜索
        keywords = self._extract_keywords(profile)

        data = {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "category": "layout",
            "skill_type": "template_layout",
            "triggers": json.dumps(list(profile.slide_induction.get("functional_keys", [])), ensure_ascii=False),
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "template_source": profile.template_source,
            "slide_count": profile.slide_count,
            "aspect_ratio": profile.aspect_ratio,
            # slide_induction 原始布局数据
            "slide_induction": json.dumps(profile.slide_induction, ensure_ascii=False),
            "semantic_model": json.dumps(profile.semantic_model.to_dict(), ensure_ascii=False),
            "template_dir": profile.template_dir,
            "image_stats": json.dumps(profile.image_stats, ensure_ascii=False),
            # memslides 增量
            "design_constraints": json.dumps(profile.design_constraints.to_dict(), ensure_ascii=False),
            "content_patterns": json.dumps(profile.content_patterns.to_dict(), ensure_ascii=False),
            "confidence": profile.confidence,
            "created_at": profile.created_at or now,
            "updated_at": now,
            "user_id": profile.user_id,
            "status": profile.status,
        }

        # 检查是否已存在同名模板（优先）或同源模板
        existing = await self.get_by_name(profile.name, profile.user_id)
        if not existing:
            existing = await self.get_by_source(profile.template_source, profile.user_id)

        if existing:
            # 更新现有记录（保留原 ID 和创建时间）
            update_fields = {k: v for k, v in data.items() if k not in ("id", "created_at")}
            set_clause = ", ".join(f"{k} = ?" for k in update_fields.keys())
            await self.db.execute(
                f"UPDATE {self.TABLE} SET {set_clause} WHERE id = ?",
                list(update_fields.values()) + [existing.id]
            )
            logger.info(f"Updated template profile (by name): {existing.id} ({profile.name})")
            return existing.id

        # 插入新记录
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        await self.db.execute(
            f"INSERT INTO {self.TABLE} ({columns}) VALUES ({placeholders})",
            list(data.values())
        )

        logger.info(f"Added template profile: {profile.id} ({profile.name})")
        return profile.id

    async def get(self, profile_id: str) -> TemplateProfile | None:
        """按 ID 获取模板档案"""
        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE id = ? AND skill_type = 'template_layout'",
            [profile_id]
        )
        if not rows:
            return None
        return self._row_to_profile(rows[0])

    async def get_by_name(self, name: str, user_id: str) -> TemplateProfile | None:
        """按模板名称获取档案（用于同名覆盖）"""
        if not name or not name.strip():
            return None

        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE name = ? AND user_id = ? AND skill_type = 'template_layout' AND status = 'active'",
            [name, user_id]
        )
        if not rows:
            return None
        return self._row_to_profile(rows[0])

    async def get_by_source(self, template_source: str, user_id: str) -> TemplateProfile | None:
        """按模板源文件获取档案"""
        # 空路径检查，避免错误匹配
        if not template_source or not template_source.strip():
            return None

        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE template_source = ? AND user_id = ? AND skill_type = 'template_layout'",
            [template_source, user_id]
        )
        if not rows:
            return None
        return self._row_to_profile(rows[0])

    async def get_any_by_source(
        self,
        template_source: str,
        exclude_user_id: str | None = None,
    ) -> TemplateProfile | None:
        """按模板源文件获取任意用户的档案"""
        if not template_source or not template_source.strip():
            return None

        conditions = [
            "template_source = ?",
            "skill_type = 'template_layout'",
            "status = 'active'",
        ]
        params: list[Any] = [template_source]
        if exclude_user_id:
            conditions.append("user_id != ?")
            params.append(exclude_user_id)

        rows = await self.db.query(
            f"""
            SELECT * FROM {self.TABLE}
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        return self._row_to_profile(rows[0])

    async def list_by_name(
        self,
        name: str,
        *,
        user_id: str | None = None,
        exclude_user_id: str | None = None,
        limit: int = 20,
    ) -> list[TemplateProfile]:
        """按模板名称列出候选档案"""
        if not name or not name.strip():
            return []

        conditions = [
            "name = ?",
            "skill_type = 'template_layout'",
            "status = 'active'",
        ]
        params: list[Any] = [name]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if exclude_user_id:
            conditions.append("user_id != ?")
            params.append(exclude_user_id)

        params.append(limit)
        rows = await self.db.query(
            f"""
            SELECT * FROM {self.TABLE}
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_profile(r) for r in rows]

    async def find_equivalent_profile(
        self,
        template_path: str,
        *,
        user_id: str | None = None,
        exclude_user_id: str | None = None,
        limit: int = 20,
    ) -> TemplateProfile | None:
        """查找与输入模板文件等价的已提取档案"""
        normalized_path = self._normalize_template_path(template_path)
        if not normalized_path:
            return None

        input_path = Path(normalized_path)
        if not input_path.exists() or not input_path.is_file():
            return None

        if user_id:
            exact_match = await self.get_by_source(normalized_path, user_id)
        else:
            exact_match = await self.get_any_by_source(
                normalized_path,
                exclude_user_id=exclude_user_id,
            )
        if exact_match:
            return exact_match

        input_fingerprint = self._get_file_fingerprint(input_path)
        if not input_fingerprint:
            return None

        candidates = await self.list_by_name(
            input_path.stem,
            user_id=user_id,
            exclude_user_id=exclude_user_id,
            limit=limit,
        )
        for candidate in candidates:
            if self._profile_matches_fingerprint(candidate, input_fingerprint):
                return candidate
        return None

    async def clone_profile_to_user(
        self,
        profile: TemplateProfile,
        user_id: str,
        narrative_style: str | None = None,
    ) -> TemplateProfile:
        """复制一个已存在的模板档案到新的 user_id"""
        clone_data = deepcopy(profile.to_dict())
        now = datetime.now().isoformat()
        clone_data["id"] = str(uuid.uuid4())
        clone_data["user_id"] = user_id
        clone_data["created_at"] = now
        clone_data["updated_at"] = now
        clone_data["status"] = "active"

        if narrative_style is not None:
            content_patterns = clone_data.get("content_patterns", {}) or {}
            content_patterns["narrative_style"] = narrative_style
            clone_data["content_patterns"] = content_patterns

        cloned_profile = TemplateProfile.from_dict(clone_data)
        stored_id = await self.add(cloned_profile)
        if stored_id:
            cloned_profile.id = stored_id
        return cloned_profile

    async def list_by_user(self, user_id: str, limit: int = 20) -> list[TemplateProfile]:
        """获取用户的所有模板档案"""
        rows = await self.db.query(
            f"""SELECT * FROM {self.TABLE}
                WHERE user_id = ? AND skill_type = 'template_layout' AND status = 'active'
                ORDER BY updated_at DESC LIMIT ?""",
            [user_id, limit]
        )
        return [self._row_to_profile(r) for r in rows]

    async def search(
        self,
        query: str,
        user_id: str,
        aspect_ratio: str | None = None,
        limit: int = 5,
    ) -> list[TemplateProfile]:
        """搜索模板档案

        优先使用 FTS5，失败则降级到语义搜索。

        Args:
            query: 搜索关键词
            user_id: 用户 ID
            aspect_ratio: 可选的宽高比过滤
            limit: 返回数量限制

        Returns:
            匹配的 TemplateProfile 列表
        """
        # 尝试 FTS5 搜索
        try:
            sql = f"""
                SELECT s.* FROM {self.TABLE} s
                JOIN {self.FTS_TABLE} fts ON s.id = fts.id
                WHERE fts MATCH ? AND s.user_id = ? AND s.skill_type = 'template_layout' AND s.status = 'active'
            """
            params = [query, user_id]

            if aspect_ratio:
                sql += " AND s.aspect_ratio = ?"
                params.append(aspect_ratio)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            rows = await self.db.query(sql, params)
            if rows:
                return [self._row_to_profile(r) for r in rows]
        except Exception as e:
            logger.debug(f"FTS search failed: {e}")

        # 降级到语义搜索
        return await self._semantic_search(query, user_id, aspect_ratio, limit)

    async def _fallback_search(
        self,
        query: str,
        user_id: str,
        aspect_ratio: str | None,
        limit: int,
    ) -> list[TemplateProfile]:
        """LIKE 降级搜索（已废弃，使用 _semantic_search 替代）"""
        like_pattern = f"%{query}%"
        sql = f"""
            SELECT * FROM {self.TABLE}
            WHERE (name LIKE ? OR description LIKE ? OR keywords LIKE ?)
            AND user_id = ? AND skill_type = 'template_layout' AND status = 'active'
        """
        params = [like_pattern, like_pattern, like_pattern, user_id]

        if aspect_ratio:
            sql += " AND aspect_ratio = ?"
            params.append(aspect_ratio)

        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.query(sql, params)
        return [self._row_to_profile(r) for r in rows]

    async def _semantic_search(
        self,
        query: str,
        user_id: str,
        aspect_ratio: str | None,
        limit: int,
    ) -> list[TemplateProfile]:
        """基于向量相似度的语义搜索

        使用 embedding 计算查询与模板的语义相似度，返回最相关的模板。
        如果 embedding_func 不可用，降级到 LIKE 搜索。
        """
        # 获取所有候选模板
        base_conditions = ["user_id = ?", "skill_type = 'template_layout'", "status = 'active'"]
        base_params = [user_id]

        if aspect_ratio:
            base_conditions.append("aspect_ratio = ?")
            base_params.append(aspect_ratio)

        sql = f"SELECT * FROM {self.TABLE} WHERE {' AND '.join(base_conditions)}"
        rows = await self.db.query(sql, base_params)

        if not rows:
            return []

        # 尝试使用向量搜索
        try:
            from memslides.memory.core.embedding import batch_cosine_similarity

            # 获取 embedding 函数
            embedding_func = self.embedding_func
            if embedding_func is None:
                logger.debug("Embedding function not available, falling back to LIKE search")
                return await self._fallback_search(query, user_id, aspect_ratio, limit)

            # 生成查询向量
            query_embedding = await embedding_func([query])
            query_vec = query_embedding[0]

            # 为每个模板生成搜索文本并计算相似度
            candidates = []
            template_texts = []
            for row in rows:
                # 组合模板的关键信息作为搜索文本
                profile = self._row_to_profile(row)
                search_text = f"{profile.name} {profile.description} {' '.join(profile.keywords)}"
                template_texts.append(search_text)
                candidates.append(profile)

            # 批量生成模板向量
            template_embeddings = await embedding_func(template_texts)

            # 计算相似度
            similarities = batch_cosine_similarity(query_vec, template_embeddings)

            # 按相似度排序
            scored_templates = list(zip(candidates, similarities))
            scored_templates.sort(key=lambda x: x[1], reverse=True)

            # 返回 top-k
            results = [t for t, score in scored_templates[:limit]]
            logger.debug(
                f"Semantic search: query='{query}', top scores={[f'{s:.3f}' for _, s in scored_templates[:limit]]}"
            )
            return results

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}, falling back to LIKE search")
            return await self._fallback_search(query, user_id, aspect_ratio, limit)

    async def delete(self, skill_id: str) -> bool:
        """软删除模板技能"""
        await self.db.execute(
            f"UPDATE {self.TABLE} SET status = 'deprecated', updated_at = ? WHERE id = ?",
            [datetime.now().isoformat(), skill_id]
        )
        return True

    def _row_to_profile(self, row: dict) -> TemplateProfile:
        """将数据库行转换为 TemplateProfile"""
        # 解析 JSON 字段
        design_constraints = DesignConstraints()
        if row.get("design_constraints"):
            try:
                design_constraints = DesignConstraints.from_dict(json.loads(row["design_constraints"]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse design_constraints: {e}")

        content_patterns = ContentPatterns()
        if row.get("content_patterns"):
            try:
                content_patterns = ContentPatterns.from_dict(json.loads(row["content_patterns"]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse content_patterns: {e}")

        # 解析 slide_induction 原始布局数据
        slide_induction = {}
        if row.get("slide_induction"):
            try:
                slide_induction = json.loads(row["slide_induction"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse slide_induction: {e}")

        semantic_model = TemplateSemanticModel()
        if row.get("semantic_model"):
            try:
                semantic_model = TemplateSemanticModel.from_dict(json.loads(row["semantic_model"]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse semantic_model: {e}")

        image_stats = {}
        if row.get("image_stats"):
            try:
                image_stats = json.loads(row["image_stats"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse image_stats: {e}")

        return TemplateProfile(
            id=row.get("id", ""),
            name=row.get("name", ""),
            description=row.get("description", ""),
            template_source=row.get("template_source", ""),
            slide_count=row.get("slide_count", 0),
            aspect_ratio=row.get("aspect_ratio", "16:9"),
            # slide_induction 原始布局数据
            slide_induction=slide_induction,
            semantic_model=semantic_model,
            template_dir=row.get("template_dir", ""),
            image_stats=image_stats,
            # memslides 增量
            design_constraints=design_constraints,
            content_patterns=content_patterns,
            confidence=row.get("confidence", 0.8),
            user_id=row.get("user_id", ""),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            status=row.get("status", "active"),
        )

    def _extract_keywords(self, profile: TemplateProfile) -> list[str]:
        """从档案中提取关键词用于搜索"""
        keywords = [profile.name]

        # 添加布局类型（从 slide_induction）
        si = profile.slide_induction or {}
        functional_keys = si.get("functional_keys", [])
        keywords.extend(functional_keys)
        for key in si.keys():
            if key not in ("functional_keys", "language", "layout_capabilities") and isinstance(si[key], dict):
                # 提取布局类型后缀（:text, :image）
                if ":text" in key:
                    keywords.append("content_slide")
                elif ":image" in key:
                    keywords.append("image_slide")

        # 添加叙事风格
        if profile.content_patterns.narrative_style:
            keywords.append(profile.content_patterns.narrative_style)

        # 添加信息密度
        if profile.content_patterns.info_density:
            keywords.append(profile.content_patterns.info_density)

        return list(set(keywords))

    def _normalize_template_path(self, template_path: str) -> str:
        if not template_path or not str(template_path).strip():
            return ""
        try:
            return str(Path(template_path).expanduser().resolve())
        except Exception:
            return str(Path(template_path).expanduser())

    def _get_file_fingerprint(self, path: Path) -> str:
        try:
            resolved = path.expanduser().resolve()
            stat = resolved.stat()
        except Exception:
            return ""

        cache_key = str(resolved)
        cache_token = (stat.st_size, stat.st_mtime_ns)
        cached = self._fingerprint_cache.get(cache_key)
        if cached and cached[0] == cache_token[0] and cached[1] == cache_token[1]:
            return cached[2]

        hasher = hashlib.sha256()
        try:
            with resolved.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except Exception:
            return ""

        fingerprint = f"{stat.st_size}:{hasher.hexdigest()}"
        self._fingerprint_cache[cache_key] = (
            cache_token[0],
            cache_token[1],
            fingerprint,
        )
        return fingerprint

    def _profile_source_candidates(self, profile: TemplateProfile) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()

        raw_paths: list[str] = []
        if profile.template_source:
            raw_paths.append(profile.template_source)
        if profile.template_dir:
            raw_paths.append(str(Path(profile.template_dir) / "source.pptx"))

        for raw_path in raw_paths:
            normalized = self._normalize_template_path(raw_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(Path(normalized))
        return candidates

    def _profile_matches_fingerprint(
        self,
        profile: TemplateProfile,
        input_fingerprint: str,
    ) -> bool:
        for candidate_path in self._profile_source_candidates(profile):
            if not candidate_path.exists():
                continue
            candidate_fingerprint = self._get_file_fingerprint(candidate_path)
            if candidate_fingerprint and candidate_fingerprint == input_fingerprint:
                return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 14: 基于风格属性的模板搜索
    # ══════════════════════════════════════════════════════════════════════════

    async def search_by_style(
        self,
        narrative_style: str = "",
        info_density: str = "",
        color_tone: str = "",
        user_id: str = "default",
        aspect_ratio: str | None = None,
        limit: int = 5,
    ) -> list[TemplateProfile]:
        """基于风格属性的模板搜索 (Stage 14)

        匹配逻辑:
        1. narrative_style 精确匹配（只要匹配就返回）
        2. 如果没有匹配，降级到关键词向量搜索

        Args:
            narrative_style: 叙事风格 (academic | business | creative | technical | educational)
            info_density: 信息密度 (low | medium | high) - 仅用于关键词搜索
            color_tone: 色调偏好 (warm | cool | neutral | vibrant | dark) - 仅用于关键词搜索
            user_id: 用户 ID
            aspect_ratio: 可选的宽高比过滤
            limit: 返回数量限制

        Returns:
            匹配的 TemplateProfile 列表，按匹配度排序
        """
        # 构建基础查询
        base_conditions = ["user_id = ?", "skill_type = 'template_layout'", "status = 'active'"]
        base_params = [user_id]

        if aspect_ratio:
            base_conditions.append("aspect_ratio = ?")
            base_params.append(aspect_ratio)

        # 只匹配 narrative_style（放宽条件）
        if narrative_style:
            exact_conditions = base_conditions.copy()
            exact_params = base_params.copy()
            exact_conditions.append("json_extract(content_patterns, '$.narrative_style') = ?")
            exact_params.append(narrative_style)

            sql = f"SELECT * FROM {self.TABLE} WHERE {' AND '.join(exact_conditions)} ORDER BY updated_at DESC LIMIT ?"
            exact_params.append(limit)

            try:
                rows = await self.db.query(sql, exact_params)
                if rows:
                    logger.debug(f"Matched {len(rows)} templates by narrative_style={narrative_style}")
                    return [self._row_to_profile(r) for r in rows]
            except Exception as e:
                logger.debug(f"Exact style match failed: {e}")

        # 降级到关键词向量搜索
        keywords = []
        if narrative_style:
            keywords.append(narrative_style)
        if info_density:
            keywords.append(info_density)
        if color_tone:
            keywords.append(color_tone)

        if keywords:
            return await self._semantic_search(" ".join(keywords), user_id, aspect_ratio, limit)

        # 无条件时返回最近使用的模板
        sql = f"SELECT * FROM {self.TABLE} WHERE {' AND '.join(base_conditions)} ORDER BY updated_at DESC LIMIT ?"
        base_params.append(limit)
        rows = await self.db.query(sql, base_params)
        return [self._row_to_profile(r) for r in rows]

    async def mark_used(self, profile_id: str) -> None:
        """更新模板的最近使用时间 (Stage 14: 支持 get_recent 的正确语义)"""
        try:
            now = datetime.now().isoformat()
            await self.db.execute(
                f"UPDATE {self.TABLE} SET last_used_at = ? WHERE id = ?",
                [now, profile_id]
            )
            logger.debug(f"[TEMPLATE] mark_used: {profile_id} at {now}")
        except Exception as e:
            logger.warning(f"[TEMPLATE] mark_used failed: {e}")

    async def get_recent(
        self,
        user_id: str,
        limit: int = 1,
    ) -> list[TemplateProfile]:
        """获取用户最近使用的模板 (Stage 14: 支持 "上次的模板" 查询)

        优先按 last_used_at（实际使用时间）排序，其次按 updated_at（存储时间）兜底。

        Args:
            user_id: 用户 ID
            limit: 返回数量

        Returns:
            最近使用的模板列表
        """
        sql = f"""
            SELECT * FROM {self.TABLE}
            WHERE user_id = ? AND skill_type = 'template_layout' AND status = 'active'
            ORDER BY
                CASE WHEN last_used_at IS NOT NULL AND last_used_at != '' THEN last_used_at ELSE updated_at END DESC
            LIMIT ?
        """
        rows = await self.db.query(sql, [user_id, limit])
        return [self._row_to_profile(r) for r in rows]

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 14: Template Usage History (Scenario-based Matching)
    # ══════════════════════════════════════════════════════════════════════════

    async def add_usage_record(self, record: "TemplateUsageRecord") -> str:
        """添加模板使用记录到历史表 (Stage 14: 支持场景匹配)

        Args:
            record: TemplateUsageRecord 对象

        Returns:
            记录 ID
        """
        # 序列化 embedding 为 JSON 字符串
        embedding_json = None
        if record.user_message_embedding is not None:
            embedding_json = json.dumps(record.user_message_embedding, ensure_ascii=False)

        data = {
            "id": record.id,
            "user_id": record.user_id,
            "template_id": record.template_id,
            "template_name": record.template_name,
            "user_message": record.user_message[:500],  # 截断至 500 字符
            "user_message_embedding": embedding_json,
            "intent": record.intent,
            "memory_intent": record.memory_intent,
            "aspect_ratio": record.aspect_ratio,
            "has_attachment": 1 if record.has_attachment else 0,
            "attachment_type": record.attachment_type,
            "success": 1 if record.success else 0,
            "created_at": record.created_at,
        }

        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))

        try:
            await self.db.execute(
                f"INSERT INTO template_usage_history ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            logger.debug(f"[TEMPLATE] Added usage record: {record.id} (template={record.template_name}, intent={record.intent})")
            return record.id
        except Exception as e:
            logger.warning(f"[TEMPLATE] Failed to add usage record: {e}")
            return record.id

    async def get_usage_history(
        self,
        user_id: str,
        limit: int = 20,
        success_only: bool = True,
        memory_intent: str = "",
        aspect_ratio: str = "",
    ) -> list["TemplateUsageRecord"]:
        """获取用户的模板使用历史 (Stage 14: 支持场景匹配)

        Args:
            user_id: 用户 ID
            limit: 返回数量限制
            success_only: 是否只返回成功的记录

        Returns:
            TemplateUsageRecord 列表，按 created_at DESC 排序
        """
        sql = "SELECT * FROM template_usage_history WHERE user_id = ?"
        params = [user_id]

        if success_only:
            sql += " AND success = 1"

        if memory_intent:
            sql += " AND memory_intent = ?"
            params.append(memory_intent)

        if aspect_ratio:
            sql += " AND aspect_ratio = ?"
            params.append(aspect_ratio)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            rows = await self.db.query(sql, params)
            return [self._row_to_usage_record(r) for r in rows]
        except Exception as e:
            logger.warning(f"[TEMPLATE] Failed to get usage history: {e}")
            return []

    def _row_to_usage_record(self, row: dict) -> "TemplateUsageRecord":
        """将数据库行转换为 TemplateUsageRecord"""
        from memslides.memory.core.models import TemplateUsageRecord

        # 反序列化 embedding
        embedding = None
        if row.get("user_message_embedding"):
            try:
                embedding = json.loads(row["user_message_embedding"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse user_message_embedding: {e}")

        return TemplateUsageRecord(
            id=row.get("id", ""),
            user_id=row.get("user_id", ""),
            template_id=row.get("template_id", ""),
            template_name=row.get("template_name", ""),
            user_message=row.get("user_message", ""),
            user_message_embedding=embedding,
            intent=row.get("intent", "explicit"),
            memory_intent=row.get("memory_intent", ""),
            aspect_ratio=row.get("aspect_ratio", ""),
            has_attachment=bool(row.get("has_attachment", 0)),
            attachment_type=row.get("attachment_type", ""),
            success=bool(row.get("success", 1)),
            created_at=row.get("created_at", ""),
        )
