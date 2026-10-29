"""ChainStore — LTM 工具链存储（原始链数据 + 提炼经验）。

重构版：支持同一签名下多条经验（通过 subkey 区分）+ keyword_embedding 持久化。

按链签名 (chain_signature) 为键，同时存储：
- 原始链数据 (chain_raw_data)：完整 cycle 数据
- 提炼经验 (chain_experiences)：lesson / anti_pattern / applicable_when / keywords / embedding
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from ...utils.constants import MAX_RAW_CHAINS_PER_KEY
from ..core.db import DatabaseBackend
from ..core.models import ChainExperience, ToolChain

logger = logging.getLogger(__name__)

RAW_TABLE = "chain_raw_data"
EXP_TABLE = "chain_experiences"


def _embedding_json(value) -> str:
    if value is None:
        return ""
    try:
        if len(value) == 0:
            return ""
    except TypeError:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value)


class ChainStore:
    """LTM 工具链存储 — 原始链数据 + 提炼经验，按签名为键。"""

    def __init__(
        self,
        db: DatabaseBackend,
        max_raw_chains_per_key: int = MAX_RAW_CHAINS_PER_KEY,
    ) -> None:
        self.db = db
        self.max_raw_chains_per_key = max_raw_chains_per_key

    @staticmethod
    def _row_to_experience(row: dict) -> ChainExperience:
        data = json.loads(row["experience_json"])
        exp = ChainExperience.from_dict(data)
        kw_emb_json = row.get("keyword_embedding_json", "")
        if kw_emb_json:
            try:
                exp.keyword_embedding = json.loads(kw_emb_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return exp

    @staticmethod
    def _merge_unique(values: list[str], *extras: str) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for value in [*values, *extras]:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged

    @staticmethod
    def _compact_text(text: str, limit: int = 180) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized

        for sep in ("\n", "。", ". ", "；", "; "):
            head = normalized.split(sep, 1)[0].strip()
            if head and len(head) <= limit:
                return head
        return normalized[: limit - 1].rstrip() + "…"

    def _build_inject_summary(self, experience: ChainExperience) -> str:
        if experience.inject_summary:
            return self._compact_text(experience.inject_summary)

        if experience.lesson:
            return self._compact_text(experience.lesson)
        if experience.anti_pattern:
            return self._compact_text(experience.anti_pattern)
        return self._compact_text(experience.applicable_when)

    def _prepare_experience_for_storage(
        self,
        experience: ChainExperience,
        existing: ChainExperience | None,
        user_id: str,
        job_id: str,
    ) -> ChainExperience:
        if not experience.subkey:
            experience.subkey = uuid.uuid4().hex[:8]

        if existing and existing.timestamp:
            experience.timestamp = existing.timestamp

        source_chain_ids = list(dict.fromkeys(experience.source_chain_ids))
        experience.source_chain_ids = source_chain_ids
        support_count = len(source_chain_ids) or 1
        if existing and existing.support_count:
            support_count = max(support_count, existing.support_count)
        experience.support_count = max(1, support_count)
        experience.merge_count = max(
            int(existing.merge_count) if existing else 0,
            experience.support_count - 1,
        )
        existing_users = existing.source_users if existing else []
        existing_jobs = existing.source_job_ids if existing else []
        experience.source_users = self._merge_unique(existing_users, *experience.source_users, user_id)
        experience.source_job_ids = self._merge_unique(existing_jobs, *experience.source_job_ids, job_id)
        experience.inject_summary = self._build_inject_summary(experience)
        return experience

    # ── 原始链数据 ──

    async def save_raw_chains(
        self,
        chain_signature: str,
        chains: list[ToolChain],
        user_id: str,
        job_id: str,
    ) -> None:
        """追加原始链数据，超出 max_raw_chains_per_key 淘汰最早的。"""
        try:
            for chain in chains:
                row = {
                    "id": uuid.uuid4().hex,
                    "chain_signature": chain_signature,
                    "user_id": user_id,
                    "tool_chain_json": json.dumps(chain.to_dict(), ensure_ascii=False),
                    "created_at": datetime.now().isoformat(),
                    "job_id": job_id,
                }
                await self.db.insert(RAW_TABLE, row)
            await self._evict_oldest(chain_signature, user_id)
        except Exception:
            logger.exception(
                "Failed to save raw chains for signature=%s", chain_signature,
            )

    async def _evict_oldest(self, chain_signature: str, user_id: str) -> None:
        """删除超出上限的最早记录。"""
        rows = await self.db.query(
            f"SELECT id FROM {RAW_TABLE} "
            "WHERE chain_signature = ? AND user_id = ? "
            "ORDER BY rowid DESC",
            (chain_signature, user_id),
        )
        if len(rows) > self.max_raw_chains_per_key:
            ids_to_delete = [r["id"] for r in rows[self.max_raw_chains_per_key:]]
            placeholders = ", ".join("?" for _ in ids_to_delete)
            await self.db.execute(
                f"DELETE FROM {RAW_TABLE} WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
            )

    async def get_raw_chains(
        self,
        chain_signature: str,
        user_id: str,
    ) -> list[ToolChain]:
        """获取某签名下的原始链数据。"""
        try:
            rows = await self.db.query(
                f"SELECT tool_chain_json FROM {RAW_TABLE} "
                "WHERE chain_signature = ? AND user_id = ? "
                "ORDER BY created_at",
                (chain_signature, user_id),
            )
            results: list[ToolChain] = []
            for row in rows:
                data = json.loads(row["tool_chain_json"])
                results.append(ToolChain.from_dict(data))
            return results
        except Exception:
            logger.exception(
                "Failed to get raw chains for signature=%s", chain_signature,
            )
            return []

    # ── 提炼经验（多条目存储）──

    async def save_experience(
        self,
        chain_signature: str,
        experience: ChainExperience,
        user_id: str,
        job_id: str = "",
    ) -> None:
        """保存提炼经验，按 (chain_signature, subkey) 全局 upsert。"""
        try:
            now = datetime.now().isoformat()
            subkey = experience.subkey or uuid.uuid4().hex[:8]
            experience.subkey = subkey

            existing_row = await self.db.query_one(
                f"SELECT id, user_id, experience_json FROM {EXP_TABLE} "
                "WHERE chain_signature = ? AND subkey = ?",
                (chain_signature, subkey),
            )
            existing_exp = None
            if existing_row:
                try:
                    existing_exp = ChainExperience.from_dict(
                        json.loads(existing_row["experience_json"])
                    )
                except (json.JSONDecodeError, TypeError):
                    existing_exp = None

            prepared = self._prepare_experience_for_storage(
                experience,
                existing_exp,
                user_id,
                job_id,
            )
            exp_json = json.dumps(prepared.to_dict(), ensure_ascii=False)

            # 序列化 keyword_embedding
            kw_emb_json = _embedding_json(prepared.keyword_embedding)

            # 统计该签名下的原始链数量（全局桶级，非按 user_id 分隔）
            count_row = await self.db.query_one(
                f"SELECT COUNT(*) AS cnt FROM {RAW_TABLE} "
                "WHERE chain_signature = ?",
                (chain_signature,),
            )
            raw_count = count_row["cnt"] if count_row else 0

            conn = await self.db._check_and_reconnect()
            row_id = existing_row["id"] if existing_row else uuid.uuid4().hex
            row_user_id = existing_row["user_id"] if existing_row and existing_row.get("user_id") else user_id
            await conn.execute(
                f"INSERT INTO {EXP_TABLE} "
                "(id, chain_signature, user_id, experience_json, raw_chain_count, "
                "last_updated, subkey, keyword_embedding_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(chain_signature, subkey) DO UPDATE SET "
                "experience_json = excluded.experience_json, "
                "raw_chain_count = excluded.raw_chain_count, "
                "last_updated = excluded.last_updated, "
                "keyword_embedding_json = excluded.keyword_embedding_json",
                (
                    row_id,
                    chain_signature,
                    row_user_id,
                    exp_json,
                    raw_count,
                    now,
                    subkey,
                    kw_emb_json,
                ),
            )
            await conn.commit()
        except Exception:
            logger.exception(
                "Failed to save experience for signature=%s", chain_signature,
            )
            raise

    async def get_experience(
        self,
        chain_signature: str,
    ) -> list[ChainExperience]:
        """按签名精确获取所有提炼经验（多条）。"""
        try:
            rows = await self.db.query(
                f"SELECT experience_json, keyword_embedding_json FROM {EXP_TABLE} "
                "WHERE chain_signature = ? "
                "ORDER BY last_updated DESC",
                (chain_signature,),
            )
            return [self._row_to_experience(row) for row in rows]
        except Exception:
            logger.exception(
                "Failed to get experience for signature=%s", chain_signature,
            )
            return []

    async def delete_experience_by_subkey(
        self,
        chain_signature: str,
        user_id: str,
        subkey: str,
    ) -> None:
        """按 (chain_signature, subkey) 精确删除单个条目。user_id 仅保留兼容。"""
        try:
            await self.db.execute(
                f"DELETE FROM {EXP_TABLE} "
                "WHERE chain_signature = ? AND subkey = ?",
                (chain_signature, subkey),
            )
        except Exception:
            logger.exception(
                "Failed to delete experience for signature=%s subkey=%s",
                chain_signature, subkey,
            )

    async def query_experiences_by_tool_with_embeddings(
        self,
        tool_name: str,
        user_id: str,
        limit: int = 20,
    ) -> list[tuple[ChainExperience, list[float] | None]]:
        """按 tool_name 匹配链签名，返回候选经验 + keyword_embedding。

        chain_experiences 是工具通用知识（非用户特定），不按 user_id 过滤。
        user_id 参数保留仅为向后兼容。
        """
        try:
            rows = await self.db.query(
                f"SELECT experience_json, keyword_embedding_json FROM {EXP_TABLE} "
                "WHERE chain_signature LIKE ? "
                "ORDER BY json_extract(experience_json, '$.support_count') DESC, "
                "json_extract(experience_json, '$.confidence') DESC, "
                "last_updated DESC "
                "LIMIT ?",
                (f"%{tool_name}%", limit),
            )
            results: list[tuple[ChainExperience, list[float] | None]] = []
            for row in rows:
                exp = self._row_to_experience(row)
                embedding = exp.keyword_embedding
                exp.keyword_embedding = embedding
                results.append((exp, embedding))
            return results
        except Exception:
            logger.exception(
                "Failed to query experiences by tool=%s", tool_name,
            )
            return []

    async def query_experiences_by_tool(
        self,
        tool_name: str,
        user_id: str,
        limit: int = 10,
    ) -> list[ChainExperience]:
        """按 tool_name 匹配链签名，返回相关经验（向后兼容接口）。

        chain_experiences 是工具通用知识（非用户特定），不按 user_id 过滤。
        user_id 参数保留仅为向后兼容。
        """
        try:
            rows = await self.db.query(
                f"SELECT experience_json FROM {EXP_TABLE} "
                "WHERE chain_signature LIKE ? "
                "ORDER BY json_extract(experience_json, '$.support_count') DESC, "
                "json_extract(experience_json, '$.confidence') DESC, "
                "last_updated DESC "
                "LIMIT ?",
                (f"%{tool_name}%", limit),
            )
            results: list[ChainExperience] = []
            for row in rows:
                data = json.loads(row["experience_json"])
                results.append(ChainExperience.from_dict(data))
            return results
        except Exception:
            logger.exception(
                "Failed to query experiences by tool=%s", tool_name,
            )
            return []

    async def clear_experiences(
        self,
        chain_signature: str,
        user_id: str,
    ) -> None:
        """删除某签名下的所有提炼经验。user_id 仅保留兼容。"""
        try:
            await self.db.execute(
                f"DELETE FROM {EXP_TABLE} "
                "WHERE chain_signature = ?",
                (chain_signature,),
            )
        except Exception:
            logger.exception(
                "Failed to clear experiences for signature=%s", chain_signature,
            )
