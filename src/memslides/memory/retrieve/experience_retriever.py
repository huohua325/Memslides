"""ExperienceTraceRetriever — ChromaDB + BM25 混合检索 ExperienceTrace

学习 ProceduralRuleRetriever 的设计:
- 写入时双路索引 (ChromaDB + BM25)
- 检索时并行查询 + RRF 融合
- 支持 search_method: "hybrid" | "vector" | "bm25"

索引字段: task_description + lessons_learned
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .strategies import RetrievalResult, rrf_merge

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed, vector search unavailable")

try:
    import bm25s

    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    logger.warning("bm25s not installed, BM25 search unavailable")


class ExperienceTraceRetriever:
    """ExperienceTrace 混合检索器 — ChromaDB + BM25 + RRF

    学习 ProceduralRuleRetriever 架构:
    - 写入时双路索引
    - 检索时并行 + RRF 融合
    - 空库优雅处理

    索引文档格式: "{task_description}\n{lessons_learned}"
    """

    _COLLECTION_NAME = "dp_experience_traces"
    _BM25_DATA_FILENAME = "experience_bm25_data.json"

    def __init__(
        self,
        db: Any,
        embedding_func: Callable,
        persist_dir: str,
        rrf_k: int = 60,
    ):
        """
        Args:
            db: AsyncDatabase 实例 (用于获取完整 trace)
            embedding_func: async (list[str]) -> np.ndarray
            persist_dir: 持久化目录
            rrf_k: RRF 融合参数 (默认 60)
        """
        self._db = db
        self._embed = embedding_func
        self._persist_dir = persist_dir
        self._rrf_k = rrf_k

        # ChromaDB
        self._chroma_client: Any = None
        self._collection: Any = None
        if CHROMADB_AVAILABLE:
            chroma_path = os.path.join(persist_dir, "chroma_experiences")
            os.makedirs(chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._COLLECTION_NAME,
                metadata={"type": "experience_traces"},
            )

        # BM25
        self._bm25_retriever: Any = None
        self._bm25_data: List[Dict] = []
        self._bm25_corpus_tokens: List[List[str]] = []
        self._tokenizer: Any = None
        if BM25S_AVAILABLE:
            try:
                from .bm25_search import BilingualTokenizer
                self._tokenizer = BilingualTokenizer()
            except ImportError:
                pass
            self._load_bm25_index()

    async def close(self) -> None:
        self._collection = None
        clear_cache = getattr(self._chroma_client, "clear_system_cache", None)
        if clear_cache:
            clear_cache()
        self._chroma_client = None
        self._bm25_retriever = None
        self._bm25_data.clear()
        self._bm25_corpus_tokens.clear()

    # ═══════════════════════════════════════════════
    # 写入接口
    # ═══════════════════════════════════════════════

    async def add_trace(
        self,
        trace_id: str,
        task_description: str,
        lessons_learned: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """双路写入 (ChromaDB + BM25)

        Args:
            trace_id: ExperienceTrace.id
            task_description: 任务描述
            lessons_learned: 学到的经验
            metadata: 元数据 (session_id, outcome, confidence, applicable_scenarios)

        Returns:
            True=新写入, False=已存在或失败
        """
        document = self._format_document(task_description, lessons_learned)

        # 1. ChromaDB 写入
        chroma_added = False
        collection = self._collection
        if collection is not None:
            try:
                existing = collection.get(ids=[trace_id])
                if not existing["ids"]:
                    embeddings = await self._embed([document])
                    embedding = embeddings[0].tolist()
                    collection.add(
                        ids=[trace_id],
                        documents=[document],
                        metadatas=[{
                            "trace_id": trace_id,
                            "task": task_description[:500],
                            "outcome": metadata.get("outcome", ""),
                            "applicable_scenarios": metadata.get("applicable_scenarios", ""),
                        }],
                        embeddings=[embedding],
                    )
                    chroma_added = True
                    logger.debug("ChromaDB: added trace %s", trace_id[:8])
            except Exception as e:
                logger.warning("ChromaDB write failed: %s", e)

        # 2. BM25 写入
        bm25_added = self._add_to_bm25(trace_id, document, metadata)

        return chroma_added or bm25_added

    def _add_to_bm25(self, trace_id: str, document: str, metadata: Dict) -> bool:
        """BM25 增量写入"""
        if not BM25S_AVAILABLE or self._tokenizer is None:
            return False

        # 去重检查
        for existing in self._bm25_data:
            if existing.get("id") == trace_id:
                return False

        tokens = self._tokenizer.tokenize([document])[0]
        if not tokens:
            return False

        self._bm25_data.append({
            "id": trace_id,
            "document": document,
            **metadata,
        })
        self._bm25_corpus_tokens.append(tokens)

        # 重建 BM25 索引
        self._bm25_retriever = bm25s.BM25(method="lucene")
        self._bm25_retriever.index(self._bm25_corpus_tokens)

        # 持久化
        self._save_bm25_index()
        logger.debug("BM25: added trace %s", trace_id[:8])
        return True

    # ═══════════════════════════════════════════════
    # 检索接口
    # ═══════════════════════════════════════════════

    async def search(
        self,
        query: str,
        user_id: str = "",
        top_k: int = 5,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """混合检索 ExperienceTrace

        Args:
            query: 用户查询 (通常是用户任务描述)
            user_id: 用户 ID (暂未使用，预留)
            top_k: 返回结果数
            method: "hybrid" | "vector" | "bm25"

        Returns:
            List[RetrievalResult]
        """
        if method == "vector":
            return await self._vector_search(query, top_k)
        elif method == "bm25":
            return await self._bm25_search(query, top_k)
        else:  # hybrid
            return await self._hybrid_search(query, top_k)

    async def _hybrid_search(
        self, query: str, top_k: int,
    ) -> List[RetrievalResult]:
        """并行 ChromaDB + BM25 → RRF 融合"""
        tasks = []
        tasks.append(self._vector_search(query, top_k * 2))
        tasks.append(self._bm25_search(query, top_k * 2))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        result_lists: List[List[RetrievalResult]] = []
        for r in raw_results:
            if isinstance(r, list) and r:
                result_lists.append(r)

        if not result_lists:
            return []
        if len(result_lists) == 1:
            return result_lists[0][:top_k]

        return self._rrf_merge(result_lists, top_k)

    async def _vector_search(
        self, query: str, top_k: int,
    ) -> List[RetrievalResult]:
        """ChromaDB 向量检索"""
        collection = self._collection
        if collection is None:
            return []

        try:
            total = collection.count()
            if total == 0:
                return []

            embeddings = await self._embed([query])
            query_embedding = embeddings[0].tolist()

            n_results = min(top_k, total)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )

            search_results: List[RetrievalResult] = []
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - distance
                search_results.append(
                    RetrievalResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results.get("documents") else "",
                        score=score,
                        source="chroma_vector",
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    )
                )
            return search_results
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

    async def _bm25_search(
        self, query: str, top_k: int,
    ) -> List[RetrievalResult]:
        """BM25 关键词检索"""
        if not BM25S_AVAILABLE or self._bm25_retriever is None or not self._bm25_data:
            return []

        try:
            query_tokens = self._tokenizer.tokenize([query])
            results_obj, scores_obj = self._bm25_retriever.retrieve(
                query_tokens, k=min(top_k, len(self._bm25_data))
            )

            search_results: List[RetrievalResult] = []
            indices = results_obj[0]
            scores = scores_obj[0]

            for i, idx in enumerate(indices):
                idx = int(idx)
                score = float(scores[i])
                if score <= 0 or idx < 0 or idx >= len(self._bm25_data):
                    continue
                item = self._bm25_data[idx]
                search_results.append(
                    RetrievalResult(
                        id=item["id"],
                        content=item["document"],
                        score=score,
                        source="bm25_keyword",
                        metadata=item,
                    )
                )
            return search_results
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    def _rrf_merge(
        self,
        result_lists: List[List[RetrievalResult]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """RRF 融合 — 委托给共享 rrf_merge()"""
        return rrf_merge(result_lists, top_k, self._rrf_k)

    # ═══════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════

    @staticmethod
    def _format_document(task_description: str, lessons_learned: str) -> str:
        """格式化 Trace 为检索文档"""
        return f"{task_description}\n{lessons_learned}"

    def _load_bm25_index(self):
        """从磁盘加载 BM25 索引"""
        if not BM25S_AVAILABLE or not self._persist_dir:
            return

        bm25_path = Path(self._persist_dir) / "bm25_experiences"
        data_path = bm25_path / self._BM25_DATA_FILENAME

        if not data_path.exists():
            return

        try:
            with open(data_path, encoding="utf-8") as f:
                self._bm25_data = json.load(f)

            if self._bm25_data and self._tokenizer:
                self._bm25_corpus_tokens = [
                    self._tokenizer.tokenize([d["document"]])[0]
                    for d in self._bm25_data
                ]
                self._bm25_retriever = bm25s.BM25(method="lucene")
                self._bm25_retriever.index(self._bm25_corpus_tokens)
                logger.info("Loaded BM25 index with %d traces", len(self._bm25_data))
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)

    def _save_bm25_index(self):
        """持久化 BM25 索引"""
        if not self._persist_dir:
            return

        try:
            bm25_path = Path(self._persist_dir) / "bm25_experiences"
            os.makedirs(bm25_path, exist_ok=True)

            data_path = bm25_path / self._BM25_DATA_FILENAME
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self._bm25_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save BM25 index: %s", e)

    async def rebuild_index(self):
        """从数据库重建全部索引

        用于初始化或索引损坏时修复。
        使用批量操作避免 O(n²) 逐条重建，并通过 asyncio.to_thread
        将同步阻塞操作移出 event loop，防止阻塞 UI。
        """
        if not self._db:
            return

        try:
            rows = await self._db.query(
                "SELECT id, task_description, lessons_learned, session_id, "
                "final_outcome as outcome, confidence, applicable_scenarios "
                "FROM experience_traces "
                "WHERE COALESCE(status, 'active') = 'active' "
                "ORDER BY created_at DESC LIMIT 1000"
            )
            logger.info("Rebuilding index for %d traces", len(rows))
            if not rows:
                return

            # Prepare documents and metadata
            docs = []
            ids = []
            metas = []
            for row in rows:
                task = row.get("task_description", "")
                lessons = row.get("lessons_learned", "")
                doc = self._format_document(task, lessons)
                docs.append(doc)
                ids.append(row["id"])
                metas.append({
                    "session_id": row.get("session_id", ""),
                    "outcome": row.get("outcome", ""),
                    "confidence": row.get("confidence", 0.5),
                    "applicable_scenarios": row.get("applicable_scenarios", ""),
                })

            # 1. ChromaDB: batch embedding + add (in thread to avoid blocking)
            collection = self._collection
            if collection is not None:
                try:
                    existing = collection.get(ids=ids)
                    existing_set = set(existing["ids"]) if existing["ids"] else set()
                    new_indices = [i for i, tid in enumerate(ids) if tid not in existing_set]

                    if new_indices:
                        new_docs = [docs[i] for i in new_indices]
                        new_ids = [ids[i] for i in new_indices]
                        new_metas = [
                            {
                                "trace_id": ids[i],
                                "task": rows[i].get("task_description", "")[:500],
                                "outcome": metas[i].get("outcome", ""),
                                "applicable_scenarios": metas[i].get("applicable_scenarios", ""),
                            }
                            for i in new_indices
                        ]

                        # Batch embed in chunks to avoid OOM
                        _EMBED_BATCH = 32
                        all_embeddings = []
                        for start in range(0, len(new_docs), _EMBED_BATCH):
                            batch = new_docs[start : start + _EMBED_BATCH]
                            embs = await self._embed(batch)
                            all_embeddings.extend(e.tolist() for e in embs)
                            # Yield control between batches
                            await asyncio.sleep(0)

                        # ChromaDB add is synchronous — run in thread
                        def _chroma_add():
                            collection.add(
                                ids=new_ids,
                                documents=new_docs,
                                metadatas=new_metas,
                                embeddings=all_embeddings,
                            )
                        await asyncio.to_thread(_chroma_add)
                        logger.info("ChromaDB: batch added %d traces", len(new_ids))
                except Exception as e:
                    logger.warning("ChromaDB batch write failed: %s", e)

            # 2. BM25: batch build index once (in thread)
            if BM25S_AVAILABLE and self._tokenizer is not None:
                try:
                    def _rebuild_bm25():
                        existing_ids = {d.get("id") for d in self._bm25_data}
                        new_data = []
                        new_tokens = []
                        for i, doc in enumerate(docs):
                            if ids[i] in existing_ids:
                                continue
                            tokens = self._tokenizer.tokenize([doc])[0]
                            if not tokens:
                                continue
                            new_data.append({"id": ids[i], "document": doc, **metas[i]})
                            new_tokens.append(tokens)

                        if new_data:
                            self._bm25_data.extend(new_data)
                            self._bm25_corpus_tokens.extend(new_tokens)

                        # Build BM25 index ONCE
                        if self._bm25_corpus_tokens:
                            self._bm25_retriever = bm25s.BM25(method="lucene")
                            self._bm25_retriever.index(self._bm25_corpus_tokens)
                        self._save_bm25_index()
                        return len(new_data)

                    added = await asyncio.to_thread(_rebuild_bm25)
                    logger.info("BM25: batch added %d traces", added)
                except Exception as e:
                    logger.warning("BM25 batch build failed: %s", e)

            logger.info("Index rebuild complete")
        except Exception as e:
            logger.warning("Index rebuild failed: %s", e)
