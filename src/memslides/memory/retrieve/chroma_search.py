"""Implementation details."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List

from .strategies import RetrievalResult

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed, ChromaVectorSearch unavailable. pip install chromadb")


class ChromaVectorSearch:
    """Implementation details."""

    name = "chroma_vector"

    def __init__(
        self,
        persist_dir: str,
        embedding_func: Callable,  # async (list[str]) -> np.ndarray
        collection_prefix: str = "dp",
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required for ChromaVectorSearch. pip install chromadb")

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._embed = embedding_func
        self._prefix = collection_prefix
        self._collections: Dict[str, Any] = {}  # cache

    async def close(self) -> None:
        self._collections.clear()
        clear_cache = getattr(self._client, "clear_system_cache", None)
        if clear_cache:
            clear_cache()
        self._client = None

    def _get_collection(self, user_id: str, mem_type: str = "episodes"):
        """Implementation details."""
        key = f"{self._prefix}_{user_id}_{mem_type}"
        if key not in self._collections:
            self._collections[key] = self._client.get_or_create_collection(
                name=key,
                metadata={"user_id": user_id, "type": mem_type},
            )
        return self._collections[key]

    async def add_episode(
        self,
        user_id: str,
        episode_id: str,
        document: str,
        metadata: dict,
        mem_type: str = "episodes",
    ) -> bool:
        """Implementation details."""
        collection = self._get_collection(user_id, mem_type)


        existing = collection.get(ids=[episode_id])
        if existing["ids"]:
            return False


        embeddings = await self._embed([document])
        embedding = embeddings[0].tolist()

        collection.add(
            ids=[episode_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return True

    async def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "default",
        mem_type: str = "episodes",
    ) -> List[RetrievalResult]:
        """Implementation details."""
        collection = self._get_collection(user_id, mem_type)

        if collection.count() == 0:
            logger.debug("ChromaDB collection empty for user %s", user_id)
            return []

        embeddings = await self._embed([query])
        query_embedding = embeddings[0].tolist()

        n_results = min(top_k, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        search_results: List[RetrievalResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances; convert to similarity
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
