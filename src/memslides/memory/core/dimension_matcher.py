"""Implementation details."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .embedding import batch_cosine_similarity

if TYPE_CHECKING:
    from .embedding import EmbeddingFunc
    from .models import UserProfile

logger = logging.getLogger(__name__)



DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "theme": (
        "主题偏好：颜色、配色方案、色彩搭配、主题色、背景色、强调色、"
        "字体、字号、font、color theme、palette、accent color、"
        "background style、dark mode、light mode、色调、暖色、冷色"
    ),
    "visual": (
        "视觉偏好：图片风格、图表类型、图标使用、动画效果、插图、"
        "diagram、chart、icon、illustration、animation、视觉元素、"
        "图形设计、数据可视化、配图、截图、照片"
    ),
    "layout": (
        "布局偏好：排版、对齐方式、间距、留白、内容密度、分栏、"
        "layout、alignment、spacing、margin、padding、grid、column、"
        "页面结构、区域划分、元素位置"
    ),
    "content": (
        "内容偏好：文字风格、文本密度、标题长度、要点格式、语言风格、"
        "bullet point、text density、title、paragraph、简洁、详细、"
        "摘要、正文、副标题、关键词、段落"
    ),
    "template": (
        "模板偏好：PPT模板选择、母版、slide master、主题模板、"
        "template selection、模板风格、避免的模板、偏好的模板"
    ),
}


SIMILARITY_THRESHOLD = 0.35


class DimensionMatcher:
    """Implementation details."""

    def __init__(self, embedding_func: EmbeddingFunc | None = None):
        self._embed = embedding_func
        self._dim_names: list[str] = []
        self._dim_matrix: np.ndarray | None = None  # shape=(n_dims, embed_dim)
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        """Implementation details."""
        if self._initialized:
            return self._dim_matrix is not None
        self._initialized = True

        if self._embed is None:
            return False

        try:
            self._dim_names = list(DIMENSION_DESCRIPTIONS.keys())
            texts = [DIMENSION_DESCRIPTIONS[d] for d in self._dim_names]
            self._dim_matrix = await self._embed(texts)
            logger.info(
                "DimensionMatcher initialized: %d dims, embed_dim=%d",
                len(self._dim_names), self._dim_matrix.shape[1],
            )
            return True
        except Exception as e:
            logger.warning("DimensionMatcher embedding init failed: %s", e)
            self._dim_matrix = None
            return False

    async def match(
        self,
        text: str,
        profile: UserProfile | None = None,
    ) -> str:
        """Implementation details."""

        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match(text)
            except Exception as e:
                logger.debug("Vector match failed, fallback to keywords: %s", e)


        if profile is not None:
            return profile.match_dimension(text)
        return "general"
    async def match_top_k(
        self,
        text: str,
        top_k: int = 3,
        profile: UserProfile | None = None,
    ) -> list[str]:
        """Implementation details."""
        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match_top_k(text, top_k)
            except Exception as e:
                logger.debug("Vector match_top_k failed, fallback to keywords: %s", e)


        if profile is not None:
            dim = profile.match_dimension(text)
            if dim != "general":
                return [dim]
        return []

    async def _vector_match_top_k(self, text: str, top_k: int) -> list[str]:
        """Implementation details."""
        query_vec = await self._embed([text])  # shape=(1, dim)
        query = query_vec[0]                    # shape=(dim,)
        sims = batch_cosine_similarity(query, self._dim_matrix)  # shape=(n,)


        sorted_indices = np.argsort(sims)[::-1]
        results: list[str] = []
        for idx in sorted_indices:
            score = float(sims[idx])
            if score < SIMILARITY_THRESHOLD:
                break
            results.append(self._dim_names[idx])
            if len(results) >= top_k:
                break

        logger.debug(
            "DimensionMatcher top_k: '%s' => %s",
            text[:50], [(d, f"{float(sims[self._dim_names.index(d)]):.3f}") for d in results],
        )
        return results

    async def _vector_match(self, text: str) -> str:
        """Implementation details."""
        query_vec = await self._embed([text])  # shape=(1, dim)
        query = query_vec[0]                    # shape=(dim,)
        sims = batch_cosine_similarity(query, self._dim_matrix)  # shape=(n,)

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < SIMILARITY_THRESHOLD:
            logger.debug(
                "DimensionMatcher: '%s' best=%.3f (%s) < threshold, => general",
                text[:50], best_score, self._dim_names[best_idx],
            )
            return "general"

        dim_name = self._dim_names[best_idx]
        logger.debug(
            "DimensionMatcher: '%s' => %s (score=%.3f)",
            text[:50], dim_name, best_score,
        )
        return dim_name

    async def match_batch(
        self,
        texts: list[str],
        profile: UserProfile | None = None,
    ) -> list[str]:
        """Implementation details."""
        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match_batch(texts)
            except Exception as e:
                logger.debug("Batch vector match failed: %s", e)


        if profile is not None:
            return [profile.match_dimension(t) for t in texts]
        return ["general"] * len(texts)

    async def _vector_match_batch(self, texts: list[str]) -> list[str]:
        """Implementation details."""
        query_matrix = await self._embed(texts)  # shape=(n_texts, dim)
        results: list[str] = []
        for i in range(len(texts)):
            sims = batch_cosine_similarity(query_matrix[i], self._dim_matrix)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score < SIMILARITY_THRESHOLD:
                results.append("general")
            else:
                results.append(self._dim_names[best_idx])
        return results
