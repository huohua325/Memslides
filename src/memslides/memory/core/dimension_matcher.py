"""DimensionMatcher — 向量化维度匹配器（Stage 12）

用 embedding 向量余弦相似度将偏好文本分配到 UserProfile 最佳维度。
维度描述向量在首次调用时计算并缓存，后续调用直接复用。
无 embedding 时降级为关键词匹配。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .embedding import batch_cosine_similarity

if TYPE_CHECKING:
    from .embedding import EmbeddingFunc
    from .models import UserProfile

logger = logging.getLogger(__name__)

# 每个维度的语义描述文本（用于生成 embedding 向量）
# 包含中英文关键词和典型表述，确保语义覆盖面
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

# 匹配阈值：低于此值的最佳匹配归入 general
SIMILARITY_THRESHOLD = 0.35


class DimensionMatcher:
    """向量化维度匹配器。

    使用流程：
        matcher = DimensionMatcher(embedding_func)
        dim = await matcher.match("把标题字体改成思源黑体")
        # => "theme"

    首次调用时会计算维度描述向量并缓存。
    """

    def __init__(self, embedding_func: EmbeddingFunc | None = None):
        self._embed = embedding_func
        self._dim_names: list[str] = []        # 按索引对应的维度名
        self._dim_matrix: np.ndarray | None = None  # shape=(n_dims, embed_dim)
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        """惰性初始化：计算维度描述向量。返回是否成功。"""
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
        """将文本匹配到最佳维度。

        Args:
            text: 偏好/约束文本
            profile: UserProfile 实例（用于关键词降级）

        Returns:
            维度名称（"theme"|"visual"|"layout"|"content"|"template"|"general"）
        """
        # 优先尝试向量匹配
        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match(text)
            except Exception as e:
                logger.debug("Vector match failed, fallback to keywords: %s", e)

        # 降级：关键词匹配
        if profile is not None:
            return profile.match_dimension(text)
        return "general"
    async def match_top_k(
        self,
        text: str,
        top_k: int = 3,
        profile: UserProfile | None = None,
    ) -> list[str]:
        """将文本匹配到 top-k 相关维度（按相似度降序）。

        Args:
            text: 偏好/约束文本
            top_k: 返回的最大维度数
            profile: UserProfile 实例（用于关键词降级）

        Returns:
            维度名称列表（最多 top_k 个），不含 "general"。
            空列表表示无匹配（调用方应全量兜底）。
        """
        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match_top_k(text, top_k)
            except Exception as e:
                logger.debug("Vector match_top_k failed, fallback to keywords: %s", e)

        # 降级：关键词匹配只能返回单维度
        if profile is not None:
            dim = profile.match_dimension(text)
            if dim != "general":
                return [dim]
        return []

    async def _vector_match_top_k(self, text: str, top_k: int) -> list[str]:
        """向量余弦相似度 top-k 匹配。"""
        query_vec = await self._embed([text])  # shape=(1, dim)
        query = query_vec[0]                    # shape=(dim,)
        sims = batch_cosine_similarity(query, self._dim_matrix)  # shape=(n,)

        # 按相似度降序排列，取 score > threshold 的 top-k
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
        """向量余弦相似度匹配。"""
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
        """批量匹配多条文本到维度。"""
        ok = await self._ensure_initialized()
        if ok:
            try:
                return await self._vector_match_batch(texts)
            except Exception as e:
                logger.debug("Batch vector match failed: %s", e)

        # 降级
        if profile is not None:
            return [profile.match_dimension(t) for t in texts]
        return ["general"] * len(texts)

    async def _vector_match_batch(self, texts: list[str]) -> list[str]:
        """批量向量匹配。"""
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
