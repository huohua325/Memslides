"""BM25 关键词搜索 — 替代 FTS5

技术选型 (基于网络调研 + 参考项目分析):
- bm25s: 比 rank_bm25 更快、支持索引持久化(save/load)、多种 BM25 变体(Lucene/BM25+)
- jieba: 中文分词 (word-level, 远优于 ICU 的 character-level)
- PyStemmer: 英文词干提取 (stemming, 如 "running" → "run")
- 自动语言检测 + 分语言处理管线 (学习 Milvus 2.6 Multi-Language Analyzer 设计)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .strategies import RetrievalResult

logger = logging.getLogger(__name__)

# ── 惰性导入 ──
try:
    import bm25s

    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    logger.warning("bm25s not installed, BM25 search unavailable. pip install bm25s")

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import Stemmer as PyStemmer

    _en_stemmer = PyStemmer.Stemmer("english")
    STEMMER_AVAILABLE = True
except ImportError:
    _en_stemmer = None
    STEMMER_AVAILABLE = False


class BilingualTokenizer:
    """双语分词器 — 学习 Milvus 2.6 Multi-Language Analyzer 设计

    核心思路: 自动检测文本中的中/英文片段 → 分别路由到最佳分词器 → 合并 tokens

    - 中文: jieba.cut_for_search() — 搜索模式，细粒度词级分词
      "把标题字体改大" → ["标题", "字体", "改大"]
    - 英文: PyStemmer 词干提取 + 正则切词
      "changing the font size" → ["chang", "font", "size"]  (stemmed)
    - 混合文本: 自动按字符级别分离中英文，分别处理后拼接
      "把font-size改为24px" → ["font", "size", "改为", "24", "px"]
    """

    def __init__(self):
        self._en_stemmer = _en_stemmer

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        """检测 CJK 统一汉字 (含扩展区)"""
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F
            or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F
        )

    def tokenize(self, texts: list[str]) -> list[list[str]]:
        """批量分词 — 兼容 bm25s.tokenize 接口"""
        return [self._tokenize_single(t) for t in texts]

    def _tokenize_single(self, text: str) -> list[str]:
        """单文本双语分词"""
        text = text.strip()
        if not text:
            return []

        # 按字符级别分离中英文片段
        segments = self._segment_by_script(text)
        tokens: list[str] = []

        for script, content in segments:
            if script == "cjk":
                tokens.extend(self._tokenize_cjk(content))
            else:
                tokens.extend(self._tokenize_latin(content))

        return tokens

    def _segment_by_script(self, text: str) -> list[tuple[str, str]]:
        """将文本按脚本类型分段: [("cjk", "标题字体"), ("latin", "font-size"), ...]"""
        segments: list[tuple[str, str]] = []
        buf: list[str] = []
        current_script: Optional[str] = None

        for ch in text:
            script = "cjk" if self._is_cjk(ch) else "latin"
            if script != current_script and buf:
                segments.append((current_script, "".join(buf)))  # type: ignore[arg-type]
                buf = []
            current_script = script
            buf.append(ch)

        if buf and current_script is not None:
            segments.append((current_script, "".join(buf)))
        return segments

    def _tokenize_cjk(self, text: str) -> list[str]:
        """中文分词: jieba 搜索模式"""
        if JIEBA_AVAILABLE:
            tokens: list[str] = []
            for t in jieba.cut_for_search(text):
                t = t.strip()
                if t and len(t) >= 1:
                    tokens.append(t)
            return tokens
        else:
            # fallback: 逐字切分 (类似 ICU character-level)
            return [ch for ch in text if ch.strip()]

    def _tokenize_latin(self, text: str) -> list[str]:
        """英文分词: 正则切词 → PyStemmer 词干提取"""
        # 提取英文单词和数字
        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
        tokens = [t for t in raw_tokens if len(t) > 1]

        # PyStemmer 词干提取 (running→run, changes→chang, etc.)
        if self._en_stemmer and tokens:
            tokens = self._en_stemmer.stemWords(tokens)

        return tokens


class BM25SearchStrategy:
    """基于 bm25s 的双语 BM25 检索策略

    相比 rank_bm25 的优势 (bm25s):
    - 速度: Numpy/Scipy 优化，比 rank_bm25 快 ~10x
    - 持久化: retriever.save() / BM25.load() — 重启不丢失索引
    - BM25 变体: Lucene (默认, Elasticsearch 同款) / BM25+ / Robertson
    - 内置停用词: bm25s.tokenize(stopwords="en") 支持多语言停用词

    增量更新策略 (学习 nemori):
    - 缓存已分词结果，新增文档只需分词新内容
    - bm25s 不支持真正增量，但通过 re-index 代价很低 (千级文档 <10ms)
    """

    name = "bm25_keyword"

    def __init__(self, persist_dir: Optional[str] = None):
        self._tokenizer = BilingualTokenizer()
        self._persist_dir = persist_dir

        # Per-user state
        self._retrievers: Dict[str, Any] = {}  # user_id → bm25s.BM25
        self._data: Dict[str, List[Dict]] = {}  # user_id → episode dicts
        self._corpus_tokens: Dict[str, list] = {}  # user_id → cached token lists

        # 尝试从磁盘加载已有索引
        if persist_dir:
            self._try_load_indices()

    _DATA_FILENAME = "data_metadata.json"

    def _try_load_indices(self):
        """启动时尝试从磁盘恢复 BM25 索引及 episode 元数据"""
        if not self._persist_dir or not BM25S_AVAILABLE:
            return
        index_dir = Path(self._persist_dir)
        if not index_dir.exists():
            return
        for user_dir in index_dir.iterdir():
            if not user_dir.is_dir():
                continue
            try:
                # load_corpus=False: corpus 已从 data_metadata.json 单独恢复，
                # 若使用 load_corpus=True，retrieve() 会返回文本而非整数索引
                retriever = bm25s.BM25.load(str(user_dir), load_corpus=False)
                user_id = user_dir.name
                self._retrievers[user_id] = retriever

                # 从 data_metadata.json 恢复 episode 元数据（含 UUID 映射）
                metadata_path = user_dir / self._DATA_FILENAME
                if metadata_path.exists():
                    with open(metadata_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self._data[user_id] = data
                    # 重新分词以恢复 corpus tokens（与索引保持一致）
                    self._corpus_tokens[user_id] = [
                        self._tokenizer.tokenize([d["document"]])[0]
                        for d in data
                    ]
                    logger.info(
                        "Loaded BM25 index + %d docs for user %s",
                        len(data), user_id,
                    )
                else:
                    # 旧版索引无元数据文件 — 安全降级，索引作废
                    self._data[user_id] = []
                    self._corpus_tokens[user_id] = []
                    logger.warning(
                        "BM25: no %s for user %s; index stale, will rebuild on next add",
                        self._DATA_FILENAME, user_id,
                    )
            except Exception as e:
                logger.debug("Failed to load BM25 index from %s: %s", user_dir, e)

    def add_episode(
        self,
        user_id: str,
        episode_id: str,
        document: str,
        metadata: dict,
    ):
        """增量添加 episode → 重建 BM25 索引"""
        if not BM25S_AVAILABLE:
            return

        if user_id not in self._data:
            self._data[user_id] = []
            self._corpus_tokens[user_id] = []

        # 去重检查
        for existing in self._data[user_id]:
            if existing.get("id") == episode_id:
                return

        # 分词 (只对新文档)
        new_tokens = self._tokenizer.tokenize([document])[0]
        if not new_tokens:
            return

        self._data[user_id].append(
            {"id": episode_id, "document": document, **metadata}
        )
        self._corpus_tokens[user_id].append(new_tokens)

        # 重建 BM25 索引 (使用自定义双语分词结果, 千级文档 <10ms)
        retriever = bm25s.BM25(method="lucene")  # Lucene 变体 — Elasticsearch 默认
        retriever.index(self._corpus_tokens[user_id])
        self._retrievers[user_id] = retriever

        # 持久化到磁盘（BM25 索引 + episode 元数据）
        if self._persist_dir:
            try:
                save_path = Path(self._persist_dir) / user_id
                os.makedirs(save_path, exist_ok=True)
                retriever.save(
                    str(save_path),
                    corpus=[d["document"] for d in self._data[user_id]],
                )
                # 单独保存 episode 元数据（含 UUID、category 等），供重启后恢复
                metadata_path = save_path / self._DATA_FILENAME
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self._data[user_id], f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("BM25 index save failed for user %s: %s", user_id, e)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "default",
    ) -> List[RetrievalResult]:
        """BM25 搜索"""
        if not BM25S_AVAILABLE:
            return []

        if user_id not in self._retrievers or not self._data.get(user_id):
            return []

        tokens = self._tokenizer.tokenize([query])[0]
        if not tokens:
            return []

        retriever = self._retrievers[user_id]
        data = self._data[user_id]

        try:
            # bm25s query returns (scores, indices) arrays
            query_tokens = self._tokenizer.tokenize([query])
            results_obj, scores_obj = retriever.retrieve(
                query_tokens, k=min(top_k, len(data))
            )

            search_results: List[RetrievalResult] = []
            # results_obj shape: (1, k) — indices; scores_obj shape: (1, k) — scores
            indices = results_obj[0]
            scores = scores_obj[0]

            for i, idx in enumerate(indices):
                idx = int(idx)
                score = float(scores[i])
                if score <= 0 or idx < 0 or idx >= len(data):
                    continue
                ep = data[idx]
                search_results.append(
                    RetrievalResult(
                        id=ep["id"],
                        content=ep["document"],
                        score=score,
                        source="bm25_keyword",
                        metadata=ep,
                    )
                )
            return search_results
        except Exception as e:
            logger.warning("BM25 search failed for user %s: %s", user_id, e)
            return []
