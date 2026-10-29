"""Implementation details."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .strategies import RetrievalResult

logger = logging.getLogger(__name__)


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
    """Implementation details."""

    def __init__(self):
        self._en_stemmer = _en_stemmer

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        """Implementation details."""
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
        """Implementation details."""
        return [self._tokenize_single(t) for t in texts]

    def _tokenize_single(self, text: str) -> list[str]:
        """Implementation details."""
        text = text.strip()
        if not text:
            return []


        segments = self._segment_by_script(text)
        tokens: list[str] = []

        for script, content in segments:
            if script == "cjk":
                tokens.extend(self._tokenize_cjk(content))
            else:
                tokens.extend(self._tokenize_latin(content))

        return tokens

    def _segment_by_script(self, text: str) -> list[tuple[str, str]]:
        """Implementation details."""
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
        """Implementation details."""
        if JIEBA_AVAILABLE:
            tokens: list[str] = []
            for t in jieba.cut_for_search(text):
                t = t.strip()
                if t and len(t) >= 1:
                    tokens.append(t)
            return tokens
        else:

            return [ch for ch in text if ch.strip()]

    def _tokenize_latin(self, text: str) -> list[str]:
        """Implementation details."""

        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
        tokens = [t for t in raw_tokens if len(t) > 1]


        if self._en_stemmer and tokens:
            tokens = self._en_stemmer.stemWords(tokens)

        return tokens


class BM25SearchStrategy:
    """Implementation details."""

    name = "bm25_keyword"

    def __init__(self, persist_dir: Optional[str] = None):
        self._tokenizer = BilingualTokenizer()
        self._persist_dir = persist_dir

        # Per-user state
        self._retrievers: Dict[str, Any] = {}  # user_id → bm25s.BM25
        self._data: Dict[str, List[Dict]] = {}  # user_id → episode dicts
        self._corpus_tokens: Dict[str, list] = {}  # user_id → cached token lists


        if persist_dir:
            self._try_load_indices()

    _DATA_FILENAME = "data_metadata.json"

    def _try_load_indices(self):
        """Implementation details."""
        if not self._persist_dir or not BM25S_AVAILABLE:
            return
        index_dir = Path(self._persist_dir)
        if not index_dir.exists():
            return
        for user_dir in index_dir.iterdir():
            if not user_dir.is_dir():
                continue
            try:


                retriever = bm25s.BM25.load(str(user_dir), load_corpus=False)
                user_id = user_dir.name
                self._retrievers[user_id] = retriever


                metadata_path = user_dir / self._DATA_FILENAME
                if metadata_path.exists():
                    with open(metadata_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self._data[user_id] = data

                    self._corpus_tokens[user_id] = [
                        self._tokenizer.tokenize([d["document"]])[0]
                        for d in data
                    ]
                    logger.info(
                        "Loaded BM25 index + %d docs for user %s",
                        len(data), user_id,
                    )
                else:

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
        """Implementation details."""
        if not BM25S_AVAILABLE:
            return

        if user_id not in self._data:
            self._data[user_id] = []
            self._corpus_tokens[user_id] = []


        for existing in self._data[user_id]:
            if existing.get("id") == episode_id:
                return


        new_tokens = self._tokenizer.tokenize([document])[0]
        if not new_tokens:
            return

        self._data[user_id].append(
            {"id": episode_id, "document": document, **metadata}
        )
        self._corpus_tokens[user_id].append(new_tokens)


        retriever = bm25s.BM25(method="lucene")
        retriever.index(self._corpus_tokens[user_id])
        self._retrievers[user_id] = retriever


        if self._persist_dir:
            try:
                save_path = Path(self._persist_dir) / user_id
                os.makedirs(save_path, exist_ok=True)
                retriever.save(
                    str(save_path),
                    corpus=[d["document"] for d in self._data[user_id]],
                )

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
        """Implementation details."""
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
