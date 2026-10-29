"""Embedding functions and vector utilities for the memory system."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

logger = logging.getLogger(__name__)

EmbeddingFunc = Callable[[list[str]], Awaitable[np.ndarray]]


def _resolve_embedding_max_batch_size(model: str, base_url: str | None) -> int:
    raw = os.environ.get("MEMSLIDES_EMBEDDING_MAX_BATCH_SIZE", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    marker = f"{model} {base_url or ''}".lower()
    if "bge-m3" in marker or "siliconflow" in marker:
        return 64
    return 2048


def _embedding_chunks(texts: list[str], batch_size: int) -> list[list[str]]:
    size = max(1, int(batch_size or 1))
    return [texts[index : index + size] for index in range(0, len(texts), size)]


async def openai_embed(
    texts: list[str],
    model: str = "BAAI/bge-m3",
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    fallback_base_url: str | None = None,
    fallback_dimensions: int | None = None,
    **_kwargs,
) -> np.ndarray:
    """调用 OpenAI / OpenAI-compatible Embedding API.

    Args:
        texts: 待编码文本列表
        model: 模型名称
        api_key: API Key（None则从环境变量读取）
        base_url: API基地址（支持代理或 OpenAI-compatible embedding 服务）
        dimensions: 可选输出维度。不要给 compatible endpoint 传该参数，除非服务明确支持

    Returns:
        np.ndarray, shape=(len(texts), dim)
    """
    from openai import AsyncOpenAI

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    primary_api_key = (
        api_key
        or os.environ.get("MEMSLIDES_EMBEDDING_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("MEMSLIDES_OPENAI_API_KEY")
    )
    primary_base_url = (
        base_url
        or os.environ.get("MEMSLIDES_EMBEDDING_BASE_URL")
        or os.environ.get("SILICONFLOW_BASE_URL")
        or os.environ.get("MEMSLIDES_OPENAI_BASE_URL")
    )
    fallback_api_key = (
        fallback_api_key
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_KEY")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    fallback_base_url = (
        fallback_base_url
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_BASE_URL")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_BASE_URL")
    )
    fallback_model = (
        fallback_model
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_MODEL")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_MODEL")
    )

    candidates: list[tuple[str, str | None, str | None, int | None]] = []

    def add_candidate(
        candidate_model: str | None,
        candidate_base_url: str | None,
        candidate_api_key: str | None,
        candidate_dimensions: int | None,
    ) -> None:
        resolved_model = str(candidate_model or "").strip()
        if not resolved_model:
            return
        resolved_base_url = str(candidate_base_url or "").strip() or None
        resolved_api_key = str(candidate_api_key or "").strip() or None
        marker = (resolved_model, resolved_base_url, resolved_api_key)
        if any((m, b, k) == marker for m, b, k, _d in candidates):
            return
        candidates.append((resolved_model, resolved_base_url, resolved_api_key, candidate_dimensions))

    add_candidate(model, primary_base_url, primary_api_key, dimensions)
    add_candidate(fallback_model, fallback_base_url, fallback_api_key, fallback_dimensions)

    errors: list[str] = []
    for candidate_model, candidate_base_url, candidate_api_key, candidate_dimensions in candidates:
        try:
            dimension_value = int(candidate_dimensions) if candidate_dimensions is not None else 0
        except (TypeError, ValueError):
            dimension_value = 0
        try:
            client = AsyncOpenAI(
                api_key=candidate_api_key,
                base_url=candidate_base_url,
            )
            embeddings: list[list[float]] = []
            max_batch_size = _resolve_embedding_max_batch_size(candidate_model, candidate_base_url)
            for batch in _embedding_chunks(texts, max_batch_size):
                request_kwargs: dict[str, object] = {"input": batch, "model": candidate_model}
                if dimension_value > 0:
                    request_kwargs["dimensions"] = dimension_value
                response = await client.embeddings.create(**request_kwargs)
                embeddings.extend(item.embedding for item in response.data)
            return np.array(embeddings, dtype=np.float32)
        except Exception as exc:
            endpoint_label = candidate_base_url or "default-openai"
            errors.append(f"{candidate_model}@{endpoint_label}: {exc}")
            logger.warning("Embedding endpoint failed (%s@%s): %s", candidate_model, endpoint_label, exc)

    raise RuntimeError(f"All embedding API endpoints failed: {errors}")


# ── Local sentence-transformers embedding ──

_local_embedding_models: dict[tuple[str, str], object] = {}
_local_embedding_lock = asyncio.Lock()
_local_embedding_failures: dict[tuple[str, str], str] = {}
# Backward-compatible alias used by existing tests/debug helpers.
_bge_m3_local_failures = _local_embedding_failures


def _local_embedding_model_candidates(model_name: str | None) -> list[str]:
    """Return local embedding model candidates before any API fallback.

    Experiments often export EMBEDDING_MODEL/BGE_M3_DIR instead of the
    MemSlides-specific MEMSLIDES_EMBEDDING_MODEL. Treat those as local
    fallback candidates so "local-first" really stays local before API.
    """
    candidates: list[str] = []

    def add(value: str | None) -> None:
        candidate = str(value or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(model_name or "BAAI/bge-m3")
    for env_name in (
        "MEMSLIDES_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
        "BGE_M3_DIR",
        "MOS_EMBEDDER_MODEL",
    ):
        add(os.environ.get(env_name))

    return candidates


def _resolve_hf_snapshot_dir(model_name: str) -> Path | None:
    """Resolve a Hugging Face cached snapshot directory for a repo id."""
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate

    if "/" not in model_name:
        return None

    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root:
        hub_root = Path(cache_root).expanduser()
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hub_root = Path(hf_home).expanduser() / "hub"
        else:
            hub_root = Path.home() / ".cache" / "huggingface" / "hub"

    repo_cache_dir = hub_root / f"models--{model_name.replace('/', '--')}"
    if not repo_cache_dir.exists():
        return None

    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    refs_main = repo_cache_dir / "refs" / "main"
    preferred_rev = ""
    if refs_main.exists():
        try:
            preferred_rev = refs_main.read_text(encoding="utf-8").strip()
        except Exception:
            preferred_rev = ""

    candidates: list[Path] = []
    if preferred_rev:
        preferred = snapshots_dir / preferred_rev
        if preferred.exists():
            candidates.append(preferred)

    try:
        others = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(p for p in others if p not in candidates)
    except Exception:
        pass

    required_files = ("modules.json", "config.json")
    for snapshot_dir in candidates:
        if all((snapshot_dir / name).exists() for name in required_files):
            return snapshot_dir

    return None


def _should_disable_safetensors(load_target: Path | None, model_name: str) -> bool:
    """Prefer PyTorch checkpoints for local models that do not ship safetensors."""
    candidate = load_target
    if candidate is None:
        model_path = Path(model_name).expanduser()
        if model_path.exists():
            candidate = model_path

    if candidate is None or not candidate.is_dir():
        return False

    has_safetensors = (
        (candidate / "model.safetensors").exists()
        or (candidate / "model.safetensors.index.json").exists()
        or any(candidate.glob("model-*.safetensors"))
    )
    if has_safetensors:
        return False

    has_pytorch_weights = (
        (candidate / "pytorch_model.bin").exists()
        or (candidate / "pytorch_model.bin.index.json").exists()
        or any(candidate.glob("pytorch_model-*.bin"))
    )
    return has_pytorch_weights


async def _load_local_embedding_model(model_name: str = "BAAI/bge-m3", device: str = "auto"):
    """Lazily load a sentence-transformers embedding model.

    使用 sentence-transformers 加载本地或 HuggingFace 模型。
    如果 HuggingFace 被墙，可设置环境变量:
        export HF_ENDPOINT=https://hf-mirror.com
    或手动下载:
        huggingface-cli download <model_name>
    """
    resolved_device = device
    if resolved_device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    model_key = (str(model_name or "BAAI/bge-m3"), str(resolved_device or "cpu"))
    if model_key not in _local_embedding_models:
        async with _local_embedding_lock:
            if model_key not in _local_embedding_models:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. "
                        "Run: pip install sentence-transformers>=3.0.0"
                    )
                logger.info("Loading local embedding model: %s ...", model_name)
                try:
                    load_target = _resolve_hf_snapshot_dir(model_name)
                    if load_target is not None:
                        logger.info("Using cached local embedding snapshot: %s", load_target)
                    sentence_transformer_kwargs = {}
                    if _should_disable_safetensors(load_target, model_name):
                        logger.info(
                            "Local embedding weights detected without safetensors; "
                            "falling back to PyTorch checkpoint loading."
                        )
                        sentence_transformer_kwargs["model_kwargs"] = {
                            "use_safetensors": False
                        }
                    _local_embedding_models[model_key] = SentenceTransformer(
                        str(load_target) if load_target is not None else model_name,
                        device=resolved_device,
                        **sentence_transformer_kwargs,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load local embedding model '{model_name}': {e}\n"
                        f"Tips:\n"
                        f"  1. 国内网络请设置: export HF_ENDPOINT=https://hf-mirror.com\n"
                        f"  2. 手动下载: huggingface-cli download {model_name}\n"
                        f"  3. 或配置 API fallback: memslides.yaml → memory.embedding.api_base_url"
                    ) from e
                logger.info("Local embedding model loaded on %s", resolved_device)
    return _local_embedding_models[model_key]


async def _load_bge_m3(model_name: str = "BAAI/bge-m3", device: str = "auto"):
    """Backward-compatible BGE-M3 loader wrapper."""
    return await _load_local_embedding_model(model_name=model_name, device=device)


async def local_sentence_transformer_embed(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 512,
    **_kwargs,
) -> np.ndarray:
    """Local embedding via sentence-transformers.

    Args:
        texts: 待编码文本列表
        model_name: HuggingFace 模型名或本地路径
        device: "auto" | "cuda" | "cpu"
        batch_size: 批处理大小
        max_length: 最大 token 长度

    Returns:
        np.ndarray, shape=(len(texts), dim)
    """
    model = await _load_local_embedding_model(model_name, device)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
    )
    return np.array(result, dtype=np.float32)


async def bge_m3_embed(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 512,
    **kwargs,
) -> np.ndarray:
    """Backward-compatible local BGE-M3 embedding wrapper."""
    return await local_sentence_transformer_embed(
        texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        **kwargs,
    )


async def local_first_embed(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 512,
    api_model: str | None = None,
    api_key: str | None = None,
    api_base_url: str | None = None,
    base_url: str | None = None,
    api_fallback_model: str | None = None,
    api_fallback_base_url: str | None = None,
    api_fallback_api_key: str | None = None,
    dimensions: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Prefer local sentence-transformers candidates, then fallback to an API.

    To keep memory retrieval meaningful, configure local and API paths to serve
    the same embedding model/dimension whenever you use persisted vectors.
    """
    local_failures: list[str] = []
    for candidate_model in _local_embedding_model_candidates(model_name):
        failure_key = (str(candidate_model or "BAAI/bge-m3"), str(device or "auto"))
        cached_failure = _local_embedding_failures.get(failure_key)
        if cached_failure:
            local_failures.append(f"{candidate_model}: {cached_failure}")
            continue
        try:
            if str(candidate_model) != str(model_name or "BAAI/bge-m3"):
                logger.info(
                    "Trying local embedding fallback model: %s",
                    candidate_model,
                )
            return await local_sentence_transformer_embed(
                texts,
                model_name=candidate_model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
        except Exception as exc:
            failure_text = str(exc)
            _local_embedding_failures[failure_key] = failure_text
            local_failures.append(f"{candidate_model}: {failure_text}")
            logger.warning("Local embedding candidate unavailable (%s): %s", candidate_model, exc)

    local_failure = "; ".join(local_failures) or "no local candidates configured"
    logger.warning(
        "All local embedding candidates failed; falling back to API if configured. %s",
        local_failure,
    )

    resolved_base_url = (
        api_base_url
        or base_url
        or os.environ.get("MEMSLIDES_EMBEDDING_API_BASE_URL")
        or os.environ.get("MEMSLIDES_EMBEDDING_BASE_URL")
        or os.environ.get("MEMSLIDES_OPENAI_BASE_URL")
    )
    resolved_api_key = (
        api_key
        or os.environ.get("MEMSLIDES_EMBEDDING_API_KEY")
        or os.environ.get("MEMSLIDES_OPENAI_API_KEY")
    )
    resolved_model = (
        api_model
        or kwargs.get("fallback_model")
        or os.environ.get("MEMSLIDES_EMBEDDING_API_MODEL")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_MODEL")
        or model_name
    )
    resolved_fallback_model = (
        api_fallback_model
        or kwargs.get("api_fallback_model")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_MODEL")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_MODEL")
    )
    resolved_fallback_base_url = (
        api_fallback_base_url
        or kwargs.get("api_fallback_base_url")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_BASE_URL")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_BASE_URL")
    )
    resolved_fallback_api_key = (
        api_fallback_api_key
        or kwargs.get("api_fallback_api_key")
        or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not resolved_base_url:
        raise RuntimeError(
            "Local embedding model failed and no embedding API fallback is configured. "
            "Set memory.embedding.api_base_url/base_url in memslides.yaml, or export "
            "MEMSLIDES_EMBEDDING_API_BASE_URL. "
            f"Local failure: {local_failure}"
        )
    return await openai_embed(
        texts,
        model=str(resolved_model or model_name),
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        fallback_model=str(resolved_fallback_model or ""),
        fallback_api_key=resolved_fallback_api_key,
        fallback_base_url=resolved_fallback_base_url,
        dimensions=dimensions,
    )


async def bge_m3_local_first_embed(*args, **kwargs) -> np.ndarray:
    """Backward-compatible alias for local_first_embed()."""
    return await local_first_embed(*args, **kwargs)


# ── Embedding 缓存层 ──

class CachedEmbedding:
    """LRU 缓存包装器 — 相同文本不重复计算 Embedding"""

    def __init__(self, embed_func: EmbeddingFunc, max_size: int = 512):
        self._func = embed_func
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def __call__(self, texts: list[str]) -> np.ndarray:
        """EmbeddingFunc 签名: (list[str]) -> np.ndarray"""
        results: list[tuple[int, np.ndarray]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
                self._cache.move_to_end(text)
                self._hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._misses += 1

        if uncached_texts:
            new_embeddings = await self._func(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                vec = new_embeddings[j]
                text = uncached_texts[j]
                self._cache[text] = vec
                results.append((idx, vec))
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype=np.float32)

    @property
    def stats(self) -> dict:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }


# ── Embedding 函数工厂 ──

def get_embedding_func(
    provider: str = "local-first",
    cache: bool = True,
    cache_size: int = 512,
    **kwargs,
) -> EmbeddingFunc:
    """Embedding 函数工厂

    Args:
        provider: "local-first" (local sentence-transformers model, then API) |
                  "local" / "sentence-transformers" (local only) |
                  "openai" (OpenAI API) | "openai-compatible" (OpenAI-compatible endpoint)
        cache: 是否启用 LRU 缓存
        cache_size: 缓存最大条目数

    Returns:
        异步 Embedding 函数
    """
    normalized_provider = str(provider or "bge-m3").strip().lower().replace("_", "-")
    if normalized_provider in {
        "bge-m3-local-first",
        "local-first",
        "bge-m3-fallback-api",
        "bge-m3-auto",
        "sentence-transformers-local-first",
        "local-api-fallback",
    }:
        func = lambda texts: local_first_embed(texts, **kwargs)
        provider_name = "local-first"
    elif normalized_provider in {
        "bge-m3",
        "bge-m3-local",
        "local-bge-m3",
        "local",
        "sentence-transformers",
        "sentence-transformer",
        "local-sentence-transformers",
    }:
        func = lambda texts: local_sentence_transformer_embed(texts, **kwargs)
        provider_name = "local"
    elif normalized_provider in {
        "openai",
        "api",
        "openai-compatible",
        "bge-m3-api",
        "bge-m3-openai",
        "embedding-api",
    }:
        func = lambda texts: openai_embed(texts, **kwargs)
        provider_name = "openai" if normalized_provider == "openai" else "openai-compatible"
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    if cache:
        wrapped = CachedEmbedding(func, max_size=cache_size)
        setattr(wrapped, "embedding_provider", provider_name)
        setattr(wrapped, "embedding_model", kwargs.get("model") or kwargs.get("model_name") or "")
        return wrapped
    setattr(func, "embedding_provider", provider_name)
    setattr(func, "embedding_model", kwargs.get("model") or kwargs.get("model_name") or "")
    return func


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度

    Args:
        a: 向量a, shape=(dim,)
        b: 向量b, shape=(dim,)

    Returns:
        float, [-1, 1]
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    if a.shape[0] != b.shape[0]:
        logger.debug(
            "Embedding dimension mismatch in cosine_similarity: %s vs %s",
            a.shape[0],
            b.shape[0],
        )
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算query与matrix中每行的余弦相似度

    Args:
        query: shape=(dim,)
        matrix: shape=(n, dim)

    Returns:
        np.ndarray, shape=(n,)
    """
    query = np.asarray(query, dtype=np.float32).flatten()
    try:
        matrix = np.asarray(matrix, dtype=np.float32)
    except ValueError:
        logger.debug("Ragged embedding matrix; returning zero similarities")
        try:
            matrix_len = len(matrix)
        except TypeError:
            matrix_len = 0
        return np.zeros(matrix_len, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    if matrix.shape[1] != query.shape[0]:
        logger.debug(
            "Embedding dimension mismatch in batch_cosine_similarity: query=%s matrix=%s",
            query.shape[0],
            matrix.shape[1],
        )
        return np.zeros(matrix.shape[0], dtype=np.float32)
    norm_q = np.linalg.norm(query)
    norms_m = np.linalg.norm(matrix, axis=1)
    if norm_q == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    dots = matrix @ query
    denom = norms_m * norm_q
    denom = np.where(denom == 0, 1.0, denom)
    return (dots / denom).astype(np.float32)
