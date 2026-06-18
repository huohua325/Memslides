"""OperationDistributor — Operation 级记忆分发（重构版）

两层检索策略：
1. 签名粗筛（SQL LIKE）— 通过 ChainStore.query_experiences_by_tool_with_embeddings
2. 向量精排（cosine similarity）— keyword_embedding vs 当前上下文向量

降级策略：
- embedding_func 不可用 → 按 confidence 排序
- 无 embedding 的条目排在有 embedding 的条目之后
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..store.chain_store import ChainStore
    from ..working_memory import WorkingMemory
    from .round_cache import RoundCache

logger = logging.getLogger(__name__)

MAX_TOOL_EXPS_PER_OP = 3
DEFAULT_TOOL_EXPS_PER_OP = 2
MAX_SAME_SIGNATURE_PER_OP = 1
MIN_TOOL_EXP_SCORE = 0.22
TOOL_EXP_CANDIDATE_LIMIT = 30


def _has_embedding_vector(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not _has_embedding_vector(a) or not _has_embedding_vector(b) or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class OperationDistributor:
    """Operation 级记忆分发器 — 两层检索。"""

    def __init__(
        self,
        round_cache: RoundCache,
        chain_store: ChainStore | None = None,
        working_memory: WorkingMemory | None = None,
        user_id: str = "",
        embedding_func: Callable | None = None,
        user_message: str = "",
        enable_ltm_tool_experience_injection: bool = True,
    ):
        self._cache = round_cache
        self._chain_store = chain_store
        self._wm = working_memory
        self._user_id = user_id
        self._embedding_func = embedding_func
        self._user_message = user_message
        self._enable_ltm_tool_experience_injection = enable_ltm_tool_experience_injection

    async def distribute(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        context: dict | None = None,
    ) -> str:
        """为单个 Operation 生成记忆注入文本。

        两层检索：签名粗筛 + 向量精排。
        空字符串表示无相关记忆。
        """
        if not self._enable_ltm_tool_experience_injection:
            return ""

        results = await self._two_layer_retrieve(tool_name, tool_args, context)
        if results:
            return self._format_chain_experiences(results)

        # 降级：走旧路径（WM cache + RoundCache）
        if self._wm and self._chain_store:
            try:
                await self._wm.chain_buffer.query_or_cache_from_ltm(
                    tool_name, self._chain_store, self._user_id,
                )
            except Exception as e:
                logger.warning(f"LTM cache query failed for {tool_name}: {e}")

        op_memory = self._cache.get_for_operation(tool_name, context)
        if op_memory.tool_experiences:
            return self._format_tool_experiences(
                op_memory.tool_experiences[:MAX_TOOL_EXPS_PER_OP],
            )
        return ""

    async def distribute_structured(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        context: dict | None = None,
    ) -> dict[str, str]:
        """返回各组件的独立文本，供 injection trace 使用。"""
        result: dict[str, str] = {
            "ltm_tool_experiences": "",
            "combined": "",
        }

        if not self._enable_ltm_tool_experience_injection:
            return result

        # 两层检索
        results = await self._two_layer_retrieve(tool_name, tool_args, context)
        if results:
            exp_text = self._format_chain_experiences(results)
            if exp_text:
                result["ltm_tool_experiences"] = exp_text
                result["combined"] = exp_text
                return result

        # 降级：走旧路径
        if self._wm and self._chain_store:
            try:
                await self._wm.chain_buffer.query_or_cache_from_ltm(
                    tool_name, self._chain_store, self._user_id,
                )
            except Exception as e:
                logger.warning(f"LTM cache query failed for {tool_name}: {e}")

        op_memory = self._cache.get_for_operation(tool_name, context)
        if op_memory.tool_experiences:
            exp_text = self._format_tool_experiences(
                op_memory.tool_experiences[:MAX_TOOL_EXPS_PER_OP],
            )
            if exp_text:
                result["ltm_tool_experiences"] = exp_text
                result["combined"] = exp_text

        return result

    async def _two_layer_retrieve(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        context: dict | None = None,
    ) -> list:
        """两层检索：签名粗筛 + 向量精排。"""
        if not self._chain_store:
            return []

        try:
            # 第一层：签名粗筛
            candidates = await self._chain_store.query_experiences_by_tool_with_embeddings(
                tool_name, self._user_id, limit=TOOL_EXP_CANDIDATE_LIMIT,
            )
            if not candidates:
                return []

            embedding_scores: dict[str, float] = {}
            query_text = self._build_query_text(tool_name, tool_args, context)
            query_terms = self._tokenize(query_text)

            # 第二层：向量精排
            if self._embedding_func and any(_has_embedding_vector(emb) for _, emb in candidates):
                try:
                    query_vec = await self._embedding_func([query_text])
                    if hasattr(query_vec, 'tolist'):
                        query_vec = query_vec[0].tolist()
                    elif isinstance(query_vec, list) and len(query_vec) > 0:
                        q = query_vec[0]
                        query_vec = q.tolist() if hasattr(q, 'tolist') else q
                    else:
                        query_vec = None

                    if _has_embedding_vector(query_vec):
                        with_emb = []
                        for exp, emb in candidates:
                            if _has_embedding_vector(emb):
                                score = _cosine_similarity(query_vec, emb)
                                with_emb.append((exp, score))
                        embedding_scores = {
                            f"{exp.chain_name}::{exp.subkey}": score
                            for exp, score in with_emb
                        }
                except Exception as e:
                    logger.warning(f"Vector ranking failed for {tool_name}: {e}")

            scored = []
            risky_context = self._is_risky_context(context)
            for exp, _ in candidates:
                exp_key = f"{exp.chain_name}::{exp.subkey}"
                role = self._classify_experience_role(exp)
                score = self._score_experience(
                    exp,
                    tool_name,
                    query_terms,
                    risky_context,
                    embedding_scores.get(exp_key, 0.0),
                )
                scored.append({
                    "exp": exp,
                    "score": score,
                    "role": role,
                })

            return self._select_experiences(scored, risky_context)

        except Exception as e:
            logger.warning(f"Two-layer retrieve failed for {tool_name}: {e}")
            return []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", str(text or "").lower()):
            cleaned = token.strip("_")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    def _build_query_text(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        context: dict | None = None,
    ) -> str:
        parts = [self._user_message, tool_name]
        if tool_args:
            try:
                parts.append(json.dumps(tool_args, ensure_ascii=False, sort_keys=True))
            except TypeError:
                parts.append(str(tool_args))
        if context:
            for key in (
                "recent_observation",
                "recent_error_message",
                "current_target",
                "used_tools",
                "agent_phase",
            ):
                value = context.get(key)
                if value:
                    if isinstance(value, (list, tuple, set)):
                        parts.append(" ".join(str(v) for v in value))
                    else:
                        parts.append(str(value))
        return " ".join(p for p in parts if p).strip()

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

    def _keyword_overlap(self, query_terms: set[str], exp: Any) -> float:
        if not query_terms:
            return 0.0
        exp_terms = self._tokenize(
            " ".join(
                [
                    getattr(exp, "chain_name", ""),
                    " ".join(getattr(exp, "tool_pipeline", []) or []),
                    getattr(exp, "applicable_when", ""),
                    getattr(exp, "inject_summary", ""),
                    " ".join(getattr(exp, "keywords", []) or []),
                ]
            )
        )
        if not exp_terms:
            return 0.0
        overlap = len(query_terms & exp_terms)
        return overlap / max(len(query_terms), min(len(exp_terms), 8), 1)

    @staticmethod
    def _is_risky_context(context: dict | None = None) -> bool:
        if not context:
            return False
        if context.get("recent_error") or int(context.get("retry_count", 0) or 0) > 0:
            return True
        recent_text = " ".join(
            str(context.get(key, "") or "")
            for key in ("recent_observation", "recent_error_message")
        ).lower()
        return any(
            term in recent_text
            for term in (
                "error",
                "failed",
                "exception",
                "not found",
                "失败",
                "报错",
                "找不到",
                "无法",
            )
        )

    @staticmethod
    def _classify_experience_role(exp: Any) -> str:
        anti = str(getattr(exp, "anti_pattern", "") or "").strip()
        combined = " ".join(
            [
                str(getattr(exp, "inject_summary", "") or ""),
                str(getattr(exp, "lesson", "") or ""),
                str(getattr(exp, "applicable_when", "") or ""),
            ]
        ).lower()
        if anti:
            return "anti_pattern"
        if any(
            term in combined
            for term in (
                "fallback",
                "失败时",
                "报错时",
                "改用",
                "改走",
                "若失败",
                "出错时",
                "找不到",
            )
        ):
            return "fallback"
        return "primary"

    def _score_experience(
        self,
        exp: Any,
        tool_name: str,
        query_terms: set[str],
        risky_context: bool,
        embedding_score: float,
    ) -> float:
        relevance = max(0.0, embedding_score)
        overlap = self._keyword_overlap(query_terms, exp)
        confidence = min(max(float(getattr(exp, "confidence", 0.0) or 0.0), 0.0), 1.0)
        support = min(max(int(getattr(exp, "support_count", 1) or 1), 1), 10) / 10.0
        role = self._classify_experience_role(exp)

        score = (
            relevance * 0.55
            + overlap * 0.20
            + confidence * 0.15
            + support * 0.10
        )
        if tool_name in (getattr(exp, "tool_pipeline", []) or []):
            score += 0.05
        if getattr(exp, "applicable_when", "") and overlap > 0:
            score += 0.05

        if role == "primary":
            score += 0.03
        elif risky_context:
            score += 0.10
        else:
            score -= 0.02

        return score

    def _experience_similarity(self, left: Any, right: Any) -> float:
        left_terms = self._tokenize(
            " ".join(
                [
                    getattr(left, "inject_summary", ""),
                    getattr(left, "lesson", ""),
                    getattr(left, "anti_pattern", ""),
                ]
            )
        )
        right_terms = self._tokenize(
            " ".join(
                [
                    getattr(right, "inject_summary", ""),
                    getattr(right, "lesson", ""),
                    getattr(right, "anti_pattern", ""),
                ]
            )
        )
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)

    def _can_select(
        self,
        item: dict[str, Any],
        selected: list[dict[str, Any]],
        signature_counts: dict[str, int],
        risky_context: bool,
    ) -> bool:
        exp = item["exp"]
        chain_name = getattr(exp, "chain_name", "")
        role = item["role"]

        same_signature_limit = MAX_SAME_SIGNATURE_PER_OP
        if risky_context and role in {"anti_pattern", "fallback"}:
            same_signature_limit = 2
        if signature_counts.get(chain_name, 0) >= same_signature_limit:
            return False

        for chosen in selected:
            chosen_exp = chosen["exp"]
            similarity = self._experience_similarity(exp, chosen_exp)
            if similarity >= 0.68:
                if getattr(chosen_exp, "chain_name", "") == chain_name:
                    return False
                if chosen["role"] == role:
                    return False
        return True

    def _select_experiences(
        self,
        scored: list[dict[str, Any]],
        risky_context: bool,
    ) -> list:
        if not scored:
            return []

        scored.sort(key=lambda item: item["score"], reverse=True)
        if scored[0]["score"] < MIN_TOOL_EXP_SCORE:
            return []

        target_k = MAX_TOOL_EXPS_PER_OP if risky_context else DEFAULT_TOOL_EXPS_PER_OP
        selected: list[dict[str, Any]] = []
        signature_counts: dict[str, int] = {}

        role_order = ["primary", "anti_pattern", "fallback"] if risky_context else ["primary", "fallback", "anti_pattern"]
        for desired_role in role_order:
            for item in scored:
                if item["role"] != desired_role:
                    continue
                if item["score"] < MIN_TOOL_EXP_SCORE:
                    continue
                if self._can_select(item, selected, signature_counts, risky_context):
                    selected.append(item)
                    chain_name = getattr(item["exp"], "chain_name", "")
                    signature_counts[chain_name] = signature_counts.get(chain_name, 0) + 1
                    break
            if len(selected) >= target_k:
                break

        for item in scored:
            if len(selected) >= target_k:
                break
            if item in selected:
                continue
            if item["score"] < MIN_TOOL_EXP_SCORE:
                continue
            if self._can_select(item, selected, signature_counts, risky_context):
                selected.append(item)
                chain_name = getattr(item["exp"], "chain_name", "")
                signature_counts[chain_name] = signature_counts.get(chain_name, 0) + 1

        return [item["exp"] for item in selected[:target_k]]

    def _format_tool_experiences(self, exps: list) -> str:
        """格式化工具经验。区分 ChainExperience 和 ExperienceTrace，降级为简单格式。"""
        try:
            from ..core.models import ChainExperience
            chain_exps = [e for e in exps if isinstance(e, ChainExperience)]
            trace_exps = [e for e in exps if not isinstance(e, ChainExperience)]
            if trace_exps:
                lines = ["## 工具使用经验"]
                for t in trace_exps:
                    lesson = getattr(t, "lessons_learned", "") or getattr(t, "lesson", "")
                    if lesson:
                        lines.append(f"- {lesson}")
                if len(lines) > 1:
                    return "\n".join(lines)
            if chain_exps:
                return self._format_chain_experiences(chain_exps)
        except Exception:
            pass
        return self._format_chain_experiences(exps)

    def _format_chain_experiences(self, exps: list) -> str:
        """压缩格式化 ChainExperience，避免多变体注入时 prompt 爆炸。"""
        lines = ["## 工具使用经验"]
        for e in exps:
            role = self._classify_experience_role(e)
            label = {
                "primary": "推荐",
                "anti_pattern": "避免",
                "fallback": "失败时",
            }.get(role, "提示")
            summary = getattr(e, "inject_summary", "") or getattr(e, "lesson", "") or getattr(e, "lessons_learned", "")
            summary = self._compact_text(summary)
            if summary:
                lines.append(f"- {label}: {summary}")

            applicable = self._compact_text(getattr(e, "applicable_when", ""), limit=110)
            if applicable:
                lines.append(f"  适用: {applicable}")
        return "\n".join(lines) if len(lines) > 1 else ""
