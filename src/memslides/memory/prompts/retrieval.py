"""Retrieval Prompts — 多查询生成"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# 多查询生成 (检索时/可选)
# ══════════════════════════════════════════════════════════════════════════════

MULTI_QUERY_PROMPT = """Given the user's design modification request and current search results,
generate 2-3 complementary search queries to find missing relevant memories.

## Original query
{original_query}

## Current results (may be insufficient)
{current_results}

## Task
Generate queries that would find memories NOT covered by current results.
Focus on: user preferences, past similar modifications, learned strategies.

## Output (JSON array of strings)
["query1", "query2"]"""
