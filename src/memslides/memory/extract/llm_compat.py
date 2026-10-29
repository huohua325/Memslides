"""Helpers for calling LLM adapters used by memory extraction components."""

from __future__ import annotations

from typing import Any

from memslides.memory.extract.tool_reasoning import extract_text_from_content


def extract_message_text(message: Any) -> str:
    """Extract visible text from provider-specific message shapes."""
    if message is None:
        return ""

    content_text = extract_text_from_content(getattr(message, "content", None))
    if content_text:
        return content_text

    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    extra = getattr(message, "__pydantic_extra__", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking", "reasoning_text", "reasoning_summary"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = extract_text_from_content(value)
            if text:
                return text
        if isinstance(value, dict):
            for subkey in ("summary", "text", "content"):
                nested = value.get(subkey)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
                if isinstance(nested, list):
                    text = extract_text_from_content(nested)
                    if text:
                        return text

    return ""


def extract_response_text(response: Any) -> str:
    """Extract text from callable- or client-style LLM responses."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    choices = getattr(response, "choices", None)
    if choices:
        try:
            text = extract_message_text(choices[0].message)
            if text:
                return text
        except Exception:
            pass

    generations = getattr(response, "generations", None)
    if generations:
        try:
            return generations[0][0].text or ""
        except Exception:
            pass

    return str(response)


async def call_llm_with_prompt(llm: Any, prompt: str) -> str:
    """Support both callable LLMs and `.run(messages=...)` clients."""
    if not llm:
        return ""

    if callable(llm) and not hasattr(llm, "run"):
        response = await llm(prompt)
        return extract_response_text(response)

    response = await llm.run(
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_response_text(response)


def resolve_llm_retry_times(llm: Any, *, minimum: int = 1) -> int:
    """Expand retries to cover all configured endpoints for memory-side LLM calls."""
    retry_times = int(getattr(llm, "retry_times", 0) or 0)
    endpoints = getattr(llm, "_endpoints", None) or []
    endpoint_count = len(endpoints) if isinstance(endpoints, list) else 0
    return max(int(minimum), retry_times, endpoint_count or 0)
