import asyncio
import json
import os
import re
from itertools import cycle
from math import ceil, gcd, lcm
from pathlib import Path
from typing import Any

import json_repair
import yaml
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from memslides.utils.constants import (
    CONTEXT_LENGTH_LIMIT,
    MCP_CALL_TIMEOUT,
    PACKAGE_DIR,
    PIXEL_MULTIPLE,
    RETRY_TIMES,
)
from memslides.utils.log import debug, error, info, logging_openai_exceptions

LLM_KEY_FALLBACKS: dict[str, str] = {
    "modify_agent": "design_agent",
}

# vLLM extends the OpenAI chat/completions request schema with extra fields
# such as `chat_template_kwargs`. When we call it through the OpenAI Python SDK,
# these non-standard fields must be placed under `extra_body`.
VLLM_EXTRA_BODY_KEYS: set[str] = {
    "add_generation_prompt",
    "chat_template",
    "chat_template_kwargs",
    "continue_final_message",
    "documents",
    "mm_processor_kwargs",
    "priority",
    "return_token_ids",
    "return_tokens_as_token_ids",
    "structured_outputs",
}

_ENV_PLACEHOLDER_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)
_NON_RETRYABLE_LLM_ERROR_MARKERS = (
    "model_not_found",
    "No available channel for model",
    "insufficient_user_quota",
    "quota",
    "billing",
    "account balance",
)


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(env_name, default)

        return _ENV_PLACEHOLDER_PATTERN.sub(_replace, value)
    return value


def _is_non_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _NON_RETRYABLE_LLM_ERROR_MARKERS)


def _merge_vllm_extra_body(request_options: dict[str, Any]) -> dict[str, Any]:
    merged = dict(request_options)
    extra_body = dict(merged.pop("extra_body", {}) or {})
    for key in list(merged.keys()):
        if key in VLLM_EXTRA_BODY_KEYS:
            extra_body[key] = merged.pop(key)
    if extra_body:
        merged["extra_body"] = extra_body
    return merged


def _raw_decode_json_candidate(text: str, start: int) -> dict | list | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, (dict, list)) else None


def get_json_from_response(response: str) -> dict | list:
    assert isinstance(response, str) and len(response) > 0, (
        "response must be a non-empty string"
    )
    response = response.strip().strip("`")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    candidates: list[dict | list] = []
    for idx, char in enumerate(response):
        if char in "[{":
            decoded = _raw_decode_json_candidate(response, idx)
            if decoded is not None:
                candidates.append(decoded)

    repaired = json_repair.loads(response)
    if isinstance(repaired, (dict, list)):
        candidates.append(repaired)
    if not candidates:
        raise ValueError("No JSON object or array could be extracted from response")
    return max(candidates, key=lambda item: len(json.dumps(item, ensure_ascii=False)))


def _align_image_size(width: int, height: int, pixel_multiple: int) -> tuple[int, int]:
    if pixel_multiple <= 1:
        return width, height

    g = gcd(width, height)
    base_w, base_h = width // g, height // g

    k = lcm(
        pixel_multiple // gcd(pixel_multiple, base_w),
        pixel_multiple // gcd(pixel_multiple, base_h),
    )
    unit_w, unit_h = base_w * k, base_h * k

    scale = max(1, ceil(max(width / unit_w, height / unit_h)))

    return unit_w * scale, unit_h * scale


def _extract_slide_key_from_tool(from_tool: Any) -> str | None:
    """从 from_tool (Function 对象或 dict) 中提取 slide 文件名作为去重 key。"""
    if from_tool is None:
        return None
    if hasattr(from_tool, "arguments"):
        args_str = getattr(from_tool, "arguments", "") or ""
    elif isinstance(from_tool, dict):
        args_str = from_tool.get("arguments", "") or ""
    else:
        return None
    try:
        args = json.loads(args_str) if isinstance(args_str, str) and args_str.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(args, dict):
        return None
    slide_file = (
        args.get("target_file")
        or args.get("file_path")
        or args.get("html_file")
        or args.get("path")
        or args.get("slide_id")
        or args.get("slide_name")
    )
    if slide_file and isinstance(slide_file, str):
        return Path(slide_file).name
    return None


def _dump_message_for_api(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(mode="json")
        except TypeError:
            return message.model_dump()
    if isinstance(message, dict):
        return dict(message)
    return dict(message)


def _normalize_role_for_api(role: Any) -> str:
    if hasattr(role, "value"):
        role = role.value
    text = str(role or "user")
    if text.startswith("Role."):
        text = text.split(".", 1)[1].lower()
    return text


def _find_latest_image_tids(messages: list) -> set[str]:
    """扫描 chat history（newest→oldest），返回每个 slide 最新图片对应的 tool_call_id 集合。
    无法识别 slide 身份的图片（非 slide 工具返回的图片）也全部保留。
    """
    slide_to_latest: dict[str, str] = {}  # slide_key → latest tool_call_id
    unknown_tids: set[str] = set()       # 无法识别 slide 的图片，全部保留
    for msg in reversed(messages):
        try:
            d = _dump_message_for_api(msg)
        except Exception:
            continue
        if _normalize_role_for_api(d.get("role", "")) != "tool":
            continue
        content = d.get("content", [])
        if not isinstance(content, list):
            continue
        has_image = any(
            isinstance(b, dict) and b.get("type") in ("image_url", "image")
            for b in content
        )
        if not has_image:
            continue
        tid = d.get("tool_call_id", "")
        slide_key = _extract_slide_key_from_tool(d.get("from_tool"))
        if slide_key:
            if slide_key not in slide_to_latest:
                slide_to_latest[slide_key] = tid
        else:
            unknown_tids.add(tid)
    return set(slide_to_latest.values()) | unknown_tids


def _is_kimi_endpoint(base_url: str | None, model: str | None) -> bool:
    base = (base_url or "").lower()
    model_name = (model or "").lower()
    return "moonshot" in base or model_name.startswith("kimi")


def _is_thinking_disabled(request_options: dict[str, Any]) -> bool:
    extra_body = request_options.get("extra_body")
    if not isinstance(extra_body, dict):
        return False

    thinking = extra_body.get("thinking")
    if thinking is False:
        return True
    if isinstance(thinking, str):
        return thinking.lower() in {"disabled", "off", "false", "none"}
    if isinstance(thinking, dict):
        thinking_type = str(thinking.get("type", "")).lower()
        return thinking_type == "disabled"
    return False


def _sanitize_messages(
    messages: list,
    preserve_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    """Normalize ChatMessage / dict messages to API-compliant dicts.

    Strips non-standard fields (is_error, from_tool, created_at, extra_info,
    reasoning_content, etc.) and only keeps role, content, tool_calls
    (assistant only), tool_call_id (tool only), and name (if present).

    Also repairs broken tool_call ↔ tool_response pairings by injecting
    placeholder tool responses for any assistant tool_call_id that has no
    matching tool response message immediately after it.
    """
    # Pre-scan: identify latest image per slide for smart deduplication
    _latest_image_tids: set[str] = _find_latest_image_tids(messages)

    sanitized = []
    for msg in messages:
        d = _dump_message_for_api(msg)

        role = _normalize_role_for_api(d.get("role", "user"))
        clean: dict[str, Any] = {"role": role}

        # --- content ---
        content = d.get("content")
        if isinstance(content, list):
            texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            multimodal = [b for b in content if isinstance(b, dict) and b.get("type") in ("image_url", "image")]
            if multimodal:
                # Keep image only if this is the latest version for its slide
                tid = d.get("tool_call_id", "")
                if tid and tid not in _latest_image_tids:
                    # Old slide image — replace with placeholder, keep text
                    new_content = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") in ("image_url", "image"):
                            slide_key = _extract_slide_key_from_tool(d.get("from_tool")) or "slide"
                            new_content.append({"type": "text", "text": f"[{slide_key} image removed: newer version available]"})
                        else:
                            new_content.append(b)
                    clean["content"] = new_content
                else:
                    clean["content"] = content
            elif texts:
                clean["content"] = "\n".join(texts) if len(texts) > 1 else texts[0]
            else:
                clean["content"] = ""
        else:
            clean["content"] = content if content is not None else ""

        # --- role-specific fields ---
        if role == "assistant" and d.get("tool_calls"):
            raw_tcs = d["tool_calls"]
            serialized_tcs = []
            for tc in raw_tcs:
                if hasattr(tc, "model_dump"):
                    serialized_tcs.append(tc.model_dump())
                elif isinstance(tc, dict):
                    serialized_tcs.append(tc)
                else:
                    serialized_tcs.append(dict(tc))
            clean["tool_calls"] = serialized_tcs
            # Some APIs require content to be None when tool_calls present
            if not clean["content"]:
                clean["content"] = None
            reasoning_content = d.get("reasoning_content")
            if (
                preserve_reasoning_content
                and isinstance(reasoning_content, str)
                and reasoning_content.strip()
            ):
                clean["reasoning_content"] = reasoning_content

        if role == "tool":
            clean["tool_call_id"] = d.get("tool_call_id", "")

        if d.get("name"):
            clean["name"] = d["name"]

        sanitized.append(clean)

    # ── Repair pass: ensure every tool_call has a matching tool response ──
    repaired: list[dict[str, Any]] = []
    for i, msg in enumerate(sanitized):
        repaired.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Collect expected tool_call_ids from this assistant message
            expected_ids = set()
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    expected_ids.add(tc_id)
            # Scan the immediately following messages for tool responses
            responded_ids = set()
            j = i + 1
            while j < len(sanitized) and sanitized[j].get("role") == "tool":
                tid = sanitized[j].get("tool_call_id", "")
                if tid:
                    responded_ids.add(tid)
                j += 1
            # Inject placeholder responses for any missing tool_call_ids
            missing = expected_ids - responded_ids
            for mid in missing:
                repaired.append({
                    "role": "tool",
                    "tool_call_id": mid,
                    "content": "[tool response unavailable]",
                })

    # ── Image count guard: strip oldest images when approaching API limit ──
    # API limit is 50 images per request; we cap at 45 to leave a safe margin.
    _MAX_IMAGES = 45
    total_images = sum(
        sum(1 for b in msg["content"] if isinstance(b, dict) and b.get("type") == "image_url")
        for msg in repaired
        if isinstance(msg.get("content"), list)
    )
    if total_images > _MAX_IMAGES:
        # Walk from oldest (index 0) forward, stripping image blocks until under limit.
        # Each stripped image is replaced with a text placeholder so context is preserved.
        for msg in repaired:
            if total_images <= _MAX_IMAGES:
                break
            if not isinstance(msg.get("content"), list):
                continue
            new_content = []
            for block in msg["content"]:
                if (isinstance(block, dict)
                        and block.get("type") == "image_url"
                        and total_images > _MAX_IMAGES):
                    new_content.append({"type": "text", "text": "[image removed: history limit]"})
                    total_images -= 1
                else:
                    new_content.append(block)
            msg["content"] = new_content

    return repaired


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            text = _extract_message_text(block)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        for key in ("text", "content", "output_text", "value", "summary"):
            value = content.get(key)
            text = _extract_message_text(value)
            if text:
                return text
        return ""
    if content is None:
        return ""
    return str(content).strip()


def _extract_reasoning_text(message: Any) -> str:
    if message is None:
        return ""

    text = _extract_message_text(getattr(message, "reasoning_content", None))
    if text:
        return text

    extra = getattr(message, "__pydantic_extra__", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking", "reasoning_text", "reasoning_summary"):
        text = _extract_message_text(extra.get(key))
        if text:
            return text
    return ""


def _extract_visible_message_text(message: Any) -> str:
    if message is None:
        return ""
    text = _extract_message_text(getattr(message, "content", None))
    if text:
        return text
    return _extract_reasoning_text(message)


class MissingToolCallError(AssertionError):
    """Raised when a tool-required completion returns no structured tool calls."""

    def __init__(self, *, model: str, assistant_message: Any):
        self.model = model
        self.assistant_message = assistant_message
        self.assistant_text = _extract_visible_message_text(assistant_message)
        super().__init__(
            f"No tool call returned from the model, got {assistant_message}"
        )


def _repair_missing_tool_call_messages(
    messages: list[Any],
    error: MissingToolCallError,
) -> list[Any]:
    repaired_messages = list(messages)
    repaired_messages.append(
        {
            "role": "assistant",
            "content": (
                error.assistant_text
                or "[assistant omitted required structured tool_calls]"
            ),
        }
    )
    repaired_messages.append(
        {
            "role": "user",
            "content": (
                "Protocol repair: your previous assistant turn omitted the required "
                "`tool_calls`. Re-issue the exact same next step now as valid "
                "structured tool calls using the provided tools. Do not restate the "
                "plan in plain text. If your provider supports tool reasoning text, "
                "keep it to one short `Reason:` line plus the tool calls."
            ),
        }
    )
    return repaired_messages


class Endpoint(BaseModel):
    """LLM Endpoint Configuration"""

    base_url: str = Field(description="API base URL")
    model: str = Field(description="Model name")
    api_key: str = Field(description="API key")
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Client parameters"
    )
    sampling_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Sampling parameters"
    )
    _client: AsyncOpenAI = PrivateAttr()

    def model_post_init(self, _) -> None:
        import httpx

        client_kwargs = dict(self.client_kwargs or {})
        proxy_url = client_kwargs.pop("proxies", None)
        timeout = client_kwargs.pop("timeout", 60.0)

        if proxy_url:
            if isinstance(proxy_url, dict):
                proxy_url = proxy_url.get("https://") or proxy_url.get("http://")
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(timeout),
                verify=False,
            )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                **client_kwargs,
            )
        else:
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                trust_env=False,
            )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                **client_kwargs,
            )

    async def close(self) -> None:
        await self._client.close()

    async def call(
        self,
        messages: list[dict[str, Any]],
        soft_response_parsing: bool,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        """Execute a chat or tool call using the endpoint client"""
        sampling_parameters = _merge_vllm_extra_body({
            **self.sampling_parameters,
            **(request_kwargs or {}),
        })
        preserve_reasoning_content = (
            _is_kimi_endpoint(self.base_url, self.model)
            and not _is_thinking_disabled(sampling_parameters)
        )
        messages = _sanitize_messages(
            messages,
            preserve_reasoning_content=preserve_reasoning_content,
        )

        tool_choice_sentinel = object()
        tool_choice = sampling_parameters.pop("tool_choice", tool_choice_sentinel)
        if tool_choice is tool_choice_sentinel:
            if preserve_reasoning_content:
                tool_choice = "auto"
            else:
                tool_choice = "required"
        if tools is not None:
            tool_kwargs = {}
            if tool_choice is not None:
                tool_kwargs["tool_choice"] = tool_choice
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **tool_kwargs,
                **sampling_parameters,
            )
        elif not soft_response_parsing and response_format is not None:
            response: ChatCompletion = await self._client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **sampling_parameters,
            )
        else:
            response: ChatCompletion = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **sampling_parameters,
            )
        assert response.choices is not None and len(response.choices) > 0, (
            f"No choices returned from the model, got {response}"
        )
        message = response.choices[0].message
        if response_format is not None:
            message.content = response_format(
                **get_json_from_response(message.content)
            ).model_dump_json(indent=2)
        visible_message_text = _extract_visible_message_text(message)
        if tools is not None and not len(message.tool_calls or []):
            if tool_choice == "auto":
                pass
            elif _is_kimi_endpoint(self.base_url, self.model) and bool(visible_message_text):
                pass
            else:
                raise MissingToolCallError(model=self.model, assistant_message=message)
        assert message.tool_calls or visible_message_text, (
            "Empty content returned from the model"
        )
        return response


class LLM(BaseModel):
    """LLM Client Manager"""

    base_url: str | None = Field(None, description="OpenAI-compatible API base URL")
    model: str | None = Field(None, description="Model name")
    api_key: str | None = Field(None, description="API key")
    identifier: str | None = Field(None, description="Display/model route name")
    is_multimodal: bool | None = Field(None, description="Whether image input is supported")
    max_concurrent: int | None = Field(None, description="Async concurrency cap")
    retry_times: int = Field(RETRY_TIMES, ge=1, description="Application retry count")
    client_kwargs: dict[str, Any] = Field(default_factory=dict, description="Client options")
    sampling_parameters: dict[str, Any] = Field(default_factory=dict, description="Request options")
    endpoints: list[dict[str, Any]] = Field(default_factory=list, description="Fallback endpoint configs")
    soft_response_parsing: bool = Field(False, description="Parse structured output from text")
    min_image_size: int | None = Field(None, description="Minimum generated image area")
    secret_logging: bool = Field(False, description="Allow endpoint details in logs")

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _endpoints: list[Endpoint] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def model_name(self) -> str:
        if not self._endpoints:
            return self.identifier or str(self.model or "unconfigured")
        model_name = self._endpoints[0].model
        return self.identifier or model_name.split("/")[-1].split(":")[0]

    def model_post_init(self, context) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent or 10000)

        endpoint_configs: list[dict[str, Any]] = []
        if self.model:
            endpoint_configs.append(
                {
                    "base_url": self.base_url,
                    "model": self.model,
                    "api_key": self.api_key,
                    "client_kwargs": self.client_kwargs,
                    "sampling_parameters": self.sampling_parameters,
                }
            )
        endpoint_configs.extend(self.endpoints)

        for endpoint_config in endpoint_configs:
            if not self._has_api_key(endpoint_config):
                continue
            self._endpoints.append(Endpoint(**endpoint_config))
        return super().model_post_init(context)

    @staticmethod
    def _has_api_key(endpoint_config: dict[str, Any]) -> bool:
        token = str(endpoint_config.get("api_key") or "").strip()
        return bool(token) and token != "missing-api-key"

    async def run(
        self,
        messages: list[dict[str, Any]] | str,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_times: int | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        """Unified interface for chat and tool calls with alternating retry"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        active_messages = list(messages)

        effective_retry_times = retry_times or self.retry_times
        max_missing_tool_repairs = min(2, max(0, effective_retry_times - 1))
        missing_tool_repairs = 0
        errors = []
        if not self._endpoints:
            raise RuntimeError(
                f"LLM route `{self.model_name}` has no configured endpoint. "
                "Set an API key in a private YAML file, a Web Studio Service Profile, "
                "or an environment variable referenced by memslides.yaml."
            )
        non_retryable_failed_endpoints: set[tuple[str, str, str]] = set()
        iter_endpoints = cycle(self._endpoints)
        endpoint = next(iter_endpoints)
        async with self._semaphore:
            for _ in range(effective_retry_times):
                if non_retryable_failed_endpoints:
                    while (
                        (
                            str(endpoint.base_url or ""),
                            str(endpoint.model or ""),
                            str(endpoint.api_key or ""),
                        )
                        in non_retryable_failed_endpoints
                        and len(non_retryable_failed_endpoints) < len(self._endpoints)
                    ):
                        endpoint = next(iter_endpoints)
                current_endpoint = endpoint
                try:
                    return await current_endpoint.call(
                        active_messages,
                        self.soft_response_parsing,
                        response_format,
                        tools,
                        request_kwargs=request_kwargs,
                    )
                except MissingToolCallError as e:
                    errors.append(f"[{current_endpoint.model}] {e}")
                    if tools is not None and missing_tool_repairs < max_missing_tool_repairs:
                        missing_tool_repairs += 1
                        active_messages = _repair_missing_tool_call_messages(
                            active_messages,
                            e,
                        )
                        info(
                            f"[{current_endpoint.model}] repairing missing tool_calls "
                            f"(attempt {missing_tool_repairs}/{max_missing_tool_repairs})"
                        )
                    endpoint = next(iter_endpoints)
                    continue
                except (AssertionError, ValidationError) as e:
                    errors.append(f"[{current_endpoint.model}] {e}")
                    endpoint = next(iter_endpoints)
                    continue
                except Exception as e:
                    errors.append(f"[{current_endpoint.model}] {e}")
                    if self.secret_logging:
                        identifier = current_endpoint
                    else:
                        identifier = current_endpoint.model
                    logging_openai_exceptions(identifier, e)
                    if _is_non_retryable_llm_error(e):
                        non_retryable_failed_endpoints.add(
                            (
                                str(current_endpoint.base_url or ""),
                                str(current_endpoint.model or ""),
                                str(current_endpoint.api_key or ""),
                            )
                        )
                        if len(non_retryable_failed_endpoints) >= len(self._endpoints):
                            info(
                                f"[{current_endpoint.model}] non-retryable route error detected on all endpoints; aborting remaining retries"
                            )
                            break
                    endpoint = next(iter_endpoints)
        raise ValueError(
            f"All models failed after {effective_retry_times} retries:\n{errors}"
        )

    async def close(self) -> None:
        for endpoint in self._endpoints:
            try:
                await endpoint.close()
            except Exception as exc:
                debug(f"Failed to close LLM endpoint {endpoint.model}: {exc}")

    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        retry_times: int | None = None,
        pixel_multiple: int = PIXEL_MULTIPLE,
    ) -> ImagesResponse:
        """Unified interface for image generation"""
        effective_retry_times = retry_times or self.retry_times
        if not self._endpoints:
            raise RuntimeError(
                f"Image route `{self.model_name}` has no configured endpoint. "
                "Set an image-generation endpoint in a private YAML file, "
                "a Web Studio Service Profile, or an environment variable."
            )
        if self.min_image_size is not None and (width * height) < int(
            self.min_image_size
        ):
            ratio = (int(self.min_image_size) / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        width, height = _align_image_size(width, height, pixel_multiple)
        max_pixels = 16_777_216
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            width = int(width * scale) // pixel_multiple * pixel_multiple
            height = int(height * scale) // pixel_multiple * pixel_multiple
        async with self._semaphore:
            errors: list[str] = []
            for retry_idx in range(max(effective_retry_times, 1)):
                endpoint = self._endpoints[retry_idx % len(self._endpoints)]
                try:
                    response = await endpoint._client.images.generate(
                        prompt=prompt,
                        model=endpoint.model,
                        size=f"{width}x{height}",
                        timeout=MCP_CALL_TIMEOUT // 5,
                        **endpoint.sampling_parameters,
                    )
                    if not response.data:
                        raise AssertionError(
                            f"Expected at least one generated image, got {response}"
                        )
                    return response

                except (AssertionError, ValidationError) as exc:
                    errors.append(f"[{endpoint.model}] {exc}")
                except Exception as exc:
                    errors.append(f"[{endpoint.model}] {exc}")
                    logging_openai_exceptions(
                        endpoint if self.secret_logging else endpoint.model,
                        exc,
                    )
        raise ValueError(
            f"All image models failed after {effective_retry_times} retries: {errors}"
        )

    async def validate(self):
        if not self._endpoints:
            raise RuntimeError(
                f"LLM route `{self.model_name}` has no configured endpoint. "
                "Set an API key before validating this route."
            )
        endpoint = self._endpoints[0]
        models = await endpoint._client.models.list()
        if not any(model.id.endswith(endpoint.model) for model in models.data):
            raise Exception(
                f"Model {endpoint.model} is not available at {endpoint.base_url}, please check your apikey or {PACKAGE_DIR / 'memslides.yaml'}\n"
            )


class EmbeddingConfig(BaseModel):
    """Embedding model configuration for memory system"""

    provider: str = Field(
        default="openai-compatible",
        description=(
            "Embedding provider: 'local-first' (try local sentence-transformers model, then API) | "
            "'local'/'sentence-transformers' (local only) | 'openai' (official OpenAI embeddings) | "
            "'openai-compatible' (OpenAI-compatible embedding endpoint)"
        ),
    )
    model: str = Field(
        default="BAAI/bge-m3",
        description="Local HuggingFace model name/path, or API model name for API-only providers",
    )
    dim: int = Field(
        default=1024,
        description="Embedding 向量维度 (BGE-M3=1024, OpenAI small=1536)",
    )
    device: str = Field(
        default="auto",
        description="推理设备: 'auto' | 'cuda' | 'cpu' (local sentence-transformers providers)",
    )
    batch_size: int = Field(
        default=32,
        description="批处理大小 (local sentence-transformers providers)",
    )
    max_length: int = Field(
        default=512,
        description="最大 token 长度 (reserved for local embedding providers)",
    )
    cache_enabled: bool = Field(
        default=True,
        description="是否启用 LRU 缓存",
    )
    cache_size: int = Field(
        default=512,
        description="LRU 缓存最大条目数",
    )
    base_url: str = Field(
        default="",
        description="Embedding API base URL (provider=openai/openai-compatible)",
    )
    api_key: str = Field(
        default="",
        description="Embedding API key (provider=openai/openai-compatible)",
    )
    api_model: str = Field(
        default="",
        description="Fallback/API embedding model name for provider=local-first",
    )
    api_base_url: str = Field(
        default="",
        description="Primary fallback/API embedding base URL for provider=local-first",
    )
    api_fallback_model: str = Field(
        default="",
        description="Secondary API embedding model name when primary API endpoint fails",
    )
    api_fallback_base_url: str = Field(
        default="",
        description="Secondary API embedding base URL when primary API endpoint fails",
    )
    api_fallback_api_key: str = Field(
        default="",
        description="Secondary API embedding key when primary API endpoint fails",
    )


class GCConfig(BaseModel):
    """Garbage collection configuration for memory compactor"""

    max_rules_per_user: int = Field(default=200, description="Maximum rules per user before GC overflow")
    decay_factor: float = Field(default=0.95, description="Priority decay factor per day")
    min_priority: float = Field(default=0.01, description="Minimum priority before deprecation")
    max_experiences_per_user: int = Field(default=500, description="Maximum experience traces per user")
    experience_max_age_days: int = Field(default=90, description="Max age in days for unused experiences")


class ToolMemoryConfig(BaseModel):
    """Tool memory configuration for chain store limits"""

    max_raw_chains_per_key: int = Field(default=20, description="每个签名下最多保留的原始链数据条数")
    max_ltm_experiences_per_tool: int = Field(default=10, description="每个 tool_name 从 LTM 最多拉取的经验条数")


class VisualAssetConfig(BaseModel):
    """Structured chart/table rendering defaults."""

    default_font_family: str = Field(
        default="Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif",
        description="Default safe font stack for structured chart/table assets",
    )
    font_dirs: list[str] = Field(
        default_factory=list,
        description="Optional extra font directories registered for structured chart/table rendering",
    )


class MemoryModulesConfig(BaseModel):
    """Stage 2.5 module switches for memory system"""

    tool_knowledge_learning: bool = Field(default=True, description="Enable tool knowledge learning")
    foresight_detection: bool = Field(default=False, description="Enable foresight detection (disabled for now)")
    profile_update: bool = Field(default=True, description="Enable profile update from episodes")
    rule_induction: bool = Field(default=True, description="Enable procedural rule induction")
    profile_injection: bool = Field(default=True, description="Enable LTM user-profile injection into WM at job start")
    wm_preference_collection: bool = Field(default=True, description="Enable collecting temporary user preferences into WM")
    wm_preference_injection: bool = Field(default=True, description="Enable WM preference injection into prompts")
    wm_experience_collection: bool = Field(default=True, description="Enable collecting round experiences into WM during the job")
    wm_experience_injection: bool = Field(default=True, description="Enable WM round-experience injection into prompts")
    chain_experience_collection: bool = Field(default=True, description="Enable collecting chain/tool experiences into WM chain buffer during the job")
    wm_round_history_injection: bool = Field(default=False, description="Enable WM round-history injection into prompts")
    wm_task_history_injection: bool = Field(default=False, description="Deprecated alias for wm_round_history_injection")
    ltm_tool_experience_injection: bool = Field(default=True, description="Enable LTM chain/tool experience injection into operation prompts")
    experience_preload: bool = Field(default=True, description="Enable preload of relevant LTM round experiences into WM at job start")
    atomic_preference_writeback: bool = Field(default=True, description="Enable job-end AtomicPreference writeback to LTM")
    profile_writeback: bool = Field(default=True, description="Enable job-end UserProfile writeback to LTM")
    round_experience_writeback: bool = Field(default=True, description="Enable job-end round experience writeback to experience_traces")
    task_experience_writeback: bool = Field(default=True, description="Deprecated alias for round_experience_writeback")
    chain_experience_writeback: bool = Field(default=True, description="Enable job-end chain/tool experience writeback to chain_experiences")


class MemoryConfig(BaseModel):
    """Memory system configuration"""

    enabled: bool = Field(default=True, description="Enable memory system")
    llm_ref: str = Field(
        default="design_agent",
        description="Which LLM config key to use for rule extraction/intent classification",
    )
    llm: dict[str, str] = Field(
        default_factory=dict,
        description="Memory-call-type → model-ref routing for memory subsystem LLM calls",
    )
    embedding: EmbeddingConfig | None = Field(default=None, description="Embedding model config")
    db_path: str = Field(default=".memory/memory.db", description="SQLite database path (per-session fallback)")
    vector_db_path: str = Field(default=".memory/vectors.json", description="Vector index path (per-session fallback)")
    # [DELETED] prompt_log_dir — historical extraction file dumps are no longer wired by default
    params_store_dir: str = Field(default=".memory/params", description="Parameter snapshot directory")
    global_db_dir: str = Field(
        default="",
        description=(
            "Global directory for shared DB and vector index across all sessions. "
            "When non-empty, db and vector index are stored here instead of per-session. "
            "Set to e.g. '~/.cache/memslides/.memory' "
            "for cross-session reuse."
        ),
    )
    gc: GCConfig = Field(default_factory=GCConfig, description="GC configuration")
    modules: MemoryModulesConfig = Field(default_factory=MemoryModulesConfig, description="Stage 2.5 module switches")
    enable_agentic_retrieval: bool = Field(
        default=False,
        description="Enable LLM-driven multi-query retrieval. Adds LLM cost per retrieval.",
    )
    memory_v2: bool = Field(
        default=False,
        description="Enable WM + 双 LTM memory architecture (Stage 12). When True, MemoryOrchestrator manages memory lifecycle.",
    )
    artifact_trace: bool = Field(
        default=False,
        description="Enable ArtifactDumper to write intermediate pipeline artifacts to .memory/ for debugging.",
    )


class MemSlidesConfig(BaseModel):
    """MemSlides Global Configuration"""

    # config
    offline_mode: bool = Field(
        default=False, description="Enable offline mode, disable all network requests"
    )
    file_path: str = Field(description="Configuration file path")
    mcp_config_file: str = Field(
        description="MCP configuration file", default=str(PACKAGE_DIR / "mcp.json")
    )
    context_folding: bool = Field(
        default=False, description="Enable context management and auto summarization"
    )
    context_window: int | None = Field(
        default=None,
        description="Context window for context management, if not set, use the default value",
    )
    max_context_folds: int = Field(
        default=999, description="Maximum number of folds for context management (999 = effectively unlimited)"
    )
    heavy_reflect: bool = Field(
        default=False,
        description="Enable heavy reflection, use rendered slide image for reflective design",
    )

    # llms
    research_agent: LLM = Field(description="Research agent model configuration")
    design_agent: LLM = Field(description="Design agent model configuration")
    long_context_model: LLM = Field(description="Long context model configuration")
    vision_model: LLM = Field(description="Vision model configuration")
    t2i_model: LLM | None = Field(
        default=None, description="Text-to-image model configuration"
    )

    # 额外 LLM 配置（如 fast_model、balanced_model 等自定义 key）
    extra_llms: dict[str, LLM] = Field(
        default_factory=dict,
        description="Additional LLM configs from YAML (keys not matching declared fields)",
    )

    # memory
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory system configuration"
    )
    tool_memory: ToolMemoryConfig = Field(
        default_factory=ToolMemoryConfig, description="Tool memory chain store limits"
    )
    visual_assets: VisualAssetConfig = Field(
        default_factory=VisualAssetConfig,
        description="Structured chart/table rendering defaults",
    )

    def model_post_init(self, context):
        if self.context_window is None:
            # Use full CONTEXT_LENGTH_LIMIT (90k) regardless of context_folding.
            # Previously context_folding used //4 (22.5k), causing premature
            # compact_history that lost round progress on batch modifications.
            self.context_window = CONTEXT_LENGTH_LIMIT

        if self.context_folding:
            info(
                f"Context folding is enabled, context window: {self.context_window}, max folds: {self.max_context_folds}"
            )
        else:
            info(f"Context folding is disabled, context window: {self.context_window}")

        self._inherit_llm_fallback_endpoints()
        return super().model_post_init(context)

    def _inherit_llm_fallback_endpoints(self) -> None:
        """Share configured fallback endpoints with singleton generation models.

        The packaged config has multi-endpoint fallback chains for research/fast
        calls, but DeckDesigner historically used a single design endpoint. When
        that account is out of quota, application retries can keep hitting the
        same exhausted endpoint. If the design model did not explicitly define
        fallbacks, inherit compatible fallback endpoints already present in the
        config so long generation jobs can fail over without duplicating secrets
        in YAML.
        """

        def _endpoint_signature(endpoint: Endpoint) -> tuple[str, str, str]:
            return (
                str(endpoint.base_url or ""),
                str(endpoint.model or ""),
                str(endpoint.api_key or ""),
            )

        def _extend_if_single_endpoint(target: LLM, sources: list[LLM]) -> None:
            target_endpoints = getattr(target, "_endpoints", [])
            if len(target_endpoints) != 1:
                return
            seen = {_endpoint_signature(endpoint) for endpoint in target_endpoints}
            for source in sources:
                for endpoint in getattr(source, "_endpoints", [])[1:]:
                    signature = _endpoint_signature(endpoint)
                    if signature in seen:
                        continue
                    target_endpoints.append(
                        Endpoint(
                            base_url=endpoint.base_url,
                            model=endpoint.model,
                            api_key=endpoint.api_key,
                            client_kwargs=dict(endpoint.client_kwargs or {}),
                            sampling_parameters=dict(endpoint.sampling_parameters or {}),
                        )
                    )
                    seen.add(signature)

        fast_model = self.extra_llms.get("fast_model")
        balanced_model = self.extra_llms.get("balanced_model")
        fallback_sources = [
            source
            for source in (self.research_agent, fast_model, balanced_model)
            if isinstance(source, LLM)
        ]
        _extend_if_single_endpoint(self.design_agent, fallback_sources)
        _extend_if_single_endpoint(self.vision_model, fallback_sources)

    @classmethod
    def load_from_file(cls, config_path: str | None = None) -> "MemSlidesConfig":
        """Load configuration from file.

        YAML keys that match declared model fields (research_agent, design_agent, etc.)
        are parsed normally.  Any *extra* top-level key whose value is a dict containing
        a ``model`` key is treated as an additional LLM definition and stored in
        ``extra_llms``.  This allows MemSlides config files to define arbitrary model aliases
        (e.g. ``fast_model``, ``balanced_model``) that can be referenced by
        ``memory.llm`` routing without changing the Python class.
        """
        env_config_path = os.getenv("MEMSLIDES_CONFIG_FILE", "").strip()
        resolved_config_path = config_path or env_config_path
        if resolved_config_path:
            config_file = Path(resolved_config_path)
        else:
            config_file = PACKAGE_DIR / "memslides.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist")
        config_data = {}
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        config_data = _expand_env_placeholders(config_data)

        mcp_config_override = os.getenv("MEMSLIDES_MCP_CONFIG_FILE", "").strip()
        if mcp_config_override:
            config_data["mcp_config_file"] = mcp_config_override

        config_data["file_path"] = str(config_file.resolve())

        # Collect extra LLM definitions (keys not in the Pydantic model)
        known_fields = set(cls.model_fields.keys())
        extra_llms: dict[str, Any] = {}
        extra_keys: list[str] = []
        for k, v in config_data.items():
            if k not in known_fields and isinstance(v, dict) and "model" in v:
                extra_llms[k] = LLM(**v)
                extra_keys.append(k)
        for k in extra_keys:
            del config_data[k]
        config_data["extra_llms"] = extra_llms

        return cls(**config_data)

    async def validate_llms(self):
        # ? t2i endpoints might not support this api
        tasks = [
            self.research_agent.validate(),
            self.design_agent.validate(),
            self.long_context_model.validate(),
            self.vision_model.validate(),
        ]
        if self.t2i_model is not None:
            tasks.append(self.t2i_model.validate())
        for llm in self.extra_llms.values():
            tasks.append(llm.validate())
        await asyncio.gather(*tasks)

    def __getitem__(self, key: str) -> Any:
        """Lookup LLM config by key, applying fallback aliases when needed."""
        return self.resolve_llm(key)[1]

    def get_optional_llm(self, key: str) -> LLM | None:
        """Return an LLM config if present, otherwise None."""
        val = getattr(self, key, None)
        if isinstance(val, LLM):
            return val
        return self.extra_llms.get(key)

    def resolve_llm(self, key: str) -> tuple[str, LLM]:
        """Resolve an LLM key with fallback support.

        Returns the actual config key used together with the LLM config.
        """
        visited: set[str] = set()
        current_key = key

        while current_key not in visited:
            visited.add(current_key)
            llm = self.get_optional_llm(current_key)
            if llm is not None:
                return current_key, llm
            fallback_key = LLM_KEY_FALLBACKS.get(current_key)
            if fallback_key is None:
                break
            current_key = fallback_key

        raise KeyError(
            f"LLM config '{key}' not found in declared fields or extra_llms"
        )


GLOBAL_CONFIG = MemSlidesConfig.load_from_file()
