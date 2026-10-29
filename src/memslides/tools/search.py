import asyncio
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Ensure direct script execution imports the local repo package first.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import aiohttp
import httpx
import markdownify
from fake_useragent import UserAgent
from fastmcp import FastMCP
from PIL import Image
from playwright.async_api import TimeoutError
from trafilatura import extract

from memslides.utils.constants import (
    MAX_RETRY_INTERVAL,
    MCP_CALL_TIMEOUT,
    RETRY_TIMES,
)
from memslides.utils.env import getenv_optional
from memslides.utils.log import debug, set_logger, warning
from memslides.utils.webview import PlaywrightConverter

mcp = FastMCP(name="MemSlidesSearchTools")

FAKE_UA = UserAgent()
TAVILY_API_URL = "https://api.tavily.com/search"


def _tavily_keys() -> list[str]:
    return [
        item.strip()
        for item in getenv_optional("MEMSLIDES_TAVILY_API_KEY").split(",")
        if item.strip().startswith("tvly")
    ]


debug(f"{len(_tavily_keys())} TAVILY keys loaded")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _fetch_url_http_timeout_s() -> float:
    return max(1.0, _env_float("MEMSLIDES_FETCH_URL_HTTP_TIMEOUT", 10.0))


def _fetch_url_browser_timeout_s() -> float:
    default = min(max(MCP_CALL_TIMEOUT // 5, 5), 30)
    return max(1.0, _env_float("MEMSLIDES_FETCH_URL_BROWSER_TIMEOUT", min(float(default), 15.0)))


def _fetch_url_total_timeout_s() -> float:
    default = min(max(MCP_CALL_TIMEOUT // 3, 10), 60)
    return max(1.0, _env_float("MEMSLIDES_FETCH_URL_TOTAL_TIMEOUT", min(float(default), 30.0)))


def _download_file_timeout_s() -> float:
    return max(1.0, _env_float("MEMSLIDES_DOWNLOAD_FILE_TIMEOUT", 15.0))


def _download_file_retries() -> int:
    raw = os.getenv("MEMSLIDES_DOWNLOAD_FILE_RETRIES", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _relay_base_url() -> str:
    return getenv_optional("MEMSLIDES_RELAY_BASE_URL").rstrip("/")


def _relay_token() -> str:
    return getenv_optional("MEMSLIDES_RELAY_TOKEN")


def _relay_allowed_hosts() -> tuple[str, ...]:
    return tuple(
        host.strip().lower()
        for host in getenv_optional("MEMSLIDES_RELAY_ALLOWED_HOSTS").split(",")
        if host.strip()
    )


def _relay_failure_cooldown_s() -> float:
    return max(1.0, _env_float("MEMSLIDES_RELAY_FAILURE_COOLDOWN_SECONDS", 300.0))


_FAILURE_COOLDOWNS: dict[str, tuple[float, str]] = {}


def _normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    path = parsed.path or "/"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            query,
            "",
        )
    )


def _url_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def _host_matches_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in allowed_hosts)


def _should_use_relay(url: str) -> bool:
    relay_base_url = _relay_base_url()
    relay_token = _relay_token()
    allowed_hosts = _relay_allowed_hosts()
    host = _url_host(url)
    if not relay_base_url or not relay_token or not host or not allowed_hosts:
        return False
    return _host_matches_allowed(host, allowed_hosts)


def _failure_payload(
    *,
    url: str,
    reason: str,
    detail: str,
    source: str,
    retry_after_seconds: int | None = None,
) -> str:
    payload = {
        "url": url,
        "available": False,
        "reason": reason,
        "detail": detail,
        "retry_after_seconds": int(retry_after_seconds or _relay_failure_cooldown_s()),
        "source": source,
    }
    return json.dumps(payload, ensure_ascii=False)


def _get_cooldown(url: str) -> str | None:
    normalized_url = _normalize_url(url)
    entry = _FAILURE_COOLDOWNS.get(normalized_url)
    if not entry:
        return None
    expires_at, payload = entry
    if expires_at <= time.time():
        _FAILURE_COOLDOWNS.pop(normalized_url, None)
        return None
    return payload


def _remember_cooldown(url: str, payload: str) -> str:
    _FAILURE_COOLDOWNS[_normalize_url(url)] = (
        time.time() + _relay_failure_cooldown_s(),
        payload,
    )
    return payload


def _relay_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_relay_token()}",
        "Content-Type": "application/json",
    }


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


async def _fetch_via_relay(url: str, *, body_only: bool) -> str:
    timeout_s = _fetch_url_total_timeout_s()
    timeout = httpx.Timeout(timeout_s, connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s)
    relay_url = f"{_relay_base_url()}/v1/fetch"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.post(
                relay_url,
                headers=_relay_headers(),
                json={
                    "url": url,
                    "body_only": body_only,
                    "prefer_api": True,
                },
            )
    except httpx.TimeoutException as exc:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_timeout",
                detail=str(exc) or exc.__class__.__name__,
                source="relay",
            ),
        )
    except Exception as exc:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_unavailable",
                detail=str(exc),
                source="relay",
            ),
        )

    payload = _response_json(response)
    if response.status_code in {401, 403}:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_forbidden",
                detail=payload.get("detail") or payload.get("reason") or response.text,
                source="relay",
                retry_after_seconds=payload.get("retry_after_seconds"),
            ),
        )
    if response.is_error or not payload.get("ok"):
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason=payload.get("reason") or "relay_error",
                detail=payload.get("detail") or response.text,
                source="relay",
                retry_after_seconds=payload.get("retry_after_seconds"),
            ),
        )

    content = payload.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _save_downloaded_bytes(output_path: Path, data: bytes) -> str:
    suffix = output_path.suffix.lower()
    ext_format_map = Image.registered_extensions()
    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            save_format = ext_format_map.get(suffix, img.format)
            note = ""
            if img.format == "WEBP" or suffix == ".webp":
                output_path = output_path.with_suffix(".png")
                save_format = "PNG"
                note = " (converted from WEBP to PNG)"
            img.save(output_path, format=save_format)
            width, height = img.size
            return f"File downloaded to {output_path} (resolution: {width}x{height}){note}"
    except Exception:
        with open(output_path, "wb") as f:
            f.write(data)
    return f"File downloaded to {output_path}"


async def _download_via_relay(url: str, output_path: Path) -> str:
    timeout_s = _download_file_timeout_s()
    timeout = httpx.Timeout(timeout_s, connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s)
    relay_url = f"{_relay_base_url()}/v1/download"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.post(
                relay_url,
                headers=_relay_headers(),
                json={"url": url},
            )
    except httpx.TimeoutException as exc:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_timeout",
                detail=str(exc) or exc.__class__.__name__,
                source="relay",
            ),
        )
    except Exception as exc:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_unavailable",
                detail=str(exc),
                source="relay",
            ),
        )

    if response.headers.get("content-type", "").lower().startswith("application/json"):
        payload = _response_json(response)
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason=payload.get("reason") or "relay_error",
                detail=payload.get("detail") or response.text,
                source="relay",
                retry_after_seconds=payload.get("retry_after_seconds"),
            ),
        )
    if response.is_error:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="relay_http_error",
                detail=f"HTTP {response.status_code}",
                source="relay",
            ),
        )

    data = await response.aread()
    return _save_downloaded_bytes(output_path, data)


def _render_markdown_from_html(html: str, *, body_only: bool) -> str:
    markdown = markdownify.markdownify(html, heading_style=markdownify.ATX)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if body_only:
        result = extract(
            html,
            output_format="markdown",
            with_metadata=True,
            include_links=True,
            include_images=True,
            include_tables=True,
        )
        return result or markdown
    return markdown


async def _fetch_via_http(url: str, *, body_only: bool) -> str:
    timeout_s = _fetch_url_http_timeout_s()
    timeout = httpx.Timeout(timeout_s, connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": FAKE_UA.random},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    content_dispo = response.headers.get("Content-Disposition", "").lower()
    if "attachment" in content_dispo or "filename=" in content_dispo:
        return f"URL {url} is a downloadable file (Content-Disposition: {content_dispo})"
    if not content_type.startswith("text/html"):
        return f"URL {url} returned {content_type}, not a web page"
    return _render_markdown_from_html(response.text, body_only=body_only)


async def _fetch_via_playwright(url: str, *, body_only: bool) -> str:
    timeout_ms = int(_fetch_url_browser_timeout_s() * 1000)
    async with PlaywrightConverter() as converter:
        await converter.page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        html = await converter.page.content()
    return _render_markdown_from_html(html, body_only=body_only)


async def tavily_request(idx: int, params: dict) -> dict[str, Any]:
    """Send Tavily API request"""
    headers = {"Content-Type": "application/json", "User-Agent": FAKE_UA.random}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            TAVILY_API_URL, headers=headers, json=params
        ) as response:
            if response.status == 200:
                return await response.json()
            body = await response.text()
            if response.status == 429:
                await asyncio.sleep(idx * MAX_RETRY_INTERVAL)
            else:
                await asyncio.sleep(RETRY_TIMES)
            warning(f"TAVILY Error [{idx:02d}] [{response.status}] body={body}")
            response.raise_for_status()
        raise RuntimeError("TAVILY request failed after retries")


async def search_with_fallback(**kwargs) -> dict[str, Any]:
    tavily_keys = _tavily_keys()
    if not tavily_keys:
        raise RuntimeError(
            "MEMSLIDES_TAVILY_API_KEY is not configured; web search is unavailable."
        )

    last_error = None
    for idx, api_key in enumerate(tavily_keys, start=1):
        try:
            params = {**kwargs, "api_key": api_key}
            return await tavily_request(idx, params)
        except Exception as e:
            warning(f"TAVILY search error with key {api_key[:16]}...: {e}")
            last_error = e

    raise RuntimeError(
        f"TAVILY search failed after {len(tavily_keys)} retries"
    ) from last_error


def _unavailable_search_payload(query: str, reason: str, *, include_images: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "available": False,
        "error": reason,
        "total_results": 0,
    }
    if include_images:
        payload["images"] = []
    else:
        payload["results"] = []
    return payload


@mcp.tool()
async def search_web(
    query: str,
    max_results: int = 3,
    time_range: Literal["month", "year"] | None = None,
) -> dict:
    """
    Search the web

    Args:
        query: Search keywords
        max_results: Maximum number of search results, default 3
        time_range: Time range filter for search results, can be "month", "year", or None

    Returns:
        dict: Dictionary containing search results
    """
    kwargs = {"query": query, "max_results": max_results, "include_images": False}
    if time_range:
        kwargs["time_range"] = time_range

    try:
        result = await search_with_fallback(**kwargs)
    except Exception as exc:
        return _unavailable_search_payload(query, str(exc))

    results = [
        {
            "url": item["url"],
            "content": item["content"],
        }
        for item in result.get("results", [])
    ]

    return {
        "query": query,
        "available": True,
        "total_results": len(results),
        "results": results,
    }


@mcp.tool()
async def search_images(
    query: str,
) -> dict:
    """
    Search for web images
    """
    try:
        result = await search_with_fallback(
            query=query,
            max_results=4,
            include_images=True,
            include_image_descriptions=True,
        )
    except Exception as exc:
        return _unavailable_search_payload(query, str(exc), include_images=True)

    images = [
        {
            "url": img["url"],
            "description": img["description"],
        }
        for img in result.get("images", [])
    ]

    return {
        "query": query,
        "available": True,
        "total_results": len(images),
        "images": images,
    }


@mcp.tool()
async def fetch_url(url: str, body_only: bool = True) -> str:
    """
    Fetch web page content

    Args:
        url: Target URL
        body_only: If True, return only main content; otherwise return full page, default True
    """

    cached_failure = _get_cooldown(url)
    if cached_failure:
        return cached_failure

    if _should_use_relay(url):
        return await _fetch_via_relay(url, body_only=body_only)

    total_timeout_s = _fetch_url_total_timeout_s()

    async def _impl() -> str:
        try:
            result = await _fetch_via_http(url, body_only=body_only)
            if result.strip():
                return result
        except httpx.TimeoutException as exc:
            return _remember_cooldown(
                url,
                _failure_payload(
                    url=url,
                    reason="http_timeout",
                    detail=str(exc) or exc.__class__.__name__,
                    source="local",
                ),
            )
        except httpx.HTTPError:
            # Fall back to a real browser only when the site rejects or mangles
            # plain HTTP requests. Connectivity timeouts should fail fast above.
            pass
        except Exception:
            pass

        try:
            return await _fetch_via_playwright(url, body_only=body_only)
        except TimeoutError:
            return _remember_cooldown(
                url,
                _failure_payload(
                    url=url,
                    reason="browser_timeout",
                    detail=f"Playwright navigation exceeded {_fetch_url_browser_timeout_s():.0f}s",
                    source="local",
                ),
            )
        except Exception as exc:
            return _remember_cooldown(
                url,
                _failure_payload(
                    url=url,
                    reason="browser_unavailable",
                    detail=str(exc),
                    source="local",
                ),
            )

    try:
        return await asyncio.wait_for(_impl(), timeout=total_timeout_s)
    except asyncio.TimeoutError:
        return _remember_cooldown(
            url,
            _failure_payload(
                url=url,
                reason="timeout",
                detail=f"fetch_url exceeded {total_timeout_s:.0f}s total budget",
                source="local",
            ),
        )


@mcp.tool()
async def download_file(url: str, output_file: str) -> str:
    """
    Download a file from a URL and save it to a local path.
    """
    # Create directory if it doesn't exist
    workspace = Path(os.getcwd()).resolve()
    output_path = (workspace / output_file).resolve()
    if not output_path.is_relative_to(workspace):
        return f"Access denied: path outside allowed workspace: {workspace}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cached_failure = _get_cooldown(url)
    if cached_failure:
        return cached_failure

    if _should_use_relay(url):
        return await _download_via_relay(url, output_path)

    retries = _download_file_retries()
    timeout_s = _download_file_timeout_s()
    timeout = httpx.Timeout(timeout_s, connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s)
    last_error = ""
    last_reason = "download_failed"
    for retry in range(retries):
        try:
            if retry:
                await asyncio.sleep(min(retry, 2))
            async with httpx.AsyncClient(
                headers={"User-Agent": FAKE_UA.random},
                follow_redirects=True,
                verify=False,
                timeout=timeout,
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    data = await response.aread()
            return _save_downloaded_bytes(output_path, data)
        except httpx.TimeoutException as exc:
            last_reason = "http_timeout"
            last_error = f"{exc.__class__.__name__}: {exc}"
        except httpx.HTTPStatusError as exc:
            last_reason = "http_error"
            last_error = f"{exc.__class__.__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            last_reason = "download_failed"
            last_error = f"{exc.__class__.__name__}: {exc}"

    return _remember_cooldown(
        url,
        _failure_payload(
            url=url,
            reason=last_reason,
            detail=last_error or f"Failed to download file from {url}",
            source="local",
        ),
    )


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m memslides.tools.search_tools <workspace>"
    work_dir = Path(sys.argv[1])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(
        f"memslides-search-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_search_tools.log",
    )

    mcp.run(show_banner=False)
