import asyncio
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 300  # 5 minutes for offline API
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
POLL_INTERVAL = 5  # seconds
POLL_TIMEOUT = 600  # 10 minutes for online API
RESPONSE_PREVIEW_LIMIT = 500
DEFAULT_ONLINE_API_BASE_URL = "https://mineru.net/api/v4"


class PdfParserOperationError(RuntimeError):
    """RuntimeError carrying enough context to diagnose PDF parser failures."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        batch_id: str | None = None,
        status: int | None = None,
        state: str | None = None,
        err_msg: str | None = None,
        url: str | None = None,
        response_snippet: str | None = None,
    ) -> None:
        self.stage = stage
        self.batch_id = batch_id
        self.status = status
        self.state = state
        self.err_msg = err_msg
        self.url = url
        self.response_snippet = response_snippet

        details = [message, f"stage={stage}"]
        if batch_id:
            details.append(f"batch_id={batch_id}")
        if status is not None:
            details.append(f"http_status={status}")
        if state:
            details.append(f"state={state}")
        if err_msg:
            details.append(f"err_msg={_truncate(err_msg)}")
        if url:
            details.append(f"url={_sanitize_url(url)}")
        if response_snippet:
            details.append(f"response={_truncate(response_snippet)}")
        super().__init__(" | ".join(details))


def _truncate(text: str | None, limit: int = RESPONSE_PREVIEW_LIMIT) -> str:
    if not text:
        return ""
    clean = str(text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _sanitize_url(url: str | None) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    path = parts.path or ""
    return f"{parts.scheme}://{parts.netloc}{path}"


def _redact_token(token: str | None) -> str:
    if not token:
        return "<missing>"
    if len(token) <= 8:
        return "<redacted>"
    return f"{token[:4]}...{token[-4:]}"


def _normalize_online_api_base_url(api_base_url: str | None = None) -> str:
    raw = (
        api_base_url
        or os.getenv("MEMSLIDES_MINERU_API_URL")
        or os.getenv("MEMSLIDES_PDF_PARSER_API_URL")
        or DEFAULT_ONLINE_API_BASE_URL
    )
    base = str(raw).strip().rstrip("/")
    if not base:
        base = DEFAULT_ONLINE_API_BASE_URL
    for suffix in (
        "/file-urls/batch",
        "/extract-results/batch",
        "/extract/task",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
    return base


def _online_api_url(path: str, api_base_url: str | None = None) -> str:
    base = _normalize_online_api_base_url(api_base_url)
    return f"{base}/{path.lstrip('/')}"


def _json_preview(payload: object) -> str:
    try:
        return _truncate(json.dumps(payload, ensure_ascii=False))
    except (TypeError, ValueError):
        return _truncate(repr(payload))


async def _read_response_text(resp: aiohttp.ClientResponse) -> str:
    try:
        text = await resp.text()
    except UnicodeDecodeError:
        raw = await resp.read()
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive logging path
        return f"<unreadable response: {exc}>"
    return text


async def _read_json_response(
    resp: aiohttp.ClientResponse,
    *,
    stage: str,
    batch_id: str | None = None,
    url: str | None = None,
) -> dict:
    body = await _read_response_text(resp)
    if resp.status != 200:
        raise PdfParserOperationError(
            "PDF parser request returned non-200 response",
            stage=stage,
            batch_id=batch_id,
            status=resp.status,
            url=url,
            response_snippet=_truncate(body),
        )
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PdfParserOperationError(
            "PDF parser response was not valid JSON",
            stage=stage,
            batch_id=batch_id,
            status=resp.status,
            url=url,
            response_snippet=_truncate(body),
        ) from exc


async def parse_pdf_offline(
    pdf_path: str, output_path: str, url: str, timeout: int = DEFAULT_TIMEOUT
) -> None:
    """Parse PDF using a local/compatible PDF parser endpoint.

    Args:
        pdf_path: Path to PDF file
        output_path: Output directory
        url: PDF parser API endpoint URL
        timeout: Request timeout in seconds (default: 300)

    Raises:
        RuntimeError: If parsing fails after retries
        asyncio.TimeoutError: If request times out
    """
    os.makedirs(output_path, exist_ok=True)
    pdf_path_obj = Path(pdf_path)
    logger.info(
        "Parsing PDF offline: %s (size=%d bytes) via %s -> %s",
        pdf_path_obj.name,
        pdf_path_obj.stat().st_size,
        _sanitize_url(url),
        output_path,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                form = aiohttp.FormData()
                form.add_field(
                    "pdf",
                    pdf_path_obj.read_bytes(),
                    filename=pdf_path_obj.name,
                    content_type="application/pdf",
                )

                logger.debug(f"Attempt {attempt}/{MAX_RETRIES}: Uploading PDF")
                async with session.post(url, data=form) as resp:
                    if resp.status != 200:
                        await _raise_parsedoc_error(
                            resp,
                            stage="offline_parse",
                            url=url,
                        )
                    content = await resp.read()

            _extract_zip_bytes(content, output_path)
            logger.info(f"PDF parsed successfully: {pdf_path_obj.name}")
            return

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed to parse PDF after {MAX_RETRIES} attempts: {e}")
            await asyncio.sleep(RETRY_DELAY * attempt)


async def parse_pdf_online(
    pdf_path: str,
    output_path: str,
    token: str,
    model_version: str = "pipeline",
    poll_timeout: int = POLL_TIMEOUT,
    api_base_url: str | None = None,
) -> None:
    """Parse PDF using PDF parser external API

    Args:
        pdf_path: PDF file path
        output_path: Output directory
        token: API Token
        model_version: Model version (pipeline/vlm). pipeline is better for text-based PDFs, vlm for scanned images.
        poll_timeout: Maximum time to wait for result in seconds (default: 600)
        api_base_url: MinerU precise parsing API base URL. Defaults to https://mineru.net/api/v4.

    Raises:
        RuntimeError: If parsing fails or times out
    """
    os.makedirs(output_path, exist_ok=True)
    pdf_path = Path(pdf_path)
    batch_id: str | None = None
    logger.info(
        "Parsing PDF online: %s (size=%d bytes, model=%s) -> %s",
        pdf_path.name,
        pdf_path.stat().st_size,
        model_version,
        output_path,
    )
    logger.debug("PDF parser token fingerprint=%s", _redact_token(token))

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        ) as session:
            batch_id, upload_url, upload_headers = await _request_upload_url(
                session,
                pdf_path.name,
                pdf_path.stem[:128],
                model_version,
                token,
                api_base_url=api_base_url,
            )
            logger.debug(
                "PDF parser upload plan ready batch_id=%s upload_target=%s header_keys=%s",
                batch_id,
                _sanitize_url(upload_url),
                sorted(upload_headers.keys()) if upload_headers else [],
            )

            await _upload_file(
                session,
                upload_url,
                pdf_path,
                upload_headers,
                batch_id=batch_id,
            )
            logger.debug("File uploaded successfully for batch_id=%s", batch_id)

            zip_url = await _poll_result(
                session,
                batch_id,
                token,
                poll_timeout,
                api_base_url=api_base_url,
            )
            logger.debug(
                "PDF parser result ready batch_id=%s download_url=%s",
                batch_id,
                _sanitize_url(zip_url),
            )

            await _download_and_extract(session, zip_url, output_path, batch_id=batch_id)
            logger.info(f"PDF parsed successfully: {pdf_path.name}")
    except Exception as e:
        logger.exception(
            "Failed to parse PDF online for %s (batch_id=%s): %s",
            pdf_path.name,
            batch_id,
            e,
        )
        raise


async def _request_upload_url(
    session: aiohttp.ClientSession,
    filename: str,
    data_id: str,
    model_version: str,
    token: str,
    *,
    api_base_url: str | None = None,
) -> tuple[str, str, dict[str, str] | None]:
    """Request upload URL, returns (batch_id, upload_url, upload_headers)"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {
        "files": [{"name": filename, "data_id": data_id}],
        "model_version": model_version,
    }
    request_url = _online_api_url("file-urls/batch", api_base_url)

    async with session.post(
        request_url, headers=headers, json=payload
    ) as resp:
        result = await _read_json_response(
            resp,
            stage="request_upload_url",
            url=request_url,
        )
        if result.get("code") != 0:
            raise PdfParserOperationError(
                "Failed to request upload URL",
                stage="request_upload_url",
                status=resp.status,
                url=request_url,
                err_msg=result.get("msg", "Unknown error"),
                response_snippet=_json_preview(result),
            )

        data = result["data"]
        upload_headers = data.get("headers", [None])[0] if "headers" in data else None
        return data["batch_id"], data["file_urls"][0], upload_headers


async def _upload_file(
    session: aiohttp.ClientSession,
    upload_url: str,
    pdf_path: Path,
    headers: dict[str, str] | None = None,
    *,
    batch_id: str | None = None,
) -> None:
    """Upload PDF file to OSS"""
    file_data = pdf_path.read_bytes()

    upload_headers = headers if headers else {}

    async with session.put(
        upload_url,
        data=file_data,
        headers=upload_headers,
        skip_auto_headers={"Content-Type"},
    ) as resp:
        if resp.status >= 400:
            raise PdfParserOperationError(
                "Failed to upload PDF to PDF parser storage",
                stage="upload_file",
                batch_id=batch_id,
                status=resp.status,
                url=upload_url,
                response_snippet=_truncate(await _read_response_text(resp)),
            )


async def _poll_result(
    session: aiohttp.ClientSession,
    batch_id: str,
    token: str,
    timeout: int,
    *,
    api_base_url: str | None = None,
) -> str:
    """Poll parsing result, returns download URL

    Args:
        session: aiohttp session
        batch_id: Batch ID from upload
        token: API token
        timeout: Maximum wait time in seconds

    Returns:
        Download URL for result zip file

    Raises:
        RuntimeError: If parsing fails or times out
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = _online_api_url(f"extract-results/batch/{batch_id}", api_base_url)
    start_time = asyncio.get_event_loop().time()
    last_state: str | None = None

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise PdfParserOperationError(
                "Polling timeout exceeded",
                stage="poll_result",
                batch_id=batch_id,
                url=url,
                err_msg=f"Polling timeout after {timeout}s. Batch may still be processing.",
            )

        async with session.get(url, headers=headers) as resp:
            result = await _read_json_response(
                resp,
                stage="poll_result",
                batch_id=batch_id,
                url=url,
            )

            if result.get("code") != 0:
                raise PdfParserOperationError(
                    "Query failed",
                    stage="poll_result",
                    batch_id=batch_id,
                    status=resp.status,
                    url=url,
                    err_msg=result.get("msg", "Unknown error"),
                    response_snippet=_json_preview(result),
                )

            extract_results = result.get("data", {}).get("extract_result") or []
            if not extract_results:
                raise PdfParserOperationError(
                    "Query returned no extract_result entries",
                    stage="poll_result",
                    batch_id=batch_id,
                    status=resp.status,
                    url=url,
                    response_snippet=_json_preview(result),
                )

            extract = extract_results[0]
            state = extract.get("state", "<missing>")

            if state != last_state:
                logger.debug(
                    "PDF parser poll state changed batch_id=%s state=%s elapsed=%.1fs",
                    batch_id,
                    state,
                    elapsed,
                )
                last_state = state

            if state == "done":
                logger.info("Parsing completed in %.1fs for batch_id=%s", elapsed, batch_id)
                return extract["full_zip_url"]
            elif state == "failed":
                raise PdfParserOperationError(
                    "Parsing failed",
                    stage="poll_result",
                    batch_id=batch_id,
                    status=resp.status,
                    state=state,
                    url=url,
                    err_msg=extract.get("err_msg", "Unknown error"),
                    response_snippet=_json_preview(extract),
                )

            logger.debug(
                "Polling... batch_id=%s state=%s elapsed=%.1fs",
                batch_id,
                state,
                elapsed,
            )
            await asyncio.sleep(POLL_INTERVAL)


async def _download_and_extract(
    session: aiohttp.ClientSession,
    zip_url: str,
    output_path: str,
    *,
    batch_id: str | None = None,
) -> None:
    """Download and extract result"""
    async with session.get(zip_url) as resp:
        if resp.status != 200:
            raise PdfParserOperationError(
                "Failed to download PDF parser result zip",
                stage="download_result",
                batch_id=batch_id,
                status=resp.status,
                url=zip_url,
                response_snippet=_truncate(await _read_response_text(resp)),
            )
        content = await resp.read()
        logger.debug(
            "Downloaded PDF parser result zip batch_id=%s bytes=%d",
            batch_id,
            len(content),
        )

    _extract_zip_bytes(content, output_path)


async def _raise_parsedoc_error(
    resp: aiohttp.ClientResponse,
    *,
    stage: str,
    url: str | None = None,
) -> None:
    """Raise a RuntimeError with parsed error content."""
    payload = _truncate(await _read_response_text(resp))
    raise PdfParserOperationError(
        "PDF parser offline request failed",
        stage=stage,
        status=resp.status,
        url=url,
        response_snippet=payload,
    )


def _extract_zip_bytes(content: bytes, output_path: str) -> None:
    """Extract zip bytes into output_path with proper cleanup.

    Args:
        content: Zip file content as bytes
        output_path: Destination directory

    Raises:
        RuntimeError: If extraction fails
    """
    zip_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(content)
            zip_path = tmp.name

        logger.debug(f"Extracting zip to {output_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            all_names = [name for name in zip_ref.namelist() if name.strip()]
            top_level = {name.split("/", 1)[0] for name in all_names}

            if len(top_level) == 1 and all("/" in name for name in all_names):
                prefix = list(top_level)[0] + "/"
            else:
                prefix = ""

            for member in zip_ref.infolist():
                if not member.is_dir():
                    rel_path = (
                        member.filename.removeprefix(prefix) if prefix else member.filename
                    )
                    dest_path = os.path.join(output_path, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())

        logger.debug(f"Extracted {len(all_names)} files")

    except Exception as e:
        logger.error(f"Failed to extract zip: {e}")
        raise RuntimeError(f"Zip extraction failed: {e}")

    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {zip_path}: {e}")
