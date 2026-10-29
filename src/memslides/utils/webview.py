import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fake_useragent import UserAgent
from filelock import FileLock
from playwright.async_api import async_playwright
from pypdf import PdfWriter

from memslides.utils.constants import PACKAGE_DIR
from memslides.utils.log import error, info, warning

logger = logging.getLogger(__name__)

_PPTX_EXPORT_DIAGNOSTICS_PREFIX = "__MEMSLIDES_PPTX_EXPORT_DIAGNOSTICS__="
_PPTX_EXPORT_NODE_RUNTIME_DIRNAME = "node-runtime"
_PPTX_EXPORT_REQUIRED_NODE_MODULES = (
    "fast-glob",
    "minimist",
    "playwright",
    "pptxgenjs",
    "sharp",
)

# Playwright browsers are stored on the configured shared filesystem rather
# than a node-local browser cache.
_PLAYWRIGHT_BROWSERS_DIRNAME = "playwright-browsers"
_PLAYWRIGHT_BROWSER_DIR_ENV_KEYS = (
    "MEMSLIDES_PLAYWRIGHT_BROWSERS_PATH",
    "PLAYWRIGHT_BROWSERS_PATH",
)
_PLAYWRIGHT_EXECUTABLE_ENV_KEYS = (
    "MEMSLIDES_CHROMIUM_EXECUTABLE",
    "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
)


@dataclass(frozen=True)
class PptxExportNodeRuntime:
    node_binary: str
    cwd: Path
    env: dict[str, str]
    node_modules: Path | None
    source: str


def _resolve_node_binary() -> str:
    """Resolve the node executable without depending on shell PATH."""
    explicit = os.getenv("MEMSLIDES_NODE_BINARY", "").strip()
    candidates = []
    if explicit:
        candidates.append(explicit)

    for name in ("node", "nodejs"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    python_bin_dir = Path(sys.executable).resolve().parent
    candidates.append(str(python_bin_dir / "node"))

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "node"))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise RuntimeError(
        "Node.js executable not found. Install node.js in the current Python "
        "environment, ensure it is on PATH, or set MEMSLIDES_NODE_BINARY to "
        "an absolute node path."
    )


def _resolve_npm_binary() -> str:
    """Resolve npm for first-run pptx_export Node runtime installation."""
    explicit = os.getenv("MEMSLIDES_NPM_BINARY", "").strip()
    candidates = []
    if explicit:
        candidates.append(explicit)

    resolved = shutil.which("npm")
    if resolved:
        candidates.append(resolved)

    python_bin_dir = Path(sys.executable).resolve().parent
    candidates.append(str(python_bin_dir / "npm"))

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "npm"))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise RuntimeError(
        "npm executable not found. Install Node.js/npm in the current environment "
        "or set MEMSLIDES_NPM_BINARY to an absolute npm path."
    )


def _env_enabled(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _get_playwright_browsers_dir() -> Path:
    for key in _PLAYWRIGHT_BROWSER_DIR_ENV_KEYS:
        configured = os.getenv(key, "").strip()
        if configured:
            return Path(configured).expanduser()
    for key in ("MEMSLIDES_DEFAULT_CACHE_ROOT", "MEMSLIDES_DATA_ROOT"):
        configured = os.getenv(key, "").strip()
        if configured:
            return Path(configured).expanduser() / _PLAYWRIGHT_BROWSERS_DIRNAME
    return PACKAGE_DIR.parent.parent / f".{_PLAYWRIGHT_BROWSERS_DIRNAME}"


def _get_configured_playwright_executable() -> Path | None:
    for key in _PLAYWRIGHT_EXECUTABLE_ENV_KEYS:
        configured = os.getenv(key, "").strip()
        if configured:
            return Path(configured).expanduser()
    return None


def _iter_playwright_binary_candidates(browsers_dir: Path) -> Iterable[Path]:
    patterns = (
        "chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe",
        "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
        "chromium_headless_shell-*/chrome-headless-shell-mac-*/chrome-headless-shell",
        "chromium-*/chrome-win64/chrome.exe",
        "chromium-*/chrome-linux64/chrome",
        "chromium-*/chrome-mac-*/Chromium.app/Contents/MacOS/Chromium",
    )
    for pattern in patterns:
        for candidate in sorted(browsers_dir.glob(pattern), reverse=True):
            yield candidate


def _find_existing_playwright_binary() -> Path | None:
    explicit = _get_configured_playwright_executable()
    if explicit and explicit.exists():
        return explicit

    browsers_dir = _get_playwright_browsers_dir()
    for candidate in _iter_playwright_binary_candidates(browsers_dir):
        if candidate.exists():
            return candidate
    return None


def _get_playwright_env() -> dict[str, str]:
    """Return env dict with Playwright browser settings for local rendering."""
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(_get_playwright_browsers_dir())
    explicit = _get_configured_playwright_executable()
    if explicit:
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = str(explicit)
    return env


def _get_expected_playwright_binary() -> Path:
    existing = _find_existing_playwright_binary()
    if existing is not None:
        return existing
    return (
        _get_playwright_browsers_dir()
        / "chromium_headless_shell-1200"
        / "chrome-headless-shell-linux64"
        / "chrome-headless-shell"
    )


def should_auto_install_playwright() -> bool:
    return _env_enabled("MEMSLIDES_PLAYWRIGHT_AUTO_INSTALL", "1")


async def _ensure_playwright_browsers() -> None:
    """Install Playwright chromium browsers if the headless-shell binary is missing."""
    expected = _get_expected_playwright_binary()
    if expected.exists():
        return
    if not should_auto_install_playwright():
        raise RuntimeError(
            "Playwright chromium is missing and auto-install is disabled. "
            "Run `python -m playwright install chromium`, or set "
            "`MEMSLIDES_PLAYWRIGHT_AUTO_INSTALL=1` to allow automatic install."
        )
    info("Playwright chromium not found, auto-installing...")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "playwright", "install", "chromium",
        env=_get_playwright_env(),
        cwd=str(PACKAGE_DIR.parent.parent),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timeout_s = float(os.getenv("MEMSLIDES_PLAYWRIGHT_INSTALL_TIMEOUT", "600"))
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"Playwright chromium auto-install timed out after {timeout_s:.0f}s. "
            "Please install it manually with `python -m playwright install chromium`, "
            "or enable automatic install."
        ) from None
    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to auto-install Playwright chromium: {detail}")
    info(f"Playwright chromium installed to {_get_playwright_browsers_dir()}")


def _pptx_export_node_package_json_path() -> Path:
    return PACKAGE_DIR / "presentation_export" / "package.json"


def _pptx_export_node_package_json_text() -> str:
    path = _pptx_export_node_package_json_path()
    if not path.exists():
        raise FileNotFoundError(f"pptx_export Node dependency manifest not found at {path}")
    return path.read_text(encoding="utf-8")


def _pptx_export_node_dependency_hash() -> str:
    data = json.loads(_pptx_export_node_package_json_text())
    dependencies = data.get("dependencies") or {}
    payload = json.dumps(dependencies, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _get_default_cache_root() -> Path:
    configured = (
        os.getenv("MEMSLIDES_DEFAULT_CACHE_ROOT", "").strip()
        or os.getenv("MEMSLIDES_DATA_ROOT", "").strip()
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "memslides"


def _get_pptx_export_node_runtime_root() -> Path:
    configured = os.getenv("MEMSLIDES_NODE_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _get_default_cache_root() / _PPTX_EXPORT_NODE_RUNTIME_DIRNAME


def _get_pptx_export_node_runtime_dir() -> Path:
    return _get_pptx_export_node_runtime_root() / f"pptx_export-{_pptx_export_node_dependency_hash()}"


def _get_preconfigured_pptx_export_node_modules() -> Path | None:
    configured = os.getenv("MEMSLIDES_PPTX_EXPORT_NODE_MODULES", "").strip()
    if configured:
        return Path(configured).expanduser()
    return None


def should_auto_install_pptx_export_node_runtime() -> bool:
    return _env_enabled("MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL", "1")


def _merge_node_path(env: dict[str, str], node_modules: Path | None) -> dict[str, str]:
    merged = dict(env)
    if node_modules is not None:
        node_path_parts = [str(node_modules)]
        existing = merged.get("NODE_PATH", "").strip()
        if existing:
            node_path_parts.append(existing)
        merged["NODE_PATH"] = os.pathsep.join(node_path_parts)
    return merged


def _missing_pptx_export_node_modules(
    node_binary: str,
    *,
    cwd: Path,
    node_modules: Path | None = None,
    required_modules: Iterable[str] = _PPTX_EXPORT_REQUIRED_NODE_MODULES,
) -> tuple[str, ...]:
    modules = tuple(required_modules)
    script = """
const modules = JSON.parse(process.argv[1]);
const missing = [];
for (const name of modules) {
  try {
    require.resolve(name);
  } catch (error) {
    missing.push(name);
  }
}
process.stdout.write(JSON.stringify(missing));
"""
    env = _merge_node_path(os.environ.copy(), node_modules)
    try:
        proc = subprocess.run(
            [node_binary, "-e", script, json.dumps(modules)],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return modules
    if proc.returncode != 0:
        return modules
    try:
        parsed = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return modules
    if not isinstance(parsed, list):
        return modules
    missing = [str(item) for item in parsed if str(item) in modules]
    return tuple(missing)


def _format_pptx_export_node_runtime_error(missing_modules: Iterable[str], *, reason: str = "") -> str:
    missing = ", ".join(sorted(set(str(item) for item in missing_modules if str(item))))
    if not missing:
        missing = "unknown"
    details = f" ({reason})" if reason else ""
    return (
        f"pptx_export Node runtime dependencies are missing: {missing}{details}. "
        "MemSlides can auto-install them on first export when npm is available. "
        "Set MEMSLIDES_PPTX_EXPORT_AUTO_INSTALL=1 to allow automatic install, "
        "or set MEMSLIDES_PPTX_EXPORT_NODE_MODULES to a preinstalled node_modules directory. "
        "For offline setup, run npm install using the package manifest under "
        "memslides/presentation_export/package.json and point MEMSLIDES_PPTX_EXPORT_NODE_MODULES at it."
    )


def _extract_missing_node_modules_from_error(details: str) -> tuple[str, ...]:
    missing = re.findall(r"Cannot find module ['\"]([^'\"]+)['\"]", details or "")
    return tuple(name for name in missing if name in _PPTX_EXPORT_REQUIRED_NODE_MODULES)


def _install_pptx_export_node_runtime(runtime_dir: Path) -> None:
    npm = _resolve_npm_binary()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    package_json = runtime_dir / "package.json"
    package_json.write_text(_pptx_export_node_package_json_text(), encoding="utf-8")
    env = os.environ.copy()
    env["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
    timeout_s = float(os.getenv("MEMSLIDES_NODE_RUNTIME_INSTALL_TIMEOUT", "600"))
    proc = subprocess.run(
        [npm, "install", "--omit=dev", "--no-audit", "--no-fund"],
        cwd=str(runtime_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to install pptx_export Node runtime with npm at {runtime_dir}: {detail}"
        )


def _runtime_env_for_node_modules(node_modules: Path | None) -> dict[str, str]:
    return _merge_node_path(_get_playwright_env(), node_modules)


def _resolve_pptx_export_node_runtime(*, auto_install: bool | None = None) -> PptxExportNodeRuntime:
    node_binary = _resolve_node_binary()
    source_root = PACKAGE_DIR.parent.parent
    preconfigured_node_modules = _get_preconfigured_pptx_export_node_modules()
    if preconfigured_node_modules is not None:
        missing = _missing_pptx_export_node_modules(
            node_binary,
            cwd=preconfigured_node_modules.parent,
            node_modules=preconfigured_node_modules,
        )
        if missing:
            raise RuntimeError(
                _format_pptx_export_node_runtime_error(
                    missing,
                    reason=f"MEMSLIDES_PPTX_EXPORT_NODE_MODULES={preconfigured_node_modules}",
                )
            )
        return PptxExportNodeRuntime(
            node_binary=node_binary,
            cwd=preconfigured_node_modules.parent,
            env=_runtime_env_for_node_modules(preconfigured_node_modules),
            node_modules=preconfigured_node_modules,
            source="env",
        )

    missing = _missing_pptx_export_node_modules(node_binary, cwd=source_root)
    if not missing:
        return PptxExportNodeRuntime(
            node_binary=node_binary,
            cwd=source_root,
            env=_runtime_env_for_node_modules(None),
            node_modules=None,
            source="ambient",
        )

    runtime_dir = _get_pptx_export_node_runtime_dir()
    node_modules = runtime_dir / "node_modules"
    missing = _missing_pptx_export_node_modules(
        node_binary,
        cwd=runtime_dir,
        node_modules=node_modules,
    )
    if not missing:
        return PptxExportNodeRuntime(
            node_binary=node_binary,
            cwd=runtime_dir,
            env=_runtime_env_for_node_modules(node_modules),
            node_modules=node_modules,
            source="cache",
        )

    should_install = should_auto_install_pptx_export_node_runtime() if auto_install is None else auto_install
    if not should_install:
        raise RuntimeError(
            _format_pptx_export_node_runtime_error(
                missing,
                reason="automatic install is disabled",
            )
        )

    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(runtime_dir) + ".lock", timeout=300)
    with lock:
        missing = _missing_pptx_export_node_modules(
            node_binary,
            cwd=runtime_dir,
            node_modules=node_modules,
        )
        if missing:
            info(f"Installing pptx_export Node runtime to {runtime_dir}...")
            _install_pptx_export_node_runtime(runtime_dir)
            missing = _missing_pptx_export_node_modules(
                node_binary,
                cwd=runtime_dir,
                node_modules=node_modules,
            )
        if missing:
            raise RuntimeError(
                _format_pptx_export_node_runtime_error(
                    missing,
                    reason=f"npm install completed but runtime is still incomplete at {runtime_dir}",
                )
            )

    return PptxExportNodeRuntime(
        node_binary=node_binary,
        cwd=runtime_dir,
        env=_runtime_env_for_node_modules(node_modules),
        node_modules=node_modules,
        source="cache",
    )

FAKE_UA = UserAgent()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

CHROMIUM_LAUNCH_FLAGS = (
    "--allow-file-access-from-files",
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-sandbox",
)

BROWSER_INIT_SCRIPT = """
() => {
  Object.defineProperty(navigator, 'webdriver', { get: () => false });
  if (!window.chrome) window.chrome = { runtime: {} };
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
}
"""

ASPECT_RATIO_PIXELS = {
    "16:9": (1280, 720),
    "4:3": (960, 720),
    "A1": (2244, 3178),
    "A2": (1587, 2244),
    "A3": (1122, 1587),
    "A4": (794, 1123),
}

_FIGURE_IMAGE_FIX_STYLE_ID = "memslides-figure-image-fix"
_FIGURE_IMAGE_FIX_CSS = """
.figure, .row, .center {
    min-width: 0 !important;
    min-height: 0 !important;
}
.figure img, .row .figure img {
    display: block;
    width: auto;
    height: auto;
    max-width: 100% !important;
    max-height: 100% !important;
    min-width: 0;
    min-height: 0;
    object-fit: contain;
}
""".strip()

_INLINE_MARGIN_ERROR_MARKERS = (
    "has margin-left which is not supported in PowerPoint",
    "has margin-right which is not supported in PowerPoint",
)

_INLINE_TEXT_TAGS = {
    "span",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "code",
    "sup",
    "sub",
}


class PlaywrightConverter:
    _playwright = None
    _browser = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.context = None
        self.page = None

    async def __aenter__(self):
        """Async context manager entry"""
        async with PlaywrightConverter._lock:
            if PlaywrightConverter._browser is None:
                os.environ.update(_get_playwright_env())
                await _ensure_playwright_browsers()
                launch_kwargs: dict[str, Any] = {
                    "headless": True,
                    "args": list(CHROMIUM_LAUNCH_FLAGS),
                }
                executable = _find_existing_playwright_binary()
                if executable and executable.exists():
                    launch_kwargs["executable_path"] = str(executable)
                PlaywrightConverter._playwright = await async_playwright().start()
                PlaywrightConverter._browser = (
                    await PlaywrightConverter._playwright.chromium.launch(**launch_kwargs)
                )

        self.context = await PlaywrightConverter._browser.new_context(
            user_agent=FAKE_UA.random,
            bypass_csp=True,
        )
        await self.context.add_init_script(BROWSER_INIT_SCRIPT)
        self.page = await self.context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit, only close context"""
        if self.context:
            await self.context.close()

    async def convert_to_pdf(
        self,
        html_files: list[str | Path],
        output_pdf: Path | str,
        aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"],
        error_sink: list[str] | None = None,
    ) -> Path:
        if isinstance(output_pdf, str):
            output_pdf = Path(output_pdf)
        pdf_files = [tempfile.mkstemp(suffix=".pdf")[1] for _ in range(len(html_files))]
        folder = output_pdf.parent / f".slide_images-pdf-{output_pdf.stem}"
        folder.mkdir(exist_ok=True, parents=True)
        _fix_all_html_files(html_files)

        page = await self.context.new_page()
        width_px, height_px = ASPECT_RATIO_PIXELS[aspect_ratio]
        await page.set_viewport_size(
            {
                "width": width_px,
                "height": height_px,
            }
        )
        if error_sink is not None:
            page.on(
                "pageerror",
                lambda exc: error_sink.append(f"Page error: {exc}"),
            )
            page.on(
                "console",
                lambda msg: error_sink.append(f"Console error: {msg.text}")
                if msg.type == "error"
                else None,
            )
        try:
            page_timeout_ms = max(5_000, int(_env_float("MEMSLIDES_EXPORT_PAGE_TIMEOUT_SEC", 60.0) * 1000))
            for idx, (html, pdf) in enumerate(zip(sorted(html_files), pdf_files), start=1):
                await page.goto(
                    Path(html).resolve().as_uri(),
                    wait_until="networkidle",
                    timeout=page_timeout_ms,
                )
                # Preview images are used by local evaluation/debug views. Capture them
                # directly from the viewport. We also build the slide PDF from the
                # same rasterized viewport because Playwright's print pipeline can
                # drop or mis-layout local <img> content that renders correctly
                # on screen.
                slide_image = folder / f"slide_{idx:02d}.jpg"
                await page.screenshot(
                    path=str(slide_image),
                    type="jpeg",
                    quality=90,
                    timeout=page_timeout_ms,
                )
                _write_single_page_pdf_from_image(slide_image, Path(pdf))
        except Exception as e:
            error(f"Failed to convert HTML to PDF: {e}")
            raise e
        finally:
            await page.close()

        with PdfWriter() as merger:
            for pdf_file in pdf_files:
                merger.append(pdf_file)

            with open(output_pdf, "wb") as f:
                merger.write(f)
        info(f"Converted PDF saved at: {output_pdf}")
        return folder


def _parse_failing_html_path(error_message: str) -> str | None:
    """Extract the failing HTML file path from pptx_export error message."""
    match = re.search(r'(/\S+\.html)', error_message)
    return match.group(1) if match else None


def _extract_pptx_export_diagnostics(error_details: str) -> tuple[list[dict[str, Any]], str]:
    diagnostics: list[dict[str, Any]] = []
    cleaned_lines: list[str] = []
    for raw_line in str(error_details or "").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith(_PPTX_EXPORT_DIAGNOSTICS_PREFIX):
            cleaned_lines.append(raw_line)
            continue
        payload_text = stripped[len(_PPTX_EXPORT_DIAGNOSTICS_PREFIX) :].strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except Exception as exc:
            warning(f"Failed to parse pptx_export diagnostics payload: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        html_file = str(payload.get("html_file", "") or "")
        for item in payload.get("diagnostics", []) or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            if html_file and not entry.get("html_file"):
                entry["html_file"] = html_file
            diagnostics.append(entry)
    cleaned = "\n".join(cleaned_lines).strip()
    return diagnostics, cleaned


def _is_shrinkable_error(error_message: str) -> bool:
    """Check if the error is a layout overflow that can be fixed by shrinking.

    Note: 尺寸不匹配 ("don't match presentation layout") 不是压缩能解决的，
    需要 Agent 重新生成该页，不应在此处理。
    """
    return 'overflows body' in error_message or 'too close to bottom edge' in error_message


def _is_inline_margin_validation_error(error_message: str) -> bool:
    return any(marker in error_message for marker in _INLINE_MARGIN_ERROR_MARKERS)


def _has_numeric_css_value(value: str) -> bool:
    return bool(re.search(r"[-+]?\d", value or ""))


def _rewrite_inline_horizontal_margins(style_text: str) -> tuple[str, bool]:
    changed = False

    def _replace(match: re.Match) -> str:
        nonlocal changed
        side = match.group("side").strip().lower()
        value = match.group("value").strip()
        if not _has_numeric_css_value(value):
            return match.group(0)
        changed = True
        return f"padding-{side}:{value}; margin-{side}:0;"

    rewritten = re.sub(
        r"margin-(?P<side>left|right)\s*:\s*(?P<value>[^;}{]+)\s*;?",
        _replace,
        style_text,
        flags=re.IGNORECASE,
    )
    return rewritten, changed


def _selector_matches_inline_target(selectors: str, target_selectors: set[str]) -> bool:
    for selector in target_selectors:
        escaped = re.escape(selector)
        if selector.startswith((".", "#")):
            pattern = rf"(?<![\w-]){escaped}(?![\w-])"
        else:
            pattern = rf"(?<![\w-]){escaped}(?![\w-])"
        if re.search(pattern, selectors):
            return True
    return False


def _rewrite_style_block_inline_margins(
    css_text: str,
    target_selectors: set[str],
) -> tuple[str, bool]:
    changed = False

    def _replace_rule(match: re.Match) -> str:
        nonlocal changed
        selectors = match.group("selectors")
        body = match.group("body")
        if selectors.lstrip().startswith("@"):
            return match.group(0)
        if not _selector_matches_inline_target(selectors, target_selectors):
            return match.group(0)
        rewritten_body, body_changed = _rewrite_inline_horizontal_margins(body)
        if not body_changed:
            return match.group(0)
        changed = True
        return f"{selectors}{{{rewritten_body}}}"

    rewritten = re.sub(
        r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}",
        _replace_rule,
        css_text,
    )
    return rewritten, changed


def _fix_inline_horizontal_margins(html_path: Path) -> bool:
    """Rewrite inline-element horizontal margins into padding for pptx_export compatibility."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("bs4 is unavailable; cannot auto-fix inline margins for %s", html_path)
        return False

    original = html_path.read_text(encoding="utf-8")
    doctype_match = re.match(r"\s*(<!DOCTYPE[^>]+>)", original, flags=re.IGNORECASE)
    soup = BeautifulSoup(original, "html.parser")
    changed = False
    target_selectors: set[str] = set()

    for tag_name in _INLINE_TEXT_TAGS:
        for element in soup.find_all(tag_name):
            target_selectors.add(tag_name)
            for class_name in element.get("class", []) or []:
                if class_name:
                    target_selectors.add(f".{class_name}")
            element_id = element.get("id")
            if element_id:
                target_selectors.add(f"#{element_id}")
            style = element.get("style")
            if not style:
                continue
            rewritten_style, style_changed = _rewrite_inline_horizontal_margins(style)
            if not style_changed:
                continue
            element["style"] = rewritten_style
            changed = True

    for style_tag in soup.find_all("style"):
        css_text = style_tag.string if style_tag.string is not None else style_tag.get_text()
        if not css_text:
            continue
        rewritten_css, css_changed = _rewrite_style_block_inline_margins(
            css_text,
            target_selectors,
        )
        if not css_changed:
            continue
        style_tag.clear()
        style_tag.append(rewritten_css)
        changed = True

    if not changed:
        return False

    new_content = str(soup)
    if doctype_match and not new_content.lstrip().lower().startswith("<!doctype"):
        new_content = f"{doctype_match.group(1)}\n{new_content.lstrip()}"

    html_path.write_text(new_content, encoding="utf-8")
    warning(f"Auto-fixed inline horizontal margins in {html_path.name}")
    return True


def _fix_inline_margins_in_inputs(html_inputs: Path | str | Iterable[Path | str]) -> bool:
    fixed_any = False
    if isinstance(html_inputs, (str, Path)):
        input_path = Path(html_inputs)
        if input_path.is_dir():
            for html_file in sorted(input_path.glob("*.html")):
                fixed_any = _fix_inline_horizontal_margins(html_file) or fixed_any
        elif input_path.exists():
            fixed_any = _fix_inline_horizontal_margins(input_path) or fixed_any
        return fixed_any

    for item in html_inputs:
        item_path = Path(item)
        if item_path.is_dir():
            for html_file in sorted(item_path.glob("*.html")):
                fixed_any = _fix_inline_horizontal_margins(html_file) or fixed_any
        elif item_path.exists():
            fixed_any = _fix_inline_horizontal_margins(item_path) or fixed_any
    return fixed_any


def _collect_html_input_files(
    html_inputs: Path | str | Iterable[Path | str],
) -> list[Path]:
    html_files: list[Path] = []
    seen: set[str] = set()

    def _add_html_file(path: Path) -> None:
        if path.suffix.lower() != ".html":
            return
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        html_files.append(resolved)

    def _walk_input(path_like: Path | str) -> None:
        input_path = Path(path_like)
        if not input_path.exists():
            return
        if input_path.is_dir():
            for html_file in sorted(input_path.glob("*.html")):
                _add_html_file(html_file)
            return
        _add_html_file(input_path)

    if isinstance(html_inputs, (str, Path)):
        _walk_input(html_inputs)
        return html_files

    for item in html_inputs:
        _walk_input(item)
    return html_files


# ── Dimension auto-fix constants (aspect_ratio → expected body px) ──
_EXPECTED_BODY_PX: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "4:3": (960, 720),
    "A1": (2244, 3178),
    "A2": (1587, 2244),
    "A3": (1122, 1587),
    "A4": (794, 1123),
}


def auto_fix_body_dimensions(
    html_path: Path,
    aspect_ratio: str = "16:9",
) -> bool:
    """Patch the <body> CSS width/height to match the expected layout dimensions.

    Returns True if a fix was applied, False otherwise.
    """
    expected = _EXPECTED_BODY_PX.get(aspect_ratio)
    if not expected:
        return False
    exp_w, exp_h = expected
    content = html_path.read_text(encoding="utf-8")

    # Match body (or html,body) rule and fix width/height inside it
    _body_re = re.compile(r'((?:html\s*,\s*)?body\s*\{)([^}]*)\}', re.DOTALL)
    m = _body_re.search(content)
    if not m:
        return False

    rule_body = m.group(2)
    original = rule_body

    def _replace_or_insert_dimension(css_body: str, prop: str, value: int) -> str:
        pattern = re.compile(rf'({prop}\s*:\s*)[^;}}]+', re.IGNORECASE)
        if pattern.search(css_body):
            return pattern.sub(rf'\g<1>{value}px', css_body)
        return f"{prop}:{value}px;{css_body}"

    rule_body = _replace_or_insert_dimension(rule_body, "width", exp_w)
    rule_body = _replace_or_insert_dimension(rule_body, "height", exp_h)

    if rule_body == original:
        return False  # nothing changed

    new_content = content[:m.start(2)] + rule_body + content[m.end(2):]
    html_path.write_text(new_content, encoding="utf-8")
    warning(
        f"Auto-fixed body dimensions in {html_path.name} → {exp_w}×{exp_h}px"
    )
    return True


_IMAGE_HINT_TOKENS = (
    "img",
    "image",
    "figure",
    "photo",
    "picture",
    "graphic",
    "media",
    "illustration",
    "logo",
    "chart",
    "diagram",
    "thumb",
    "thumbnail",
)


def _token_looks_image_like(token: str) -> bool:
    normalized = str(token or "").strip().lower()
    if not normalized or normalized == "slide":
        return False
    for hint in _IMAGE_HINT_TOKENS:
        if (
            normalized == hint
            or normalized.startswith(f"{hint}-")
            or normalized.endswith(f"-{hint}")
            or f"-{hint}-" in normalized
            or normalized.startswith(hint)
            or normalized.endswith(hint)
        ):
            return True
    return False


def _selector_looks_image_like(selector_text: str) -> bool:
    normalized = str(selector_text or "").strip().lower()
    if not normalized:
        return False
    if re.search(r"(^|[^a-z0-9_-])img([^a-z0-9_-]|$)", normalized):
        return True
    tokens = re.findall(r"[a-z0-9_-]+", normalized)
    return any(_token_looks_image_like(token) for token in tokens)


def _tag_attrs_look_image_like(tag_name: str, attrs_text: str) -> bool:
    if tag_name == "img":
        return True
    normalized_attrs = str(attrs_text or "").lower()
    if "background-image" in normalized_attrs:
        return True
    tokens = re.findall(r"[a-z0-9_-]+", normalized_attrs)
    return any(_token_looks_image_like(token) for token in tokens)


def _scale_dimension_declarations(css_text: str, scale_factor: float) -> str:
    def _scale(match: re.Match) -> str:
        prop = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)
        return f"{prop}:{value * scale_factor:.1f}{unit}"

    css_text = re.sub(
        r'((?:max-)?width)\s*:\s*([\d.]+)(px|vh|vw)',
        _scale,
        css_text,
        flags=re.IGNORECASE,
    )
    css_text = re.sub(
        r'((?:max-)?height)\s*:\s*([\d.]+)(px|vh|vw)',
        _scale,
        css_text,
        flags=re.IGNORECASE,
    )
    return css_text


def _scale_image_like_css_rules(content: str, scale_factor: float) -> str:
    def _rewrite_rule(match: re.Match) -> str:
        selector = match.group(1)
        declarations = match.group(2)
        if not _selector_looks_image_like(selector):
            return match.group(0)
        return f"{selector}{{{_scale_dimension_declarations(declarations, scale_factor)}}}"

    return re.sub(r'([^{}]+)\{([^{}]*)\}', _rewrite_rule, content)


def _scale_image_like_inline_dimensions(content: str, scale_factor: float) -> str:
    style_attr_re = re.compile(
        r'(?P<prefix>\sstyle\s*=\s*)(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
        re.IGNORECASE | re.DOTALL,
    )

    def _scale_img_attr(match: re.Match) -> str:
        attr = match.group(1)
        quote = match.group(2)
        value = float(match.group(3))
        return f'{attr}={quote}{value * scale_factor:.0f}{quote}'

    def _rewrite_opening_tag(match: re.Match) -> str:
        tag_name = match.group("tag").lower()
        attrs = match.group("attrs")
        if not _tag_attrs_look_image_like(tag_name, attrs):
            return match.group(0)

        updated_attrs = attrs

        def _rewrite_style_attr(style_match: re.Match) -> str:
            scaled = _scale_dimension_declarations(style_match.group("value"), scale_factor)
            return (
                f'{style_match.group("prefix")}{style_match.group("quote")}'
                f'{scaled}{style_match.group("quote")}'
            )

        updated_attrs = style_attr_re.sub(_rewrite_style_attr, updated_attrs)
        if tag_name == "img":
            updated_attrs = re.sub(
                r'(width)\s*=\s*(["\'])([\d.]+)\2',
                _scale_img_attr,
                updated_attrs,
                flags=re.IGNORECASE,
            )
            updated_attrs = re.sub(
                r'(height)\s*=\s*(["\'])([\d.]+)\2',
                _scale_img_attr,
                updated_attrs,
                flags=re.IGNORECASE,
            )
        return f"<{match.group('tag')}{updated_attrs}>"

    return re.sub(
        r'<(?P<tag>[a-zA-Z][a-zA-Z0-9:-]*)(?P<attrs>[^<>]*?)>',
        _rewrite_opening_tag,
        content,
    )


def shrink_images_only(html_path: Path, scale_factor: float = 0.70) -> None:
    """Shrink only image dimensions (CSS + HTML attributes) without touching text.

    This is the first-priority overflow fix: images can tolerate more aggressive
    scaling than text before readability suffers.

    Only image-like selectors/elements are touched so slide canvas rules such
    as `.slide { width: 1280px; height: 720px; }` are preserved.
    """
    content = html_path.read_text(encoding='utf-8')
    content = _scale_image_like_css_rules(content, scale_factor)
    content = _scale_image_like_inline_dimensions(content, scale_factor)
    html_path.write_text(content, encoding='utf-8')


def shrink_text_only(html_path: Path, scale_factor: float = 0.90) -> None:
    """Shrink only font-size and line-height without touching image dimensions.

    This is the second-priority overflow fix, applied only when image shrinking
    alone was not enough to resolve the overflow.
    """
    content = html_path.read_text(encoding='utf-8')

    def _scale(match: re.Match) -> str:
        prop = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)
        return f'{prop}:{value * scale_factor:.1f}{unit}'

    content = re.sub(r'(font-size)\s*:\s*([\d.]+)(pt|px)', _scale, content)
    content = re.sub(r'(line-height)\s*:\s*([\d.]+)(pt|px)', _scale, content)

    html_path.write_text(content, encoding='utf-8')


def repair_slide_layout_for_export(
    html_inputs: Path | str | Iterable[Path | str],
    *,
    aspect_ratio: str = "16:9",
    failure_message: str = "",
) -> int:
    """Apply deterministic, conservative layout repairs before relaxed export."""
    expected = _EXPECTED_BODY_PX.get(aspect_ratio, _EXPECTED_BODY_PX["16:9"])
    exp_w, exp_h = expected
    css = f"""
html, body {{
    width: {exp_w}px !important;
    height: {exp_h}px !important;
    margin: 0 !important;
    overflow: hidden !important;
}}
body, body * {{
    box-sizing: border-box !important;
}}
body {{
    max-width: {exp_w}px !important;
    max-height: {exp_h}px !important;
}}
section, .slide, .page, .canvas, .deck-slide {{
    max-width: {exp_w}px !important;
    max-height: {exp_h}px !important;
    overflow: hidden !important;
}}
img, svg, canvas, video {{
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
}}
""".strip()
    patched = 0
    for html_path in _iter_html_paths(html_inputs):
        try:
            before = html_path.read_text(encoding="utf-8")
            auto_fix_body_dimensions(html_path, aspect_ratio)
            shrink_images_only(html_path, 0.82)
            shrink_text_only(html_path, 0.94)
            content = html_path.read_text(encoding="utf-8")
            content = _inject_repair_style(content, css)
            if content != before:
                html_path.write_text(content, encoding="utf-8")
                patched += 1
        except Exception as exc:  # noqa: BLE001
            warning(f"Deterministic layout repair failed for {html_path}: {exc}")
    if patched:
        warning(
            "Applied deterministic export layout repair to %s slide(s)%s",
            patched,
            f" after: {failure_message.splitlines()[0][:160]}" if failure_message else "",
        )
    return patched


def _inject_repair_style(content: str, css: str) -> str:
    style = f'<style id="memslides-export-layout-repair">\n{css}\n</style>'
    if 'id="memslides-export-layout-repair"' in content:
        return re.sub(
            r'<style\s+id=["\']memslides-export-layout-repair["\'][^>]*>.*?</style>',
            style,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
    head_match = re.search(r"</head\s*>", content, flags=re.IGNORECASE)
    if head_match:
        return content[:head_match.start()] + style + "\n" + content[head_match.start():]
    return style + "\n" + content


def auto_shrink_slide_html(html_path: Path, scale_factor: float = 0.90) -> None:
    """Reduce font-size, line-height, and image dimensions in an HTML slide to fix overflow.

    IMPORTANT: Image dimension shrinking is limited to image-like selectors and
    elements so slide canvas dimensions (e.g. 1280×720 for 16:9) are never
    accidentally shrunk.

    NOTE: For the graduated shrink strategy (images first, then text), use
    :func:`shrink_images_only` and :func:`shrink_text_only` separately via
    :func:`convert_html_to_pptx_with_retry`. This function is kept for
    backward compatibility and shrinks everything at once.
    """
    content = html_path.read_text(encoding='utf-8')

    def _scale(match: re.Match) -> str:
        prop = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)
        return f'{prop}:{value * scale_factor:.1f}{unit}'

    # Text scaling (never appears in body rule, but safe anyway)
    content = re.sub(r'(font-size)\s*:\s*([\d.]+)(pt|px)', _scale, content)
    content = re.sub(r'(line-height)\s*:\s*([\d.]+)(pt|px)', _scale, content)

    content = _scale_image_like_css_rules(content, scale_factor)
    content = _scale_image_like_inline_dimensions(content, scale_factor)
    html_path.write_text(content, encoding='utf-8')


def _fix_unwrapped_text_in_div(html_path: Path) -> bool:
    """Fix DIV elements with unwrapped text by wrapping them in <p> tags.

    pptx_export requires all text to be wrapped in <p>, <h1>-<h6>, <ul>, or <ol>.
    This function fixes common patterns like <div class="x">text</div>.

    Returns True if any fixes were applied.
    """
    content = html_path.read_text(encoding='utf-8')
    original = content

    # Pattern: <div ...>bare text</div> where bare text has no child tags
    # We need to wrap the text in <p>
    def wrap_text_in_p(match):
        opening_tag = match.group(1)
        inner = match.group(2)
        # Check if inner already has block-level tags
        if re.search(r'<(p|h[1-6]|ul|ol|li|div|section|article|header|footer|nav|aside|table)\b', inner, re.I):
            return match.group(0)  # Already has proper tags
        # Wrap bare text in <p>
        inner_stripped = inner.strip()
        if inner_stripped and not inner_stripped.startswith('<'):
            return f'{opening_tag}<p>{inner}</p></div>'
        return match.group(0)

    # Match <div ...>content</div> patterns
    content = re.sub(
        r'(<div[^>]*>)([^<]+)(</div>)',
        lambda m: f'{m.group(1)}<p>{m.group(2)}</p>{m.group(3)}',
        content
    )

    if content != original:
        html_path.write_text(content, encoding='utf-8')
        logger.info(f"Fixed unwrapped text in {html_path.name}")
        return True
    return False


def _ensure_figure_image_containment(html_path: Path) -> bool:
    """Inject containment CSS for common figure wrappers generated by slide HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("bs4 is unavailable; cannot auto-fix figure images for %s", html_path)
        return False

    original = html_path.read_text(encoding="utf-8")
    doctype_match = re.match(r"\s*(<!DOCTYPE[^>]+>)", original, flags=re.IGNORECASE)
    soup = BeautifulSoup(original, "html.parser")

    if not soup.select(".figure img, .row .figure img"):
        return False

    if soup.find("style", attrs={"id": _FIGURE_IMAGE_FIX_STYLE_ID}):
        return False

    style_tag = soup.new_tag("style", id=_FIGURE_IMAGE_FIX_STYLE_ID)
    style_tag.string = _FIGURE_IMAGE_FIX_CSS

    if soup.head is not None:
        soup.head.append(style_tag)
    elif soup.html is not None:
        soup.html.insert(0, style_tag)
    else:
        soup.insert(0, style_tag)

    new_content = str(soup)
    if doctype_match and not new_content.lstrip().lower().startswith("<!doctype"):
        new_content = f"{doctype_match.group(1)}\n{new_content.lstrip()}"

    if new_content == original:
        return False

    html_path.write_text(new_content, encoding="utf-8")
    warning(f"Auto-fixed figure image containment in {html_path.name}")
    return True


def _iter_html_paths(html_inputs: Path | str | Iterable[Path | str]) -> list[Path]:
    html_paths: list[Path] = []
    if isinstance(html_inputs, (str, Path)):
        input_path = Path(html_inputs)
        if input_path.is_dir():
            html_paths.extend(sorted(input_path.glob("*.html")))
        elif input_path.exists():
            html_paths.append(input_path)
        return html_paths

    for item in html_inputs:
        item_path = Path(item)
        if item_path.is_dir():
            html_paths.extend(sorted(item_path.glob("*.html")))
        elif item_path.exists():
            html_paths.append(item_path)
    return html_paths


def _write_single_page_pdf_from_image(image_path: Path, pdf_path: Path) -> None:
    """Render a viewport screenshot as a single-page PDF at CSS-pixel slide size."""
    import fitz
    from PIL import Image

    with Image.open(image_path) as image:
        width_px, height_px = image.size

    # Chrome CSS pixels map to 96 DPI. Convert to PDF points so a 1280x720
    # screenshot becomes the standard 13.333" x 7.5" slide.
    width_pt = width_px * 72 / 96
    height_pt = height_px * 72 / 96

    pdf = fitz.open()
    page = pdf.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()


def _fix_all_html_files(html_inputs: Path | str | Iterable[Path | str]) -> None:
    """Apply HTML auto-fixes that stabilize both screen and export rendering."""
    for html_file in _iter_html_paths(html_inputs):
        _fix_inline_horizontal_margins(html_file)
        _fix_unwrapped_text_in_div(html_file)
        _ensure_figure_image_containment(html_file)


def _split_html_input_for_cli(
    html_inputs: Path | str | Iterable[Path | str],
) -> tuple[Path | None, list[Path]]:
    if isinstance(html_inputs, (str, Path)):
        candidates = [Path(html_inputs)]
    else:
        candidates = [Path(item) for item in html_inputs]

    if not candidates:
        raise ValueError("No HTML inputs provided")

    html_dir: Path | None = None
    html_files: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"HTML input does not exist: {candidate}")
        if candidate.is_dir():
            if html_dir is not None or html_files:
                raise ValueError("html_inputs cannot mix directories and files")
            html_dir = candidate.resolve()
        else:
            html_files.append(candidate.resolve())
    return html_dir, html_files


async def convert_html_to_pptx(
    html_inputs: Path | str | Iterable[Path | str],
    output_pptx: Path | str | None = None,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
    skip_layout_validation: bool = False,
) -> Path:
    # Auto-fix unwrapped text in DIV elements before conversion
    _fix_all_html_files(html_inputs)

    script_path = PACKAGE_DIR / "presentation_export" / "export_cli.js"
    if not script_path.exists():
        raise FileNotFoundError(f"pptx_export CLI not found at {script_path}")

    if output_pptx is None:
        fd, temp_path = tempfile.mkstemp(suffix=".pptx")
        os.close(fd)
        output_path = Path(temp_path)
    else:
        output_path = Path(output_pptx)

    html_dir, html_files = _split_html_input_for_cli(html_inputs)
    node_runtime = _resolve_pptx_export_node_runtime()
    cmd = [node_runtime.node_binary, str(script_path), "--layout", aspect_ratio]
    if skip_layout_validation:
        cmd.append("--skip-layout-validation")
    if html_dir is not None:
        cmd.extend(["--html_dir", str(html_dir.resolve())])
    else:
        for html_file in html_files:
            cmd.extend(["--html", str(html_file)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output", str(output_path)])

    await _ensure_playwright_browsers()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        env=node_runtime.env,
        cwd=node_runtime.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(
            "pptx_export timed out after 120s — likely Chromium hung during rendering. "
            "Check slide HTML for complex/broken content."
        )
    if process.returncode != 0:
        details = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()
        diagnostics, cleaned_details = _extract_pptx_export_diagnostics(details)
        display_details = cleaned_details or details
        missing_node_modules = _extract_missing_node_modules_from_error(details)
        if missing_node_modules:
            raise RuntimeError(_format_pptx_export_node_runtime_error(missing_node_modules))
        error = RuntimeError(f"pptx_export failed: {display_details.split('at pptx_export (')[0].strip()}")
        if diagnostics:
            setattr(error, "pptx_export_diagnostics", diagnostics)
        raise error

    return output_path


async def convert_html_to_pptx_with_retry(
    html_inputs: Path | str | Iterable[Path | str],
    output_pptx: Path | str | None = None,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
    max_retries: int = 3,
    image_scale_factor: float = 0.70,
    text_scale_factor: float = 0.90,
    session_id: str = "",
    experience_writer: Any = None,
    allow_skip_layout_validation_fallback: bool = True,
    # Deprecated — kept for backward compatibility; ignored when both
    # image_scale_factor and text_scale_factor are provided.
    scale_factor: float | None = None,
    preserve_source_html: bool = False,
) -> Path:
    """Convert HTML to PPTX, auto-shrinking content on layout validation errors.

    Graduated shrink strategy (images first, text second):
      1. Try normal conversion with layout validation.
      2. On overflow, first shrink *images only* (up to ``image_retries`` times).
         Images tolerate more aggressive scaling before readability suffers.
      3. If still overflowing, shrink *text* (font-size & line-height) while
         keeping images at their already-reduced size.
      4. If all retries fail, do a final attempt with layout validation
         skipped so a .pptx is always produced.

    The total number of retry attempts is ``max_retries``.  The first half
    (rounded up) targets images; the remainder targets text.
    """
    # Backward compat: if caller passes old `scale_factor`, use it for both
    if scale_factor is not None:
        image_scale_factor = scale_factor
        text_scale_factor = scale_factor

    # Split retries: images get the first half (rounded up), text gets the rest
    image_retries = (max_retries + 1) // 2  # e.g. 2 out of 3
    text_retries = max_retries - image_retries  # e.g. 1 out of 3

    last_error: Exception | None = None
    original_html_by_path: dict[Path, str] = {}
    if preserve_source_html:
        for html_file in _collect_html_input_files(html_inputs):
            try:
                original_html_by_path[html_file] = html_file.read_text(encoding="utf-8")
            except Exception as e:
                warning(f"Failed to snapshot source HTML for {html_file}: {e}")

    def _apply_shrink(html_file: Path, attempt: int) -> str:
        """Apply the appropriate shrink strategy based on attempt number.

        Returns a human-readable label for logging.
        """
        if attempt < image_retries:
            shrink_images_only(html_file, image_scale_factor)
            return f"images-only (scale={image_scale_factor})"
        else:
            shrink_text_only(html_file, text_scale_factor)
            return f"text-only (scale={text_scale_factor})"

    try:
        for attempt in range(max_retries + 1):
            try:
                return await convert_html_to_pptx(
                    html_inputs, output_pptx, aspect_ratio,
                )
            except RuntimeError as e:
                last_error = e
                error_msg = str(e)

                if _is_inline_margin_validation_error(error_msg):
                    failing_path = _parse_failing_html_path(error_msg)
                    fixed = False
                    if failing_path and Path(failing_path).exists():
                        fixed = _fix_inline_horizontal_margins(Path(failing_path))
                    else:
                        fixed = _fix_inline_margins_in_inputs(html_inputs)
                    if fixed:
                        warning(
                            "Auto-fixed inline margin compatibility issue for pptx_export, retrying conversion"
                        )
                        try:
                            return await convert_html_to_pptx(
                                html_inputs, output_pptx, aspect_ratio,
                            )
                        except RuntimeError as retry_e:
                            last_error = retry_e
                            error_msg = str(retry_e)

                # 尺寸不匹配：先尝试自动修正 body CSS，修正后重试一次
                if "don't match presentation layout" in error_msg:
                    failing_path = _parse_failing_html_path(error_msg)
                    if failing_path and Path(failing_path).exists():
                        fixed = auto_fix_body_dimensions(
                            Path(failing_path), aspect_ratio
                        )
                        if fixed:
                            # 修正成功，立即重试（不计入 shrink 次数）
                            warning(
                                f"Auto-fixed dimension mismatch in "
                                f"{Path(failing_path).name}, retrying conversion"
                            )
                            try:
                                return await convert_html_to_pptx(
                                    html_inputs, output_pptx, aspect_ratio,
                                )
                            except RuntimeError as retry_e:
                                # 修正尺寸后仍有其他错误（如 overflow），
                                # 让它走正常的 shrink 重试流程
                                last_error = retry_e
                                error_msg = str(retry_e)
                                if "don't match presentation layout" not in error_msg:
                                    # 尺寸已修正，但有 overflow 等可压缩错误
                                    if _is_shrinkable_error(error_msg) and attempt < max_retries:
                                        fp = _parse_failing_html_path(error_msg)
                                        if fp and Path(fp).exists():
                                            label = _apply_shrink(Path(fp), attempt)
                                            info(
                                                f"Layout retry after dimension fix: auto-shrank {Path(fp).name} "
                                                f"({label}, attempt {attempt + 1}/{max_retries})"
                                            )
                                        continue
                                # 尺寸修正后仍然尺寸不匹配，放弃

                    # 记录到 ExperienceTrace
                    if experience_writer and session_id:
                        try:
                            from .layout_overflow_recorder import record_dimension_mismatch
                            if failing_path and Path(failing_path).exists():
                                await record_dimension_mismatch(
                                    html_path=Path(failing_path),
                                    session_id=session_id,
                                    experience_writer=experience_writer,
                                    error_msg=error_msg,
                                )
                        except Exception as rec_err:
                            warning(f"Failed to record dimension mismatch: {rec_err}")
                    # 自动修正失败，抛出清晰错误让 Agent 重新生成
                    raise RuntimeError(
                        f"HTML aspect ratio mismatch - regenerate slide with correct dimensions. "
                        f"File: {failing_path or 'unknown'}. {error_msg}"
                    ) from e

                if not _is_shrinkable_error(error_msg):
                    raise

                if attempt >= max_retries:
                    break

                failing_path = _parse_failing_html_path(error_msg)
                if failing_path and Path(failing_path).exists():
                    label = _apply_shrink(Path(failing_path), attempt)
                    info(
                        f"Layout retry: auto-shrank {Path(failing_path).name} "
                        f"({label}, attempt {attempt + 1}/{max_retries})"
                    )
                else:
                    input_path = (
                        Path(html_inputs) if isinstance(html_inputs, (str, Path)) else None
                    )
                    if input_path and input_path.is_dir():
                        for html_file in sorted(input_path.glob("*.html")):
                            _apply_shrink(html_file, attempt)
                        phase = "images" if attempt < image_retries else "text"
                        info(
                            f"Layout retry: auto-shrank all slides ({phase}) in {input_path.name} "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                    else:
                        raise

        if not allow_skip_layout_validation_fallback:
            if last_error is not None:
                raise last_error
            raise RuntimeError("pptx_export layout validation failed after retries")

        repair_slide_layout_for_export(
            html_inputs,
            aspect_ratio=aspect_ratio,
            failure_message=str(last_error or ""),
        )
        try:
            return await convert_html_to_pptx(
                html_inputs,
                output_pptx,
                aspect_ratio,
            )
        except RuntimeError as repair_e:
            last_error = repair_e

        # All shrink retries exhausted — skip layout validation as final fallback
        warning(
            "Layout validation errors persist after graduated shrink retries "
            f"(images×{image_retries} + text×{text_retries}), "
            "converting with layout validation skipped"
        )

        # 记录布局溢出到 ExperienceTrace
        if experience_writer and session_id:
            try:
                from .layout_overflow_recorder import record_layout_overflow
                failing_path = _parse_failing_html_path(str(last_error)) if last_error else None
                if failing_path and Path(failing_path).exists():
                    await record_layout_overflow(
                        html_path=Path(failing_path),
                        session_id=session_id,
                        experience_writer=experience_writer,
                        shrink_attempts=max_retries,
                        scale_factor=image_scale_factor,
                    )
            except Exception as e:
                warning(f"Failed to record layout overflow (non-fatal): {e}")

        return await convert_html_to_pptx(
            html_inputs, output_pptx, aspect_ratio, skip_layout_validation=True,
        )
    finally:
        if preserve_source_html:
            for html_file, original_html in original_html_by_path.items():
                try:
                    current_html = html_file.read_text(encoding="utf-8") if html_file.exists() else ""
                    if current_html != original_html:
                        html_file.write_text(original_html, encoding="utf-8")
                except Exception as e:
                    warning(f"Failed to restore source HTML for {html_file}: {e}")
