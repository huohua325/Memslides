import base64
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

# Ensure direct script execution imports the local repo package first.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from fastmcp import FastMCP
from PIL import Image

try:
    from markitdown import MarkItDown
except ImportError as exc:  # pragma: no cover - optional dependency
    MarkItDown = None  # type: ignore[assignment]
    _MARKITDOWN_IMPORT_ERROR = str(exc)
else:
    _MARKITDOWN_IMPORT_ERROR = ""

from memslides.utils.log import info, set_logger, warning
from memslides.utils.pdf_parser_client import parse_pdf_offline, parse_pdf_online
from memslides.utils.pdf_image_extractor import (
    FigureInfo,
    build_pdf_parser_to_hd_mapping,
    extract_hd_images,
    parse_caption_label,
)

mcp = FastMCP(name="MemSlidesDocumentTools")

IMAGE_EXTENSIONS = [
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
]
MINERU_API_URL = os.getenv("MEMSLIDES_MINERU_API_URL", None)
MINERU_API_KEY = os.getenv("MEMSLIDES_PDF_PARSER_API_KEY", None)
PDF_CONVERSION_BACKEND = os.getenv(
    "MEMSLIDES_PDF_CONVERSION_BACKEND", "auto"
).strip().lower()
MINERU_REQUEST_TIMEOUT_SEC = int(
    os.getenv("MEMSLIDES_MINERU_REQUEST_TIMEOUT_SEC", "180")
)
MINERU_POLL_TIMEOUT_SEC = int(
    os.getenv("MEMSLIDES_MINERU_POLL_TIMEOUT_SEC", "180")
)
# 启用高清图片提取（默认开启）
EXTRACT_HD_IMAGES = (
    os.getenv("MEMSLIDES_EXTRACT_HD_IMAGES", "true").lower() == "true"
)
FIGURE_MANIFEST = "figure_manifest.json"


def _slugify_output_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return slug or "document"


def _pdf_parser_backend_mode() -> str:
    if MINERU_API_KEY:
        return "online_api_key"
    if MINERU_API_URL:
        return "compatible_endpoint"
    return "unconfigured"


def _resolve_workspace_output_folder(file_path: str, output_folder: str) -> Path:
    requested = Path(output_folder).expanduser()
    workspace = os.getenv("MEMSLIDES_WORKSPACE", "").strip()
    if not workspace:
        return requested.resolve()

    workspace_path = Path(workspace).expanduser().resolve()
    try:
        resolved_requested = requested.resolve()
    except FileNotFoundError:
        resolved_requested = (Path.cwd() / requested).resolve()

    if workspace_path == resolved_requested or workspace_path in resolved_requested.parents:
        return resolved_requested

    safe_name = _slugify_output_name(Path(file_path).stem)
    return (workspace_path / "converted" / safe_name).resolve()


def _default_workspace_output_folder(file_path: str) -> Path:
    workspace = os.getenv("MEMSLIDES_WORKSPACE", "").strip()
    base = Path(workspace).expanduser() if workspace else Path.cwd()
    safe_name = _slugify_output_name(Path(file_path).stem)
    return (base / "converted" / safe_name).resolve()


def _find_pdf_parser_json(output_path: Path, stem: str, suffix: str) -> Path | None:
    """查找PDF parser输出的JSON文件（可能带原文件名前缀）"""
    # 尝试 stem_suffix.json 格式（如 MemoBrain_model.json）
    candidate = output_path / f"{stem}_{suffix}.json"
    if candidate.exists():
        return candidate
    # 尝试 suffix.json 格式
    candidate = output_path / f"{suffix}.json"
    if candidate.exists():
        return candidate
    # 模糊搜索
    for f in output_path.glob(f"*{suffix}*.json"):
        return f
    return None


def _figure_manifest_records(figures: list[FigureInfo], images_dir: Path) -> list[dict]:
    records = []
    for fig in figures:
        label_kind, label_num, caption_text = parse_caption_label(fig.caption)
        filename = Path(fig.image_path).name
        records.append({
            "filename": filename,
            "path": str((images_dir / filename).resolve()),
            "page": fig.page_num + 1,
            "category": fig.category,
            "caption": caption_text,
            "label": f"{label_kind.title()} {label_num}" if label_kind and label_num is not None else "",
            "label_number": label_num,
            "pdf_parser_img_name": fig.pdf_parser_img_name,
            "width": fig.width,
            "height": fig.height,
        })
    records.sort(key=lambda item: (item["page"], item["label_number"] or 10**9, item["filename"]))
    return records


def _write_figure_manifest(converted_dir: Path, figures: list[FigureInfo]) -> None:
    images_dir = converted_dir / "images"
    manifest_path = converted_dir / FIGURE_MANIFEST
    records = _figure_manifest_records(figures, images_dir)
    manifest_path.write_text(
        json.dumps({"figures": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_pdf_parser_json_files(output_dir: Path) -> None:
    """Remove intermediate PDF parser JSONs while preserving exported figure metadata."""
    for f in output_dir.glob("*.json"):
        if f.name == FIGURE_MANIFEST:
            continue
        try:
            os.remove(f)
        except OSError:
            pass


def _matched_figures_from_mapping(
    figures: list[FigureInfo],
    mapping: dict[str, str],
) -> list[FigureInfo]:
    matched = []
    seen_pairs: set[tuple[str, str]] = set()

    for fig in figures:
        pdf_parser_name = fig.pdf_parser_img_name
        if not pdf_parser_name:
            continue
        if mapping.get(pdf_parser_name) != fig.image_path:
            continue
        pair = (pdf_parser_name, fig.image_path)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        matched.append(fig)

    return matched


def _install_exact_hd_images(
    pdf_parser_images_dir: Path,
    matched_figures: list[FigureInfo],
) -> None:
    pdf_parser_images_dir.mkdir(parents=True, exist_ok=True)

    for fig in matched_figures:
        src = Path(fig.image_path)
        if not src.exists():
            continue

        dst = pdf_parser_images_dir / src.name
        shutil.copy2(src, dst)

        old_thumb = pdf_parser_images_dir / fig.pdf_parser_img_name if fig.pdf_parser_img_name else None
        if old_thumb and old_thumb != dst and old_thumb.exists():
            old_thumb.unlink()


def _clear_directory_contents(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def _normalize_local_image_paths(markdown: str, output_path: Path) -> str:
    for match in re.findall(r"!\[.*?\]\((.*?)\)", markdown):
        local_path = match.split()[0].strip("\"'")
        p = Path(local_path)
        if (output_path / local_path).exists():
            p = output_path / local_path
        if p.exists():
            markdown = markdown.replace(local_path, str(p.resolve()))
    return markdown


def _summarize_images(output_path: Path) -> str:
    images = output_path.glob("images/*")
    images_with_info = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                images_with_info.append((img_path, *img.size))
        except Exception:
            continue

    images_with_info.sort(key=lambda x: int(x[1]), reverse=True)
    return "Found {count} images\n{details}".format(
        count=len(images_with_info),
        details="".join(
            [f"- {img[0]}: {img[1]}x{img[2]}\n" for img in images_with_info]
        ),
    )


async def _convert_pdf_with_pdf_parser(
    file_path: str,
    output_path: Path,
    markdown_file: Path,
    pdf_stem: str,
) -> tuple[str, list[FigureInfo]]:
    info(
        "Converting PDF via PDF parser backend (request_timeout=%ss, poll_timeout=%ss)",
        MINERU_REQUEST_TIMEOUT_SEC,
        MINERU_POLL_TIMEOUT_SEC,
    )
    if MINERU_API_KEY:
        await parse_pdf_online(
            file_path,
            str(output_path),
            MINERU_API_KEY,
            poll_timeout=MINERU_POLL_TIMEOUT_SEC,
            api_base_url=MINERU_API_URL,
        )
    elif MINERU_API_URL:
        await parse_pdf_offline(
            file_path,
            str(output_path),
            MINERU_API_URL,
            timeout=MINERU_REQUEST_TIMEOUT_SEC,
        )
    else:
        raise RuntimeError("PDF parser backend requested but no PDF parser credential is configured.")

    model_json_path = None
    content_list_json_path = None
    layout_json_path = None
    for f in output_path.glob("*"):
        if f.name.lower().endswith(".md"):
            os.rename(f, str(markdown_file))
        elif f.name.lower().endswith(".pdf"):
            os.remove(f)
        elif f.name.lower().endswith(".json"):
            fname_lower = f.name.lower()
            if "model" in fname_lower:
                model_json_path = f
            elif "content_list" in fname_lower:
                content_list_json_path = f
            elif "layout" in fname_lower:
                layout_json_path = f

    if not model_json_path:
        model_json_path = _find_pdf_parser_json(output_path, pdf_stem, "model")
    if not content_list_json_path:
        content_list_json_path = _find_pdf_parser_json(output_path, pdf_stem, "content_list")
    if not layout_json_path:
        layout_json_path = _find_pdf_parser_json(output_path, pdf_stem, "layout")

    if not markdown_file.exists():
        raise RuntimeError("PDF parser conversion did not produce a markdown file.")

    markdown = markdown_file.read_text(encoding="utf-8")

    matched_figures: list[FigureInfo] = []
    if EXTRACT_HD_IMAGES:
        hd_images_dir = output_path / "images_hd"
        try:
            figures = extract_hd_images(
                file_path,
                str(hd_images_dir),
                pdf_parser_model_json=str(model_json_path) if model_json_path else None,
                pdf_parser_layout_json=str(layout_json_path) if layout_json_path else None,
                pdf_parser_content_list_json=str(content_list_json_path) if content_list_json_path else None,
                min_size=100,
                dpi=300,
            )
            if figures:
                mapping = build_pdf_parser_to_hd_mapping(figures, output_path / "images")
                matched_figures = _matched_figures_from_mapping(figures, mapping)
                if matched_figures:
                    _install_exact_hd_images(output_path / "images", matched_figures)
                    _write_figure_manifest(output_path, matched_figures)
                markdown = _replace_with_hd_images(
                    markdown,
                    output_path / "images",
                    hd_images_dir,
                    figures,
                    mapping=mapping,
                )
        except Exception as exc:
            warning(f"Failed to extract HD images from PDF parser output: {exc}")
        finally:
            if hd_images_dir.exists():
                shutil.rmtree(hd_images_dir, ignore_errors=True)

    _cleanup_pdf_parser_json_files(output_path)
    return markdown, matched_figures


def _convert_with_markitdown(
    file_path: str,
    output_path: Path,
) -> tuple[str, list[FigureInfo]]:
    if MarkItDown is None:
        raise RuntimeError(f"markitdown is unavailable: {_MARKITDOWN_IMPORT_ERROR}")

    info("Converting document via MarkItDown fallback.")
    convert_result = MarkItDown().convert_local(file_path, keep_data_uris=True)
    markdown = parse_base64_images(
        convert_result.text_content, output_path / "images"
    )

    figures: list[FigureInfo] = []
    if file_path.lower().endswith(".pdf") and EXTRACT_HD_IMAGES:
        try:
            figures = extract_hd_images(
                file_path,
                str(output_path / "images"),
                min_size=100,
                dpi=300,
            )
            if figures:
                _write_figure_manifest(output_path, figures)
        except Exception as exc:
            warning(f"Failed to extract local PDF images during MarkItDown fallback: {exc}")

    return markdown, figures


async def _convert_document_to_markdown(
    file_path: str | None = None,
    output_folder: str | None = None,
    filePath: str | None = None,
    outputFolder: str | None = None,
) -> dict:
    """Convert a file to markdown, it could accept pdf, docx, doc, etc.
    Args:
        file_path: The path of the file to be converted
        output_folder: The folder to save the converted markdown and images, should be empty or not exist

    Returns:
        The converted results, with file saved to the specified path
    """
    # Be tolerant to camelCase argument drift from tool-calling models.
    file_path = file_path or filePath
    output_folder = output_folder or outputFolder

    if not file_path:
        return {
            "success": False,
            "error": "Error: missing required argument `file_path`",
        }
    warnings: list[str] = []
    if not output_folder:
        output_path = _default_workspace_output_folder(file_path)
        warnings.append(
            f"`output_folder` was omitted; defaulted to {output_path}."
        )
    else:
        output_path = _resolve_workspace_output_folder(file_path, output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    markdown_file = output_path / f"{Path(file_path).stem}.md"
    if len(os.listdir(output_path)) != 0:
        if markdown_file.exists():
            warnings.append("Output folder already contained converted markdown; reused cached conversion.")
            return {
                "success": True,
                "markdown_file": str(markdown_file),
                "images": _summarize_images(output_path),
                "backend_requested": PDF_CONVERSION_BACKEND,
                "backend_used": "cached",
                "fallback_used": False,
                "warnings": warnings,
                "cached": True,
            }
        return {
            "success": False,
            "error": "Error: output folder should be empty or not exist",
        }
    if not os.path.exists(file_path):
        return {"success": False, "error": f"Error: file {file_path} does not exist"}

    pdf_stem = Path(file_path).stem
    backend_requested = PDF_CONVERSION_BACKEND
    backend_used = "markitdown"
    fallback_used = False
    file_size = Path(file_path).stat().st_size

    info(
        "convert_to_markdown start file=%s output_dir=%s backend_requested=%s file_size=%d",
        file_path,
        output_path,
        backend_requested,
        file_size,
    )

    try:
        if file_path.lower().endswith(".pdf"):
            should_try_pdf_parser = backend_requested in {"auto", "pdf_parser"} and (
                MINERU_API_KEY or MINERU_API_URL
            )
            if should_try_pdf_parser:
                info(
                    "Trying PDF parser for %s using mode=%s",
                    file_path,
                    _pdf_parser_backend_mode(),
                )
                try:
                    markdown, _ = await _convert_pdf_with_pdf_parser(
                        file_path=file_path,
                        output_path=output_path,
                        markdown_file=markdown_file,
                        pdf_stem=pdf_stem,
                    )
                    backend_used = "pdf_parser"
                except Exception as exc:
                    if backend_requested == "pdf_parser":
                        raise
                    fallback_used = True
                    warning(
                        "PDF parser PDF conversion failed for %s using mode=%s, falling back to MarkItDown: %s: %s",
                        file_path,
                        _pdf_parser_backend_mode(),
                        exc.__class__.__name__,
                        exc,
                    )
                    warnings.append(f"PDF parser fallback: {exc}")
                    _clear_directory_contents(output_path)
                    markdown, _ = _convert_with_markitdown(file_path, output_path)
                    backend_used = "markitdown"
            else:
                info(
                    "Skipping PDF parser for %s because backend_requested=%s and mode=%s",
                    file_path,
                    backend_requested,
                    _pdf_parser_backend_mode(),
                )
                markdown, _ = _convert_with_markitdown(file_path, output_path)
                backend_used = "markitdown"
        else:
            markdown, _ = _convert_with_markitdown(file_path, output_path)
            backend_used = "markitdown"
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "skipped": False,
            "backend_requested": backend_requested,
            "backend_used": backend_used,
            "fallback_used": fallback_used,
            "error": str(exc),
        }

    markdown = _normalize_local_image_paths(markdown, output_path)
    with open(str(markdown_file), "w", encoding="utf-8") as f:
        f.write(markdown)

    return {
        "success": True,
        "markdown_file": str(markdown_file),
        "images": _summarize_images(output_path),
        "backend_requested": backend_requested,
        "backend_used": backend_used,
        "fallback_used": fallback_used,
        "warnings": warnings,
    }


@mcp.tool()
async def convert_to_markdown(
    file_path: str | None = None,
    output_folder: str | None = None,
    filePath: str | None = None,
    outputFolder: str | None = None,
) -> dict:
    return await _convert_document_to_markdown(
        file_path=file_path,
        output_folder=output_folder,
        filePath=filePath,
        outputFolder=outputFolder,
    )


@mcp.tool()
async def list_document_figures(converted_dir: str) -> dict:
    """List all figures and tables extracted from a converted document with HD image paths and metadata.

    Args:
        converted_dir: The converted document directory (containing images/ and markdown)

    Returns:
        dict with figure list including paths, dimensions, and captions
    """
    converted = Path(converted_dir)
    images_dir = converted / "images"
    if not images_dir.exists():
        return {"success": False, "error": "No images directory found", "figures": []}

    # 从 markdown 中提取 caption 信息
    captions = {}
    for md_file in converted.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        # 匹配 ![...](path)\nFigure N: caption 或 Table N: caption
        for m in re.finditer(
            r"!\[[^\]]*\]\(([^)]+)\)\s*\n\s*((?:Figure|Table|图|表)\s*\d+[^.\n]*\.?[^\n]*)",
            content,
        ):
            img_ref = m.group(1)
            caption = m.group(2).strip()
            img_name = Path(img_ref).name
            captions[img_name] = caption

    manifest_data = {}
    manifest_path = converted / FIGURE_MANIFEST
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in raw.get("figures", []):
                filename = item.get("filename")
                if filename:
                    manifest_data[filename] = item
        except json.JSONDecodeError:
            warning(f"Failed to parse figure manifest: {manifest_path}")

    figure_list = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            continue
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        manifest_item = manifest_data.get(img_path.name, {})
        caption = captions.get(img_path.name) or manifest_item.get("caption", "")
        label_kind, label_num, _ = parse_caption_label(caption)
        figure_list.append({
            "path": str(img_path.resolve()),
            "filename": img_path.name,
            "width": w,
            "height": h,
            "caption": caption,
            "page": manifest_item.get("page"),
            "category": manifest_item.get("category", label_kind or ""),
            "label": manifest_item.get("label") or (f"{label_kind.title()} {label_num}" if label_kind and label_num is not None else ""),
            "label_number": manifest_item.get("label_number", label_num),
        })

    figure_list.sort(
        key=lambda x: (
            x.get("page") if x.get("page") is not None else 10**9,
            x.get("label_number") if x.get("label_number") is not None else 10**9,
            x["filename"],
        )
    )

    return {
        "success": True,
        "total": len(figure_list),
        "figures": figure_list,
    }


def parse_base64_images(markdown: str, image_dir: Path) -> str:
    """Save base64 images to local, and convert those links to local paths"""
    image_dir.mkdir(exist_ok=True, parents=True)
    for image_match in re.finditer(
        r"!\[([^\]]*)\]\((data:image/([^;]+);base64,([^)]+))\)", markdown
    ):
        _, data_uri, image_format, base64_data = image_match.groups()

        if image_format.lower() not in IMAGE_EXTENSIONS:
            markdown = markdown.replace(image_match.group(0), "")
            warning(f"Unsupported image format: {image_format}, image will be ignored")
            continue

        image_data = base64.b64decode(base64_data)
        image_path = image_dir / (uuid.uuid4().hex[:8] + "." + image_format)

        with open(image_path, "wb") as f:
            f.write(image_data)

        # Replace data URI with relative path
        markdown = markdown.replace(data_uri, str(image_path))

    return markdown


def _replace_with_hd_images(
    markdown: str,
    pdf_parser_images_dir: Path,
    hd_images_dir: Path,
    figures: list[FigureInfo],
    mapping: dict[str, str] | None = None,
) -> str:
    """
    将Markdown中的PDF parser缩略图替换为高清图片

    只使用精确映射；若缺少精确映射，则保留原 PDF parser 缩略图，
    避免再靠附近 caption 做语义猜测而把图重新绑错。
    """
    mapping = mapping or build_pdf_parser_to_hd_mapping(figures, pdf_parser_images_dir)
    pattern = re.compile(
        r'!\[([^\]]*)\]\(([^)]+)\)(\s*\n\s*((?:Figure|Table|图|表)\s*\d+[^.\n]*\.?[^\n]*))?'
    )

    def _resolve_hd_path(img_basename: str, caption_text: str) -> str | None:
        return mapping.get(img_basename)

    def _rewrite(match: re.Match) -> str:
        alt_text = match.group(1)
        img_ref = match.group(2)
        suffix = match.group(3) or ""
        img_basename = Path(img_ref).name
        caption_text = match.group(4) or ""
        hd_path = _resolve_hd_path(img_basename, caption_text)
        if not hd_path:
            return match.group(0)
        new_ref = f"images/{Path(hd_path).name}"
        return f"![{alt_text}]({new_ref}){suffix}"

    return pattern.sub(_rewrite, markdown)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m memslides.tools.document_tools <workspace>"
    work_dir = Path(sys.argv[1])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(
        f"memslides-document-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_document_tools.log",
    )

    mcp.run(show_banner=False)
