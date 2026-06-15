from __future__ import annotations

import mimetypes
import os
import re
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from memslides.utils.webview import (
    _find_existing_playwright_binary,
    _resolve_pptx_export_node_runtime,
    _resolve_node_binary,
)

OPERATIONS_MARKER = ".operations"
MAX_ARTIFACTS = 300

ALLOWED_SUFFIXES = {
    ".html",
    ".css",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".pptx",
    ".pdf",
    ".sqlite",
    ".db",
}


@dataclass(frozen=True)
class ServedFile:
    path: Path
    media_type: str


def path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_workspace_file(workspace: Path, rel_path: str) -> ServedFile:
    candidate = (workspace / rel_path).resolve()
    if not path_within(candidate, workspace):
        raise ValueError("File is outside the session workspace")
    if ".runtime_secrets" in candidate.parts:
        raise ValueError("Runtime secret files are not downloadable")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"File type is not allowed: {candidate.suffix}")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(rel_path)
    media_type = _media_type_for(candidate)
    return ServedFile(path=candidate, media_type=media_type)


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if suffix == ".pdf":
        return "application/pdf"
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def list_slides(workspace: Path, session_id: str) -> list[dict[str, Any]]:
    slide_dirs = _candidate_slide_dirs(workspace)
    slides: list[dict[str, Any]] = []
    for slide_dir in slide_dirs:
        for html_file in sorted(slide_dir.glob("*.html")):
            rel = html_file.relative_to(workspace).as_posix()
            slides.append(
                {
                    "name": html_file.name,
                    "path": rel,
                    "url": f"/api/sessions/{session_id}/preview?{urlencode({'path': rel})}",
                    "updated_at": html_file.stat().st_mtime,
                    "size": html_file.stat().st_size,
                }
            )
        if slides:
            break
    return slides


def list_artifacts(workspace: Path, session_id: str) -> dict[str, Any]:
    files = _collect_workspace_files(workspace, session_id)
    prioritized = _prioritize_artifacts(files, limit=MAX_ARTIFACTS)
    version_history = _build_version_history(files)
    current_deck = _build_current_deck(files, workspace)
    inputs = _build_inputs_summary(workspace, session_id)
    visible_count = (
        len(current_deck.get("files") or [])
        + sum(len(group.get("files") or []) for group in version_history)
        + sum(len(group.get("files") or []) for group in inputs)
    )
    return {
        "workspace": str(workspace),
        "slides": list_slides(workspace, session_id),
        "current_deck": current_deck,
        "version_history": version_history,
        "inputs": inputs,
        "files": prioritized,
        "file_count": len(files),
        "visible_file_count": visible_count,
        "capabilities": detect_capabilities(),
    }


def list_files_tree(workspace: Path, session_id: str, *, root: str, rel_path: str = "") -> dict[str, Any]:
    root_name = str(root or "").strip().lower()
    root_dir = _tree_root_dir(workspace, root_name)
    target = _resolve_tree_target(root_dir, rel_path)
    children = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if _skip_tree_entry(child, workspace, root_name):
            continue
        stat = child.stat()
        child_rel = child.relative_to(root_dir).as_posix()
        workspace_rel = child.relative_to(workspace).as_posix()
        kind = "dir" if child.is_dir() else _artifact_kind(child)
        node: dict[str, Any] = {
            "name": child.name,
            "path": child_rel,
            "workspace_path": workspace_rel,
            "kind": kind,
            "is_dir": child.is_dir(),
            "size": 0 if child.is_dir() else stat.st_size,
            "updated_at": stat.st_mtime,
        }
        if child.is_file() and child.suffix.lower() in ALLOWED_SUFFIXES:
            node["download_url"] = f"/api/files?session_id={session_id}&path={workspace_rel}&download=1"
        children.append(node)

    return {
        "root": root_name,
        "path": "" if target == root_dir else target.relative_to(root_dir).as_posix(),
        "children": children,
    }


def list_templates(cache_root: Path) -> list[dict[str, Any]]:
    templates_dir = cache_root / "templates"
    if not templates_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    template_dirs: list[Path] = []
    for candidate in sorted(templates_dir.rglob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not candidate.is_dir():
            continue
        if (candidate / "analysis.json").exists() or (candidate / "slide_induction.json").exists():
            template_dirs.append(candidate)
    for template_dir in template_dirs:
        analysis = template_dir / "analysis.json"
        induction = template_dir / "slide_induction.json"
        source = template_dir / "source.pptx"
        description = template_dir / "description.txt"
        summary = template_summary(template_dir)
        owner_user_id = "system"
        if "templates" in template_dir.parts:
            try:
                templates_index = template_dir.parts.index("templates")
                owner_user_id = template_dir.parts[templates_index + 1] if len(template_dir.parts) > templates_index + 2 else "system"
            except Exception:
                owner_user_id = "system"
        results.append(
            {
                "id": template_dir.name,
                "name": _read_first_line(description) or template_dir.name,
                "path": str(template_dir),
                "owner_user_id": owner_user_id,
                "visibility": "system" if owner_user_id == "system" else "private",
                "has_analysis": analysis.exists(),
                "has_induction": induction.exists(),
                "source_pptx": str(source) if source.exists() else "",
                "updated_at": template_dir.stat().st_mtime,
                **summary,
            }
        )
    return results[:100]


def template_summary(template_dir: Path) -> dict[str, Any]:
    analysis = _read_json(template_dir / "analysis.json")
    induction = _read_json(template_dir / "slide_induction.json")
    layout_keys: list[str] = []
    if isinstance(induction, dict):
        layout_keys = [
            key for key, value in induction.items()
            if key not in {"functional_keys", "language", "layout_capabilities"} and isinstance(value, dict)
        ]
    image_count = len(list((template_dir / "images").glob("*"))) if (template_dir / "images").exists() else 0
    return {
        "slide_count": int(analysis.get("slide_count") or analysis.get("total_slides") or 0) if isinstance(analysis, dict) else 0,
        "layout_count": len(layout_keys),
        "image_count": image_count,
        "aspect_ratio": str(analysis.get("aspect_ratio") or "") if isinstance(analysis, dict) else "",
        "language": (
            str((induction.get("language") or {}).get("lid") or "")
            if isinstance(induction, dict) and isinstance(induction.get("language"), dict)
            else ""
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def detect_capabilities() -> dict[str, dict[str, str]]:
    export = detect_export_capabilities()
    return {
        "web_search": _capability(bool(os.getenv("MEMSLIDES_TAVILY_API_KEY")), "MEMSLIDES_TAVILY_API_KEY not set"),
        "docker": _capability(_which("docker"), "docker executable not found"),
        "office_export": _capability(_which("soffice") or _which("unoconvert"), "soffice/unoconvert not found"),
        "playwright": export["playwright_chromium"],
        "node": export["node"],
        "pptxgenjs": export["pptxgenjs"],
        "pdf_screenshot_export": export["pdf_screenshot_export"],
    }


@lru_cache(maxsize=1)
def detect_export_capabilities() -> dict[str, dict[str, str]]:
    node_ok = False
    node_message = "Node.js executable not found"
    node_path = ""
    try:
        node_path = _resolve_node_binary()
        node_ok = True
        node_message = node_path
    except Exception as exc:  # noqa: BLE001
        node_message = str(exc)

    pptx_ok = False
    pptx_message = "Node.js is unavailable"
    if node_ok:
        try:
            runtime = _resolve_pptx_export_node_runtime(auto_install=False)
            pptx_ok = True
            pptx_message = f"pptx_export Node runtime resolved from {runtime.source}"
        except Exception as exc:  # noqa: BLE001
            pptx_message = str(exc)

    chromium = _find_existing_playwright_binary()
    chromium_ok = bool(chromium and chromium.exists())
    chromium_message = str(chromium) if chromium_ok else "Playwright Chromium is not installed yet"
    try:
        import PIL  # noqa: F401
        import fitz  # noqa: F401

        pdf_ok = chromium_ok
        pdf_message = "Screenshot PDF export prerequisites are available" if pdf_ok else chromium_message
    except Exception as exc:  # noqa: BLE001
        pdf_ok = False
        pdf_message = str(exc)

    return {
        "node": _capability(node_ok, node_message),
        "pptxgenjs": _capability(pptx_ok, pptx_message),
        "playwright_chromium": _capability(chromium_ok, chromium_message),
        "cjk_fonts": _capability(True, "CJK font fallback is configured in pptx_export"),
        "pdf_screenshot_export": _capability(pdf_ok, pdf_message),
    }


def _candidate_slide_dirs(workspace: Path) -> list[Path]:
    return [workspace / "outputs", workspace / "slides", workspace]


def count_current_slides(workspace: Path) -> int:
    for slide_dir in _candidate_slide_dirs(workspace):
        if not slide_dir.exists() or not slide_dir.is_dir():
            continue
        count = len(list(slide_dir.glob("*.html")))
        if count:
            return count
    return 0


def _collect_workspace_files(workspace: Path, session_id: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    stale_exports = _stale_current_exports(workspace)
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if any(part in {".locks", ".runtime_secrets", "__pycache__"} for part in path.parts):
            continue
        if _is_internal_runtime_artifact(path):
            continue
        if ".interrupted" in path.parts and path.suffix.lower() in {".pptx", ".pdf"}:
            continue
        rel = path.relative_to(workspace).as_posix()
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": rel,
                "kind": _artifact_kind(path),
                "stable": _is_stable_download(path),
                "stale": str(path.resolve()) in stale_exports,
                "snapshot_label": _snapshot_label(path),
                "size": stat.st_size,
                "updated_at": stat.st_mtime,
                "download_url": f"/api/files?session_id={session_id}&path={rel}&download=1",
            }
        )
    return files


def _build_current_deck(files: list[dict[str, Any]], workspace: Path) -> dict[str, Any]:
    live_exports = [
        file for file in files
        if _is_live_deck_export(file) and not file.get("stable")
    ]
    pptx = _pick_best_export([file for file in live_exports if file.get("kind") == "pptx"])
    pdf = _pick_best_export([file for file in live_exports if file.get("kind") == "pdf"])
    status = "waiting"
    if pptx or pdf:
        status = "ready"
        if (pptx and pptx.get("stale")) or (pdf and pdf.get("stale")):
            status = "previous"
    html_slides_count = count_current_slides(workspace)
    files_list = [item for item in [pptx, pdf] if item]
    latest_time = max((float(item.get("updated_at") or 0.0) for item in files_list), default=0.0)
    return {
        "pptx": pptx,
        "pdf": pdf,
        "html_slides_count": html_slides_count,
        "status": status if files_list else ("html_ready" if html_slides_count else "waiting"),
        "updated_at": latest_time or None,
        "files": files_list,
    }


def _build_version_history(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable = [file for file in files if file.get("stable") and file.get("kind") in {"pptx", "pdf"}]
    grouped = _group_snapshot_versions(stable)
    if grouped:
        return grouped

    fallback = [file for file in files if file.get("stale") and file.get("kind") in {"pptx", "pdf"} and not file.get("stable")]
    if not fallback:
        return []
    return [{
        "version_label": "Previous deck",
        "snapshot_label": "previous_live_exports",
        "updated_at": max(float(file.get("updated_at") or 0.0) for file in fallback),
        "pptx": _pick_best_export([file for file in fallback if file.get("kind") == "pptx"]),
        "pdf": _pick_best_export([file for file in fallback if file.get("kind") == "pdf"]),
        "files": _pick_version_files(fallback),
    }]


def _build_inputs_summary(workspace: Path, session_id: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    upload_files = _collect_group_files(workspace, session_id, workspace / "uploads")
    attachment_files = _collect_group_files(workspace, session_id, workspace / "attachments", skip_nested_images=True)
    attachment_files = _filter_runtime_attachment_copies(attachment_files, upload_files)
    if upload_files:
        groups.append(
            _input_group(
                "source_paper",
                "Source paper",
                upload_files,
                hint=_input_hint(upload_files, prefix="paper"),
            )
        )
    if attachment_files:
        groups.append(
            _input_group(
                "attachments",
                "Attachments",
                attachment_files,
                hint=_input_hint(attachment_files, prefix="attachment"),
            )
        )
    reference_template = _collect_group_files(workspace, session_id, workspace / "reference_template")
    if reference_template:
        groups.append(
            _input_group(
                "reference_template",
                "Reference template",
                reference_template,
                hint=_input_hint(reference_template, prefix="template"),
            )
        )

    inducted = _current_session_template_artifacts(workspace, session_id)
    if inducted:
        groups.append(
            {
                "id": "inducted_template",
                "label": "Inducted template artifacts",
                "hint": _input_hint(inducted, prefix="artifact"),
                "count": len(inducted),
                "files": inducted,
            }
        )
    return groups


def _filter_runtime_attachment_copies(
    attachment_files: list[dict[str, Any]],
    upload_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not attachment_files or not upload_files:
        return attachment_files
    upload_keys = {
        _input_duplicate_key(file)
        for file in upload_files
        if _input_duplicate_key(file) is not None
    }
    if not upload_keys:
        return attachment_files
    return [
        file
        for file in attachment_files
        if _input_duplicate_key(file) not in upload_keys
    ]


def _input_duplicate_key(file: dict[str, Any]) -> tuple[str, int] | None:
    name = str(file.get("name") or "").strip().lower()
    if not name:
        return None
    try:
        size = int(file.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    return (name, size)


def _input_group(group_id: str, label: str, files: list[dict[str, Any]], *, hint: str) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "hint": hint,
        "count": len(files),
        "files": files,
    }


def _input_hint(files: list[dict[str, Any]], *, prefix: str) -> str:
    names = [str(file.get("name") or "") for file in files if str(file.get("name") or "").strip()]
    if not names:
        return "No files"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} · {names[1]}"
    return f"{names[0]} · {names[1]} · +{len(names) - 2} more"


def _collect_group_files(
    workspace: Path,
    session_id: str,
    base_dir: Path,
    *,
    skip_nested_images: bool = False,
) -> list[dict[str, Any]]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if skip_nested_images and "images" in path.parts and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
            continue
        if any(part in {".locks", ".runtime_secrets", "__pycache__"} for part in path.parts):
            continue
        rel = path.relative_to(workspace).as_posix()
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": rel,
                "kind": _artifact_kind(path),
                "size": stat.st_size,
                "updated_at": stat.st_mtime,
                "download_url": f"/api/files?session_id={session_id}&path={rel}&download=1",
            }
        )
    return items


def _current_session_template_artifacts(workspace: Path, session_id: str) -> list[dict[str, Any]]:
    request = _read_json(workspace / ".input_request.json")
    template_value = str(request.get("template") or "") if isinstance(request, dict) else ""
    template_dir = Path(template_value).expanduser() if template_value else None
    template_id = str(request.get("template_id") or "") if isinstance(request, dict) else ""
    candidates: list[Path] = []
    if template_dir is not None and template_dir.exists() and template_dir.is_dir():
        candidates.append(template_dir)
    if not candidates and template_id:
        maybe = _infer_template_cache_dir(workspace) / template_id
        if maybe.exists() and maybe.is_dir():
            candidates.append(maybe)
    if not candidates:
        return []

    files: list[dict[str, Any]] = []
    for candidate in candidates:
        files.extend(_collect_external_template_files(candidate, session_id))
    return files[:12]


def _collect_external_template_files(template_dir: Path, session_id: str) -> list[dict[str, Any]]:
    allowed = {"analysis.json", "slide_induction.json", "description.txt", "source.pptx"}
    files: list[dict[str, Any]] = []
    for path in sorted(template_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in allowed and path.parent.name != "images":
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "kind": _artifact_kind(path),
                "size": stat.st_size,
                "updated_at": stat.st_mtime,
                "external": True,
                "download_url": "",
            }
        )
    return files


def _infer_template_cache_dir(workspace: Path) -> Path:
    web_root = _infer_web_root(workspace)
    cache_root = web_root.parent if web_root.name == "web" else web_root
    return cache_root / "templates"


def _infer_web_root(workspace: Path) -> Path:
    for parent in workspace.parents:
        if parent.name == "web":
            return parent
    return workspace.parent


def _group_snapshot_versions(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for file in files:
        key = str(file.get("snapshot_label") or file.get("path") or "")
        grouped.setdefault(key, []).append(file)

    ordered = sorted(
        grouped.items(),
        key=lambda item: max(float(file.get("updated_at") or 0.0) for file in item[1]),
    )

    result: list[dict[str, Any]] = []
    for index, (snapshot_label, items) in enumerate(ordered):
        result.append(
            {
                "version_label": _version_label(snapshot_label, index),
                "snapshot_label": snapshot_label,
                "updated_at": max(float(file.get("updated_at") or 0.0) for file in items),
                "pptx": _pick_best_export([file for file in items if file.get("kind") == "pptx"]),
                "pdf": _pick_best_export([file for file in items if file.get("kind") == "pdf"]),
                "files": _pick_version_files(items),
            }
        )
    return list(reversed(result))


def _pick_version_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    pptx = _pick_best_export([file for file in files if file.get("kind") == "pptx"])
    pdf = _pick_best_export([file for file in files if file.get("kind") == "pdf"])
    if pptx:
        chosen.append(pptx)
    if pdf:
        chosen.append(pdf)
    if chosen:
        return chosen
    return sorted(files, key=lambda file: (-float(file.get("updated_at") or 0.0), str(file.get("path") or "")))[:2]


def _is_live_deck_export(file: dict[str, Any]) -> bool:
    path = str(file.get("path") or "")
    kind = str(file.get("kind") or "")
    if kind == "pptx":
        return not path.startswith(("uploads/", "attachments/", "reference_template/", "template_shell/"))
    if kind != "pdf":
        return False
    if path.startswith(("uploads/", "attachments/", "reference_template/", "template_shell/")):
        return False
    name = str(file.get("name") or "").lower()
    return (
        path.startswith("outputs/")
        or name.startswith("manuscript")
        or name.startswith("modification_")
        or name in {"presentation.pdf", "deck.pdf", "slides.pdf"}
    )


def _pick_best_export(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not files:
        return None
    return sorted(files, key=_export_choice_key)[0]


def _export_choice_key(file: dict[str, Any]) -> tuple[int, int, float, str]:
    name = str(file.get("name") or "")
    match = re.search(r"modification_(\d+)", name, flags=re.IGNORECASE)
    revision_rank = int(match.group(1)) if match else (-1 if "manuscript" in name.lower() else 0)
    return (
        0 if not file.get("stale") else 1,
        -revision_rank,
        -float(file.get("updated_at") or 0.0),
        str(file.get("path") or ""),
    )


def _version_label(snapshot_label: str, index: int) -> str:
    if index == 0:
        return "Initial deck"
    return f"Revision {index}"


def _tree_root_dir(workspace: Path, root: str) -> Path:
    mapping = {
        "current": workspace,
        "history": workspace / "downloads",
        "inputs": workspace,
        "slides": next((item for item in _candidate_slide_dirs(workspace) if item.exists()), workspace),
        "debug": workspace,
    }
    if root not in mapping:
        raise ValueError(f"Unsupported tree root: {root}")
    root_dir = mapping[root]
    if root in {"history"} and not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def _resolve_tree_target(root_dir: Path, rel_path: str) -> Path:
    target = (root_dir / rel_path).resolve()
    if not path_within(target, root_dir):
        raise ValueError("Tree path is outside the allowed root")
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(rel_path)
    return target


def _skip_tree_entry(path: Path, workspace: Path, root: str) -> bool:
    if any(part in {".locks", ".runtime_secrets", "__pycache__"} for part in path.parts):
        return True
    if _is_internal_runtime_artifact(path):
        return True
    rel_tuple = path.relative_to(workspace).parts if path.exists() and path != workspace and path_within(path, workspace) else ()
    rel_parts = set(rel_tuple)
    top = rel_tuple[0] if rel_tuple else ""
    if root == "current":
        if path.is_dir() and path.name in {"downloads", "uploads", "attachments", "reference_template", ".history", ".memory", ".rollback", ".interrupted"}:
            return True
        if path.is_file() and "downloads" in rel_parts:
            return True
        return False
    if root == "history":
        return False
    if root == "slides":
        if path.is_dir():
            return False
        return path.suffix.lower() != ".html"
    if root == "inputs":
        if path == workspace:
            return False
        allowed = {"uploads", "attachments", "reference_template", "template_shell"}
        if top in allowed:
            if "images" in path.parts and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
                return True
            return False
        return True
    if root == "debug":
        if path == workspace:
            return False
        allowed = {".history", ".memory", ".rollback", ".interrupted"}
        if top in allowed:
            return False
        if top.startswith(".slide_images"):
            return False
        if path.name == "operation_state.json":
            return False
        return True
    return False


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if "memory" in path.parts or ".memory" in path.parts:
        return "memory"
    if ".history" in path.parts or suffix in {".log", ".jsonl"}:
        return "log"
    if suffix == ".html":
        return "slide"
    if suffix == ".pptx":
        return "pptx"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        return "image"
    return "file"


def _is_stable_download(path: Path) -> bool:
    return "downloads" in path.parts and path.suffix.lower() in {".pptx", ".pdf"}


def _is_internal_runtime_artifact(path: Path) -> bool:
    if OPERATIONS_MARKER in path.parts:
        return True
    if path.name == "operation_state.json":
        return True
    return False


def _snapshot_label(path: Path) -> str:
    if "downloads" not in path.parts:
        return ""
    parts = list(path.parts)
    idx = parts.index("downloads")
    if idx + 1 >= len(parts):
        return ""
    return parts[idx + 1]


def _prioritize_artifacts(files: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(files, key=_artifact_sort_key)
    return ordered[:limit]


def _artifact_sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    path = str(item.get("path") or "")
    kind = str(item.get("kind") or "")
    stable = bool(item.get("stable"))
    updated = float(item.get("updated_at") or 0.0)
    if stable and kind in {"pptx", "pdf"}:
        return (0, -updated, path)
    if kind in {"pptx", "pdf"}:
        return (1, -updated, path)
    if kind == "slide":
        return (2, -updated, path)
    if kind == "memory":
        return (3, -updated, path)
    if kind == "log":
        return (4, -updated, path)
    return (5, -updated, path)


def _stale_current_exports(workspace: Path) -> set[str]:
    newest_slide = 0.0
    for slide_dir in (workspace / "outputs", workspace / "slides", workspace):
        if not slide_dir.exists() or not slide_dir.is_dir():
            continue
        for slide in slide_dir.glob("slide_*.html"):
            if slide.is_file():
                newest_slide = max(newest_slide, slide.stat().st_mtime)
        if newest_slide:
            break
    if not newest_slide:
        return set()

    stale: set[str] = set()
    for pattern in ("*.pptx", "*.pdf", "outputs/*.pptx", "outputs/*.pdf"):
        for path in workspace.glob(pattern):
            if not path.is_file() or "downloads" in path.parts:
                continue
            if ".interrupted" in path.parts:
                continue
            if path.stat().st_mtime + 1.0 < newest_slide:
                stale.add(str(path.resolve()))
    return stale


def _read_first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception:
        return ""


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def _capability(ok: bool, reason: str) -> dict[str, str]:
    return {"status": "available" if ok else "unavailable", "reason": "" if ok else reason}


def render_preview_html(workspace: Path, session_id: str, rel_path: str) -> str:
    served = resolve_workspace_file(workspace, rel_path)
    if served.path.suffix.lower() != ".html":
        raise ValueError("Preview only supports HTML slides")
    html = served.path.read_text(encoding="utf-8", errors="replace")
    return rewrite_workspace_asset_urls(html, workspace, served.path.parent, session_id)


def rewrite_workspace_asset_urls(html: str, workspace: Path, base_dir: Path, session_id: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        for attr in ("src", "href", "poster"):
            value = tag.get(attr)
            if isinstance(value, str):
                tag[attr] = _rewrite_asset_reference(value, workspace, base_dir, session_id)
        srcset = tag.get("srcset")
        if isinstance(srcset, str):
            tag["srcset"] = _rewrite_srcset(srcset, workspace, base_dir, session_id)
        style = tag.get("style")
        if isinstance(style, str):
            tag["style"] = _rewrite_css_urls(style, workspace, base_dir, session_id)
    for style_tag in soup.find_all("style"):
        if style_tag.string:
            style_tag.string.replace_with(_rewrite_css_urls(style_tag.string, workspace, base_dir, session_id))
    return str(soup)


def _rewrite_srcset(value: str, workspace: Path, base_dir: Path, session_id: str) -> str:
    parts: list[str] = []
    for item in value.split(","):
        tokens = item.strip().split()
        if not tokens:
            continue
        tokens[0] = _rewrite_asset_reference(tokens[0], workspace, base_dir, session_id)
        parts.append(" ".join(tokens))
    return ", ".join(parts)


def _rewrite_css_urls(css: str, workspace: Path, base_dir: Path, session_id: str) -> str:
    pattern = re.compile(r"url\((?P<quote>['\"]?)(?P<url>[^)'\"\s]+)(?P=quote)\)")

    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote") or ""
        url = match.group("url")
        rewritten = _rewrite_asset_reference(url, workspace, base_dir, session_id)
        return f"url({quote}{rewritten}{quote})"

    return pattern.sub(repl, css)


def _rewrite_asset_reference(value: str, workspace: Path, base_dir: Path, session_id: str) -> str:
    raw = value.strip()
    if not raw or raw.startswith(("#", "data:", "blob:", "http://", "https:", "mailto:", "javascript:")):
        return value
    if raw.startswith("file://"):
        raw = raw.removeprefix("file://")
    path_part, suffix = _split_url_suffix(raw)
    candidate = Path(path_part)
    if not candidate.is_absolute():
        candidate = (base_dir / path_part).resolve()
    else:
        candidate = candidate.resolve()
    if not path_within(candidate, workspace):
        return value
    if not candidate.exists() or not candidate.is_file():
        return value
    rel = candidate.relative_to(workspace).as_posix()
    return f"/api/files?{urlencode({'session_id': session_id, 'path': rel})}{suffix}"


def _split_url_suffix(value: str) -> tuple[str, str]:
    indices = [idx for idx in (value.find("#"), value.find("?")) if idx >= 0]
    if not indices:
        return value, ""
    idx = min(indices)
    return value[:idx], value[idx:]
