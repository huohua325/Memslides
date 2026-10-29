from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from memslides.utils.log import error, info, set_logger

_REQUIRED_TOOLS = frozenset(
    {
        "write_html_file",
        "write_new_slide_file",
        "read_slide_snapshot",
        "plan_slide_patch",
        "apply_slide_patch",
    }
)
_REQUIRED_PARAMS = {
    "write_html_file": {"file_path", "content", "force_regenerate", "expected_hash"},
    "write_new_slide_file": {"file_path", "content"},
    "read_slide_snapshot": {"slide_path"},
    "plan_slide_patch": {"slide_path", "edit_intent", "target_scope", "requested_properties"},
    "apply_slide_patch": {"snapshot_id", "patch_ops", "expected_hash"},
}
_SCHEMA_REPORT_FILE = "memslides_deck_server_schema.json"


def _load_deck_module():
    return importlib.import_module("memslides.tools.deck_runtime")


async def _registered_deck_tools_async(
    deck_module: Any | None = None,
) -> dict[str, Any]:
    module = deck_module or _load_deck_module()
    list_tools = getattr(module.mcp, "list_tools", None)
    if callable(list_tools):
        try:
            tools = await list_tools(run_middleware=False)
        except TypeError:
            tools = await list_tools()
        return {
            str(getattr(tool, "name", "")): tool
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
        }

    # Compatibility with older FastMCP versions.
    tool_manager = getattr(module.mcp, "_tool_manager", None)
    legacy_tools = getattr(tool_manager, "_tools", None)
    if isinstance(legacy_tools, dict):
        return legacy_tools
    return {}


def _registered_deck_tools(
    deck_module: Any | None = None,
) -> dict[str, Any]:
    return asyncio.run(_registered_deck_tools_async(deck_module))


def _tool_parameter_names(tool: Any) -> set[str]:
    parameters = getattr(tool, "parameters", None)
    if not isinstance(parameters, dict):
        return set()
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        return set()
    return {str(name) for name in properties.keys()}


def collect_deck_schema_report(
    registered_tools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deck_module = _load_deck_module()
    tools = registered_tools if registered_tools is not None else _registered_deck_tools(
        deck_module
    )

    missing_tools = sorted(_REQUIRED_TOOLS - set(tools.keys()))
    schema_issues: list[str] = []
    parameter_report: dict[str, dict[str, Any]] = {}
    for tool_name, required_params in _REQUIRED_PARAMS.items():
        available_params = sorted(_tool_parameter_names(tools.get(tool_name)))
        missing_params = sorted(required_params - set(available_params))
        parameter_report[tool_name] = {
            "available_params": available_params,
            "missing_params": missing_params,
        }
        if missing_params:
            schema_issues.append(
                f"{tool_name} missing params: {', '.join(missing_params)}"
            )

    return {
        "ok": not missing_tools and not schema_issues,
        "python_executable": sys.executable,
        "deck_module_file": str(Path(deck_module.__file__).resolve()),
        "registered_tool_count": len(tools),
        "registered_tools": sorted(tools.keys()),
        "required_tools": sorted(_REQUIRED_TOOLS),
        "missing_tools": missing_tools,
        "schema_issues": schema_issues,
        "parameter_report": parameter_report,
    }


def validate_deck_schema_report(report: dict[str, Any]) -> None:
    if report.get("ok"):
        return

    details: list[str] = []
    missing_tools = report.get("missing_tools") or []
    if missing_tools:
        details.append("missing tools: " + ", ".join(str(x) for x in missing_tools))
    for issue in report.get("schema_issues") or []:
        details.append(str(issue))

    raise RuntimeError(
        "Deck server schema self-check failed: "
        + "; ".join(details)
        + ". Refusing to expose a stale deck MCP schema."
    )


def _write_schema_report(workspace: Path, report: dict[str, Any]) -> Path:
    history_dir = workspace / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)
    report_path = history_dir / _SCHEMA_REPORT_FILE
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def run_deck_server(workspace: Path | str) -> None:
    work_dir = Path(workspace)
    if not work_dir.exists():
        raise FileNotFoundError(f"Workspace {work_dir} does not exist.")

    os.environ["MEMSLIDES_WORKSPACE"] = str(work_dir.resolve())
    os.chdir(work_dir)
    set_logger(
        f"memslides-deck-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_deck_tools.log",
    )

    schema_report = collect_deck_schema_report()
    report_path = _write_schema_report(work_dir, schema_report)
    validate_deck_schema_report(schema_report)

    info(
        "MemSlides deck server schema self-check passed using %s; report saved to %s",
        schema_report["deck_module_file"],
        report_path,
    )

    deck_module = _load_deck_module()
    deck_module.mcp.run(show_banner=False)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m memslides.tools.deck_tools <workspace>")

    try:
        run_deck_server(args[0])
    except Exception as exc:
        try:
            error("Deck server startup failed: %s", exc)
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
