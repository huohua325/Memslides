from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from memslides.tools.research import mcp as _research_mcp
from memslides.tools.search import mcp as _web_mcp
from memslides.utils.log import set_logger


mcp = FastMCP(name="MemSlidesSearchTools")


async def _registered_tools(source: FastMCP) -> list:
    list_tools = getattr(source, "list_tools", None)
    if callable(list_tools):
        try:
            return list(await list_tools(run_middleware=False))
        except TypeError:
            return list(await list_tools())

    # Compatibility with FastMCP 2.x, where tools are kept behind the manager
    # and the public list_tools API is not available yet.
    tool_manager = getattr(source, "_tool_manager", None)
    legacy_tools = getattr(tool_manager, "_tools", None)
    if isinstance(legacy_tools, dict):
        return list(legacy_tools.values())

    get_tools = getattr(tool_manager, "get_tools", None)
    if callable(get_tools):
        tools = get_tools()
        if isinstance(tools, dict):
            return list(tools.values())
        return list(tools or [])

    return []


async def _copy_registered_tools_async(source: FastMCP) -> None:
    tools = await _registered_tools(source)
    registered = {
        tool.name
        for tool in await _registered_tools(mcp)
    }
    for tool in tools:
        if tool.name not in registered:
            mcp.add_tool(tool)


def _copy_registered_tools(source: FastMCP) -> None:
    asyncio.run(_copy_registered_tools_async(source))


_copy_registered_tools(_web_mcp)
_copy_registered_tools(_research_mcp)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m memslides.tools.search_tools <workspace>")

    work_dir = Path(args[0])
    if not work_dir.exists():
        raise FileNotFoundError(f"Workspace {work_dir} does not exist.")
    os.chdir(work_dir)
    set_logger(
        f"memslides-search-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_search_tools.log",
    )
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
