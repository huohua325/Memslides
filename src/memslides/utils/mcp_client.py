from __future__ import annotations

import asyncio
import contextlib
import copy
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import logger as stdio_logger
from mcp.client.stdio import stdio_client

from memslides.utils.constants import MCP_CALL_TIMEOUT, MCP_CONNECT_TIMEOUT
from memslides.utils.log import error, exception, info, warning
from memslides.utils.typings import MCPServer


stdio_logger.setLevel("WARNING")


@dataclass
class _ServerHandle:
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    runner: asyncio.Task[None] | None = None


class MCPClient:
    def __init__(self, envs: dict[str, Any]):
        self.envs = envs
        self.sessions: dict[str, ClientSession] = {}
        self._handles: dict[str, _ServerHandle] = {}

    async def tool_execute(
        self,
        server_id: str,
        tool_name: str,
        tool_params: dict | None,
    ):
        session = self.sessions.get(server_id)
        if session is None:
            raise ValueError(f"Server {server_id} is not connected.")
        return await asyncio.wait_for(
            session.call_tool(tool_name, tool_params),
            timeout=MCP_CALL_TIMEOUT,
        )

    async def connect_server(self, server_id: str, config: MCPServer) -> None:
        if server_id in self.sessions:
            return
        if server_id in self._handles:
            raise RuntimeError(f"Server {server_id} is already connecting.")

        prepared = copy.deepcopy(config)
        prepared.env.update(self.envs)
        prepared._process_escape()

        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        handle = _ServerHandle()
        self._handles[server_id] = handle

        async def run_server() -> None:
            try:
                async with AsyncExitStack() as exit_stack:
                    session = await self._open_session(server_id, prepared, exit_stack)
                    self.sessions[server_id] = session
                    if not ready.done():
                        ready.set_result(None)
                    await handle.stop_event.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    exception("MCP session %s stopped unexpectedly", server_id)
            finally:
                self.sessions.pop(server_id, None)
                self._handles.pop(server_id, None)
                if not ready.done():
                    ready.set_exception(RuntimeError(f"MCP session {server_id} closed before initialization."))
                info("MCP session %s closed", server_id)

        handle.runner = asyncio.create_task(run_server(), name=f"memslides-mcp-{server_id}")
        try:
            await asyncio.wait_for(ready, timeout=max(1.0, MCP_CONNECT_TIMEOUT + 5))
        except Exception:
            await self._close_server(server_id)
            raise

    async def _open_session(
        self,
        server_id: str,
        config: MCPServer,
        exit_stack: AsyncExitStack,
    ) -> ClientSession:
        if config.command:
            return await self._open_stdio_session(server_id, config, exit_stack)
        if config.url:
            return await self._open_sse_session(server_id, config, exit_stack)
        raise ValueError("MCP server config must define either command or url.")

    async def _open_stdio_session(
        self,
        server_id: str,
        config: MCPServer,
        exit_stack: AsyncExitStack,
    ) -> ClientSession:
        try:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
            )
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=MCP_CONNECT_TIMEOUT)
            info("Connected to server %s.", server_id)
            return session
        except TimeoutError:
            error("Timeout connecting to server %s", server_id)
            raise
        except Exception as exc:
            error("Error connecting to server %s: %s", server_id, exc)
            raise

    async def _open_sse_session(
        self,
        server_id: str,
        config: MCPServer,
        exit_stack: AsyncExitStack,
    ) -> ClientSession:
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(
                sse_client(config.url, config.header)
            )
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream, MCP_CONNECT_TIMEOUT)
            )
            await asyncio.wait_for(session.initialize(), timeout=MCP_CONNECT_TIMEOUT)
            info("Connected to server %s.", server_id)
            return session
        except TimeoutError:
            error("Timeout connecting to SSE server %s", server_id)
            raise
        except Exception as exc:
            error("Error connecting to SSE server %s: %s", server_id, exc)
            raise

    async def list_tools(self, server_id: str) -> dict[str, Any]:
        session = self.sessions.get(server_id)
        if session is None:
            warning("Server %s not connected, cannot list tools.", server_id)
            return {}
        response = await session.list_tools()
        return {tool.name: tool for tool in response.tools}

    async def _close_server(self, server_id: str) -> None:
        timeout = float(os.environ.get("MEMSLIDES_MCP_CLOSE_TIMEOUT_SEC", "10") or "10")
        handle = self._handles.get(server_id)
        if handle is None:
            self.sessions.pop(server_id, None)
            return

        handle.stop_event.set()
        runner = handle.runner
        if runner is None:
            self.sessions.pop(server_id, None)
            self._handles.pop(server_id, None)
            return

        try:
            await asyncio.wait_for(runner, timeout=max(1.0, timeout))
        except asyncio.TimeoutError:
            warning("MCP close timed out for %s after %.1fs", server_id, timeout)
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runner
        except Exception as exc:
            if "ClosedResourceError" not in type(exc).__name__:
                warning("MCP close warning for %s: %s", server_id, exc)
        finally:
            self.sessions.pop(server_id, None)
            self._handles.pop(server_id, None)

    async def cleanup(self) -> None:
        server_ids = list(self.sessions.keys() | self._handles.keys())
        for server_id in server_ids:
            try:
                await self._close_server(server_id)
            except TimeoutError:
                warning("Timeout during cleanup")
            except Exception as exc:
                error("Error during cleanup for %s: %s", server_id, exc)
