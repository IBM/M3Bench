#!/usr/bin/env python3
"""
mcp_http_bridge.py — stdio → streamable-HTTP MCP bridge (OPT-IN, Code Engine only).

Why this exists
---------------
Locally, benchmark clients reach the MCP servers via `docker exec -i -e MCP_DOMAIN=<d>
<container> python /app/mcp_dispatch.py` — i.e. MCP over **stdio**, with the domain
injected as an env var into a freshly-spawned server process. Code Engine has no
`docker exec`, so the servers must be reachable over **HTTP**.

This bridge fronts the *unchanged* `mcp_dispatch.py`. It reproduces the exact
per-domain tool scoping:

    HTTP client connects to   ->  https://<app>/mcp/<domain>
    bridge spawns             ->  python /app/mcp_dispatch.py   (env MCP_DOMAIN=<domain>,
                                                                 CAPABILITY_ID inherited)
    bridge proxies            ->  initialize / list_tools / call_tool

A client that connects to /mcp/address gets ONLY the address tools — the subprocess
behind that path was launched with a single MCP_DOMAIN, so list_tools() is scoped
exactly as it is under docker exec. `verify_checksum(capability_id, domain, tools)`
on the client passes unchanged. No MCP server file is modified.

Only runs when the container is started with SERVE_MODE=http (see entrypoint-unified.sh).
The default (docker-exec / local) path is untouched.

Requires: mcp>=1.9 (StreamableHTTPSessionManager), starlette, uvicorn — all in the image.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from contextvars import ContextVar
from typing import Dict, Optional

import uvicorn
from starlette.types import Receive, Scope, Send

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

logging.basicConfig(
    level=os.environ.get("BRIDGE_LOG_LEVEL", "INFO"),
    format="%(asctime)s bridge[%(levelname)s] %(message)s",
)
log = logging.getLogger("mcp_http_bridge")

CAPABILITY_ID = os.environ.get("CAPABILITY_ID", "").strip()
# Overridable for local testing (point at a dummy stdio MCP server).
DISPATCH = os.environ.get("BRIDGE_DISPATCH", "/app/mcp_dispatch.py")

# Domain for the request currently being handled. Set by the ASGI wrapper before
# delegating to the session manager; read by the proxied list_tools/call_tool
# handlers. anyio copies the current context into the per-request task the session
# manager spawns, so the value propagates correctly.
_current_domain: ContextVar[str] = ContextVar("current_domain", default="")


# ---------------------------------------------------------------------------
# Downstream backend: a fresh stdio subprocess (mcp_dispatch.py) pinned to a
# single MCP_DOMAIN, created AND torn down within the calling request's own task
# scope. We deliberately do NOT cache/share sessions across requests — a
# ClientSession's internal task group belongs to the task that created it, and
# reusing it from another request task trips anyio's cancel-scope bookkeeping
# ("attempted to exit a cancel scope that isn't the current task's").
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def spawn_backend(domain: str):
    env = os.environ.copy()
    if domain:
        env["MCP_DOMAIN"] = domain
    params = StdioServerParameters(command=sys.executable, args=[DISPATCH], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            yield sess


# ---------------------------------------------------------------------------
# The single low-level proxy Server. Its handlers forward each request to a
# fresh domain-pinned subprocess (domain from the contextvar set per request).
# ---------------------------------------------------------------------------
proxy = Server(f"vakra-capability-{CAPABILITY_ID or '?'}")


@proxy.list_tools()
async def _list_tools():
    domain = _current_domain.get()
    async with spawn_backend(domain) as sess:
        tools = (await sess.list_tools()).tools
        log.info("list_tools domain=%r -> %d tools", domain, len(tools))
        return tools


@proxy.call_tool()
async def _call_tool(name: str, arguments: dict):
    domain = _current_domain.get()
    async with spawn_backend(domain) as sess:
        result = await sess.call_tool(name, arguments or {})
        # Forward the downstream content blocks verbatim.
        return result.content


session_manager = StreamableHTTPSessionManager(
    app=proxy,
    event_store=None,
    json_response=True,   # plain JSON responses — no SSE plumbing needed for the benchmark
    stateless=True,       # each request self-contained; domain is re-derived from the path
)


# ---------------------------------------------------------------------------
# Minimal ASGI app: /healthz + /mcp/<domain>/...  (domain -> contextvar -> manager)
# ---------------------------------------------------------------------------
async def _send_json(send: Send, status: int, body: bytes) -> None:
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": body})


async def app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "lifespan":
        # Keep the session manager's task group alive for the whole app lifetime,
        # and tear down any spawned subprocesses on shutdown.
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    cm = session_manager.run()
                    await cm.__aenter__()
                    scope.setdefault("state", {})["_mgr_cm"] = cm
                    await send({"type": "lifespan.startup.complete"})
                except Exception as e:  # noqa: BLE001
                    await send({"type": "lifespan.startup.failed", "message": str(e)})
            elif message["type"] == "lifespan.shutdown":
                cm = scope.get("state", {}).get("_mgr_cm")
                if cm is not None:
                    with contextlib.suppress(Exception):
                        await cm.__aexit__(None, None, None)
                await send({"type": "lifespan.shutdown.complete"})
                return

    if scope["type"] != "http":
        return

    path = scope.get("path", "")

    if path in ("/healthz", "/health"):
        import json as _json
        body = _json.dumps({
            "status": "ok",
            "capability_id": CAPABILITY_ID,
        }).encode()
        await _send_json(send, 200, body)
        return

    # Expect /mcp/<domain>[/...]
    parts = [p for p in path.split("/") if p != ""]
    if len(parts) >= 2 and parts[0] == "mcp":
        domain = parts[1]
        token = _current_domain.set(domain)
        try:
            await session_manager.handle_request(scope, receive, send)
        finally:
            _current_domain.reset(token)
        return

    import json as _json
    await _send_json(
        send, 404,
        _json.dumps({
            "error": "not found",
            "hint": "connect to /mcp/<domain> (e.g. /mcp/address); health at /healthz",
        }).encode(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VAKRA MCP stdio->HTTP bridge")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BRIDGE_PORT", "8080")))
    parser.add_argument("--host", type=str, default=os.environ.get("BRIDGE_HOST", "0.0.0.0"))
    args = parser.parse_args()

    if CAPABILITY_ID not in {"1", "2", "3", "4"}:
        log.error("CAPABILITY_ID must be one of 1..4; got %r", CAPABILITY_ID)
        sys.exit(1)

    log.info("MCP HTTP bridge starting: capability=%s on %s:%d  (endpoint: /mcp/<domain>)",
             CAPABILITY_ID, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
