#!/usr/bin/env python3
"""
MCP Tools Explorer — FastAPI backend.

Usage:
    cd <project_root>
    pip install fastapi uvicorn
    uvicorn tools_explorer.app:app --reload --port 7860

Then open http://localhost:7860
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

CAPABILITY_DESCRIPTIONS = {
    1: "Slot-filling agent — SQL + selection tools via RouterMCPServer",
    2: "Dashboard APIs — ReAct agent with M3 REST / FastAPI tools",
    3: "Multihop Reasoning — M3 REST or BPO tools (routed by domain)",
    4: "Multiturn — M3 REST tools + semantic retriever combined",
}

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="MCP Tools Explorer", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_configs():
    from benchmark.mcp_client import load_mcp_config
    # MCP_CONFIG lets the explorer point at Code Engine instead of local containers,
    # e.g. MCP_CONFIG=benchmark/mcp_connection_config.ce.yaml (after sourcing
    # deploy/ce/.ce_urls.env). Defaults to the local docker-exec config.
    override = os.environ.get("MCP_CONFIG")
    config_path = Path(override) if override else PROJECT_ROOT / "benchmark" / "mcp_connection_config.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return load_mcp_config(str(config_path))


def _schema_to_dict(schema) -> dict:
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    if hasattr(schema, "dict"):
        return schema.dict()
    return {}



# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/capabilities")
def get_capabilities():
    configs = _load_configs()
    return [
        {
            "id": cap_id,
            "container_name": cfg.container_name or "",
            "description": CAPABILITY_DESCRIPTIONS.get(cap_id, f"Capability {cap_id}"),
        }
        for cap_id, cfg in sorted(configs.items())
    ]


@app.get("/api/domains")
def get_domains(capability_id: int = 0):
    """Return domains for a capability by scanning data/test/capability_N_*/input/."""
    test_dir = PROJECT_ROOT / "data" / "test"
    if not test_dir.exists():
        return ["address", "airline", "bpo", "hockey", "olympics"]
    dirs = sorted(test_dir.glob(f"capability_{capability_id}_*")) if capability_id else sorted(test_dir.iterdir())
    domains: set[str] = set()
    for d in dirs:
        for p in (d / "input").glob("*.json"):
            domains.add(p.stem)
    return sorted(domains)


@app.get("/api/tools")
def list_tools(capability_id: int, domain: str):
    """Sync route — FastAPI runs this in a thread pool, so asyncio.run() gets
    its own clean event loop (same pattern as benchmark_runner / CLI utils)."""
    from benchmark.mcp_client import create_client_and_connect
    import asyncio

    configs = _load_configs()
    cfg = configs.get(capability_id)
    if not cfg:
        raise HTTPException(404, f"Capability {capability_id} not found in config")

    async def _run():
        tools = None
        try:
            async with create_client_and_connect(cfg, domain=domain) as session:
                result = await session.list_tools()
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": _schema_to_dict(tool.inputSchema),
                    }
                    for tool in result.tools
                ]
        except Exception as e:
            if tools is not None:
                # Data already received — cleanup threw (TaskGroup teardown on
                # servers that exit after responding, e.g. cap 2/3 M3 REST).
                logger.debug("MCP session cleanup error (data already received): %s", e)
            else:
                raise
        return tools

    try:
        return asyncio.run(_run())
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        logger.error("list_tools cap=%s domain=%s failed:\n%s", capability_id, domain, traceback.format_exc())
        raise HTTPException(503, detail)


class InvokeRequest(BaseModel):
    capability_id: int
    domain: str
    tool: str
    args: Dict[str, Any] = {}


@app.post("/api/invoke")
def invoke_tool(body: InvokeRequest):
    """Sync route — same pattern as list_tools; asyncio.run() owns its event loop."""
    from benchmark.mcp_client import create_client_and_connect
    import asyncio

    configs = _load_configs()
    cfg = configs.get(body.capability_id)
    if not cfg:
        raise HTTPException(404, f"Capability {body.capability_id} not found")

    async def _run():
        response = None
        try:
            async with create_client_and_connect(cfg, domain=body.domain) as session:
                result = await session.call_tool(body.tool, body.args)
                content = []
                for item in result.content:
                    if hasattr(item, "text"):
                        try:
                            content.append({"type": "json", "data": json.loads(item.text)})
                        except (json.JSONDecodeError, TypeError):
                            content.append({"type": "text", "data": item.text})
                    else:
                        content.append({"type": type(item).__name__, "data": str(item)})
                response = {"content": content, "isError": bool(result.isError)}
        except Exception as e:
            if response is not None:
                logger.debug("MCP session cleanup error (data already received): %s", e)
            else:
                raise
        return response

    try:
        return asyncio.run(_run())
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        logger.error("invoke_tool cap=%s domain=%s tool=%s failed:\n%s",
                     body.capability_id, body.domain, body.tool, traceback.format_exc())
        raise HTTPException(503, detail)
