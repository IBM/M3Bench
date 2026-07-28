import contextlib
from dataclasses import dataclass
import logging
import os
import subprocess
import sys
import yaml

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import AsyncGenerator, Dict, List, Optional

from benchmark.utils import _assert_container_running, detect_container_runtime

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CONTAINER_NAME = "fastapi-mcp-server"


@dataclass
class MCPConnectionConfig:
    """Connection settings for a single task's MCP server."""
    mode: str = "stdio"
    container_name: str = DEFAULT_CONTAINER_NAME
    container_runtime: Optional[str] = None  # None = auto-detect
    container_command: Optional[List[str]] = None
    container_env: Optional[Dict[str, str]] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    server_url: Optional[str] = None
    # Optional second MCP server launched in the same container (e.g. Task 3:
    # primary=BPO server, secondary=M3 REST server).  When set, benchmark_runner
    # opens a second docker-exec session and merges the tool lists.
    secondary_container_command: Optional[List[str]] = None


def load_mcp_config(config_path: str) -> Dict[int, MCPConnectionConfig]:
    """Load MCP connection config from YAML file.

    Returns a dict mapping capability_id (int) to MCPConnectionConfig.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks", {})
    result = {}
    for k, v in tasks.items():
        cmd = v.get("command", None)
        # "python" in config means the current Python executable (virtualenv-safe)
        if cmd == "python":
            cmd = sys.executable
        result[int(k)] = MCPConnectionConfig(
            mode=v.get("mode", "stdio"),
            container_name=v.get("container_name", DEFAULT_CONTAINER_NAME),
            container_runtime=v.get("container_runtime", None),
            container_command=v.get("container_command", None),
            container_env=v.get("container_env", None),
            command=cmd,
            args=v.get("args", None),
            # Expand ${ENV} in server_url so a committed CE config can reference the
            # per-app URLs written by deploy/ce/3_deploy_apps.sh (see .ce_urls.env).
            server_url=(os.path.expandvars(v["server_url"]) if v.get("server_url") else None),
            secondary_container_command=v.get("secondary_container_command", None),
        )
    return result


@contextlib.asynccontextmanager
async def create_client_and_connect(
    cfg: MCPConnectionConfig, domain: str = ""
) -> AsyncGenerator[ClientSession, None]:
    """Async context manager yielding an initialized ClientSession.

    Connection mode is determined by cfg.mode:
    - "stdio" without cfg.command: container exec (default)
    - "stdio" with cfg.command: local subprocess
    - "websocket": WebSocket connection

    Yields:
        ClientSession — an initialized MCP client session ready to use
    """
    if cfg.mode == "stdio":
        if cfg.command:
            # Local subprocess mode
            logger.info("Starting MCP server: %s %s", cfg.command, " ".join(cfg.args or []))
            env = os.environ.copy()
            if domain:
                env["MCP_DOMAIN"] = domain
            server_params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args or [],
                env=env,
            )
        else:
            # Container exec mode (FastAPIMCPServer)
            runtime = cfg.container_runtime
            if not runtime:
                try:
                    runtime = detect_container_runtime()
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Cannot start MCP server in container mode: {e}"
                    ) from e
                logger.info("Auto-detected container runtime: %s", runtime)
            if not cfg.container_name:
                raise ValueError(
                    "stdio mode requires either command or container_name"
                )
            _assert_container_running(runtime, cfg.container_name)
            exec_env = {"MCP_DOMAIN": domain} if domain else {}
            if cfg.container_env:
                exec_env.update(cfg.container_env)
            env_args = []
            for k, v in exec_env.items():
                env_args += ["-e", f"{k}={v}"]
            cmd = cfg.container_command or ["python", "mcp_server.py"]
            full_args = ["exec", "-i"] + env_args + [cfg.container_name] + cmd
            logger.info("Starting MCP server: %s %s", runtime, " ".join(full_args))
            server_params = StdioServerParameters(
                command=runtime,
                args=["exec", "-i"] + env_args + [cfg.container_name] + cmd,
                env=None,
            )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except FileNotFoundError as e:
            cmd = server_params.command
            raise RuntimeError(
                f"MCP server command not found: {cmd!r}. "
                "Ensure the command is installed and available in PATH."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP server via stdio: {e}"
            ) from e

    elif cfg.mode in ("http", "streamable-http"):
        # Remote MCP over streamable HTTP (e.g. Code Engine). Per-domain scoping is
        # preserved by the server side: the domain is carried in the URL path
        # (/mcp/<domain>) and an X-MCP-Domain header, and the bridge spawns a
        # subprocess pinned to that MCP_DOMAIN — so list_tools() returns only that
        # domain's tools, exactly as in stdio/docker-exec mode.
        if not cfg.server_url:
            raise ValueError("http mode requires server_url")
        if "${" in cfg.server_url:
            raise RuntimeError(
                f"server_url still contains an unexpanded variable: {cfg.server_url!r}. "
                "Run the CE deploy (deploy/ce/3_deploy_apps.sh) and "
                "`source deploy/ce/.ce_urls.env` before benchmarking against Code Engine."
            )
        if "{domain}" in cfg.server_url:
            url = cfg.server_url.format(domain=domain)
        else:
            url = f"{cfg.server_url.rstrip('/')}/{domain}" if domain else cfg.server_url
        headers = {"X-MCP-Domain": domain} if domain else {}
        logger.info("Connecting to MCP server via streamable HTTP: %s", url)
        from mcp.client.streamable_http import streamablehttp_client
        try:
            async with streamablehttp_client(url, headers=headers) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP server at {url!r}: {e}"
            ) from e

    elif cfg.mode == "websocket":
        if not cfg.server_url:
            raise ValueError("websocket mode requires server_url")
        logger.info("Connecting to MCP server via websocket: %s", cfg.server_url)
        from mcp.client.websocket import websocket_client
        try:
            async with websocket_client(cfg.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP server at {cfg.server_url!r}: {e}"
            ) from e

    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode!r}. Must be 'stdio', 'http', or 'websocket'"
        )
    
def stop_mcp_server(cfg: MCPConnectionConfig):
    """Stop a running MCP server process.

    For container stdio mode, force-kills the mcp_server.py process inside the
    container.  For subprocess stdio and websocket modes the transport's own
    context manager handles teardown, so this is a no-op.
    """
    if cfg.mode in ("websocket", "http", "streamable-http"):
        # Network transports are closed by the client context manager; the CE
        # bridge tears down its per-domain subprocess on session end. Nothing to do.
        return

    if cfg.mode == "stdio" and not cfg.command and cfg.container_name:
        # Container exec mode: pkill the server process inside the container.
        runtime = cfg.container_runtime or detect_container_runtime()
        try:
            kill_cmd = [
                runtime, "exec", cfg.container_name,
                "pkill", "-f", "python mcp_server.py"
            ]
            subprocess.run(kill_cmd, capture_output=True, timeout=5)
            logger.info("Server stopped.")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout while stopping server")
        except Exception:
            pass
    # Local subprocess stdio: the stdio_client context manager terminates the
    # child process on exit, so no explicit kill is needed here.

