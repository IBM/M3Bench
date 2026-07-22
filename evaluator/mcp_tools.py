from __future__ import annotations

import os
import json
import asyncio
import warnings
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from constant import PRED_OUTPUT_KEY, PRED_OUTPUT_SEQUENCE_KEY
from utils import _ensure_list

ToolSpec = Dict[str, Any]


# -----------------------------
# Prediction dialogue helpers
# -----------------------------

def extract_toolcalls_for_mcp(
    pred_dialogue: Dict[str, Any],
    limit_last_n: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """
    FastAPIMCPToolClient.call_mcp_tools expects:
      tools = [
        [  # one dialogue (ds)
          [ {"name": "...", "arguments": {...}}, ... ],  # one turn
          ...
        ],
        ...
      ]

    Here we return a SINGLE dialogue's tools shape:
      [ [tool, tool...], [tool...], ... ]
    """
    turns = _ensure_list(pred_dialogue.get(PRED_OUTPUT_KEY, []))
    dialogue_tools: List[List[Dict[str, Any]]] = []
    for t in turns:
        seq = t.get(PRED_OUTPUT_SEQUENCE_KEY, {})
        turn_tools: List[Dict[str, Any]] = []
        if "tool_call" not in seq:
            continue
        for raw_tc in seq["tool_call"]:
            if not isinstance(raw_tc, dict):
                continue
            if isinstance(raw_tc, dict) and "name" in raw_tc:
                turn_tools.append(
                    {"name": raw_tc["name"], "arguments": raw_tc.get("arguments", {})})
        if limit_last_n is not None:
            turn_tools = turn_tools[-limit_last_n:]
        dialogue_tools.append(turn_tools)
    return dialogue_tools

def inject_mcp_responses(
    pred_entry: Dict[str, Any],
    mcp_dialogue_responses: List[List[Any]],
    type: str, # "pred" / "gt"
    capability_name:str,
) -> None:
    """
    Inject tool responses into each turn's sequence.tool_response.

    Assumption (per your note):
      - sequence["tool_call"] is a flat list
      - sequence["tool_response"] is a flat list
      - they should be the same length (1:1 aligned) for downstream scoring
    """
    turns = _ensure_list(pred_entry.get(PRED_OUTPUT_KEY, []))
    query_tool_present=False    

    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue

        seq = turn.get(PRED_OUTPUT_SEQUENCE_KEY) or {}
        if not isinstance(seq, dict):
            seq = {}

        tool_calls = _ensure_list(seq.get("tool_call", []))

        # One flat response list per turn
        turn_resps = mcp_dialogue_responses[i] if i < len(mcp_dialogue_responses) else []
        turn_resps = _ensure_list(turn_resps)

        if type == "pred" and len(tool_calls) > len(turn_resps):
            tool_calls = tool_calls[-len(turn_resps):] if turn_resps else []

        if "multiturn" in capability_name:
            for tool in tool_calls:
                if "query_" in tool["name"]:
                    query_tool_present=True
                    break

        # If tool_call exists, enforce 1:1 length alignment
        if tool_calls:
            n = len(tool_calls)
            if (query_tool_present):
                truncated_responses=turn_resps[:n]
                if type == "gt":
                    for idx, tool in enumerate(tool_calls):
                        if "query_" not in tool["name"]:
                            seq["tool_response"][idx] = truncated_responses[idx]
                        else:
                            seq["tool_response"][idx] = [item["text"] for item in seq["tool_response"][idx]] # Only text in chunks is retained
                elif type=="pred":
                    for idx, tool in enumerate(tool_calls):
                        if "query_" in tool["name"]:
                            try:
                                truncated_responses[idx] = [item["text"] for item in json.loads(truncated_responses[idx])["results"]] # Only text in chunks is retained
                            except Exception as e:
                                truncated_responses[idx] = [f"Error processing response: {str(e)}"]
                    seq["tool_response"] = truncated_responses
            else:
                if len(turn_resps) >= n:
                    seq["tool_response"] = turn_resps[:n]
                else:
                    seq["tool_response"] = turn_resps + ([None] * (n - len(turn_resps)))
        else:
            # No tool calls: just store whatever we have (or empty list)
            seq["tool_response"] = turn_resps

        seq["tool_call"] = tool_calls
        turn[PRED_OUTPUT_SEQUENCE_KEY] = seq

        # If you require strict alignment only when tool_calls exist:
        if tool_calls:
            assert len(seq["tool_call"]) == len(seq["tool_response"])

# -----------------------------
# Schema coercion helpers
# -----------------------------

def _coerce_value_to_schema(value: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """
    Coerce a single value to match a (subset of) JSON schema.
    Supports: integer, number, boolean, string, array, object.
    """
    if schema is None:
        return value

    # Handle schema unions like {"type": ["integer", "null"]}
    sch_type = schema.get("type")
    if isinstance(sch_type, list):
        # Prefer first non-null type
        sch_type = next((t for t in sch_type if t != "null"), sch_type[0] if sch_type else None)

    if sch_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            # allow "1", "001", "-3"
            if s and (s.isdigit() or (s[0] == "-" and s[1:].isdigit())):
                return int(s)
        return value  # can't safely coerce

    if sch_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            try:
                return float(s)
            except ValueError:
                return value
        return value

    if sch_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "t", "1", "yes", "y"}:
                return True
            if s in {"false", "f", "0", "no", "n"}:
                return False
        return value

    if sch_type == "string":
        # if server wants string, stringify scalars
        if isinstance(value, str):
            return value
        return str(value)

    if sch_type == "array":
        items_schema = schema.get("items")
        if isinstance(value, list):
            return [_coerce_value_to_schema(v, items_schema) for v in value]
        # If a single scalar provided where array expected, wrap it
        return [_coerce_value_to_schema(value, items_schema)]

    if sch_type == "object":
        if not isinstance(value, dict):
            return value
        return _coerce_args_to_schema(value, schema)

    return value


def _coerce_args_to_schema(args: Dict[str, Any], input_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Coerce argument dict based on a tool's JSON inputSchema.
    """
    if not input_schema or not isinstance(input_schema, dict):
        return args

    properties = input_schema.get("properties", {}) if isinstance(input_schema.get("properties", {}), dict) else {}

    coerced: Dict[str, Any] = {}
    for k, v in args.items():
        prop_schema = properties.get(k)
        coerced[k] = _coerce_value_to_schema(v, prop_schema)

    return coerced


# -----------------------------
# Batch tool execution
# -----------------------------

async def execute_tools_batch(
    session: ClientSession,
    dialogue_tools_batch: List[List[List[Dict[str, Any]]]],
    schema_map: Optional[Dict[str, Any]],
    is_gt: bool = False,
) -> List[List[List[Any]]]:
    """
    Execute a batch of tool calls using a raw MCP session, with schema-based
    argument coercion to recover types lost during JSON serialization.

    Args:
        session: An initialized MCP ClientSession
        dialogue_tools_batch: [
            [ [tool...], [tool...], ... ],  # dialogue 0
            [ [tool...], [tool...], ... ],  # dialogue 1
        ]
        schema_map: {tool_name: inputSchema} mapping used to coerce argument types

    Returns:
        List[List[List[Any]]]: Response strings in the same structure as input
    """
    all_dialogue_responses: List[List[List[Any]]] = []

    for dialogue_tools in dialogue_tools_batch:
        dialogue_responses: List[List[Any]] = []

        for turn_tools in dialogue_tools:
            turn_responses: List[Any] = []

            for tool in turn_tools:
                tool_name = tool["name"]
                raw_args = tool.get("arguments", {}) or {}
                if is_gt:
                    if (("input_value" in raw_args.keys()) or "data_label" in raw_args.keys()) and len(turn_responses)>0:
                        if "handle" in turn_responses[-1]:
                            previous_response = json.loads(turn_responses[-1])
                            if "input_value" in raw_args.keys():
                                raw_args["input_value"] = previous_response["results"]
                            if "data_label" in raw_args.keys():
                                raw_args["data_label"] = previous_response["handle"]

                # Parse JSON string args if needed (JSON round-trip safety)
                if isinstance(raw_args, str):
                    raw_args = json.loads(raw_args)

                # Coerce args to match MCP tool input schema
                input_schema = (schema_map or {}).get(tool_name)
                tool_args = _coerce_args_to_schema(raw_args, input_schema)

                try:
                    result = await session.call_tool(tool_name, tool_args)
                    # Extract text from MCP response
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            turn_responses.append(content_item.text)
                        else:
                            turn_responses.append(str(content_item))
                    else:
                        turn_responses.append("")
                except Exception as e:
                    turn_responses.append(f"Error: {str(e)}")

            dialogue_responses.append(turn_responses)

        all_dialogue_responses.append(dialogue_responses)

    return all_dialogue_responses


@dataclass(frozen=True)
class MCPConnectionConfig:
    domain: str
    uuid: Optional[str] = None
    run_type: str = "local"  # "local" | "docker"
    container_runtime: Optional[str] = None
    container_name: Optional[str] = None


class MCPToolClientBase(ABC):
    """
    DEPRECATED: Use benchmark.mcp_client.create_client_and_connect() instead.

    Base class for connecting to an MCP server and calling tools.
    - Subclasses define how server params are built (FastAPI/Python/BPO).
    - Subclasses define how tool invocation happens for (FastAPI/Python/BPO).
    - Shared connect+initialize lifecycle is implemented once here.
    """

    def __init__(self, config: MCPConnectionConfig):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. "
            "Use benchmark.mcp_client.create_client_and_connect() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.run_type not in ("local", "docker"):
            raise ValueError("run_type must be 'local' or 'docker'")

        if self.config.run_type == "docker":
            if not self.config.container_runtime or not self.config.container_name:
                raise ValueError("container_runtime and container_name are required for docker mode")

    @abstractmethod
    def build_server_params(self) -> StdioServerParameters:
        """Subclasses must return StdioServerParameters appropriate for their MCP server."""
        raise NotImplementedError

    @asynccontextmanager
    async def connect_to_mcp_server(self) -> AsyncIterator[ClientSession]:
        """
        Connects to the MCP server (stdio) and yields an initialized ClientSession.
        """
        server_params = self.build_server_params()

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def call_mcp_tools(self, session: ClientSession, tools: Sequence[ToolSpec], schema_map: Optional[Dict[str, Any]]) -> List[Any]:
        """
        Calls MCP tools using an already-connected and initialized ClientSession.
        """
        raise NotImplementedError

    async def run_tools(self, tools: Optional[Sequence[ToolSpec]] = None, schema_map: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Connect to MCP server and call the provided tools.
        """
        async with self.connect_to_mcp_server() as session:
            return await self.call_mcp_tools(session, tools, schema_map)
        
    async def list_loaded_tools(self) -> List[Dict[str, Any]]:
        """
        Connects to the MCP server and returns the list of tools that the server reports as loaded.

        Returns a normalized list of dicts, e.g.:
          [{"name": "...", "description": "...", "inputSchema": {...}}, ...]
        """
        async with self.connect_to_mcp_server() as session:
            # Prefer native MCP method if available
            if hasattr(session, "list_tools"):
                resp = await session.list_tools()  # MCP SDKs often return an object with .tools
                tool_list = getattr(resp, "tools", resp)

        # 1) Build a schema map: tool_name -> inputSchema
        schema_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(tool_list, list):
            for t in tool_list:
                # t may be dict or object
                name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                input_schema = t.get("inputSchema") if isinstance(t, dict) else getattr(t, "inputSchema", None)
                if isinstance(name, str) and isinstance(input_schema, dict):
                    schema_map[name] = input_schema
            return tool_list, schema_map

    def _coerce_args_to_schema(self, args: Dict[str, Any], input_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Delegate to module-level _coerce_args_to_schema."""
        return _coerce_args_to_schema(args, input_schema)

    def _coerce_value_to_schema(self, value: Any, schema: Optional[Dict[str, Any]]) -> Any:
        """Delegate to module-level _coerce_value_to_schema."""
        return _coerce_value_to_schema(value, schema)


class FastAPIMCPToolClient(MCPToolClientBase):
    """
    MCP client for FastAPI-backed tools.
    Preserves your original behavior:
      - local: runs environment/m3/rest/mcp_server.py with FASTAPI_BASE_URL + MCP_DOMAINS env
      - docker: docker/podman exec -e MCP_DOMAINS=... <container> python mcp_server.py
    """

    def __init__(
        self,
        config: MCPConnectionConfig,
        local_server_script: str = "environment/m3/rest/mcp_server.py",
        docker_server_script: str = "environment/m3/rest/mcp_server.py",
        fastapi_url: str = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000"),
    ):
        super().__init__(config)
        self.local_server_script = local_server_script
        self.docker_server_script = docker_server_script
        self.fastapi_url = fastapi_url

    def build_server_params(self) -> StdioServerParameters:
        if self.config.run_type == "docker":
            exec_args = [
                "exec",
                "-i",
                "-e",
                f"MCP_DOMAINS={self.config.domain}",
                self.config.container_name,  # validated in base
                "python",
                self.docker_server_script,
            ]
            return StdioServerParameters(
                command=self.config.container_runtime,  # validated in base
                args=exec_args,
                env=None,
            )
        elif self.config.run_type =="local":
            return StdioServerParameters(
                command="python",
                args=[self.local_server_script],
                env={
                    "FASTAPI_BASE_URL": self.fastapi_url,
                    "MCP_DOMAIN": self.config.domain,
                },
            )
    
    async def call_mcp_tools(self, session: ClientSession, tools: Sequence[ToolSpec], schema_map: Optional[Dict[str, Any]]) -> List[Any]:
        """
        Calls MCP tools using an already-connected and initialized ClientSession.
        """
        results: List[Any] = []

        for ds in tools:  # loop over data samples
            dialogue_response: List[Any] = []
            for turn in ds:  # loop over turns
                turn_response: List[Any] = []
                for tool in turn:  # loop over tool calls
                    name = tool["name"]
                    raw_args = tool.get("arguments", {}) or {}

                    # If args sometimes arrive as JSON string, parse it.
                    if isinstance(raw_args, str):
                        raw_args = json.loads(raw_args)

                    if not isinstance(raw_args, dict):
                        raise TypeError(f"Tool arguments must be a dict (or JSON string), got {type(raw_args)} for {name}")

                    # 2) Coerce args according to schema if available
                    input_schema = schema_map.get(name)
                    args = self._coerce_args_to_schema(raw_args, input_schema)

                    # 3) Call tool
                    result = await session.call_tool(name, args)

                    # Keep your previous extraction style; adjust if needed
                    turn_response.append(result.content[0].text if getattr(result, "content", None) else None)

                dialogue_response.append(turn_response)
            results.append(dialogue_response)

        return results

class PythonMCPToolClient(MCPToolClientBase):
    """
    MCP client for Python tools.
    Update local_server_script/env according to your python-tools MCP server.
    """

    def __init__(
        self,
        config: MCPConnectionConfig,
        local_server_script: str = "environment/m3/python/mcp_server.py",
        docker_server_script: str = "mcp_server.py",
    ):
        super().__init__(config)
        self.local_server_script = local_server_script
        self.docker_server_script = docker_server_script

    def build_server_params(self) -> StdioServerParameters:
        if self.config.run_type == "docker":
            exec_args = [
                "exec",
                "-i",
                "-e",
                f"MCP_DOMAINS={self.config.domain}",
                self.config.container_name,
                "python",
                self.docker_server_script,
            ]
            return StdioServerParameters(
                command=self.config.container_runtime,
                args=exec_args,
                env=None,
            )

        return StdioServerParameters(
            command="python",
            args=[self.local_server_script],
            env={
                "MCP_DOMAINS": self.config.domain,
                # add python-tools specific env vars here if needed
            },
        )


class BPOMCPToolClient(MCPToolClientBase):
    """
    MCP client for BPO tools.
    Update local_server_script/env according to your BPO MCP server.
    """

    def __init__(
        self,
        config: MCPConnectionConfig,
        local_server_script: str = "environment/m3/bpo/mcp_server.py",
        docker_server_script: str = "mcp_server.py",
    ):
        super().__init__(config)
        self.local_server_script = local_server_script
        self.docker_server_script = docker_server_script

    def build_server_params(self) -> StdioServerParameters:
        if self.config.run_type == "docker":
            exec_args = [
                "exec",
                "-i",
                "-e",
                f"MCP_DOMAINS={self.config.domain}",
                self.config.container_name,
                "python",
                self.docker_server_script,
            ]
            return StdioServerParameters(
                command=self.config.container_runtime,
                args=exec_args,
                env=None,
            )

        return StdioServerParameters(
            command="python",
            args=[self.local_server_script],
            env={
                "MCP_DOMAINS": self.config.domain,
                # add BPO-specific env vars here if needed
            },
        )

def test(tools: List[ToolSpec]) -> List[Any]:
    async def _run():
        client = FastAPIMCPToolClient(
            config=MCPConnectionConfig(domain="california_schools", run_type="local")
        )
        return await client.run_tools(tools)

    return asyncio.run(_run())

if __name__ == "__main__":
    # This example gives a super huge output.
    # tools = [[[
    #     {
    #         "name": "get_school_details_by_open_close_dates",
    #         "arguments": {"open_year": "1991", "close_year": "2000"},
    #     }
    # ]]]

    tools = [
        {
            "name": "get_school_details_by_open_close_dates",
            "arguments": {"open_year": "1991"},
        }
    ]

    results = test(tools=tools)
