#!/usr/bin/env python3
"""
Benchmark Runner
================

Runs LLM agents against MCP tool servers and records trajectories + answers.

Tasks:
  Task 2  -> fastapi-mcp-server   (M3 SQL tools)
  Capability 4  -> retriever-mcp-server (ChromaDB retriever)

Setup:
  pip install langchain-openai langchain mcp langchain-anthropic langgraph langchain-ollama sentence-transformers

MCP connection settings are read from a YAML config file
(default: benchmark/mcp_connection_config.yaml). Override with --mcp-config.

Usage:
  # Single task, single domain
  python benchmark_runner.py --capability_id 2 --domain hockey

  # Single task, multiple domains
  python benchmark_runner.py --capability_id 2 --domain hockey --domain address

  # Multiple tasks (sequential, default)
  python benchmark_runner.py --capability_id 2 4

  # Multiple tasks in parallel
  python benchmark_runner.py --capability_id 2 4 --parallel

  # Limit samples per domain
  python benchmark_runner.py --capability_id 2 --max-samples-per-domain 5

  # Choose provider/model
  python benchmark_runner.py --capability_id 2 --provider anthropic --model claude-sonnet-4-5-20250929
  python benchmark_runner.py --capability_id 2 --provider ollama --model llama3.1:8b

  # Enable tool shortlisting (top-k tools per query)
  python benchmark_runner.py --capability_id 2 --top-k-tools 10

  # Custom output directory
  python benchmark_runner.py --capability_id 2 --output my_results/

  # Use a custom MCP connection config
  python benchmark_runner.py --capability_id 2 --mcp-config my_mcp_config.yaml

  # List available tools for a domain (does not run the benchmark)
  python benchmark_runner.py --capability_id 2 --domain hockey --list-tools

Output:
  Results saved to: output/capability_{id}_{timestamp}/<domain>.json
  e.g. output/capability_2_feb_18_11_21am/hockey.json
"""
import os
import asyncio
from contextlib import AsyncExitStack
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm


def _setup_phoenix(endpoint: str, project_name: str = "enterprise-benchmark") -> bool:
    """Configure Phoenix/Arize OTEL tracing for LangChain.

    Returns True if tracing was successfully enabled, False otherwise
    (missing packages or Phoenix unreachable — benchmark still runs).
    """
    try:
        from phoenix.otel import register  # type: ignore[import]
        from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore[import]
    except ImportError:
        logging.getLogger(__name__).warning(
            "Phoenix packages not installed. "
            "Run: pip install arize-phoenix-otel openinference-instrumentation-langchain"
            "\nContinuing without tracing."
        )
        return False

    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logging.getLogger(__name__).info(
            "Phoenix tracing enabled → %s (project: %s)", endpoint, project_name
        )
        return True
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to connect to Phoenix at %s: %s\nContinuing without tracing.",
            endpoint,
            exc,
        )
        return False


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)                                         
logging.getLogger("langchain").setLevel(logging.WARNING)                                      
logging.getLogger("langgraph").setLevel(logging.WARNING)
logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)

from agents.agent_interface import (
    AgentInterface,
    LangGraphReActAgent,
)
from agents.llm import create_llm
from agents.mcp_tool_wrapper import MCPToolWrapper


from benchmark.mcp_client import (
    load_mcp_config,
    create_client_and_connect,
    stop_mcp_server,
    MCPConnectionConfig
    )
from benchmark.runner_helpers import (
    save_results_ground_truth,
    save_tools_log,
    load_benchmark_data,
    log_trajectory,
    log_message_history,
    make_output_dir,
    BenchmarkItem,
    BenchmarkResult,
    CapabilityLogger,
)
from benchmark.validate_clients import list_tools_for_domains
from environment.m3.python_tools.mcp.mcp_server import DataPeek

load_dotenv()


# Default MCP connection config file path
DEFAULT_MCP_CONFIG = str(
    Path(__file__).parent / "benchmark" / "mcp_connection_config.yaml"
)
# Timeout for agent execution (seconds)
AGENT_TIMEOUT_SECONDS = float(os.environ.get("AGENT_TIMEOUT_SECONDS", "300"))


async def run_benchmark_for_domain(
    domain: str,
    items: List[BenchmarkItem],
    cfg: MCPConnectionConfig,
    capability_id: int,
    llm,
    max_samples: Optional[int] = None,
    top_k_tools: int = 0,
    max_iterations: Optional[int] = None,
    tlog: CapabilityLogger = None,
) -> List[BenchmarkResult]:
    """Run benchmark for a single domain - starts MCP server once."""
    import time
    if tlog is None:
        tlog = print  # type: ignore[assignment]

    # Limit samples if requested
    if max_samples and max_samples < len(items):
        items = items[:max_samples]

    tlog("\n" + "#" * 60)
    tlog(f"# DOMAIN: {domain} ({len(items)} items)")
    tlog("#" * 60)

    results: List[BenchmarkResult] = []

    try:
        async with AsyncExitStack() as stack:
            # Primary MCP server (always present).
            # For Task 3, cfg.container_command points to bpo_router.py which
            # exec's into the correct server (BPO or M3 REST) based on MCP_DOMAIN.
            session = await stack.enter_async_context(
                create_client_and_connect(cfg, domain)
            )
            # Client-side checksum verification: confirm the server returned
            # the expected tools for this (capability_id, domain) pair.
            from environment.tool_checksums import verify_checksum
            raw_tools = (await session.list_tools()).tools
            verify_checksum(capability_id, domain, raw_tools)

            wrapper = MCPToolWrapper(session)
            tools = await wrapper.get_tools()
            tlog(f"  Loaded {len(tools)} tools for domain '{domain}'")

            agent = _get_agent(capability_id, llm, tools, top_k_tools, max_iterations)

            get_data_tool = next(
                (t for t in tools if t.name == "get_data"), None
            )

            # Run all queries for this domain
            for i, item in enumerate(tqdm(items,desc=f"[{capability_id}|{domain}]")):
                query_suffix = (
                    "..." if len(item.query) > 80 else ""
                )
                tlog(
                    f"\n  [{i+1}/{len(items)}]"
                    f" Query: {item.query[:80]}{query_suffix}"
                )

                # Pre-compute shortlisted tools before the agent runs so the
                # list is captured even when the agent times out or errors.
                if hasattr(agent, '_shortlister') and agent._shortlister is not None:
                    _shortlisted = [t.name for t in agent._shortlister.shortlist(item.query, agent._tools)]
                else:
                    _shortlisted = [t.name for t in tools]

                result = BenchmarkResult(
                    uuid=item.uuid,
                    domain=domain,
                    query=item.query,
                    turn_id=item.turn_id,
                    context=item.context,
                    all_tools=[t.name for t in tools],
                    shortlisted_tools=_shortlisted,
                )

                start_time = time.perf_counter()

                try:
                    if get_data_tool:
                        tlog(f"    Switching to universe: {item.uuid}")
                        data_result = await get_data_tool.ainvoke(
                            {"tool_universe_id": item.uuid}
                        )
                        parsed_data: DataPeek = json.loads(data_result)

                        # Handle MCP TextContent format
                        if isinstance(parsed_data, list) and parsed_data:
                            first_item = parsed_data[0]
                            if (
                                isinstance(first_item, dict)
                                and "text" in first_item
                            ):
                                parsed_data = json.loads(first_item["text"])
                            else:
                                parsed_data = first_item

                        if (
                            isinstance(parsed_data, dict)
                            and "error" in parsed_data
                        ):
                            raise RuntimeError(
                                f"Universe switch failed: {parsed_data['error']}"
                            )

                        tlog("    Universe loaded successfully")
                        assert isinstance(agent, LangGraphReActAgent)
                        agent._initial_data_handle = parsed_data["handle"]
                        agent._initial_data_peek = parsed_data
                        tlog(f"    Initial data handle: {agent._initial_data_handle}")

                    if capability_id in [4]:
                        if not item.context: # Single Turn Dialogues in Capability 4
                            response = await asyncio.wait_for(
                                agent.run(input=item.query, additional_instructions=item.additional_instructions),
                                timeout=AGENT_TIMEOUT_SECONDS
                            )
                        else:
                            response = await asyncio.wait_for(
                                agent.run(input=item.context, additional_instructions=item.additional_instructions),
                                timeout=AGENT_TIMEOUT_SECONDS
                            )
                    else:
                        response = await asyncio.wait_for(
                            agent.run(item.query),
                            timeout=AGENT_TIMEOUT_SECONDS
                        )                        

                    result.answer = response.content
                    result.tool_calls = response.tool_calls
                    result.trajectory = response.trajectory
                    result.status = "success"
                    elapsed = time.perf_counter() - start_time
                    tlog(
                        f"    Status: success"
                        f" | Tools: {len(result.tool_calls)}"
                        f" | Trajectory steps:"
                        f" {len(result.trajectory)}"
                        f" | Time: {elapsed:.2f}s"
                    )
                    # Log the answer
                    answer_preview = (
                        result.answer[:200]
                        if result.answer else "(empty)"
                    )
                    ans_suffix = (
                        "..." if len(result.answer) > 200 else ""
                    )
                    tlog(
                        f"    Answer: {answer_preview}{ans_suffix}"
                    )
                    # Log trajectory summary
                    log_trajectory(result, tlog)
                    log_message_history(result, tlog)
                except asyncio.TimeoutError:
                    result.status = "error"
                    result.error = (
                        f"Agent timed out after"
                        f" {AGENT_TIMEOUT_SECONDS} seconds"
                    )
                    tlog(
                        f"    Status: timeout after"
                        f" {AGENT_TIMEOUT_SECONDS}s"
                    )
                except Exception as e:
                    import traceback
                    result.status = "error"
                    result.error = f"{type(e).__name__} "+str(e)
                    tlog(f"    Status: error | {type(e).__name__}: {str(e)[:200]}")
                    tlog(f"    Traceback: {traceback.format_exc()}")

                result.duration_s = time.perf_counter() - start_time
                results.append(result)

        tlog(f"\n  Server stopped for domain '{domain}'")
    except ExceptionGroup as eg:
        tlog(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            tlog(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(cfg)
            raise

    return results


def _get_agent(capability_id: int, llm, tools, top_k_tools: int = 0, max_iterations: Optional[int] = None) -> AgentInterface:
    """Return the appropriate agent for the given capability_id."""
    kwargs = dict(llm=llm, tools=tools, top_k_tools=top_k_tools)
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if capability_id == 1:
        kwargs["initial_data_handle"] = "placeholder"
    return LangGraphReActAgent(**kwargs)


async def run_capability(
    capability_id: int,
    cfg: MCPConnectionConfig,
    provider: str = "ollama",
    model: Optional[str] = None,
    max_samples_per_domain: Optional[int] = None,
    output_dir: Optional[str] = None,
    domains: Optional[List[str]] = None,
    top_k_tools: int = 0,
    max_iterations: Optional[int] = None,
    restart: bool = False,
    temperature: float = 0.0,
) -> List[BenchmarkResult]:
    """Run benchmark for a given capability_id, iterating over all domain files."""

    all_items, _ = load_benchmark_data(capability_id=capability_id, domains=domains)

    # Group items by domain
    items_by_domain: Dict[str, List[BenchmarkItem]] = {}
    for item in all_items:
        items_by_domain.setdefault(item.domain, []).append(item)

    domain_list = sorted(items_by_domain)

    # Create output dir early so the log file lives alongside the results
    out_dir = make_output_dir(capability_id, output_dir)
    tlog = CapabilityLogger(capability_id, out_dir / "run.log")

    tlog(f"Capability ID: {capability_id}")
    tlog(f"Mode: {cfg.mode}")
    if not cfg.command and cfg.mode == "stdio":
        tlog(f"Container name: {cfg.container_name}")
    elif cfg.mode in ("http", "streamable-http"):
        tlog(f"Server URL: {cfg.server_url}")
    tlog(f"Processing {len(domain_list)} domain(s): {domain_list}")

    # Skip domains that already have output files (resume support)
    completed = {p.stem for p in out_dir.glob("*.json")}
    if completed:
        skipped = [d for d in domain_list if d in completed]
        domain_list = [d for d in domain_list if d not in completed]
        tlog(f"Skipping {len(skipped)} already-completed domain(s): {skipped}")
        tlog(f"Remaining domain(s) to run: {domain_list}")

    if max_samples_per_domain:
        tlog(f"Max samples per domain: {max_samples_per_domain}")

    # Skip already-completed domains when restarting
    if restart:
        completed = {p.stem for p in out_dir.glob("*.json")}
        if completed:
            tlog(f"Restart mode: skipping {len(completed)} already-completed domain(s): {sorted(completed)}")
            domain_list = [d for d in domain_list if d not in completed]

    llm = create_llm(provider=provider, model=model, temperature=temperature)

    # Process each domain, writing output incrementally
    all_results: List[BenchmarkResult] = []
    for domain in domain_list:
        items = items_by_domain[domain]
        tlog(f"\nLoaded {len(items)} items for domain '{domain}'")

        domain_results = await run_benchmark_for_domain(
            domain=domain,
            items=items,
            cfg=cfg,
            capability_id=capability_id,
            llm=llm,
            max_samples=max_samples_per_domain,
            top_k_tools=top_k_tools,
            max_iterations=max_iterations,
            tlog=tlog,
        )
        all_results.extend(domain_results)
        save_results_ground_truth(domain_results, out_dir)
        save_tools_log(domain_results, out_dir)

    results = all_results

    # Summary
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]

    tlog("\n" + "=" * 60)
    tlog("BENCHMARK SUMMARY")
    tlog("=" * 60)
    tlog(f"  Total items: {len(results)}")
    tlog(f"  Successful: {len(successful)}")
    tlog(f"  Failed: {len(failed)}")
    tlog(f"  Log file: {out_dir / 'run.log'}")
    tlog.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Runner for MCP Server"
    )
    parser.add_argument(
        "--capability_id",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        required=True,
        help="Capability ID to run, must be one of [1, 2, 3, 4]"
    )
    parser.add_argument(
        "--domain",
        type=str,
        action="append",
        default=None,
        help=(
            "Domain(s) to process"
            " (can specify multiple times, default: all domains)"
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools for the specified domain(s) and exit"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple capability_ids in parallel using asyncio.gather (default: sequential)"
    )
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=None,
        help="Maximum number of benchmark items per domain (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: output/capability_{id}_{timestamp}/ in CWD)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=[
            "anthropic", "openai", "ollama",
            "litellm", "watsonx", "rits",
        ],
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider-specific default)"
    )
    parser.add_argument(
        "--top-k-tools",
        type=int,
        default=128,
        help="Enable tool shortlisting: keep top-k tools per query"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum agent iterations per query (default: 10 for task 1, provider default otherwise)"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Resume a previous run: skip domains whose output file already"
            " exists in the output directory (requires --output to point"
            " to a previous run's directory)"
        ),
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=DEFAULT_MCP_CONFIG,
        help=(
            f"Path to MCP connection config YAML file"
            f" (default: {DEFAULT_MCP_CONFIG})"
        ),
    )
    parser.add_argument(
        "--phoenix",
        action="store_true",
        default=False,
        help=(
            "Enable Phoenix/Arize OTEL tracing. "
            "Requires arize-phoenix-otel and openinference-instrumentation-langchain. "
            "Start Phoenix with: docker compose --profile phoenix up -d"
        ),
    )
    parser.add_argument(
        "--phoenix-endpoint",
        type=str,
        default="http://localhost:6006/v1/traces",
        help=(
            "Phoenix OTLP HTTP endpoint (default: http://localhost:6006/v1/traces). "
            "Only used when --phoenix is set."
        ),
    )
    parser.add_argument(
        "--phoenix-project",
        type=str,
        default="enterprise-benchmark",
        help="Phoenix project name for grouping traces (default: enterprise-benchmark)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)"
    )

    args = parser.parse_args()
    capability_ids = args.capability_id  # list of ints now

    mode = "parallel" if args.parallel and len(capability_ids) > 1 else "sequential"
    print("="*60)
    print(f"Benchmark Runner ({mode}, capabilities: {capability_ids})")
    print("="*60)

    if args.phoenix:
        tracing_ok = _setup_phoenix(
            endpoint=args.phoenix_endpoint,
            project_name=args.phoenix_project,
        )
        if tracing_ok:
            print(f"Phoenix tracing enabled → {args.phoenix_endpoint}")
        else:
            print("Phoenix tracing unavailable — continuing without it.")

    # Load MCP connection config from YAML
    mcp_configs = load_mcp_config(args.mcp_config)

    def _make_run_task_coro(tid: int):
        task_cfg = mcp_configs.get(tid, MCPConnectionConfig())
        return run_capability(
            capability_id=tid,
            cfg=task_cfg,
            provider=args.provider,
            model=args.model,
            max_samples_per_domain=args.max_samples_per_domain,
            output_dir=args.output,
            domains=args.domain,
            top_k_tools=args.top_k_tools,
            max_iterations=args.max_iterations,
            restart=args.restart,
            temperature=args.temperature
        )

    def _make_list_tools_coro(tid: int):
        task_cfg = mcp_configs.get(tid, MCPConnectionConfig())
        return list_tools_for_domains(
            capability_id=tid,
            cfg=task_cfg,
            domains=args.domain,
        )

    # Handle --list-tools mode
    if args.list_tools:
        async def _list_all():
            coros = [_make_list_tools_coro(tid) for tid in capability_ids]
            if args.parallel and len(coros) > 1:
                await asyncio.gather(*coros)
            else:
                for c in coros:
                    await c
        asyncio.run(_list_all())
        return

    # Run tasks
    async def _run_all():
        coros = [_make_run_task_coro(tid) for tid in capability_ids]
        if args.parallel and len(coros) > 1:
            await asyncio.gather(*coros)
        else:
            for c in coros:
                await c

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
