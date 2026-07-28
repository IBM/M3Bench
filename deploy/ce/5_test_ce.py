#!/usr/bin/env python3
"""
Step 5 — Smoke-test the deployed VAKRA Code Engine apps.

For each capability it:
  1. GETs /healthz on the app (reports capability_id the bridge sees).
  2. Opens an MCP session for a sample domain over HTTP (same client path the
     benchmark runner uses) and calls list_tools() — proving per-domain scoping
     works end-to-end from your machine against Code Engine.

Run AFTER deploying:
    source deploy/ce/.ce_urls.env
    python deploy/ce/5_test_ce.py                 # all capabilities, default domains
    python deploy/ce/5_test_ce.py --capability 2 --domain address
    python deploy/ce/5_test_ce.py --verify-checksums
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.mcp_client import load_mcp_config, create_client_and_connect  # noqa: E402

# Representative domain(s) to probe per capability. Override with --domain.
DEFAULT_DOMAINS = {1: "address", 2: "address", 3: "bpo", 4: "address"}

GREEN, RED, CYAN, RESET = "\033[0;32m", "\033[0;31m", "\033[0;36m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = CYAN = RESET = ""


def _health(base_url: str) -> str:
    import httpx
    url = base_url.rstrip("/") + "/healthz"
    try:
        r = httpx.get(url, timeout=30)
        return f"{r.status_code} {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {e}"


async def _list_tools(cfg, domain: str):
    async with create_client_and_connect(cfg, domain=domain) as session:
        return (await session.list_tools()).tools


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test VAKRA on Code Engine")
    ap.add_argument("--capability", type=int, default=0, help="Only test this capability (1-4)")
    ap.add_argument("--domain", default="", help="Override the probe domain")
    ap.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "mcp_connection_config.ce.yaml"))
    ap.add_argument("--verify-checksums", action="store_true",
                    help="Also run verify_checksum(cap, domain, tools)")
    args = ap.parse_args()

    configs = load_mcp_config(args.config)
    caps = [args.capability] if args.capability else [1, 2, 3, 4]

    verify = None
    if args.verify_checksums:
        from environment.tool_checksums import verify_checksum as verify

    n_pass = n_fail = 0
    for cap in caps:
        cfg = configs.get(cap)
        print(f"\n{CYAN}=== Capability {cap} ==={RESET}")
        if not cfg:
            print(f"{RED}  [FAIL] capability {cap} not in {args.config}{RESET}"); n_fail += 1; continue

        base = os.environ.get(f"VAKRA_CAP{cap}_URL", "")
        if not base:
            print(f"{RED}  [FAIL] VAKRA_CAP{cap}_URL not set — did you 'source deploy/ce/.ce_urls.env'?{RESET}")
            n_fail += 1
            continue

        print(f"  url: {base}")
        print(f"  health: {_health(base)}")

        domain = args.domain or DEFAULT_DOMAINS.get(cap, "address")
        try:
            tools = asyncio.run(_list_tools(cfg, domain))
            names = [t.name for t in tools]
            print(f"{GREEN}  [PASS] list_tools(domain={domain!r}) -> {len(tools)} tools{RESET}")
            print(f"         e.g. {names[:6]}{' ...' if len(names) > 6 else ''}")
            if verify:
                try:
                    verify(cap, domain, tools)
                    print(f"{GREEN}  [PASS] checksum matches for (cap {cap}, {domain}){RESET}")
                except Exception as e:  # noqa: BLE001
                    print(f"{RED}  [FAIL] checksum: {e}{RESET}"); n_fail += 1; continue
            n_pass += 1
        except Exception as e:  # noqa: BLE001
            print(f"{RED}  [FAIL] list_tools(domain={domain!r}): {type(e).__name__}: {e}{RESET}")
            n_fail += 1

    print(f"\n{'='*40}\n {GREEN}{n_pass} passed{RESET}, {RED if n_fail else ''}{n_fail} failed{RESET}\n{'='*40}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
