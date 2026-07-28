# Using the deployed VAKRA endpoints

For people who want to **use** the already-running Code Engine apps — call the MCP tools,
run the benchmark, or browse tools — **without deploying anything**. (Deployers: see
[RUNBOOK.md](RUNBOOK.md).)

## What's deployed — the contract

Four public HTTPS apps, each an MCP server for one capability, reachable over
**streamable HTTP** at `/mcp/<domain>`:

| Capability | App URL (base) | What it serves |
|---|---|---|
| 1 — bi_apis (slot-fill/selection) | `https://vakra-cap1.<hash>.us-east.codeengine.appdomain.cloud` | RouterMCP tools |
| 2 — dashboard_apis (SQL/REST) | `https://vakra-cap2.<hash>…` | M3 REST tools |
| 3 — multihop_reasoning | `https://vakra-cap3.<hash>…` | BPO **or** M3 REST (by domain) |
| 4 — multiturn (REST + retriever) | `https://vakra-cap4.<hash>…` | M3 REST + semantic retriever |

- **MCP endpoint:** `<base>/mcp/<domain>` (streamable HTTP). The domain is scoped per
  connection — you only see that domain's tools.
- **Health:** `GET <base>/healthz` → `{"status":"ok","capability_id":"N"}`.
- **Auth:** none — the endpoints are **public**. Treat the URLs as semi-secret.

### Get the URLs
Ask the deployer for `deploy/ce/.ce_urls.env` (it lists all four), or if you have Code
Engine access:
```bash
ibmcloud ce project select --name ce-project-routing
for i in 1 2 3 4; do ibmcloud ce app get -n vakra-cap$i --output url; done
```

### Valid domains
`<domain>` must be one this capability serves. The definitive list per capability is the
task-file names under `data/test/capability_<id>_*/input/` in the repo, or just browse them
in the hosted explorer (below). Examples for cap 2: `california_schools, card_games,
chicago_crime, financial, formula_1, movie, movielens`.

---

## Option A — call the MCP tools directly (any MCP client)

Minimal Python client (needs `pip install "mcp>=1.9"`):
```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

BASE   = "https://vakra-cap2.<hash>.us-east.codeengine.appdomain.cloud"
DOMAIN = "california_schools"

async def main():
    url = f"{BASE}/mcp/{DOMAIN}"
    async with streamablehttp_client(url, headers={"X-MCP-Domain": DOMAIN}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print(f"{len(tools)} tools:", [t.name for t in tools][:5], "...")
            # result = await s.call_tool(tools[0].name, { ... })

asyncio.run(main())
```
Any streamable-HTTP MCP client works the same way — point it at `<base>/mcp/<domain>`. The
`X-MCP-Domain` header is optional (the path already carries the domain).

Quick curl health check (no MCP client needed):
```bash
curl -s https://vakra-cap2.<hash>.us-east.codeengine.appdomain.cloud/healthz
```

---

## Option B — run the VAKRA benchmark against these endpoints

Same runner as local; only `--mcp-config` differs.

```bash
git clone <this repo> && cd vakra-july25
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[agents,mcp,init]" openai        # + langchain-ibm for --provider watsonx

# point the client at the deployed apps
source deploy/ce/.ce_urls.env                      # from the deployer; sets VAKRA_CAP1..4_URL
# benchmark inputs are read locally — fetch just data/test if you don't have it (~45 MB):
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-research/VAKRA', repo_type='dataset', allow_patterns=['test/**'], local_dir='data')"

CFG=benchmark/mcp_connection_config.ce.yaml
# no-LLM sanity:
python benchmark_runner.py --capability_id 2 --domain california_schools --list-tools --mcp-config $CFG
# a real run (pick a provider — see RUNBOOK.md "LLM providers"):
export WATSONX_APIKEY=... WATSONX_PROJECT_ID=...
python benchmark_runner.py --capability_id 2 --domain california_schools \
    --max-samples-per-domain 2 --provider watsonx --model openai/gpt-oss-120b --mcp-config $CFG
```
`.ce_urls.env` fills the `${VAKRA_CAPn_URL}` placeholders in
`benchmark/mcp_connection_config.ce.yaml`; the runner appends `/mcp/<domain>` per run, so
domain scoping is automatic. Full provider/env details: [RUNBOOK.md](RUNBOOK.md) → *LLM providers*.

---

## Option C — browse tools in the hosted explorer

If the deployer ran step 4, there's a hosted explorer:
```
https://vakra-explorer.<hash>.us-east.codeengine.appdomain.cloud
```
Pick a capability + domain to list tools and invoke them from the browser — no setup.

---

## Notes & limits

- **Public, unauthenticated** — anyone with a URL can call the tools. Don't post the URLs
  publicly; ask the deployer to add a bearer-token gate in `mcp_http_bridge.py` if needed.
- **Per-call latency:** the bridge spawns a fresh subprocess per MCP call, so heavy
  `--parallel` runs carry some overhead. Fine for interactive/small runs.
- **cap 3 `bpo` domain** needs the BPO data baked into the image (deployer's build step) —
  other cap 3 domains use the M3 REST path.
- If a call fails, the deployer can check `ibmcloud ce app logs -n vakra-cap<N>`.
