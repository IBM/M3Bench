# VAKRA on Code Engine + COS — Reproduce-From-Scratch Runbook

This is the **complete, ordered, battle-tested** procedure to put VAKRA's benchmark
environment on IBM Code Engine with its ~35 GB of data in IBM COS, and to benchmark
against it. Every step lists the command, what it does, the expected output, and the
**gotchas we actually hit** (all already fixed in the scripts — the notes explain why).

It is an **additive, admin-only** path: it never changes the local docker-compose flow,
existing Code Engine apps, or other COS buckets. All resources are named `vakra-*`.

> TL;DR happy path (details below):
> ```
> ibmcloud login --sso
> python deploy/ce/0_push_data_to_cos.py --stream          # data -> COS  (~35 GB, hours)
> cd deploy/ce
> VAKRA_CE_ADMIN=1 YES=1 ./1_create_bucket_and_pds.sh       # secret + data store
> VAKRA_CE_ADMIN=1 YES=1 ./2_build_push_image.sh            # build image (~15 min)
> VAKRA_CE_ADMIN=1 YES=1 ./3_deploy_apps.sh                 # 4 CE apps
> cd .. && source deploy/ce/.ce_urls.env
> python deploy/ce/5_test_ce.py                             # expect 4/4 pass
> python benchmark_runner.py --capability_id 2 --domain california_schools --max-samples-per-domain 2 \
>       --provider openai --model gpt-4o --mcp-config benchmark/mcp_connection_config.ce.yaml
> ```
> All `python …` commands run in the project venv created in Prerequisites (activate it first).

---

## Architecture (what gets deployed and why)

Locally, benchmark clients reach the MCP servers via `docker exec` (stdio), with the
domain injected as `MCP_DOMAIN` into a freshly-spawned server process. **Code Engine has
no `docker exec` and no local bind mounts**, so two seams are bridged — *for CE only*:

1. **Transport** — `docker/mcp_http_bridge.py` serves MCP over HTTP at `/mcp/<domain>`.
   It runs only when the container starts with `SERVE_MODE=http`. For each request it
   spawns a fresh, domain-pinned `mcp_dispatch.py` subprocess (same `MCP_DOMAIN`
   mechanism as `docker exec -e`), so `list_tools()` is scoped to exactly one domain.
2. **Data** — the ~35 GB dataset lives in a COS bucket, mounted into each app at the
   paths the servers already expect (`/app/db`, `/app/environment/configs`, and for
   cap 4 `/app/retrievers/chroma_data` + `/app/retrievers/queries`).

```
your laptop:  benchmark_runner.py --mcp-config …ce.yaml   (reads data/test/ locally for inputs)
                     │  MCP over HTTPS  (/mcp/<domain>)
                     ▼
   4 Code Engine apps: vakra-cap1..4   (same image, SERVE_MODE=http, CAPABILITY_ID=1..4)
     mcp_http_bridge ──spawn(MCP_DOMAIN)──► mcp_dispatch.py ──► FastAPI backends :8000/:8001
     data mounted read-only from COS pds  ◄── vakra-store ◄── bucket vakra-benchmark-data-us-east
```

The 4 capabilities all run one image (`icr.io/routing_namespace/vakra-benchmark:latest`),
differentiated only by `CAPABILITY_ID` and which pds subpaths they mount.

---

## Account conventions (reused from the `routing` account — nothing to fill in)

| Setting | Value |
|---|---|
| Region / resource group / CE project | `us-east` / `routing` / `ce-project-routing` |
| Registry / push secret | `icr.io/routing_namespace` / `icr-secret-1` |
| Capability image | `icr.io/routing_namespace/vakra-benchmark:latest` |
| Explorer image | `icr.io/routing_namespace/vakra-explorer:latest` |
| COS instance | reused from `~/palette/.cos_creds.json` (apikey + HMAC keys) |
| COS bucket (NEW) | `vakra-benchmark-data-us-east` |
| CE HMAC secret / data store | `vakra-cos-access` / `vakra-store` |
| Apps | `vakra-cap1..4`, `vakra-explorer` |

All overridable via env — see `config.sh`.

---

## Prerequisites

- `ibmcloud` CLI with the `code-engine` and `cloud-object-storage` plugins.
- `rsync`, `uuidgen`, and either `uv` or `python3 -m venv`.
- COS creds at `~/palette/.cos_creds.json` (already present; must contain `cos_hmac_keys`).

```bash
ibmcloud login --sso
ibmcloud plugin install code-engine -f
ibmcloud plugin install cloud-object-storage -f
ibmcloud target -r us-east -g routing
ibmcloud ce project select --name ce-project-routing
```

### Create ONE project venv (used for the push, the smoke test, and the runner)

Use a real CPython **3.11/3.12 framework build** (e.g. Homebrew `python3.12`). Don't mix
Python tools — a venv created by one Python but relinked to another corrupts it (the
`No module named 'encodings'` / `base_prefix='/install'` failure we hit; uv's
build-standalone Pythons can trigger it, so a framework `python3.12` is the safe choice).

```bash
cd <repo-root>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[agents,mcp,init]" boto3 openai
```

This installs the package's **core** deps (pandas, numpy, dotenv, pyyaml, sentence-transformers)
plus the `agents` (langchain/langgraph), `mcp`, and `init` (huggingface_hub) extras — the
complete set the push, smoke test, and runner all need. **`requirements_benchmark.txt` alone
is NOT enough** (it omits core deps like `pandas`/`dotenv`). Every `python …` command below
assumes this venv is **activated**. Add a provider client only if you use it:
`langchain-ibm` for `--provider watsonx`, `anthropic` for `--provider anthropic`.

Sanity-check it boots (a healthy base, not `/install`):
`python -c "import sys; print(sys.base_prefix)"`.

> **Admin gate:** every resource-creating script requires `VAKRA_CE_ADMIN=1` and prompts
> once. Add `YES=1` (or pass `-y`) to skip the prompt.

---

## Step 1 — Push the data to COS

Reuses the existing HuggingFace download (`benchmark_setup.py --download-data`) and mirrors
`data/{databases,indexed_documents,queries,test}` + `environment/configs` to the bucket
with the same layout the containers expect. The bucket is **auto-created** here (idempotent).

```bash
cd <repo-root>
python3 deploy/ce/0_push_data_to_cos.py --stream
```

- **Use `--stream`** unless you have ≥ ~40 GB free disk. It downloads one subtree →
  uploads it → deletes it → next, so peak local use is ~25 GB (the `databases` subtree)
  instead of the full ~35 GB. The full path (`0_push_data_to_cos.py` with no flag) needs
  ~35 GB free and refuses to run under 35 GB.
- Other flags: `--skip-download` (upload an existing `data/`), `--only databases,configs`,
  `--dry-run`.
- **Resumable:** HF download resumes via `.incomplete` files; the upload skips objects whose
  size already matches. Safe to re-run after any interruption.

**Verify what landed** (any Python with boto3):
```bash
python3 - <<'PY'
import json, boto3
from botocore.client import Config
c=json.load(open('/Users/anu/palette/.cos_creds.json'))['credentials']
r="us-east"; b=f"vakra-benchmark-data-{r}"
cli=boto3.client("s3",endpoint_url=f"https://s3.{r}.cloud-object-storage.appdomain.cloud",
  aws_access_key_id=c['cos_hmac_keys']['access_key_id'],
  aws_secret_access_key=c['cos_hmac_keys']['secret_access_key'],config=Config(signature_version="s3v4"))
n=tot=0; per={}
for pg in cli.get_paginator("list_objects_v2").paginate(Bucket=b):
  for o in pg.get("Contents",[]):
    n+=1; tot+=o["Size"]; t=o["Key"].split("/")[0]; per[t]=per.get(t,0)+o["Size"]
print(f"{b}: {n} objects, {tot/1e9:.1f} GB"); [print(f"  {k:20s}{v/1e9:6.2f} GB") for k,v in sorted(per.items())]
PY
```
Expected ≈ `databases 26.9 GB · indexed_documents 7.8 GB · test 0.05 GB · configs`.

> **Gotchas we hit here**
> - *Only 31 GB free / disk 97 % full* → the full download won't fit. Free ~10 GB and use
>   `--stream` (peak ~25 GB). The script guards this and points you to `--stream`.
> - *HF download hung for hours on one file* (dead connection, no read timeout). Fix: the
>   script is resumable — Ctrl-C and re-run; set `HF_HUB_DOWNLOAD_TIMEOUT=30` so a stall
>   errors-and-retries instead of hanging. `HF_HUB_ENABLE_HF_TRANSFER=1` (with
>   `pip install hf_transfer`) is faster/more resilient.
> - *A 5 GB `.sqlite` seemed "stuck"* — it was mid-multipart; a multipart object only appears
>   in the bucket once all parts finish. Not stuck.
> - *`boto3` not found* even after `pip install` → it went to a different interpreter. Install
>   into the same Python you run the script with (`python3 -m pip install boto3`).

---

## Step 2 — HMAC secret + persistent data store

The bucket already exists (Step 1). This adds a Code Engine HMAC access secret and registers
the bucket as a **persistent data store** (`vakra-store`) that the apps mount.

```bash
cd <repo-root>/deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./1_create_bucket_and_pds.sh
```

Expected tail:
```
  bucket already exists and is accessible — reusing.
  Creating CE HMAC secret 'vakra-cos-access' ...
  Creating persistent data store 'vakra-store' -> bucket 'vakra-benchmark-data-us-east' ...
Done.
```

> **Gotcha:** `bucket-create` returned *"The requested bucket name is not available"* (COS
> names are globally unique — it's yours from Step 1). The script now falls back to
> `ibmcloud cos bucket-head`; if the bucket is accessible, it reuses it instead of failing.

---

## Step 3 — Build + push the image (cloud buildrun)

Builds the *same* image docker-compose uses (`docker/Dockerfile.unified`) in the cloud and
pushes to ICR. The 35 GB dataset is **excluded** from the build context (staged copy).

```bash
cd <repo-root>/deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./2_build_push_image.sh          # ~10–20 min (large ML image)
```

Watch the streamed log for the `pip install`, the granite embedding-model pre-download, and
the final push. Ends with `Buildrun status: succeeded`.

> **Gotchas we hit here**
> - *cap3 `bpo` failed at runtime with `candidate_data.parquet not found`.* The rsync that
>   stages the build context used `--exclude 'data/'` — unanchored, so rsync dropped **every**
>   `data/` dir, including the small in-repo `environment/bpo/data/` the BPO server needs.
>   Fixed: the exclude is now `--exclude '/data/'` (anchored to the top-level 35 GB only), and
>   the root `.dockerignore` uses `/data/` too. In-repo data dirs are now baked in.
> - *`mcp` behaved inconsistently across rebuilds.* Both images now pin `mcp==1.27.0` (the
>   version proven locally), which also busts stale BuildKit pip-layer caches.

---

## Step 4 — Deploy the 4 capability apps

Creates `vakra-cap1..4` (same image, `SERVE_MODE=http`), mounts the COS data store, and writes
the public URLs to `deploy/ce/.ce_urls.env`.

```bash
cd <repo-root>/deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./3_deploy_apps.sh
```

- **Default `DATA_SOURCE=mount`** (COS pds mount). Fallback: `DATA_SOURCE=sync` downloads the
  data to the container's ephemeral disk at startup (bigger instances; use if the s3fs mount
  is too slow, or if cap 4's ChromaDB needs writes):
  `VAKRA_CE_ADMIN=1 YES=1 DATA_SOURCE=sync ./3_deploy_apps.sh`
- Re-running **deletes + recreates** each app. The app name/URL is stable across recreate.

Confirm health (Ready + `/healthz` on all 4):
```bash
cd <repo-root> && source deploy/ce/.ce_urls.env
ibmcloud ce project select --name ce-project-routing >/dev/null
for i in 1 2 3 4; do n="vakra-cap$i"; u=$(eval echo \$VAKRA_CAP${i}_URL)
  printf "%-11s " "$n"; ibmcloud ce app get -n "$n" -o json | python3 -c "import sys,json;print('Ready='+ {c['type']:c['status'] for c in json.load(sys.stdin)['status']['conditions']}.get('Ready','?'))"
  curl -s -m 15 "$u/healthz"; echo; done
```

> **Gotchas we hit here**
> - *`ce app update` → "Mount directory … is a duplicate".* `app update` can't re-apply an
>   existing `--mount-data-store` and has no `--unmount`. Fixed: the script now **delete +
>   recreate**s each app (with `app delete --wait`).
> - *CE doesn't re-pull an unchanged `:latest`.* The script stamps a per-deploy `DEPLOY_REV`
>   env so every deploy is a new revision and pulls the freshly-built image.
> - *The bridge crashed with "attempted to exit a cancel scope…".* The first bridge version
>   cached one backend subprocess and reused it across HTTP request tasks — an anyio
>   scope violation. Fixed: spawn a fresh, scope-local subprocess per request (validated with
>   a local dummy-server harness before rebuilding).

---

## Step 5 — Smoke test (functional health)

Proves each app lists tools for a sample domain over MCP (exercises the data mount + per-domain
scoping) — the real "is it working" check.

```bash
cd <repo-root> && source deploy/ce/.ce_urls.env
python deploy/ce/5_test_ce.py
# add --verify-checksums to assert the exact tool set per (capability, domain)
```
Expected: `4 passed, 0 failed` — e.g. cap1 → 9 tools, cap2 → 139 (`address`), cap4 → 144,
cap3 (`bpo`) → BPO tools.

---

## Step 6 — Benchmark against Code Engine (the runner)

The runner is **unchanged** vs. a local run — only `--mcp-config` differs. Per-domain
scoping is automatic: `create_client_and_connect(cfg, domain)` connects to
`…/mcp/<domain>`, so `list_tools()` returns just that domain's tools and
`verify_checksum` still passes.

**Three prerequisites** (beyond a deployed, green Step 5):

1. **The project venv from Prerequisites, activated** (`source .venv/bin/activate`). It
   already has the agent stack + `openai`. Add `anthropic` if you use `--provider anthropic`.
2. **Local benchmark inputs.** The runner reads task inputs from **local**
   `data/test/capability_<id>_*/input/*.json` (only *tools* come from CE). If `data/test/`
   isn't present, fetch just it (~45 MB):
   ```bash
   cd <repo-root>
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download("ibm-research/VAKRA", repo_type="dataset",
                     allow_patterns=["test/**"], local_dir="data")
   PY
   ```
   **Valid `--domain` values are the task-file names there** — e.g. cap 2 has
   `california_schools, card_games, chicago_crime, financial, formula_1, movie, movielens, …`.
   (Note: the server also serves domains that have no benchmark tasks — e.g. `address` works
   in `5_test_ce.py`'s `list_tools` but isn't a cap 2 *benchmark* domain. Pick a wrong domain
   and the runner prints the available list.)
3. **An LLM provider + key** (see the provider table below). `--provider` is one of
   `openai|watsonx|rits|anthropic|litellm|ollama` (default `ollama`).

**Run it (point the same runner at CE via the CE config):**
```bash
cd <repo-root>
source .venv/bin/activate                       # the project venv from Prerequisites
source deploy/ce/.ce_urls.env
CFG=benchmark/mcp_connection_config.ce.yaml

# (a) zero-LLM sanity — just list a domain's tools via CE (no key needed):
python benchmark_runner.py --capability_id 2 --domain california_schools --list-tools --mcp-config $CFG

# (b) tiny real benchmark — 2 samples, cap2/california_schools, OpenAI:
export OPENAI_API_KEY=sk-...
python benchmark_runner.py --capability_id 2 --domain california_schools \
    --max-samples-per-domain 2 --provider openai --model gpt-4o --mcp-config $CFG

# (c) all of cap 2's domains, in parallel:
python benchmark_runner.py --capability_id 2 --parallel \
    --provider openai --model gpt-4o --mcp-config $CFG
```

Results are written to `output/capability_<id>_<timestamp>/<domain>.json`. To run locally
instead of CE, drop `--mcp-config $CFG` (defaults to the docker-exec config).

### LLM providers (env to export before running)

| `--provider` | `--model` | Required env | Notes |
|---|---|---|---|
| `openai` | `gpt-4o` (any) | `OPENAI_API_KEY` | client already installed |
| `watsonx` | `openai/gpt-oss-120b` (default) | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` (or `WATSONX_SPACE_ID`); optional `WATSONX_URL` (default `https://us-south.ml.cloud.ibm.com`) | **the reliable gpt-oss-120b path**; needs `pip install langchain-ibm`; model must exist in your region |
| `anthropic` | `claude-sonnet-4-5-…` | `ANTHROPIC_API_KEY` | needs `pip install anthropic` |
| `rits` | `gpt-oss-120b` (short key) | `RITS_API_KEY` | URL = `{base}/{short-key}/v1/chat/completions`; `MODEL_NAME_MAPPING` in `agents/llm.py` maps short→payload. If it 404s, the endpoint name drifted — fix the mapping (prefer watsonx for 120b). |
| `litellm` | provider-specific | `LITELLM_API_KEY`, `LITELLM_BASE_URL` | |
| `ollama` | e.g. `llama3.1:8b` | — (local) | default provider |

**gpt-oss-120b via watsonx** (recommended):
```bash
export WATSONX_APIKEY=<ibm-cloud-api-key>
export WATSONX_PROJECT_ID=<watsonx-project-id>      # or WATSONX_SPACE_ID
# export WATSONX_URL=https://us-south.ml.cloud.ibm.com   # set if your region differs
python benchmark_runner.py --capability_id 2 --domain california_schools \
    --max-samples-per-domain 2 --provider watsonx --model openai/gpt-oss-120b --mcp-config $CFG
```

> Notes: `--list-tools` is the best first check (no LLM, no cost — just proves CE
> connectivity + scoping). `--max-samples-per-domain N` keeps a first run cheap. If a local
> step ever needs the BPO parquet, `pip install pyarrow`.

## Using the deployed endpoints (for consumers, not deployers)

If you just want to *use* the already-deployed apps (call the MCP endpoints, run the
benchmark, browse tools) — not deploy them — see **[USING.md](USING.md)**. For a condensed
command reference, see **[CHEATSHEET.md](CHEATSHEET.md)**.

---

## Step 7 — (optional) Hosted MCP Tools Explorer

```bash
cd <repo-root>/deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./4_deploy_explorer.sh
```
Prints the explorer URL. Or run it locally against CE:
```bash
source deploy/ce/.ce_urls.env
MCP_CONFIG=benchmark/mcp_connection_config.ce.yaml \
  python -m uvicorn tools_explorer.app:app --port 7860
```

> **Gotchas we hit here**
> - *Build: `"/benchmark": not found`.* The root `.dockerignore` (tuned for the main image)
>   excludes `benchmark/`, which the explorer needs. Fixed: the explorer build drops that
>   `.dockerignore` from its (already data-free) staged context.
> - *Runtime: `cannot import name 'streamablehttp_client'`.* Stale cached pip layer with an old
>   `mcp`. Fixed by pinning `mcp==1.27.0` (busts the cache).
> - *Deploy: duplicate mount* — same delete+recreate fix as Step 4.
> - cap3 `bpo` only works in the explorer **after** Step 3/4 include the BPO parquet.

---

## Ship a new version later
```bash
cd deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./2_build_push_image.sh && VAKRA_CE_ADMIN=1 YES=1 ./3_deploy_apps.sh
```
Data changed? Re-run Step 1 (only changed objects upload).

## Teardown (only `vakra-*`; never touches other resources)
```bash
cd deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./teardown.sh                  # keeps the bucket + data
VAKRA_CE_ADMIN=1 YES=1 ./teardown.sh --delete-bucket  # also empties + deletes the bucket
```

---

## Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| `only N GB free` refusal in Step 1 | Free ~10 GB and use `--stream` (peak ~25 GB). |
| HF download stalls for a long time | Ctrl-C + re-run (resumable); set `HF_HUB_DOWNLOAD_TIMEOUT=30`. |
| `boto3` not found | `python3 -m pip install boto3` into the *same* interpreter you run scripts with. |
| `bucket name is not available` (Step 2) | It's yours; script now reuses via `bucket-head`. Re-run. |
| `candidate_data.parquet not found` (cap3) | The in-repo `environment/bpo/data/` must be baked in — rebuild with the anchored `/data/` excludes (Step 3). |
| `Mount directory … is a duplicate` | `app update` can't re-mount; scripts now delete+recreate. Re-run. |
| App serves old code after rebuild | `DEPLOY_REV` forces a new revision; or delete+recreate. |
| Bridge `cancel scope` crash | Fresh-subprocess-per-request bridge (current code). Rebuild. |
| Explorer `streamablehttp_client` ImportError | Pin `mcp==1.27.0` (busts stale pip cache). Rebuild explorer. |
| cap4 ChromaDB write errors on the mount | Redeploy with `DATA_SOURCE=sync`. |
| Runner: `ModuleNotFoundError: No module named 'encodings'` / `base_prefix='/install'` | A **corrupt `.venv`** (mixed Pythons, or a build-standalone base). `deactivate; rm -rf .venv`, then recreate with a framework Python per Prerequisites: `python3.12 -m venv .venv`. |
| Diagnosing any app | `ibmcloud ce app logs -n vakra-capN` and `ibmcloud ce app events -n vakra-capN`. |
