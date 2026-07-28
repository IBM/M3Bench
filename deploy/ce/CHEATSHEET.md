# VAKRA on Code Engine — Cheatsheet

One-page command reference. Full detail + gotchas: [RUNBOOK.md](RUNBOOK.md) ·
consumer guide: [USING.md](USING.md). All admin scripts need `VAKRA_CE_ADMIN=1`
(add `YES=1` to skip the prompt). Run from the repo root unless noted.

## 0. One-time setup
```bash
ibmcloud login --sso
ibmcloud plugin install code-engine -f && ibmcloud plugin install cloud-object-storage -f
ibmcloud target -r us-east -g routing && ibmcloud ce project select --name ce-project-routing

# ONE project venv (framework python; don't mix Python tools)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[agents,mcp,init]" boto3 openai         # + langchain-ibm for watsonx
```

## Deploy from scratch
```bash
# 1. data -> COS (~35 GB; --stream = low disk, peak ~25 GB; resumable)
python deploy/ce/0_push_data_to_cos.py --stream

cd deploy/ce
VAKRA_CE_ADMIN=1 YES=1 ./1_create_bucket_and_pds.sh      # HMAC secret + data store
VAKRA_CE_ADMIN=1 YES=1 ./2_build_push_image.sh           # cloud build -> ICR (~15 min)
VAKRA_CE_ADMIN=1 YES=1 ./3_deploy_apps.sh                # 4 apps; writes .ce_urls.env
VAKRA_CE_ADMIN=1 YES=1 ./4_deploy_explorer.sh            # optional hosted explorer
cd ..
```

## Verify
```bash
source deploy/ce/.ce_urls.env
python deploy/ce/5_test_ce.py                            # health + list_tools per cap (expect 4/4)
python deploy/ce/5_test_ce.py --verify-checksums         # also assert exact tool sets
```

## Benchmark against CE (same runner, `--mcp-config` differs)
```bash
source .venv/bin/activate && source deploy/ce/.ce_urls.env
CFG=benchmark/mcp_connection_config.ce.yaml
# need data/test locally? python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-research/VAKRA',repo_type='dataset',allow_patterns=['test/**'],local_dir='data')"

# no-LLM sanity (best first check):
python benchmark_runner.py --capability_id 2 --domain california_schools --list-tools --mcp-config $CFG

# real run — OpenAI:
export OPENAI_API_KEY=sk-...
python benchmark_runner.py --capability_id 2 --domain california_schools \
    --max-samples-per-domain 2 --provider openai --model gpt-4o --mcp-config $CFG

# real run — gpt-oss-120b on watsonx (needs: pip install langchain-ibm):
export WATSONX_APIKEY=... WATSONX_PROJECT_ID=...          # optional WATSONX_URL (default us-south)
python benchmark_runner.py --capability_id 2 --domain california_schools \
    --max-samples-per-domain 2 --provider watsonx --model openai/gpt-oss-120b --mcp-config $CFG
```
Results → `output/capability_<id>_<timestamp>/<domain>.json`. Drop `--mcp-config $CFG` to run local.

## Use the endpoints directly (any MCP client)
```
GET  <base>/healthz
MCP  <base>/mcp/<domain>          # streamable HTTP
base = https://vakra-cap{1..4}.<hash>.us-east.codeengine.appdomain.cloud   (see .ce_urls.env)
```

## Operate
```bash
# health / logs / events
source deploy/ce/.ce_urls.env; curl -s "$VAKRA_CAP2_URL/healthz"
ibmcloud ce app logs   -n vakra-cap3
ibmcloud ce app events -n vakra-cap3

# ship a new version (rebuild + redeploy; redeploy delete+recreates, re-pulls :latest)
cd deploy/ce && VAKRA_CE_ADMIN=1 YES=1 ./2_build_push_image.sh && VAKRA_CE_ADMIN=1 YES=1 ./3_deploy_apps.sh

# data changed? re-run step 0 (only changed objects upload)
# cap4 ChromaDB write issues on the mount? redeploy with sync:
VAKRA_CE_ADMIN=1 YES=1 DATA_SOURCE=sync ./3_deploy_apps.sh

# teardown (only vakra-*)
VAKRA_CE_ADMIN=1 YES=1 ./teardown.sh                     # --delete-bucket to also drop data
```

## Facts
- Names: apps `vakra-cap1..4` + `vakra-explorer` · bucket `vakra-benchmark-data-us-east` ·
  data store `vakra-store` · image `icr.io/routing_namespace/vakra-benchmark:latest`.
- Region `us-east`, RG `routing`, project `ce-project-routing`, push secret `icr-secret-1`.
- MCP pinned `mcp==1.27.0`. Data default `DATA_SOURCE=mount` (COS pds), fallback `sync`.
- Valid `--domain` = task-file names under `data/test/capability_<id>_*/input/` (server also
  serves extra domains like `address` that have no benchmark tasks).

## First-aid (full table in RUNBOOK.md)
| Symptom | Fix |
|---|---|
| disk-full during push | free ~10 GB, use `--stream` |
| HF download hangs | Ctrl-C + re-run (resumable); `HF_HUB_DOWNLOAD_TIMEOUT=30` |
| `bucket name not available` | it's yours — re-run step 1 (reuses via bucket-head) |
| `Mount … is a duplicate` | scripts delete+recreate — re-run |
| runner `No module named 'encodings'` | corrupt `.venv` — `rm -rf .venv`, recreate with ONE tool |
| RITS 120b 404 | endpoint drift — use watsonx for 120b, or fix `MODEL_NAME_MAPPING` |
| cap3 `bpo` missing parquet | rebuild image (bakes `environment/bpo/data/`) |
