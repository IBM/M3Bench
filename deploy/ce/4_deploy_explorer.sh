#!/bin/bash
# ============================================================
# Step 4 (OPTIONAL) — Build + deploy the MCP Tools Explorer as a 5th Code Engine
# app, pointed at the 4 deployed capability apps. Gives you a shareable hosted
# explorer. (You can also just run it locally — see the README.)
#
# Requires step 3 to have run (reads deploy/ce/.ce_urls.env).
# ============================================================
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/config.sh"
admin_guard "${1:-}"
require_login
ce_target

if [[ ! -f "$URLS_ENV_FILE" ]]; then
  echo "Missing $URLS_ENV_FILE — run ./3_deploy_apps.sh first."; exit 1
fi
source "$URLS_ENV_FILE"

# --- Build the explorer image (cloud-side buildrun from a clean staged context) ---
STAGE="$(mktemp -d -t vakra-explorer-XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
rsync -a --exclude '/data/' --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
      --exclude '*.sqlite*' "$APP_ROOT/" "$STAGE/"
# The repo's root .dockerignore is tuned for the MAIN image (it excludes benchmark/,
# *.json, etc.). The explorer image needs benchmark/ + tools_explorer/, so drop that
# ignore file from the (already data-free) staged context.
rm -f "$STAGE/.dockerignore"

uuid=$(uuidgen | tr '[:upper:]' '[:lower:]' | awk -F- '{print $1}')
BUILD_NAME="vakra-explorer-build-${uuid}"
echo "Building explorer image -> $EXPLORER_IMAGE_REF ..."
ibmcloud ce buildrun submit \
  --name "$BUILD_NAME" \
  --source "$STAGE" \
  --strategy dockerfile \
  --dockerfile deploy/ce/Dockerfile.explorer \
  --image "$EXPLORER_IMAGE_REF" \
  --registry-secret "$REGISTRY_SECRET_NAME" \
  --size medium \
  --timeout 900
ibmcloud ce buildrun logs -f -n "$BUILD_NAME"

# --- Deploy the explorer app ---
args=(
  --image "$EXPLORER_IMAGE_REF"
  --registry-secret "$REGISTRY_SECRET_NAME"
  --port 8080
  --min-scale "${MIN_SCALE}" --max-scale 1
  --cpu 0.5 --memory 1G
  --env "MCP_CONFIG=benchmark/mcp_connection_config.ce.yaml"
  --env "VAKRA_CAP1_URL=$VAKRA_CAP1_URL"
  --env "VAKRA_CAP2_URL=$VAKRA_CAP2_URL"
  --env "VAKRA_CAP3_URL=$VAKRA_CAP3_URL"
  --env "VAKRA_CAP4_URL=$VAKRA_CAP4_URL"
  # Mount the small test/ tree so the domain dropdown is populated from real data.
  --mount-data-store "/app/data/test=${PDS_NAME}:test"
)

# Delete + recreate (ce app update can't re-apply an existing --mount-data-store).
if ibmcloud ce app get -n "$APP_EXPLORER" >/dev/null 2>&1; then
  echo "Deleting existing explorer app '$APP_EXPLORER' for a clean redeploy ..."
  ibmcloud ce app delete --name "$APP_EXPLORER" --force --wait --ignore-not-found
fi
echo "Creating explorer app '$APP_EXPLORER' ..."
ibmcloud ce app create --name "$APP_EXPLORER" "${args[@]}"

URL=$(ibmcloud ce app get --name "$APP_EXPLORER" --output url)
echo ""
echo "==================================================="
echo " MCP Tools Explorer (hosted) is live:"
echo "   $URL"
echo "==================================================="
