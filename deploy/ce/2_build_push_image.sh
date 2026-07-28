#!/bin/bash
# ============================================================
# Step 2 — Build the SAME image docker-compose builds (docker/Dockerfile.unified)
# with a Code Engine buildrun (cloud-side; no local docker needed) and push it to
# icr.io/routing_namespace/vakra-benchmark:latest.
#
# The 32 GB dataset is NOT part of the image — it's excluded from the build
# context (staged copy below), so the image stays lean and data lives in COS.
# ============================================================
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/config.sh"
admin_guard "${1:-}"
require_login
ce_target

if ! ibmcloud ce secret get -n "$REGISTRY_SECRET_NAME" >/dev/null 2>&1; then
  echo "Registry secret '$REGISTRY_SECRET_NAME' not found in '$CE_PROJECT_NAME'."; exit 1
fi

# Stage a clean build context that EXCLUDES the giant data/ dir, .git, venvs, etc.
# This captures the current working tree (including uncommitted bridge changes)
# and guarantees we never upload 32 GB to the buildrun.
STAGE="$(mktemp -d -t vakra-build-XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
echo "Staging build context (excluding data/, .git, venvs) -> $STAGE ..."
rsync -a \
  --exclude '/data/' \
  --exclude '.git/' \
  --exclude '.venv/' --exclude '*.venv/' --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.sqlite*' \
  --exclude '/output/' \
  --exclude 'parlant-data/' \
  "$APP_ROOT/" "$STAGE/"
# NOTE: '/data/' is anchored (leading slash) so ONLY the top-level 34 GB data/ is
# skipped — small in-repo data dirs like environment/bpo/data/ (needed by the BPO
# server) are kept. A bare 'data/' would recursively drop those too.

CTX_SIZE=$(du -sh "$STAGE" 2>/dev/null | cut -f1)
echo "Build context size: ${CTX_SIZE:-?}"

uuid=$(uuidgen | tr '[:upper:]' '[:lower:]' | awk -F- '{print $1}')
BUILD_NAME="vakra-build-${uuid}"

echo "Submitting buildrun '$BUILD_NAME' -> $IMAGE_REF (image is large; ~10-20 min) ..."
ibmcloud ce buildrun submit \
  --name "$BUILD_NAME" \
  --source "$STAGE" \
  --strategy dockerfile \
  --dockerfile docker/Dockerfile.unified \
  --image "$IMAGE_REF" \
  --registry-secret "$REGISTRY_SECRET_NAME" \
  --size xlarge \
  --timeout 2400

ibmcloud ce buildrun logs -f -n "$BUILD_NAME"

# Surface the final status so a failed build doesn't look like success.
STATUS=$(ibmcloud ce buildrun get -n "$BUILD_NAME" -o json 2>/dev/null \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('status',{}).get('reason','') or json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
echo ""
echo "Buildrun status: ${STATUS:-see logs above}"
echo "Image: $IMAGE_REF"
echo "Next: VAKRA_CE_ADMIN=1 ./3_deploy_apps.sh"
