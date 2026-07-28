#!/bin/bash
# ============================================================
# Step 1 — Create the NEW COS bucket, an HMAC access secret, and a Code Engine
# persistent data store (pds) that mounts that bucket into the apps.
#
# Reuses the EXISTING palette COS instance (never creates a new instance, never
# touches other buckets). Safe to re-run: skips/refreshes what already exists.
# ============================================================
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/config.sh"
admin_guard "${1:-}"
require_login
ce_target

if [[ -z "${COS_INSTANCE_CRN:-}" || -z "${COS_HMAC_ACCESS_KEY_ID:-}" ]]; then
  echo "Could not read COS creds (apikey/CRN/HMAC) from $COS_CREDS_JSON"; exit 1
fi

# --- 1a. Create the bucket (regional, in $REGION). Reuse if already owned. ---
echo "Configuring COS CLI for instance ${COS_INSTANCE_CRN} ..."
ibmcloud cos config crn --crn "$COS_INSTANCE_CRN" >/dev/null 2>&1 || true

echo "Creating bucket '$COS_BUCKET' in region '$REGION' ..."
if ibmcloud cos bucket-create --bucket "$COS_BUCKET" --region "$REGION" \
      --ibm-service-instance-id "$COS_INSTANCE_CRN" 2>/tmp/vakra_cos_err; then
  echo "  bucket created."
elif ibmcloud cos bucket-head --bucket "$COS_BUCKET" --region "$REGION" >/dev/null 2>&1; then
  # Create can fail with "already exists" / "name is not available" (the COS
  # namespace is global). If we can head the bucket, it's ours — reuse it.
  echo "  bucket already exists and is accessible — reusing."
else
  echo "  bucket-create failed and the bucket is not accessible with these creds:"
  cat /tmp/vakra_cos_err
  exit 1
fi

# --- 1b. HMAC access secret that the pds uses to read the bucket ------------
echo "Creating CE HMAC secret '$COS_ACCESS_SECRET_NAME' ..."
ibmcloud ce secret delete --name "$COS_ACCESS_SECRET_NAME" -f >/dev/null 2>&1 || true
ibmcloud ce secret create --name "$COS_ACCESS_SECRET_NAME" \
  --format hmac \
  --access-key-id "$COS_HMAC_ACCESS_KEY_ID" \
  --secret-access-key "$COS_HMAC_SECRET_ACCESS_KEY"

# --- 1c. Persistent data store bound to the bucket -------------------------
echo "Creating persistent data store '$PDS_NAME' -> bucket '$COS_BUCKET' ..."
if ibmcloud ce pds get --name "$PDS_NAME" >/dev/null 2>&1; then
  echo "  pds '$PDS_NAME' already exists — deleting and recreating to ensure fresh binding."
  ibmcloud ce pds delete --name "$PDS_NAME" -f >/dev/null 2>&1 || true
fi
ibmcloud ce pds create --name "$PDS_NAME" \
  --cos-access-secret "$COS_ACCESS_SECRET_NAME" \
  --cos-bucket-name "$COS_BUCKET" \
  --cos-bucket-location "$REGION"

echo ""
echo "Done."
echo "  bucket : $COS_BUCKET"
echo "  secret : $COS_ACCESS_SECRET_NAME (hmac)"
echo "  pds    : $PDS_NAME"
echo "Next: VAKRA_CE_ADMIN=1 ./2_build_push_image.sh"
