#!/bin/bash
# ============================================================
# Teardown — remove ONLY the vakra-* Code Engine resources this path created.
# Never touches other apps, other buckets, or the shared COS instance.
#
# By default the COS bucket + its 32 GB of data are KEPT (redeploy is cheap).
# Pass --delete-bucket to also empty and delete the bucket.
# ============================================================
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/config.sh"
admin_guard "${1:-}"
require_login
ce_target

DELETE_BUCKET=0
for a in "$@"; do [[ "$a" == "--delete-bucket" ]] && DELETE_BUCKET=1; done

echo "Deleting Code Engine apps ..."
for name in "$APP_CAP1" "$APP_CAP2" "$APP_CAP3" "$APP_CAP4" "$APP_EXPLORER"; do
  ibmcloud ce app delete --name "$name" -f >/dev/null 2>&1 && echo "  deleted app $name" || echo "  (no app $name)"
done

echo "Deleting persistent data store '$PDS_NAME' ..."
ibmcloud ce pds delete --name "$PDS_NAME" -f >/dev/null 2>&1 && echo "  deleted pds" || echo "  (no pds)"

echo "Deleting HMAC secret '$COS_ACCESS_SECRET_NAME' ..."
ibmcloud ce secret delete --name "$COS_ACCESS_SECRET_NAME" -f >/dev/null 2>&1 && echo "  deleted secret" || echo "  (no secret)"

rm -f "$URLS_ENV_FILE" 2>/dev/null || true

if [[ "$DELETE_BUCKET" == "1" ]]; then
  echo "Emptying and deleting bucket '$COS_BUCKET' ..."
  ibmcloud cos config crn --crn "$COS_INSTANCE_CRN" >/dev/null 2>&1 || true
  # Delete all objects, then the bucket. (Requires the cos plugin.)
  ibmcloud cos objects-delete-all --bucket "$COS_BUCKET" --region "$REGION" --force >/dev/null 2>&1 || true
  ibmcloud cos bucket-delete --bucket "$COS_BUCKET" --region "$REGION" --force >/dev/null 2>&1 \
    && echo "  deleted bucket" || echo "  (could not delete bucket — delete manually if needed)"
else
  echo "Kept bucket '$COS_BUCKET' (pass --delete-bucket to remove it)."
fi

echo "Teardown complete."
