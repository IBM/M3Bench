#!/usr/bin/env python3
"""
cos_sync.py — pull a COS bucket prefix to a local directory (S3 / HMAC).

Used ONLY when the container runs with DATA_SOURCE=sync (the fallback to the
default COS pds mount). It downloads the benchmark data from the VAKRA bucket to
the container's local (ephemeral) disk at startup, giving the servers full
local-filesystem semantics — useful if the s3fs mount proves too slow, or for the
writable ChromaDB path in capability 4.

Reads from env (set by deploy/ce/3_deploy_apps.sh via the COS access secret):
    COS_ENDPOINT           e.g. https://s3.us-east.cloud-object-storage.appdomain.cloud
    COS_BUCKET             e.g. vakra-benchmark-data-us-east
    COS_ACCESS_KEY_ID      HMAC access key id
    COS_SECRET_ACCESS_KEY  HMAC secret access key

Usage:
    python cos_sync.py --prefix databases --dest /app/db
    python cos_sync.py --prefix configs   --dest /app/environment/configs
"""
import argparse
import os
import sys
from pathlib import Path


def _client():
    import boto3  # from ibm COS S3-compatible endpoint; boto3 is installed in the image
    from botocore.client import Config

    endpoint = os.environ["COS_ENDPOINT"]
    key = os.environ["COS_ACCESS_KEY_ID"]
    secret = os.environ["COS_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


def sync(prefix: str, dest: str) -> None:
    bucket = os.environ["COS_BUCKET"]
    prefix = prefix.strip("/") + "/"
    dest_root = Path(dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    client = _client()
    paginator = client.get_paginator("list_objects_v2")

    n_files = 0
    n_bytes = 0
    n_skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = dest_root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # Skip if same size already present (cheap resume).
            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                n_skipped += 1
                continue
            client.download_file(bucket, key, str(local_path))
            n_files += 1
            n_bytes += obj["Size"]
            if n_files % 25 == 0:
                print(f"  [{prefix}] {n_files} files, {n_bytes / 1e9:.2f} GB ...", flush=True)

    print(f"  [{prefix}] done: downloaded {n_files} file(s) ({n_bytes / 1e9:.2f} GB), "
          f"skipped {n_skipped} already-present.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync a COS prefix to a local dir")
    ap.add_argument("--prefix", required=True, help="Bucket prefix, e.g. 'databases'")
    ap.add_argument("--dest", required=True, help="Local destination directory")
    args = ap.parse_args()

    for var in ("COS_ENDPOINT", "COS_BUCKET", "COS_ACCESS_KEY_ID", "COS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            print(f"cos_sync: missing env {var}", file=sys.stderr)
            sys.exit(1)

    print(f"cos_sync: {os.environ['COS_BUCKET']}/{args.prefix} -> {args.dest}", flush=True)
    sync(args.prefix, args.dest)


if __name__ == "__main__":
    main()
