#!/usr/bin/env python3
"""
Step 0 — Push VAKRA benchmark data to the new COS bucket.

Reuses the EXISTING download (`make download` -> benchmark_setup.py) to pull the
HuggingFace dataset to local `data/`, then mirrors it (plus environment/configs)
to the COS bucket with the SAME layout the containers expect:

    data/databases         -> <bucket>/databases/...
    data/indexed_documents -> <bucket>/indexed_documents/...
    data/queries           -> <bucket>/queries/...          (if present)
    data/test              -> <bucket>/test/...             (for the CE explorer)
    environment/configs    -> <bucket>/configs/...

Idempotent: objects whose size already matches are skipped.

Prereqs:  pip install boto3        (only needed on the machine that pushes)
          COS service-credential JSON at $COS_CREDS_JSON (default ~/.cos_creds.json)

Usage:
    # from repo root
    python deploy/ce/0_push_data_to_cos.py                 # download (if needed) + upload
    python deploy/ce/0_push_data_to_cos.py --skip-download  # upload only
    python deploy/ce/0_push_data_to_cos.py --only databases,configs
    python deploy/ce/0_push_data_to_cos.py --dry-run
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "ibm-research/VAKRA"

# Top-level dataset subtrees to stream (download -> upload -> delete), largest first.
# 'configs' comes from the repo (tiny) and is handled separately.
HF_PREFIXES = ["databases", "indexed_documents", "queries", "test"]

# local dir  ->  bucket prefix
UPLOAD_MAP = {
    REPO_ROOT / "data" / "databases":         "databases",
    REPO_ROOT / "data" / "indexed_documents": "indexed_documents",
    REPO_ROOT / "data" / "queries":           "queries",
    REPO_ROOT / "data" / "test":              "test",
    REPO_ROOT / "environment" / "configs":    "configs",
}


def _cos_creds():
    creds_path = Path(os.environ.get("COS_CREDS_JSON", str(Path.home() / ".cos_creds.json")))
    # Prefer env (exported by config.sh); fall back to the creds JSON directly.
    endpoint = os.environ.get("COS_ENDPOINT")
    bucket = os.environ.get("COS_BUCKET")
    key = os.environ.get("COS_HMAC_ACCESS_KEY_ID")
    secret = os.environ.get("COS_HMAC_SECRET_ACCESS_KEY")
    region = os.environ.get("REGION", "us-east")
    if not (key and secret) and creds_path.exists():
        c = json.load(open(creds_path))["credentials"]
        key = key or c["cos_hmac_keys"]["access_key_id"]
        secret = secret or c["cos_hmac_keys"]["secret_access_key"]
    endpoint = endpoint or f"https://s3.{region}.cloud-object-storage.appdomain.cloud"
    bucket = bucket or f"vakra-benchmark-data-{region}"
    if not (key and secret):
        sys.exit("Could not resolve COS HMAC keys. Source deploy/ce/config.sh or set "
                 "COS_HMAC_ACCESS_KEY_ID / COS_HMAC_SECRET_ACCESS_KEY.")
    return endpoint, bucket, key, secret


def _client(endpoint, key, secret):
    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        sys.exit("boto3 is required for the upload. Run:  pip install boto3")
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=key, aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


def _ensure_bucket(client, bucket: str, region: str) -> None:
    """Create the bucket if it doesn't exist (idempotent). Safe if step 1 also creates it."""
    from botocore.exceptions import ClientError
    try:
        client.head_bucket(Bucket=bucket)
        print(f"  bucket '{bucket}' exists — reusing.")
        return
    except ClientError:
        pass
    try:
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": f"{region}-standard"},
        )
        print(f"  created bucket '{bucket}' in {region}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  bucket '{bucket}' already owned — reusing.")
        else:
            raise


def _existing_sizes(client, bucket, prefix):
    sizes = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def upload_dir(client, bucket, local_dir: Path, prefix: str, dry_run: bool):
    if not local_dir.exists():
        print(f"  [skip] {local_dir}  (not present locally)")
        return 0, 0
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                         multipart_chunksize=64 * 1024 * 1024,
                         max_concurrency=8, use_threads=True)
    remote_sizes = {} if dry_run else _existing_sizes(client, bucket, prefix)
    n_up = n_skip = 0
    total_bytes = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}"
        size = path.stat().st_size
        if remote_sizes.get(key) == size:
            n_skip += 1
            continue
        total_bytes += size
        if dry_run:
            print(f"  [would upload] {key}  ({size/1e6:.1f} MB)")
        else:
            client.upload_file(str(path), bucket, key, Config=cfg)
            n_up += 1
            if n_up % 25 == 0:
                print(f"  [{prefix}] uploaded {n_up} files, {total_bytes/1e9:.2f} GB ...", flush=True)
    print(f"  [{prefix}] {'would upload' if dry_run else 'uploaded'} {n_up if not dry_run else '?'} "
          f"file(s), skipped {n_skip} unchanged  ({total_bytes/1e9:.2f} GB)")
    return n_up, n_skip


def _download_prefix(prefix: str) -> None:
    """Download just one dataset subtree into data/<prefix> (skips train/)."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=[f"{prefix}/**"],
        local_dir=str(REPO_ROOT / "data"),
    )


def _free_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(str(path))
    return free / 1e9


def stream_push(client, bucket, only, dry_run):
    """Low-disk path: for each subtree, download -> upload -> delete, so peak local
    usage is one subtree (~25 GB for databases) instead of the full ~32 GB."""
    for prefix in HF_PREFIXES:
        if only and prefix not in only:
            continue
        local_dir = REPO_ROOT / "data" / prefix
        print(f"\n--- stream '{prefix}': download -> upload -> delete "
              f"(free: {_free_gb(REPO_ROOT):.1f} GB) ---")
        if not dry_run:
            try:
                _download_prefix(prefix)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {prefix}: download failed/empty: {e}")
        if not local_dir.exists():
            print(f"  [skip] {prefix}: not present in dataset")
            continue
        upload_dir(client, bucket, local_dir, prefix, dry_run)
        if not dry_run:
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"  [freed] {local_dir}  (free now: {_free_gb(REPO_ROOT):.1f} GB)")

    # configs live in the repo (tiny) — upload directly, no download/delete.
    if not only or "configs" in only:
        cfg = REPO_ROOT / "environment" / "configs"
        print(f"\n--- environment/configs -> {bucket}/configs/ ---")
        upload_dir(client, bucket, cfg, "configs", dry_run)


def main():
    ap = argparse.ArgumentParser(description="Push VAKRA data to COS")
    ap.add_argument("--stream", action="store_true",
                    help="Low-disk: download+upload+delete one subtree at a time "
                         "(peak ~25 GB). Recommended if free disk < ~40 GB.")
    ap.add_argument("--skip-download", action="store_true", help="Don't download; upload existing data/ only")
    ap.add_argument("--only", default="", help="Comma list of prefixes (databases,indexed_documents,queries,test,configs)")
    ap.add_argument("--dry-run", action="store_true", help="List what would upload, don't upload")
    args = ap.parse_args()

    only = {p.strip() for p in args.only.split(",") if p.strip()}
    endpoint, bucket, key, secret = _cos_creds()
    region = os.environ.get("REGION", "us-east")
    client = None if args.dry_run else _client(endpoint, key, secret)
    if client is not None:
        _ensure_bucket(client, bucket, region)

    if args.stream:
        print(f"=== STREAM push to COS bucket '{bucket}' ({endpoint}) ===")
        print(f"    free disk: {_free_gb(REPO_ROOT):.1f} GB  (peak use ~25 GB during 'databases')")
        stream_push(client, bucket, only, args.dry_run)
    else:
        if not args.skip_download:
            free = _free_gb(REPO_ROOT)
            if free < 35:
                sys.exit(
                    f"Only {free:.1f} GB free; a full download needs ~35 GB.\n"
                    "Use the low-disk path instead:  python deploy/ce/0_push_data_to_cos.py --stream\n"
                    "(or free up disk, or --skip-download if data/ is already present)."
                )
            print("=== Downloading data (reuses `make download`) ===")
            rc = subprocess.run([sys.executable, str(REPO_ROOT / "benchmark_setup.py"), "--download-data"],
                                cwd=str(REPO_ROOT)).returncode
            if rc != 0:
                sys.exit(f"data download failed (rc={rc})")

        print(f"\n=== Uploading to COS bucket '{bucket}' ({endpoint}) ===")
        for local_dir, prefix in UPLOAD_MAP.items():
            if only and prefix not in only:
                continue
            print(f"\n--- {local_dir.relative_to(REPO_ROOT)} -> {bucket}/{prefix}/ ---")
            upload_dir(client, bucket, local_dir, prefix, args.dry_run)

    print("\nDone." + ("  (dry run — nothing uploaded)" if args.dry_run else ""))
    if not args.dry_run:
        print("Next: cd deploy/ce && VAKRA_CE_ADMIN=1 ./1_create_bucket_and_pds.sh")


if __name__ == "__main__":
    main()
