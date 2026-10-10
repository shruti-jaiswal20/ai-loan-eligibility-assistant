"""
AWS S3 integration (RFP Section 10: file storage on S3).

What is stored where, once S3 is configured (S3_BUCKET_NAME + credentials or an
AWS_ENDPOINT_URL for LocalStack):
  applications/uploads/<applicant_id>/<file>.pdf   documents uploaded in New Assessment
  audit_log/<timestamp>_<event>_<applicant>.json   one object per audit event (see utils/audit_log.py)
  policy_docs/*.pdf, applications/forms/*.pdf      pushed by sync_directory() / Settings > Sync

If S3 is NOT configured every function degrades to a harmless local no-op
(files stay on disk), so development and tests never need AWS.
Network calls use short timeouts and a single attempt: an unreachable bucket
must never freeze the UI. Callers in the request path catch S3Error.

Usage from a shell:  python -m utils.s3_storage sync
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ai_loan_assistant.s3")

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "")  # set for LocalStack, blank for real AWS


class S3Error(Exception):
    pass


# Circuit breaker: after a failed S3 call, skip S3 for a while so an outage costs ONE timeout,
# not one per audit event (which would stall every assessment by several seconds).
CIRCUIT_OPEN_SECONDS = 60
_down_until = 0.0


def _circuit_open() -> bool:
    return time.time() < _down_until


def _trip_circuit() -> None:
    global _down_until
    _down_until = time.time() + CIRCUIT_OPEN_SECONDS


def _s3_configured() -> bool:
    return bool(BUCKET_NAME and (os.getenv("AWS_ACCESS_KEY_ID") or AWS_ENDPOINT_URL
                                 or os.getenv("AWS_PROFILE") or os.getenv("AWS_ROLE_ARN")
                                 or os.getenv("S3_USE_INSTANCE_ROLE") == "1"))


def _client():
    import boto3
    from botocore.config import Config
    kwargs = {"region_name": AWS_REGION,
              "config": Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1})}
    if AWS_ENDPOINT_URL:
        kwargs["endpoint_url"] = AWS_ENDPOINT_URL
        kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID", "test")
        kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
    return boto3.client("s3", **kwargs)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or "file"


def ensure_bucket() -> None:
    """Create the bucket if missing. Only works with permissions to create buckets
    (LocalStack / dev). On real AWS the bucket is provisioned in the deployment guide."""
    from botocore.exceptions import BotoCoreError, ClientError
    client = _client()
    try:
        client.head_bucket(Bucket=BUCKET_NAME)
    except (ClientError, BotoCoreError):
        try:
            client.create_bucket(Bucket=BUCKET_NAME,
                                 **({} if AWS_REGION == "us-east-1" else
                                    {"CreateBucketConfiguration": {"LocationConstraint": AWS_REGION}}))
            logger.info("Created S3 bucket %s", BUCKET_NAME)
        except (ClientError, BotoCoreError) as e:
            raise S3Error(f"Bucket {BUCKET_NAME!r} is missing and could not be created: {e}") from e


def put_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """Upload bytes. Returns s3://bucket/key, or a local marker when S3 is not configured."""
    if not _s3_configured():
        return f"local-only://{key}"
    if _circuit_open():
        raise S3Error("S3 skipped: it failed recently (retrying in under a minute)")
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        ensure_bucket()
        _client().put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    except S3Error:
        _trip_circuit()
        raise
    except (ClientError, BotoCoreError) as e:
        logger.error("S3 put failed for %s: %s", key, e)
        _trip_circuit()
        raise S3Error(str(e)) from e
    return f"s3://{BUCKET_NAME}/{key}"


def put_json(obj: dict, key: str) -> str:
    return put_bytes(json.dumps(obj, default=str).encode("utf-8"), key, "application/json")


def upload_file(local_path: str, s3_key: str) -> str:
    ctype = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    return put_bytes(Path(local_path).read_bytes(), s3_key, ctype)


def store_document(local_path: str | Path, applicant_id: str, filename: str) -> str:
    """Store an uploaded application document. Returns its S3 URI, or the local path
    (file://...) if S3 is not configured or the upload failed — never raises."""
    key = f"applications/uploads/{_safe_name(applicant_id)}/{_safe_name(filename)}"
    try:
        uri = upload_file(str(local_path), key)
        if uri.startswith("local-only://"):
            return f"file://{Path(local_path).resolve()}"
        return uri
    except S3Error as e:
        logger.warning("Document stored locally only (S3 failed): %s", e)
        return f"file://{Path(local_path).resolve()}"


def sync_directory(local_dir: str | Path, prefix: str, pattern: str = "*.pdf") -> dict:
    """Upload every matching file. Returns {'uploaded': n, 'failed': [..], 'configured': bool}."""
    out = {"uploaded": 0, "failed": [], "configured": _s3_configured()}
    if not out["configured"]:
        return out
    for f in sorted(Path(local_dir).glob(pattern)):
        try:
            upload_file(str(f), f"{prefix.rstrip('/')}/{f.name}")
            out["uploaded"] += 1
        except S3Error as e:
            out["failed"].append(f"{f.name}: {e}")
    return out


def list_objects(prefix: str = "") -> list[str]:
    if not _s3_configured():
        return []
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        resp = _client().list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    except (ClientError, BotoCoreError) as e:
        raise S3Error(str(e)) from e
    return [o["Key"] for o in resp.get("Contents", [])]


def status() -> dict:
    """Real connectivity check (head_bucket) for the Settings page."""
    info = {"configured": _s3_configured(), "bucket": BUCKET_NAME or None,
            "endpoint": AWS_ENDPOINT_URL or "AWS", "reachable": None, "error": None}
    if not info["configured"]:
        return info
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        _client().head_bucket(Bucket=BUCKET_NAME)
        info["reachable"] = True
        global _down_until
        _down_until = 0.0
    except (ClientError, BotoCoreError) as e:
        info["reachable"] = False
        info["error"] = str(e)[:200]
    return info


def sync_all() -> dict:
    return {
        "policy_docs": sync_directory(BASE_DIR / "data" / "policy_docs", "policy_docs"),
        "application_forms": sync_directory(BASE_DIR / "data" / "applications", "applications/forms", "*_form.pdf"),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        print(json.dumps(sync_all(), indent=2))
    else:
        print(json.dumps(status(), indent=2))
