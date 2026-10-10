"""S3 integration tested against moto (an in-process AWS simulator): real boto3 calls,
no network, no credentials. Proves the wiring works — it does not prove a real AWS account is set up."""
import json
from pathlib import Path

import pytest

pytest.importorskip("moto")
import boto3
from moto import mock_aws

from utils import s3_storage, audit_log
from workflow.graph import run_until_human_review

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "test-loan-bucket"


@pytest.fixture(autouse=True)
def _reset_breaker(monkeypatch):
    monkeypatch.setattr(s3_storage, "_down_until", 0.0)


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setattr(s3_storage, "BUCKET_NAME", BUCKET)
    monkeypatch.setattr(s3_storage, "AWS_ENDPOINT_URL", "")
    monkeypatch.setattr(s3_storage, "AWS_REGION", "us-east-1")
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


def _keys(client):
    return [o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


def test_status_reports_bucket_state(aws):
    assert s3_storage.status()["reachable"] is False          # bucket doesn't exist yet
    aws.create_bucket(Bucket=BUCKET)
    assert s3_storage.status()["reachable"] is True


def test_document_is_stored_in_s3_and_bucket_auto_created(aws):
    pdf = ROOT / "data/applications/APP-1001_form.pdf"
    uri = s3_storage.store_document(pdf, "APP-1001", "my form.pdf")
    assert uri == f"s3://{BUCKET}/applications/uploads/APP-1001/my_form.pdf"
    body = aws.get_object(Bucket=BUCKET, Key="applications/uploads/APP-1001/my_form.pdf")["Body"].read()
    assert body == pdf.read_bytes()


def test_audit_events_are_mirrored_one_object_each(aws):
    audit_log.log_event("unit_test_event", "APP-X", {"detail": 1})
    keys = _keys(aws)
    assert len(keys) == 1 and keys[0].startswith("audit_log/") and keys[0].endswith("_unit_test_event_APP-X.json")
    rec = json.loads(aws.get_object(Bucket=BUCKET, Key=keys[0])["Body"].read())
    assert rec["event_type"] == "unit_test_event" and rec["detail"] == 1


def test_full_assessment_writes_audit_trail_to_s3(aws):
    rec = json.loads((ROOT / "data/applications/APP-1001.json").read_text())
    run_until_human_review(rec | {"applicant_id": "APP-S3-FLOW"},
                           {"path": str(ROOT / "data/applications/APP-1001_form.pdf"), "filename": "f.pdf"})
    events = " ".join(_keys(aws))
    assert "assessment_started" in events and "document_processed" in events
    assert "assessment_recommendation" in events


def test_sync_directory_uploads_policy_pdfs(aws):
    res = s3_storage.sync_directory(ROOT / "data/policy_docs", "policy_docs")
    assert res["uploaded"] == 2 and res["failed"] == []
    assert set(_keys(aws)) == {"policy_docs/loan_eligibility_policy.pdf",
                               "policy_docs/risk_assessment_guidelines.pdf"}


def test_unconfigured_s3_is_a_safe_noop(monkeypatch):
    monkeypatch.setattr(s3_storage, "BUCKET_NAME", "")
    assert s3_storage.status()["configured"] is False
    assert s3_storage.sync_directory(ROOT / "data/policy_docs", "x")["uploaded"] == 0
    uri = s3_storage.store_document(ROOT / "data/applications/APP-1001_form.pdf", "APP-1001", "f.pdf")
    assert uri.startswith("file://")
    audit_log.log_event("noop_event", "APP-X", {})            # must not raise


def test_s3_outage_never_breaks_the_workflow(monkeypatch):
    """Configured but unreachable (bad endpoint): the assessment must still complete."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x"); monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setattr(s3_storage, "BUCKET_NAME", BUCKET)
    monkeypatch.setattr(s3_storage, "AWS_ENDPOINT_URL", "http://127.0.0.1:9")   # nothing listens here
    rec = json.loads((ROOT / "data/applications/APP-1001.json").read_text())
    state = run_until_human_review(rec | {"applicant_id": "APP-S3-DOWN"})
    assert state["recommendation"] and not state.get("errors")
    uri = s3_storage.store_document(ROOT / "data/applications/APP-1001_form.pdf", "APP-1001", "f.pdf")
    assert uri.startswith("file://")                            # fell back to local, did not raise


def test_outage_costs_one_timeout_not_one_per_event(monkeypatch):
    import time
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x"); monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setattr(s3_storage, "BUCKET_NAME", BUCKET)
    monkeypatch.setattr(s3_storage, "AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    audit_log.log_event("first", "A", {})                      # pays the failure, opens the breaker
    t = time.time()
    for i in range(10):
        audit_log.log_event("later", "A", {"i": i})            # skipped instantly
    assert time.time() - t < 1.0
    assert s3_storage._circuit_open()
