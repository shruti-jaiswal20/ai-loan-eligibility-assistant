"""Document Intake Agent: real PDF parsing, cross-checking, and workflow integration.
Ground truth = the JSON record the form PDF was generated from."""
import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from agents.document_agent import process_document, parse_fields, cross_check
from workflow.graph import run_until_human_review

APPS = Path(__file__).resolve().parent.parent / "data" / "applications"
FIELDS = ["name", "age", "city", "employment_type", "years_at_current_job", "monthly_income",
          "existing_monthly_emi", "existing_loans_count", "credit_score", "bank_default_history",
          "requested_loan_amount", "loan_purpose", "tenure_years", "collateral_value"]


def _record(app_id):
    return json.loads((APPS / f"{app_id}.json").read_text())


@pytest.mark.parametrize("app_id", ["APP-1000", "APP-1007", "APP-1023", "APP-1039"])
def test_every_field_is_extracted_correctly(app_id):
    rec = _record(app_id)
    res = process_document(APPS / f"{app_id}_form.pdf", rec)
    assert res["status"] == "ok" and res["missing_fields"] == []
    assert res["document_applicant_id"] == app_id
    for f in FIELDS:
        exp, got = rec[f], res["fields"][f]
        if isinstance(exp, float):
            assert got == pytest.approx(exp, rel=1e-3), f
        else:
            assert got == exp, f
    assert res["discrepancies"] == []          # form and record agree
    assert rec["name"] in res["summary"]


def test_mismatch_with_record_is_reported():
    rec = _record("APP-1003")
    rec["monthly_income"] = rec["monthly_income"] * 2
    rec["credit_score"] = 800
    res = process_document(APPS / "APP-1003_form.pdf", rec)
    keys = " ".join(res["discrepancies"])
    assert "monthly_income" in keys and "credit_score" in keys


def test_small_rounding_is_not_a_discrepancy():
    assert cross_check({"monthly_income": 100000.0}, {"monthly_income": 100500.0}) == []   # 0.5%
    assert cross_check({"monthly_income": 100000.0}, {"monthly_income": 120000.0}) != []   # ~17%


def test_pdf_without_text_layer_fails_honestly(tmp_path):
    p = tmp_path / "scan.pdf"
    c = canvas.Canvas(str(p)); c.rect(50, 50, 200, 200); c.save()       # drawing only, no text
    res = process_document(p, {})
    assert res["status"] == "failed" and "OCR" in res["reason"]


def test_corrupt_file_does_not_raise(tmp_path):
    p = tmp_path / "bad.pdf"; p.write_bytes(b"not a pdf at all")
    res = process_document(p, {})
    assert res["status"] == "failed" and res["reason"]


def test_unrelated_text_pdf_is_not_parsed(tmp_path):
    p = tmp_path / "letter.pdf"
    c = canvas.Canvas(str(p)); c.drawString(72, 700, "Dear Sir, please find my CV attached."); c.save()
    res = process_document(p, {})
    assert res["status"] == "failed" and res["fields"] == {}


def test_parse_handles_inline_label_value_format():
    f, _ = parse_fields("Credit Score: 712\nMonthly Income (INR): 85,000\nPrior Default History: Yes")
    assert f == {"credit_score": 712, "monthly_income": 85000.0, "bank_default_history": True}


# ---- workflow integration -------------------------------------------------

def _doc(app_id):
    return {"path": str(APPS / f"{app_id}_form.pdf"), "filename": f"{app_id}_form.pdf"}


def test_workflow_uses_matching_document():
    rec = _record("APP-1001")                      # LOW-risk, small loan
    state = run_until_human_review(rec, _doc("APP-1001"))
    assert state["document_status"] == "ok" and state["document_discrepancies"] == []
    assert "[OK] document_extraction" in state["trace"]
    assert state["recommendation"] == "APPROVE"    # nothing blocks approval


def test_discrepant_document_blocks_auto_approval():
    rec = _record("APP-1001")
    rec["applicant_id"] = "APP-T-MISMATCH"
    rec["monthly_income"] = rec["monthly_income"] + 80000       # officer entered something else
    state = run_until_human_review(rec, _doc("APP-1001"))
    assert state["document_discrepancies"]
    assert state["recommendation"] == "REFER_FOR_MANUAL_REVIEW"
    assert state["escalation_reasons"] and "ESCALATED" in state["recommendation_explanation"]
    assert state["requires_human_approval"] is True


def test_no_document_is_skipped_cleanly():
    state = run_until_human_review(_record("APP-1001") | {"applicant_id": "APP-T-NODOC"})
    assert state["document_status"] == "none"
    assert state.get("escalation_reasons") == []
