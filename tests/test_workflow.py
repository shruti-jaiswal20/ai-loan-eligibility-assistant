"""
Smoke tests for the agentic workflow. Run with:  pytest tests/ -v

These run fully offline (MOCK LLM mode, local TF-IDF embeddings) so they
work in any CI environment with no API keys / network access required.
"""
import json
from pathlib import Path

import pytest

from workflow.graph import run_until_human_review, resume_after_human_decision

BASE_DIR = Path(__file__).resolve().parent.parent
APPLICATIONS_DIR = BASE_DIR / "data" / "applications"


def _load_first_applicant():
    files = sorted(APPLICATIONS_DIR.glob("APP-*.json"))
    assert files, "Run data_generation/generate_synthetic_data.py before testing."
    with open(files[0]) as f:
        return json.load(f)


def test_workflow_produces_recommendation():
    applicant = _load_first_applicant()
    state = run_until_human_review(applicant)

    assert state["route"] in {"expedited", "standard"}
    assert "financial_summary" in state and state["financial_summary"]
    assert state["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert state["recommendation"] in {"APPROVE", "REJECT", "REFER_FOR_MANUAL_REVIEW"}
    assert state["requires_human_approval"] is True
    assert state.get("human_decision") is None
    assert not state.get("errors")


def test_high_risk_defaulter_is_rejected_or_referred():
    applicant = {
        "applicant_id": "APP-TEST-HIGHRISK",
        "name": "Test Defaulter",
        "monthly_income": 40000,
        "existing_monthly_emi": 25000,
        "requested_loan_amount": 2000000,
        "tenure_years": 10,
        "credit_score": 580,
        "bank_default_history": True,
        "collateral_value": 0,
        "years_at_current_job": 0.4,
        "employment_type": "Salaried",
    }
    state = run_until_human_review(applicant)
    assert state["risk_level"] == "HIGH"
    assert state["recommendation"] in {"REJECT", "REFER_FOR_MANUAL_REVIEW"}


def test_human_approval_gate_and_resume():
    applicant = _load_first_applicant()
    state = run_until_human_review(applicant)
    assert state.get("human_decision") is None  # paused at the gate

    final_state = resume_after_human_decision(
        applicant["applicant_id"], "APPROVED", "test_reviewer", "looks fine"
    )
    assert final_state["human_decision"] == "APPROVED"
    assert final_state["human_reviewer"] == "test_reviewer"
    assert "[OK] finalize" in final_state["trace"]


def test_low_amount_high_credit_routes_expedited():
    applicant = {
        "applicant_id": "APP-TEST-EXPEDITED",
        "name": "Test FastTrack",
        "monthly_income": 150000,
        "existing_monthly_emi": 5000,
        "requested_loan_amount": 400000,
        "tenure_years": 5,
        "credit_score": 780,
        "bank_default_history": False,
        "collateral_value": 500000,
        "years_at_current_job": 5,
        "employment_type": "Salaried",
    }
    state = run_until_human_review(applicant)
    assert state["route"] == "expedited"


def test_invalid_input_is_rejected_before_agents_run():
    """Negative income should be caught by Pydantic validation and never
    reach the agents (no financial_metrics/risk_level should be produced)."""
    applicant = {
        "applicant_id": "APP-TEST-INVALID",
        "name": "Bad Data",
        "monthly_income": -5000,  # invalid: must be > 0
        "existing_monthly_emi": 0,
        "requested_loan_amount": 500000,
        "tenure_years": 5,
        "credit_score": 720,
        "bank_default_history": False,
        "collateral_value": 0,
        "years_at_current_job": 2,
        "employment_type": "Salaried",
    }
    state = run_until_human_review(applicant)
    assert state["is_valid"] is False
    assert any("monthly_income" in e for e in state["validation_errors"])
    assert "financial_metrics" not in state
    assert "risk_level" not in state


def test_financial_and_policy_retrieval_run_as_parallel_branches():
    """Both parallel branches (Financial Analysis, Policy Retrieval) should
    each contribute their own [OK] trace entry, proving the fan-out/join
    executed both nodes rather than skipping one."""
    applicant = _load_first_applicant()
    state = run_until_human_review(applicant)
    trace_text = " | ".join(state["trace"])
    assert "[OK] financial_analysis" in trace_text
    assert "[OK] policy_retrieval" in trace_text
    assert "[OK] risk_assessment" in trace_text
    assert state.get("policy_chunks")  # RAG branch actually retrieved something
    assert state.get("financial_metrics")  # Financial branch actually computed something


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
