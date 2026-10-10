"""Boundary tests for the deterministic risk rules (they back the expedited path, mock mode and the dashboard)."""
import pytest
from agents.risk_agent import _rule_based_risk


def m(dti=0.2, credit=750, default=False, tenure=5):
    return {"dti_ratio": dti, "credit_score": credit, "prior_default": default, "years_at_current_job": tenure}


@pytest.mark.parametrize("kw, expected", [
    (dict(dti=0.35), "LOW"), (dict(dti=0.351), "MEDIUM"), (dict(dti=0.355), "MEDIUM"), (dict(dti=0.50), "MEDIUM"),
    (dict(dti=0.501), "HIGH"),
    (dict(credit=700), "LOW"), (dict(credit=699), "MEDIUM"), (dict(credit=650), "MEDIUM"), (dict(credit=649), "HIGH"),
    (dict(tenure=1), "LOW"), (dict(tenure=0.99), "MEDIUM"), (dict(tenure=0.3), "MEDIUM"), (dict(tenure=0), "MEDIUM"),
    (dict(default=True), "HIGH"), (dict(default=True, dti=0.1, credit=850), "HIGH"),
])
def test_risk_band_boundaries(kw, expected):
    assert _rule_based_risk(m(**kw))[0] == expected
