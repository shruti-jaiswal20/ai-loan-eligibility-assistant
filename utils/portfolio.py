"""Portfolio-level numbers shared by the dashboard and the AI Assistant.

These are a fast RULE-BASED pass over the synthetic applicant file (no LLM, no
agent run) — they describe the portfolio, not the outcome of agent assessments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.financial_agent import compute_financial_metrics
from agents.recommendation_agent import _decide
from agents.risk_agent import _rule_based_risk

APPLICATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "applications"


def load_applicants_df() -> pd.DataFrame:
    p = APPLICATIONS_DIR / "applicants.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def score_applicants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, r in df.iterrows():
        a = r.to_dict()
        risk, _ = _rule_based_risk(compute_financial_metrics(a))
        decision, _ = _decide(a, risk)
        rows.append({**a, "risk_level": risk, "recommendation": decision})
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict:
    n = len(scored)
    if n == 0:
        return {"total": 0}
    return {
        "total": n,
        "approve_recommended": int((scored["recommendation"] == "APPROVE").sum()),
        "manual_review": int((scored["recommendation"] == "REFER_FOR_MANUAL_REVIEW").sum()),
        "reject_recommended": int((scored["recommendation"] == "REJECT").sum()),
        "by_risk": {k: int(v) for k, v in scored["risk_level"].value_counts().items()},
    }
