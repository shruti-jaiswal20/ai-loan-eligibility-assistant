"""
Pure presentation-layer helpers: turn the backend's financial_metrics /
risk_level / recommendation into the bullet lists and summary chips the UI
shows. No underwriting decisions are made here — those all happen in
agents/*.py — this module only *explains* them.
"""


def positive_negative_factors(metrics: dict, risk_level: str) -> tuple[list[str], list[str]]:
    positives, negatives = [], []

    credit = metrics.get("credit_score", 0)
    dti = metrics.get("dti_ratio", 0)
    tenure = metrics.get("years_at_current_job", 0)
    default = metrics.get("prior_default", False)
    ltv = metrics.get("ltv_ratio")

    if credit >= 750:
        positives.append(f"Excellent credit score ({credit})")
    elif credit >= 700:
        positives.append(f"Good credit score ({credit})")
    elif credit >= 650:
        negatives.append(f"Fair credit score ({credit}) — below preferred 700+ band")
    else:
        negatives.append(f"Poor credit score ({credit}) — below policy minimum")

    if dti <= 0.35:
        positives.append(f"Healthy debt-to-income ratio ({dti:.1%})")
    elif dti <= 0.50:
        negatives.append(f"DTI ratio ({dti:.1%}) is above the 35% low-risk limit (policy maximum 50%)")
    else:
        negatives.append(f"DTI ratio ({dti:.1%}) exceeds the policy maximum of 50%")

    if tenure >= 1:
        positives.append(f"Stable employment history ({tenure} years at current job)")
    else:
        negatives.append(f"Limited employment tenure ({tenure} years)")

    if default:
        negatives.append("Prior loan default on record")
    else:
        positives.append("No prior default history")

    if ltv is not None:
        if ltv <= 0.8:
            positives.append(f"Loan-to-value ratio ({ltv:.0%}) within policy limit")
        else:
            negatives.append(f"Loan-to-value ratio ({ltv:.0%}) exceeds the 80% policy limit")
    else:
        negatives.append("No collateral declared for this application")

    return positives, negatives


def assessment_summary_tags(metrics: dict, risk_level: str) -> dict:
    """Returns the four summary chips shown on the 'Completed' screen."""
    credit = metrics.get("credit_score", 0)
    dti = metrics.get("dti_ratio", 0)

    financial_health = "Good" if credit >= 700 and dti <= 0.4 else (
        "Fair" if credit >= 650 else "Weak")
    credit_risk = "Low" if credit >= 750 else ("Medium" if credit >= 650 else "High")
    policy_compliance = "Pass" if risk_level != "HIGH" else "Flagged"

    return {
        "Financial Health": financial_health,
        "Credit Risk": credit_risk,
        "Policy Compliance": policy_compliance,
        "Overall Risk": risk_level.title(),
    }
