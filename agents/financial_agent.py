"""
Financial Analysis Agent
------------------------
Tool-using agent: computes hard financial ratios from the applicant record
(a deterministic "tool"/function, not left to the LLM to hallucinate), then
asks the LLM to produce a plain-language financial summary grounded in
those computed numbers.
"""
from agents.llm_client import generate


def compute_emi(principal: float, tenure_years: float, annual_rate: float = 0.10) -> float:
    """Standard reducing-balance EMI formula - the agent's 'tool'."""
    r = annual_rate / 12
    n = tenure_years * 12
    if n <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def compute_financial_metrics(applicant: dict) -> dict:
    income = applicant.get("monthly_income", 0)
    existing_emi = applicant.get("existing_monthly_emi", 0)
    loan_amount = applicant.get("requested_loan_amount", 0)
    tenure = applicant.get("tenure_years", 5)

    proposed_emi = round(compute_emi(loan_amount, tenure), 2)
    dti_ratio = round((existing_emi + proposed_emi) / income, 3) if income else 1.0
    ltv_ratio = None
    collateral = applicant.get("collateral_value", 0)
    if collateral and collateral > 0:
        ltv_ratio = round(loan_amount / collateral, 3)

    income_multiple = round(loan_amount / (income * 12), 2) if income else None

    return {
        "proposed_emi": proposed_emi,
        "dti_ratio": dti_ratio,
        "ltv_ratio": ltv_ratio,
        "annual_income_multiple": income_multiple,
        "credit_score": applicant.get("credit_score"),
        "employment_type": applicant.get("employment_type"),
        "years_at_current_job": applicant.get("years_at_current_job"),
        "prior_default": applicant.get("bank_default_history", False),
    }


def _mock_summary(applicant: dict, metrics: dict) -> str:
    return (
        f"{applicant.get('name', 'The applicant')} has a monthly income of "
        f"INR {applicant.get('monthly_income', 0):,}, a credit score of "
        f"{metrics['credit_score']}, and {metrics['years_at_current_job']} years at their "
        f"current job ({metrics['employment_type']}). The proposed EMI for the requested "
        f"loan of INR {applicant.get('requested_loan_amount', 0):,} over "
        f"{applicant.get('tenure_years')} years is approximately INR {metrics['proposed_emi']:,.0f}, "
        f"resulting in a debt-to-income (DTI) ratio of {metrics['dti_ratio']:.2f}. "
        + (f"Loan-to-value ratio is {metrics['ltv_ratio']:.2f}. "
           if metrics.get("ltv_ratio") is not None else "No collateral was declared. ")
        + (f"Note: applicant has a prior default on record." if metrics["prior_default"] else
           "No prior default history on record.")
    )


def run_financial_analysis(applicant: dict) -> dict:
    metrics = compute_financial_metrics(applicant)

    system_prompt = (
        "You are a Financial Analysis Agent for a bank's loan assessment system. "
        "Given computed financial metrics, write a concise (3-4 sentence), factual "
        "summary for a human loan officer. Do not invent numbers not provided."
    )
    user_prompt = (
        f"Applicant: {applicant}\n\nComputed metrics: {metrics}\n\n"
        "Write the financial summary."
    )

    summary = generate(
        system_prompt, user_prompt,
        mock_fn=lambda: _mock_summary(applicant, metrics),
    )
    return {"financial_metrics": metrics, "financial_summary": summary}
