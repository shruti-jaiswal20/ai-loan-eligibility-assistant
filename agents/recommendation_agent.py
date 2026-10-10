"""
Recommendation Agent
---------------------
Synthesizes the Financial Analysis Agent and Risk Agent outputs into a
final AI recommendation. Per bank policy (see loan_eligibility_policy.pdf,
Section 7), the AI may only *recommend* - a human must always give final
sign-off, which is enforced by the `requires_human_approval` flag that the
workflow graph checks before finalizing any case.
"""
from agents.llm_client import generate

APPROVAL_AUTHORITY_THRESHOLD = 1_000_000  # INR - matches policy doc Section 7


def _decide(applicant: dict, risk_level: str) -> tuple[str, bool]:
    amount = applicant.get("requested_loan_amount", 0)

    if risk_level == "HIGH":
        return "REJECT", True
    if risk_level == "MEDIUM":
        return "REFER_FOR_MANUAL_REVIEW", True
    # LOW risk
    if amount <= APPROVAL_AUTHORITY_THRESHOLD:
        return "APPROVE", True  # AI can recommend, human must still sign off
    return "REFER_FOR_MANUAL_REVIEW", True


def _mock_explanation(applicant, risk_level, metrics, decision) -> str:
    return (
        f"Recommendation: {decision}. Based on a {risk_level} risk classification, "
        f"a DTI ratio of {metrics['dti_ratio']:.2f}, and a credit score of "
        f"{metrics['credit_score']}, this application "
        + ("meets the bank's low-risk fast-track criteria and is recommended for "
           "approval, subject to mandatory human sign-off."
           if decision == "APPROVE" else
           "does not meet the low-risk fast-track criteria and is referred to a "
           "human underwriter for manual review."
           if decision == "REFER_FOR_MANUAL_REVIEW" else
           "carries risk factors (high DTI, low credit score, or prior default) "
           "that fall outside acceptable policy thresholds and is recommended for "
           "rejection, pending senior underwriter confirmation.")
    )


def run_recommendation(applicant: dict, metrics: dict, risk_level: str,
                        risk_explanation: str) -> dict:
    decision, requires_human = _decide(applicant, risk_level)

    system_prompt = (
        "You are the Recommendation Agent in a bank's loan assessment system. "
        "Given the risk classification and financial metrics, write a 2-3 sentence "
        "recommendation explanation for the human reviewer. Never claim the loan is "
        "final/disbursed - your output is always a RECOMMENDATION pending human approval."
    )
    user_prompt = (
        f"Decision: {decision}\nRisk level: {risk_level}\nRisk explanation: {risk_explanation}\n"
        f"Financial metrics: {metrics}\n\nWrite the recommendation explanation."
    )

    explanation = generate(
        system_prompt, user_prompt,
        mock_fn=lambda: _mock_explanation(applicant, risk_level, metrics, decision),
    )

    return {
        "recommendation": decision,
        "recommendation_explanation": explanation,
        "requires_human_approval": requires_human,
    }
