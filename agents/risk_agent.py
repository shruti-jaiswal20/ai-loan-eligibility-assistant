"""
Risk Agent
----------
Split into two independent phases so they can run as parallel LangGraph
nodes (see workflow/graph.py):

  1. run_policy_retrieval(applicant)   — pure RAG lookup, needs only the
     raw applicant fields, so it can run concurrently with the Financial
     Analysis Agent instead of waiting for it.
  2. run_risk_classification(...)      — the join step: combines the
     Financial Analysis Agent's computed metrics with the retrieved policy
     context to produce the LOW/MEDIUM/HIGH classification.

The router's decision (`route`) is used here too: on the "expedited"
fast-track path we skip the LLM call entirely and use the deterministic
rule-based classifier, since a small, high-credit, no-default application
doesn't need generative reasoning to accept/flag it — this keeps the
Router's decision consequential (fewer LLM calls, lower latency/cost) even
though both routes still run the same parallel retrieval step.
"""
from agents.llm_client import generate
from rag.retriever import retrieve_policy_context, format_context


def _rule_based_risk(metrics: dict) -> tuple[str, str]:
    """Deterministic fallback classifier mirroring the policy PDFs -
    used as the MOCK-mode output, the expedited fast-track path, and a
    safety cross-check for the LLM path."""
    dti = metrics["dti_ratio"]
    credit = metrics["credit_score"]
    default = metrics["prior_default"]
    tenure = metrics["years_at_current_job"]

    if default or dti > 0.50 or credit < 650:
        return "HIGH", (
            f"Classified HIGH risk because "
            + ("the applicant has a prior default on record. " if default else "")
            + (f"DTI ratio is {dti:.2f} (> 0.50 threshold). " if dti > 0.50 else "")
            + (f"Credit score is {credit} (< 650 threshold). " if credit < 650 else "")
        )
    # Policy: LOW needs DTI <= 0.35 AND tenure >= 1y; anything between the LOW and HIGH limits is MEDIUM
    # (the policy text says '0.36-0.50' for convenience, but values such as 0.355 must not fall in a gap).
    if 650 <= credit <= 699 or 0.35 < dti <= 0.50 or tenure < 1:
        return "MEDIUM", (
            f"Classified MEDIUM risk based on credit score {credit}, DTI ratio "
            f"{dti:.2f}, and {tenure} years at current job — falls into the "
            "policy's 'Fair' / medium-risk band requiring manual underwriter review."
        )
    return "LOW", (
        f"Classified LOW risk: credit score {credit} (>=700), DTI ratio {dti:.2f} "
        f"(<=0.35), no prior default, and stable employment tenure of {tenure} years."
    )


def run_policy_retrieval(applicant: dict) -> dict:
    """Phase 1 (parallel branch B). Builds its query from raw applicant
    fields only — deliberately NOT from Financial Agent's computed metrics —
    so this node has no dependency on the Financial Analysis node and both
    can be scheduled by LangGraph in the same parallel super-step."""
    query = (
        f"loan risk classification credit score {applicant.get('credit_score')} "
        f"default history {applicant.get('bank_default_history')} "
        f"loan amount {applicant.get('requested_loan_amount')} "
        f"collateral {applicant.get('collateral_value', 0)}"
    )
    chunks = retrieve_policy_context(query, k=4)
    context = format_context(chunks)
    sources = sorted(set(c["source"] for c in chunks))
    return {"policy_context": context, "policy_sources": sources, "policy_chunks": chunks}


def run_risk_classification(applicant: dict, metrics: dict, route: str,
                             policy_context: str, policy_chunks: list) -> dict:
    """Phase 2 (join step). Runs after BOTH the Financial Analysis node and
    the Policy Retrieval node have completed."""
    rule_risk, rule_explanation = _rule_based_risk(metrics)

    if route == "expedited":
        # Fast-track: skip the LLM call entirely. This is the router's
        # decision actually changing behaviour, not just a cosmetic label.
        return {"risk_level": rule_risk,
                "risk_explanation": f"[Expedited fast-track — rule-based check] {rule_risk}: {rule_explanation}"}

    system_prompt = (
        "You are a Risk Agent for a bank's loan assessment system. Classify the "
        "application as LOW, MEDIUM, or HIGH risk using ONLY the provided policy "
        "context and financial metrics. Start your answer with the risk level in "
        "capital letters, then a short explanation citing the specific policy rule."
    )
    user_prompt = (
        f"Policy context:\n{policy_context}\n\n"
        f"Financial metrics: {metrics}\n\n"
        f"Classify the risk level and explain why."
    )

    def mock_fn():
        return f"{rule_risk}: {rule_explanation}"

    llm_output = generate(system_prompt, user_prompt, mock_fn=mock_fn)

    # Parse risk level defensively; fall back to the deterministic rule engine
    # if the LLM output doesn't clearly state one (robustness / error handling).
    risk_level = rule_risk
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in llm_output.upper()[:20]:
            risk_level = level
            break

    return {"risk_level": risk_level, "risk_explanation": llm_output}
