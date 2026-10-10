"""
Tools the AI Assistant can call. They are the ONLY way the assistant touches the system:

  assess_applicant   run the full multi-agent assessment for an existing applicant ID
  explain_case       explain an already-assessed case (risk factors, policy evidence, route)
  search_policy      answer from the policy PDFs through the RAG retriever
  portfolio_stats    portfolio numbers (rule-based pass over the applicant file)
  list_cases         cases assessed in this process, optionally filtered by review status

There is deliberately NO approve / reject / disburse tool: final decisions are made by a human
on the Human Review screen. Even if a language model asks for such a tool, execute_tool()
returns an error for any name not in TOOLS.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rag.retriever import retrieve_policy_context
from utils.audit_log import log_event
from utils.insights import positive_negative_factors
from utils.portfolio import load_applicants_df, score_applicants, summarize
from workflow.graph import get_state, list_case_ids, run_until_human_review

logger = logging.getLogger("ai_loan_assistant.chat_tools")

APPLICATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "applications"
APP_ID_RE = re.compile(r"^APP-[A-Za-z0-9]{1,12}$")      # blocks path tricks like ../../etc


def _valid_id(applicant_id) -> str | None:
    return applicant_id if isinstance(applicant_id, str) and APP_ID_RE.match(applicant_id) else None


def _summary(state: dict) -> dict:
    m = state.get("financial_metrics", {})
    a = state.get("applicant", {})
    return {
        "applicant_id": a.get("applicant_id"), "name": a.get("name"),
        "route": state.get("route"), "risk_level": state.get("risk_level"),
        "ai_recommendation": state.get("recommendation"),
        "requires_human_approval": state.get("requires_human_approval", True),
        "human_decision": state.get("human_decision"),
        "dti_percent": round(m.get("dti_ratio", 0) * 100, 1), "credit_score": m.get("credit_score"),
        "monthly_income": a.get("monthly_income"), "requested_amount": a.get("requested_loan_amount"),
        "proposed_emi": m.get("proposed_emi"),
        "escalation_reasons": state.get("escalation_reasons", []),
        "policy_sources": state.get("policy_sources", []),
        "document_status": state.get("document_status", "none"),
    }


def tool_assess_applicant(applicant_id: str) -> dict:
    if not _valid_id(applicant_id):
        return {"error": "Please give an applicant ID like APP-1005."}
    existing = get_state(applicant_id)
    if existing and existing.get("recommendation"):
        return {**_summary(existing), "note": "already assessed in this session; showing the existing result",
                "next_step": "Open Human Review to record the final decision." if not existing.get("human_decision") else "Decision already recorded."}
    path = APPLICATIONS_DIR / f"{applicant_id}.json"
    if not path.exists():
        return {"error": f"No applicant {applicant_id} exists in the applicant file."}
    state = run_until_human_review(json.loads(path.read_text()))
    if not state.get("is_valid", True):
        return {"error": "Validation failed: " + "; ".join(state.get("validation_errors", []))}
    return {**_summary(state), "next_step": "Open Human Review to approve, reject or request more info."}


def tool_explain_case(applicant_id: str) -> dict:
    if not _valid_id(applicant_id):
        return {"error": "Please give an applicant ID like APP-1005."}
    state = get_state(applicant_id)
    if not state or not state.get("recommendation"):
        return {"error": f"{applicant_id} has not been assessed yet. Ask me to assess it first."}
    pos, neg = positive_negative_factors(state["financial_metrics"], state["risk_level"])
    return {
        **_summary(state), "route_reason": state.get("route_reason"),
        "positive_factors": pos, "risk_factors": neg,
        "risk_explanation": state.get("risk_explanation"),
        "recommendation_explanation": state.get("recommendation_explanation"),
        "financial_summary": state.get("financial_summary"),
        "policy_evidence": [{"source": c["source"], "text": c["text"][:240]}
                            for c in state.get("policy_chunks", [])[:3]],
        "document_summary": state.get("document_summary", ""),
        "document_discrepancies": state.get("document_discrepancies", []),
    }


def tool_search_policy(question: str) -> dict:
    q = (question or "").strip()[:300]
    if not q:
        return {"error": "Ask a policy question, e.g. 'What DTI ratio needs manual review?'"}
    chunks = retrieve_policy_context(q, k=3)
    return {"question": q, "passages": [{"source": c["source"], "text": c["text"][:520]} for c in chunks]}


def tool_portfolio_stats() -> dict:
    stats = summarize(score_applicants(load_applicants_df()))
    ids = list_case_ids()
    decided = sum(1 for i in ids if (get_state(i) or {}).get("human_decision"))
    return {**stats, "basis": "rule-based pass over the applicant file, not agent assessments",
            "assessed_this_session": len(ids), "awaiting_human_decision": len(ids) - decided,
            "human_decisions_recorded": decided}


def tool_list_cases(status: str = "all") -> dict:
    status = status if status in {"all", "pending_review", "decided"} else "all"
    out = []
    for i in list_case_ids():
        st = get_state(i) or {}
        if not st.get("recommendation"):
            continue
        decided = bool(st.get("human_decision"))
        if (status == "pending_review" and decided) or (status == "decided" and not decided):
            continue
        out.append({"applicant_id": i, "name": st["applicant"].get("name"), "risk_level": st.get("risk_level"),
                    "ai_recommendation": st.get("recommendation"), "human_decision": st.get("human_decision")})
    return {"status_filter": status, "count": len(out), "cases": out}


TOOLS = {
    "assess_applicant": tool_assess_applicant, "explain_case": tool_explain_case,
    "search_policy": tool_search_policy, "portfolio_stats": tool_portfolio_stats,
    "list_cases": tool_list_cases,
}


def _fn(name, description, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


_ID = {"applicant_id": {"type": "string", "description": "Applicant ID such as APP-1005"}}
TOOL_SCHEMAS = [
    _fn("assess_applicant", "Run the full multi-agent loan assessment for an existing applicant. It only "
        "produces a recommendation; a human must make the final decision.", _ID, ["applicant_id"]),
    _fn("explain_case", "Explain an already-assessed case: risk factors, policy evidence, routing, escalation.",
        _ID, ["applicant_id"]),
    _fn("search_policy", "Answer a question from the bank's loan policy documents (RAG).",
        {"question": {"type": "string"}}, ["question"]),
    _fn("portfolio_stats", "Portfolio-level counts: totals, risk bands, recommendations."),
    _fn("list_cases", "List cases assessed in this session.",
        {"status": {"type": "string", "enum": ["all", "pending_review", "decided"]}}),
]


def execute_tool(name: str, args: dict) -> dict:
    """Run a tool safely. Unknown tools and any exception become an error result."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. I can only assess, explain, search policy, show stats and list cases."}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be an object."}
    try:
        result = fn(**{k: v for k, v in args.items() if k in fn.__code__.co_varnames[:fn.__code__.co_argcount]})
    except Exception as e:  # noqa: BLE001
        logger.exception("Tool %s failed", name)
        return {"error": f"The {name} tool failed: {e}"}
    log_event("assistant_tool_call", str(args.get("applicant_id", "-")),
              {"tool": name, "args": args, "ok": "error" not in result})
    return result
