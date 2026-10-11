"""
Agentic workflow for the AI Loan Eligibility Assessment Assistant.

Matches RFP Theme 11's required workflow type: "Sequential + Router workflow",
AND the cross-cutting requirement in RFP Section 7 for "Sequential and
parallel execution" — the Financial Analysis Agent and the Policy/RAG
Retrieval step run as genuinely parallel LangGraph nodes (see the fan-out
below), joining before the Risk Agent classifies.

Full checklist coverage (RFP Section 7):
  - Multi-agent workflow (Financial -> Risk -> Recommendation)            [x]
  - Sequential AND parallel execution (financial_analysis ‖ policy_retrieval) [x]
  - Router-based decision making (expedited vs standard; changes Risk    [x]
    Agent behaviour, not just a label)
  - RAG pipeline (policy_retrieval node, grounded in policy PDFs)         [x]
  - Human approval step (graph interrupts before finalizing)             [x]
  - Tool-using agents (EMI/DTI/LTV calculators as agent tools)           [x]
  - Memory/context handling (LangGraph checkpointer, keyed per applicant) [x]
  - Error handling and retries (input validation + per-node try/except +  [x]
    LLM-level retry with backoff)

Graph shape:

                document_extraction   (reads the uploaded PDF, if any; cross-checks it)
                       |
                 validate_input --(invalid)--> END
                       |(valid)
                     router
                    /        \\
        financial_analysis   policy_retrieval      <- run in PARALLEL
                    \\        /
                  risk_assessment   (join: waits for both branches)
                       |
                  recommendation
                       |
             [INTERRUPT: human_approval_gate]
                       |
                    finalize
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from workflow.state import LoanAssessmentState
from agents.document_agent import process_document
from agents.financial_agent import run_financial_analysis
from agents.risk_agent import run_policy_retrieval, run_risk_classification
from agents.recommendation_agent import run_recommendation
from utils.schemas import validate_applicant
from utils.audit_log import log_event
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("ai_loan_assistant.workflow")

EXPEDITED_LOAN_CEILING = 800_000
EXPEDITED_CREDIT_FLOOR = 720


def _safe_step(node_name):
    """Decorator giving every node uniform error handling: a failure is
    captured into state['errors'] (additive) and logged, rather than
    crashing the whole graph. Satisfies the RFP's error-handling
    requirement without every agent needing its own try/except."""
    def decorator(fn):
        def wrapper(state: LoanAssessmentState) -> dict:
            started = time.perf_counter()
            try:
                result = dict(fn(state))
                elapsed_ms = (time.perf_counter() - started) * 1000
                result["trace"] = [f"[OK] {node_name} ({elapsed_ms:.1f} ms)"]
                return result
            except Exception as e:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.exception("Node %s failed", node_name)
                return {"errors": [f"{node_name}: {e}"],
                        "trace": [f"[ERROR] {node_name} ({elapsed_ms:.1f} ms): {e}"]}
        return wrapper
    return decorator


# ---------------------------------------------------------------- nodes ----

@_safe_step("document_extraction")
def document_extraction_node(state: LoanAssessmentState) -> dict:
    """Document Intake Agent. Skips cleanly when no PDF was uploaded; when one
    was, it really parses it and cross-checks it against the applicant record."""
    doc = state.get("document")
    if not doc or not doc.get("path"):
        return {"document_status": "none", "document_fields": {}, "document_summary": "",
                "document_discrepancies": [], "document_missing": [], "document_reason": ""}
    result = process_document(doc["path"], state["applicant"], doc.get("filename"),
                              summary=doc.get("summary"))
    log_event("document_processed", state["applicant"].get("applicant_id", "?"), {
        "filename": result["filename"], "status": result["status"],
        "fields_found": len(result["fields"]), "discrepancies": result["discrepancies"],
        "storage_uri": doc.get("storage_uri"),
    })
    return {
        "document_status": result["status"], "document_fields": result["fields"],
        "document_summary": result["summary"], "document_discrepancies": result["discrepancies"],
        "document_missing": result["missing_fields"], "document_reason": result["reason"],
    }


@_safe_step("validate_input")
def validate_input_node(state: LoanAssessmentState) -> dict:
    is_valid, errors = validate_applicant(state["applicant"])
    if not is_valid:
        logger.warning("Validation failed for %s: %s", state["applicant"].get("applicant_id"), errors)
    return {"is_valid": is_valid, "validation_errors": errors}


@_safe_step("router")
def router_node(state: LoanAssessmentState) -> dict:
    applicant = state["applicant"]
    amount = applicant.get("requested_loan_amount", 0)
    credit_score = applicant.get("credit_score", 0)
    default = applicant.get("bank_default_history", False)

    thresholds = state.get("policy_thresholds") or {}
    expedited_ceiling = thresholds.get("expedited_loan_ceiling", EXPEDITED_LOAN_CEILING)
    expedited_credit_floor = thresholds.get("expedited_credit_floor", EXPEDITED_CREDIT_FLOOR)
    if amount <= expedited_ceiling and credit_score >= expedited_credit_floor and not default:
        return {
            "route": "expedited",
            "route_reason": (
                f"Loan amount INR {amount:,} <= INR {expedited_ceiling:,} and "
                f"credit score {credit_score} >= {expedited_credit_floor} with no default "
                "history -> routed to the expedited fast-track path (Risk Agent will skip "
                "the LLM call and use the deterministic rule engine)."
            ),
        }
    return {
        "route": "standard",
        "route_reason": (
            f"Loan amount INR {amount:,} or credit score {credit_score} falls outside "
            "fast-track thresholds, or a prior default exists -> routed to the standard "
            "path (Risk Agent will use full LLM+RAG-grounded classification)."
        ),
    }


@_safe_step("financial_analysis")
def financial_analysis_node(state: LoanAssessmentState) -> dict:
    return run_financial_analysis(state["applicant"])


@_safe_step("policy_retrieval")
def policy_retrieval_node(state: LoanAssessmentState) -> dict:
    """Parallel branch B — runs concurrently with financial_analysis_node.
    Depends only on the router's output, not on Financial Agent's metrics."""
    return run_policy_retrieval(state["applicant"])


@_safe_step("risk_assessment")
def risk_assessment_node(state: LoanAssessmentState) -> dict:
    """Join node: LangGraph schedules this only after BOTH parallel
    predecessors (financial_analysis, policy_retrieval) have completed."""
    return run_risk_classification(
        state["applicant"], state["financial_metrics"], state["route"],
        state.get("policy_context", ""), state.get("policy_chunks", []),
        state.get("policy_thresholds"),
    )


@_safe_step("recommendation")
def recommendation_node(state: LoanAssessmentState) -> dict:
    result = run_recommendation(
        state["applicant"], state["financial_metrics"],
        state["risk_level"], state["risk_explanation"],
    )
    # Policy (risk_assessment_guidelines.pdf, Section 5): missing or inconsistent data
    # must stop automated approval and go to a human analyst.
    reasons = []
    if state.get("document_discrepancies"):
        reasons.append("Uploaded document does not match the application data: "
                       + "; ".join(state["document_discrepancies"][:3]))
    if state.get("document_status") in ("failed", "partial"):
        reasons.append("Uploaded document could not be fully read"
                       + (f" ({state.get('document_reason')})" if state.get("document_reason") else "")
                       + " — manual verification required.")
    if reasons:
        if result["recommendation"] == "APPROVE":
            result["recommendation"] = "REFER_FOR_MANUAL_REVIEW"
        result["recommendation_explanation"] += "  ESCALATED: " + " ".join(reasons)
    result["escalation_reasons"] = reasons
    return result


@_safe_step("human_approval_gate")
def human_approval_gate_node(state: LoanAssessmentState) -> dict:
    """Deliberate interrupt point. The graph is invoked with
    interrupt_before=['human_approval_gate']; execution pauses here until
    the Streamlit UI calls resume_after_human_decision(). No auto-disbursal
    ever happens without an explicit human decision."""
    return {}


@_safe_step("finalize")
def finalize_node(state: LoanAssessmentState) -> dict:
    return {}


# ---------------------------------------------------------------- graph ----

def route_after_validation(state: LoanAssessmentState) -> str:
    return "router" if state.get("is_valid") else "invalid"


def fan_out_after_router(state: LoanAssessmentState) -> list[str]:
    """Always fan out to BOTH parallel branches. The router's decision
    (state['route']) still matters downstream — see risk_assessment_node —
    it just doesn't change *which* nodes run next, only *how* the join
    node behaves once both branches complete."""
    return ["financial_analysis", "policy_retrieval"]


_CHECKPOINTER = None


def build_graph():
    global _CHECKPOINTER
    graph = StateGraph(LoanAssessmentState)

    graph.add_node("document_extraction", document_extraction_node)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("router", router_node)
    graph.add_node("financial_analysis", financial_analysis_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("human_approval_gate", human_approval_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("document_extraction")
    graph.add_edge("document_extraction", "validate_input")

    graph.add_conditional_edges(
        "validate_input", route_after_validation, {"router": "router", "invalid": END}
    )

    # Router-based decision making: fans out to two nodes every time, but
    # route_reason/route recorded here changes risk_assessment's behaviour.
    graph.add_conditional_edges("router", fan_out_after_router)

    # Parallel execution: both branches read only from "router"'s output,
    # so LangGraph schedules them in the same super-step.
    graph.add_edge("financial_analysis", "risk_assessment")
    graph.add_edge("policy_retrieval", "risk_assessment")

    graph.add_edge("risk_assessment", "recommendation")
    graph.add_edge("recommendation", "human_approval_gate")
    graph.add_edge("human_approval_gate", "finalize")
    graph.add_edge("finalize", END)

    # Durable checkpoints survive Streamlit reruns, container restarts, and
    # process restarts. Mount /app/data as a persistent Docker volume in production.
    checkpoint_path = Path(os.getenv(
        "LANGGRAPH_CHECKPOINT_DB",
        str(Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"),
    ))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    _CHECKPOINTER = checkpointer
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval_gate"],
    )
    return compiled


_APP = None
_CASES: dict[str, float] = {}   # applicant_id -> time of last assessment (process-wide registry)


def list_case_ids() -> list[str]:
    return [k for k, _ in sorted(_CASES.items(), key=lambda kv: kv[1], reverse=True)]


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_until_human_review(applicant: dict, document: dict | None = None,
                           policy_thresholds: dict | None = None,
                           progress_callback=None) -> LoanAssessmentState:
    """Runs the graph from the start up to (but not including) the human
    approval gate. `document` = {"path","filename","storage_uri"} for an uploaded PDF.
    Returns the accumulated state for the UI to display."""
    app = get_app()
    thread_id = applicant["applicant_id"]
    config = {"configurable": {"thread_id": thread_id}}
    # Starting a new assessment for the same applicant must not inherit a
    # previous run's trace or human decision. The saved assessment history
    # remains available independently in assessment_history.json.
    if _CHECKPOINTER is not None:
        try:
            _CHECKPOINTER.delete_thread(thread_id)
        except Exception:
            logger.debug("No prior checkpoint to clear for %s", thread_id, exc_info=True)
    initial_state: LoanAssessmentState = {
        "applicant": applicant, "document": document,
        "policy_thresholds": policy_thresholds or {},
        "errors": [], "trace": [],
    }

    log_event("assessment_started", applicant["applicant_id"],
              {"applicant": applicant, "has_document": bool(document)})
    # Stream actual node completions so the UI can report genuine progress
    # rather than animating a fake timeline.
    for update in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_update in update.items():
            if progress_callback:
                try:
                    progress_callback(node_name, node_update or {})
                except Exception:  # UI feedback must never break the workflow
                    logger.debug("Progress callback failed for %s", node_name, exc_info=True)
    snapshot = app.get_state(config)
    result = snapshot.values if snapshot else {}
    import time as _t
    _CASES[applicant["applicant_id"]] = _t.time()

    if not result.get("is_valid", True):
        log_event("assessment_rejected_invalid_input", applicant["applicant_id"],
                   {"validation_errors": result.get("validation_errors", [])})
    else:
        log_event("assessment_recommendation", applicant["applicant_id"], {
            "route": result.get("route"), "risk_level": result.get("risk_level"),
            "recommendation": result.get("recommendation"),
            "escalation_reasons": result.get("escalation_reasons", []),
        })
    return result


def resume_after_human_decision(applicant_id: str, decision: str,
                                 reviewer: str, comment: str = "") -> LoanAssessmentState:
    """Resumes a paused graph after the human reviewer makes a decision."""
    app = get_app()
    config = {"configurable": {"thread_id": applicant_id}}
    app.update_state(config, {
        "human_decision": decision,
        "human_reviewer": reviewer,
        "human_comment": comment,
    })
    result = app.invoke(None, config=config)
    log_event("human_decision", applicant_id, {
        "decision": decision, "reviewer": reviewer, "comment": comment,
    })
    return result


def get_state(applicant_id: str) -> LoanAssessmentState:
    app = get_app()
    config = {"configurable": {"thread_id": applicant_id}}
    snapshot = app.get_state(config)
    return snapshot.values if snapshot else {}
