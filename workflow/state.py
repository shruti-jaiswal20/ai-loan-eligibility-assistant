"""Shared state object threaded through every node of the LangGraph workflow.

`errors` and `trace` use an additive reducer (operator.add) rather than plain
overwrite, because two nodes now run in parallel (Financial Analysis and
Policy/RAG Retrieval, see workflow/graph.py). Without a reducer, LangGraph's
default "last write wins" merge would silently drop one branch's trace/error
entries whenever both branches write in the same super-step.
"""
from __future__ import annotations

import operator
from typing import TypedDict, Optional, List, Dict, Any, Annotated


class LoanAssessmentState(TypedDict, total=False):
    # --- input ---
    applicant: Dict[str, Any]
    policy_thresholds: Dict[str, Any]
    document: Optional[Dict[str, Any]]      # {"path", "filename", "storage_uri"} if a PDF was uploaded

    # --- document intake agent output ---
    document_status: str                    # "none" | "ok" | "partial" | "failed"
    document_fields: Dict[str, Any]
    document_summary: str
    document_discrepancies: List[str]       # mismatches vs. the applicant record
    document_missing: List[str]
    document_reason: str
    escalation_reasons: List[str]           # why automated approval was blocked

    # --- router decision ---
    route: str                      # "expedited" | "standard"
    route_reason: str

    # --- financial analysis agent output (parallel branch A) ---
    financial_metrics: Dict[str, Any]
    financial_summary: str

    # --- policy/RAG retrieval output (parallel branch B) ---
    policy_context: str
    policy_sources: List[str]
    policy_chunks: List[Dict[str, str]]

    # --- risk agent output (joins branch A + B) ---
    risk_level: str                 # "LOW" | "MEDIUM" | "HIGH"
    risk_explanation: str

    # --- recommendation agent output ---
    recommendation: str             # "APPROVE" | "REJECT" | "REFER_FOR_MANUAL_REVIEW"
    recommendation_explanation: str
    requires_human_approval: bool

    # --- human-in-the-loop ---
    human_decision: Optional[str]   # "APPROVED" | "REJECTED" | "REQUEST_MORE_INFO" | None
    human_reviewer: Optional[str]
    human_comment: Optional[str]

    # --- validation (runs before any agent) ---
    is_valid: bool
    validation_errors: List[str]

    # --- process/meta (additive: safe to write from parallel branches) ---
    errors: Annotated[List[str], operator.add]
    trace: Annotated[List[str], operator.add]   # human-readable log of each step
