"""Page-level render functions for the AI Loan Eligibility Assessment
Assistant dashboard UI. Each function renders one screen; app.py routes
to these based on st.session_state.page / wizard_step."""
from __future__ import annotations

import hashlib
import json
import time
import random
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from agents.financial_agent import compute_financial_metrics
from agents.risk_agent import _rule_based_risk
from agents.recommendation_agent import _decide
from agents.llm_client import USE_MOCK, MODEL_NAME
from workflow.graph import (run_until_human_review, resume_after_human_decision, get_state,
                             EXPEDITED_LOAN_CEILING, EXPEDITED_CREDIT_FLOOR)
from agents.chat_agent import chat_turn, get_history, new_session_id, assistant_mode
from agents.document_agent import process_document, find_discrepancies, FIELD_MAP
from utils.s3_storage import (_s3_configured, BUCKET_NAME, store_document,
                              status as s3_status, sync_all)
from utils.audit_log import read_recent_events
from utils.assessment_store import save_assessment_history
from utils.portfolio import score_applicants

from ui.components import (inr, badge, badge_html, topbar, stat_card, card_heading,
                            step_row, factor_list, RISK_KIND, RISK_LABEL, DECISION_KIND)
from ui.insights import positive_negative_factors, assessment_summary_tags
from ui.workflow_diagram import build_workflow_dot
from ui.pdf_report import build_report_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
APPLICATIONS_DIR = BASE_DIR / "data" / "applications"
POLICY_DIR = BASE_DIR / "data" / "policy_docs"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
MAX_UPLOAD_MB = 10

LOAN_TYPES = ["Home Loan", "Personal Loan", "Vehicle Loan", "Business Loan", "Education Loan"]
TENURES = [3, 5, 7, 10, 15, 20]


# ------------------------------------------------------------------ data --

@st.cache_data
def load_applicants_df() -> pd.DataFrame:
    csv_path = APPLICATIONS_DIR / "applicants.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_data
def bulk_rule_based_status(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based pass over every synthetic applicant (no LLM) — see utils/portfolio.py."""
    return score_applicants(df)


def _status_label(decision: str) -> str:
    return {"APPROVE": "Approved", "REJECT": "High Risk",
            "REFER_FOR_MANUAL_REVIEW": "Manual Review"}.get(decision, decision)


# --------------------------------------------------------------- sidebar --

NAV_ITEMS = [
    ("dashboard", "📊", "Dashboard"),
    ("new_assessment", "➕", "New Assessment"),
    ("assistant", "💬", "AI Assistant"),
    ("applications", "📄", "Applications"),
    ("assessments", "🧪", "Assessments"),
    ("agent_execution", "🤖", "Agent Execution"),
    ("documents", "📁", "Documents"),
    ("reports", "📈", "Reports"),
    ("settings", "⚙️", "Settings"),
]


def _quick_ask():
    q = (st.session_state.get("quick_q") or "").strip()
    if q:
        st.session_state.pending_chat_question = q
        st.session_state.page = "assistant"
        st.session_state.quick_q = ""


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo">LA</div>
                <div>
                    <div class="name">AI LOAN<br/>ASSESSMENT</div>
                    <div class="sub">INTELLIGENT UNDERWRITING</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for key, icon, label in NAV_ITEMS:
            is_active = st.session_state.page == key
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         type="primary" if is_active else "secondary",
                         width="stretch"):
                st.session_state.page = key
                if key == "new_assessment" and not st.session_state.get("keep_wizard"):
                    st.session_state.wizard_step = "form"
                st.session_state.keep_wizard = False
                st.rerun()

        with st.popover("💬 Quick ask", width="stretch"):
            st.text_input("Ask the assistant", key="quick_q", placeholder="e.g. What DTI needs manual review?")
            st.button("Ask", key="quick_ask_btn", on_click=_quick_ask, width="stretch")

        st.toggle("🌙 Dark mode", key="dark_mode")
        if os.getenv("APP_LOGIN_PASSWORD_SHA256", "").strip() and st.session_state.get("authenticated"):
            if st.button("Sign out", key="sign_out", width="stretch"):
                st.session_state.authenticated = False
                st.rerun()
        st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
        llm_state = "Mock (offline)" if USE_MOCK else f"Live · {MODEL_NAME}"
        s3_state = "Connected" if _s3_configured() else "Local mode"
        st.markdown(
            f"""
            <div class="sidebar-status-row"><span class="dot" style="background:{'#f59e0b' if USE_MOCK else '#16a34a'}"></span> LLM Engine: {llm_state}</div>
            <div class="sidebar-status-row"><span class="dot" style="background:{'#16a34a' if _s3_configured() else '#64748b'}"></span> S3 Storage: {s3_state}</div>
            <div class="sidebar-status-row"><span class="dot"></span> Vector DB: Online</div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------- dashboard --

def _dashboard_greeting() -> str:
    """Return a time-appropriate greeting using the configured app timezone.

    Set APP_TIMEZONE if the deployment should use a timezone other than India.
    """
    timezone_name = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
    try:
        now = datetime.now(ZoneInfo(timezone_name))
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now()

    if 5 <= now.hour < 12:
        greeting = "Good morning"
    elif 12 <= now.hour < 17:
        greeting = "Good afternoon"
    elif 17 <= now.hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Good night"
    reviewer = (st.session_state.get("reviewer_name") or "Analyst").strip()
    return f"{greeting}, {reviewer} 👋"


def page_dashboard():
    topbar("Dashboard", _dashboard_greeting(),
           "Monitor loan applications and AI workflow activity.",
           st.session_state.reviewer_name)

    df = load_applicants_df()
    if df.empty:
        st.warning("No synthetic data found. Run `python data_generation/generate_synthetic_data.py`.")
        return
    with st.spinner("Loading portfolio metrics…"):
        scored = bulk_rule_based_status(df)

    total = len(scored)
    approved = (scored["recommendation"] == "APPROVE").sum()
    manual = (scored["recommendation"] == "REFER_FOR_MANUAL_REVIEW").sum()
    high_risk = (scored["risk_level"] == "HIGH").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("📋", "Total Applications", str(total), "blue")
    with c2: stat_card("✅", "Approved", str(approved), "green", f"{approved/total:.1%} of total")
    with c3: stat_card("🕵️", "Manual Review", str(manual), "amber", f"{manual/total:.1%} of total")
    with c4: stat_card("⚠️", "High Risk", str(high_risk), "red", f"{high_risk/total:.1%} of total")

    st.write("")
    left, right = st.columns([1, 2])
    with left:
        with st.container(border=True):
            card_heading("🟢", "Risk Distribution")
            counts = scored["risk_level"].value_counts()
            colors_map = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626"}
            labels = [f"{k.title()} Risk" for k in counts.index]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=counts.values, hole=0.6,
                marker=dict(colors=[colors_map.get(k, "#94a3b8") for k in counts.index]),
                textinfo="percent",
            )])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260,
                               showlegend=True, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
            st.download_button(
                "⬇️ Export portfolio CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="loan_portfolio.csv",
                mime="text/csv",
                key="dashboard_csv_export",
            )

    # Trend from durable assessment history (only actual completed workflow runs).
    history = st.session_state.get("assessment_history", {})
    if history:
        trend_rows = []
        for app_id, record in history.items():
            state = record.get("state", {})
            ts = record.get("timestamp")
            if ts and state.get("risk_level"):
                trend_rows.append({"Date": pd.to_datetime(ts).date(),
                                   "Risk": state.get("risk_level", "N/A"),
                                   "Application": app_id})
        if trend_rows:
            trend = pd.DataFrame(trend_rows).groupby(["Date", "Risk"]).size().unstack(fill_value=0)
            st.markdown("### Assessment trend")
            st.line_chart(trend)

    with right:
        with st.container(border=True):
            hcol1, hcol2 = st.columns([3, 1])
            with hcol1:
                card_heading("📑", "Recent Applications")
            with hcol2:
                if st.button("➕ New Assessment", key="dash_new_assessment", width="stretch"):
                    st.session_state.page = "new_assessment"
                    st.session_state.wizard_step = "form"
                    st.rerun()

            recent = scored.head(8)
            for _, r in recent.iterrows():
                cols = st.columns([1.3, 1.8, 1.5, 1.1, 1.5, 1.1])
                cols[0].write(f"**{r['applicant_id']}**")
                cols[1].write(r["name"])
                cols[2].write(inr(r["requested_loan_amount"]))
                cols[3].markdown(badge(r["risk_level"], RISK_KIND[r["risk_level"]]), unsafe_allow_html=True)
                cols[4].markdown(badge(_status_label(r["recommendation"]),
                                        DECISION_KIND.get(r["recommendation"], "gray")),
                                  unsafe_allow_html=True)
                if cols[5].button("Assess", key=f"assess_{r['applicant_id']}"):
                    _jump_to_assessment(r["applicant_id"])
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.caption(f"Showing {len(recent)} of {total} applications — see the Applications tab for the full list.")


def _jump_to_assessment(applicant_id: str):
    import json
    with open(APPLICATIONS_DIR / f"{applicant_id}.json") as f:
        applicant = json.load(f)
    st.session_state.current_applicant = applicant
    st.session_state.current_document = None
    st.session_state.page = "new_assessment"
    st.session_state.wizard_step = "progress"
    st.session_state.keep_wizard = True
    st.session_state.progress_started = False
    st.rerun()


# ---------------------------------------------------------- new assessment --

def page_new_assessment():
    step = st.session_state.wizard_step
    if step == "form":
        _wizard_form()
    elif step == "progress":
        _wizard_progress()
    elif step == "results":
        _wizard_results()
    elif step == "review":
        _wizard_review()
    elif step == "completed":
        _wizard_completed()


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ingest_upload(uploaded) -> dict:
    """Save the uploaded PDF once (content-addressed) and run the Document Intake Agent on it."""
    data = uploaded.getvalue()
    empty = {"sha": None, "filename": uploaded.name, "path": None, "status": "failed", "fields": {},
             "missing_fields": [], "summary": "", "discrepancies": [], "document_applicant_id": None}
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return {**empty, "reason": f"File is larger than {MAX_UPLOAD_MB} MB."}
    sha = hashlib.sha256(data).hexdigest()[:12]
    cache = st.session_state.setdefault("doc_cache", {})
    if sha not in cache:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / f"{sha}.pdf"
        if not path.exists():
            path.write_bytes(data)
        res = process_document(path, None, uploaded.name)
        res["sha"] = sha
        cache[sha] = res
    return cache[sha]


def _render_document_card(doc: dict, current: dict):
    """Shows what the Document Intake Agent read and how it compares with the form."""
    kind = {"ok": "green", "partial": "amber", "failed": "red"}[doc["status"]]
    label = {"ok": "READ OK", "partial": "PARTIAL", "failed": "NOT READABLE"}[doc["status"]]
    with st.container(border=True):
        card_heading("📎", "Document Intake Agent")
        st.markdown(f"{badge(label, kind)} &nbsp; **{doc['filename']}**", unsafe_allow_html=True)
        if doc["status"] == "failed":
            st.error(doc.get("reason") or "The document could not be read.")
            st.caption("You can still continue — the case will be flagged for manual verification.")
            return
        st.caption(f"{len(doc['fields'])} of {len(FIELD_MAP)} form fields read from the PDF text layer.")
        st.write(doc["summary"])
        if doc["missing_fields"]:
            st.warning("Required fields not found: " + ", ".join(doc["missing_fields"]))
        issues = find_discrepancies(doc["fields"], doc.get("document_applicant_id"), current)
        if issues:
            st.warning("**Document and form disagree** — approval will be blocked and the case sent to manual review:")
            for i in issues[:6]:
                st.markdown(f"- {i}")
            if len(issues) > 6:
                st.caption(f"…and {len(issues) - 6} more differences.")
        else:
            st.success("Document and form values agree.")


def _wizard_form():
    topbar("Dashboard / New Assessment", "New Loan Assessment", "",
           st.session_state.reviewer_name)

    df = load_applicants_df()
    manual = "— Manual Entry —"
    ids = [manual] + (df["applicant_id"].tolist() if not df.empty else [])

    col1, col2 = st.columns([1, 1.4])

    doc = None
    with col1:
        with st.container(border=True):
            card_heading("⬆️", "Upload Loan Application")
            uploaded = st.file_uploader("Drag & drop PDF here, or click to browse",
                                         type=["pdf"], key="upload_pdf")
            if uploaded:
                doc = _ingest_upload(uploaded)
            else:
                st.caption("Supported: PDF · Maximum size: 10MB. The Document Intake Agent reads the "
                           "form, summarizes it and checks it against the data below. "
                           "Try any file from the Documents page (APP-####_form.pdf).")
        doc_card = st.container()      # filled in after the form values are known

    with col2:
        with st.container(border=True):
            card_heading("🧾", "Customer Information")
            selected_id = st.selectbox("Customer ID", ids, key="form_customer_id")

            doc_ok = bool(doc and doc["status"] in ("ok", "partial"))
            if selected_id != manual:
                prefill = df[df["applicant_id"] == selected_id].iloc[0].to_dict()   # system of record
            else:
                prefill = dict(doc["fields"]) if doc_ok else {}                      # prefill from the document
            sfx = f"{selected_id}_{doc['sha'] if doc_ok else 'nodoc'}"

            name = st.text_input("Customer Name", value=prefill.get("name", ""), key=f"form_name_{sfx}")
            loan_type = st.selectbox("Loan Type", LOAN_TYPES, index=0, key=f"form_loan_type_{sfx}")
            requested_amount = st.number_input(
                "Requested Amount (INR)", min_value=50000, max_value=20000000, step=50000,
                value=_clamp(int(prefill.get("requested_loan_amount", 500000)), 50000, 20000000),
                key=f"form_amount_{sfx}")
            tenure_default = int(prefill["tenure_years"]) if prefill.get("tenure_years") is not None else None
            tenure = st.selectbox("Tenure (Years)", TENURES,
                                   index=TENURES.index(tenure_default) if tenure_default in TENURES else 2,
                                   key=f"form_tenure_{sfx}")
            purpose = st.text_input("Purpose", value=prefill.get("loan_purpose", "Home Renovation"),
                                     key=f"form_purpose_{sfx}")

            emp_types = ["Salaried", "Self-Employed", "Business Owner"]
            with st.expander("Applicant financial profile (used by the AI agents)",
                              expanded=selected_id == manual):
                fc1, fc2 = st.columns(2)
                with fc1:
                    income = st.number_input("Monthly Income (INR)", min_value=10000, step=1000,
                                              value=max(10000, int(prefill.get("monthly_income", 60000))),
                                              key=f"form_income_{sfx}")
                    credit_score = st.slider("Credit Score", 300, 900,
                                              _clamp(int(prefill.get("credit_score", 720)), 300, 900),
                                              key=f"form_credit_{sfx}")
                    employment_type = st.selectbox(
                        "Employment Type", emp_types,
                        index=emp_types.index(prefill["employment_type"])
                        if prefill.get("employment_type") in emp_types else 0,
                        key=f"form_employment_{sfx}")
                with fc2:
                    existing_emi = st.number_input("Existing Monthly EMI (INR)", min_value=0, step=500,
                                                     value=max(0, int(prefill.get("existing_monthly_emi", 0))),
                                                     key=f"form_emi_{sfx}")
                    years_job = st.number_input("Years at Current Job", min_value=0.0, step=0.5,
                                                  value=max(0.0, float(prefill.get("years_at_current_job", 2.0))),
                                                  key=f"form_years_{sfx}")
                    collateral = st.number_input("Collateral Value (INR, 0 if none)", min_value=0, step=50000,
                                                   value=max(0, int(prefill.get("collateral_value", 0))),
                                                   key=f"form_collateral_{sfx}")
                default_history = st.checkbox("Prior bank default on record",
                                               value=bool(prefill.get("bank_default_history", False)),
                                               key=f"form_default_{sfx}")

            applicant_id = selected_id if selected_id != manual else \
                st.session_state.setdefault("manual_id", f"APP-M{random.randint(1000, 9999)}")
            applicant = {
                "applicant_id": applicant_id,
                "name": name or "Unnamed Applicant",
                "monthly_income": income,
                "existing_monthly_emi": existing_emi,
                "existing_loans_count": prefill.get("existing_loans_count", 0),
                "credit_score": credit_score,
                "bank_default_history": default_history,
                "requested_loan_amount": requested_amount,
                "loan_purpose": purpose,
                "loan_type": loan_type,
                "tenure_years": tenure,
                "collateral_value": collateral,
                "years_at_current_job": years_job,
                "employment_type": employment_type,
                "city": prefill.get("city", "N/A"),
                "age": prefill.get("age", "N/A"),
            }

            st.write("")
            start = st.button("Start AI Assessment →", type="primary", width="stretch")

    if doc:
        with doc_card:
            _render_document_card(doc, applicant)

    if start:
        document = None
        if doc and doc.get("path"):
            uri = store_document(doc["path"], applicant_id, doc["filename"])
            document = {"path": doc["path"], "filename": doc["filename"],
                        "storage_uri": uri, "summary": doc.get("summary")}
        st.session_state.current_applicant = applicant
        st.session_state.current_document = document
        st.session_state.wizard_step = "progress"
        st.session_state.progress_started = False
        st.session_state.pop("manual_id", None)
        st.rerun()


def _wizard_progress():
    applicant = st.session_state.current_applicant
    document = st.session_state.get("current_document")
    topbar(f"Dashboard / New Assessment / {applicant['applicant_id']}",
           "AI Assessment in Progress", "", st.session_state.reviewer_name)

    left, right = st.columns([1, 1.3])

    if not st.session_state.get("progress_started"):
        st.session_state.progress_started = True
        st.session_state.progress_start_ts = time.time()

        with left:
            with st.container(border=True):
                card_heading("📋", "Workflow Steps")
                ph = {k: st.empty() for k in
                      ["extract", "validate", "router", "financial", "rag", "risk", "recommend", "human"]}
                bar = st.progress(0, text="Starting assessment...")

                with ph["extract"].container():
                    step_row("Document Intake Agent", "active")
                bar.progress(10, text="Running the workflow...")

                # Rows and progress bar update from real LangGraph node-completion events.
                progress_values = {
                    "document_extraction": (15, "extract"),
                    "validate_input": (25, "validate"),
                    "router": (35, "router"),
                    "financial_analysis": (55, "financial"),
                    "policy_retrieval": (55, "rag"),
                    "risk_assessment": (75, "risk"),
                    "recommendation": (90, "recommend"),
                    "human_approval_gate": (100, "human"),
                }
                def on_workflow_step(node_name, update):
                    if node_name not in progress_values:
                        return
                    pct, slot = progress_values[node_name]
                    detail = ""
                    label = {
                        "document_extraction": "Document Intake Agent",
                        "validate_input": "Input Validation (Pydantic)",
                        "router": "Router",
                        "financial_analysis": "Financial Analysis Agent",
                        "policy_retrieval": "Policy / RAG Retrieval Agent",
                        "risk_assessment": "Risk Agent (join)",
                        "recommendation": "Recommendation Agent",
                        "human_approval_gate": "Human Approval",
                    }[node_name]
                    if node_name == "document_extraction":
                        status = update.get("document_status", "none")
                        detail = "no document uploaded" if status == "none" else status
                        row_state = "skipped" if status == "none" else ("done" if status == "ok" else "warn")
                    elif node_name == "validate_input":
                        valid = update.get("is_valid", False)
                        detail = "All fields valid" if valid else "Failed validation"
                        row_state = "done" if valid else "warn"
                    elif node_name == "router":
                        detail = update.get("route", "routed")
                        row_state = "done"
                    elif node_name == "financial_analysis":
                        metrics = update.get("financial_metrics", {})
                        detail = f"DTI {metrics.get('dti_ratio', 0):.0%}"
                        row_state = "done"
                    elif node_name == "policy_retrieval":
                        detail = f"{len(update.get('policy_chunks', []))} clauses retrieved"
                        row_state = "done"
                    elif node_name == "risk_assessment":
                        detail = f"Risk: {update.get('risk_level', 'classified')}"
                        row_state = "done"
                    elif node_name == "recommendation":
                        detail = update.get("recommendation", "complete")
                        row_state = "done"
                    else:
                        detail, row_state = "Paused for reviewer", "pending"
                    with ph[slot].container():
                        step_row(label, row_state, detail)
                    bar.progress(pct, text=f"{label} completed")
                try:
                    state = run_until_human_review(
                        applicant, document, st.session_state.get("policy_thresholds", {}),
                        progress_callback=on_workflow_step,
                    )
                except Exception as e:
                    st.error(f"Assessment failed unexpectedly: {e}")
                    return
                st.session_state.current_state = state
                # Persist immediately, before the human-review step, so a refresh
                # does not erase the assessment results already produced.
                st.session_state.assessment_history[applicant["applicant_id"]] = {
                    "state": state,
                    "timestamp": datetime.now(),
                }
                save_assessment_history(st.session_state.assessment_history)

                dstatus = state.get("document_status", "none")
                n_issues = len(state.get("document_discrepancies", []))
                with ph["extract"].container():
                    if dstatus == "none":
                        step_row("Document Intake Agent", "skipped", "no document uploaded")
                    elif dstatus == "ok":
                        step_row("Document Intake Agent", "warn" if n_issues else "done",
                                 f"{len(state['document_fields'])} fields read · "
                                 + (f"{n_issues} mismatch(es)" if n_issues else "matches the form"))
                    else:
                        step_row("Document Intake Agent", "warn",
                                 "partly/not readable — manual check")
                bar.progress(20, text="Validating required fields and value ranges...")

                if not state.get("is_valid", True):
                    with ph["validate"].container():
                        step_row("Input Validation (Pydantic)", "pending", "Failed")
                    bar.progress(100, text="Validation failed.")
                    st.error("This application failed input validation and was not "
                             "sent to the AI agents:")
                    for err in state.get("validation_errors", []):
                        st.code(err)
                    if st.button("← Back to form"):
                        st.session_state.wizard_step = "form"
                        st.session_state.progress_started = False
                        st.rerun()
                    return

                with ph["validate"].container():
                    step_row("Input Validation (Pydantic)", "done", "All fields valid")
                is_expedited = state.get("route") == "expedited"
                with ph["router"].container():
                    step_row("Router", "done", "expedited fast-track" if is_expedited else "standard path")
                bar.progress(40, text="Financial and Policy/RAG agents ran in parallel...")
                with ph["financial"].container():
                    step_row("Financial Analysis Agent", "done",
                             f"DTI {state['financial_metrics']['dti_ratio']:.0%} (parallel branch A)")
                with ph["rag"].container():
                    step_row("Policy / RAG Retrieval Agent", "done",
                             f"{len(state.get('policy_chunks', []))} clauses retrieved (parallel branch B)")
                bar.progress(75, text="Classifying risk...")
                with ph["risk"].container():
                    step_row("Risk Agent (join)", "done", f"Risk classified: {state.get('risk_level')}")
                with ph["recommend"].container():
                    step_row("Recommendation Agent", "done", state.get("recommendation", ""))
                bar.progress(100, text="Awaiting human approval...")
                with ph["human"].container():
                    step_row("Human Approval", "pending", "Routed to reviewer")

        with right:
            with st.container(border=True):
                card_heading("🕸️", "Workflow Visualization")
                done = {"start", "validate", "router", "financial", "risk", "rag", "recommend"}
                if dstatus != "none":
                    done.add("document")
                dot = build_workflow_dot(completed_steps=done, active_step=None,
                                         route="expedited" if is_expedited else "standard")
                st.graphviz_chart(dot, width="stretch")

        st.session_state.progress_end_ts = time.time()
        st.session_state.wizard_step = "results"
        st.rerun()


def _wizard_results():
    state = st.session_state.current_state
    applicant = st.session_state.current_applicant
    metrics = state["financial_metrics"]
    risk_level = state["risk_level"]

    topbar(f"Dashboard / New Assessment / {applicant['applicant_id']}",
           "Assessment Results", f"Application ID: {applicant['applicant_id']}",
           st.session_state.reviewer_name)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat_card("🛡️", "Overall Risk", RISK_LABEL[risk_level], RISK_KIND[risk_level])
    with c2: stat_card("💳", "Credit Score", str(metrics["credit_score"]), "blue")
    with c3: stat_card("📉", "DTI Ratio", f"{metrics['dti_ratio']:.1%}",
                        "green" if metrics["dti_ratio"] <= 0.35 else "amber")
    with c4: stat_card("💰", "Monthly Income", inr(applicant["monthly_income"]), "blue")
    with c5: stat_card("🏠", "Loan Amount", inr(applicant["requested_loan_amount"]), "blue")

    if state.get("escalation_reasons"):
        st.warning("**Escalated to manual review** (policy: inconsistent or unreadable data stops automated approval)\n\n"
                   + "\n".join(f"- {r}" for r in state["escalation_reasons"]), icon="🚩")

    st.write("")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            card_heading("📊", "Financial Analysis")
            st.write(state["financial_summary"])
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            fc1.metric("Monthly Income", inr(applicant["monthly_income"]))
            fc1.metric("Existing EMI", inr(applicant["existing_monthly_emi"]))
            fc2.metric("Proposed EMI", inr(metrics["proposed_emi"]))
            fc2.metric("Loan Amount", inr(applicant["requested_loan_amount"]))
            st.caption("Debt-to-Income Ratio")
            st.progress(min(metrics["dti_ratio"], 1.0),
                        text=f"{metrics['dti_ratio']:.1%} — policy threshold is 50%")

        with st.container(border=True):
            card_heading("⚖️", "Risk Analysis")
            positives, negatives = positive_negative_factors(metrics, risk_level)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Positive Factors**")
                factor_list(positives, positive=True)
            with rc2:
                st.markdown("**Risk Factors**")
                factor_list(negatives, positive=False)
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.write(state["risk_explanation"])

    with right:
        if state.get("document_status", "none") != "none":
            with st.container(border=True):
                card_heading("📎", "Uploaded Document")
                dkind = {"ok": "green", "partial": "amber", "failed": "red"}[state["document_status"]]
                st.markdown(f"{badge(state['document_status'].upper(), dkind)} &nbsp; "
                            f"**{(st.session_state.get('current_document') or {}).get('filename', 'document')}**",
                            unsafe_allow_html=True)
                if state["document_status"] == "failed":
                    st.error(state.get("document_reason") or "Could not be read.")
                else:
                    st.write(state.get("document_summary", ""))
                    if state.get("document_discrepancies"):
                        for d in state["document_discrepancies"][:6]:
                            st.markdown(f"- ⚠️ {d}")
                        if len(state["document_discrepancies"]) > 6:
                            st.caption(f"…and {len(state['document_discrepancies']) - 6} more differences.")
                    else:
                        st.caption("✓ Matches the application data.")
                uri = (st.session_state.get("current_document") or {}).get("storage_uri")
                if uri:
                    st.caption(f"Stored at: `{uri}`")
        with st.container(border=True):
            card_heading("📚", "Policy Evidence (RAG)")
            chunks = state.get("policy_chunks", [])
            if not chunks:
                st.caption("No policy evidence retrieved.")
            for c in chunks:
                with st.container(border=True):
                    st.markdown(f"**📄 {c['source']}**")
                    st.caption(c["text"][:220] + ("…" if len(c["text"]) > 220 else ""))
        st.write("")
        if st.button("Continue to Human Review →", type="primary", width="stretch"):
            st.session_state.wizard_step = "review"
            st.rerun()


def _wizard_review():
    state = st.session_state.current_state
    applicant = st.session_state.current_applicant
    risk_level = state["risk_level"]
    recommendation = state["recommendation"]

    topbar(f"Dashboard / New Assessment / {applicant['applicant_id']}",
           "Human Review / Approval", "", st.session_state.reviewer_name)

    left, right = st.columns([1, 1.2])

    with left:
        with st.container(border=True):
            card_heading("🧭", "AI Recommendation")
            st.markdown(badge("REQUIRES MANUAL REVIEW" if state.get("requires_human_approval") else "AUTO-ELIGIBLE",
                               "amber" if state.get("requires_human_approval") else "green"),
                        unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**Risk Level:** {badge(RISK_LABEL[risk_level], RISK_KIND[risk_level])}",
                        unsafe_allow_html=True)
            st.markdown(f"**AI Recommendation:** {badge(recommendation, DECISION_KIND.get(recommendation,'gray'))}",
                        unsafe_allow_html=True)
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.markdown("**Reason**")
            st.write(state["recommendation_explanation"])
            for r in state.get("escalation_reasons", []):
                st.warning(r, icon="🚩")
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.markdown("**Supporting Documents**")
            with st.expander("📄 Financial Analysis Report"):
                st.write(state["financial_summary"])
            with st.expander("📄 Risk Analysis Report"):
                st.write(state["risk_explanation"])
            with st.expander(f"📄 Policy Evidence ({len(state.get('policy_chunks', []))})"):
                for c in state.get("policy_chunks", []):
                    st.caption(f"**{c['source']}**: {c['text'][:180]}…")

    with right:
        with st.container(border=True):
            card_heading("✍️", "Reviewer Decision")
            comment = st.text_area("Comments",
                                    placeholder="Applicant's DTI is acceptable with current income. Approving with standard terms.",
                                    key="review_comment", height=100)
            reviewer = st.text_input("Reviewer Name", value=st.session_state.reviewer_name, key="review_reviewer")
            st.caption(f"Date: {datetime.now().strftime('%d %b %Y')}")

            b1, b2, b3 = st.columns(3)
            decision = None
            if b1.button("✅ Approve", type="primary", width="stretch"):
                decision = "APPROVED"
            if b2.button("❌ Reject", width="stretch"):
                decision = "REJECTED"
            if b3.button("🔄 Request More Info", width="stretch"):
                decision = "REQUEST_MORE_INFO"

            if decision:
                st.session_state.reviewer_name = reviewer or "Analyst"
                with st.spinner("Recording decision and finalizing case..."):
                    final_state = resume_after_human_decision(
                        applicant["applicant_id"], decision, reviewer or "Analyst", comment)
                st.session_state.current_state = final_state
                if decision == "APPROVED":
                    st.balloons()
                st.session_state.assessment_history[applicant["applicant_id"]] = {
                    "state": final_state, "timestamp": datetime.now(),
                }
                save_assessment_history(st.session_state.assessment_history)
                st.session_state.wizard_step = "completed"
                st.rerun()


def _wizard_completed():
    state = st.session_state.current_state
    applicant = st.session_state.current_applicant
    metrics = state["financial_metrics"]
    decision = state.get("human_decision", "PENDING")

    topbar(f"Dashboard / New Assessment / {applicant['applicant_id']}",
           "Assessment Completed", "", st.session_state.reviewer_name)

    left, right = st.columns([1, 1.2])

    with left:
        with st.container(border=True):
            card_heading("🏁", "Final Decision")
            st.markdown(
                f'<div style="text-align:center;padding:14px 0;">'
                f'<span class="badge badge-{DECISION_KIND.get(decision,"gray")}" '
                f'style="font-size:1.05rem;padding:10px 22px;">{decision}</span></div>',
                unsafe_allow_html=True,
            )
            st.write(f"**Approved by:** {state.get('human_reviewer','N/A')}")
            st.write(f"**Date:** {datetime.now().strftime('%d %b %Y, %H:%M')}")
            if state.get("human_comment"):
                st.write(f"**Comment:** {state['human_comment']}")

            st.write("")
            pdf_bytes = build_report_pdf(state)
            st.download_button("⬇️ Download Report (PDF)", data=pdf_bytes,
                                file_name=f"{applicant['applicant_id']}_assessment_report.pdf",
                                mime="application/pdf", width="stretch")
            if st.button("📄 View Full Report", width="stretch"):
                st.session_state.show_full_report = not st.session_state.get("show_full_report", False)
            if st.button("➕ Start New Assessment", width="stretch"):
                st.session_state.wizard_step = "form"
                st.session_state.current_applicant = None
                st.session_state.current_state = None
                st.session_state.progress_started = False
                st.rerun()

        if st.session_state.get("show_full_report"):
            with st.container(border=True):
                card_heading("📃", "Full Report")
                st.write("**Financial Analysis**"); st.write(state["financial_summary"])
                st.write("**Risk Assessment**"); st.write(state["risk_explanation"])
                st.write("**Recommendation**"); st.write(state["recommendation_explanation"])

    with right:
        with st.container(border=True):
            card_heading("🧮", "Assessment Summary")
            tags = assessment_summary_tags(metrics, state["risk_level"])
            tc = st.columns(4)
            tag_kind = {"Good": "green", "Fair": "amber", "Weak": "red",
                        "Low": "green", "Medium": "amber", "High": "red",
                        "Pass": "green", "Flagged": "red"}
            for i, (label, val) in enumerate(tags.items()):
                with tc[i]:
                    st.caption(label)
                    badge_html(val, tag_kind.get(val, "gray"))

        with st.container(border=True):
            card_heading("📈", "Credit Score")
            score = int(metrics.get("credit_score", 0))
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 30}},
                gauge={"axis": {"range": [300, 900]},
                       "bar": {"color": "#2f6fed"},
                       "steps": [
                           {"range": [300, 650], "color": "#fee2e2"},
                           {"range": [650, 700], "color": "#fef3c7"},
                           {"range": [700, 900], "color": "#dcfce7"},
                       ],
                       "threshold": {"line": {"color": "#0f172a", "width": 3},
                                     "thickness": 0.8, "value": 700}},
            ))
            gauge.update_layout(height=210, margin=dict(t=15, b=5, l=20, r=20))
            st.plotly_chart(gauge, width="stretch", key="completed_credit_gauge")

        with st.container(border=True):
            card_heading("🕸️", "AI Workflow Overview")
            dstatus = state.get("document_status", "none")
            if dstatus == "none":
                step_row("Document Intake", "skipped", "no document uploaded")
            else:
                step_row("Document Intake", "done" if dstatus == "ok" and not state.get("document_discrepancies")
                         else "warn", f"{len(state.get('document_fields', {}))} fields read")
            for label in ["Validation", "Router", "Financial Analysis (parallel)",
                          "Policy Retrieval — RAG (parallel)", "Risk Analysis (join)",
                          "Recommendation", "Human Approval"]:
                step_row(label, "done")
            elapsed = st.session_state.get("progress_end_ts", time.time()) - \
                st.session_state.get("progress_start_ts", time.time())
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.metric("Total Time Taken", f"{max(elapsed, 0.1):.1f} sec")



# ---------------------------------------------------------------- assistant --

CHAT_SUGGESTIONS = ["Assess APP-1005", "What DTI ratio needs manual review?",
                    "How many applications are high risk?", "Show pending cases"]


def _ask(question: str):
    st.session_state.pending_chat_question = question


def _new_chat():
    st.session_state.chat_session_id = new_session_id()


def _open_case(applicant_id: str, step: str):
    state = get_state(applicant_id)
    st.session_state.current_applicant = state["applicant"]
    st.session_state.current_state = state
    st.session_state.current_document = None
    st.session_state.page = "new_assessment"
    st.session_state.wizard_step = step
    st.session_state.keep_wizard = True


def _tool_label(tc: dict) -> str:
    try:
        a = json.loads(tc["function"].get("arguments") or "{}")
    except ValueError:
        a = {}
    detail = a.get("applicant_id") or (a.get("question") or "")[:30] or a.get("status") or ""
    return f"{tc['function']['name']}({detail})"


def page_assistant():
    topbar("AI Assistant", "AI Assistant",
           "Ask about cases, the policy and the portfolio — or have me run an assessment.",
           st.session_state.reviewer_name)
    sid = st.session_state.setdefault("chat_session_id", new_session_id())

    c1, c2 = st.columns([5, 1])
    with c1:
        if assistant_mode() == "llm":
            st.caption("🟢 I can assess, explain, look up policy and show stats — final decisions always stay with a human reviewer.")
        else:
            st.caption("🟡 Rule-based mode (no OPENAI_API_KEY): I understand a fixed set of requests and answer "
                       "from templates — this is not generative text. Add an API key for open-ended chat.")
    with c2:
        st.button("New chat", on_click=_new_chat, width="stretch")

    history = get_history(sid)
    if not history:
        st.markdown("**Try one of these:**")
        st.caption("⌨️ Keyboard tip: press `?` to view available Streamlit keyboard shortcuts.")
        cols = st.columns(len(CHAT_SUGGESTIONS))
        for col, q in zip(cols, CHAT_SUGGESTIONS):
            col.button(q, key=f"sugg_{q}", on_click=_ask, args=(q,), width="stretch")

    last_idx = max((i for i, m in enumerate(history) if m["role"] == "assistant" and not m.get("tool_calls")),
                   default=-1)
    used: list[str] = []
    for idx, m in enumerate(history):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        elif m["role"] == "assistant" and m.get("tool_calls"):
            used += [_tool_label(tc) for tc in m["tool_calls"]]
        elif m["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(m["content"])
                notes = []
                if used:
                    notes.append("🔧 " + ", ".join(used))
                if (m.get("meta") or {}).get("note"):
                    notes.append("⚠️ " + m["meta"]["note"])
                if notes:
                    st.caption(" · ".join(notes))
                used = []
                focus = (m.get("meta") or {}).get("focus")
                case = get_state(focus) if (idx == last_idx and focus) else None
                if case and case.get("recommendation"):
                    b1, b2, _ = st.columns([1, 1.3, 3])
                    b1.button("Open results", key=f"open_res_{idx}", on_click=_open_case, args=(focus, "results"))
                    b2.button("Open human review" if not case.get("human_decision") else "Open final report",
                              key=f"open_rev_{idx}", on_click=_open_case,
                              args=(focus, "review" if not case.get("human_decision") else "completed"))

    typed = st.chat_input("Ask about a case, the policy or the portfolio…")
    question = typed or st.session_state.pop("pending_chat_question", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Thinking…"):
            chat_turn(sid, question)
        st.rerun()

# ------------------------------------------------------------- applications --

def page_applications():
    topbar("Applications", "All Applications", "Browse and act on every synthetic loan application.",
           st.session_state.reviewer_name)
    df = load_applicants_df()
    if df.empty:
        st.warning("No synthetic data found.")
        return
    scored = bulk_rule_based_status(df)

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    search = fc1.text_input("🔎 Search by name or ID", key="app_search")
    risk_filter = fc2.selectbox("Risk", ["All", "LOW", "MEDIUM", "HIGH"], key="app_risk_filter")
    status_filter = fc3.selectbox("Status", ["All", "Approved", "Manual Review", "High Risk"],
                                   key="app_status_filter")

    filtered = scored.copy()
    if search:
        mask = filtered["name"].str.contains(search, case=False, na=False) | \
               filtered["applicant_id"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if status_filter != "All":
        filtered = filtered[filtered["recommendation"].apply(_status_label) == status_filter]

    st.caption(f"{len(filtered)} of {len(scored)} applications")
    page_size = st.selectbox("Rows per page", [10, 20, 40, 80], index=1, key="applications_page_size")
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    if int(st.session_state.get("applications_page_number", 1)) > page_count:
        st.session_state["applications_page_number"] = 1
    current_page = st.number_input("Page", min_value=1, max_value=page_count,
                                   step=1, key="applications_page_number")
    start_idx = (int(current_page) - 1) * page_size
    page_df = filtered.iloc[start_idx:start_idx + page_size]
    st.download_button(
        "⬇️ Export filtered applications CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_loan_applications.csv",
        mime="text/csv",
        key="applications_csv_export",
    )

    with st.container(border=True):
        header = st.columns([1.2, 1.8, 1.5, 1.1, 1.5, 1.1])
        for h, label in zip(header, ["ID", "Customer", "Loan Amount", "Risk", "Status", ""]):
            h.markdown(f"**{label}**")
        st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
        for _, r in page_df.iterrows():
            cols = st.columns([1.2, 1.8, 1.5, 1.1, 1.5, 1.1])
            cols[0].write(r["applicant_id"])
            cols[1].write(r["name"])
            cols[2].write(inr(r["requested_loan_amount"]))
            cols[3].markdown(badge(r["risk_level"], RISK_KIND[r["risk_level"]]), unsafe_allow_html=True)
            cols[4].markdown(badge(_status_label(r["recommendation"]),
                                    DECISION_KIND.get(r["recommendation"], "gray")), unsafe_allow_html=True)
            if cols[5].button("Assess", key=f"app_assess_{r['applicant_id']}"):
                _jump_to_assessment(r["applicant_id"])


# -------------------------------------------------------------- assessments --

def page_assessments():
    topbar("Assessments", "Assessment History", "Persisted cases processed by the AI workflow.",
           st.session_state.reviewer_name)
    history = st.session_state.assessment_history
    if not history:
        st.info("No assessments run yet in this session. Start one from **New Assessment** or **Applications**.")
        return

    for app_id, record in sorted(history.items(), key=lambda kv: kv[1]["timestamp"], reverse=True):
        state = record["state"]
        with st.container(border=True):
            cols = st.columns([1.4, 2, 1.4, 1.4, 1.4, 1])
            cols[0].write(f"**{app_id}**")
            cols[1].write(state["applicant"]["name"])
            cols[2].markdown(badge(state["risk_level"], RISK_KIND[state["risk_level"]]), unsafe_allow_html=True)
            cols[3].markdown(badge(state["recommendation"], DECISION_KIND.get(state["recommendation"], "gray")),
                              unsafe_allow_html=True)
            human = state.get("human_decision")
            cols[4].markdown(badge(human or "Pending review", DECISION_KIND.get(human, "gray") if human else "gray"),
                              unsafe_allow_html=True)
            if cols[5].button("View", key=f"view_{app_id}"):
                st.session_state.current_applicant = state["applicant"]
                st.session_state.current_state = state
                st.session_state.page = "new_assessment"
                st.session_state.keep_wizard = True
                st.session_state.wizard_step = "completed" if human else "review"
                st.rerun()


# ---------------------------------------------------------- agent execution --

def _completed_nodes(trace: list[str]) -> set[str]:
    """Diagram nodes that really ran OK, derived from the execution trace (not hard-coded)."""
    ok = {t[5:].split(" (", 1)[0] for t in trace if t.startswith("[OK] ")}
    mapping = {"document_extraction": "document", "validate_input": "validate", "router": "router",
               "financial_analysis": "financial", "policy_retrieval": "rag", "risk_assessment": "risk",
               "recommendation": "recommend", "human_approval_gate": "human"}
    return {"start"} | {v for k, v in mapping.items() if k in ok}


def page_agent_execution():
    topbar("Agent Execution", "Agent Execution Trace", "Inspect exactly what each agent did for a given case.",
           st.session_state.reviewer_name)
    history = st.session_state.assessment_history
    if not history:
        st.info("No assessments run yet in this session.")
        return

    app_id = st.selectbox("Select an assessed application", list(history.keys()), key="agent_exec_select")
    state = history[app_id]["state"]

    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True):
            card_heading("🕸️", "Workflow Path")
            dot = build_workflow_dot(
                completed_steps=_completed_nodes(state.get("trace", [])),
                route=state.get("route"),
            )
            st.graphviz_chart(dot, width="stretch")
    with right:
        with st.container(border=True):
            card_heading("🧾", "Execution Log")
            for line in state.get("trace", []):
                ok = line.startswith("[OK]")
                st.markdown(
                    f'<div class="step-row"><div class="step-icon {"step-done" if ok else "step-pending"}">'
                    f'{"✓" if ok else "!"}</div><div style="font-family:monospace;font-size:0.82rem;">{line}</div></div>',
                    unsafe_allow_html=True,
                )
            if state.get("errors"):
                st.error("Errors encountered:")
                for e in state["errors"]:
                    st.code(e)


# --------------------------------------------------------------- documents --

def page_documents():
    topbar("Documents", "Document Library", "Bank policy documents (RAG corpus) and generated application forms.",
           st.session_state.reviewer_name)

    with st.container(border=True):
        card_heading("📚", "Policy Documents (RAG knowledge base)")
        for pdf in sorted(POLICY_DIR.glob("*.pdf")):
            cols = st.columns([3, 1, 1])
            cols[0].write(f"📄 {pdf.name}")
            cols[1].caption(f"{pdf.stat().st_size/1024:.0f} KB")
            with open(pdf, "rb") as f:
                cols[2].download_button("Download", f.read(), file_name=pdf.name,
                                         mime="application/pdf", key=f"dl_policy_{pdf.name}")

    with st.container(border=True):
        card_heading("🗂️", "Sample Application Forms")
        forms = sorted(APPLICATIONS_DIR.glob("*_form.pdf"))
        if not forms:
            st.caption("No sample forms generated.")
        for pdf in forms:
            cols = st.columns([3, 1, 1])
            cols[0].write(f"📄 {pdf.name}")
            cols[1].caption(f"{pdf.stat().st_size/1024:.0f} KB")
            with open(pdf, "rb") as f:
                cols[2].download_button("Download", f.read(), file_name=pdf.name,
                                         mime="application/pdf", key=f"dl_form_{pdf.name}")


# ----------------------------------------------------------------- reports --

def page_reports():
    topbar("Reports", "Assessment Reports", "Generate a formatted PDF report for any completed case.",
           st.session_state.reviewer_name)
    history = st.session_state.assessment_history
    completed = {k: v for k, v in history.items() if v["state"].get("human_decision")}
    if not completed:
        st.info("No finalized (human-reviewed) assessments yet. Complete one in **New Assessment**.")
        return

    app_id = st.selectbox("Select a finalized application", list(completed.keys()), key="report_select")
    state = completed[app_id]["state"]

    with st.container(border=True):
        card_heading("📊", "Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", state["risk_level"])
        c2.metric("AI Recommendation", state["recommendation"])
        c3.metric("Human Decision", state["human_decision"])
        st.write(state["recommendation_explanation"])

    pdf_bytes = build_report_pdf(state)
    st.download_button("⬇️ Generate & Download PDF Report", data=pdf_bytes,
                        file_name=f"{app_id}_assessment_report.pdf", mime="application/pdf",
                        type="primary")


# ---------------------------------------------------------------- settings --

def page_settings():
    topbar("Settings", "Settings", "System configuration and defaults.", st.session_state.reviewer_name)

    with st.container(border=True):
        card_heading("🧑‍💼", "Reviewer Profile")
        name = st.text_input("Default reviewer name", value=st.session_state.reviewer_name, key="settings_reviewer")
        if name != st.session_state.reviewer_name:
            st.session_state.reviewer_name = name
            st.success("Updated.")

    with st.container(border=True):
        card_heading("🧠", "LLM Engine")
        st.write(f"**Mode:** {'Mock (rule-based, offline)' if USE_MOCK else 'Live API'}")
        st.write(f"**Model:** {MODEL_NAME}")
       # st.caption("Set OPENAI_API_KEY in your .env file to switch from mock to live generative responses.")

    with st.container(border=True):
        card_heading("📐", "Policy Thresholds (session only)")
        st.caption("Changes apply to assessments started after you save them in this browser session. "
                   "They are not written back to bank policy documents.")
        # Defensive initialization: Settings may be opened with a stale session
        # created before policy thresholds were introduced.
        threshold_defaults = {
            "expedited_loan_ceiling": 800000,
            "expedited_credit_floor": 720,
            "high_dti": 0.50,
            "low_dti": 0.35,
            "high_credit_floor": 650,
            "low_credit_floor": 700,
            "minimum_job_tenure": 1.0,
        }
        saved_thresholds = st.session_state.get("policy_thresholds")
        if not isinstance(saved_thresholds, dict):
            saved_thresholds = {}
        current = {**threshold_defaults, **saved_thresholds}
        st.session_state["policy_thresholds"] = current
        t1, t2 = st.columns(2)
        with t1:
            exp_loan = st.number_input("Expedited loan ceiling (INR)", min_value=50000,
                                       max_value=20000000, step=50000,
                                       value=int(current["expedited_loan_ceiling"]), key="edit_exp_loan")
            exp_credit = st.slider("Expedited minimum credit score", 300, 900,
                                   int(current["expedited_credit_floor"]), key="edit_exp_credit")
            high_credit = st.slider("High-risk credit floor", 300, 750,
                                    int(current["high_credit_floor"]), key="edit_high_credit")
            low_credit = st.slider("Low-risk credit floor", 500, 850,
                                   int(current["low_credit_floor"]), key="edit_low_credit")
        with t2:
            low_dti = st.slider("Low-risk maximum DTI", 0.10, 0.60,
                                float(current["low_dti"]), step=0.01, key="edit_low_dti")
            high_dti = st.slider("High-risk DTI threshold", 0.20, 0.90,
                                 float(current["high_dti"]), step=0.01, key="edit_high_dti")
            min_tenure = st.number_input("Minimum job tenure (years)", min_value=0.0,
                                         max_value=20.0, step=0.5,
                                         value=float(current["minimum_job_tenure"]), key="edit_min_tenure")
        if st.button("💾 Apply thresholds to future assessments", key="save_policy_thresholds"):
            if high_credit >= low_credit:
                st.error("Low-risk credit floor must be higher than the high-risk credit floor.")
            elif low_dti >= high_dti:
                st.error("Low-risk DTI threshold must be lower than the high-risk DTI threshold.")
            else:
                st.session_state.policy_thresholds = {
                    "expedited_loan_ceiling": int(exp_loan),
                    "expedited_credit_floor": int(exp_credit),
                    "high_dti": float(high_dti),
                    "low_dti": float(low_dti),
                    "high_credit_floor": int(high_credit),
                    "low_credit_floor": int(low_credit),
                    "minimum_job_tenure": float(min_tenure),
                }
                st.success("Thresholds saved for this session.")

    with st.container(border=True):
        card_heading("🎨", "Appearance")
        st.caption("Use the sidebar toggle to switch between light and dark mode.")
        st.write(f"**Current mode:** {'Dark' if st.session_state.get('dark_mode') else 'Light'}")

    with st.container(border=True):
        card_heading("ℹ️", "About this application")
        st.write("**Application:** AI Loan Eligibility Assessment Assistant")
        st.write("**Version:** 1.1.0")
        st.write("**Build:** Docker / Streamlit")
        st.write(f"**Python runtime:** {os.sys.version.split()[0]}")
        st.write("**Checkpoint store:** SQLite")
        st.caption("AI recommendations support human review and do not constitute a final lending decision.")

    with st.container(border=True):
        card_heading("☁️", "AWS S3 Integration")
        info = s3_status()
        st.write(f"**Bucket:** {info['bucket'] or 'not configured'}  ·  **Endpoint:** {info['endpoint']}")
        if not info["configured"]:
            st.info("S3 is not configured — uploaded documents and audit events stay on local disk. "
                    "Set S3_BUCKET_NAME and AWS credentials in `.env` to enable S3.")
        elif info["reachable"]:
            st.success("Connected: the bucket is reachable. Uploaded documents and every audit event are written to S3.")
        else:
            st.error(f"Configured but not reachable: {info['error']}")
        if st.button("Sync policy PDFs and sample forms to S3", disabled=not info["configured"],
                     key="s3_sync_btn"):
            with st.spinner("Uploading..."):
                res = sync_all()
            n = sum(v["uploaded"] for v in res.values())
            failed = [f for v in res.values() for f in v["failed"]]
            (st.success if not failed else st.warning)(
                f"Uploaded {n} files." + (f" {len(failed)} failed: {failed[:2]}" if failed else ""))

    with st.container(border=True):
        card_heading("🛡️", "Input Validation")
        st.write("Every application is validated against a Pydantic schema "
                  "(`utils/schemas.py`) before any agent runs — invalid income, "
                  "out-of-range credit scores, or missing required fields are "
                  "rejected with a clear message instead of producing a broken "
                  "financial calculation downstream.")

    with st.container(border=True):
        card_heading("📜", "Audit Log (data/audit_log/audit_log.jsonl)")
        events = read_recent_events(limit=15)
        if not events:
            st.caption("No audit events recorded yet in this session.")
        else:
            for ev in events:
                ts = ev.get("timestamp", "")[:19].replace("T", " ")
                st.markdown(
                    f'<div class="step-row"><div class="step-icon step-done">✓</div>'
                    f'<div style="font-family:monospace;font-size:0.8rem;">'
                    f'{ts} · <b>{ev.get("event_type")}</b> · {ev.get("applicant_id")}</div></div>',
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        card_heading("🗒️", "Log File")
        log_path = BASE_DIR / "logs" / "app.log"
        if log_path.exists():
            st.caption(f"Structured logs are written to `{log_path}`.")
            with st.expander("View last 30 log lines"):
                lines = log_path.read_text(encoding="utf-8").splitlines()[-30:]
                st.code("\n".join(lines), language="text")
        else:
            st.caption("No log file yet — run an assessment to generate one.")
