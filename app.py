"""
AI Loan Eligibility Assessment Assistant - Streamlit frontend.

Redesigned dashboard UI: dark sidebar navigation, card-based layout,
a step-by-step "New Assessment" wizard (upload -> agent progress ->
results -> human review -> completed), plus Applications, Assessments,
Agent Execution, Documents, Reports and Settings pages.

Run locally:   streamlit run app.py
"""
import hashlib
import hmac
import os
import streamlit as st
from ui.components import render_footer

from ui.theme import THEME_CSS
from utils.assessment_store import load_assessment_history
from ui.pages import (
    render_sidebar, page_dashboard, page_new_assessment, page_applications,
    page_assessments, page_agent_execution, page_documents, page_reports,
    page_settings, page_assistant,
)

st.set_page_config(
    page_title="AI Loan Eligibility Assessment Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
def _render_login() -> bool:
    """Always show the login screen; fail closed until credentials are configured."""
    expected_hash = os.getenv("APP_LOGIN_PASSWORD_SHA256", "").strip().lower()
    expected_user = os.getenv("APP_LOGIN_USERNAME", "analyst").strip()
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        "<div style='max-width:520px;margin:8vh auto 0;text-align:center'>"
        "<div style='font-size:2.4rem'>🏦</div><h1>AI Loan Assessment</h1>"
        "<p>Sign in to continue to the underwriting workspace.</p></div>",
        unsafe_allow_html=True,
    )

    if not expected_hash:
        st.warning("Login is not configured yet. Set APP_LOGIN_USERNAME and APP_LOGIN_PASSWORD_SHA256 in the application's .env file, then restart the Docker container.")

    with st.form("login_form"):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

    if submitted:
        if not expected_hash:
            st.error("Sign-in is unavailable until an administrator configures the login credentials.")
        else:
            supplied_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
            if hmac.compare_digest(username, expected_user) and hmac.compare_digest(supplied_hash, expected_hash):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    return False


def init_session_state():
    defaults = {
        "page": "dashboard",
        "wizard_step": "form",
        "current_applicant": None,
        "current_state": None,
        "assessment_history": load_assessment_history(),
        "reviewer_name": os.getenv("APP_REVIEWER_NAME", "Analyst"),
        "dark_mode": False,
        "policy_thresholds": {
            "expedited_loan_ceiling": 800000,
            "expedited_credit_floor": 720,
            "high_dti": 0.50,
            "low_dti": 0.35,
            "high_credit_floor": 650,
            "low_credit_floor": 700,
            "minimum_job_tenure": 1.0,
        },
        "progress_started": False,
        "keep_wizard": False,
        "show_full_report": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state or st.session_state.get(k) is None:
            # Copy mutable defaults so sessions never share the same object.
            st.session_state[k] = v.copy() if isinstance(v, dict) else v

    # Repair partial state left by older app versions or an interrupted rerun.
    # This prevents Settings from failing if policy_thresholds is missing keys.
    thresholds = defaults["policy_thresholds"]
    existing = st.session_state.get("policy_thresholds")
    if not isinstance(existing, dict):
        existing = {}
    st.session_state["policy_thresholds"] = {**thresholds, **existing}


def main():
    init_session_state()
    if not _render_login():
        return
    from ui.theme import DARK_THEME_CSS
    # Render the sidebar toggle first, then inject CSS using its current value.
    # Streamlit reruns the script after a toggle change, so the whole page updates.
    render_sidebar()
    st.markdown(
        THEME_CSS + (DARK_THEME_CSS if st.session_state.get("dark_mode", False) else ""),
        unsafe_allow_html=True,
    )

    page = st.session_state.page
    if page == "dashboard":
        page_dashboard()
    elif page == "new_assessment":
        page_new_assessment()
    elif page == "assistant":
        page_assistant()
    elif page == "applications":
        page_applications()
    elif page == "assessments":
        page_assessments()
    elif page == "agent_execution":
        page_agent_execution()
    elif page == "documents":
        page_documents()
    elif page == "reports":
        page_reports()
    elif page == "settings":
        page_settings()


if __name__ == "__main__":
    main()

render_footer()
