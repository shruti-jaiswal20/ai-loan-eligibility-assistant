"""
AI Loan Eligibility Assessment Assistant - Streamlit frontend.

Redesigned dashboard UI: dark sidebar navigation, card-based layout,
a step-by-step "New Assessment" wizard (upload -> agent progress ->
results -> human review -> completed), plus Applications, Assessments,
Agent Execution, Documents, Reports and Settings pages.

Run locally:   streamlit run app.py
"""
import streamlit as st

from ui.theme import THEME_CSS
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
st.markdown(THEME_CSS, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "page": "dashboard",
        "wizard_step": "form",
        "current_applicant": None,
        "current_state": None,
        "assessment_history": {},
        "reviewer_name": "Analyst",
        "progress_started": False,
        "keep_wizard": False,
        "show_full_report": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    init_session_state()
    render_sidebar()

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
