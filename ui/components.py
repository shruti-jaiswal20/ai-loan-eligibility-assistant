"""Small reusable rendering helpers used across every page."""
import streamlit as st

RISK_KIND = {"LOW": "green", "MEDIUM": "amber", "HIGH": "red"}
RISK_LABEL = {"LOW": "LOW RISK", "MEDIUM": "MEDIUM RISK", "HIGH": "HIGH RISK"}
DECISION_KIND = {
    "APPROVE": "green", "APPROVED": "green",
    "REJECT": "red", "REJECTED": "red",
    "REFER_FOR_MANUAL_REVIEW": "amber", "REQUEST_MORE_INFO": "amber",
}
STEP_ICON = {"done": ("✓", "step-done"), "active": ("●", "step-active"), "pending": ("○", "step-pending"),
             "skipped": ("–", "step-pending"), "warn": ("!", "step-warn")}


def inr(amount) -> str:
    try:
        return f"₹{amount:,.0f}"
    except (TypeError, ValueError):
        return "₹0"


def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def badge_html(text: str, kind: str = "gray"):
    st.markdown(badge(text, kind), unsafe_allow_html=True)


def topbar(breadcrumb: str, title: str, subtitle: str = "", reviewer_name: str = "Analyst"):
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="breadcrumb">{breadcrumb}</div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle" style="{'display:none;' if not subtitle else ''}">{subtitle}</div>
            </div>
            <div class="topbar-user">
                <div class="topbar-avatar">👤</div>
                <div>{reviewer_name}<div style="font-weight:400;color:#94a3b8;font-size:0.72rem;">Analyst</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(icon: str, label: str, value: str, kind: str = "blue", delta: str | None = None):
    icon_bg = {
        "blue": "rgba(47,111,237,0.12)", "green": "rgba(22,163,74,0.12)",
        "amber": "rgba(217,119,6,0.12)", "red": "rgba(220,38,38,0.12)",
    }[kind]
    icon_fg = {"blue": "#2f6fed", "green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}[kind]
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon" style="background:{icon_bg};color:{icon_fg};">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                {f'<div class="stat-delta" style="color:{icon_fg};">{delta}</div>' if delta else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def card_heading(icon: str, text: str):
    st.markdown(f'<div class="card-heading">{icon} {text}</div>', unsafe_allow_html=True)


def step_row(label: str, state: str, detail: str = ""):
    """state: 'done' | 'active' | 'pending'"""
    icon, cls = STEP_ICON[state]
    detail_html = f'<span style="color:#94a3b8;font-size:0.78rem;"> — {detail}</span>' if detail else ""
    st.markdown(
        f"""
        <div class="step-row">
            <div class="step-icon {cls}">{icon}</div>
            <div>{label}{detail_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def factor_list(items: list[str], positive: bool):
    cls = "factor-pos" if positive else "factor-neg"
    if not items:
        st.caption("None identified.")
        return
    html = "".join(f'<div class="factor-item {cls}">{i}</div>' for i in items)
    st.markdown(html, unsafe_allow_html=True)
