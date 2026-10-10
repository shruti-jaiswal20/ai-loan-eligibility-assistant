
"""Custom CSS injected once at app start to reskin Streamlit into the
dark-sidebar / card-based dashboard look."""

THEME_CSS = """
<style>
:root {
    --navy: #0f172a;
    --navy-2: #16213a;
    --accent: #2f6fed;
    --accent-soft: rgba(47,111,237,0.14);
    --green: #16a34a;
    --green-soft: rgba(22,163,74,0.12);
    --amber: #d97706;
    --amber-soft: rgba(217,119,6,0.12);
    --red: #dc2626;
    --red-soft: rgba(220,38,38,0.12);
    --gray-text: #64748b;
    --card-radius: 14px;
}

/* ---- overall app background ---- */
[data-testid="stAppViewContainer"] {
    background: #f4f6fb;
}

.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        var(--navy) 0%,
        var(--navy-2) 100%
    );
    min-width: 250px !important;
}

/* Sidebar text and branding */
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Sidebar buttons: left-align icons and labels */
section[data-testid="stSidebar"] [data-testid="stButton"] > button {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    width: 100% !important;
    background: transparent;
    border: none;
    padding: 0.55rem 0.9rem;
    border-radius: 10px;
    font-size: 0.92rem;
    font-weight: 500;
    transition: background 0.15s ease;
}

/* Ensure the button's inner content is left-aligned */
section[data-testid="stSidebar"] [data-testid="stButton"] > button > div {
    justify-content: flex-start !important;
    width: 100%;
}

/* Ensure button labels use the available width */
section[data-testid="stSidebar"] [data-testid="stButton"] > button p {
    text-align: left !important;
    width: 100%;
    margin: 0;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08);
    color: #fff !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] p {
    color: #fff !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
}

/* Quick-ask popover */
section[data-testid="stSidebar"] [data-testid="stPopoverButton"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px;
}

section[data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover {
    background: rgba(255,255,255,0.14) !important;
}

section[data-testid="stSidebar"] [data-baseweb="popover"] *,
section[data-testid="stSidebar"] [data-testid="stPopoverBody"] * {
    color: #1e293b !important;
}

/* Optional active navigation styling */
.nav-active button {
    background: var(--accent) !important;
    color: #fff !important;
}

.nav-active button p {
    color: #fff !important;
}

/* Sidebar brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 4px 18px 4px;
}

.sidebar-brand .logo {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.sidebar-brand .name {
    font-weight: 700;
    color: #fff !important;
    font-size: 0.95rem;
    line-height: 1.1;
}

.sidebar-brand .sub {
    color: #94a3b8 !important;
    font-size: 0.7rem;
}

.sidebar-section-label {
    color: #64748b !important;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 14px 6px 4px 6px;
    font-weight: 600;
}

.sidebar-status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 6px;
    font-size: 0.78rem;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    background: #16a34a;
}

/* ---- top bar ---- */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
}

.breadcrumb {
    color: var(--gray-text);
    font-size: 0.85rem;
}

.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #111827;
    margin: 2px 0 0 0;
}

.page-subtitle {
    color: var(--gray-text);
    font-size: 0.9rem;
    margin-top: 2px;
}

.topbar-user {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #fff;
    border-radius: 999px;
    padding: 6px 14px 6px 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-size: 0.85rem;
    font-weight: 600;
    color: #111827;
}

.topbar-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--accent-soft);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

/* ---- cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--card-radius) !important;
    border: 1px solid #e7eaf2 !important;
    background: #fff;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}

/* ---- badges ---- */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.badge-green {
    background: var(--green-soft);
    color: var(--green);
}

.badge-amber {
    background: var(--amber-soft);
    color: var(--amber);
}

.badge-red {
    background: var(--red-soft);
    color: var(--red);
}

.badge-blue {
    background: var(--accent-soft);
    color: var(--accent);
}

.badge-gray {
    background: #f1f5f9;
    color: #475569;
}

/* ---- stat card ---- */
.stat-card {
    padding: 4px 6px 2px 6px;
}

.stat-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}

.stat-label {
    color: var(--gray-text);
    font-size: 0.82rem;
    margin-top: 2px;
}

.stat-delta {
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 6px;
}

/* ---- section heading inside cards ---- */
.card-heading {
    font-weight: 700;
    font-size: 1rem;
    color: #0f172a;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- checklist / step rows ---- */
.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    font-size: 0.88rem;
}

.step-icon {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
}

.step-done {
    background: var(--green-soft);
    color: var(--green);
}

.step-active {
    background: var(--accent-soft);
    color: var(--accent);
}

.step-pending {
    background: #f1f5f9;
    color: #94a3b8;
}

.step-warn {
    background: var(--amber-soft);
    color: var(--amber);
}

/* ---- factor lists ---- */
.factor-item {
    font-size: 0.85rem;
    padding: 4px 0;
}

.factor-pos::before {
    content: "✓  ";
    color: var(--green);
    font-weight: 700;
}

.factor-neg::before {
    content: "✗  ";
    color: var(--red);
    font-weight: 700;
}

hr.soft {
    border: none;
    border-top: 1px solid #eef0f5;
    margin: 10px 0;
}

/* ---- buttons in main area ---- */
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);
}

.stButton > button[kind="primary"]:hover {
    background: #1d55c9;
    border-color: #1d55c9;
}

/* ---- file uploader dashed box ---- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
}
</style>
"""
