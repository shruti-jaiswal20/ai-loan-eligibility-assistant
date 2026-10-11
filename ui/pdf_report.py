"""Generates the downloadable PDF assessment report shown on the
'Assessment Completed' screen."""
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, ListFlowable, ListItem, KeepTogether)
from xml.sax.saxutils import escape

from ui.insights import positive_negative_factors, assessment_summary_tags


def _pdf_text(value) -> str:
    """Normalize common Unicode punctuation/currency symbols for built-in PDF fonts.

    ReportLab's built-in Helvetica fonts do not cover all Unicode characters. Replacing
    these symbols prevents black-square glyphs in generated reports.
    """
    text = str(value)
    replacements = {
        "₹": "INR ", "\u00a0": " ", "\u202f": " ",
        "–": "-", "—": "-", "‑": "-", "−": "-",
        "≥": ">=", "≤": "<=", "≠": "!=", "×": "x",
        "·": "|", "•": "-", "…": "...", "’": "'", "‘": "'",
        "“": '"', "”": '"', "€": "EUR ", "£": "GBP ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    # Replace any remaining non-ASCII characters that Helvetica may not support.
    return text.encode("cp1252", errors="replace").decode("cp1252")


def build_report_pdf(state: dict) -> bytes:
    buf = io.BytesIO()

    def draw_footer(canvas, document):
        canvas.saveState()
        width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#dbe3ef"))
        canvas.line(2 * cm, 1.55 * cm, width - 2 * cm, 1.55 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(2 * cm, 1.1 * cm,
                          "Confidential | AI-generated recommendation only | Human underwriting decision required")
        canvas.drawRightString(width - 2 * cm, 1.1 * cm, f"Page {document.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], fontName="Helvetica-Bold",
                              fontSize=20, leading=24, textColor=colors.HexColor("#0f172a"),
                              spaceAfter=5))
    styles.add(ParagraphStyle(name="BrandSub", parent=styles["Normal"], fontSize=8,
                              textColor=colors.HexColor("#64748b"), spaceAfter=12))
    styles.add(ParagraphStyle(name="FooterNote", parent=styles["BodyText"], fontSize=8,
                              leading=10, textColor=colors.HexColor("#475569")))
    story = []

    applicant = state.get("applicant", {})
    metrics = state.get("financial_metrics", {})
    risk_level = state.get("risk_level", "N/A")

    story.append(Paragraph("AI LOAN ASSESSMENT", styles["BrandSub"]))
    story.append(Paragraph("Loan Eligibility Assessment Report", styles["H1c"]))
    story.append(Paragraph(
        f"Application ID: {escape(_pdf_text(applicant.get('applicant_id', 'N/A')))} &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["Normal"]))
    risk_color = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626"}.get(risk_level, "#64748b")
    risk_banner = Table([[Paragraph(
        f'<font color="white"><b>OVERALL RISK: {escape(_pdf_text(risk_level))}</b></font>',
        styles["Normal"])]], colWidths=[15 * cm])
    risk_banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(risk_color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 8))
    story.append(risk_banner)
    story.append(Spacer(1, 14))

    # Applicant summary table
    rows = [["Field", "Value"],
            ["Applicant Name", escape(_pdf_text(applicant.get("name", "N/A")))],
            ["Loan Purpose", escape(_pdf_text(applicant.get("loan_purpose", "N/A")))],
            ["Requested Amount", f"INR {applicant.get('requested_loan_amount',0):,.0f}"],
            ["Tenure", f"{applicant.get('tenure_years','N/A')} years"],
            ["Monthly Income", f"INR {applicant.get('monthly_income',0):,.0f}"],
            ["Credit Score", str(metrics.get("credit_score", "N/A"))],
            ["DTI Ratio", f"{metrics.get('dti_ratio',0):.0%}"],
            ["Overall Risk", risk_level]]
    table = Table(rows, colWidths=[6 * cm, 9 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Financial Analysis", styles["Heading2"]))
    story.append(Paragraph(escape(_pdf_text(state.get("financial_summary", ""))), styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Risk Assessment", styles["Heading2"]))
    story.append(Paragraph(f"<b>Risk Level: {escape(_pdf_text(risk_level))}</b>", styles["BodyText"]))
    story.append(Paragraph(escape(_pdf_text(state.get("risk_explanation", ""))), styles["BodyText"]))
    story.append(Spacer(1, 6))

    pos, neg = positive_negative_factors(metrics, risk_level)
    story.append(Paragraph("Positive factors:", styles["Heading4"]))
    story.append(ListFlowable([ListItem(Paragraph(p, styles["BodyText"])) for p in pos],
                               bulletType="bullet"))
    story.append(Paragraph("Risk factors:", styles["Heading4"]))
    story.append(ListFlowable([ListItem(Paragraph(n, styles["BodyText"])) for n in neg],
                               bulletType="bullet"))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Policy Evidence (RAG-retrieved)", styles["Heading2"]))
    story.append(Paragraph(f"Sources consulted: {escape(_pdf_text(', '.join(state.get('policy_sources', []))))}",
                            styles["BodyText"]))
    story.append(Spacer(1, 10))

    if state.get("document_status", "none") != "none":
        story.append(Paragraph("Uploaded Document Check", styles["Heading2"]))
        story.append(Paragraph(f"Status: <b>{escape(_pdf_text(state['document_status']).upper())}</b>. "
                                f"{escape(_pdf_text(state.get('document_summary') or state.get('document_reason', '')))}",
                                styles["BodyText"]))
        for d in state.get("document_discrepancies", []):
            story.append(Paragraph(f"&bull; Mismatch: {escape(_pdf_text(d))}", styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("AI Recommendation", styles["Heading2"]))
    story.append(Paragraph(f"<b>{escape(_pdf_text(state.get('recommendation','N/A')))}</b>", styles["BodyText"]))
    story.append(Paragraph(escape(_pdf_text(state.get("recommendation_explanation", ""))), styles["BodyText"]))
    for r in state.get("escalation_reasons", []):
        story.append(Paragraph(f"<b>Escalated:</b> {escape(_pdf_text(r))}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Human Review Decision", styles["Heading2"]))
    story.append(Paragraph(
        f"<b>Final Decision: {escape(_pdf_text(state.get('human_decision','Pending')))}</b><br/>"
        f"Reviewer: {escape(_pdf_text(state.get('human_reviewer','N/A')))}<br/>"
        f"Comment: {escape(_pdf_text(state.get('human_comment','') or '—'))}",
        styles["BodyText"]))

    story.append(Spacer(1, 16))
    reviewer = str(state.get("human_reviewer") or "Reviewer")
    initials = "".join(part[0] for part in reviewer.split()[:2]).upper() or "—"
    sign_rows = [
        ["Reviewer initials", "Review date", "Decision"],
        [initials, datetime.now().strftime("%d %b %Y"), str(state.get("human_decision", "Pending"))],
    ]
    sign_table = Table(sign_rows, colWidths=[5 * cm, 5 * cm, 5 * cm])
    sign_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Paragraph("Reviewer sign-off", styles["Heading2"]))
    story.append(sign_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Important:</b> This report is decision support only. It may contain AI-generated "
        "summaries and must be checked against source documents and applicable lending policy. "
        "It is not a loan offer, approval guarantee, or authorization to disburse funds.",
        styles["FooterNote"],
    ))

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buf.getvalue()
