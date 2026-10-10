"""Generates the downloadable PDF assessment report shown on the
'Assessment Completed' screen."""
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, ListFlowable, ListItem)

from ui.insights import positive_negative_factors, assessment_summary_tags


def build_report_pdf(state: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceAfter=6))
    story = []

    applicant = state.get("applicant", {})
    metrics = state.get("financial_metrics", {})
    risk_level = state.get("risk_level", "N/A")

    story.append(Paragraph("AI Loan Eligibility Assessment Report", styles["H1c"]))
    story.append(Paragraph(
        f"Application ID: {applicant.get('applicant_id','N/A')} &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 14))

    # Applicant summary table
    rows = [["Field", "Value"],
            ["Applicant Name", applicant.get("name", "N/A")],
            ["Loan Purpose", applicant.get("loan_purpose", "N/A")],
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
    story.append(Paragraph(state.get("financial_summary", ""), styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Risk Assessment", styles["Heading2"]))
    story.append(Paragraph(f"<b>Risk Level: {risk_level}</b>", styles["BodyText"]))
    story.append(Paragraph(state.get("risk_explanation", ""), styles["BodyText"]))
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
    story.append(Paragraph(f"Sources consulted: {', '.join(state.get('policy_sources', []))}",
                            styles["BodyText"]))
    story.append(Spacer(1, 10))

    if state.get("document_status", "none") != "none":
        story.append(Paragraph("Uploaded Document Check", styles["Heading2"]))
        story.append(Paragraph(f"Status: <b>{state['document_status'].upper()}</b>. "
                                f"{state.get('document_summary') or state.get('document_reason', '')}",
                                styles["BodyText"]))
        for d in state.get("document_discrepancies", []):
            story.append(Paragraph(f"&bull; Mismatch: {d}", styles["BodyText"]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("AI Recommendation", styles["Heading2"]))
    story.append(Paragraph(f"<b>{state.get('recommendation','N/A')}</b>", styles["BodyText"]))
    story.append(Paragraph(state.get("recommendation_explanation", ""), styles["BodyText"]))
    for r in state.get("escalation_reasons", []):
        story.append(Paragraph(f"<b>Escalated:</b> {r}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Human Review Decision", styles["Heading2"]))
    story.append(Paragraph(
        f"<b>Final Decision: {state.get('human_decision','Pending')}</b><br/>"
        f"Reviewer: {state.get('human_reviewer','N/A')}<br/>"
        f"Comment: {state.get('human_comment','') or '—'}",
        styles["BodyText"]))

    doc.build(story)
    return buf.getvalue()
