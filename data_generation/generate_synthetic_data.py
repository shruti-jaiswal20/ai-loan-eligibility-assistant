"""
Generates all synthetic artifacts required by the RFP for Theme 11
(AI Loan Eligibility Assessment Assistant):

  1. data/applications/applicants.csv   -> synthetic customer financial data
  2. data/applications/APP-*.json       -> individual loan application records
  3. data/policy_docs/*.pdf             -> bank loan policy PDFs (RAG corpus)
  4. data/applications/APP-*_form.pdf   -> one filled application form per applicant

Run:  python data_generation/generate_synthetic_data.py
"""
import json
import os
import random
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE_DIR = Path(__file__).resolve().parent.parent
APPLICATIONS_DIR = BASE_DIR / "data" / "applications"
POLICY_DIR = BASE_DIR / "data" / "policy_docs"
APPLICATIONS_DIR.mkdir(parents=True, exist_ok=True)
POLICY_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan",
               "Meera", "Arjun", "Isha", "Rahul", "Divya", "Sanjay", "Neha", "Amit"]
LAST_NAMES = ["Sharma", "Patel", "Reddy", "Iyer", "Gupta", "Nair", "Singh",
              "Kapoor", "Menon", "Joshi", "Verma", "Rao", "Das", "Chatterjee"]
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Salaried", "Salaried", "Business Owner"]
CITIES = ["Mumbai", "Pune", "Nashik", "Bengaluru", "Delhi", "Hyderabad", "Chennai"]
LOAN_PURPOSES = ["Home Purchase", "Home Renovation", "Vehicle Purchase",
                 "Education", "Business Expansion", "Personal / Medical"]


def make_applicant(i: int) -> dict:
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    age = random.randint(23, 58)
    employment_type = random.choice(EMPLOYMENT_TYPES)
    monthly_income = random.randint(28000, 250000)
    existing_emi = round(monthly_income * random.uniform(0.0, 0.45), 2)
    credit_score = random.randint(560, 830)
    requested_loan_amount = random.choice([300000, 500000, 800000, 1200000,
                                            2000000, 3500000, 5000000, 7500000])
    tenure_years = random.choice([3, 5, 7, 10, 15, 20])
    existing_loans_count = random.randint(0, 3)
    years_at_current_job = round(random.uniform(0.5, 20), 1)
    collateral_value = requested_loan_amount * random.uniform(0.6, 1.4) \
        if random.random() > 0.3 else 0
    bank_default_history = random.choice([False, False, False, False, True])

    dti_ratio = round((existing_emi + estimate_new_emi(requested_loan_amount, tenure_years))
                       / monthly_income, 3)

    return {
        "applicant_id": f"APP-{1000 + i}",
        "name": name,
        "age": age,
        "city": random.choice(CITIES),
        "employment_type": employment_type,
        "years_at_current_job": years_at_current_job,
        "monthly_income": monthly_income,
        "existing_monthly_emi": existing_emi,
        "existing_loans_count": existing_loans_count,
        "credit_score": credit_score,
        "bank_default_history": bank_default_history,
        "requested_loan_amount": requested_loan_amount,
        "loan_purpose": random.choice(LOAN_PURPOSES),
        "tenure_years": tenure_years,
        "collateral_value": round(collateral_value, 2),
        "estimated_dti_ratio": dti_ratio,
    }


def estimate_new_emi(principal, tenure_years, annual_rate=0.10):
    r = annual_rate / 12
    n = tenure_years * 12
    if r == 0:
        return principal / n
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return emi


def generate_applicants(n=40):
    applicants = [make_applicant(i) for i in range(n)]
    df = pd.DataFrame(applicants)
    df.to_csv(APPLICATIONS_DIR / "applicants.csv", index=False)

    for a in applicants:
        with open(APPLICATIONS_DIR / f"{a['applicant_id']}.json", "w") as f:
            json.dump(a, f, indent=2)

    print(f"Generated {n} synthetic applicant records -> {APPLICATIONS_DIR}")
    return df


def _pdf_doc(path, title):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceAfter=14))
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    story = [Paragraph(title, styles["H1c"]), Spacer(1, 6)]
    return doc, story, styles


def generate_loan_policy_pdf():
    path = POLICY_DIR / "loan_eligibility_policy.pdf"
    doc, story, styles = _pdf_doc(path, "Bank Loan Eligibility &amp; Underwriting Policy")

    sections = [
        ("1. Purpose",
         "This document defines the eligibility criteria, debt ratios, credit score "
         "bands, and approval authority levels used to assess retail loan applications "
         "(home, vehicle, personal, and business loans)."),
        ("2. Minimum Eligibility Criteria",
         "Applicants must be between 21 and 60 years of age at loan maturity, have a "
         "minimum monthly income of INR 25,000, and a minimum of 1 year of continuous "
         "employment or 2 years of business vintage for self-employed applicants."),
        ("3. Credit Score Bands",
         "Credit Score 750+: Excellent - eligible for best interest rate and fast-track "
         "processing. 700-749: Good - standard processing. 650-699: Fair - additional "
         "documentation and manual underwriting required. Below 650: Poor - loan is "
         "declined unless strong collateral (>= 150% of loan value) is provided and "
         "approved by senior underwriter."),
        ("4. Debt-to-Income (DTI) Ratio Rules",
         "DTI ratio is computed as (existing EMIs + proposed EMI) / monthly income. "
         "DTI <= 0.35: Low risk, eligible for standard approval. DTI above 0.35 and up to 0.50: "
         "Medium risk, requires risk-agent review and may require a co-applicant or "
         "reduced loan amount. DTI > 0.50: High risk, application must be routed to "
         "manual/human underwriter review and is generally declined unless offset by "
         "high-value collateral."),
        ("5. Collateral and Loan-to-Value (LTV)",
         "For secured loans, the Loan-to-Value ratio must not exceed 80% of the "
         "appraised collateral value. Loans with no collateral above INR 1,000,000 "
         "require mandatory human approval regardless of credit score."),
        ("6. Default History",
         "Any applicant with a prior bank default or written-off loan in the last 5 "
         "years is automatically flagged High Risk and requires senior underwriter "
         "sign-off before any approval can be issued."),
        ("7. Approval Authority Matrix",
         "Loans up to INR 1,000,000 with Low Risk classification may be auto-approved "
         "by the AI recommendation system subject to a human reviewer's final sign-off. "
         "Loans between INR 1,000,000 and INR 5,000,000, or classified Medium Risk, "
         "require review by a branch credit manager. Loans above INR 5,000,000, or "
         "classified High Risk, require committee-level human approval. In all cases, "
         "final disbursal requires an explicit human approval step; the AI system may "
         "only recommend, never auto-disburse."),
        ("8. Documentation Requirements",
         "Salaried applicants must submit the last 3 months' salary slips, 6 months' "
         "bank statements, and Form 16. Self-employed applicants must submit 2 years of "
         "ITR, audited financials, and 12 months' bank statements."),
        ("9. Rejection &amp; Re-application",
         "Rejected applicants may reapply after 90 days. The rejection reason must be "
         "clearly communicated, referencing the specific policy clause that was not met."),
    ]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    print(f"Generated policy PDF -> {path}")


def generate_risk_guidelines_pdf():
    path = POLICY_DIR / "risk_assessment_guidelines.pdf"
    doc, story, styles = _pdf_doc(path, "Credit Risk Assessment Guidelines")

    sections = [
        ("1. Risk Classification Framework",
         "Every application must be classified into one of three risk bands: LOW, "
         "MEDIUM, or HIGH. The classification drives the routing decision in the "
         "agentic workflow (auto-recommend path vs. mandatory manual review path)."),
        ("2. LOW Risk Criteria (all must be true)",
         "Credit score >= 700 AND DTI ratio <= 0.35 AND no prior default history AND "
         "employment tenure >= 1 year. LOW risk applications may proceed on the "
         "fast-track / expedited router path."),
        ("3. MEDIUM Risk Criteria",
         "Credit score between 650 and 699, OR DTI ratio above 0.35 and up to 0.50, OR "
         "employment tenure below 1 year, with no default history. MEDIUM "
         "risk applications follow the standard sequential path with full risk-agent "
         "analysis and require branch-manager human approval."),
        ("4. HIGH Risk Criteria (any one triggers HIGH)",
         "Credit score < 650, OR DTI ratio > 0.50, OR any prior default history, OR "
         "requested amount with zero collateral exceeds INR 1,000,000. HIGH risk "
         "applications are always routed to mandatory senior human review; the AI "
         "system must not issue an auto-approval recommendation."),
        ("5. Escalation Triggers",
         "If any single financial data field is missing or inconsistent (e.g., income "
         "reported as zero, negative EMI), the workflow must halt automated processing "
         "and escalate directly to a human analyst rather than guessing values."),
        ("6. Explainability Requirement",
         "Every risk classification produced by the AI system must be accompanied by a "
         "plain-language explanation referencing the specific policy clauses and "
         "financial ratios that drove the decision, so a human reviewer can validate "
         "the reasoning quickly."),
    ]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    print(f"Generated risk guidelines PDF -> {path}")


def generate_sample_application_pdf(applicant: dict):
    path = APPLICATIONS_DIR / f"{applicant['applicant_id']}_form.pdf"
    doc, story, styles = _pdf_doc(path, f"Loan Application Form - {applicant['applicant_id']}")

    rows = [["Field", "Value"]] + [
        ["Applicant Name", applicant["name"]],
        ["Age", str(applicant["age"])],
        ["City", applicant["city"]],
        ["Employment Type", applicant["employment_type"]],
        ["Years at Current Job", str(applicant["years_at_current_job"])],
        ["Monthly Income (INR)", f"{applicant['monthly_income']:,}"],
        ["Existing Monthly EMI (INR)", f"{applicant['existing_monthly_emi']:,}"],
        ["Existing Loans Count", str(applicant["existing_loans_count"])],
        ["Credit Score", str(applicant["credit_score"])],
        ["Prior Default History", str(applicant["bank_default_history"])],
        ["Requested Loan Amount (INR)", f"{applicant['requested_loan_amount']:,}"],
        ["Loan Purpose", applicant["loan_purpose"]],
        ["Tenure (Years)", str(applicant["tenure_years"])],
        ["Collateral Value (INR)", f"{applicant['collateral_value']:,}"],
    ]
    table = Table(rows, colWidths=[7 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    doc.build(story)


def main():
    df = generate_applicants(n=40)
    generate_loan_policy_pdf()
    generate_risk_guidelines_pdf()
    # One filled application-form PDF per applicant: these are the documents the
    # Document Intake Agent reads (upload any of them in New Assessment).
    for _, row in df.iterrows():
        generate_sample_application_pdf(row.to_dict())
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
