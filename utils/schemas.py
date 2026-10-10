"""
Input validation for loan applications.

Runs as the very first node in the LangGraph workflow (before any agent is
invoked) so malformed input — missing fields, negative income, an
out-of-range credit score — is caught and reported clearly instead of
producing a nonsensical DTI/EMI calculation or crashing an agent deep in
the pipeline. This is the concrete "error handling for incorrect inputs"
mechanism referenced in the project report.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ApplicantInput(BaseModel):
    applicant_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    monthly_income: float = Field(..., gt=0, description="Must be positive")
    existing_monthly_emi: float = Field(0, ge=0)
    requested_loan_amount: float = Field(..., gt=0)
    tenure_years: float = Field(..., gt=0, le=40)
    credit_score: int = Field(..., ge=300, le=900)
    bank_default_history: bool = False
    collateral_value: float = Field(0, ge=0)
    years_at_current_job: float = Field(0, ge=0, le=60)
    employment_type: Literal["Salaried", "Self-Employed", "Business Owner"] = "Salaried"
    loan_purpose: Optional[str] = ""
    city: Optional[str] = "N/A"
    age: Optional[object] = "N/A"

    @field_validator("existing_monthly_emi")
    @classmethod
    def emi_not_absurd(cls, v, info):
        # Sanity check rather than a hard business rule: catches obvious
        # data-entry mistakes (e.g. an EMI larger than any plausible income).
        if v > 10_000_000:
            raise ValueError("existing_monthly_emi is implausibly large")
        return v

    model_config = ConfigDict(extra="allow")


def validate_applicant(applicant: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_error_messages). Never raises."""
    try:
        ApplicantInput(**applicant)
        return True, []
    except ValidationError as e:
        messages = []
        for err in e.errors():
            field = ".".join(str(p) for p in err["loc"]) or "input"
            messages.append(f"{field}: {err['msg']}")
        return False, messages
