"""Streamlit AppTest: executes the real app script headlessly and simulates clicks/typing."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["dashboard", "new_assessment", "assistant", "applications", "assessments",
         "agent_execution", "documents", "reports", "settings"]


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page):
    at = app()
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception


def test_assistant_chat_round_trip_and_open_case_button():
    at = app()
    at.session_state["page"] = "assistant"
    at.run()
    at.chat_input[0].set_value("Assess APP-1016").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "APP-1016" in text and "AI recommendation" in text
    assert any(b.label == "Open human review" for b in at.button)

    # follow-up resolves through memory, still no exceptions
    at.chat_input[0].set_value("why is its risk like that?").run()
    assert not at.exception and "Why" in " ".join(m.value for m in at.markdown)

    # the button hands the chat-assessed case to the real Human Review screen
    [b for b in at.button if b.label == "Open human review"][-1].click().run()
    assert at.session_state["page"] == "new_assessment" and at.session_state["wizard_step"] == "review"
    assert not at.exception
    [b for b in at.button if "Approve" in b.label][0].click().run()          # a HUMAN approves
    assert at.session_state["current_state"]["human_decision"] == "APPROVED"


def test_assistant_refuses_to_approve_in_the_ui():
    at = app()
    at.session_state["page"] = "assistant"
    at.run()
    at.chat_input[0].set_value("approve APP-1001").run()
    assert "can't approve" in " ".join(m.value for m in at.markdown)


def test_wizard_end_to_end_with_human_decision():
    at = app()
    applicant = json.loads((ROOT / "data/applications/APP-1020.json").read_text())
    at.session_state["page"] = "new_assessment"
    at.session_state["wizard_step"] = "progress"
    at.session_state["current_applicant"] = applicant
    at.session_state["current_document"] = None
    at.session_state["progress_started"] = False
    at.run()
    assert not at.exception and at.session_state["wizard_step"] == "results"
    at.session_state["wizard_step"] = "review"
    at.run()
    [b for b in at.button if "Reject" in b.label][0].click().run()
    assert at.session_state["current_state"]["human_decision"] == "REJECTED"
    at.session_state["page"] = "reports"
    at.run()
    assert not at.exception


def test_wizard_with_uploaded_document_shows_agent_findings():
    at = app()
    applicant = json.loads((ROOT / "data/applications/APP-1001.json").read_text())
    applicant["applicant_id"] = "APP-UI-DOC"
    applicant["monthly_income"] += 90000                              # disagrees with the PDF
    at.session_state["page"] = "new_assessment"
    at.session_state["wizard_step"] = "progress"
    at.session_state["current_applicant"] = applicant
    at.session_state["current_document"] = {"path": str(ROOT / "data/applications/APP-1001_form.pdf"),
                                            "filename": "APP-1001_form.pdf", "storage_uri": "file://x"}
    at.session_state["progress_started"] = False
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "Escalated to manual review" in text and "monthly_income" in text
