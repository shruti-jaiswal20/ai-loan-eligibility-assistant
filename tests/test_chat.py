"""AI Assistant: rule-based mode, conversation memory, safety, and the LLM tool-calling loop
(driven through the real OpenAI client against a scripted local server)."""
import json
import pytest
import tenacity

from agents import chat_agent, chat_tools, llm_client
from agents.chat_agent import chat_turn, get_history, new_session_id
from workflow.graph import get_state
from tests.fake_openai import FakeOpenAI, tool_call, text


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_client._chat_retry.retry, "wait", tenacity.wait_none())   # keep failure tests fast
    return monkeypatch


def tools_used(r):
    return [c["name"] for c in r["tool_calls"]]


# ---------------------------------------------------------------- rule-based mode

def test_assess_then_followup_uses_memory():
    sid = new_session_id()
    r1 = chat_turn(sid, "Assess APP-1012")
    assert tools_used(r1) == ["assess_applicant"] and "APP-1012" in r1["reply"] and "Risk" in r1["reply"]
    assert get_state("APP-1012").get("recommendation")                 # a real workflow run happened
    r2 = chat_turn(sid, "why was it flagged? and its DTI?")           # no ID given -> uses the focus applicant
    assert tools_used(r2) == ["explain_case"]
    assert json.loads(r2["tool_calls"][0]["args"])["applicant_id"] == "APP-1012"
    assert "APP-1012" in r2["reply"] and "Why" in r2["reply"]
    assert len(get_history(sid)) >= 6                                   # memory persisted across turns


def test_policy_question_is_answered_from_rag_with_sources():
    r = chat_turn(new_session_id(), "What DTI ratio needs manual review?")
    assert tools_used(r) == ["search_policy"]
    assert ".pdf" in r["reply"] and "DTI" in r["reply"]


def test_portfolio_stats_match_dashboard_numbers():
    from utils.portfolio import load_applicants_df, score_applicants, summarize
    expected = summarize(score_applicants(load_applicants_df()))
    r = chat_turn(new_session_id(), "How many applications are high risk?")
    assert tools_used(r) == ["portfolio_stats"]
    assert f"HIGH {expected['by_risk']['HIGH']}" in r["reply"] and str(expected["total"]) in r["reply"]


def test_list_pending_cases_shows_assessed_case():
    sid = new_session_id()
    chat_turn(sid, "assess APP-1013")
    r = chat_turn(sid, "show pending cases")
    assert tools_used(r) == ["list_cases"] and "APP-1013" in r["reply"] and "awaiting human decision" in r["reply"]


@pytest.mark.parametrize("msg", ["approve APP-1014", "Please reject it and disburse the loan", "can you sanction APP-1015?"])
def test_assistant_never_decides_and_makes_no_tool_call(msg):
    sid = new_session_id()
    chat_turn(sid, "assess APP-1014")
    before = get_state("APP-1014").get("human_decision")
    r = chat_turn(sid, msg)
    assert r["tool_calls"] == [] and "human" in r["reply"].lower() and "can't" in r["reply"]
    assert get_state("APP-1014").get("human_decision") == before is None   # nothing was decided


def test_explain_unassessed_case_says_to_assess_first():
    r = chat_turn(new_session_id(), "explain APP-1039")
    assert "not been assessed" in r["reply"]


def test_unknown_applicant_and_bad_ids_are_rejected():
    assert "No applicant" in chat_turn(new_session_id(), "assess APP-9999")["reply"]
    for bad in ["../../etc/passwd", "APP-1;rm", "app-" + "1" * 40, None, 123]:
        assert "error" in chat_tools.execute_tool("assess_applicant", {"applicant_id": bad})


def test_unknown_tool_name_is_refused():
    out = chat_tools.execute_tool("approve_loan", {"applicant_id": "APP-1001"})
    assert "Unknown tool" in out["error"]


def test_gibberish_gets_help_not_a_crash():
    assert "assess" in chat_turn(new_session_id(), "zzzz qqq")["reply"].lower()
    assert chat_turn(new_session_id(), "   ")["reply"] == ""


# ------------------------------------------------------------- LLM tool-calling loop

def test_llm_chooses_tool_then_answers(llm_env):
    with FakeOpenAI([tool_call("search_policy", {"question": "high DTI"}),
                     text("A DTI above 50% is HIGH risk (loan_eligibility_policy.pdf).")]) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        r = chat_turn(new_session_id(), "what happens with a very high DTI?")
    assert r["mode"] == "llm" and tools_used(r) == ["search_policy"]
    assert r["reply"].startswith("A DTI above 50%")
    assert len(srv.requests) == 2
    assert srv.requests[0]["tools"] and {t["function"]["name"] for t in srv.requests[0]["tools"]} == set(chat_tools.TOOLS)
    assert srv.requests[0]["messages"][0]["role"] == "system"
    tool_msgs = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "passages" in tool_msgs[0]["content"]          # real RAG output fed back to the model
    for req in srv.requests:                                             # internal bookkeeping never leaks to the API
        assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id", "name"} for m in req["messages"])


def test_llm_cannot_approve_even_if_it_tries(llm_env):
    with FakeOpenAI([tool_call("approve_loan", {"applicant_id": "APP-1001"}),
                     text("I can't approve loans; please use the Human Review screen.")]) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        r = chat_turn(new_session_id(), "approve APP-1001")
    tool_result = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "Unknown tool" in tool_result
    assert get_state("APP-1001").get("human_decision") is None


def test_llm_malformed_tool_arguments_do_not_crash(llm_env):
    bad = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "assess_applicant", "arguments": "{not json"}}]}
    with FakeOpenAI([bad, text("Sorry, which applicant?")]) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        r = chat_turn(new_session_id(), "assess someone")
    assert r["reply"] == "Sorry, which applicant?"


def test_llm_tool_loop_is_capped(llm_env):
    with FakeOpenAI([tool_call("portfolio_stats", {}, f"c{i}") for i in range(20)]) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        r = chat_turn(new_session_id(), "stats forever")
    assert len(srv.requests) == chat_agent.MAX_STEPS and "too many steps" in r["reply"]


def test_llm_outage_falls_back_to_rule_based(llm_env):
    with FakeOpenAI([], status=500) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        r = chat_turn(new_session_id(), "What is the minimum credit score?")
    assert r["mode"] == "fallback" and tools_used(r) == ["search_policy"] and ".pdf" in r["reply"]
    assert "unavailable" in (r["note"] or "")
    assert len(srv.requests) >= 3                                        # tenacity retried before giving up


def test_generate_degrades_to_rule_based_text_on_outage(llm_env):
    with FakeOpenAI([], status=500) as srv:
        llm_env.setenv("OPENAI_BASE_URL", srv.url)
        llm_env.setattr(llm_client._call_with_retry.retry, "wait", tenacity.wait_none())
        assert llm_client.generate("s", "u", mock_fn=lambda: "RULE-BASED") == "RULE-BASED"
