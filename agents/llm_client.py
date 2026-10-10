"""
Central LLM client used by every agent.

- If OPENAI_API_KEY is set, calls the OpenAI Chat Completions API
  (any OpenAI-compatible endpoint works via OPENAI_BASE_URL).
- Otherwise runs in MOCK mode: deterministic, rule-based text so the whole
  workflow is demoable with no API key.

Reliability (RFP: "error handling and retry mechanisms"):
- tenacity retries transient API errors with exponential backoff (3 attempts);
  the SDK's own retries are disabled so retries are not layered twice.
- a 30 s request timeout (the SDK default is 10 minutes, which would freeze the UI).
- if every attempt fails, generate() degrades to the agent's rule-based output
  instead of crashing the workflow.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger("ai_loan_assistant.llm")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT", "30"))


def use_mock() -> bool:
    """Evaluated at call time so the mode follows the environment."""
    return not bool(os.getenv("OPENAI_API_KEY"))


USE_MOCK = use_mock()   # snapshot used only for display (sidebar / settings)


class LLMCallError(Exception):
    """Transient API failure (retried)."""


class LLMUnavailable(Exception):
    """No API key, or all retries failed (not retried)."""


def _client():
    from openai import OpenAI
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS)


def _real_openai_call(system_prompt: str, user_prompt: str) -> str:
    from openai import APIError, APITimeoutError, RateLimitError
    model = os.getenv("OPENAI_MODEL", MODEL_NAME)
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except (APIError, APITimeoutError, RateLimitError) as e:
        raise LLMCallError(str(e)) from e


@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(LLMCallError))
def _call_with_retry(system_prompt: str, user_prompt: str) -> str:
    return _real_openai_call(system_prompt, user_prompt)


def generate(system_prompt: str, user_prompt: str, mock_fn=None) -> str:
    """Unified entry point for text generation. `mock_fn` produces the rule-based
    output used in mock mode and as the fallback if the API is unavailable."""
    fallback = (mock_fn() if mock_fn else "[no rule-based fallback available]")
    if use_mock():
        logger.info("LLM MOCK mode active (no OPENAI_API_KEY) - using rule-based output.")
        return fallback
    try:
        out = _call_with_retry(system_prompt, user_prompt)
        if out:
            return out
        logger.warning("LLM returned empty content; using rule-based output.")
    except Exception as e:  # noqa: BLE001 - last-resort safety net
        logger.error("LLM call failed after retries (%s); using rule-based output.", e)
    return fallback


# ---- tool-calling (used by the AI Assistant) --------------------------------------

_API_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}


def to_api_messages(messages: list[dict]) -> list[dict]:
    """Drop our internal bookkeeping keys (e.g. 'meta') — the API rejects unknown fields."""
    return [{k: v for k, v in m.items() if k in _API_KEYS} for m in messages]


def _real_chat(messages: list[dict], tools: list[dict]) -> dict:
    from openai import APIError, APITimeoutError, RateLimitError
    model = os.getenv("OPENAI_MODEL", MODEL_NAME)
    try:
        resp = _client().chat.completions.create(
            model=model, messages=to_api_messages(messages), tools=tools,
            tool_choice="auto", temperature=0.2,
        )
    except (APIError, APITimeoutError, RateLimitError) as e:
        raise LLMCallError(str(e)) from e
    m = resp.choices[0].message
    out: dict = {"role": "assistant", "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                             for tc in m.tool_calls]
    return out


@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(LLMCallError))
def _chat_retry(messages: list[dict], tools: list[dict]) -> dict:
    return _real_chat(messages, tools)


def chat_with_tools(messages: list[dict], tools: list[dict]) -> dict:
    """One tool-calling turn. Returns an OpenAI-style assistant message.
    Raises LLMUnavailable when there is no key or the API keeps failing."""
    if use_mock():
        raise LLMUnavailable("no OPENAI_API_KEY")
    try:
        return _chat_retry(messages, tools)
    except Exception as e:  # noqa: BLE001
        raise LLMUnavailable(str(e)) from e
