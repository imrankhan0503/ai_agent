"""Tests the OpenAI-compatible adapter end to end with a scripted fake OpenAI client."""
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import llm  # noqa: E402
import pipeline  # noqa: E402

TICKET = {"ticket_id": "T-Y", "customer_email": "anna.lindqvist@example.com", "subject": "Refund", "body": "Refund please"}

TRIAGE = dict(category="refund_return", priority="medium", sentiment="neutral", language="English",
              summary="Wants refund", legal_or_safety_risk=False, prompt_injection_suspected=False)
RESEARCH = dict(findings="ok", policies_applied=["KB-001"], refund_eligible=True, recommended_refund_eur=14.9,
                customer_verified=True, confidence=0.95)
DRAFT = dict(reply="Hi", proposed_action="refund", action_amount_eur=14.9, internal_note="")
QA = dict(verdict="approve", issues=[], score=0.95)


def call(name, args, cid):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


class FakeOpenAI:
    """Script: list of lists of tool calls, one list per model response."""

    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.seen.append(kw["messages"])
        calls = self.script.pop(0)
        msg = NS(content=None, tool_calls=calls)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def make(script):
    c = llm.OpenAICompatClient.__new__(llm.OpenAICompatClient)
    c._client = FakeOpenAI(script)
    c.messages = NS(create=c._create)
    return c


def test_full_pipeline_through_adapter_with_tool_loop():
    script = [
        [call("submit_triage", TRIAGE, "c1")],
        [call("lookup_customer", {"email": TICKET["customer_email"]}, "c2")],  # research uses a tool first
        [call("submit_research", RESEARCH, "c3")],
        [call("submit_draft", DRAFT, "c4")],
        [call("submit_qa", QA, "c5")],
    ]
    client = make(script)
    r = pipeline.process_ticket(client, TICKET)
    assert r["status"] == "auto_resolved"
    assert r["tokens_in"] == 50
    # the tool result must have been sent back in OpenAI 'tool' format
    third_call_msgs = client._client.seen[2]
    assert any(m["role"] == "tool" and "Anna" in m["content"] for m in third_call_msgs)


def test_provider_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "x-model")
    assert llm.model_name() == "x-model"
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert llm.has_key()
