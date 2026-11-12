"""Offline tests: a fake LLM client scripts each agent's answer, so no API key is needed."""
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import auth  # noqa: E402
import pipeline  # noqa: E402
import tools  # noqa: E402


class FakeClient:
    """Plays back a script of tool calls: list of (tool_name, input) per agent run, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        name, inp = self.script.pop(0)
        block = NS(type="tool_use", name=name, input=inp, id=f"id{len(self.script)}")
        return NS(content=[block], usage=NS(input_tokens=100, output_tokens=50))


TICKET = {"ticket_id": "T-X", "customer_email": "anna.lindqvist@example.com", "subject": "Refund", "body": "Refund please"}
TRIAGE = dict(category="refund_return", priority="medium", sentiment="neutral", language="English",
              summary="Wants refund", legal_or_safety_risk=False, prompt_injection_suspected=False)
RESEARCH = dict(findings="ok", policies_applied=["KB-001"], refund_eligible=True, recommended_refund_eur=14.9,
                customer_verified=True, confidence=0.95)
DRAFT = dict(reply="Hi", proposed_action="refund", action_amount_eur=14.9, internal_note="")
QA_OK = dict(verdict="approve", issues=[], score=0.95)


def run(triage=TRIAGE, research=RESEARCH, draft=DRAFT, qa=QA_OK, extra=()):
    script = [("submit_triage", triage), ("submit_research", research), ("submit_draft", draft), ("submit_qa", qa), *extra]
    return pipeline.process_ticket(FakeClient(script), TICKET)


def test_auto_resolve_happy_path():
    assert run()["status"] == "auto_resolved"


def test_refund_over_limit_needs_review():
    r = run(draft={**DRAFT, "action_amount_eur": 189.0})
    assert r["status"] == "needs_review"


def test_legal_risk_escalates():
    assert run(triage={**TRIAGE, "legal_or_safety_risk": True})["status"] == "escalated"


def test_privacy_always_human():
    assert run(triage={**TRIAGE, "category": "privacy"})["status"] == "escalated"


def test_injection_escalates():
    assert run(triage={**TRIAGE, "prompt_injection_suspected": True})["status"] == "escalated"


def test_unverified_customer_cannot_get_action():
    assert run(research={**RESEARCH, "customer_verified": False})["status"] == "needs_review"


def test_revision_loop():
    r = run(qa={"verdict": "revise", "issues": ["tone"], "score": 0.6},
            extra=[("submit_draft", DRAFT), ("submit_qa", QA_OK)])
    assert r["status"] == "auto_resolved"


def test_pipeline_error_goes_to_human():
    class Boom:
        messages = NS(create=lambda **k: (_ for _ in ()).throw(RuntimeError("api down")))
    r = pipeline.process_ticket(Boom(), TICKET)
    assert r["status"] == "needs_review" and "api down" in r["reasons"][0]


def test_tools():
    assert tools.lookup_order("a-1001")["price"] == 189.0
    assert "error" in tools.lookup_customer("nobody@x.com")
    assert tools.search_knowledge_base("my item is damaged")[0]["id"] == "KB-002"


def test_auth():
    raw = json.dumps([{"username": "u", "role": "admin", "hash": auth.hash_password("secret")}])
    assert auth.authenticate("u", "secret", raw)["role"] == "admin"
    assert auth.authenticate("u", "wrong", raw) is None
    assert auth.authenticate("nobody", "secret", raw) is None
