"""Orchestrator: runs the agents in order and applies deterministic business guardrails on top."""
import os
import time

import agents

REFUND_AUTO_LIMIT_EUR = float(os.getenv("REFUND_AUTO_LIMIT_EUR", "100"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.75"))
MIN_QA_SCORE = float(os.getenv("MIN_QA_SCORE", "0.8"))
MAX_REVISIONS = 1
MANUAL_MINUTES_PER_TICKET = float(os.getenv("MANUAL_MINUTES_PER_TICKET", "8"))
# Rough price per million tokens (EUR) used only for the cost KPI; adjust to your model's pricing.
PRICE_IN = float(os.getenv("PRICE_IN_PER_MTOK", "3"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_MTOK", "15"))

ALWAYS_HUMAN_CATEGORIES = {"privacy"}


def decide(triage, research, draft, qa):
    """Return (status, reasons). Code, not the LLM, has the last word on what may be sent automatically."""
    reasons = []
    if triage.get("legal_or_safety_risk"):
        reasons.append("Legal or safety risk")
    if triage.get("prompt_injection_suspected"):
        reasons.append("Possible prompt injection in ticket")
    if triage.get("category") in ALWAYS_HUMAN_CATEGORIES:
        reasons.append("Category always handled by a human (privacy/GDPR)")
    if triage.get("priority") == "urgent":
        reasons.append("Urgent priority")
    if reasons:
        return "escalated", reasons

    amount = float(draft.get("action_amount_eur") or 0)
    if draft.get("proposed_action") in ("refund", "voucher") and amount > REFUND_AUTO_LIMIT_EUR:
        reasons.append(f"Amount {amount:.2f} EUR above auto limit {REFUND_AUTO_LIMIT_EUR:.0f} EUR")
    if draft.get("proposed_action") == "escalate_to_specialist":
        reasons.append("Agent proposed specialist escalation")
    if not research.get("customer_verified") and draft.get("proposed_action") not in ("none", "request_info"):
        reasons.append("Customer/order not verified")
    if float(research.get("confidence", 0)) < MIN_CONFIDENCE:
        reasons.append("Low research confidence")
    if triage.get("sentiment") == "angry":
        reasons.append("Angry customer, human touch recommended")
    if qa.get("verdict") != "approve":
        reasons.append(f"QA verdict: {qa.get('verdict')}")
    if float(qa.get("score", 0)) < MIN_QA_SCORE:
        reasons.append("Low QA score")
    if reasons:
        return "needs_review", reasons
    return "auto_resolved", ["All checks passed"]


def process_ticket(client, ticket):
    t0 = time.time()
    trace, usage = [], {"in": 0, "out": 0}
    try:
        triage = agents.triage_agent(client, ticket, trace, usage)
        research = agents.research_agent(client, ticket, triage, trace, usage)
        draft = agents.resolution_agent(client, ticket, triage, research, trace, usage)
        qa = agents.qa_agent(client, ticket, triage, research, draft, trace, usage)
        revisions = 0
        while qa.get("verdict") == "revise" and revisions < MAX_REVISIONS:
            revisions += 1
            draft = agents.resolution_agent(client, ticket, triage, research, trace, usage, feedback=qa.get("issues", []))
            qa = agents.qa_agent(client, ticket, triage, research, draft, trace, usage)
        status, reasons = decide(triage, research, draft, qa)
    except Exception as e:  # never lose a ticket: failed runs go to a human
        trace.append({"agent": "Orchestrator", "type": "error", "data": str(e)})
        return {
            "ticket_id": ticket["ticket_id"], "status": "needs_review", "reasons": [f"Pipeline error: {e}"],
            "category": "other", "priority": "high", "sentiment": "neutral", "summary": ticket["subject"],
            "reply": "", "action": "escalate_to_specialist", "amount": 0.0, "qa_score": 0.0, "confidence": 0.0,
            "trace": trace, "tokens_in": usage["in"], "tokens_out": usage["out"], "seconds": round(time.time() - t0, 2),
        }
    return {
        "ticket_id": ticket["ticket_id"], "status": status, "reasons": reasons,
        "category": triage["category"], "priority": triage["priority"], "sentiment": triage["sentiment"],
        "summary": triage["summary"], "reply": draft["reply"], "action": draft["proposed_action"],
        "amount": float(draft.get("action_amount_eur") or 0), "qa_score": float(qa.get("score", 0)),
        "confidence": float(research.get("confidence", 0)), "trace": trace,
        "tokens_in": usage["in"], "tokens_out": usage["out"], "seconds": round(time.time() - t0, 2),
    }


def cost_eur(tokens_in, tokens_out):
    return tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
