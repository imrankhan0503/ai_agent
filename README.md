# SupportDesk AI: multi-agent customer support

## The business challenge
Support teams drown in repetitive tickets. Each ticket costs ~8 minutes of manual work (read, look up the order, check policy, write a reply), answers are inconsistent, and risky tickets (legal threats, safety incidents, GDPR, big refunds) can slip through.

SupportDesk AI automates the full workflow with **four cooperating agents** and keeps humans in control of everything risky.

## Architecture

```
Ticket --> [1 Triage] --> [2 Research] --> [3 Resolution] --> [4 QA/Compliance]
            classify        tools: customer   draft reply in     approve / revise / escalate
            priority        lookup, order     customer's         (1 automatic revision loop)
            risk flags      lookup, policy    language
                            search (KB)
                                                   |
                                  Orchestrator guardrails (code, not LLM)
                                                   |
              auto_resolved  |  needs_review  |  escalated  --> Human review queue
```

| Agent | Role | Tools |
|---|---|---|
| Triage | category, priority, sentiment, language, legal/safety and prompt-injection flags | none |
| Research | verifies customer + order, finds applicable policy | `lookup_customer`, `lookup_order`, `search_knowledge_base` |
| Resolution | writes the reply and proposes an action | none |
| QA / Compliance | checks facts, policy, tone, language; can send the draft back for revision | none |

**Guardrails enforced in code** (`pipeline.decide`): legal/safety risk, GDPR/privacy, suspected prompt injection, urgent priority are always escalated. Refunds over 100 EUR, unverified customers, low confidence, angry customers or a low QA score need human review. Only tickets passing every check are auto-resolved. Pipeline errors never lose a ticket; they go to a human.

**Security:** login with PBKDF2-hashed passwords and roles (admin/agent), ticket text treated as untrusted input, full audit trail of every agent step and tool call, human approver recorded.

## Measurable outcome (Dashboard page)
Auto-resolution rate, average handling time, estimated hours saved, AI cost per ticket and outcome/category breakdown. Assumptions are configurable via environment variables.

## Run it

```bash
pip install -r requirements.txt
python make_user.py admin admin        # prints a JSON user entry
export ANTHROPIC_API_KEY=sk-ant-...
export APP_USERS='[{"username":"admin","role":"admin","hash":"<salt$hash>"}]'
streamlit run app.py
```

Docker:
```bash
docker build -t supportdesk .
docker run -p 8501:8501 -v sd_data:/data -e ANTHROPIC_API_KEY=... -e APP_USERS='[...]' supportdesk
```

Cloud (Streamlit Community Cloud): push to GitHub, deploy `app.py`, add `ANTHROPIC_API_KEY` and `APP_USERS` under *Secrets*. Give the examiner a username and password.

## Free AI provider (no payment)
The agents can run on any OpenAI-compatible API with a free tier. Set these secrets instead of `ANTHROPIC_API_KEY`:

```
LLM_PROVIDER = "openai"
LLM_API_KEY = "<key from aistudio.google.com or console.groq.com>"
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"   # Gemini
LLM_MODEL = "gemini-2.5-flash"                                                # check the current model name
```
Groq: `LLM_BASE_URL = "https://api.groq.com/openai/v1"` and a Groq model that supports tool calling, e.g. `llama-3.3-70b-versatile`. Free tiers are rate limited; the app retries automatically, so a full demo run may take a few minutes.

## Demo script
1. Log in, open **Process tickets > Demo data**, run the 12 tickets.
2. **Dashboard**: see the KPIs.
3. **Review queue**: the overheating charger/lawyer ticket, the GDPR request and the prompt-injection ticket (T-012) are escalated; the angry customer and the 189 EUR headphone refund need review. Edit and approve one.
4. **Audit trail**: open any ticket to see each agent's tool calls and reasoning outputs.

## Tests
`pytest` runs offline with a scripted fake LLM (no API key): guardrails, revision loop, error handling, tools and authentication.

## Limitations and next steps
Data is mock (JSON files); sending replies is simulated. Production would connect a helpdesk API (Zendesk, Freshdesk), a real order system, and run as a background worker polling the inbox.
