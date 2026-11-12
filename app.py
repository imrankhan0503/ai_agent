"""SupportDesk AI - multi-agent customer support triage and resolution. Run: streamlit run app.py"""
import csv
import io
import json
import os

import pandas as pd
import streamlit as st

import auth
import db
import llm
import pipeline

st.set_page_config(page_title="SupportDesk AI", page_icon="🎧", layout="wide")

# Allow configuration through Streamlit Cloud secrets as well as environment variables.
try:
    for key in ("ANTHROPIC_API_KEY", "APP_USERS", "CLAUDE_MODEL", "LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

STATUS_LABEL = {
    "auto_resolved": "Auto-resolved",
    "needs_review": "Needs review",
    "escalated": "Escalated",
    "human_approved": "Approved by human",
    "human_rejected": "Rejected by human",
}


def get_client():
    return llm.make_client()


# ---------- login ----------
def login_screen():
    st.title("🎧 SupportDesk AI")
    st.caption("Multi-agent customer support: triage, research, resolution and QA.")
    if not auth.load_users():
        st.error("No users configured. Set APP_USERS (see README, use make_user.py).")
        st.stop()
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            user = auth.authenticate(u, p)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Wrong username or password.")
    st.stop()


if "user" not in st.session_state:
    login_screen()

user = st.session_state.user
st.sidebar.title("🎧 SupportDesk AI")
st.sidebar.write(f"Logged in as **{user['username']}** ({user['role']})")
pages = ["Process tickets", "Review queue", "Dashboard", "Audit trail"]
page = st.sidebar.radio("Navigate", pages)
if st.sidebar.button("Log out"):
    del st.session_state.user
    st.rerun()
if not llm.has_key():
    st.sidebar.error("API key is not set (ANTHROPIC_API_KEY, or LLM_API_KEY when LLM_PROVIDER=openai).")
else:
    st.sidebar.caption(f"Model: {llm.model_name()}")


def run_batch(tickets):
    client = get_client()
    bar = st.progress(0.0, text="Starting agents...")
    done = []
    for i, t in enumerate(tickets):
        bar.progress(i / len(tickets), text=f"Processing {t['ticket_id']} ({i + 1}/{len(tickets)})")
        res = pipeline.process_ticket(client, t)
        db.save_result(t, res, user["username"])
        done.append(res)
    bar.progress(1.0, text="Done")
    return done


# ---------- pages ----------
if page == "Process tickets":
    st.header("Process tickets")
    tab1, tab2, tab3 = st.tabs(["Single ticket", "Upload CSV", "Demo data"])
    with tab1:
        with st.form("single"):
            tid = st.text_input("Ticket id", value=f"T-{len(db.all_tickets()) + 100}")
            email = st.text_input("Customer email")
            subj = st.text_input("Subject")
            body = st.text_area("Message", height=150)
            go = st.form_submit_button("Run agents")
        if go and email and body:
            res = run_batch([{"ticket_id": tid, "customer_email": email, "subject": subj, "body": body}])[0]
            st.success(f"{STATUS_LABEL[res['status']]}: {', '.join(res['reasons'])}")
            st.text_area("Draft reply", res["reply"], height=200)
    with tab2:
        st.caption("CSV columns: ticket_id, customer_email, subject, body")
        up = st.file_uploader("CSV file", type="csv")
        if up and st.button("Process uploaded tickets"):
            rows = list(csv.DictReader(io.StringIO(up.getvalue().decode("utf-8"))))
            missing = {"ticket_id", "customer_email", "subject", "body"} - set(rows[0].keys() if rows else [])
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            else:
                run_batch(rows)
                st.success(f"Processed {len(rows)} tickets. See the Review queue and Dashboard.")
    with tab3:
        st.caption("12 realistic tickets incl. an angry customer, a legal/safety threat, GDPR and a prompt-injection attempt.")
        sample = os.path.join(os.path.dirname(__file__), "data", "sample_tickets.csv")
        rows = list(csv.DictReader(open(sample, encoding="utf-8")))
        st.dataframe(pd.DataFrame(rows), width="stretch")
        if st.button("Run agents on demo tickets"):
            run_batch(rows)
            st.success("Done. Open the Dashboard.")

elif page == "Review queue":
    st.header("Human review queue")
    queue = [t for t in db.all_tickets() if t["status"] in ("needs_review", "escalated")]
    if not queue:
        st.info("Nothing waiting for review.")
    for t in queue:
        icon = "🚨" if t["status"] == "escalated" else "👀"
        with st.expander(f"{icon} {t['ticket_id']} · {t['subject']} · {t['priority']} priority"):
            st.markdown(f"**From:** {t['customer_email']}  \n**Summary:** {t['summary']}")
            st.markdown(f"**Why it is here:** {', '.join(json.loads(t['reasons']))}")
            st.text_area("Customer message", t["body"], height=90, disabled=True, key=f"b{t['ticket_id']}")
            edited = st.text_area("Reply (editable)", t["reply"], height=180, key=f"r{t['ticket_id']}")
            st.caption(f"Proposed action: {t['action']} {t['amount']:.2f} EUR")
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("Approve & send", key=f"a{t['ticket_id']}"):
                db.review(t["ticket_id"], user["username"], "human_approved", edited)
                st.rerun()
            if c2.button("Reject", key=f"x{t['ticket_id']}"):
                db.review(t["ticket_id"], user["username"], "human_rejected", edited)
                st.rerun()
    st.caption("Sending is simulated: connect an email/helpdesk API in pipeline hooks for production.")

elif page == "Dashboard":
    st.header("Business impact")
    rows = db.all_tickets()
    if not rows:
        st.info("No tickets processed yet. Run the demo data first.")
        st.stop()
    df = pd.DataFrame(rows)
    n = len(df)
    auto = (df.status == "auto_resolved").sum()
    review_n = df.status.isin(["needs_review", "human_approved", "human_rejected"]).sum()
    esc = (df.status == "escalated").sum()
    cost = sum(pipeline.cost_eur(r["tokens_in"], r["tokens_out"]) for r in rows)
    saved_min = auto * pipeline.MANUAL_MINUTES_PER_TICKET + (n - auto) * pipeline.MANUAL_MINUTES_PER_TICKET * 0.5
    c = st.columns(5)
    c[0].metric("Tickets processed", n)
    c[1].metric("Auto-resolved", f"{auto / n:.0%}")
    c[2].metric("Avg. agent time", f"{df.seconds.mean():.1f} s")
    c[3].metric("Est. time saved", f"{saved_min / 60:.1f} h")
    c[4].metric("AI cost / ticket", f"{cost / n:.3f} EUR")
    st.caption(
        f"Assumes {pipeline.MANUAL_MINUTES_PER_TICKET:.0f} min manual handling per ticket; "
        "humans still reviewing a ticket count as 50% saved (drafting is done). Configurable via env vars."
    )
    a, b = st.columns(2)
    a.subheader("Outcome")
    a.bar_chart(df.status.map(lambda s: STATUS_LABEL.get(s, s)).value_counts())
    b.subheader("Category")
    b.bar_chart(df.category.value_counts())
    st.subheader("All tickets")
    st.dataframe(
        df[["ticket_id", "subject", "category", "priority", "sentiment", "status", "action", "amount", "qa_score", "seconds"]],
        width="stretch",
    )
    if user["role"] == "admin" and st.button("Delete all tickets (admin)"):
        db.clear_all()
        st.rerun()

elif page == "Audit trail":
    st.header("Audit trail")
    st.caption("Every agent step, tool call and decision reason is stored for each ticket.")
    rows = db.all_tickets()
    if not rows:
        st.info("No tickets yet.")
        st.stop()
    pick = st.selectbox("Ticket", [r["ticket_id"] for r in rows])
    t = db.get_ticket(pick)
    st.markdown(f"**Status:** {STATUS_LABEL.get(t['status'], t['status'])} · **Processed by:** {t['processed_by']} · "
                f"**Reviewed by:** {t['reviewed_by'] or '-'} · **Tokens:** {t['tokens_in']} in / {t['tokens_out']} out")
    st.markdown(f"**Decision reasons:** {', '.join(json.loads(t['reasons']))}")
    for step in json.loads(t["trace"]):
        if step["type"] == "tool":
            st.markdown(f"🔧 **{step['agent']}** called `{step['tool']}`")
            st.json({"input": step["input"], "output": step["output"]}, expanded=False)
        elif step["type"] == "result":
            st.markdown(f"✅ **{step['agent']} agent** result")
            st.json(step["data"], expanded=False)
        else:
            st.error(f"{step['agent']}: {step['data']}")
