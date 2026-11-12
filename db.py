"""SQLite persistence: processed tickets, human decisions and the audit trail."""
import json
import os
import sqlite3
import time

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "supportdesk.db"))


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY, customer_email TEXT, subject TEXT, body TEXT,
            created_at REAL, status TEXT, category TEXT, priority TEXT, sentiment TEXT, summary TEXT,
            reply TEXT, action TEXT, amount REAL, qa_score REAL, confidence REAL, reasons TEXT,
            trace TEXT, tokens_in INTEGER, tokens_out INTEGER, seconds REAL,
            reviewed_by TEXT, reviewed_at REAL, final_reply TEXT, processed_by TEXT)"""
    )
    return c


def save_result(ticket, result, user):
    with conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?)""",
            (
                result["ticket_id"], ticket["customer_email"], ticket["subject"], ticket["body"], time.time(),
                result["status"], result["category"], result["priority"], result["sentiment"], result["summary"],
                result["reply"], result["action"], result["amount"], result["qa_score"], result["confidence"],
                json.dumps(result["reasons"], ensure_ascii=False), json.dumps(result["trace"], ensure_ascii=False),
                result["tokens_in"], result["tokens_out"], result["seconds"], user,
            ),
        )


def all_tickets():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM tickets ORDER BY created_at DESC")]


def get_ticket(ticket_id):
    with conn() as c:
        r = c.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return dict(r) if r else None


def review(ticket_id, user, decision, final_reply):
    """decision: 'human_approved' or 'human_rejected'."""
    with conn() as c:
        c.execute(
            "UPDATE tickets SET status=?, reviewed_by=?, reviewed_at=?, final_reply=? WHERE ticket_id=?",
            (decision, user, time.time(), final_reply, ticket_id),
        )


def clear_all():
    with conn() as c:
        c.execute("DELETE FROM tickets")
