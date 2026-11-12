"""Tools the Research agent can call. They read the company's knowledge base and order data."""
import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"


def _load(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def search_knowledge_base(query: str, limit: int = 3):
    kb = _load("knowledge_base.json")
    words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for art in kb:
        haystack = (art["title"] + " " + " ".join(art["tags"]) + " " + art["text"]).lower()
        hay_words = set(re.findall(r"\w+", haystack))
        score = len(words & hay_words)
        for tag in art["tags"]:
            if tag in query.lower():
                score += 3
        if score:
            scored.append((score, art))
    scored.sort(key=lambda x: -x[0])
    return [
        {"id": a["id"], "title": a["title"], "text": a["text"]} for _, a in scored[:limit]
    ] or [{"note": "No matching policy article found."}]


def lookup_order(order_id: str):
    data = _load("orders.json")
    order = data["orders"].get(order_id.strip().upper())
    if not order:
        return {"error": f"Order {order_id} not found."}
    return {"order_id": order_id.strip().upper(), **order}


def lookup_customer(email: str):
    data = _load("orders.json")
    cust = data["customers"].get(email.strip().lower())
    if not cust:
        return {"error": f"No customer account found for {email}."}
    return {"email": email.strip().lower(), **cust}


TOOL_FUNCS = {
    "search_knowledge_base": lambda a: search_knowledge_base(a.get("query", "")),
    "lookup_order": lambda a: lookup_order(a.get("order_id", "")),
    "lookup_customer": lambda a: lookup_customer(a.get("email", "")),
}

TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the company policy knowledge base. Returns the most relevant policy articles.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords describing the issue"}},
            "required": ["query"],
        },
    },
    {
        "name": "lookup_order",
        "description": "Look up an order by id (e.g. A-1001): item, price, status, dates, carrier.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "lookup_customer",
        "description": "Look up a customer by email: name, loyalty tier, country and their order ids.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
]


def dispatch(name: str, args: dict):
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"error": f"Unknown tool {name}"}
    try:
        return fn(args)
    except Exception as e:  # tool errors go back to the model instead of crashing the run
        return {"error": str(e)}
