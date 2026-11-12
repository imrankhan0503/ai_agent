"""LLM provider layer. The agents talk to a small Anthropic-style interface; this file supplies it.

Provider choice (environment variables or Streamlit secrets):
  LLM_PROVIDER=anthropic (default)  -> needs ANTHROPIC_API_KEY, optional CLAUDE_MODEL
  LLM_PROVIDER=openai               -> any OpenAI-compatible API (Google Gemini, Groq, ...)
                                       needs LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Free options (check each site for current limits and model names):
  Gemini: LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   key from aistudio.google.com
  Groq:   LLM_BASE_URL=https://api.groq.com/openai/v1                              key from console.groq.com
"""
import json
import os
import time
from types import SimpleNamespace as NS

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


def provider():
    return os.getenv("LLM_PROVIDER", "anthropic").lower()


def model_name():
    if provider() == "openai":
        return os.getenv("LLM_MODEL", "gemini-2.5-flash")
    return os.getenv("CLAUDE_MODEL", DEFAULT_ANTHROPIC_MODEL)


def has_key():
    if provider() == "openai":
        return bool(os.getenv("LLM_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def make_client():
    if provider() == "openai":
        return OpenAICompatClient(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        )
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class OpenAICompatClient:
    """Exposes client.messages.create(...) in Anthropic style on top of an OpenAI-compatible chat API."""

    def __init__(self, api_key, base_url):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages = NS(create=self._create)

    @staticmethod
    def _convert_messages(system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if m["role"] == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                else:  # list of tool_result dicts
                    for r in content:
                        out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]})
            else:
                text = "".join(b.text for b in content if b.type == "text")
                calls = [
                    {"id": b.id, "type": "function",
                     "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)}}
                    for b in content if b.type == "tool_use"
                ]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    def _create(self, model, max_tokens, system, tools, messages):
        oai_tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}}
            for t in tools
        ]
        kwargs = dict(model=model, messages=self._convert_messages(system, messages), tools=oai_tools,
                      tool_choice="required", max_tokens=max_tokens)
        for attempt in range(5):  # free tiers rate-limit: back off and retry
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt == 4 or "429" not in str(e) and "rate" not in str(e).lower():
                    raise
                time.sleep(5 * (attempt + 1))
        msg = resp.choices[0].message
        blocks = []
        if msg.content:
            blocks.append(NS(type="text", text=msg.content))
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(NS(type="tool_use", id=tc.id, name=tc.function.name, input=args))
        u = getattr(resp, "usage", None)
        usage = NS(input_tokens=getattr(u, "prompt_tokens", 0) or 0, output_tokens=getattr(u, "completion_tokens", 0) or 0)
        return NS(content=blocks, usage=usage)
