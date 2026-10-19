"""LLM wrapper: every model call goes through LLM.chat, which records one LLMCall event per attempt.

Records (L bucket): who assembled the prompt, the origin of every message, sampling parameters,
seed, effort, reasoning text, tool calls, finish reason, tokens (new / cached / output / reasoning),
wall time, retries. `output` tokens include `reasoning` tokens.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from grader.record import AgentRole, LLMCall, MessagePart, RunRecorder, TokenUsage, new_id

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
             openai.RateLimitError)


def to_api(m: MessagePart) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                           for tc in m.tool_calls]
    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id
    return d


@dataclass
class LLMResult:
    llm_call_id: str
    content: str
    reasoning: str | None
    tool_calls: list[dict[str, Any]]   # {"id", "name", "arguments": dict, "parse_error": str | None}
    finish_reason: str | None
    event: LLMCall

    def as_message(self) -> MessagePart:
        """The assistant turn to append to the conversation (reasoning is not resent)."""
        return MessagePart(role="assistant", origin="model", content=self.content,
                           tool_calls=[{k: tc[k] for k in ("id", "name", "arguments")} for tc in self.tool_calls])


@dataclass
class LLM:
    rec: RunRecorder
    model: str = "Qwen/Qwen3.8-27B"
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = 0
    max_tokens: int = 32768
    effort_via: str = field(default_factory=lambda: os.environ.get("GRADER_EFFORT_VIA", "param"))
    max_attempts: int = 3
    default_effort: str = "xhigh"              # the template's own default, made explicit
    client: Any = None

    def __post_init__(self):
        self.client = self.client or AsyncOpenAI(api_key="EMPTY", base_url=self.base_url, timeout=1800)

    async def _count_tokens(self, text: str) -> int:
        """Fallback when the server does not report reasoning tokens: vLLM's /tokenize endpoint."""
        def post() -> int:
            req = urllib.request.Request(self.base_url.rsplit("/v1", 1)[0] + "/tokenize",
                                         data=json.dumps({"model": self.model, "prompt": text,
                                                          "add_special_tokens": False}).encode(),
                                         headers={"Content-Type": "application/json"})
            return int(json.loads(urllib.request.urlopen(req, timeout=30).read())["count"])
        try:
            return await asyncio.to_thread(post)
        except Exception:
            return 0

    async def chat(self, messages: list[MessagePart], *, agent_id: str, role: AgentRole, prompt_author: str,
                   step_id: str | None = None, tools: list[dict] | None = None,
                   effort: str | None = None) -> LLMResult:
        llm_call_id = new_id("llm")
        effort = effort or self.default_effort
        kwargs: dict[str, Any] = dict(model=self.model, messages=[to_api(m) for m in messages],
                                      temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                                      extra_body={"top_k": self.top_k})
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs["tools"] = tools
        if effort:
            if self.effort_via == "template":
                kwargs["extra_body"]["chat_template_kwargs"] = {"reasoning_effort": effort}
            else:
                kwargs["reasoning_effort"] = effort
        meta = dict(llm_call_id=llm_call_id, agent_id=agent_id, role=role, prompt_author=prompt_author,
                    step_id=step_id, messages=messages, model=self.model, temperature=self.temperature,
                    top_p=self.top_p, top_k=self.top_k, seed=self.seed, reasoning_effort=effort,
                    max_tokens=self.max_tokens)

        for attempt in range(1, self.max_attempts + 1):
            start = time.time()
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except RETRYABLE as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                if attempt == self.max_attempts:
                    raise
                await asyncio.sleep(2 ** attempt)
        end = time.time()

        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args, err = json.loads(raw), None
                if not isinstance(args, dict):
                    args, err = {}, f"arguments are not a JSON object: {raw[:200]}"
            except json.JSONDecodeError as e:
                args, err = {}, f"invalid JSON arguments ({e}): {raw[:200]}"
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "parse_error": err})

        u = resp.usage
        ptd, ctd = getattr(u, "prompt_tokens_details", None), getattr(u, "completion_tokens_details", None)
        reasoning_tokens = getattr(ctd, "reasoning_tokens", None) if ctd else None
        if not reasoning_tokens and reasoning:
            reasoning_tokens = await self._count_tokens(reasoning)
        usage = TokenUsage(prompt=u.prompt_tokens, cached=(getattr(ptd, "cached_tokens", None) or 0) if ptd else 0,
                           output=u.completion_tokens, reasoning=reasoning_tokens or 0)

        ev = self.rec.emit(LLMCall(reasoning=reasoning, content=msg.content, tool_calls=calls, usage=usage,
                                   finish_reason=choice.finish_reason, start=start, end=end, attempt=attempt,
                                   **meta))
        return LLMResult(llm_call_id, msg.content or "", reasoning, calls, choice.finish_reason, ev)
