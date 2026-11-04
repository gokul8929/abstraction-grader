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

from grader import models as model_profiles
from grader.argfix import coerce_args
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
    if m.reasoning:                                  # only set for profiles with resend_reasoning (gpt-oss)
        d["reasoning_content"] = m.reasoning
        d["reasoning"] = m.reasoning
    return d


class ContextExhausted(Exception):
    """The conversation no longer fits in the model's context window."""


class ModelOutputError(Exception):
    """The model kept producing output the server could not parse (e.g. a malformed tool-call header)."""


def _unparseable(e: Exception) -> bool:
    s = str(e)
    return "message header" in s or "unexpected tokens" in s or "HarmonyError" in s


@dataclass
class LLMResult:
    llm_call_id: str
    content: str
    reasoning: str | None
    tool_calls: list[dict[str, Any]]   # {"id", "name", "arguments": dict, "parse_error": str | None}
    finish_reason: str | None
    event: LLMCall
    resend_reasoning: bool = False

    def as_message(self) -> MessagePart:
        """The assistant turn to append to the conversation (reasoning resent only if the profile needs it)."""
        return MessagePart(role="assistant", origin="model", content=self.content,  # coerced args are resent
                           reasoning=self.reasoning if self.resend_reasoning else None,
                           tool_calls=[{k: tc[k] for k in ("id", "name", "arguments")} for tc in self.tool_calls])


@dataclass
class LLM:
    rec: RunRecorder
    model: str | None = None                       # default: the profile's served model id
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    temperature: float | None = None               # default: the profile's value, else generation_config.json
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = 0
    max_tokens: int = 32768
    effort_via: str = field(default_factory=lambda: os.environ.get("GRADER_EFFORT_VIA", "param"))
    max_attempts: int = 3
    default_effort: str | None = None              # default: the profile's default setting
    profile_key: str | None = None                 # grader.models profile (env GRADER_MODEL, default qwen3.8-27b)
    profile: Any = None
    context_margin: int = 256                     # tokens kept free beyond the measured prompt
    min_output_tokens: int = 1024                 # below this much room, the context counts as exhausted
    client: Any = None

    def __post_init__(self):
        self.profile = self.profile or model_profiles.get(self.profile_key)
        self.model = self.model or self.profile.hf_id
        self.default_effort = self.default_effort or self.profile.default_setting
        gen = model_profiles.generation_defaults(self.profile.hf_id)
        for k in ("temperature", "top_p", "top_k"):
            if getattr(self, k) is None:
                v = getattr(self.profile, k)
                setattr(self, k, v if v is not None else gen.get(k))
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

    async def _prompt_size(self, kwargs: dict, effort: str | None) -> tuple[int, int] | None:
        """(prompt tokens, context limit) from vLLM's /tokenize, rendered exactly like the chat call."""
        body = {"model": self.model, "messages": kwargs["messages"], "add_generation_prompt": True}
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        ctk = dict(kwargs["extra_body"].get("chat_template_kwargs", {}))
        if kwargs.get("reasoning_effort"):
            ctk.setdefault("reasoning_effort", kwargs["reasoning_effort"])
        if ctk:
            body["chat_template_kwargs"] = ctk

        def post() -> tuple[int, int] | None:
            req = urllib.request.Request(self.base_url.rsplit("/v1", 1)[0] + "/tokenize",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return (int(r["count"]), int(r["max_model_len"])) if r.get("max_model_len") else None
        try:
            return await asyncio.to_thread(post)
        except Exception:
            return None

    async def chat(self, messages: list[MessagePart], *, agent_id: str, role: AgentRole, prompt_author: str,
                   step_id: str | None = None, tools: list[dict] | None = None,
                   effort: str | None = None) -> LLMResult:
        llm_call_id = new_id("llm")
        effort = effort or self.default_effort
        if effort not in self.profile.settings:
            raise ValueError(f"setting {effort!r} not valid for {self.profile.key}: {list(self.profile.settings)}")
        kwargs: dict[str, Any] = dict(model=self.model, messages=[to_api(m) for m in messages],
                                      max_tokens=self.max_tokens, extra_body={})
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["extra_body"]["top_k"] = self.top_k
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs["tools"] = tools
        tool_choice = "required" if tools and role in self.profile.force_tool_call else None
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        extras = self.profile.settings[effort]
        if "reasoning_effort" in extras:
            if self.effort_via == "template":
                kwargs["extra_body"]["chat_template_kwargs"] = {"reasoning_effort": extras["reasoning_effort"]}
            else:
                kwargs["reasoning_effort"] = extras["reasoning_effort"]
        if "chat_template_kwargs" in extras:
            kwargs["extra_body"].setdefault("chat_template_kwargs", {}).update(extras["chat_template_kwargs"])
        meta = dict(llm_call_id=llm_call_id, agent_id=agent_id, role=role, prompt_author=prompt_author,
                    step_id=step_id, messages=messages, model=self.model, temperature=self.temperature,
                    top_p=self.top_p, top_k=self.top_k, seed=self.seed, reasoning_effort=effort,
                    max_tokens=self.max_tokens, tool_choice=tool_choice)

        size = await self._prompt_size(kwargs, effort)
        if size:
            prompt_tokens, limit = size
            room = limit - prompt_tokens - self.context_margin
            if room < self.min_output_tokens:
                now = time.time()
                self.rec.emit(LLMCall(start=now, end=now, error=f"context_exhausted: prompt {prompt_tokens} "
                                      f"tokens, limit {limit}", **meta))
                raise ContextExhausted(f"prompt {prompt_tokens} tokens leaves {room} of {limit}")
            if room < kwargs["max_tokens"]:
                kwargs["max_tokens"] = meta["max_tokens"] = room

        for attempt in range(1, self.max_attempts + 1):
            start = time.time()
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                if "maximum context length" in str(e):
                    raise ContextExhausted(str(e)) from e
                raise
            except RETRYABLE as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                bad_output = isinstance(e, openai.InternalServerError) and _unparseable(e)
                if attempt == self.max_attempts:
                    if bad_output:
                        raise ModelOutputError(str(e)[:500]) from e
                    raise
                if bad_output and self.seed is not None:   # same seed would reproduce the same bad output
                    kwargs["seed"] = meta["seed"] = self.seed + 1009 * attempt
                await asyncio.sleep(2 ** attempt)
        end = time.time()

        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        calls = []
        for tc in msg.tool_calls or []:
            name, renamed = tc.function.name or "", []
            if "<|" in name:                           # control token leaked into the name (vLLM openai parser)
                clean = name.split("<|")[0].strip()
                renamed.append(f"name: {name!r} -> {clean!r}")
                name = clean
            raw = tc.function.arguments or "{}"
            try:
                args, err = json.loads(raw), None
                if not isinstance(args, dict):
                    args, err = {}, f"arguments are not a JSON object: {raw[:200]}"
            except json.JSONDecodeError as e:
                args, err = {}, f"invalid JSON arguments ({e}): {raw[:200]}"
            schema = next((t["function"].get("parameters") for t in (tools or [])
                           if t.get("function", {}).get("name") == name), None)
            args, coerced = coerce_args(args, schema)
            calls.append({"id": tc.id, "name": name, "arguments": args, "parse_error": err,
                          "coerced": renamed + coerced})

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
        return LLMResult(llm_call_id, msg.content or "", reasoning, calls, choice.finish_reason, ev,
                         resend_reasoning=self.profile.resend_reasoning)
