"""Patch: handle a full context window gracefully, and record vLLM's max_model_len.

Run from ~/abstraction-grader:  python patch_context.py
"""


def sub(p, old, new):
    s = open(p).read()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:80]}"
    open(p, "w").write(s.replace(old, new))


# ---------------- record.py: schema v3, max_model_len in run_start ----------------
R = "grader/record.py"
sub(R, 'SCHEMA_VERSION = "2"', 'SCHEMA_VERSION = "3"')
sub(R, "    chat_template_sha256: str | None = None\n",
       "    chat_template_sha256: str | None = None\n    max_model_len: int | None = None             # vLLM context limit for this run\n")
sub(R, '''        vllm_version = json.loads(urllib.request.urlopen(f"{vllm_url}/version", timeout=3).read()).get("version")
    except Exception:
        pass
''', '''        vllm_version = json.loads(urllib.request.urlopen(f"{vllm_url}/version", timeout=3).read()).get("version")
    except Exception:
        pass
    max_model_len = None
    try:
        import urllib.request
        models = json.loads(urllib.request.urlopen(f"{vllm_url}/v1/models", timeout=3).read())
        max_model_len = models["data"][0].get("max_model_len")
    except Exception:
        pass
''')
sub(R, '        "vllm_version": vllm_version,\n',
       '        "vllm_version": vllm_version,\n        "max_model_len": max_model_len,\n')

# ---------------- llm.py: fit output to remaining context; ContextExhausted ----------------
L = "grader/llm.py"
sub(L, "@dataclass\nclass LLMResult:", '''class ContextExhausted(Exception):
    """The conversation no longer fits in the model's context window."""


@dataclass
class LLMResult:''')
sub(L, '    default_effort: str = "xhigh"              # the template\'s own default, made explicit\n',
       '''    default_effort: str = "xhigh"              # the template's own default, made explicit
    context_margin: int = 256                     # tokens kept free beyond the measured prompt
    min_output_tokens: int = 1024                 # below this much room, the context counts as exhausted
''')
sub(L, "    async def chat(self, messages", '''    async def _prompt_size(self, kwargs: dict, effort: str | None) -> tuple[int, int] | None:
        """(prompt tokens, context limit) from vLLM's /tokenize, rendered exactly like the chat call."""
        body = {"model": self.model, "messages": kwargs["messages"], "add_generation_prompt": True}
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        if effort:
            body["chat_template_kwargs"] = {"reasoning_effort": effort}

        def post() -> tuple[int, int] | None:
            req = urllib.request.Request(self.base_url.rsplit("/v1", 1)[0] + "/tokenize",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return (int(r["count"]), int(r["max_model_len"])) if r.get("max_model_len") else None
        try:
            return await asyncio.to_thread(post)
        except Exception:
            return None

    async def chat(self, messages''')
sub(L, '''                    max_tokens=self.max_tokens)

        for attempt in range(1, self.max_attempts + 1):''', '''                    max_tokens=self.max_tokens)

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

        for attempt in range(1, self.max_attempts + 1):''')
sub(L, '''            except RETRYABLE as e:''', '''            except openai.BadRequestError as e:
                self.rec.emit(LLMCall(start=start, end=time.time(), attempt=attempt,
                                      error=f"{type(e).__name__}: {e}", **meta))
                if "maximum context length" in str(e):
                    raise ContextExhausted(str(e)) from e
                raise
            except RETRYABLE as e:''')

# ---------------- agent.py: a full context ends the step / run cleanly ----------------
A = "grader/agent.py"
sub(A, "from grader.llm import LLM\n", "from grader.llm import LLM, ContextExhausted\n")
sub(A, '''            r = await self.llm.chat(msgs, agent_id=agent, role="subagent", prompt_author="orchestrator",
                                    step_id=step.step_id, tools=tools, effort=self.effort)''',
'''            try:
                r = await self.llm.chat(msgs, agent_id=agent, role="subagent", prompt_author="orchestrator",
                                        step_id=step.step_id, tools=tools, effort=self.effort)
            except ContextExhausted:
                return StepResult(step.step_id, f"Stopped: context limit reached. Last message: {last[:1000]}",
                                  finished=False)''')
sub(A, "    async def run(self, question: str) -> AgentResult:\n",
       '''    async def run(self, question: str) -> AgentResult:
        try:
            return await self._run(question)
        except ContextExhausted:                       # the orchestrator's own context is full
            return AgentResult("", "context_exhausted", 0)

    async def _run(self, question: str) -> AgentResult:
''')
print("patched record.py, llm.py, agent.py")

# ---------------- batch.py: one run per cell — prefer non-error, then newest ----------------
B = "grader/batch.py"
sub(B, '''        key = (start.dataset, start.query_id, start.reasoning_effort, int(flags.get("seed", 0)))
        done[key] = {"run": ev_file.parent.name, "valid": end.validation["is_valid"], "end": end.end_reason,
                     "answer": end.answer, **end.totals}''',
'''        key = (start.dataset, start.query_id, start.reasoning_effort, int(flags.get("seed", 0)))
        cand = {"run": ev_file.parent.name, "valid": end.validation["is_valid"], "end": end.end_reason,
                "answer": end.answer, "_ts": start.ts, **end.totals}
        old = done.get(key)
        rank = lambda r: (not r["end"].startswith("error"), r["_ts"])   # finished beats error; then newest
        if old is None or rank(cand) > rank(old):
            done[key] = cand''')
print("patched batch.py")
