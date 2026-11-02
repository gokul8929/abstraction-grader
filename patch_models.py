"""Patch: run any model profile (Qwen3.8-27B, gpt-oss-120b, GLM-4.5-Air) through the same harness.

Run from ~/abstraction-grader after copying grader/models.py in:  python patch_models.py
Every replacement must match exactly once, so the script stops if the files differ from what it expects.
Schema 3 -> 4: RunStart.model_profile; MessagePart.reasoning (reasoning resent to the model, gpt-oss only).
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


assert Path("grader/models.py").exists(), "copy grader/models.py into the repo first"
if 'SCHEMA_VERSION = "4"' in Path("grader/record.py").read_text():
    raise SystemExit("already patched (schema 4); nothing to do")

# ---------------- record.py ----------------
R = "grader/record.py"
sub(R, 'SCHEMA_VERSION = "3"', 'SCHEMA_VERSION = "4"')
sub(R, "    model: str\n    reasoning_effort: str | None = None\n    reflection: bool = False\n",
       "    model: str\n    model_profile: str | None = None               # grader.models profile key\n"
       "    reasoning_effort: str | None = None              # the run's setting: effort level or thinking mode\n"
       "    reflection: bool = False\n")
sub(R, "    tool_calls: list[dict[str, Any]] = []          # assistant turns that called tools\n",
       "    tool_calls: list[dict[str, Any]] = []          # assistant turns that called tools\n"
       "    reasoning: str | None = None                   # reasoning resent to the model on later turns (gpt-oss)\n")
sub(R, '''        "chat_template_sha256": _sha_file(ref.parent.parent / "snapshots" / ref.read_text().strip()
                                          / "chat_template.jinja") if ref.exists() else None,''',
       '''        "chat_template_sha256": _template_sha(ref.parent.parent / "snapshots" / ref.read_text().strip())
                                if ref.exists() else None,''')
sub(R, "def environment(", '''def _template_sha(snap: Path) -> str | None:
    """SHA-256 of the chat template: chat_template.jinja if present, else tokenizer_config.json's chat_template."""
    if (snap / "chat_template.jinja").exists():
        return _sha_file(snap / "chat_template.jinja")
    cfg = snap / "tokenizer_config.json"
    if cfg.exists():
        t = json.loads(cfg.read_text()).get("chat_template")
        if t:
            return hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
    return None


def environment(''')

# ---------------- llm.py ----------------
L = "grader/llm.py"
sub(L, "from grader.record import AgentRole,", "from grader import models as model_profiles\nfrom grader.record import AgentRole,")
sub(L, '''    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id
    return d''', '''    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id
    if m.reasoning:                                  # only set for profiles with resend_reasoning (gpt-oss)
        d["reasoning_content"] = m.reasoning
        d["reasoning"] = m.reasoning
    return d''')
sub(L, '''    finish_reason: str | None
    event: LLMCall

    def as_message(self) -> MessagePart:
        """The assistant turn to append to the conversation (reasoning is not resent)."""
        return MessagePart(role="assistant", origin="model", content=self.content,
                           tool_calls=[{k: tc[k] for k in ("id", "name", "arguments")} for tc in self.tool_calls])''',
    '''    finish_reason: str | None
    event: LLMCall
    resend_reasoning: bool = False

    def as_message(self) -> MessagePart:
        """The assistant turn to append to the conversation (reasoning resent only if the profile needs it)."""
        return MessagePart(role="assistant", origin="model", content=self.content,
                           reasoning=self.reasoning if self.resend_reasoning else None,
                           tool_calls=[{k: tc[k] for k in ("id", "name", "arguments")} for tc in self.tool_calls])''')
sub(L, '''    model: str = "Qwen/Qwen3.8-27B"
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = 0''', '''    model: str | None = None                       # default: the profile's served model id
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    temperature: float | None = None               # default: the profile's value, else generation_config.json
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = 0''')
sub(L, '''    default_effort: str = "xhigh"              # the template's own default, made explicit''',
       '''    default_effort: str | None = None              # default: the profile's default setting
    profile_key: str | None = None                 # grader.models profile (env GRADER_MODEL, default qwen3.8-27b)
    profile: Any = None''')
sub(L, '''    def __post_init__(self):
        self.client = self.client or AsyncOpenAI(api_key="EMPTY", base_url=self.base_url, timeout=1800)''',
    '''    def __post_init__(self):
        self.profile = self.profile or model_profiles.get(self.profile_key)
        self.model = self.model or self.profile.hf_id
        self.default_effort = self.default_effort or self.profile.default_setting
        gen = model_profiles.generation_defaults(self.profile.hf_id)
        for k in ("temperature", "top_p", "top_k"):
            if getattr(self, k) is None:
                v = getattr(self.profile, k)
                setattr(self, k, v if v is not None else gen.get(k))
        self.client = self.client or AsyncOpenAI(api_key="EMPTY", base_url=self.base_url, timeout=1800)''')
sub(L, '''        body = {"model": self.model, "messages": kwargs["messages"], "add_generation_prompt": True}
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        if effort:
            body["chat_template_kwargs"] = {"reasoning_effort": effort}''',
    '''        body = {"model": self.model, "messages": kwargs["messages"], "add_generation_prompt": True}
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        ctk = dict(kwargs["extra_body"].get("chat_template_kwargs", {}))
        if kwargs.get("reasoning_effort"):
            ctk.setdefault("reasoning_effort", kwargs["reasoning_effort"])
        if ctk:
            body["chat_template_kwargs"] = ctk''')
sub(L, '''        effort = effort or self.default_effort
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
                kwargs["reasoning_effort"] = effort''',
    '''        effort = effort or self.default_effort
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
        extras = self.profile.settings[effort]
        if "reasoning_effort" in extras:
            if self.effort_via == "template":
                kwargs["extra_body"]["chat_template_kwargs"] = {"reasoning_effort": extras["reasoning_effort"]}
            else:
                kwargs["reasoning_effort"] = extras["reasoning_effort"]
        if "chat_template_kwargs" in extras:
            kwargs["extra_body"].setdefault("chat_template_kwargs", {}).update(extras["chat_template_kwargs"])''')
sub(L, "        return LLMResult(llm_call_id, msg.content or \"\", reasoning, calls, choice.finish_reason, ev)",
       "        return LLMResult(llm_call_id, msg.content or \"\", reasoning, calls, choice.finish_reason, ev,\n"
       "                         resend_reasoning=self.profile.resend_reasoning)")

# ---------------- run.py ----------------
U = "grader/run.py"
sub(U, "  python -m grader.run --dataset stockindex --query 1 --effort medium --seed 0\n",
       "  python -m grader.run --dataset stockindex --query 1 --effort medium --seed 0\n"
       "  python -m grader.run --model glm-4.5-air --dataset yelp --query 1 --effort nothink\n")
sub(U, 'GPUS = int(os.environ.get("GRADER_GPUS", "2"))                       # tensor-parallel GPUs held by vLLM\n',
       'GPUS_ENV = os.environ.get("GRADER_GPUS")       # override; default = the model profile\'s tensor-parallel GPUs\n')
sub(U, "def totals(events: list[Event], wall: float) -> dict:", "def totals(events: list[Event], wall: float, gpus: int) -> dict:")
sub(U, "    gpu_s = llm_s * GPUS\n", "    gpu_s = llm_s * gpus\n")
sub(U, '"gpu_s": round(gpu_s, 2), "gpus": GPUS,', '"gpu_s": round(gpu_s, 2), "gpus": gpus,')
sub(U, "async def run_query(dataset: str, query: str, effort: str | None, seed: int, hints: bool, tag: str) -> Path:",
       "async def run_query(dataset: str, query: str, effort: str | None, seed: int, hints: bool, tag: str,\n"
       "                    model: str | None = None) -> Path:")
sub(U, "    llm = LLM(rec, seed=seed)\n", "    llm = LLM(rec, seed=seed, profile_key=model)\n    gpus = int(GPUS_ENV or llm.profile.gpus)\n")
sub(U, '''    rec.emit(RunStart(dataset=dataset, query_id=str(query), question=question, model=llm.model,
                      reasoning_effort=effort,''', '''    rec.emit(RunStart(dataset=dataset, query_id=str(query), question=question, model=llm.model,
                      model_profile=llm.profile.key, reasoning_effort=effort,''')
sub(U, '[f"seed={seed}", f"tag={tag}"]', '[f"seed={seed}", f"tag={tag}", f"model={llm.profile.key}"]')
sub(U, "totals=totals(load_events(run_dir), wall)", "totals=totals(load_events(run_dir), wall, gpus)")
sub(U, "    env = environment(model=llm.model)\n",
       '    env = environment(model=llm.model, vllm_url=llm.base_url.rsplit("/v1", 1)[0])\n')

sub(U, '''    ap.add_argument("--effort", choices=["low", "medium", "xhigh"])''',
       '''    ap.add_argument("--model", default=None, help="grader.models profile key (default qwen3.8-27b)")
    ap.add_argument("--effort", help="the profile's setting: effort level, or think / nothink for GLM")''')
sub(U, "asyncio.run(run_query(a.dataset, a.query, a.effort, a.seed, a.hints, a.tag))",
       "asyncio.run(run_query(a.dataset, a.query, a.effort, a.seed, a.hints, a.tag, a.model))")

# ---------------- batch.py ----------------
B = "grader/batch.py"
sub(B, "from grader.record import load_events\n", "from grader import models as model_profiles\nfrom grader.record import load_events\n")
sub(B, "def finished(tag: str, include_errors: bool = False) -> dict[tuple, dict]:",
       "def finished(tag: str, include_errors: bool = False, model: str | None = None) -> dict[tuple, dict]:")
sub(B, '''        if flags.get("tag") != tag or (end.end_reason.startswith("error") and not include_errors):
            continue''', '''        if flags.get("tag") != tag or (end.end_reason.startswith("error") and not include_errors):
            continue
        if model and flags.get("model", "qwen3.8-27b") != model:
            continue''')
sub(B, '''    cells = grid(a.datasets, a.efforts, a.seeds, a.tag)
    done = finished(a.tag)''', '''    cells = grid(a.datasets, a.efforts, a.seeds, a.tag)
    done = finished(a.tag, model=a.model)''')
sub(B, '''        cmd = [sys.executable, "-m", "grader.run", "--dataset", d, "--query", q, "--effort", e,''',
       '''        cmd = [sys.executable, "-m", "grader.run", "--model", a.model, "--dataset", d, "--query", q, "--effort", e,''')
sub(B, "        r = finished(a.tag).get((d, q, e, s))", "        r = finished(a.tag, model=a.model).get((d, q, e, s))")
sub(B, "    done = finished(a.tag, include_errors=True)\n    if not done:", "    done = finished(a.tag, include_errors=True, model=a.model)\n    if not done:")
sub(B, '''    for e in ["low", "medium", "xhigh", "ALL"]:''', '''    for e in sorted({k[2] for k in done}) + ["ALL"]:''')
sub(B, '''    ap.add_argument("--efforts", default="low,medium,xhigh", type=lambda s: s.split(","))''',
       '''    ap.add_argument("--model", default="qwen3.8-27b", help="grader.models profile key")
    ap.add_argument("--efforts", default=None, type=lambda s: s.split(","), help="default: all of the profile's settings")
    ap.add_argument("--queries", default=None, type=lambda s: s.split(","), help="dataset:query list, e.g. yelp:1 (smoke tests)")''')
sub(B, "    a = ap.parse_args()\n    summary(a) if a.summary else run_batch(a)",
       "    a = ap.parse_args()\n    a.efforts = a.efforts or list(model_profiles.get(a.model).settings)\n"
       "    summary(a) if a.summary else run_batch(a)")
sub(B, '''def grid(datasets, efforts, seeds, tag):
    cells = [(d, q, e, s) for d in datasets for q in queries(d) for e in efforts for s in seeds]''',
    '''def grid(datasets, efforts, seeds, tag, only=None):
    cells = [(d, q, e, s) for d in datasets for q in queries(d) for e in efforts for s in seeds
             if not only or f"{d}:{q}" in only]''')
sub(B, "    cells = grid(a.datasets, a.efforts, a.seeds, a.tag)\n", "    cells = grid(a.datasets, a.efforts, a.seeds, a.tag, a.queries)\n")
print("patched: record.py (schema 4), llm.py, run.py, batch.py")
