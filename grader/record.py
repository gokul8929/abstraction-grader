"""Run record for the abstraction grader.

Every component writes typed events through one RunRecorder per run:
  <run_dir>/events.jsonl          run events (this module)
  <run_dir>/server_<engine>.jsonl MCP server logs, joined on call_id
  <run_dir>/results/<call_id>.json tool results (read by the Python server's load())
Reviews are written separately to reviews.jsonl (Review model below).
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

SCHEMA_VERSION = "2"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ============================ run events ============================
class Event(BaseModel):
    type: str
    run_id: str = ""
    seq: int = 0
    ts: float = 0.0


class RunStart(Event):
    type: Literal["run_start"] = "run_start"
    schema_version: str = SCHEMA_VERSION
    dataset: str
    query_id: str
    question: str
    model: str
    reasoning_effort: str | None = None
    reflection: bool = False
    hints: bool = False
    code_commit: str | None = None
    dab_commit: str | None = None
    node: str | None = None
    gpu: str | None = None
    vllm_version: str | None = None
    model_revision: str | None = None
    chat_template_sha256: str | None = None
    server_flags: list[str] = []


class PlanStep(BaseModel):
    step_id: str
    description: str
    assigned_agent: str
    databases: list[str] = []
    depends_on: list[str] = []


class PlanCommitted(Event):                       # O bucket
    type: Literal["plan_committed"] = "plan_committed"
    plan_id: str
    version: int
    author: str
    steps: list[PlanStep]
    revises: str | None = None                    # plan_id this replaces; never overwritten
    revision_reason: str | None = None
    llm_call_id: str | None = None


InterpretationKind = Literal["source_choice", "column_meaning", "metric_definition", "join_key",
                             "normalisation", "filter_boundary", "entity_match", "other"]


class Interpretation(Event):                      # D2 bucket
    type: Literal["interpretation"] = "interpretation"
    interp_id: str
    agent_id: str
    step_id: str | None = None
    kind: InterpretationKind
    subject: str                                  # e.g. "index_trade.Date"
    statement: str                                # e.g. "dates come in mixed formats; parse all of them"
    source: Literal["data", "model_knowledge", "question"] | None = None   # where the judgement comes from
    evidence: list[str] = []                      # call_ids whose results support it (required for "data")
    llm_call_id: str | None = None


MessageOrigin = Literal["system_template", "task_context", "user_question", "orchestrator", "subagent",
                        "reflection", "tool_result", "model"]


class MessagePart(BaseModel):
    role: str
    origin: MessageOrigin
    content: str
    template_id: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = []          # assistant turns that called tools


class TokenUsage(BaseModel):
    prompt: int = 0
    cached: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def new_prompt(self) -> int:
        return self.prompt - self.cached


AgentRole = Literal["orchestrator", "subagent", "reflection"]


class LLMCall(Event):                             # L bucket
    type: Literal["llm_call"] = "llm_call"
    llm_call_id: str
    agent_id: str
    role: AgentRole
    prompt_author: str                            # who assembled these messages (L1)
    step_id: str | None = None
    messages: list[MessagePart]                   # each tagged with its origin (L2)
    model: str
    temperature: float | None = None              # sampling parameters (L0)
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    reasoning: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] = []
    usage: TokenUsage = TokenUsage()
    start: float
    end: float
    finish_reason: str | None = None              # "stop", "tool_calls", "length"
    attempt: int = 1                              # retries
    error: str | None = None


class ContextRewrite(Event):                      # L2: in-flight rewriting
    type: Literal["context_rewrite"] = "context_rewrite"
    agent_id: str
    kind: Literal["truncate", "summarise", "evict"]
    before_chars: int
    after_chars: int
    summary: str | None = None
    llm_call_id: str | None = None


class ToolCall(Event):                            # O (step link) + D (result)
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    server: str
    tool: str
    arguments: dict[str, Any]
    agent_id: str
    step_id: str | None = None                    # None = unplanned
    llm_call_id: str | None = None
    ok: bool
    error: str | None = None
    result_file: str | None = None
    result_sha256: str | None = None
    full_result_chars: int = 0
    shown_to_model_chars: int = 0                 # what the model actually saw
    loaded_results: list[str] = []                # for run_python: call_ids it loaded (D1)
    start: float
    end: float


class Reflection(Event):
    type: Literal["reflection"] = "reflection"
    reflection_id: str
    agent_id: str
    reviewed: list[str]                           # step_ids / call_ids examined
    verdict: Literal["accept", "revise", "retry", "abort"]
    notes: str = ""
    triggered_plan_id: str | None = None
    llm_call_id: str | None = None


class RunEnd(Event):
    type: Literal["run_end"] = "run_end"
    answer: str
    end_reason: str
    totals: dict[str, Any] = {}
    validation: dict[str, Any] | None = None


AnyEvent = Annotated[Union[RunStart, PlanCommitted, Interpretation, LLMCall, ContextRewrite,
                           ToolCall, Reflection, RunEnd], Field(discriminator="type")]
_EVENT_ADAPTER = TypeAdapter(AnyEvent)


# ============================ reviews (human cost) ============================
class RevealStep(BaseModel):
    level: str                                    # e.g. "result", "O1-Spec", "D2", "L1"
    opened_at: float
    items_opened: int = 0
    chars_read: int = 0


class Review(BaseModel):
    review_id: str
    run_id: str
    reviewer_id: str
    grade_shown: str                              # starting view, e.g. "D3/O2/L3"
    steps: list[RevealStep]
    started: float
    ended: float
    verdict: Literal["accept", "reject", "unresolved"]
    error_step_id: str | None = None              # where they think it went wrong
    confidence: int = Field(ge=1, le=7)
    mode: Literal["verify", "rewrite"]
    chars_typed: int = 0
    interventions: int = 0
    tlx: dict[str, int] = {}                      # NASA-TLX: mental, physical, temporal, performance, effort, frustration

    @property
    def time_to_verdict(self) -> float:
        return self.ended - self.started


# ============================ writer / reader ============================
class RunRecorder:
    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or new_id("run")
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, event: Event) -> Event:
        with self._lock:
            self._seq += 1
            event.run_id, event.seq, event.ts = self.run_id, self._seq, time.time()
            with open(self.path, "a") as f:
                f.write(event.model_dump_json() + "\n")
        return event

    def save_result(self, call_id: str, result: Any) -> tuple[str, str, int]:
        """Store a tool result for load(); return (path, sha256, size in chars)."""
        text = json.dumps(result, sort_keys=True, default=str)
        path = self.results_dir / f"{call_id}.json"
        path.write_text(text)
        return str(path), hashlib.sha256(text.encode()).hexdigest(), len(text)

    def server_log(self, engine: str) -> str:
        return str(self.run_dir / f"server_{engine}.jsonl")


def load_events(run_dir: str | Path) -> list[Event]:
    lines = Path(run_dir, "events.jsonl").read_text().splitlines()
    return [_EVENT_ADAPTER.validate_json(line) for line in lines if line.strip()]


def append_review(path: str | Path, review: Review) -> None:
    with open(path, "a") as f:
        f.write(review.model_dump_json() + "\n")


# ============================ environment capture ============================
def _cmd(args: list[str], cwd: str | None = None) -> str | None:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _sha_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def environment(code_dir: str = "~/abstraction-grader", dab_dir: str = "~/DataAgentBench",
                model: str = "Qwen/Qwen3.8-27B", vllm_url: str = "http://127.0.0.1:8000") -> dict:
    """Facts for RunStart: node, GPU, commits, vLLM version, exact model revision."""
    hf = Path(os.environ.get("HF_HOME", f"/tmp/{os.environ.get('USER', '')}/hf"))
    ref = hf / "hub" / f"models--{model.replace('/', '--')}" / "refs" / "main"
    vllm_version = None
    try:
        import urllib.request
        vllm_version = json.loads(urllib.request.urlopen(f"{vllm_url}/version", timeout=3).read()).get("version")
    except Exception:
        pass
    return {
        "node": socket.gethostname(),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "code_commit": _cmd(["git", "rev-parse", "HEAD"], os.path.expanduser(code_dir)),
        "dab_commit": _cmd(["git", "rev-parse", "HEAD"], os.path.expanduser(dab_dir)),
        "model_revision": ref.read_text().strip() if ref.exists() else None,
        "chat_template_sha256": _sha_file(ref.parent.parent / "snapshots" / ref.read_text().strip()
                                          / "chat_template.jinja") if ref.exists() else None,
        "vllm_version": vllm_version,
    }
