"""MCP client for the grader: one ToolHub per run.

- starts one stdio server per engine the dataset needs (+ python)
- shows the model OpenAI-style tool schemas WITHOUT call_id
- injects a fresh call_id into every call (joins events.jsonl with server_<engine>.jsonl)
- saves every successful result to <run>/results/<call_id>.json for run_python's load()
- shows the model a capped view and records how much it actually saw
- emits one ToolCall event per call, including failed and unknown ones
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mcp import Client, StdioServerParameters

from grader.record import RunRecorder, ToolCall, new_id

SERVERS_DIR = Path(__file__).resolve().parent.parent / "servers"
SEP = "__"                                         # tool name shown to model: "<engine>__<tool>"
ENGINE_OF = {"sqlite": "sqlite", "duckdb": "duckdb", "postgres": "postgres", "postgresql": "postgres",
             "mongo": "mongo", "mongodb": "mongo"}
PASS_ENV = ["PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "MONGO_URI",
            "GRADER_MAX_ROWS", "GRADER_EXEC_TIMEOUT"]
LOAD_RE = re.compile(r"""load\(\s*['"]([^'"]+)['"]""")


def engines_for(dataset_dir: str | Path) -> list[str]:
    cfg = yaml.safe_load(open(Path(dataset_dir).expanduser() / "db_config.yaml"))["db_clients"]
    return sorted({ENGINE_OF[c["db_type"]] for c in cfg.values()}) + ["python"]


def _strip_call_id(schema: dict) -> dict:
    s = json.loads(json.dumps(schema))
    s.get("properties", {}).pop("call_id", None)
    if "required" in s:
        s["required"] = [r for r in s["required"] if r != "call_id"]
    s.pop("title", None)
    return s


@dataclass
class ToolOutcome:
    text: str                                      # what goes back to the model
    event: ToolCall


class ToolHub:
    def __init__(self, rec: RunRecorder, dataset_dir: str | Path, engines: list[str] | None = None,
                 show_chars: int | None = None):
        self.rec = rec
        self.dataset_dir = str(Path(dataset_dir).expanduser())
        self.engines = engines or engines_for(self.dataset_dir)
        self.show_chars = show_chars or int(os.environ.get("GRADER_SHOW_CHARS", "6000"))
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, tuple[str, Any]] = {}   # model-facing name -> (engine, mcp Tool)
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "ToolHub":
        await self._stack.__aenter__()
        for engine in self.engines:
            env = {"DAB_DATASET_DIR": self.dataset_dir, "GRADER_RUN_DIR": str(self.rec.run_dir),
                   "GRADER_SERVER_LOG": self.rec.server_log(engine),
                   "PYTHONPATH": str(SERVERS_DIR.parent)}
            env |= {k: os.environ[k] for k in PASS_ENV if k in os.environ}
            params = StdioServerParameters(command=sys.executable,
                                           args=[str(SERVERS_DIR / f"{engine}_server.py")], env=env)
            client = await self._stack.enter_async_context(Client(params))
            self.clients[engine] = client
            for tool in (await client.list_tools()).tools:
                self.tools[f"{engine}{SEP}{tool.name}"] = (engine, tool)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.__aexit__(*exc)

    def openai_tools(self, engines: list[str] | None = None) -> list[dict]:
        """Tool list for the chat API; restrict to some engines for a given sub-agent."""
        return [{"type": "function",
                 "function": {"name": name,
                              "description": f"[{engine}] {tool.description or ''}".strip(),
                              "parameters": _strip_call_id(tool.input_schema)}}
                for name, (engine, tool) in self.tools.items()
                if engines is None or engine in engines]

    async def call(self, name: str, arguments: dict[str, Any] | None, agent_id: str,
                   step_id: str | None = None, llm_call_id: str | None = None) -> ToolOutcome:
        args = dict(arguments or {})
        args.pop("call_id", None)                  # the model never chooses call_id
        call_id = new_id("call")
        start = time.time()
        engine, _, tool_name = name.partition(SEP)
        base = dict(call_id=call_id, server=engine if name in self.tools else "unknown",
                    tool=tool_name or name, arguments=args, agent_id=agent_id,
                    step_id=step_id, llm_call_id=llm_call_id, start=start)

        if name not in self.tools:
            msg = f"Unknown tool '{name}'. Available: {', '.join(self.tools)}"
            ev = self.rec.emit(ToolCall(ok=False, error=msg, end=time.time(), **base))
            return ToolOutcome(f"ERROR: {msg}", ev)

        loaded = LOAD_RE.findall(args.get("code", "")) if tool_name == "run_python" else []
        try:
            res = await self.clients[engine].call_tool(tool_name, {**args, "call_id": call_id})
        except Exception as e:                     # server crashed or transport failed
            res, crash = None, f"{type(e).__name__}: {e}"
        end = time.time()

        if res is None or res.is_error:
            msg = crash if res is None else "\n".join(getattr(c, "text", "") for c in res.content)
            ev = self.rec.emit(ToolCall(ok=False, error=msg, loaded_results=loaded, end=end, **base))
            return ToolOutcome(f"ERROR: {msg}", ev)

        result = res.structured_content
        if result is None:
            result = {"text": "\n".join(getattr(c, "text", "") for c in res.content)}
        path, sha, size = self.rec.save_result(call_id, result)

        full = json.dumps(result, default=str)
        if len(full) <= self.show_chars:
            shown = full
        else:
            shown = (full[: self.show_chars] + f"\n... [cut at {self.show_chars} of {len(full)} chars. "
                     f"Full result: load('{call_id}') in python__run_python]")
        text = f"[call_id: {call_id}]\n{shown}"
        ev = self.rec.emit(ToolCall(ok=True, result_file=path, result_sha256=sha, full_result_chars=size,
                                    shown_to_model_chars=min(len(full), self.show_chars),
                                    loaded_results=loaded, end=end, **base))
        return ToolOutcome(text, ev)
