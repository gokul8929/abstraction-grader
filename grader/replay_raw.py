"""Replay an orchestrator call that came back empty, and show the raw tokens the model generated.

  python -m grader.replay_raw --tag tune-gptoss_r3 --model gpt-oss-120b            # needs the vLLM server up
  python -m grader.replay_raw --run runs/stockindex/q1/run-577fafb217 --model gpt-oss-120b

For each run that ended no_final_answer, the first orchestrator call after the last executed plan is sent
again exactly as recorded (same messages, tools, effort, sampling), with the server asked to return the
generated token ids. Each replay is tried with several seeds, and once with tool_choice="required".
Prints, per attempt: finish_reason, output tokens, what the server parsed (content / reasoning / tool calls)
and the raw generated text including harmony control tokens. Writes runs/_batch/<tag>_replay.md.
Nothing in the run folders is changed (replays are recorded in a scratch folder).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import types
import urllib.request
from pathlib import Path

from grader.agent import ORCH_TOOLS
from grader.batch import finished
from grader.llm import LLM
from grader.record import RunRecorder, load_events
from grader.run import RUNS

SPECIAL = {200002: "<|return|>", 200003: "<|constrain|>", 200005: "<|channel|>", 200006: "<|start|>",
           200007: "<|end|>", 200008: "<|message|>", 200012: "<|call|>"}


class Tap:
    """Wraps the OpenAI client: asks vLLM for the generated token ids and keeps the raw response."""

    def __init__(self, real, extra: dict):
        self.real, self.extra, self.last = real, extra, None

    async def create(self, **kw):
        kw.setdefault("extra_body", {})["return_token_ids"] = True
        kw.update(self.extra)
        self.last = await self.real.chat.completions.create(**kw)
        return self.last


def _detok(base_url: str, model: str, ids: list[int]) -> str:
    root = base_url.rsplit("/v1", 1)[0]
    body = json.dumps({"model": model, "tokens": ids}).encode()
    try:
        req = urllib.request.Request(root + "/detokenize", body, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt"]
    except Exception as e:                          # fall back to showing the control tokens only
        return f"(detokenize failed: {e}) ids: " + " ".join(SPECIAL.get(i, str(i)) for i in ids[:200])


def _target(run_dir: Path):
    orch = [e for e in load_events(run_dir) if e.type == "llm_call" and e.agent_id == "orchestrator"]
    idx = max((i for i, e in enumerate(orch) if "Step summaries:" in (e.messages[-1].content or "")), default=None)
    return orch[idx] if idx is not None else None


async def replay(run_dir: Path, model_key: str, seeds: list[int]) -> list[str]:
    ev = _target(run_dir)
    if ev is None:
        return ["- no orchestrator call after an executed plan"]
    out = [f"- replaying {ev.llm_call_id} (effort {ev.reasoning_effort}, recorded seed {ev.seed}, "
           f"recorded output tokens {ev.usage.output}, reasoning tokens {ev.usage.reasoning})"]
    base = ev.seed if ev.seed is not None else 0
    seeds = seeds or [base, base + 1, base + 2]             # the recorded seed first: should reproduce the empty turn
    trials = [(s, {}) for s in seeds] + [(seeds[0], {"tool_choice": "required"})]
    for seed, extra in trials:
        rec = RunRecorder(Path(tempfile.mkdtemp()) / "replay")
        llm = LLM(rec, seed=seed, profile_key=model_key)
        tap = Tap(llm.client, extra)
        llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=tap))
        label = f"seed {seed}" + (f", {extra}" if extra else "")
        try:
            r = await llm.chat(ev.messages, agent_id="orchestrator", role="orchestrator", prompt_author="harness",
                               tools=ORCH_TOOLS, effort=ev.reasoning_effort)
        except Exception as e:
            out.append(f"  - {label}: ERROR {type(e).__name__}: {str(e)[:300]}")
            continue
        ch = tap.last.choices[0]
        ids = (getattr(ch, "model_extra", None) or {}).get("token_ids") or getattr(ch, "token_ids", None) or []
        raw = _detok(llm.base_url, llm.model, ids) if ids else "(server returned no token ids)"
        out += [f"  - {label}: finish_reason={r.finish_reason} · output tokens {tap.last.usage.completion_tokens}",
                f"    - parsed tool_calls: {[c['name'] for c in r.tool_calls] or '(none)'}",
                f"    - parsed content: {r.content[:200]!r}",
                f"    - parsed reasoning: {(r.reasoning or '')[:200]!r}",
                f"    - raw generated text: {raw[:1500]!r}"]
    return out


async def main_async(a):
    if a.run:
        runs = [Path(a.run)]
        name = Path(a.run).name
    else:
        name = a.tag
        runs = []
        for (d, q, e, s), r in sorted(finished(a.tag, include_errors=True).items()):
            run_dir = RUNS / d / f"q{q}" / r["run"]
            end = next((x for x in reversed(load_events(run_dir)) if x.type == "run_end"), None)
            if end is not None and end.end_reason == "no_final_answer":
                runs.append(run_dir)
    parts = [f"# replay of empty orchestrator turns: {name}", ""]
    for run_dir in runs:
        parts += [f"## {run_dir}", ""] + await replay(run_dir, a.model, a.seeds) + [""]
    text = "\n".join(parts)
    out = RUNS / "_batch" / f"{name}_replay.md"
    out.write_text(text)
    print(text)
    print(f"\nwritten to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--run")
    ap.add_argument("--model", default=os.environ.get("GRADER_MODEL", "gpt-oss-120b"))
    ap.add_argument("--seeds", default=None, type=lambda s: [int(x) for x in s.split(",")],
                    help="default: the recorded seed, then +1, +2")
    a = ap.parse_args()
    if not (a.tag or a.run):
        ap.error("give --tag or --run")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
