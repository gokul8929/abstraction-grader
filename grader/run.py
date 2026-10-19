"""Run one DAB query through the agent and record everything.

  python -m grader.run --dataset stockindex --query 1 --effort medium --seed 0

Writes runs/<dataset>/q<query>/<run_id>/ : events.jsonl, server_*.jsonl, results/
Scores the answer with the query's own validate.py, and writes totals (tokens, time, GPU-seconds, Rs).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

from grader.agent import DataAgent
from grader.client import ToolHub, engines_for
from grader.llm import LLM
from grader.record import Event, RunEnd, RunRecorder, RunStart, environment, new_id

DAB = Path(os.environ.get("DAB_DIR", "~/DataAgentBench")).expanduser()
RUNS = Path(os.environ.get("GRADER_RUNS", Path(__file__).resolve().parent.parent / "runs"))
GPUS = int(os.environ.get("GRADER_GPUS", "2"))                       # tensor-parallel GPUs held by vLLM
INR_PER_GPU_HOUR = float(os.environ.get("GRADER_INR_PER_GPU_HOUR", "35"))
FLAGS = ["GRADER_SHOW_CHARS", "GRADER_MAX_ROWS", "GRADER_EXEC_TIMEOUT"]


def load_validator(query_dir: Path):
    spec = importlib.util.spec_from_file_location(f"dab_validate_{query_dir.parent.name}_{query_dir.name}",
                                                  query_dir / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate


def totals(events: list[Event], wall: float) -> dict:
    llm = [e for e in events if e.type == "llm_call"]
    tools = [e for e in events if e.type == "tool_call"]
    llm_s = sum(e.end - e.start for e in llm)
    gpu_s = llm_s * GPUS
    u = lambda f: sum(getattr(e.usage, f) for e in llm)
    return {
        "wall_s": round(wall, 2),
        "llm_calls": len(llm), "llm_errors": sum(1 for e in llm if e.error),
        "llm_s": round(llm_s, 2), "tool_calls": len(tools), "tool_errors": sum(1 for e in tools if not e.ok),
        "tool_s": round(sum(e.end - e.start for e in tools), 2),
        "prompt_tokens": u("prompt"), "cached_tokens": u("cached"), "new_prompt_tokens": u("prompt") - u("cached"),
        "output_tokens": u("output"), "reasoning_tokens": u("reasoning"),
        "plans": sum(1 for e in events if e.type == "plan_committed"),
        "interpretations": sum(1 for e in events if e.type == "interpretation"),
        "gpu_s": round(gpu_s, 2), "gpus": GPUS, "inr_per_gpu_hour": INR_PER_GPU_HOUR,
        "agent_inr": round(gpu_s / 3600 * INR_PER_GPU_HOUR, 4),
    }


async def run_query(dataset: str, query: str, effort: str | None, seed: int, hints: bool, tag: str) -> Path:
    ds_dir = DAB / f"query_{dataset}"
    q_dir = ds_dir / f"query{query}"
    question = json.loads((q_dir / "query.json").read_text())
    engines = engines_for(ds_dir)
    if "postgres" in engines or "mongo" in engines:
        from grader.dbload import ensure_loaded       # loads Postgres / Mongo databases for this dataset
        ensure_loaded(ds_dir)

    run_id = new_id("run")
    run_dir = RUNS / dataset / f"q{query}" / run_id
    rec = RunRecorder(run_dir, run_id)
    llm = LLM(rec, seed=seed)
    effort = effort or llm.default_effort
    env = environment(model=llm.model)
    rec.emit(RunStart(dataset=dataset, query_id=str(query), question=question, model=llm.model,
                      reasoning_effort=effort, reflection=False, hints=hints,
                      server_flags=[f"{k}={os.environ[k]}" for k in FLAGS if k in os.environ] + [f"seed={seed}", f"tag={tag}"],
                      **env))
    t0, answer, reason = time.time(), "", "error"
    try:
        async with ToolHub(rec, ds_dir) as hub:
            res = await DataAgent(rec, llm, hub, ds_dir, effort=effort, hints=hints).run(question)
        answer, reason = res.answer, res.end_reason
    except Exception as e:
        reason = f"error: {type(e).__name__}: {e}"
        traceback.print_exc()
    wall = time.time() - t0

    try:
        ok, why = load_validator(q_dir)(answer)
        validation = {"is_valid": bool(ok), "reason": why}
    except Exception as e:
        validation = {"is_valid": False, "reason": f"validator error: {type(e).__name__}: {e}"}

    from grader.record import load_events
    rec.emit(RunEnd(answer=answer, end_reason=reason, totals=totals(load_events(run_dir), wall), validation=validation))
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hints", action="store_true")
    ap.add_argument("--tag", default="dev")
    a = ap.parse_args()
    run_dir = asyncio.run(run_query(a.dataset, a.query, a.effort, a.seed, a.hints, a.tag))

    from grader.record import load_events
    end = load_events(run_dir)[-1]
    t = end.totals
    print(f"\nrun: {run_dir}")
    print(f"answer: {end.answer[:300]}")
    print(f"end: {end.end_reason} | valid: {end.validation['is_valid']} ({end.validation['reason']})")
    print(f"time: wall {t['wall_s']}s, llm {t['llm_s']}s, tools {t['tool_s']}s | calls: llm {t['llm_calls']}, "
          f"tools {t['tool_calls']} ({t['tool_errors']} failed) | plans {t['plans']}, interpretations {t['interpretations']}")
    print(f"tokens: new {t['new_prompt_tokens']}, cached {t['cached_tokens']}, out {t['output_tokens']} "
          f"(reasoning {t['reasoning_tokens']}) | GPU {t['gpu_s']}s = Rs {t['agent_inr']}")


if __name__ == "__main__":
    main()
