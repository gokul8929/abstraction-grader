"""Run a grid of DAB queries unattended, resumable, then summarise.

  python -m grader.batch --tag pilot1                      # 13 pilot queries x low/medium/xhigh x seed 0
  python -m grader.batch --tag pilot1 --dry-run            # show what would run
  python -m grader.batch --tag pilot1 --summary            # table + per-effort aggregates

Each run is a separate process (python -m grader.run), so one crash never stops the batch.
Order is shuffled with a fixed seed so time-of-day effects do not line up with one condition.
Re-running the same command skips runs that already finished (safe after an allocation ends).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from grader import models as model_profiles
from grader.record import load_events
from grader.run import DAB, RUNS


def queries(dataset: str) -> list[str]:
    return sorted((m.group(1) for p in (DAB / f"query_{dataset}").iterdir()
                   if p.is_dir() and (m := re.fullmatch(r"query(\d+)", p.name))), key=int)


def finished(tag: str, include_errors: bool = False, model: str | None = None) -> dict[tuple, dict]:
    """(dataset, query, effort, seed) -> summary, for runs of this tag that reached run_end.
    Runs that ended in a harness error (e.g. vLLM down) are left out unless include_errors, so a resume redoes them."""
    done = {}
    for ev_file in RUNS.glob("*/q*/run-*/events.jsonl"):
        try:
            events = load_events(ev_file.parent)
        except Exception:
            continue
        start, end = events[0], events[-1]
        if start.type != "run_start" or end.type != "run_end":
            continue
        flags = dict(f.split("=", 1) for f in start.server_flags if "=" in f)
        if flags.get("tag") != tag or (end.end_reason.startswith("error") and not include_errors):
            continue
        if model and flags.get("model", "qwen3.8-27b") != model:
            continue
        key = (start.dataset, start.query_id, start.reasoning_effort, int(flags.get("seed", 0)))
        cand = {"run": ev_file.parent.name, "valid": end.validation["is_valid"], "end": end.end_reason,
                "answer": end.answer, "_ts": start.ts, **end.totals}
        old = done.get(key)
        rank = lambda r: (not r["end"].startswith("error"), r["_ts"])   # finished beats error; then newest
        if old is None or rank(cand) > rank(old):
            done[key] = cand
    return done


def vllm_up(url: str = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")) -> bool:
    try:
        return urllib.request.urlopen(url + "/models", timeout=10).status == 200
    except Exception:
        return False


def grid(datasets, efforts, seeds, tag, only=None):
    cells = [(d, q, e, s) for d in datasets for q in queries(d) for e in efforts for s in seeds
             if not only or f"{d}:{q}" in only]
    random.Random(tag).shuffle(cells)
    return cells


def run_batch(a) -> None:
    cells = grid(a.datasets, a.efforts, a.seeds, a.tag, a.queries)
    done = finished(a.tag, model=a.model)
    todo = [c for c in cells if c not in done]
    print(f"[batch {a.tag}] {len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run", flush=True)
    log_dir = RUNS / "_batch"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{a.tag}.log", "a")
    for i, (d, q, e, s) in enumerate(todo, 1):
        cmd = [sys.executable, "-m", "grader.run", "--model", a.model, "--dataset", d, "--query", q, "--effort", e,
               "--seed", str(s), "--tag", a.tag] + (["--hints"] if a.hints else [])
        if a.dry_run:
            print("  would run:", " ".join(cmd[2:]))
            continue
        if not vllm_up():
            print(f"[batch {a.tag}] vLLM is not answering - stopping. Restart it, then rerun this command to resume.")
            break
        t = time.time()
        print(f"[{i}/{len(todo)}] {d} q{q} effort={e} seed={s} ...", end=" ", flush=True)
        log.write(f"\n===== {time.strftime('%F %T')} {d} q{q} {e} seed={s}\n"); log.flush()
        try:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout,
                               cwd=Path(__file__).resolve().parent.parent)
            status = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            status = f"TIMEOUT after {a.timeout}s"
        r = finished(a.tag, model=a.model).get((d, q, e, s))
        print(f"{status}, {time.time() - t:.0f}s, valid={r['valid'] if r else 'no run_end'}", flush=True)
    log.close()


def summary(a) -> None:
    done = finished(a.tag, include_errors=True, model=a.model)
    if not done:
        print(f"no finished runs for tag {a.tag}"); return
    print(f"{'dataset':11s} {'q':>2s} {'effort':6s} {'seed':>4s} {'valid':5s} {'wall_s':>7s} {'Rs':>6s} "
          f"{'think':>6s} {'llm':>4s} {'tool':>4s} {'fail':>4s} {'plans':>5s} {'interp':>6s}  end")
    for (d, q, e, s), r in sorted(done.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2], kv[0][3])):
        print(f"{d:11s} {q:>2s} {e:6s} {s:>4d} {str(r['valid']):5s} {r['wall_s']:7.0f} {r['agent_inr']:6.2f} "
              f"{r['reasoning_tokens']:6d} {r['llm_calls']:4d} {r['tool_calls']:4d} {r['tool_errors']:4d} "
              f"{r['plans']:5d} {r['interpretations']:6d}  {r['end']}")
    by = defaultdict(list)
    for (d, q, e, s), r in done.items():
        by[e].append(r); by["ALL"].append(r)
    print(f"\n{'effort':6s} {'runs':>4s} {'correct':>8s} {'mean_s':>7s} {'mean_Rs':>8s} {'mean_think':>10s} {'total_Rs':>9s}")
    for e in sorted({k[2] for k in done}) + ["ALL"]:
        rs = by.get(e)
        if not rs:
            continue
        n = len(rs)
        print(f"{e:6s} {n:4d} {sum(r['valid'] for r in rs):4d}/{n:<3d} {sum(r['wall_s'] for r in rs) / n:7.0f} "
              f"{sum(r['agent_inr'] for r in rs) / n:8.2f} {sum(r['reasoning_tokens'] for r in rs) / n:10.0f} "
              f"{sum(r['agent_inr'] for r in rs):9.2f}")
    if a.json:
        out = RUNS / "_batch" / f"{a.tag}_summary.json"
        out.write_text(json.dumps([{"dataset": d, "query": q, "effort": e, "seed": s, **r}
                                   for (d, q, e, s), r in done.items()], indent=1))
        print(f"\nwritten: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="stockindex,bookreview,yelp", type=lambda s: s.split(","))
    ap.add_argument("--model", default="qwen3.8-27b", help="grader.models profile key")
    ap.add_argument("--efforts", default=None, type=lambda s: s.split(","), help="default: all of the profile's settings")
    ap.add_argument("--queries", default=None, type=lambda s: s.split(","), help="dataset:query list, e.g. yelp:1 (smoke tests)")
    ap.add_argument("--seeds", default="0", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--hints", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--json", action="store_true", help="with --summary: also write a JSON file")
    a = ap.parse_args()
    a.efforts = a.efforts or list(model_profiles.get(a.model).settings)
    summary(a) if a.summary else run_batch(a)


if __name__ == "__main__":
    main()
