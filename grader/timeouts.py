"""Close runs that the batch killed at its per-run time limit, so they count as failures.

  python -m grader.timeouts --tag B-glimmer --model muse-glimmer-30b             # list and close them
  python -m grader.timeouts --tag B-glimmer --model muse-glimmer-30b --dry-run   # only list

A run killed by `batch --timeout` never writes run_end, so summaries left it out and a resubmit retried it
(which would resample a slow configuration until it happens to finish). This appends a run_end with
end_reason "timeout", no answer and validation is_valid=False, plus totals over the events it did record.
Only runs whose last event is older than --older-than seconds are touched (default 600), so a run still
in progress is never closed. The batch calls close_timed_out() itself right after killing a run.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from grader import models
from grader.record import RunEnd, load_events
from grader.run import RUNS, totals


def close_timed_out(tag: str, model: str | None, limit_s: int, older_than: float = 600,
                    key: tuple | None = None, dry_run: bool = False) -> list[Path]:
    closed = []
    for f in RUNS.glob("*/q*/run-*/events.jsonl"):
        try:
            ev = load_events(f.parent)
        except Exception:
            continue
        if not ev or ev[0].type != "run_start" or ev[-1].type == "run_end":
            continue
        start = ev[0]
        flags = dict(x.split("=", 1) for x in start.server_flags if "=" in x)
        if flags.get("tag") != tag or (model and flags.get("model", "qwen3.8-27b") != model):
            continue
        k = (start.dataset, start.query_id, start.reasoning_effort, int(flags.get("seed", 0)))
        if key is not None and k != key:
            continue
        if time.time() - ev[-1].ts < older_than:
            continue                                     # still running (or only just killed: pass older_than=0)
        closed.append(f.parent)
        print(f"{'would close' if dry_run else 'closed'}: {f.parent}  {k}  events={len(ev)}")
        if dry_run:
            continue
        gpus = models.get(flags.get("model", "qwen3.8-27b")).gpus
        end = RunEnd(run_id=start.run_id, seq=ev[-1].seq + 1, ts=time.time(), answer="", end_reason="timeout",
                     totals=totals(ev, float(limit_s), gpus),       # killed at the limit
                     validation={"is_valid": False, "reason": f"no answer within the {limit_s}s per-run limit"})
        with open(f, "a") as fh:
            fh.write(end.model_dump_json() + "\n")
    return closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=1800, help="the per-run limit the batch used (for the record)")
    ap.add_argument("--older-than", type=float, default=600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = close_timed_out(a.tag, a.model, a.limit, a.older_than, dry_run=a.dry_run)
    print(f"{len(n)} run(s)")


if __name__ == "__main__":
    main()
