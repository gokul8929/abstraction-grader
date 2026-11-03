"""Rescore runs that ended 'no_final_answer' because the model wrote its answer as plain text.

  python -m grader.rescore_text --tag pilot2-gptoss --model gpt-oss-120b

Some models (gpt-oss) give the final answer as ordinary text instead of calling final_answer. The harness
used to stop at that point with 'no_final_answer'. From now on it accepts that text as the answer, ending
the run as 'answered_text' (patch_textanswer.py). Nothing before that point changes, so for runs already
recorded the outcome can be computed from the record: the answer is the orchestrator's last non-empty
message. This script does that, scores it with the query's own validate.py, and writes
runs/_batch/<tag>_rescored.json. The original events.jsonl files are not modified.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from grader.batch import finished
from grader.record import load_events
from grader.run import DAB, RUNS, load_validator


def text_answer(run_dir) -> str:
    calls = [e for e in load_events(run_dir) if e.type == "llm_call" and e.role == "orchestrator" and not e.error]
    for e in reversed(calls):
        if (e.content or "").strip():
            return e.content.strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    done = finished(a.tag, include_errors=True, model=a.model)
    out, by = [], defaultdict(lambda: [0, 0, 0])          # setting -> [runs, valid before, valid after]
    for (d, q, e, s), r in sorted(done.items()):
        run_dir = RUNS / d / f"q{q}" / r["run"]
        valid, end, answer, why = r["valid"], r["end"], r["answer"], None
        if r["end"] == "no_final_answer":
            answer = text_answer(run_dir)
            if answer:
                ok, why = load_validator(DAB / f"query_{d}" / f"query{q}")(answer)
                valid, end = bool(ok), "answered_text"
        out.append({"dataset": d, "query": q, "effort": e, "seed": s, "run": r["run"], "end_original": r["end"],
                    "end": end, "valid_original": r["valid"], "valid": valid, "answer": answer, "reason": why})
        for k in (e, "ALL"):
            by[k][0] += 1; by[k][1] += r["valid"]; by[k][2] += valid
    path = RUNS / "_batch" / f"{a.tag}_rescored.json"
    path.write_text(json.dumps(out, indent=1))
    changed = [x for x in out if x["end"] == "answered_text"]
    print(f"{len(changed)} runs had a plain-text answer; {sum(x['valid'] for x in changed)} of them correct\n")
    for x in changed:
        print(f"{x['dataset']:10s} q{x['query']} {x['effort']:7s} seed {x['seed']}  valid={x['valid']}  "
              f"answer: {x['answer'][:90]!r}")
    print(f"\n{'setting':8s} {'runs':>4s} {'before':>7s} {'after':>6s}")
    for k in sorted(k for k in by if k != "ALL") + ["ALL"]:
        n, b, c = by[k]
        print(f"{k:8s} {n:4d} {b:4d}/{n:<3d} {c:3d}/{n}")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
