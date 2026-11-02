"""Check a smoke batch before the real one starts: every run must finish normally and use tools.

  python -m grader.smoke_check --tag smoke-gptoss --model gpt-oss-120b

Exit code 0 = fine to continue; 1 = stop (prints why). Correctness is NOT required here, only that the
model talks to the harness properly: tool calls parse, the run ends with an answer, no harness errors.
"""
from __future__ import annotations

import argparse
import sys

from grader.batch import finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    done = finished(a.tag, include_errors=True, model=a.model)
    if not done:
        print(f"SMOKE FAIL: no finished runs for tag {a.tag}")
        sys.exit(1)
    bad = []
    for (d, q, e, s), r in sorted(done.items()):
        print(f"{d} q{q} {e:8s} seed {s}: end={r['end']!r} valid={r['valid']} tools={r['tool_calls']} "
              f"(failed {r['tool_errors']}) llm={r['llm_calls']} (errors {r['llm_errors']}) "
              f"reasoning_tokens={r['reasoning_tokens']} wall={r['wall_s']:.0f}s")
        if r["end"] != "answered":
            bad.append(f"{d} q{q} {e}: ended {r['end']!r}")
        elif r["tool_calls"] == 0:
            bad.append(f"{d} q{q} {e}: answered without a single tool call (tool calling not working?)")
        elif r["llm_errors"] > 0:
            bad.append(f"{d} q{q} {e}: {r['llm_errors']} model-call errors")
    if bad:
        print("SMOKE FAIL:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
