"""Show what the orchestrator actually emitted on its last turns, for runs that ended without an answer.

  python -m grader.diag_orch --tag tune-gptoss_r2            # runs ending no_final_answer / answered_text
  python -m grader.diag_orch --tag tune-gptoss_r2 --all      # every run

For each orchestrator model call after the last executed plan: finish_reason, the tool calls it made
(names exactly as the server parsed them), the visible content, and the tail of its reasoning.
Read-only. Output also goes to runs/_batch/<tag>_orch.md.
"""
from __future__ import annotations

import argparse

from grader.batch import finished
from grader.record import load_events
from grader.run import RUNS


def _tail(t: str | None, n: int) -> str:
    t = (t or "").strip()
    return t if len(t) <= n else "…" + t[-n:]


def describe(run_dir) -> list[str]:
    ev = load_events(run_dir)
    orch = [e for e in ev if e.type == "llm_call" and e.agent_id == "orchestrator"]
    # calls after the last plan was executed = calls whose input ends with the step-summary report
    last_report = max((i for i, e in enumerate(orch)
                       if "Step summaries:" in (e.messages[-1].content or "")), default=None)
    tail = orch[last_report:] if last_report is not None else orch[-3:]
    out = []
    for e in tail:
        names = [f"{c.get('name')!r}" + (f" parse_error={c.get('parse_error')!r}" if c.get("parse_error") else "")
                 for c in e.tool_calls]
        out += [f"- call {e.llm_call_id} · finish_reason={e.finish_reason} · attempt={e.attempt}"
                f"{' · ERROR ' + e.error[:200] if e.error else ''}",
                f"  - tool_calls: {', '.join(names) or '(none)'}",
                f"  - content: {(e.content or '').strip()[:300]!r}",
                f"  - reasoning tail: {_tail(e.reasoning, 600)!r}",
                f"  - last input message: {e.messages[-1].role}/{e.messages[-1].origin}: "
                f"{(e.messages[-1].content or '')[:160]!r}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    done = finished(a.tag, include_errors=True)
    parts = [f"# {a.tag}: orchestrator's last turns", ""]
    for (d, q, e, s), r in sorted(done.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2], kv[0][3])):
        run_dir = RUNS / d / f"q{q}" / r["run"]
        end = next((x for x in reversed(load_events(run_dir)) if x.type == "run_end"), None)
        reason = getattr(end, "end_reason", None) or getattr(end, "reason", None) or "?"
        if not a.all and reason not in ("no_final_answer", "answered_text"):
            continue
        parts += [f"## {d} q{q} · {e} · seed {s} · end={reason}", f"`{run_dir}`", ""] + describe(run_dir) + [""]
    text = "\n".join(parts)
    out = RUNS / "_batch" / f"{a.tag}_orch.md"
    out.write_text(text)
    print(text)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
