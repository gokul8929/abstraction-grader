"""Dump what happened in each wrong run of a batch, for reading and for the reviewer study.

  python -m grader.inspect_runs --tag pilot1              # wrong runs  -> runs/_batch/pilot1_wrong.md
  python -m grader.inspect_runs --tag pilot1 --all        # every run   -> runs/_batch/pilot1_all.md

Per run: question, DAB ground truth, the agent's answer and the validator's reason; the committed plan(s)
(O1-Spec); interpretations with their source (D2); step summaries the orchestrator saw; failed tool calls;
context-limit events. Read-only: nothing in the run folders is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grader.batch import finished
from grader.record import load_events
from grader.run import DAB, RUNS


def _cut(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " …"


def describe(run_dir: Path) -> str:
    ev = load_events(run_dir)
    start, end = ev[0], ev[-1]
    q_dir = DAB / f"query_{start.dataset}" / f"query{start.query_id}"
    gt_file = q_dir / "ground_truth.csv"
    gt = gt_file.read_text() if gt_file.exists() else "(no ground_truth.csv)"
    out = [f"## {start.dataset} q{start.query_id} · effort {start.reasoning_effort} · "
           f"{'CORRECT' if end.validation['is_valid'] else 'WRONG'}",
           f"`{run_dir}`", "",
           f"**Question:** {start.question}", "",
           f"**Ground truth:** `{_cut(gt, 300)}`", "",
           f"**Answer:** {_cut(end.answer, 600) or '(empty)'}", "",
           f"**Validator:** {end.validation['reason']} · **end:** {end.end_reason}", "",
           f"**Cost:** {end.totals['wall_s']} s, ₹{end.totals['agent_inr']}, "
           f"{end.totals['llm_calls']} LLM / {end.totals['tool_calls']} tool calls, "
           f"{end.totals['reasoning_tokens']} thinking tokens", ""]

    plans = [e for e in ev if e.type == "plan_committed"]
    for p in plans:
        why = f" — revises {p.revises}: {p.revision_reason}" if p.revises else ""
        out.append(f"**Plan v{p.version}**{why}")
        for s in p.steps:
            dep = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
            out.append(f"- `{s.step_id}` [{', '.join(s.databases) or 'no DB'}]{dep}: {_cut(s.description, 400)}")
        out.append("")

    interps = [e for e in ev if e.type == "interpretation"]
    if interps:
        out.append("**Interpretations (D2)**")
        for i in interps:
            ev_ids = f" · evidence {', '.join(i.evidence)}" if i.evidence else ""
            out.append(f"- `{i.step_id}` {i.kind} · source **{i.source or '?'}**{ev_ids} · *{i.subject}*: "
                       f"{_cut(i.statement, 300)}")
        out.append("")

    orch = [e for e in ev if e.type == "llm_call" and e.role == "orchestrator" and not e.error]
    if orch:
        reports = [m.content for m in orch[-1].messages if m.origin == "subagent"]
        if reports:
            out += ["**Step summaries the orchestrator saw (last round)**", "", "```",
                    _cut(reports[-1], 2500), "```", ""]

    failed = [e for e in ev if e.type == "tool_call" and not e.ok]
    if failed:
        out.append("**Failed tool calls**")
        for t in failed:
            out.append(f"- `{t.step_id}` {t.server}.{t.tool}: {_cut(t.error, 200)}")
        out.append("")

    ctx = [e for e in ev if e.type == "llm_call" and e.error and "context" in e.error]
    clamped = [e for e in ev if e.type == "llm_call" and not e.error and e.max_tokens and e.max_tokens < 32768]
    if ctx or clamped:
        out.append(f"**Context:** {len(clamped)} calls had their output allowance reduced; "
                   f"{len(ctx)} hit the limit" + (f" ({_cut(ctx[0].error, 120)})" if ctx else ""))
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--all", action="store_true", help="include correct runs too")
    a = ap.parse_args()
    done = finished(a.tag, include_errors=True)
    cells = sorted(done.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2], kv[0][3]))
    chosen = [(k, r) for k, r in cells if a.all or not r["valid"]]
    parts = [f"# {a.tag}: {'all' if a.all else 'wrong'} runs ({len(chosen)} of {len(cells)})", ""]
    for (d, q, e, s), r in chosen:
        parts += [describe(RUNS / d / f"q{q}" / r["run"]), "---", ""]
    out = RUNS / "_batch" / f"{a.tag}_{'all' if a.all else 'wrong'}.md"
    out.write_text("\n".join(parts))
    print(f"{len(chosen)} runs written to {out}")


if __name__ == "__main__":
    main()
