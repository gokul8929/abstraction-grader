"""Export every run of a batch as a BLINDED review packet, split by disclosure level.

  python -m grader.export_review --tag pilot1
    -> runs/_batch/pilot1_review/all_blind.md   (what the reviewer reads: no ground truth, no verdicts)
    -> runs/_batch/pilot1_review/key.json       (run ID -> dataset, query, effort, valid, ground truth; keep away from reviewers)

Each run gets a random ID (R01..), and its record is laid out in reveal order:
  START  question + answer                               (D3 / O2 / L3 start point: result, outcome, reasoning)
  [D3]   result rows the answer rests on
  [L3]   the orchestrator's final reasoning
  [D2]   interpretations (with source and evidence)
  [O1-Spec] committed plan(s) and revisions
  [O1-Exec / O0] tool-call trace, grouped by plan step
  [D1]   which databases/results each step used (access path, lineage)
  [L2/L1/L0] how worker prompts were written and by whom; sampling settings
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from grader.batch import finished
from grader.record import load_events
from grader.run import DAB, RUNS


def _cut(text, n):
    text = str(text or "").strip()
    return text if len(text) <= n else text[:n] + " …"


def _result_text(run_dir: Path, call_id: str, n: int) -> str:
    f = run_dir / "results" / f"{call_id}.json"
    if not f.exists():
        return f"({call_id}: no stored result)"
    return _cut(f.read_text(), n)


def packet(run_dir: Path, rid: str) -> str:
    ev = load_events(run_dir)
    start, end = ev[0], ev[-1]
    llm = [e for e in ev if e.type == "llm_call" and not e.error]
    tools = [e for e in ev if e.type == "tool_call"]
    out = [f"## {rid}", "", f"**Question:** {start.question}", "", f"**Answer:** {_cut(end.answer, 1500) or '(empty)'}", ""]

    # D3: results the final step pointed to (finish_step result_call_ids), else the last successful result
    finish_ids = [cid for e in llm for tc in e.tool_calls if tc["name"] == "finish_step"
                  for cid in (tc["arguments"].get("result_call_ids") or [])]
    ok_ids = [t.call_id for t in tools if t.ok]
    shown = [c for c in finish_ids if (run_dir / "results" / f"{c}.json").exists()][-3:] or ok_ids[-1:]
    out.append("### [D3] Result rows the answer rests on")
    out += [f"- `{c}`: {_result_text(run_dir, c, 1200)}" for c in shown] or ["- (none)"]
    out.append("")

    orch = [e for e in llm if e.role == "orchestrator"]
    out.append("### [L3] Orchestrator's final reasoning")
    out.append(_cut(orch[-1].reasoning if orch else "", 1500) or "(none)")
    out.append("")

    out.append("### [D2] Interpretations")
    interps = [e for e in ev if e.type == "interpretation"]
    out += [f"- `{i.step_id}` {i.kind} · source {i.source or '?'} · *{i.subject}*: {_cut(i.statement, 400)}"
            for i in interps] or ["- (none recorded)"]
    out.append("")

    out.append("### [O1-Spec] Plan")
    for p in [e for e in ev if e.type == "plan_committed"]:
        why = f" (revision: {_cut(p.revision_reason, 200)})" if p.revises else ""
        out.append(f"Plan v{p.version}{why}")
        out += [f"- `{s.step_id}` [{', '.join(s.databases) or 'no DB'}]"
                f"{' after ' + ', '.join(s.depends_on) if s.depends_on else ''}: {_cut(s.description, 500)}"
                for s in p.steps]
    out.append("")

    out.append("### [O1-Exec / O0] Tool-call trace, by plan step")
    for t in tools:
        args = t.arguments.get("sql") or t.arguments.get("code") or t.arguments.get("pipeline") \
            or t.arguments.get("filter") or {k: v for k, v in t.arguments.items() if k != "db_name"}
        status = "ok" if t.ok else f"ERROR {_cut(t.error, 120)}"
        out.append(f"- `{t.step_id or 'unplanned'}` {t.server}.{t.tool}"
                   f"{' [' + t.arguments['db_name'] + ']' if 'db_name' in t.arguments else ''}: "
                   f"`{_cut(json.dumps(args) if not isinstance(args, str) else args, 400)}` → {status}")
    out.append("")

    out.append("### [D1] Access path and lineage")
    dbs = sorted({(t.server, t.arguments.get('db_name')) for t in tools if t.arguments.get('db_name')})
    out.append("- databases touched: " + (", ".join(f"{s}:{d}" for s, d in dbs) or "none"))
    loads = [(t.call_id, t.loaded_results) for t in tools if t.loaded_results]
    out += [f"- python `{c}` loaded results {', '.join(l)}" for c, l in loads] or ["- no python loads of stored results"]
    out.append("")

    out.append("### [L2 / L1 / L0] Prompt composition, authorship, sampling")
    first_worker = {}
    for e in llm:
        if e.role == "subagent" and e.agent_id not in first_worker:
            first_worker[e.agent_id] = e
    for a, e in first_worker.items():
        instr = [m.content for m in e.messages if m.origin == "orchestrator"]
        deps = [m for m in e.messages if m.origin == "subagent"]
        out.append(f"- {a}: instruction written by **orchestrator**: {_cut(instr[0] if instr else '', 300)}"
                   f"{' · + summaries from earlier steps' if deps else ''}")
    if llm:
        e = llm[0]
        out.append(f"- sampling: temperature {e.temperature}, top_p {e.top_p}, top_k {e.top_k}, seed {e.seed}, "
                   f"effort {e.reasoning_effort}, max_tokens {e.max_tokens}")
    out += ["", "---", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    done = finished(a.tag, include_errors=True)
    cells = sorted(done.items())
    random.Random(f"{a.tag}-review").shuffle(cells)
    out_dir = RUNS / "_batch" / f"{a.tag}_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, key = [f"# Blinded review packet: {a.tag} ({len(cells)} runs)", ""], {}
    for i, ((d, q, e, s), r) in enumerate(cells, 1):
        rid = f"R{i:02d}"
        run_dir = RUNS / d / f"q{q}" / r["run"]
        parts.append(packet(run_dir, rid))
        gt = DAB / f"query_{d}" / f"query{q}" / "ground_truth.csv"
        key[rid] = {"dataset": d, "query": q, "effort": e, "seed": s, "valid": r["valid"], "run": str(run_dir),
                    "ground_truth": gt.read_text()[:500] if gt.exists() else None}
    (out_dir / "all_blind.md").write_text("\n".join(parts))
    (out_dir / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(cells)} runs -> {out_dir}/all_blind.md (blinded) and key.json (answers; do not show reviewers)")


if __name__ == "__main__":
    main()
