"""Patch: per-system prompt variants (condition B, tuned systems).

Run from ~/abstraction-grader after copying grader/prompts/ in:  python patch_prompts.py
A variant is a folder grader/prompts/<name>/ with orchestrator.md and/or worker.md. Their text is APPENDED to
the shared base prompts (orchestrator_v2 / worker_v2), so every system keeps the same core instructions and
the tuning is visible as a separate block. Template ids become e.g. 'orchestrator_v2+gptoss_r1@1a2b3c4d'
(variant name + hash of the addendum), so each prompt revision is identifiable in the record (L2).
Select with --prompts <name> on grader.run / grader.batch; no variant = the base prompts (as before).
"""
from pathlib import Path


def sub(p, old, new, count=1):
    s = Path(p).read_text()
    assert s.count(old) == count, f"{p}: expected {count}x, found {s.count(old)}:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


A, U, B = "grader/agent.py", "grader/run.py", "grader/batch.py"
if "def prompt_variant" in Path(A).read_text():
    raise SystemExit("already patched; nothing to do")

sub(A, 'KINDS = ["source_choice",', '''PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def prompt_variant(name: str | None) -> tuple[str, str, str, str]:
    """(orchestrator system text, its template id, worker system text, its template id) for a variant."""
    import hashlib                                   # local: does not depend on the module's import block
    if not name:
        return ORCH_SYSTEM, ORCH_TEMPLATE, WORKER_SYSTEM, WORKER_TEMPLATE
    d = PROMPTS_DIR / name
    if not d.is_dir():
        raise ValueError(f"unknown prompt variant {name!r}: no folder {d}")
    out = []
    for base, tid, fname in ((ORCH_SYSTEM, ORCH_TEMPLATE, "orchestrator.md"), (WORKER_SYSTEM, WORKER_TEMPLATE, "worker.md")):
        f = d / fname
        add = f.read_text().strip() if f.exists() else ""
        if add:
            out += [base + "\\n\\n" + add, f"{tid}+{name}@{hashlib.sha256(add.encode()).hexdigest()[:8]}"]
        else:
            out += [base, tid]
    return tuple(out)


KINDS = ["source_choice",''')
sub(A, '''                 effort: str | None = None, hints: bool = False, max_worker_turns: int = 25,''',
       '''                 effort: str | None = None, hints: bool = False, max_worker_turns: int = 25, prompts: str | None = None,''')
sub(A, '''        self.rec, self.llm, self.hub, self.effort = rec, llm, hub, effort''',
       '''        self.rec, self.llm, self.hub, self.effort = rec, llm, hub, effort
        self.orch_system, self.orch_tid, self.worker_system, self.worker_tid = prompt_variant(prompts)''')
sub(A, "template_id=ORCH_TEMPLATE", "template_id=self.orch_tid", count=2)
sub(A, "content=ORCH_SYSTEM)", "content=self.orch_system)")
sub(A, "template_id=WORKER_TEMPLATE", "template_id=self.worker_tid", count=3)
sub(A, "content=WORKER_SYSTEM)", "content=self.worker_system)")

sub(U, '''async def run_query(dataset: str, query: str, effort: str | None, seed: int, hints: bool, tag: str,
                    model: str | None = None) -> Path:''', '''async def run_query(dataset: str, query: str, effort: str | None, seed: int, hints: bool, tag: str,
                    model: str | None = None, prompts: str | None = None) -> Path:''')
sub(U, '[f"seed={seed}", f"tag={tag}", f"model={llm.profile.key}"]',
       '[f"seed={seed}", f"tag={tag}", f"model={llm.profile.key}", f"prompts={prompts or \'base\'}"]')
sub(U, "DataAgent(rec, llm, hub, ds_dir, effort=effort, hints=hints)",
       "DataAgent(rec, llm, hub, ds_dir, effort=effort, hints=hints, prompts=prompts)")
sub(U, '''    ap.add_argument("--tag", default="dev")''',
       '''    ap.add_argument("--tag", default="dev")
    ap.add_argument("--prompts", default=None, help="prompt variant folder under grader/prompts (default: base prompts)")''')
sub(U, "asyncio.run(run_query(a.dataset, a.query, a.effort, a.seed, a.hints, a.tag, a.model))",
       "asyncio.run(run_query(a.dataset, a.query, a.effort, a.seed, a.hints, a.tag, a.model, a.prompts))")

sub(B, '''               "--seed", str(s), "--tag", a.tag] + (["--hints"] if a.hints else [])''',
       '''               "--seed", str(s), "--tag", a.tag] + (["--hints"] if a.hints else []) \\
              + (["--prompts", a.prompts] if a.prompts else [])''')
sub(B, '''    ap.add_argument("--hints", action="store_true")''',
       '''    ap.add_argument("--hints", action="store_true")
    ap.add_argument("--prompts", default=None, help="prompt variant folder under grader/prompts")''')
print("patched: agent.py, run.py, batch.py (prompt variants via --prompts)")
