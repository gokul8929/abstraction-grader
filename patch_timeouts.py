"""Patch: a run killed at the batch's per-run time limit is recorded as end_reason "timeout" (a failure).

Run from ~/abstraction-grader after copying grader/timeouts.py in:  python patch_timeouts.py
Before, the killed run had no run_end: summaries left it out and a resubmit retried it.
"""
from pathlib import Path

B = "grader/batch.py"
s = Path(B).read_text()
if "close_timed_out" in s:
    raise SystemExit("already patched; nothing to do")
old = '''        except subprocess.TimeoutExpired:
            status = f"TIMEOUT after {a.timeout}s"'''
new = '''        except subprocess.TimeoutExpired:
            status = f"TIMEOUT after {a.timeout}s"
            from grader.timeouts import close_timed_out
            close_timed_out(a.tag, a.model, a.timeout, older_than=0, key=(d, q, e, s))'''
assert s.count(old) == 1, "timeout handler not found"
Path(B).write_text(s.replace(old, new))
print("patched: batch.py (timed-out runs get run_end 'timeout')")
