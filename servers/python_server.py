"""MCP server that runs Python snippets for the agent.

Each call runs in a fresh subprocess inside the run's work folder, with limits.
Earlier tool results are readable via load("<call_id>") from GRADER_RESULTS_DIR.

Env vars:
  GRADER_RUN_DIR        run folder; code runs in <run>/work, results in <run>/results
  GRADER_EXEC_TIMEOUT   seconds per snippet (default 600)
"""
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from servers.common import CallLog

RUN_DIR = Path(os.path.expanduser(os.environ["GRADER_RUN_DIR"]))
WORK_DIR = RUN_DIR / "work"
RESULTS_DIR = RUN_DIR / "results"
TIMEOUT = int(os.environ.get("GRADER_EXEC_TIMEOUT", "600"))
MAX_OUTPUT = 20_000          # characters of stdout/stderr returned to the model
MAX_FILE_BYTES = 1 << 30     # 1 GB per file written

WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREAMBLE = '''
import json as _json, os as _os
from pathlib import Path as _Path
_RESULTS = _Path(_os.environ["GRADER_RESULTS_DIR"])

def load(call_id):
    """Return the result of an earlier tool call, by its call id."""
    p = _RESULTS / f"{call_id}.json"
    if not p.exists():
        raise KeyError(f"No stored result {call_id!r}. Available: {available()}")
    return _json.loads(p.read_text())

def available():
    """List the call ids of stored results."""
    return sorted(q.stem for q in _RESULTS.glob("*.json"))
'''

mcp = MCPServer("python")
log = CallLog("python")


class ExecResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 5, TIMEOUT + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))


def _clip(s: str) -> tuple[str, bool]:
    return (s, False) if len(s) <= MAX_OUTPUT else (s[-MAX_OUTPUT:], True)


@mcp.tool()
def run_python(code: str, call_id: str = "") -> ExecResult:
    """Run a Python snippet (pandas available) and return what it prints.
    Use load("<call_id>") to read the result of an earlier tool call and available()
    to list them. Print anything you want to see; nothing else is returned."""
    start = time.time()
    script = WORK_DIR / f"snippet_{call_id or int(start * 1000)}.py"
    script.write_text(PREAMBLE + "\n" + code)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(WORK_DIR),
           "GRADER_RESULTS_DIR": str(RESULTS_DIR), "PYTHONHASHSEED": "0"}
    try:
        proc = subprocess.run([sys.executable, "-I", str(script)], cwd=WORK_DIR, env=env,
                              capture_output=True, text=True, timeout=TIMEOUT,
                              preexec_fn=_limits)
    except subprocess.TimeoutExpired:
        log.write({"call_id": call_id, "tool": "run_python", "ok": False, "code": code,
                   "script": str(script), "error": f"timeout after {TIMEOUT}s",
                   "duration_s": time.time() - start})
        raise ToolError(f"Code timed out after {TIMEOUT} seconds.")
    stdout, t1 = _clip(proc.stdout)
    stderr, t2 = _clip(proc.stderr)
    ok = proc.returncode == 0
    log.write({"call_id": call_id, "tool": "run_python", "ok": ok, "code": code,
               "script": str(script), "exit_code": proc.returncode,
               "stdout_chars": len(proc.stdout), "stderr_tail": proc.stderr[-2000:],
               "duration_s": time.time() - start})
    if not ok:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise ToolError(f"Code failed (exit {proc.returncode}).\n"
                        f"stdout (last part):\n{stdout[-2000:]}\nerror:\n{tail}")
    return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode,
                      truncated=t1 or t2)


if __name__ == "__main__":
    mcp.run()
