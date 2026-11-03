"""Patch: malformed plans never crash a run.

Run from ~/abstraction-grader after patch_coerce.py:  python patch_robust.py
  * argfix.py: a JSON-looking string that is not valid JSON (e.g. Python-style quotes {'a': 1}) is also
    converted, via ast.literal_eval (safe: literals only), and the conversion is recorded as such.
  * agent.py: plan validation checks the shape of every step first (an object with step_id, description,
    and list-valued databases / depends_on) and returns an error message to the model instead of raising.
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


F, A = "grader/argfix.py", "grader/agent.py"
if "literal_eval" in Path(F).read_text():
    raise SystemExit("already patched; nothing to do")

sub(F, "import json\n", "import ast\nimport json\n")
sub(F, '''def _coerce(v: Any, schema: dict | None, path: str, notes: list[str]) -> Any:''',
       '''def _parse(text: str, want: type, path: str, notes: list[str], label: str):
    """JSON first; then a Python literal (single quotes, True/None). Returns the value or None."""
    try:
        v = json.loads(text)
        how = "JSON string"
    except json.JSONDecodeError:
        try:
            v = ast.literal_eval(text.strip())
            how = "Python-literal string"
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if isinstance(v, want):
        notes.append(f"{path}: {how} -> {label}")
        return v
    return None


def _coerce(v: Any, schema: dict | None, path: str, notes: list[str]) -> Any:''')
sub(F, '''        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return v
            if isinstance(parsed, list):
                notes.append(f"{path}: JSON string -> array")
                v = parsed''', '''        if isinstance(v, str):
            parsed = _parse(v, list, path, notes, "array")
            if parsed is None:
                return v
            v = parsed''')
sub(F, '''        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return v
            if isinstance(parsed, dict):
                notes.append(f"{path}: JSON string -> object")
                v = parsed''', '''        if isinstance(v, str):
            parsed = _parse(v, dict, path, notes, "object")
            if parsed is None:
                return v
            v = parsed''')

sub(A, '''    def _validate(self, steps: list[dict]) -> str | None:
        ids = [s.get("step_id") for s in steps]''', '''    def _validate(self, steps: list[dict]) -> str | None:
        shape = ("steps must be a JSON array of objects, each with step_id (string), description (string), "
                 "databases (array of names) and depends_on (array of step_ids).")
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            return shape
        for s in steps:
            if not isinstance(s.get("description"), str) or not s["description"].strip():
                return "Every step needs a non-empty description. " + shape
            if not isinstance(s.get("databases", []), list) or not isinstance(s.get("depends_on", []), list):
                return shape
        ids = [s.get("step_id") for s in steps]''')
print("patched: argfix.py (Python-literal strings), agent.py (plan shape checked before use)")
