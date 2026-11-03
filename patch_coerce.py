"""Patch: convert tool-call argument types to the declared schema before the agent or a server sees them.

Run from ~/abstraction-grader after patch_models.py, with grader/argfix.py copied in:  python patch_coerce.py
Fixes the GLM crashes (plan `steps` arriving as a JSON string) and the "rejected arguments" errors
(e.g. a Mongo pipeline arriving as a list where the schema says string). Every conversion is recorded in
the llm_call event's tool call entry under "coerced".
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


assert Path("grader/argfix.py").exists(), "copy grader/argfix.py into the repo first"
L = "grader/llm.py"
if "coerce_args" in Path(L).read_text():
    raise SystemExit("already patched; nothing to do")
sub(L, "from grader import models as model_profiles\n",
       "from grader import models as model_profiles\nfrom grader.argfix import coerce_args\n")
sub(L, '            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "parse_error": err})',
       '''            schema = next((t["function"].get("parameters") for t in (tools or [])
                           if t.get("function", {}).get("name") == tc.function.name), None)
            args, coerced = coerce_args(args, schema)
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "parse_error": err,
                          "coerced": coerced})''')
sub(L, '        return MessagePart(role="assistant", origin="model", content=self.content,',
       '        return MessagePart(role="assistant", origin="model", content=self.content,  # coerced args are resent')
print("patched: llm.py (argument types follow the tool schema; conversions recorded)")
