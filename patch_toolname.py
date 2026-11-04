"""Patch: strip harmony control tokens that leak into tool-call names.

Run from ~/abstraction-grader:  python patch_toolname.py
vLLM's openai tool parser sometimes returns gpt-oss tool names with a channel token attached, e.g.
'postgres__run_query<|channel|>commentary', which the harness then rejects as an unknown tool. The name is
cut at the first '<|' and the change is recorded in the call's `coerced` list (visible in the record, L0).
Names without '<|' are untouched, so other models are unaffected.
"""
from pathlib import Path

L = "grader/llm.py"
s = Path(L).read_text()
if "control token leaked" in s:
    raise SystemExit("already patched; nothing to do")
old = '''        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"'''
new = '''        for tc in msg.tool_calls or []:
            name, renamed = tc.function.name or "", []
            if "<|" in name:                           # control token leaked into the name (vLLM openai parser)
                clean = name.split("<|")[0].strip()
                renamed.append(f"name: {name!r} -> {clean!r}")
                name = clean
            raw = tc.function.arguments or "{}"'''
for o, n in ((old, new),
             ('''if t.get("function", {}).get("name") == tc.function.name), None)''',
              '''if t.get("function", {}).get("name") == name), None)'''),
             ('''            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args, "parse_error": err,
                          "coerced": coerced})''',
              '''            calls.append({"id": tc.id, "name": name, "arguments": args, "parse_error": err,
                          "coerced": renamed + coerced})''')):
    assert s.count(o) == 1, f"expected text not found exactly once:\n{o[:120]}"
    s = s.replace(o, n)
Path(L).write_text(s)
print("patched: llm.py (control tokens stripped from tool names, recorded in coerced)")
