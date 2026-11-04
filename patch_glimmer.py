"""Patch: add the Muse Glimmer-30B profile (Meta, Aug 2026; replaces GLM-4.5-Air in the paper).

Run from ~/abstraction-grader after patch_toolchoice.py:  python patch_glimmer.py
Dense 29.6B (text decoder + 1.8B vision encoder, skipped with --language-model-only), BF16 on 2 GPUs.
Reasoning strength low/medium/high/xhigh via the chat-template kwarg reasoning_strength (Meta deploy docs);
we use low/medium/xhigh like Qwen (lowest, middle, highest). Sampling per the model card: T=1.0, top_p 0.95,
top_k 64. vLLM parsers: muse_glimmer (tool calls and reasoning; vLLM >= 0.28).
"""
from pathlib import Path

M = "grader/models.py"
s = Path(M).read_text()
if '"muse-glimmer-30b"' in s:
    raise SystemExit("already patched; nothing to do")
old = '''              "nothink = no reasoning text, so nothing exists at L3"),
]}'''
new = '''              "nothink = no reasoning text, so nothing exists at L3"),
    ModelProfile(
        key="muse-glimmer-30b", hf_id="meta-models/Muse-Glimmer-30B",
        settings={e: {"chat_template_kwargs": {"reasoning_strength": e}} for e in ("low", "medium", "xhigh")},
        default_setting="xhigh",
        gpus=2, temperature=1.0, top_p=0.95, top_k=64,
        serve_args=["--tensor-parallel-size 2", "--language-model-only", "--generation-config auto",
                    "--tool-call-parser muse_glimmer", "--reasoning-parser muse_glimmer"] + COMMON,
        notes="BF16 dense; reasoning strength via chat-template kwarg reasoning_strength (reasoning cannot be "
              "turned off); sampling per the model card; Meta notes outputs are not reproducible even with a seed"),
]}'''
assert s.count(old) == 1, "end of PROFILES list not found"
Path(M).write_text(s.replace(old, new))
print("patched: models.py (muse-glimmer-30b profile)")
