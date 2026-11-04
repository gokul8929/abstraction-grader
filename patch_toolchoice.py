"""Patch: force a tool call on gpt-oss orchestrator turns (tool_choice="required").

Run from ~/abstraction-grader after patch_toolname.py:  python patch_toolchoice.py
Replays of the empty orchestrator turns showed gpt-oss writing the tool name as a harmony channel,
'<|channel|>final_answer<|message|>{"answer": ...}<|return|>', which vLLM drops (not analysis/commentary/final),
so the harness saw an empty turn. With tool_choice="required" every replay produced a valid final_answer call.
A profile lists the agent roles that get it (gpt-oss: orchestrator); other profiles are unchanged.
Each llm_call event records the tool_choice it was sent with (L0).
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


M, L, R = "grader/models.py", "grader/llm.py", "grader/record.py"
if "force_tool_call" in Path(M).read_text():
    raise SystemExit("already patched; nothing to do")

sub(M, '''    resend_reasoning: bool = False            # pass the model's reasoning back on later turns (gpt-oss needs this)
''', '''    resend_reasoning: bool = False            # pass the model's reasoning back on later turns (gpt-oss needs this)
    force_tool_call: tuple[str, ...] = ()     # agent roles whose turns are sent with tool_choice="required"
''')
sub(M, '''        gpus=4, temperature=1.0, top_p=1.0, top_k=None, resend_reasoning=True,''',
       '''        gpus=4, temperature=1.0, top_p=1.0, top_k=None, resend_reasoning=True, force_tool_call=("orchestrator",),''')
sub(M, '''              "reasoning must be returned to the model between tool-call turns"),''',
       '''              "reasoning must be returned to the model between tool-call turns; orchestrator turns use "
              "tool_choice=required (otherwise it can write final_answer as a harmony channel, which vLLM drops)"),''')

sub(R, '''    max_tokens: int | None = None
    reasoning: str | None = None''', '''    max_tokens: int | None = None
    tool_choice: str | None = None                # "required" when the profile forces a tool call (L0)
    reasoning: str | None = None''')

sub(L, '''        if tools:
            kwargs["tools"] = tools
''', '''        if tools:
            kwargs["tools"] = tools
        tool_choice = "required" if tools and role in self.profile.force_tool_call else None
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
''')
sub(L, '''                    top_p=self.top_p, top_k=self.top_k, seed=self.seed, reasoning_effort=effort,
                    max_tokens=self.max_tokens)''', '''                    top_p=self.top_p, top_k=self.top_k, seed=self.seed, reasoning_effort=effort,
                    max_tokens=self.max_tokens, tool_choice=tool_choice)''')
print("patched: models.py (force_tool_call; gpt-oss orchestrator), llm.py (tool_choice), record.py (tool_choice recorded)")
