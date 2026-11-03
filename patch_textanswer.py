"""Patch: accept results that a model writes as plain text instead of calling the finishing tool.

Run from ~/abstraction-grader:  python patch_textanswer.py
Interface-only change, the same for every model:
  * orchestrator: if, after the plan has run, the model writes text instead of calling final_answer
    (or revise_plan) even after the reminders, that text becomes the answer; end reason 'answered_text'.
  * worker: if the model writes its step result as a JSON object {"summary": ..., "result_call_ids": [...]}
    instead of calling finish_step, that object finishes the step. A message 'harness:text_finish'
    is recorded in the run so the event is visible in the record.
Models that call the tools (all Qwen runs so far) never reach either branch, so their behaviour is unchanged.
"""
from pathlib import Path


def sub(p, old, new):
    s = Path(p).read_text()
    assert s.count(old) == 1, f"{p}: expected text not found exactly once:\n{old[:120]}"
    Path(p).write_text(s.replace(old, new))


A = "grader/agent.py"
if "answered_text" in Path(A).read_text():
    raise SystemExit("already patched; nothing to do")

sub(A, '''                if call is None:
                    return AgentResult("", "no_plan" if version == 0 else "no_final_answer", version)''',
       '''                if call is None:
                    if version == 0:
                        return AgentResult("", "no_plan", version)
                    text = next((m.content.strip() for m in reversed(msgs)
                                 if m.role == "assistant" and (m.content or "").strip()), "")
                    return AgentResult(text, "answered_text" if text else "no_final_answer", version)''')

sub(A, '''            if not r.tool_calls:
                if nudges >= self.max_nudges and r.finish_reason != "length":''',
       '''            if not r.tool_calls:
                written = _text_finish(r.content)
                if written:                                # step result written as JSON text, not via finish_step
                    msgs.append(MessagePart(role="user", origin="system_template", template_id=WORKER_TEMPLATE,
                                            content="harness:text_finish (step result taken from your message)"))
                    return StepResult(step.step_id, written[0], written[1])
                if nudges >= self.max_nudges and r.finish_reason != "length":''')

sub(A, "@dataclass\nclass StepResult:", '''def _text_finish(content: str | None) -> tuple[str, list[str]] | None:
    """A worker message that is a JSON object with a 'summary' (optionally in a ``` fence) -> (summary, ids)."""
    t = (content or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("summary"), str) or not obj["summary"].strip():
        return None
    ids = obj.get("result_call_ids") or []
    return obj["summary"], [str(x) for x in ids] if isinstance(ids, list) else []


@dataclass
class StepResult:''')
print("patched: agent.py (plain-text final answers -> 'answered_text'; JSON-text step results -> finish)")
