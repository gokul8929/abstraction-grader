import os
import shutil
import sys
import time

sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
from pydantic import ValidationError

from grader.record import (Interpretation, LLMCall, MessagePart, PlanCommitted, PlanStep, Review,
                           RevealStep, RunEnd, RunRecorder, RunStart, TokenUsage, ToolCall,
                           append_review, environment, load_events)

RUN = f"/tmp/{os.environ['USER']}/test_record_run"
shutil.rmtree(RUN, ignore_errors=True)
rec = RunRecorder(RUN)

env = environment()
print("environment:", env)
rec.emit(RunStart(dataset="stockindex", query_id="1", question="test?", model="Qwen/Qwen3.8-27B",
                  reasoning_effort="medium", **{k: env[k] for k in
                  ("node", "gpu", "code_commit", "dab_commit", "model_revision", "vllm_version")}))
rec.emit(PlanCommitted(plan_id="plan-1", version=1, author="orchestrator",
                       steps=[PlanStep(step_id="s1", description="get exchanges", assigned_agent="sql-1")]))
rec.emit(Interpretation(interp_id="i1", agent_id="sql-1", step_id="s1", kind="normalisation",
                        subject="index_trade.Date", statement="mixed date formats; parse all"))
t = time.time()
rec.emit(LLMCall(llm_call_id="llm-1", agent_id="sql-1", role="subagent", prompt_author="orchestrator",
                 step_id="s1", messages=[MessagePart(role="system", origin="system_template",
                 template_id="subagent_v1", content="You are...")], model="Qwen/Qwen3.8-27B",
                 temperature=1.0, seed=7, usage=TokenUsage(prompt=1000, cached=800, output=120, reasoning=90),
                 start=t, end=t + 2))
path, sha, size = rec.save_result("call-1", {"rows": [["NYSE", "USD"]]})
rec.emit(ToolCall(call_id="call-1", server="sqlite", tool="run_query", arguments={"sql": "SELECT 1"},
                  agent_id="sql-1", step_id="s1", llm_call_id="llm-1", ok=True, result_file=path,
                  result_sha256=sha, full_result_chars=size, shown_to_model_chars=size, start=t, end=t + 0.1))
rec.emit(RunEnd(answer="NYSE", end_reason="answered", validation={"is_valid": True}))

events = load_events(RUN)
print("events read back:", [(e.seq, e.type) for e in events])
print("new prompt tokens:", events[3].usage.new_prompt)
print("result file exists:", os.path.exists(path), "| sha:", sha[:12])

try:
    Interpretation(interp_id="x", agent_id="a", kind="guess", subject="s", statement="t")
except ValidationError:
    print("bad interpretation kind rejected: True")

r = Review(review_id="rev-1", run_id=rec.run_id, reviewer_id="gokul", grade_shown="D3/O2/L3",
           steps=[RevealStep(level="result", opened_at=t), RevealStep(level="O1-Spec", opened_at=t + 30)],
           started=t, ended=t + 95, verdict="accept", confidence=6, mode="verify",
           tlx={"mental": 40, "effort": 30})
append_review(f"{RUN}/reviews.jsonl", r)
print("review time to verdict:", r.time_to_verdict, "s")
