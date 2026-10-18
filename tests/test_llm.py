import asyncio, os, shutil, sys
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
from grader.llm import LLM
from grader.record import MessagePart, RunRecorder, load_events

RUN = f"/tmp/{os.environ['USER']}/test_llm_run"
shutil.rmtree(RUN, ignore_errors=True)
rec = RunRecorder(RUN)
Q = [MessagePart(role="system", origin="system_template", template_id="test", content="You are careful."),
     MessagePart(role="user", origin="user_question",
                 content="A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball? Reply with just the number.")]
TOOL = [{"type": "function", "function": {"name": "sqlite__run_query", "description": "Run a read-only SQL query.",
         "parameters": {"type": "object", "properties": {"db_name": {"type": "string"}, "sql": {"type": "string"}},
                        "required": ["db_name", "sql"]}}}]

async def effort_test(via):
    llm = LLM(rec, effort_via=via)
    for effort in ["low", "xhigh"]:
        try:
            r = await llm.chat(Q, agent_id="t", role="subagent", prompt_author="test", effort=effort)
            u = r.event.usage
            print(f"  [{via}] {effort:5s} answer={r.content.strip()!r} reasoning_tokens={u.reasoning} "
                  f"output={u.output} time={r.event.end - r.event.start:.1f}s")
        except Exception as e:
            print(f"  [{via}] {effort}: FAILED {type(e).__name__}: {str(e)[:150]}")

async def main():
    print("1. effort levels (reasoning tokens should differ clearly):")
    await effort_test("param")
    await effort_test("template")

    llm = LLM(rec)
    print("2. tool call:")
    msgs = [MessagePart(role="user", origin="user_question",
                        content="Count the rows in table trades of database stocks. Use the tool.")]
    r = await llm.chat(msgs, agent_id="t", role="subagent", prompt_author="test", tools=TOOL, effort="low")
    print("  finish:", r.finish_reason, "| calls:", r.tool_calls)
    if r.tool_calls:
        msgs += [r.as_message(), MessagePart(role="tool", origin="tool_result", tool_call_id=r.tool_calls[0]["id"],
                                             content='{"columns": ["n"], "rows": [[4213]]}')]
        r2 = await llm.chat(msgs, agent_id="t", role="subagent", prompt_author="test", tools=TOOL, effort="low")
        print("  after result:", r2.content.strip()[:100])

    print("3. prefix cache (second call should show cached > 0):")
    long_sys = MessagePart(role="system", origin="system_template", content="Rules:\n" + "Be precise and brief.\n" * 300)
    for i in range(2):
        r = await llm.chat([long_sys, Q[1]], agent_id="t", role="subagent", prompt_author="test", effort="low")
        u = r.event.usage
        print(f"  call {i + 1}: prompt={u.prompt} cached={u.cached} new={u.new_prompt}")

    print("events recorded:", len(load_events(RUN)))

asyncio.run(main())
