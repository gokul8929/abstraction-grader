import asyncio, os, shutil, sys, time
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
from grader.agent import DataAgent
from grader.client import ToolHub
from grader.llm import LLM
from grader.record import RunRecorder, load_events

DS = os.path.expanduser("~/DataAgentBench/query_stockindex")
RUN = f"/tmp/{os.environ['USER']}/test_agent_run"
shutil.rmtree(RUN, ignore_errors=True)
Q = sys.argv[1] if len(sys.argv) > 1 else "List every table in every database, with its number of rows."

async def main():
    rec = RunRecorder(RUN); t = time.time()
    async with ToolHub(rec, DS) as hub:
        res = await DataAgent(rec, LLM(rec), hub, DS, effort="medium").run(Q)
    print(f"\nANSWER ({res.end_reason}, {res.plans} plan(s), {time.time() - t:.0f}s):\n{res.answer}\n")
    for e in load_events(RUN):
        if e.type == "plan_committed":
            for s in e.steps: print(f"  plan v{e.version} {s.step_id} {s.databases} dep={s.depends_on}: {s.description[:90]}")
        elif e.type == "tool_call":
            print(f"  tool  {e.agent_id:14s} {e.server}.{e.tool} ok={e.ok} {str(e.arguments)[:80]}")
        elif e.type == "interpretation":
            print(f"  interp {e.agent_id:13s} {e.kind}: {e.subject} - {e.statement[:70]}")
        elif e.type == "llm_call":
            u = e.usage
            print(f"  llm   {e.agent_id:14s} {e.end - e.start:5.1f}s new={u.new_prompt} cached={u.cached} out={u.output} think={u.reasoning} {e.finish_reason}")
asyncio.run(main())
