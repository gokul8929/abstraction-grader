import asyncio, json, os, shutil, sys
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
from grader.client import ToolHub, engines_for
from grader.record import RunRecorder, load_events

DS = os.path.expanduser("~/DataAgentBench/query_stockindex")
RUN = f"/tmp/{os.environ['USER']}/test_client_run"
shutil.rmtree(RUN, ignore_errors=True)

async def main():
    rec = RunRecorder(RUN)
    print("engines:", engines_for(DS))
    async with ToolHub(rec, DS, show_chars=500) as hub:
        tools = hub.openai_tools()
        print("tools:", [t["function"]["name"] for t in tools])
        print("call_id hidden:", all("call_id" not in t["function"]["parameters"].get("properties", {}) for t in tools))

        o = await hub.call("sqlite__list_databases", {}, agent_id="sql-1", step_id="s1")
        db = json.load(open(o.event.result_file))["databases"][0]
        o = await hub.call("sqlite__list_tables", {"db_name": db}, agent_id="sql-1", step_id="s1")
        table = next(iter(json.load(open(o.event.result_file))["tables"]))
        o = await hub.call("sqlite__run_query", {"db_name": db, "sql": f'SELECT * FROM "{table}"'},
                           agent_id="sql-1", step_id="s1")
        print("query shown to model:", o.event.shown_to_model_chars, "of", o.event.full_result_chars, "chars")
        cid = o.event.call_id

        o = await hub.call("python__run_python",
                           {"code": f"r = load('{cid}')\nprint(type(r).__name__, len(r))"}, agent_id="py-1", step_id="s2")
        print("python:", o.text.replace("\n", " ")[:200], "| loaded:", o.event.loaded_results)
        o = await hub.call("sqlite__run_query", {"db_name": db, "sql": "SELEC 1"}, agent_id="sql-1")
        print("bad sql:", o.text[:100])
        o = await hub.call("postgres__run_query", {}, agent_id="sql-1")
        print("unknown tool:", o.text[:40])

    events = load_events(RUN)
    ids = {e.call_id for e in events}
    logged = {json.loads(l)["call_id"] for f in os.listdir(RUN) if f.startswith("server_")
              for l in open(f"{RUN}/{f}") if l.strip()}
    print("events:", [(e.tool, e.ok) for e in events])
    print("every server log line joins an event:", logged <= ids, f"({len(logged)} logged)")

asyncio.run(main())
