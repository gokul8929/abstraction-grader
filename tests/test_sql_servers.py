import asyncio
import json
import os
import sys

USER = os.environ["USER"]
os.environ["DAB_DATASET_DIR"] = os.path.expanduser("~/DataAgentBench/query_stockindex")
os.environ["GRADER_SERVER_LOG"] = f"/tmp/{USER}/test_sql_servers.jsonl"
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))

from mcp import Client
from servers import duckdb_server, sqlite_server


async def check(name, server):
    print(f"\n=== {name} ===")
    async with Client(server) as c:
        tools = await c.list_tools()
        for t in tools.tools:
            print(f"  tool {t.name}: {t.description}")
        dbs = (await c.call_tool("list_databases", {"call_id": f"{name}-1"})).structured_content
        print("  databases:", dbs)
        db = dbs["databases"][0]
        tl = (await c.call_tool("list_tables", {"db_name": db, "call_id": f"{name}-2"})).structured_content
        table = sorted(tl["tables"])[0]
        print("  tables:", json.dumps(tl)[:200])
        r = await c.call_tool("run_query", {"db_name": db, "sql": f'SELECT * FROM {table} LIMIT 2',
                                            "call_id": f"{name}-3"})
        print("  query ok:", not r.is_error, json.dumps(r.structured_content)[:250])
        r = await c.call_tool("run_query", {"db_name": db, "sql": "SELEC broken", "call_id": f"{name}-4"})
        print("  bad SQL ->", r.is_error, "|", r.content[0].text[:70])


async def main():
    await check("sqlite", sqlite_server.mcp)
    await check("duckdb", duckdb_server.mcp)


asyncio.run(main())
print("\nlog lines:", sum(1 for _ in open(f"/tmp/{USER}/test_sql_servers.jsonl")))