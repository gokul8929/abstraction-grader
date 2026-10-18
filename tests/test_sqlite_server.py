import asyncio
import json
import os
import sys

USER = os.environ["USER"]
os.environ["DAB_DATASET_DIR"] = os.path.expanduser("~/DataAgentBench/query_stockindex")
os.environ["GRADER_SERVER_LOG"] = f"/tmp/{USER}/test_server_sqlite.jsonl"
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))

from mcp import Client
from servers.sqlite_server import mcp


async def main():
    async with Client(mcp) as c:
        tools = await c.list_tools()
        print("tools:", [t.name for t in tools.tools])

        r = await c.call_tool("list_databases", {"call_id": "t1"})
        print("databases:", r.structured_content)

        r = await c.call_tool("list_tables", {"db_name": "indexinfo_database", "call_id": "t2"})
        print("tables:", json.dumps(r.structured_content)[:300])

        r = await c.call_tool("run_query", {"db_name": "indexinfo_database",
                                            "sql": "SELECT * FROM index_info LIMIT 3",
                                            "call_id": "t3"})
        print("query error?", r.is_error)
        print("rows:", json.dumps(r.structured_content)[:300])

        r = await c.call_tool("run_query", {"db_name": "nope", "sql": "SELECT 1", "call_id": "t4"})
        print("bad db gives error?", r.is_error)
        
        r = await c.call_tool("run_query", {"db_name": "indexinfo_database",
                                            "sql": "SELEC broken", "call_id": "t5"})
        print("bad SQL gives error?", r.is_error, "|", r.content[0].text[:80])

asyncio.run(main())
print("\nserver log:")
os.system(f"tail -4 /tmp/{USER}/test_server_sqlite.jsonl")
