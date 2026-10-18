import asyncio
import json
import os
import sys

USER = os.environ["USER"]
DAB = os.path.expanduser("~/DataAgentBench")
os.environ["DAB_DATASET_DIR"] = f"{DAB}/query_bookreview"
os.environ["GRADER_SERVER_LOG"] = f"/tmp/{USER}/test_postgres.jsonl"
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
sys.path.insert(0, DAB)

from common_scaffold.tools.db_utils import postgres_utils   # DAB's own loader
from mcp import Client
from servers import postgres_server


async def main():
    async with Client(postgres_server.mcp) as c:
        dbs = (await c.call_tool("list_databases", {"call_id": "pg-1"})).structured_content
        print("databases:", dbs)
        db = dbs["databases"][0]
        tl = (await c.call_tool("list_tables", {"db_name": db, "call_id": "pg-2"})).structured_content
        print("tables:", json.dumps(tl)[:250])
        table = sorted(tl["tables"])[0]
        r = await c.call_tool("run_query", {"db_name": db, "sql": f"SELECT * FROM {table} LIMIT 2",
                                            "call_id": "pg-3"})
        print("query ok:", not r.is_error, json.dumps(r.structured_content)[:250])
        r = await c.call_tool("run_query", {"db_name": db, "sql": f"DELETE FROM {table}",
                                            "call_id": "pg-4"})
        print("write blocked:", r.is_error, "|", r.content[0].text[:90])


cfg = postgres_server.CFG
for name, c in cfg.items():
    print(f"loading {c['db_name']} from {c['sql_file']} ...")
    postgres_utils.load_db(f"{DAB}/query_bookreview/{c['sql_file']}", c["db_name"])
try:
    asyncio.run(main())
finally:
    for c in cfg.values():
        postgres_utils.clean_up(c["db_name"])
    print("cleaned up")