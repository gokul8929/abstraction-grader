import asyncio
import json
import os
import sys

USER = os.environ["USER"]
DAB = os.path.expanduser("~/DataAgentBench")
os.environ["DAB_DATASET_DIR"] = f"{DAB}/query_yelp"
os.environ["GRADER_SERVER_LOG"] = f"/tmp/{USER}/test_mongo.jsonl"
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))
sys.path.insert(0, DAB)

from common_scaffold.tools.db_utils import mongo_utils   # DAB's own loader (also loads .env)
from mcp import Client
from servers import mongo_server


async def main():
    async with Client(mongo_server.mcp) as c:
        print("tools:", [t.name for t in (await c.list_tools()).tools])
        dbs = (await c.call_tool("list_databases", {"call_id": "m-1"})).structured_content
        db = dbs["databases"][0]
        cl = (await c.call_tool("list_collections", {"db_name": db, "call_id": "m-2"})).structured_content
        print("collections:", json.dumps(cl)[:300])
        coll = cl["collections"][0]["name"]

        r = await c.call_tool("find", {"db_name": db, "collection": coll, "limit": 2, "call_id": "m-3"})
        print("find ok:", not r.is_error, json.dumps(r.structured_content)[:250])

        r = await c.call_tool("find", {"db_name": db, "collection": coll, "limit": 1,
                                       "filter": "{}", "projection": '{"_id": 0}', "call_id": "m-4"})
        print("find with JSON-text args ok:", not r.is_error)

        r = await c.call_tool("aggregate", {"db_name": db, "collection": coll,
                                            "pipeline": [{"$count": "n"}], "call_id": "m-5"})
        print("aggregate:", r.structured_content)

        r = await c.call_tool("aggregate", {"db_name": db, "collection": coll,
                                            "pipeline": [{"$out": "hacked"}], "call_id": "m-6"})
        print("write blocked:", r.is_error, "|", r.content[0].text[:80])


cfg = mongo_server.CFG
for c in cfg.values():
    print(f"restoring {c['db_name']} from {c['dump_folder']} ...")
    mongo_utils.load_db(f"{DAB}/query_yelp/{c['dump_folder']}", c["db_name"])
try:
    asyncio.run(main())
finally:
    for c in cfg.values():
        mongo_utils.clean_up(c["db_name"])
    print("cleaned up")
