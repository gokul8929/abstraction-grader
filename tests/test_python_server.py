import asyncio
import json
import os
import shutil
import sys

USER = os.environ["USER"]
RUN = f"/tmp/{USER}/test_python_run"
shutil.rmtree(RUN, ignore_errors=True)
os.environ["GRADER_RUN_DIR"] = RUN
os.environ["GRADER_EXEC_TIMEOUT"] = "5"
os.environ["GRADER_SERVER_LOG"] = f"/tmp/{USER}/test_python.jsonl"
sys.path.insert(0, os.path.expanduser("~/abstraction-grader"))

from mcp import Client
from servers import python_server

# pretend an earlier SQL call stored its result, as the agent's client will do
os.makedirs(f"{RUN}/results", exist_ok=True)
json.dump({"columns": ["city", "stars"], "rows": [["Tampa", 4.5], ["Tampa", 3.0], ["Reno", 5.0]],
           "truncated": False}, open(f"{RUN}/results/call-1.json", "w"))


async def main():
    async with Client(python_server.mcp) as c:
        code = """
import pandas as pd
r = load("call-1")
df = pd.DataFrame(r["rows"], columns=r["columns"])
print(df.groupby("city")["stars"].mean().to_dict())
print("available:", available())
"""
        r = await c.call_tool("run_python", {"code": code, "call_id": "p-1"})
        print("pandas + load ok:", not r.is_error, "|", r.structured_content["stdout"].strip())

        r = await c.call_tool("run_python", {"code": "1/0", "call_id": "p-2"})
        print("error reported:", r.is_error, "|", r.content[0].text.splitlines()[-1])

        r = await c.call_tool("run_python", {"code": "while True: pass", "call_id": "p-3"})
        print("timeout enforced:", r.is_error, "|", r.content[0].text)

        r = await c.call_tool("run_python", {"code": "import os; print(os.getcwd())", "call_id": "p-4"})
        print("runs in work dir:", r.structured_content["stdout"].strip())


asyncio.run(main())
