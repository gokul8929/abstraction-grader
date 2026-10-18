"""MCP server for the SQLite databases of one DAB dataset.

Env vars:
  DAB_DATASET_DIR    e.g. ~/DataAgentBench/query_stockindex
  GRADER_SERVER_LOG  JSONL file for the server-side (D bucket) record
  GRADER_MAX_ROWS    max rows returned per query (default 200)
"""
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

DATASET_DIR = Path(os.path.expanduser(os.environ["DAB_DATASET_DIR"]))
LOG_PATH = Path(os.path.expanduser(
    os.environ.get("GRADER_SERVER_LOG", f"/tmp/{os.environ['USER']}/server_sqlite.jsonl")))
MAX_ROWS = int(os.environ.get("GRADER_MAX_ROWS", "200"))

_cfg = yaml.safe_load(open(DATASET_DIR / "db_config.yaml"))["db_clients"]
DBS = {name: (DATASET_DIR / c["db_path"]).resolve()
       for name, c in _cfg.items() if c["db_type"] == "sqlite"}

mcp = MCPServer("sqlite")


class Column(BaseModel):
    name: str
    type: str


class DatabaseList(BaseModel):
    databases: list[str]


class TableList(BaseModel):
    db_name: str
    tables: dict[str, list[Column]]


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


def _log(entry: dict) -> None:
    entry = {"ts": time.time(), "engine": "sqlite", **entry}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _connect(db_name: str) -> sqlite3.Connection:
    if db_name not in DBS:
        raise ToolError(f"Unknown database {db_name!r}. Available: {sorted(DBS)}")
    # read-only: DAB tasks never write; write support (with pre-state) comes later
    return sqlite3.connect(f"file:{DBS[db_name]}?mode=ro", uri=True)


def _source_info(db_name: str) -> dict:
    p = DBS[db_name]
    return {"db_name": db_name, "source": str(p), "source_mtime": os.path.getmtime(p)}


@mcp.tool()
def list_databases(call_id: str = "") -> DatabaseList:
    """List the SQLite databases available for this task."""
    result = DatabaseList(databases=sorted(DBS))
    _log({"call_id": call_id, "tool": "list_databases", "ok": True, **result.model_dump()})
    return result


@mcp.tool()
def list_tables(db_name: str, call_id: str = "") -> TableList:
    """List tables and their columns in a SQLite database."""
    start = time.time()
    try:
        con = _connect(db_name)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            schema = {t: [Column(name=c[1], type=c[2])
                          for c in con.execute(f'PRAGMA table_info("{t}")')]
                      for t in tables}
        finally:
            con.close()
    except (ToolError, sqlite3.Error) as e:
        _log({"call_id": call_id, "tool": "list_tables", "db_name": db_name,
              "ok": False, "error": str(e), "duration_s": time.time() - start})
        raise e if isinstance(e, ToolError) else ToolError(f"SQLite error: {e}")
    _log({"call_id": call_id, "tool": "list_tables", "ok": True, **_source_info(db_name),
          "tables": tables, "duration_s": time.time() - start})
    return TableList(db_name=db_name, tables=schema)


@mcp.tool()
def run_query(db_name: str, sql: str, call_id: str = "") -> QueryResult:
    """Run a read-only SQL query on a SQLite database and return the rows."""
    start = time.time()
    try:
        con = _connect(db_name)
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS + 1)
        finally:
            con.close()
    except (ToolError, sqlite3.Error) as e:
        _log({"call_id": call_id, "tool": "run_query", "db_name": db_name, "sql": sql,
              "ok": False, "error": str(e), "duration_s": time.time() - start})
        raise e if isinstance(e, ToolError) else ToolError(f"SQLite error: {e}")
    truncated = len(rows) > MAX_ROWS
    rows = [list(r) for r in rows[:MAX_ROWS]]
    _log({"call_id": call_id, "tool": "run_query", "ok": True, **_source_info(db_name),
          "sql": sql, "columns": columns, "row_count": len(rows),
          "truncated": truncated, "duration_s": time.time() - start})
    return QueryResult(columns=columns, rows=rows, truncated=truncated)


if __name__ == "__main__":
    mcp.run()