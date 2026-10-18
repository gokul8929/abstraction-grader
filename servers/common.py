"""Shared pieces for the grader's SQL MCP servers: config, logging, result models, tools."""
import datetime
import decimal
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel


# ---------- result shapes (give the tools an output schema) ----------
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


# ---------- config ----------
def dataset_dir() -> Path:
    return Path(os.path.expanduser(os.environ["DAB_DATASET_DIR"]))


def load_db_configs(db_type: str) -> dict[str, dict]:
    """Entries of this db_type from the dataset's db_config.yaml, keyed by db name."""
    cfg = yaml.safe_load(open(dataset_dir() / "db_config.yaml"))["db_clients"]
    return {name: c for name, c in cfg.items() if c["db_type"] == db_type}


def file_dbs(db_type: str) -> dict[str, Path]:
    """For file-based engines (sqlite, duckdb): db name -> absolute file path."""
    return {name: (dataset_dir() / c["db_path"]).resolve()
            for name, c in load_db_configs(db_type).items()}


def file_source_info(paths: dict[str, Path]) -> Callable[[str], dict]:
    def info(db_name: str) -> dict:
        p = paths[db_name]
        return {"db_name": db_name, "source": str(p), "source_mtime": os.path.getmtime(p)}
    return info


def max_rows() -> int:
    return int(os.environ.get("GRADER_MAX_ROWS", "200"))


# ---------- server-side record (D bucket) ----------
class CallLog:
    def __init__(self, engine: str):
        self.engine = engine
        default = f"/tmp/{os.environ['USER']}/server_{engine}.jsonl"
        self.path = Path(os.path.expanduser(os.environ.get("GRADER_SERVER_LOG", default)))

    def write(self, entry: dict) -> None:
        entry = {"ts": time.time(), "engine": self.engine, **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return str(v)


def dbapi_run(con, sql: str, limit: int) -> tuple[list[str], list]:
    """Run SQL through a standard DB-API cursor; fetch at most limit+1 rows."""
    cur = con.cursor()
    cur.execute(sql)
    if not cur.description:
        return [], []
    return [d[0] for d in cur.description], cur.fetchmany(limit + 1)


# ---------- the three standard tools for any SQL engine ----------
def build_sql_server(
    engine: str,                                   # e.g. "sqlite", "duckdb"
    label: str,                                    # e.g. "SQLite", shown to the model
    dbs: dict,                                     # db name -> whatever connect() needs
    connect: Callable[[str], Any],                 # db name -> open DB-API connection
    describe: Callable[[Any], dict[str, list[Column]]],
    source_info: Callable[[str], dict],
    db_errors: tuple,                              # the engine's own exception types
    run: Callable = dbapi_run,
) -> MCPServer:
    mcp = MCPServer(engine)
    log = CallLog(engine)
    limit = max_rows()

    def _open(db_name: str):
        if db_name not in dbs:
            raise ToolError(f"Unknown database {db_name!r}. Available: {sorted(dbs)}")
        return connect(db_name)

    def _fail(tool: str, call_id: str, start: float, e: Exception, **extra):
        log.write({"call_id": call_id, "tool": tool, "ok": False, "error": str(e),
                   "duration_s": time.time() - start, **extra})
        if isinstance(e, ToolError):
            raise e
        raise ToolError(f"{label} error: {e}") from e

    def list_databases(call_id: str = "") -> DatabaseList:
        result = DatabaseList(databases=sorted(dbs))
        log.write({"call_id": call_id, "tool": "list_databases", "ok": True, **result.model_dump()})
        return result

    def list_tables(db_name: str, call_id: str = "") -> TableList:
        start = time.time()
        try:
            con = _open(db_name)
            try:
                schema = describe(con)
            finally:
                con.close()
        except (ToolError, *db_errors) as e:
            _fail("list_tables", call_id, start, e, db_name=db_name)
        log.write({"call_id": call_id, "tool": "list_tables", "ok": True, **source_info(db_name),
                   "tables": sorted(schema), "duration_s": time.time() - start})
        return TableList(db_name=db_name, tables=schema)

    def run_query(db_name: str, sql: str, call_id: str = "") -> QueryResult:
        start = time.time()
        try:
            con = _open(db_name)
            try:
                columns, rows = run(con, sql, limit)
            finally:
                con.close()
        except (ToolError, *db_errors) as e:
            _fail("run_query", call_id, start, e, db_name=db_name, sql=sql)
        truncated = len(rows) > limit
        rows = [[jsonable(v) for v in r] for r in rows[:limit]]
        log.write({"call_id": call_id, "tool": "run_query", "ok": True, **source_info(db_name),
                   "sql": sql, "columns": columns, "row_count": len(rows),
                   "truncated": truncated, "duration_s": time.time() - start})
        return QueryResult(columns=columns, rows=rows, truncated=truncated)

    list_databases.__doc__ = f"List the {label} databases available for this task."
    list_tables.__doc__ = f"List tables and their columns in a {label} database."
    run_query.__doc__ = (f"Run a read-only SQL query on a {label} database "
                         f"(use {label} SQL syntax). Returns at most {limit} rows.")
    for fn in (list_databases, list_tables, run_query):
        mcp.tool()(fn)
    return mcp