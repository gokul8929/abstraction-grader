"""MCP server for the PostgreSQL databases of one DAB dataset.

The databases must already be loaded (DAB's postgres_utils.load_db) — the task runner
does that before a run. Connection: PG_HOST, PG_PORT, PG_USER, PG_PASSWORD env vars.
"""
import os

import psycopg2

from servers.common import Column, build_sql_server, dataset_dir, load_db_configs

CFG = load_db_configs("postgres")
DBS = {name: c["db_name"] for name, c in CFG.items()}   # config name -> postgres database name


def connect(db_name: str):
    con = psycopg2.connect(
        dbname=DBS[db_name],
        host=os.environ.get("PG_HOST", "127.0.0.1"),
        port=int(os.environ.get("PG_PORT", "5433")),
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASSWORD", ""),
    )
    con.set_session(readonly=True, autocommit=True)
    return con


def describe(con) -> dict[str, list[Column]]:
    with con.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position
        """)
        rows = cur.fetchall()
    schema: dict[str, list[Column]] = {}
    for s, t, c, ty in rows:
        name = t if s == "public" else f"{s}.{t}"
        schema.setdefault(name, []).append(Column(name=c, type=ty))
    return schema


def source_info(db_name: str) -> dict:
    dump = (dataset_dir() / CFG[db_name]["sql_file"]).resolve()
    return {"db_name": db_name, "pg_database": DBS[db_name],
            "source": str(dump), "source_mtime": os.path.getmtime(dump)}


mcp = build_sql_server("postgres", "PostgreSQL", DBS, connect, describe,
                       source_info, (psycopg2.Error,))

if __name__ == "__main__":
    mcp.run()
    