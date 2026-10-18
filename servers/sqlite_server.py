"""MCP server for the SQLite databases of one DAB dataset (env: DAB_DATASET_DIR)."""
import sqlite3

from servers.common import Column, build_sql_server, file_dbs, file_source_info

DBS = file_dbs("sqlite")


def connect(db_name: str):
    return sqlite3.connect(f"file:{DBS[db_name]}?mode=ro", uri=True)  # read-only


def describe(con) -> dict[str, list[Column]]:
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {t: [Column(name=c[1], type=c[2]) for c in con.execute(f'PRAGMA table_info("{t}")')]
            for t in tables}


mcp = build_sql_server("sqlite", "SQLite", DBS, connect, describe,
                       file_source_info(DBS), (sqlite3.Error,))

if __name__ == "__main__":
    mcp.run()