"""MCP server for the DuckDB databases of one DAB dataset (env: DAB_DATASET_DIR)."""
import duckdb

from servers.common import Column, build_sql_server, file_dbs, file_source_info

DBS = file_dbs("duckdb")


def connect(db_name: str):
    return duckdb.connect(str(DBS[db_name]), read_only=True)


def describe(con) -> dict[str, list[Column]]:
    rows = con.execute("""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position
    """).fetchall()
    schema: dict[str, list[Column]] = {}
    for s, t, c, ty in rows:
        name = t if s == "main" else f"{s}.{t}"
        schema.setdefault(name, []).append(Column(name=c, type=ty))
    return schema


mcp = build_sql_server("duckdb", "DuckDB", DBS, connect, describe,
                       file_source_info(DBS), (duckdb.Error,))

if __name__ == "__main__":
    mcp.run()