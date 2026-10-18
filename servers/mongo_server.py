"""MCP server for the MongoDB databases of one DAB dataset.

The databases must already be restored (DAB's mongo_utils.load_db) — the task runner
does that before a run. Connection: MONGO_URI env var.
Read-only by construction: only find / aggregate / listing are exposed, and
aggregate stages that write ($out, $merge) are rejected.
"""
import json
import os
import time
from typing import Any

import pymongo
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from servers.common import CallLog, DatabaseList, dataset_dir, jsonable, load_db_configs, max_rows

CFG = load_db_configs("mongo")
DBS = {name: c["db_name"] for name, c in CFG.items()}   # config name -> mongo database name
URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27018/")
WRITE_STAGES = {"$out", "$merge"}

mcp = MCPServer("mongo")
log = CallLog("mongo")
LIMIT = max_rows()


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    sample_fields: list[str]


class CollectionList(BaseModel):
    db_name: str
    collections: list[CollectionInfo]


class DocumentResult(BaseModel):
    documents: list[dict[str, Any]]
    truncated: bool


def _source_info(db_name: str) -> dict:
    dump = (dataset_dir() / CFG[db_name]["dump_folder"]).resolve()
    return {"db_name": db_name, "mongo_database": DBS[db_name],
            "source": str(dump), "source_mtime": os.path.getmtime(dump)}


def _parse(value, default, what: str):
    """Accept JSON objects or JSON text from the model."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ToolError(f"{what} is not valid JSON: {e}")
    return value


def _db(db_name: str):
    if db_name not in DBS:
        raise ToolError(f"Unknown database {db_name!r}. Available: {sorted(DBS)}")
    client = pymongo.MongoClient(URI, serverSelectionTimeoutMS=5000)
    return client, client[DBS[db_name]]


def _fail(tool: str, call_id: str, start: float, e: Exception, **extra):
    log.write({"call_id": call_id, "tool": tool, "ok": False, "error": str(e),
               "duration_s": time.time() - start, **extra})
    if isinstance(e, ToolError):
        raise e
    raise ToolError(f"MongoDB error: {e}") from e


@mcp.tool()
def list_databases(call_id: str = "") -> DatabaseList:
    """List the MongoDB databases available for this task."""
    result = DatabaseList(databases=sorted(DBS))
    log.write({"call_id": call_id, "tool": "list_databases", "ok": True, **result.model_dump()})
    return result


@mcp.tool()
def list_collections(db_name: str, call_id: str = "") -> CollectionList:
    """List collections in a MongoDB database, with document counts and the field names
    of one sample document (MongoDB has no fixed schema)."""
    start = time.time()
    try:
        client, db = _db(db_name)
        try:
            infos = []
            for name in sorted(db.list_collection_names()):
                sample = db[name].find_one() or {}
                infos.append(CollectionInfo(name=name,
                                            document_count=db[name].estimated_document_count(),
                                            sample_fields=list(sample.keys())))
        finally:
            client.close()
    except (ToolError, pymongo.errors.PyMongoError) as e:
        _fail("list_collections", call_id, start, e, db_name=db_name)
    log.write({"call_id": call_id, "tool": "list_collections", "ok": True, **_source_info(db_name),
               "collections": [i.name for i in infos], "duration_s": time.time() - start})
    return CollectionList(db_name=db_name, collections=infos)


@mcp.tool()
def find(db_name: str, collection: str, filter: dict[str, Any] | str | None = None,
         projection: dict[str, Any] | str | None = None, limit: int = 50,
         call_id: str = "") -> DocumentResult:
    """Find documents in a MongoDB collection. filter and projection use MongoDB query
    syntax, e.g. filter={"city": "Tampa"}, projection={"name": 1, "_id": 0}."""
    start = time.time()
    n = max(1, min(limit, LIMIT))
    try:
        flt = _parse(filter, {}, "filter")
        proj = _parse(projection, None, "projection")
        client, db = _db(db_name)
        try:
            docs = list(db[collection].find(flt, proj).limit(n + 1))
        finally:
            client.close()
    except (ToolError, pymongo.errors.PyMongoError) as e:
        _fail("find", call_id, start, e, db_name=db_name, collection=collection,
              filter=filter, projection=projection)
    truncated = len(docs) > n
    docs = [jsonable(d) for d in docs[:n]]
    log.write({"call_id": call_id, "tool": "find", "ok": True, **_source_info(db_name),
               "collection": collection, "filter": flt, "projection": proj,
               "row_count": len(docs), "truncated": truncated, "duration_s": time.time() - start})
    return DocumentResult(documents=docs, truncated=truncated)


@mcp.tool()
def aggregate(db_name: str, collection: str, pipeline: list[dict[str, Any]] | str,
              call_id: str = "") -> DocumentResult:
    """Run a read-only MongoDB aggregation pipeline (a list of stages such as $match,
    $group, $sort, $project). Stages that write ($out, $merge) are not allowed."""
    start = time.time()
    try:
        stages = _parse(pipeline, [], "pipeline")
        if not isinstance(stages, list):
            raise ToolError("pipeline must be a list of stages")
        for st in stages:
            if WRITE_STAGES & set(st):
                raise ToolError(f"Write stages are not allowed: {sorted(WRITE_STAGES & set(st))}")
        client, db = _db(db_name)
        try:
            docs = list(db[collection].aggregate(stages + [{"$limit": LIMIT + 1}]))
        finally:
            client.close()
    except (ToolError, pymongo.errors.PyMongoError) as e:
        _fail("aggregate", call_id, start, e, db_name=db_name, collection=collection,
              pipeline=pipeline)
    truncated = len(docs) > LIMIT
    docs = [jsonable(d) for d in docs[:LIMIT]]
    log.write({"call_id": call_id, "tool": "aggregate", "ok": True, **_source_info(db_name),
               "collection": collection, "pipeline": stages, "row_count": len(docs),
               "truncated": truncated, "duration_s": time.time() - start})
    return DocumentResult(documents=docs, truncated=truncated)


if __name__ == "__main__":
    mcp.run()
