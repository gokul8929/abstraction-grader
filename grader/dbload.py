"""Load a dataset's Postgres / MongoDB databases with DAB's own loaders, once per database.

A marker /tmp/$USER/dab_loaded/<engine>_<db>.json records the source path and its mtime.
A database that exists without a matching marker (e.g. half-loaded) is dropped and reloaded.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from grader.client import ENGINE_OF

DAB = Path(os.environ.get("DAB_DIR", "~/DataAgentBench")).expanduser()
MARKERS = Path(f"/tmp/{os.environ.get('USER', '')}/dab_loaded")


def _dab_utils():
    if str(DAB) not in sys.path:
        sys.path.insert(0, str(DAB))
    from common_scaffold.tools.db_utils import mongo_utils, postgres_utils
    return postgres_utils, mongo_utils


def _pg_exists(name: str) -> bool:
    import psycopg2
    con = psycopg2.connect(host=os.environ.get("PG_HOST", "127.0.0.1"), port=int(os.environ.get("PG_PORT", "5433")),
                           user=os.environ.get("PG_USER", "postgres"), password=os.environ.get("PG_PASSWORD") or None,
                           dbname="postgres")
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        found = cur.fetchone() is not None
    con.close()
    return found


def _mongo_exists(name: str) -> bool:
    import pymongo
    client = pymongo.MongoClient(os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27018/"), serverSelectionTimeoutMS=5000)
    try:
        return name in client.list_database_names()
    finally:
        client.close()


def ensure_loaded(dataset_dir: str | Path) -> None:
    d = Path(dataset_dir).expanduser()
    cfg = yaml.safe_load(open(d / "db_config.yaml"))["db_clients"]
    MARKERS.mkdir(parents=True, exist_ok=True)
    for c in cfg.values():
        engine = ENGINE_OF[c["db_type"]]
        if engine not in ("postgres", "mongo"):
            continue
        db = c["db_name"]
        src = d / (c["sql_file"] if engine == "postgres" else c["dump_folder"])
        stamp = {"source": str(src), "mtime": src.stat().st_mtime}
        marker = MARKERS / f"{engine}_{db}.json"
        exists = _pg_exists(db) if engine == "postgres" else _mongo_exists(db)
        if exists and marker.exists() and json.loads(marker.read_text()) == stamp:
            continue
        pg, mongo = _dab_utils()
        utils = pg if engine == "postgres" else mongo
        if exists:
            print(f"[dbload] {engine} {db}: no matching marker, dropping and reloading", file=sys.stderr)
            utils.clean_up(db)
        print(f"[dbload] loading {engine} {db} from {src} ...", file=sys.stderr)
        utils.load_db(str(src), db)
        marker.write_text(json.dumps(stamp))
        print(f"[dbload] {engine} {db}: done", file=sys.stderr)
