"""SQLite persistence for Web UI Config Sets, Pipelines, and Temporal runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import loghawk.config as config


DB_PATH = Path(config.LH_DATA_DIR) / "db" / "loghawk_webui.sqlite3"
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False
_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}


class ActivePipelineRunError(Exception):
    def __init__(self, workflow_id: str):
        super().__init__(f"Pipeline already has an active run: {workflow_id}")
        self.workflow_id = workflow_id


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH, timeout=15) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS config_sets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    external_data_use INTEGER NOT NULL,
                    source_bucket TEXT,
                    source_batch TEXT,
                    train_sources_json TEXT NOT NULL,
                    detect_sources_json TEXT NOT NULL,
                    output_bucket TEXT NOT NULL,
                    output_batch TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pipelines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    config_set_id TEXT NOT NULL REFERENCES config_sets(id),
                    run_mode TEXT NOT NULL CHECK (
                        run_mode IN ('train', 'detect', 'train-detect')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL REFERENCES pipelines(id),
                    workflow_id TEXT NOT NULL UNIQUE,
                    temporal_run_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS pipeline_runs_pipeline_started
                    ON pipeline_runs(pipeline_id, started_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_pipeline
                    ON pipeline_runs(pipeline_id) WHERE status = 'RUNNING';
                """
            )
        _SCHEMA_READY = True


def _connect() -> sqlite3.Connection:
    _ensure_schema()
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _config_set_from_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["external_data_use"] = bool(result["external_data_use"])
    result["train_sources"] = json.loads(result.pop("train_sources_json"))
    result["detect_sources"] = json.loads(result.pop("detect_sources_json"))
    return result


def list_config_sets() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM config_sets ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [_config_set_from_row(row) for row in rows]


def get_config_set(config_set_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM config_sets WHERE id = ?", (config_set_id,)
        ).fetchone()
    return _config_set_from_row(row)


def save_config_set(data: dict, config_set_id: str | None = None) -> dict | None:
    external = bool(data["external_data_use"])
    source_bucket = data.get("source_bucket") if not external else None
    source_batch = data.get("source_batch") if not external else None
    output_bucket = data.get("output_bucket") if external else source_bucket
    output_batch = data.get("output_batch") if external else source_batch
    now = _now()
    train_json = json.dumps(data.get("train_sources", []))
    detect_json = json.dumps(data.get("detect_sources", []))

    with _connect() as connection:
        if config_set_id is None:
            config_set_id = uuid4().hex
            connection.execute(
                """INSERT INTO config_sets
                   (id, name, external_data_use, source_bucket, source_batch,
                    train_sources_json, detect_sources_json, output_bucket,
                    output_batch, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    config_set_id, data["name"], int(external), source_bucket,
                    source_batch, train_json, detect_json, output_bucket,
                    output_batch, now, now,
                ),
            )
        else:
            cursor = connection.execute(
                """UPDATE config_sets SET
                   name = ?, external_data_use = ?, source_bucket = ?,
                   source_batch = ?, train_sources_json = ?,
                   detect_sources_json = ?, output_bucket = ?, output_batch = ?,
                   updated_at = ? WHERE id = ?""",
                (
                    data["name"], int(external), source_bucket, source_batch,
                    train_json, detect_json, output_bucket, output_batch, now,
                    config_set_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
    return get_config_set(config_set_id)


def create_pipeline(data: dict) -> dict:
    pipeline_id = uuid4().hex
    now = _now()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO pipelines
               (id, name, config_set_id, run_mode, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                pipeline_id, data["name"], data["config_set_id"],
                data["run_mode"], now, now,
            ),
        )
    return get_pipeline(pipeline_id)  # type: ignore[return-value]


def update_pipeline(pipeline_id: str, data: dict) -> dict | None:
    with _connect() as connection:
        cursor = connection.execute(
            """UPDATE pipelines SET name = ?, config_set_id = ?,
               run_mode = ?, updated_at = ? WHERE id = ?""",
            (
                data["name"], data["config_set_id"], data["run_mode"],
                _now(), pipeline_id,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return get_pipeline(pipeline_id)


def _pipeline_with_latest_run(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["latest_run"] = None
    return result


def list_pipelines() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """SELECT p.*, c.name AS config_set_name
               FROM pipelines p JOIN config_sets c ON c.id = p.config_set_id
               ORDER BY p.name COLLATE NOCASE"""
        ).fetchall()
    pipelines = []
    for row in rows:
        pipeline = _pipeline_with_latest_run(row)
        pipeline["latest_run"] = get_latest_run(pipeline["id"])
        pipelines.append(pipeline)
    return pipelines


def get_pipeline(pipeline_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT p.*, c.name AS config_set_name
               FROM pipelines p JOIN config_sets c ON c.id = p.config_set_id
               WHERE p.id = ?""",
            (pipeline_id,),
        ).fetchone()
    if row is None:
        return None
    pipeline = _pipeline_with_latest_run(row)
    pipeline["latest_run"] = get_latest_run(pipeline_id)
    return pipeline


def get_latest_run(pipeline_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT * FROM pipeline_runs WHERE pipeline_id = ?
               ORDER BY started_at DESC LIMIT 1""",
            (pipeline_id,),
        ).fetchone()
    return dict(row) if row else None


def list_pipeline_runs(pipeline_id: str) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """SELECT * FROM pipeline_runs WHERE pipeline_id = ?
               ORDER BY started_at DESC""",
            (pipeline_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_active_run(pipeline_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT * FROM pipeline_runs
               WHERE pipeline_id = ? AND status = 'RUNNING' LIMIT 1""",
            (pipeline_id,),
        ).fetchone()
    return dict(row) if row else None


def reserve_run(pipeline_id: str, workflow_id: str) -> dict:
    run = {
        "id": uuid4().hex,
        "pipeline_id": pipeline_id,
        "workflow_id": workflow_id,
        "temporal_run_id": None,
        "status": "RUNNING",
        "started_at": _now(),
        "completed_at": None,
        "error": None,
    }
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        active = connection.execute(
            """SELECT workflow_id FROM pipeline_runs
               WHERE pipeline_id = ? AND status = 'RUNNING' LIMIT 1""",
            (pipeline_id,),
        ).fetchone()
        if active:
            raise ActivePipelineRunError(active["workflow_id"])
        connection.execute(
            """INSERT INTO pipeline_runs
               (id, pipeline_id, workflow_id, temporal_run_id, status,
                started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(run[key] for key in (
                "id", "pipeline_id", "workflow_id", "temporal_run_id",
                "status", "started_at", "completed_at", "error",
            )),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return run


def update_run(
    workflow_id: str,
    *,
    status: str,
    temporal_run_id: str | None = None,
    error: str | None = None,
) -> dict | None:
    status = status.upper()
    completed_at = _now() if status in _TERMINAL_STATUSES else None
    with _connect() as connection:
        connection.execute(
            """UPDATE pipeline_runs SET status = ?,
               temporal_run_id = COALESCE(?, temporal_run_id),
               completed_at = COALESCE(?, completed_at), error = ?
               WHERE workflow_id = ?""",
            (status, temporal_run_id, completed_at, error, workflow_id),
        )
        row = connection.execute(
            "SELECT * FROM pipeline_runs WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
    return dict(row) if row else None


def get_run(workflow_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT r.*, p.name AS pipeline_name, p.run_mode,
                      p.config_set_id, c.name AS config_set_name
               FROM pipeline_runs r
               JOIN pipelines p ON p.id = r.pipeline_id
               JOIN config_sets c ON c.id = p.config_set_id
               WHERE r.workflow_id = ?""",
            (workflow_id,),
        ).fetchone()
    return dict(row) if row else None
