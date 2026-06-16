"""Lightweight SQLite-backed application persistence for packaged Electron use."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def default_storage_root() -> Path:
    if os.getenv("DEPTHLENS_USER_DATA_DIR"):
        return Path(os.environ["DEPTHLENS_USER_DATA_DIR"]).expanduser()
    return Path.home() / ".depthlenspro"


def init_storage(root: Path | None = None) -> dict[str, Any]:
    base = (root or default_storage_root()).expanduser()
    dirs = [
        base,
        base / "artifacts" / "depth-maps",
        base / "artifacts" / "reconstructions",
        base / "artifacts" / "thumbnails",
        base / "cache",
        base / "logs",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    db = base / "app.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS workspace_sessions (id TEXT PRIMARY KEY, name TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS workspace_items (id TEXT PRIMARY KEY, session_id TEXT, source_path_hash TEXT, source_display_name TEXT, result_artifact_id TEXT, metadata_json TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, kind TEXT, path TEXT, sha256 TEXT, size_bytes INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS model_assets (model_id TEXT, engine TEXT, path TEXT, sha256 TEXT, size_bytes INTEGER, validation_status TEXT, installed_at TEXT, PRIMARY KEY(model_id, engine))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS benchmark_runs (id TEXT PRIMARY KEY, model_id TEXT, engine TEXT, provider TEXT, latency_ms REAL, throughput_fps REAL, memory_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS diagnostics_history (id TEXT PRIMARY KEY, kind TEXT, payload_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "available": True,
        "path": os.fspath(base),
        "database": os.fspath(db),
        "schema_version": SCHEMA_VERSION,
    }


def get_setting(key: str, default: Any = None, root: Path | None = None) -> Any:
    info = init_storage(root)
    conn = sqlite3.connect(info["database"])
    try:
        row = conn.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default
    finally:
        conn.close()


def set_setting(key: str, value: Any, root: Path | None = None) -> None:
    info = init_storage(root)
    conn = sqlite3.connect(info["database"])
    try:
        conn.execute(
            "INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=CURRENT_TIMESTAMP",
            (key, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()
