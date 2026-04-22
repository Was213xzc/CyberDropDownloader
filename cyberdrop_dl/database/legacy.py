from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

LEGACY_TABLES = frozenset({"media", "files", "hash"})
NEW_TABLES = frozenset({"media_items", "media_files", "media_hashes"})
MIN_SUPPORTED_LEGACY_VERSION = Version("6.6.0")


@dataclass(slots=True, kw_only=True)
class DatabaseState:
    db_exists: bool
    tables: frozenset[str]
    version: Version | None

    @property
    def has_new_schema(self) -> bool:
        return "alembic_version" in self.tables or bool(self.tables & NEW_TABLES)

    @property
    def has_legacy_tables(self) -> bool:
        return bool(self.tables & LEGACY_TABLES)


def inspect_database_state(db_path: Path) -> DatabaseState:
    if not db_path.is_file():
        return DatabaseState(db_exists=False, tables=frozenset(), version=None)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = frozenset(row[0] for row in cursor.fetchall())
        version = _get_legacy_version(conn, tables)
    finally:
        conn.close()

    return DatabaseState(db_exists=True, tables=tables, version=version)


def ensure_supported_legacy_state(db_path: Path, state: DatabaseState) -> None:
    if not state.has_legacy_tables or state.has_new_schema:
        return
    if state.version is None:
        msg = (
            f"Unsupported legacy database at {db_path}: missing schema_version. "
            "Only databases from version 6.6.0 or newer can be migrated automatically."
        )
        raise SystemExit(msg)
    if state.version < MIN_SUPPORTED_LEGACY_VERSION:
        msg = (
            f"Unsupported legacy database at {db_path}: found schema version {state.version}, "
            f"but automatic migration requires {MIN_SUPPORTED_LEGACY_VERSION} or newer."
        )
        raise SystemExit(msg)


def _get_legacy_version(conn: sqlite3.Connection, tables: frozenset[str]) -> Version | None:
    if "schema_version" not in tables:
        return None
    cursor = conn.execute("SELECT version FROM schema_version ORDER BY ROWID DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        return None
    try:
        return Version(row[0])
    except InvalidVersion:
        return None
