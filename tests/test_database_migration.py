from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from cyberdrop_dl.database import Database, FileQuery, MediaDefaults, MediaLookupKey


def _create_legacy_database(path: Path, *, version: str | None = "8.10.0") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE media (
              domain TEXT,
              url_path TEXT,
              referer TEXT,
              album_id TEXT,
              download_path TEXT,
              download_filename TEXT,
              original_filename TEXT,
              file_size INT,
              duration FLOAT,
              completed INTEGER NOT NULL,
              created_at TIMESTAMP,
              completed_at TIMESTAMP,
              PRIMARY KEY (domain, url_path, original_filename)
            );
            CREATE TABLE files (
              folder TEXT,
              download_filename TEXT,
              original_filename TEXT,
              file_size INT,
              referer TEXT,
              date INT,
              PRIMARY KEY (folder, download_filename)
            );
            CREATE TABLE "hash" (
              folder TEXT,
              download_filename TEXT,
              hash_type TEXT,
              hash TEXT,
              PRIMARY KEY (folder, download_filename, hash_type)
            );
            """
        )
        if version is not None:
            conn.executescript(
                """
                CREATE TABLE schema_version (
                    version VARCHAR(50) NOT NULL PRIMARY KEY,
                    applied_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

        created_at = datetime(2026, 1, 2, 3, 4, 5).isoformat()
        completed_at = datetime(2026, 1, 2, 4, 5, 6).isoformat()
        conn.execute(
            """
            INSERT INTO media (
                domain, url_path, referer, album_id, download_path, download_filename,
                original_filename, file_size, duration, completed, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bunkr",
                "/one.mp4",
                "https://bunkr.site/a/album-1",
                "album-1",
                str(path.parent / "downloads"),
                "saved-one.mp4",
                "one.mp4",
                321,
                12.5,
                1,
                created_at,
                completed_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO files (folder, download_filename, original_filename, file_size, referer, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(path.parent / "downloads"),
                "saved-one.mp4",
                "one.mp4",
                321,
                "https://bunkr.site/a/album-1",
                1_700_000_000,
            ),
        )
        conn.execute(
            """
            INSERT INTO "hash" (folder, download_filename, hash_type, hash)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(path.parent / "downloads"),
                "saved-one.mp4",
                "xxh128",
                "legacyhash",
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def test_database_imports_supported_legacy_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    _create_legacy_database(db_path, version="8.10.0")
    database = Database(db_path, ignore_history=False)
    await database.startup()

    try:
        stored = await database.get_media_item(
            MediaLookupKey(
                domain="bunkr",
                db_path="/one.mp4",
                referer="https://bunkr.site/a/album-1",
                original_filename="one.mp4",
            ),
            MediaDefaults(download_path=str(tmp_path / "downloads"), original_filename="one.mp4"),
        )
        files = await database.get_files(
            FileQuery(folder=str(tmp_path / "downloads"), download_filename="saved-one.mp4")
        )
        assert stored.completed is True
        assert stored.file_size == 321
        assert stored.duration == 12.5
        assert len(files) == 1
        assert ("xxh128", "legacyhash") in {(hash_row.hash_type, hash_row.hash) for hash_row in files[0].hashes}
    finally:
        await database.close()


async def test_database_rejects_legacy_schema_older_than_660(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    _create_legacy_database(db_path, version="6.5.0")

    with pytest.raises(SystemExit, match="requires 6.6.0 or newer"):
        await Database(db_path, ignore_history=False).startup()


async def test_database_rejects_unversioned_legacy_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    _create_legacy_database(db_path, version=None)

    with pytest.raises(SystemExit, match="missing schema_version"):
        await Database(db_path, ignore_history=False).startup()
