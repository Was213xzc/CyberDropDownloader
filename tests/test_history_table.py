from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from cyberdrop_dl.database import Database, FileQuery, MediaDefaults, MediaLookupKey
from cyberdrop_dl.database.mappers import apply_media_row, media_defaults_from_media_item, media_lookup_from_media_item
from cyberdrop_dl.data_structures import AbsoluteHttpURL
from cyberdrop_dl.data_structures.url_objects import MediaItem


async def _create_database(tmp_path: Path) -> tuple[Database, Path]:
    db_path = tmp_path / "history.db"
    database = Database(db_path, ignore_history=False)
    await database.startup()
    return database, db_path


def _build_media_item(tmp_path: Path) -> MediaItem:
    download_folder = tmp_path / "downloads"
    download_folder.mkdir(parents=True, exist_ok=True)
    return MediaItem(
        url=AbsoluteHttpURL("https://bunkr.site/f/one.mp4"),
        domain="bunkr",
        referer=AbsoluteHttpURL("https://bunkr.site/a/album-1"),
        download_folder=download_folder,
        filename="one.mp4",
        original_filename="one.mp4",
        ext=".mp4",
        db_path="/one.mp4",
        album_id="album-1",
    )


async def test_database_creates_lookup_indexes(tmp_path: Path) -> None:
    database, db_path = await _create_database(tmp_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        index_names = {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()
        await database.close()

    assert "ix_media_items_referer_domain_completed" in index_names
    assert "ix_media_items_domain_album_completed" in index_names
    assert "ix_media_items_download_filename" in index_names
    assert "ix_media_items_domain_filename_size_completed" in index_names
    assert "ix_media_hashes_hash_type_hash" in index_names


async def test_get_media_item_creates_missing_row_and_returns_existing_row_unchanged(tmp_path: Path) -> None:
    database, _ = await _create_database(tmp_path)
    key = MediaLookupKey(
        domain="bunkr",
        db_path="/one.mp4",
        referer="https://bunkr.site/a/album-1",
        original_filename="one.mp4",
    )
    defaults = MediaDefaults(download_path=str(tmp_path / "downloads"), original_filename="one.mp4", album_id="album-1")

    created = await database.get_media_item(key, defaults)
    existing = await database.get_media_item(
        key,
        MediaDefaults(download_path="ignored", original_filename="different.mp4", album_id="other"),
    )

    try:
        assert created.id == existing.id
        assert existing.download_path == str(tmp_path / "downloads")
        assert existing.original_filename == "one.mp4"
        assert existing.album_id == "album-1"
        assert existing.completed is False
    finally:
        await database.close()


async def test_concurrent_get_media_item_creates_single_row(tmp_path: Path) -> None:
    database, _ = await _create_database(tmp_path)
    key = MediaLookupKey(
        domain="bunkr",
        db_path="/concurrent.mp4",
        referer="https://bunkr.site/a/album-1",
        original_filename="concurrent.mp4",
    )
    defaults = MediaDefaults(
        download_path=str(tmp_path / "downloads"),
        original_filename="concurrent.mp4",
        album_id="album-1",
    )

    try:
        rows = await asyncio.gather(*(database.get_media_item(key, defaults) for _ in range(40)))

        assert len({row.id for row in rows}) == 1
        assert rows[0].original_filename == "concurrent.mp4"
    finally:
        await database.close()


async def test_update_media_item_persists_media_files_and_hashes(tmp_path: Path) -> None:
    database, _ = await _create_database(tmp_path)
    media_item = _build_media_item(tmp_path)

    row = await database.get_media_item(
        media_lookup_from_media_item(media_item),
        media_defaults_from_media_item(media_item),
    )
    apply_media_row(media_item, row)

    media_item.download_filename = "saved-one.mp4"
    media_item.filename = media_item.download_filename
    media_item.complete_file = media_item.download_folder / media_item.download_filename
    media_item.complete_file.write_bytes(b"downloaded")
    media_item.filesize = media_item.complete_file.stat().st_size
    media_item.duration = 12.5
    media_item.hash = "abc123"
    media_item.db_completed = True

    await database.update_media_item(media_item)
    stored = await database.get_media_item(
        media_lookup_from_media_item(media_item),
        media_defaults_from_media_item(media_item),
    )
    files = await database.get_files(
        FileQuery(folder=str(media_item.download_folder), download_filename=media_item.download_filename)
    )

    try:
        assert stored.download_filename == "saved-one.mp4"
        assert stored.file_size == media_item.filesize
        assert stored.duration == 12.5
        assert stored.completed is True
        assert await database.check_complete_by_referer("bunkr", media_item.referer) is True
        assert await database.check_complete_by_filename_size("bunkr", "saved-one.mp4", media_item.filesize) is True
        assert await database.check_download_filename_exists("saved-one.mp4") is True
        assert len(files) == 1
        assert files[0].file_size == media_item.filesize
        assert ("xxh128", "abc123") in {(hash_row.hash_type, hash_row.hash) for hash_row in files[0].hashes}
    finally:
        await database.close()


async def test_apply_media_row_restores_saved_download_folder(tmp_path: Path) -> None:
    database, _ = await _create_database(tmp_path)
    media_item = _build_media_item(tmp_path)

    row = await database.get_media_item(
        media_lookup_from_media_item(media_item),
        media_defaults_from_media_item(media_item),
    )
    moved_folder = tmp_path / "moved-downloads"
    moved_folder.mkdir(parents=True, exist_ok=True)
    media_item.download_folder = moved_folder
    media_item.complete_file = None

    try:
        apply_media_row(media_item, row)

        assert media_item.download_folder == Path(row.download_path)
        assert media_item.complete_file is None
    finally:
        await database.close()
