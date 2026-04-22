from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .types import FileRow, HashRow, MediaDefaults, MediaItemRow, MediaLookupKey

if TYPE_CHECKING:
    from cyberdrop_dl.data_structures.url_objects import MediaItem


def media_lookup_from_media_item(media_item: MediaItem) -> MediaLookupKey:
    return MediaLookupKey(
        domain=media_item.domain,
        db_path=media_item.db_path,
        referer=str(media_item.referer),
        original_filename=media_item.original_filename,
    )


def media_defaults_from_media_item(media_item: MediaItem) -> MediaDefaults:
    return MediaDefaults(
        download_path=str(media_item.download_folder),
        original_filename=media_item.original_filename,
        album_id=media_item.album_id,
        download_filename=media_item.download_filename,
        file_size=media_item.filesize,
        duration=media_item.duration,
    )


def apply_media_row(media_item: MediaItem, row: MediaItemRow) -> None:
    media_item.download_folder = Path(row.download_path)
    if row.download_filename:
        media_item.download_filename = row.download_filename
        media_item.filename = row.download_filename
        media_item.complete_file = media_item.download_folder / row.download_filename
    if row.file_size is not None:
        media_item.filesize = row.file_size
    if row.duration is not None:
        media_item.duration = row.duration
    media_item.db_completed = row.completed


def file_row_from_path(
    file: Path | str,
    *,
    original_filename: str | None = None,
    referer: str | None = None,
    media_item_id: int | None = None,
) -> FileRow:
    path = Path(file)
    if not path.is_absolute():
        path = path.absolute()
    stat = path.stat()
    return FileRow(
        folder=str(path.parent),
        download_filename=path.name,
        original_filename=original_filename,
        file_size=stat.st_size,
        referer=referer,
        date=int(stat.st_mtime),
        media_item_id=media_item_id,
    )


def media_item_hashes(media_item: MediaItem) -> tuple[HashRow, ...]:
    if not media_item.hash:
        return ()
    return (HashRow(hash_type="xxh128", hash=media_item.hash),)
