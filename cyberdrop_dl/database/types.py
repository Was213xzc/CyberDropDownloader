from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True, kw_only=True)
class MediaLookupKey:
    domain: str
    db_path: str
    referer: str
    original_filename: str | None = None


@dataclass(slots=True, kw_only=True)
class MediaDefaults:
    download_path: str
    original_filename: str
    album_id: str | None = None
    download_filename: str | None = None
    file_size: int | None = None
    duration: float | None = None
    created_at: datetime | None = None


@dataclass(slots=True, kw_only=True)
class MediaItemRow:
    id: int
    domain: str
    db_path: str
    referer: str
    album_id: str | None
    download_path: str
    download_filename: str | None
    original_filename: str
    file_size: int | None
    duration: float | None
    completed: bool
    created_at: datetime | None
    completed_at: datetime | None


@dataclass(slots=True, kw_only=True)
class HashRow:
    hash_type: str
    hash: str


@dataclass(slots=True, kw_only=True)
class FileQuery:
    folder: str | None = None
    download_filename: str | None = None
    hash_type: str | None = None
    hash_value: str | None = None
    file_size: int | None = None
    media_item_id: int | None = None


@dataclass(slots=True, kw_only=True)
class FileRow:
    folder: str
    download_filename: str
    original_filename: str | None = None
    file_size: int | None = None
    referer: str | None = None
    date: int | None = None
    media_item_id: int | None = None
    hashes: tuple[HashRow, ...] = field(default_factory=tuple)


@dataclass(slots=True, kw_only=True)
class RetryMediaRow:
    referer: str
    download_path: str
    completed_at: datetime | None
    created_at: datetime | None
