from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from cyberdrop_dl.utils.logger import log

from .legacy import DatabaseState, ensure_supported_legacy_state, inspect_database_state
from .mappers import file_row_from_path
from .orm import MediaFileRecord, MediaHashRecord, MediaItemRecord
from .types import FileQuery, FileRow, HashRow, MediaDefaults, MediaItemRow, MediaLookupKey, RetryMediaRow

if TYPE_CHECKING:
    from yarl import URL

    from cyberdrop_dl.crawlers import Crawler
    from cyberdrop_dl.data_structures.url_objects import MediaItem


_FETCH_MANY_SIZE = 1000
_BUNKR_FAILURE_FILE_SIZE = 322509
_BUNKR_FAILURE_HASH = "eb669b6362e031fa2b0f1215480c4e30"


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Database:
    def __init__(self, db_path: Path, ignore_history: bool) -> None:
        self._db_path = db_path
        self.ignore_history = ignore_history
        self._engine: AsyncEngine
        self._sessionmaker: async_sessionmaker[AsyncSession]

    async def startup(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db_state = inspect_database_state(self._db_path)
        ensure_supported_legacy_state(self._db_path, db_state)

        await asyncio.to_thread(self._run_migrations)

        self._engine = create_async_engine(self._async_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

        if db_state.has_legacy_tables and await self._new_schema_is_empty():
            await self._import_legacy_database(db_state)

    async def close(self) -> None:
        await self._engine.dispose()

    async def check_complete(self, domain: str, url: URL | str, referer: URL | str, db_path: str) -> bool:
        if self.ignore_history:
            return False

        async with self._sessionmaker() as session:
            stmt = (
                select(MediaItemRecord)
                .where(MediaItemRecord.domain == domain, MediaItemRecord.url_path == db_path)
                .order_by(MediaItemRecord.completed.desc(), MediaItemRecord.id.asc())
            )
            record = (await session.scalars(stmt)).first()
            if record is None:
                return False

            completed = bool(record.completed)
            referer_str = str(referer)
            if completed and record.referer != referer_str:
                record.referer = referer_str
                await session.commit()
            return completed

    async def check_complete_by_referer(self, domain: str | None, referer: URL | str) -> bool:
        if self.ignore_history:
            return False

        stmt = select(MediaItemRecord.id).where(
            MediaItemRecord.referer == str(referer),
            MediaItemRecord.completed.is_(True),
        )
        if domain is not None:
            stmt = stmt.where(MediaItemRecord.domain == domain)

        async with self._sessionmaker() as session:
            return (await session.scalar(stmt.limit(1))) is not None

    async def check_album(self, domain: str, album_id: str) -> dict[str, int]:
        if self.ignore_history:
            return {}

        async with self._sessionmaker() as session:
            stmt = select(MediaItemRecord.url_path, MediaItemRecord.completed).where(
                MediaItemRecord.domain == domain,
                MediaItemRecord.album_id == album_id,
            )
            rows = await session.execute(stmt)
            return {url_path: int(completed) for url_path, completed in rows.all()}

    async def get_media_item(self, key: MediaLookupKey, defaults: MediaDefaults) -> MediaItemRow:
        async with self._sessionmaker() as session:
            record = await self._find_media_item_record(session, key)
            if record is None:
                created_at = defaults.created_at or _utcnow()
                record = MediaItemRecord(
                    domain=key.domain,
                    url_path=key.db_path,
                    referer=key.referer,
                    album_id=defaults.album_id,
                    download_path=defaults.download_path,
                    download_filename=defaults.download_filename,
                    original_filename=key.original_filename or defaults.original_filename,
                    file_size=defaults.file_size,
                    duration=defaults.duration,
                    completed=False,
                    created_at=created_at,
                )
                session.add(record)
                await session.commit()
                await session.refresh(record)
            return self._map_media_item(record)

    async def update_media_item(self, media_item: MediaItem) -> None:
        if media_item.is_segment:
            return

        key = MediaLookupKey(
            domain=media_item.domain,
            db_path=media_item.db_path,
            referer=str(media_item.referer),
            original_filename=media_item.original_filename,
        )
        defaults = MediaDefaults(
            download_path=str(media_item.download_folder),
            original_filename=media_item.original_filename,
            album_id=media_item.album_id,
            download_filename=media_item.download_filename,
            file_size=media_item.filesize,
            duration=media_item.duration,
        )
        file_row, hashes = await self._build_media_file_update(media_item)

        async with self._sessionmaker() as session:
            record = await self._find_media_item_record(session, key)
            now = _utcnow()
            if record is None:
                record = MediaItemRecord(
                    domain=key.domain,
                    url_path=key.db_path,
                    referer=key.referer,
                    album_id=defaults.album_id,
                    download_path=defaults.download_path,
                    download_filename=defaults.download_filename,
                    original_filename=key.original_filename or defaults.original_filename,
                    file_size=defaults.file_size,
                    duration=defaults.duration,
                    completed=False,
                    created_at=now,
                )
                session.add(record)
                await session.flush()

            completed = bool(record.completed)
            completed_at = record.completed_at
            if media_item.db_completed is not None:
                completed = media_item.db_completed
                if completed:
                    completed_at = completed_at or now
                else:
                    completed_at = None

            record.referer = key.referer
            record.album_id = media_item.album_id
            record.download_path = str(media_item.download_folder)
            record.download_filename = media_item.download_filename or record.download_filename
            record.original_filename = media_item.original_filename
            record.file_size = (
                file_row.file_size
                if file_row and file_row.file_size is not None
                else media_item.filesize
                if media_item.filesize is not None
                else record.file_size
            )
            record.duration = media_item.duration if media_item.duration is not None else record.duration
            record.completed = completed
            record.completed_at = completed_at
            record.created_at = record.created_at or now

            await session.flush()
            if file_row is not None:
                file_row.media_item_id = record.id
                await self._upsert_file(session, file_row, list(hashes))
            await session.commit()

    async def get_all_media_items(self, after: date, before: date) -> AsyncGenerator[list[RetryMediaRow]]:
        async for rows in self._yield_retry_rows(
            select(MediaItemRecord)
            .where(
                func.date(func.coalesce(MediaItemRecord.completed_at, datetime(1970, 1, 1))) >= after.isoformat(),
                func.date(func.coalesce(MediaItemRecord.completed_at, datetime(1970, 1, 1))) <= before.isoformat(),
            )
            .order_by(MediaItemRecord.completed_at.desc(), MediaItemRecord.id.asc())
        ):
            yield rows

    async def get_failed_media_items(self) -> AsyncGenerator[list[RetryMediaRow]]:
        async for rows in self._yield_retry_rows(
            select(MediaItemRecord)
            .where(MediaItemRecord.completed.is_(False))
            .order_by(MediaItemRecord.id.asc())
        ):
            yield rows

    async def get_files(self, query: FileQuery) -> list[FileRow]:
        async with self._sessionmaker() as session:
            stmt = select(MediaFileRecord).options(selectinload(MediaFileRecord.hashes))
            if query.hash_type is not None or query.hash_value is not None:
                stmt = stmt.join(MediaFileRecord.hashes)
            if query.folder is not None:
                stmt = stmt.where(MediaFileRecord.folder == query.folder)
            if query.download_filename is not None:
                stmt = stmt.where(MediaFileRecord.download_filename == query.download_filename)
            if query.hash_type is not None:
                stmt = stmt.where(MediaHashRecord.hash_type == query.hash_type)
            if query.hash_value is not None:
                stmt = stmt.where(MediaHashRecord.hash == query.hash_value)
            if query.file_size is not None:
                stmt = stmt.where(MediaFileRecord.file_size == query.file_size)
            if query.media_item_id is not None:
                stmt = stmt.where(MediaFileRecord.media_item_id == query.media_item_id)
            stmt = stmt.order_by(MediaFileRecord.date.asc(), MediaFileRecord.id.asc())

            records = (await session.scalars(stmt)).unique().all()
            return [self._map_file(record) for record in records]

    async def update_files(self, file: FileRow, hashes: list[HashRow] | None = None) -> None:
        async with self._sessionmaker() as session:
            await self._upsert_file(session, file, hashes or [])
            await session.commit()

    async def update_previously_unsupported(self, crawlers: dict[str, Crawler]) -> None:
        domains_to_update = {
            crawler.DOMAIN: f"http%{crawler.PRIMARY_URL.host}%"
            for crawler in crawlers.values()
            if crawler.UPDATE_UNSUPPORTED
        }
        if not domains_to_update:
            return

        async with self._sessionmaker() as session:
            for domain, like_pattern in domains_to_update.items():
                stmt = select(MediaItemRecord).where(
                    MediaItemRecord.domain == "no_crawler",
                    MediaItemRecord.referer.like(like_pattern),
                )
                for record in (await session.scalars(stmt)).all():
                    duplicate_stmt = select(MediaItemRecord.id).where(
                        MediaItemRecord.domain == domain,
                        MediaItemRecord.url_path == record.url_path,
                        MediaItemRecord.original_filename == record.original_filename,
                        MediaItemRecord.id != record.id,
                    )
                    duplicate = await session.scalar(duplicate_stmt.limit(1))
                    if duplicate is None:
                        record.domain = domain
                    else:
                        await session.delete(record)
            await session.commit()

    async def check_complete_by_filename_size(self, domain: str, filename: str | None, file_size: int | None) -> bool:
        if self.ignore_history or not filename or file_size is None:
            return False

        async with self._sessionmaker() as session:
            stmt = select(MediaItemRecord.id).where(
                MediaItemRecord.domain == domain,
                MediaItemRecord.download_filename == filename,
                MediaItemRecord.file_size == file_size,
                MediaItemRecord.completed.is_(True),
            )
            return (await session.scalar(stmt.limit(1))) is not None

    async def check_download_filename_exists(self, filename: str) -> bool:
        async with self._sessionmaker() as session:
            stmt = select(MediaItemRecord.id).where(MediaItemRecord.download_filename == filename)
            return (await session.scalar(stmt.limit(1))) is not None

    async def get_all_bunkr_failed(self) -> AsyncGenerator[list[RetryMediaRow]]:
        async for rows in self._yield_retry_rows(
            select(MediaItemRecord)
            .where(MediaItemRecord.file_size == _BUNKR_FAILURE_FILE_SIZE)
            .order_by(MediaItemRecord.id.asc())
        ):
            yield rows

        async for rows in self._yield_retry_rows(
            select(MediaItemRecord)
            .join(MediaItemRecord.files)
            .join(MediaFileRecord.hashes)
            .where(MediaHashRecord.hash == _BUNKR_FAILURE_HASH)
            .order_by(MediaItemRecord.id.asc())
        ):
            yield rows

    @property
    def _async_url(self) -> str:
        return f"sqlite+aiosqlite:///{self._db_path.resolve().as_posix()}"

    @property
    def _sync_url(self) -> str:
        return f"sqlite:///{self._db_path.resolve().as_posix()}"

    def _run_migrations(self) -> None:
        from alembic import command
        from alembic.config import Config

        config = Config()
        config.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
        config.set_main_option("sqlalchemy.url", self._sync_url)
        command.upgrade(config, "head")

    async def _find_media_item_record(self, session: AsyncSession, key: MediaLookupKey) -> MediaItemRecord | None:
        stmt = select(MediaItemRecord).where(
            MediaItemRecord.domain == key.domain,
            MediaItemRecord.url_path == key.db_path,
        )
        if key.original_filename is not None:
            exact = await session.scalar(stmt.where(MediaItemRecord.original_filename == key.original_filename))
            if exact is not None:
                return exact

        candidates = (await session.scalars(stmt.order_by(MediaItemRecord.id.asc()))).all()
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        matching_referer = [candidate for candidate in candidates if candidate.referer == key.referer]
        if len(matching_referer) == 1:
            return matching_referer[0]
        return None

    async def _build_media_file_update(self, media_item: MediaItem) -> tuple[FileRow | None, tuple[HashRow, ...]]:
        download_filename = media_item.download_filename
        if not download_filename:
            return None, ()

        if media_item.complete_file and await asyncio.to_thread(media_item.complete_file.is_file):
            file_row = file_row_from_path(
                media_item.complete_file,
                original_filename=media_item.original_filename,
                referer=str(media_item.referer),
            )
        elif media_item.filesize is not None:
            file_row = FileRow(
                folder=str(media_item.download_folder),
                download_filename=download_filename,
                original_filename=media_item.original_filename,
                file_size=media_item.filesize,
                referer=str(media_item.referer),
            )
        else:
            return None, ()

        hashes = ()
        if media_item.hash:
            hashes = (HashRow(hash_type="xxh128", hash=media_item.hash),)
        return file_row, hashes

    async def _upsert_file(self, session: AsyncSession, file: FileRow, hashes: list[HashRow]) -> None:
        stmt = (
            select(MediaFileRecord)
            .options(selectinload(MediaFileRecord.hashes))
            .where(
                MediaFileRecord.folder == file.folder,
                MediaFileRecord.download_filename == file.download_filename,
            )
        )
        record = await session.scalar(stmt)
        if record is None:
            record = MediaFileRecord(
                media_item_id=file.media_item_id,
                folder=file.folder,
                download_filename=file.download_filename,
                original_filename=file.original_filename,
                file_size=file.file_size,
                referer=file.referer,
                date=file.date,
            )
            session.add(record)
            await session.flush()
            existing_hashes: dict[str, MediaHashRecord] = {}
        else:
            if file.media_item_id is not None:
                record.media_item_id = file.media_item_id
            if file.original_filename is not None:
                record.original_filename = file.original_filename
            if file.file_size is not None:
                record.file_size = file.file_size
            if file.referer is not None:
                record.referer = file.referer
            if file.date is not None:
                record.date = file.date
            existing_hashes = {hash_record.hash_type: hash_record for hash_record in record.hashes}
        for hash_row in hashes:
            hash_record = existing_hashes.get(hash_row.hash_type)
            if hash_record is None:
                session.add(
                    MediaHashRecord(
                        media_file_id=record.id,
                        hash_type=hash_row.hash_type,
                        hash=hash_row.hash,
                    )
                )
            else:
                hash_record.hash = hash_row.hash

    async def _yield_retry_rows(self, stmt: Any) -> AsyncGenerator[list[RetryMediaRow]]:
        offset = 0
        while True:
            async with self._sessionmaker() as session:
                rows = (await session.scalars(stmt.limit(_FETCH_MANY_SIZE).offset(offset))).all()
            if not rows:
                return
            yield [self._map_retry_media(row) for row in rows]
            offset += _FETCH_MANY_SIZE

    async def _new_schema_is_empty(self) -> bool:
        async with self._sessionmaker() as session:
            for table_name in ("media_items", "media_files", "media_hashes"):
                exists = await session.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
                if exists.first() is not None:
                    return False
        return True

    async def _import_legacy_database(self, db_state: DatabaseState) -> None:
        log(f"Importing legacy database rows from {self._db_path}")
        async with self._sessionmaker() as session:
            media_rows = await session.execute(
                text(
                    """
                    SELECT domain, url_path, referer, album_id, download_path, download_filename,
                           original_filename, file_size, duration, completed, created_at, completed_at
                    FROM media
                    """
                )
            )
            media_by_file: dict[tuple[str, str], int] = {}
            for row in media_rows.mappings():
                original_filename = row["original_filename"] or ""
                record = MediaItemRecord(
                    domain=row["domain"],
                    url_path=row["url_path"],
                    referer=row["referer"],
                    album_id=row["album_id"],
                    download_path=row["download_path"],
                    download_filename=row["download_filename"] or None,
                    original_filename=original_filename,
                    file_size=row["file_size"],
                    duration=row["duration"],
                    completed=bool(row["completed"]),
                    created_at=self._coerce_datetime(row["created_at"]),
                    completed_at=self._coerce_datetime(row["completed_at"]),
                )
                session.add(record)
                await session.flush()
                if record.download_filename:
                    media_by_file[(record.download_path, record.download_filename)] = record.id

            file_lookup: dict[tuple[str, str], MediaFileRecord] = {}
            if "files" in db_state.tables:
                file_rows = await session.execute(
                    text(
                        """
                        SELECT folder, download_filename, original_filename, file_size, referer, date
                        FROM files
                        """
                    )
                )
                for row in file_rows.mappings():
                    file_record = MediaFileRecord(
                        media_item_id=media_by_file.get((row["folder"], row["download_filename"])),
                        folder=row["folder"],
                        download_filename=row["download_filename"],
                        original_filename=row["original_filename"],
                        file_size=row["file_size"],
                        referer=row["referer"],
                        date=row["date"],
                    )
                    session.add(file_record)
                    await session.flush()
                    file_lookup[(file_record.folder, file_record.download_filename)] = file_record

            if "hash" in db_state.tables and file_lookup:
                hash_rows = await session.execute(
                    text(
                        """
                        SELECT folder, download_filename, hash_type, hash
                        FROM "hash"
                        """
                    )
                )
                for row in hash_rows.mappings():
                    file_record = file_lookup.get((row["folder"], row["download_filename"]))
                    if file_record is None:
                        continue
                    stmt = select(MediaHashRecord).where(
                        MediaHashRecord.media_file_id == file_record.id,
                        MediaHashRecord.hash_type == row["hash_type"],
                    )
                    existing = await session.scalar(stmt)
                    if existing is None:
                        session.add(
                            MediaHashRecord(
                                media_file_id=file_record.id,
                                hash_type=row["hash_type"],
                                hash=row["hash"],
                            )
                        )
                    else:
                        existing.hash = row["hash"]

            await session.commit()
        log("Legacy database import finished")

    def _map_media_item(self, record: MediaItemRecord) -> MediaItemRow:
        return MediaItemRow(
            id=record.id,
            domain=record.domain,
            db_path=record.url_path,
            referer=record.referer,
            album_id=record.album_id,
            download_path=record.download_path,
            download_filename=record.download_filename,
            original_filename=record.original_filename,
            file_size=record.file_size,
            duration=record.duration,
            completed=bool(record.completed),
            created_at=record.created_at,
            completed_at=record.completed_at,
        )

    def _map_file(self, record: MediaFileRecord) -> FileRow:
        return FileRow(
            folder=record.folder,
            download_filename=record.download_filename,
            original_filename=record.original_filename,
            file_size=record.file_size,
            referer=record.referer,
            date=record.date,
            media_item_id=record.media_item_id,
            hashes=tuple(HashRow(hash_type=hash_record.hash_type, hash=hash_record.hash) for hash_record in record.hashes),
        )

    def _map_retry_media(self, record: MediaItemRecord) -> RetryMediaRow:
        return RetryMediaRow(
            referer=record.referer,
            download_path=record.download_path,
            completed_at=record.completed_at,
            created_at=record.created_at,
        )

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)


__all__ = [
    "Database",
    "FileQuery",
    "FileRow",
    "HashRow",
    "MediaDefaults",
    "MediaItemRow",
    "MediaLookupKey",
    "RetryMediaRow",
]
