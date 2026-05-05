from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from cyberdrop_dl.clients.download_client import DownloadClient
from cyberdrop_dl.config.config_model import CompressionOptions, ConfigSettings
from cyberdrop_dl.config.global_model import GlobalSettings
from cyberdrop_dl.crawlers.crawler import Crawler
from cyberdrop_dl.data_structures.url_objects import AbsoluteHttpURL
from cyberdrop_dl.database.types import MediaItemRow
from cyberdrop_dl.managers.compression_manager import CompressionManager
from tests.test_compression import FakeCompressionOwner, _reset_test_dir


async def _noop_process(media_item: Any, domain: str) -> None:
    return None


async def _noop_handle(media_item: Any, downloaded: bool = False) -> None:
    return None


def _fake_manager(**kwargs: Any) -> Any:
    settings = kwargs.pop("settings_data", ConfigSettings())
    global_settings = kwargs.pop("global_settings_data", GlobalSettings())
    running = asyncio.Event()
    running.set()
    base = {
        "config": settings,
        "config_manager": SimpleNamespace(settings_data=settings, global_settings_data=global_settings, loaded_config="test"),
        "states": SimpleNamespace(RUNNING=running),
        "parsed_args": SimpleNamespace(cli_only_args=SimpleNamespace(retry_any=False)),
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_existing_file_recovery_queues_uncompressed_video_but_not_marked_video() -> None:
    root = _reset_test_dir()
    try:
        owner = FakeCompressionOwner(CompressionOptions())
        manager = CompressionManager(cast("Any", owner))
        queued: list[tuple[str, str, bool]] = []

        async def fake_enqueue(
            domain: str,
            media_item: Any,
            process_completed: Any,
            handle_completion: Any,
            *,
            downloaded: bool = True,
            finalize_download: Any = None,
            completion_lock: asyncio.Lock | None = None,
        ) -> None:
            queued.append((domain, media_item.filename, downloaded))

        manager.enqueue_completed_download = fake_enqueue

        video = root / "video.mp4"
        video.write_bytes(b"data")
        video_item = cast(
            "Any",
            SimpleNamespace(
                complete_file=video,
                download_folder=root,
                filename="video.mp4",
                download_filename="video.mp4",
                is_segment=False,
            ),
        )

        marked = root / "[COMPRESSED] clip.mp4"
        marked.write_bytes(b"data")
        marked_item = cast(
            "Any",
            SimpleNamespace(
                complete_file=marked,
                download_folder=root,
                filename=marked.name,
                download_filename=marked.name,
                is_segment=False,
            ),
        )

        async def run() -> None:
            assert await manager.enqueue_existing_file_if_needed(
                "example.com",
                video_item,
                _noop_process,
                _noop_handle,
                downloaded=False,
            )
            assert not await manager.enqueue_existing_file_if_needed(
                "example.com",
                marked_item,
                _noop_process,
                _noop_handle,
                downloaded=False,
                allow_images=False,
            )

        asyncio.run(run())

        assert queued == [("example.com", "video.mp4", False)]
    finally:
        for path in root.glob("*"):
            path.unlink(missing_ok=True)
        root.rmdir()


def test_existing_compressed_pair_removes_duplicate_source_and_updates_completion() -> None:
    root = _reset_test_dir()
    try:
        owner = FakeCompressionOwner(CompressionOptions())
        manager = CompressionManager(cast("Any", owner))
        callbacks: list[tuple[str, Any, Any, Any]] = []

        source = root / "video.mp4"
        marked = root / "[COMPRESSED] video.mp4"
        source.write_bytes(b"x" * 100)
        marked.write_bytes(b"x" * 50)

        media_item = cast(
            "Any",
            SimpleNamespace(
                complete_file=source,
                download_folder=root,
                filename="video.mp4",
                download_filename="video.mp4",
                filesize=100,
                is_segment=False,
            ),
        )

        async def process_completed(media_item: Any, domain: str) -> None:
            callbacks.append(("process", domain, media_item.complete_file, media_item.download_filename))

        async def handle_completion(media_item: Any, downloaded: bool = False) -> None:
            callbacks.append(("handle", downloaded, media_item.complete_file, media_item.filesize))

        handled = asyncio.run(
            manager.enqueue_existing_file_if_needed(
                "example.com",
                media_item,
                process_completed,
                handle_completion,
                downloaded=False,
                allow_images=False,
            )
        )

        assert handled is True
        assert not source.exists()
        assert marked.exists()
        assert media_item.complete_file == marked
        assert media_item.download_filename == marked.name
        assert media_item.filesize == 50
        assert callbacks == [
            ("process", "example.com", marked, marked.name),
            ("handle", False, marked, 50),
        ]
    finally:
        for path in root.glob("*"):
            path.unlink(missing_ok=True)
        root.rmdir()


def test_download_client_requeues_existing_local_file_for_compression() -> None:
    root = _reset_test_dir()
    try:
        complete_file = root / "video.mp4"
        complete_file.write_bytes(b"data")
        calls: list[str] = []
        db_updates: list[str] = []

        class FakeDatabase:
            async def update_media_item(self, media_item: Any) -> None:
                db_updates.append(media_item.download_filename or media_item.filename)

        class FakeProgress:
            def __init__(self) -> None:
                self.args: list[bool] = []

            def add_previously_completed(self, increase_total: bool = True) -> None:
                self.args.append(increase_total)

        class FakeCompressionManager:
            async def enqueue_existing_file_if_needed(
                self,
                domain: str,
                media_item: Any,
                process_completed: Any,
                handle_completion: Any,
                *,
                downloaded: bool = False,
                allow_images: bool = True,
            ) -> bool:
                calls.append(f"enqueue:{domain}:{downloaded}:{allow_images}:{media_item.filename}")
                return True

        class FakeClientManager:
            def __init__(self, manager: Any) -> None:
                self.manager = manager

            async def check_http_status(self, resp: Any, *, download: bool = False) -> None:
                return None

            def check_content_length(self, headers: dict[str, str]) -> None:
                return None

        manager = _fake_manager(
            database=FakeDatabase(),
            progress_manager=SimpleNamespace(download_progress=FakeProgress()),
            compression_manager=FakeCompressionManager(),
            log_manager=SimpleNamespace(write_skipped_duplicate_url_log=lambda media_item: calls.append("skip-log")),
        )
        client = DownloadClient(cast("Any", manager), cast("Any", FakeClientManager(manager)))
        client.process_completed = lambda media_item, domain: calls.append("process_completed")  # type: ignore[assignment]
        client.handle_media_item_completion = lambda media_item, downloaded=False: calls.append("handle_completion")  # type: ignore[assignment]
        media_item = cast(
            "Any",
            SimpleNamespace(
                url=AbsoluteHttpURL("https://example.com/video.mp4"),
                referer=AbsoluteHttpURL("https://example.com/post"),
                domain="example.com",
                download_folder=root,
                filename="video.mp4",
                original_filename="video.mp4",
                download_filename="video.mp4",
                filesize=4,
                ext=".mp4",
                db_path="/video",
                is_segment=False,
                complete_file=None,
                partial_file=None,
                task_id=None,
                datetime=None,
            ),
        )
        response = SimpleNamespace(status=200, headers={"Content-Length": "4", "Content-Type": "video/mp4"})

        result = asyncio.run(client._process_response(media_item, "example.com", 0, response))

        assert result is False
        assert calls == ["enqueue:example.com:False:True:video.mp4"]
        assert db_updates == ["video.mp4"]
        assert manager.progress_manager.download_progress.args == [False]
    finally:
        for path in root.glob("*"):
            path.unlink(missing_ok=True)
        root.rmdir()


class RecoveryCrawler(Crawler):
    DOMAIN = "example.com"
    PRIMARY_URL = AbsoluteHttpURL("https://example.com")
    SUPPORTED_PATHS = {"Files": ("/file",)}

    async def fetch(self, scrape_item: Any) -> None:
        return None


def test_completed_video_is_requeued_for_compression_recovery() -> None:
    root = _reset_test_dir()
    try:
        stored_root = root / "stored"
        stored_root.mkdir()
        (stored_root / "video.mp4").write_bytes(b"data")
        calls: list[str] = []

        class FakeDatabase:
            async def get_media_item(self, key: Any, defaults: Any) -> MediaItemRow:
                return MediaItemRow(
                    id=1,
                    domain="example.com",
                    db_path="/video",
                    referer="https://example.com/post",
                    album_id=None,
                    download_path=str(stored_root),
                    download_filename="video.mp4",
                    original_filename="video.mp4",
                    file_size=4,
                    duration=None,
                    completed=True,
                    created_at=None,
                    completed_at=None,
                )

            async def update_media_item(self, media_item: Any) -> None:
                calls.append(f"update:{media_item.download_filename}")

        class FakeCompressionManager:
            async def enqueue_existing_file_if_needed(
                self,
                domain: str,
                media_item: Any,
                process_completed: Any,
                handle_completion: Any,
                *,
                downloaded: bool = False,
                allow_images: bool = True,
            ) -> bool:
                calls.append(
                    f"enqueue:{domain}:{downloaded}:{allow_images}:{media_item.filename}:{media_item.download_folder == stored_root}"
                )
                return True

        class FakeDownloadProgress:
            def add_previously_completed(self, increase_total: bool = True) -> None:
                calls.append(f"previous:{increase_total}")

        manager = _fake_manager(
            database=FakeDatabase(),
            progress_manager=SimpleNamespace(download_progress=FakeDownloadProgress()),
            compression_manager=FakeCompressionManager(),
        )
        crawler = RecoveryCrawler(cast("Any", manager))
        crawler.downloader = SimpleNamespace(client=SimpleNamespace(process_completed=_noop_process, handle_media_item_completion=_noop_handle))
        media_item = cast(
            "Any",
            SimpleNamespace(
                url=AbsoluteHttpURL("https://example.com/video.mp4"),
                referer=AbsoluteHttpURL("https://example.com/post"),
                domain="example.com",
                download_folder=root / "new-run-folder",
                filename="video.mp4",
                original_filename="video.mp4",
                download_filename=None,
                filesize=None,
                ext=".mp4",
                db_path="/video",
                album_id=None,
                duration=None,
                datetime=None,
                is_segment=False,
                complete_file=None,
            ),
        )

        asyncio.run(crawler.handle_media_item(media_item))

        assert calls == [
            "update:video.mp4",
            "previous:True",
            "enqueue:example.com:False:False:video.mp4:True",
        ]
    finally:
        for path in root.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            path.rmdir()
        root.rmdir()
