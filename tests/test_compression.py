from __future__ import annotations

import asyncio
import contextlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from cyberdrop_dl.clients.download_client import DownloadClient
from cyberdrop_dl.config.config_model import CompressionOptions, ConfigSettings
from cyberdrop_dl.managers.compression_manager import CompressionManager
from cyberdrop_dl.utils import yaml

if TYPE_CHECKING:
    from cyberdrop_dl.data_structures.url_objects import MediaItem

TEST_DIR = Path("tmp_compression_tests")


class FakeProgressManager:
    def __init__(self) -> None:
        self.results: list[tuple[str, int]] = []

    def add_compression_result(self, status: str, bytes_saved: int = 0) -> None:
        self.results.append((status, bytes_saved))


class FakeLogManager:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def write_compression_report(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


class FakeCompressionOwner:
    def __init__(self, options: CompressionOptions | None = None) -> None:
        self.config = SimpleNamespace(compression_options=options or CompressionOptions())
        self.progress_manager = FakeProgressManager()
        self.log_manager = FakeLogManager()


def _reset_test_dir() -> Path:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir()
    return TEST_DIR


def _media_item(path: Path, url: str = "https://example.com/media") -> MediaItem:
    return cast(
        "MediaItem",
        SimpleNamespace(
            complete_file=path,
            filesize=path.stat().st_size,
            is_segment=False,
            url=url,
        ),
    )


def test_compression_options_defaults_validation_and_yaml_serialization() -> None:
    root = _reset_test_dir()
    config_file = root / "config.yaml"
    try:
        options = ConfigSettings().compression_options
        assert options.enabled is True
        assert options.compress_videos is True
        assert options.compress_images is True
        assert options.video_backend == "pynv"
        assert options.video_codec == "hevc"
        assert options.video_workers_per_gpu == 2
        assert options.hevc_cq == 23
        assert options.av1_cq == 26
        assert options.bf == 3

        assert CompressionOptions.model_validate({"video_workers_per_gpu": 99}).video_workers_per_gpu == 2
        assert CompressionOptions.model_validate({"video_workers_per_gpu": 0}).video_workers_per_gpu == 1
        assert CompressionOptions.model_validate({"video_codec": "AV1"}).video_codec == "av1"

        yaml.save(config_file, ConfigSettings())
        serialized_config = yaml.load(config_file)
        assert serialized_config["compression_options"]["video_codec"] == "hevc"
        assert serialized_config["compression_options"]["video_workers_per_gpu"] == 2
        assert ConfigSettings.model_validate(serialized_config).compression_options.hevc_cq == 23
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_video_compression_skips_when_pynv_is_unavailable() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"not a real video, but PyNv is checked before probing")
        owner = FakeCompressionOwner()
        compression_manager = CompressionManager(cast("Any", owner))
        compression_manager._import_pynv = lambda: None

        result = asyncio.run(compression_manager.compress_media_item(_media_item(video)))

        assert result is not None
        assert result.status == "skipped"
        assert result.error == "PyNvVideoCodec is not installed"
        assert owner.progress_manager.results == [("skipped", 0)]
        assert owner.log_manager.rows[0]["status"] == "skipped"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pynv_encoder_kwargs_use_gpu_buffers_constqp_and_b_frames() -> None:
    compression_manager = CompressionManager(cast("Any", FakeCompressionOwner()))

    hevc_kwargs = compression_manager._pynv_transcode_kwargs("hevc", 23)
    av1_kwargs = compression_manager._pynv_transcode_kwargs("av1", 26)

    assert hevc_kwargs["codec"] == "hevc"
    assert hevc_kwargs["constqp"] == 23
    assert av1_kwargs["codec"] == "av1"
    assert av1_kwargs["constqp"] == 26
    assert hevc_kwargs["bf"] == 3
    assert hevc_kwargs["usedevicememory"] is True
    assert hevc_kwargs["usecpuinputbuffer"] is False
    assert hevc_kwargs["format"] == "NV12"
    assert hevc_kwargs["preset"] == "P6"
    assert hevc_kwargs["tuning_info"] == "high_quality"
    assert "gpu_id" not in hevc_kwargs


def test_pynv_transcode_uses_installed_transcoder_api_shape() -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class FakeTranscoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("init", args, kwargs))

        def transcode_with_mux(self) -> None:
            calls.append(("transcode_with_mux", (), {}))

    fake_pynv = SimpleNamespace(Transcoder=FakeTranscoder)
    compression_manager = CompressionManager(cast("Any", FakeCompressionOwner()))
    source = Path("input.mp4")
    output = Path("output.mp4")

    compression_manager._transcode_with_pynv(cast("Any", fake_pynv), source, output, 1, "hevc", 23)

    assert calls[0][0] == "init"
    assert calls[0][1] == ()
    assert calls[0][2]["enc_file_path"] == str(source)
    assert calls[0][2]["muxed_file_path"] == str(output)
    assert calls[0][2]["gpu_id"] == 1
    assert calls[0][2]["cuda_context"] == 0
    assert calls[0][2]["cuda_stream"] == 0
    assert calls[0][2]["codec"] == "hevc"
    assert calls[0][2]["constqp"] == "23"
    assert calls[0][2]["usedevicememory"] == "true"
    assert calls[1] == ("transcode_with_mux", (), {})


def test_video_slots_allow_at_most_two_jobs_per_gpu() -> None:
    owner = FakeCompressionOwner(CompressionOptions(gpu_ids=[0, 1], video_workers_per_gpu=2))
    compression_manager = CompressionManager(cast("Any", owner))
    active_by_gpu = {0: 0, 1: 0}
    peak_by_gpu = {0: 0, 1: 0}

    async def run_job(gpu_id: int) -> None:
        async with compression_manager._video_slot(gpu_id):
            active_by_gpu[gpu_id] += 1
            peak_by_gpu[gpu_id] = max(peak_by_gpu[gpu_id], active_by_gpu[gpu_id])
            await asyncio.sleep(0.01)
            active_by_gpu[gpu_id] -= 1

    async def run_all_jobs() -> None:
        await asyncio.gather(*(run_job(gpu_id) for gpu_id in (0, 1) for _ in range(6)))

    asyncio.run(run_all_jobs())

    assert peak_by_gpu == {0: 2, 1: 2}


def test_resolve_pynv_timestamped_segment_output() -> None:
    root = _reset_test_dir()
    try:
        source = root / "video.mp4"
        expected_output = root / "video.compressed.mp4"
        timestamped_output = root / "video.compressed_0.00_12.50.mp4"
        source.write_bytes(b"source")
        timestamped_output.write_bytes(b"compressed")

        compression_manager = CompressionManager(cast("Any", FakeCompressionOwner()))

        assert compression_manager._resolve_pynv_output(source, expected_output) == timestamped_output
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_image_compression_preserves_extension_and_replaces_only_when_smaller() -> None:
    root = _reset_test_dir()
    try:
        image_path = root / "sample.jpg"
        image = Image.new("RGB", (384, 384))
        image.putdata(
            [
                ((x * 17 + y * 3) % 256, (x * 7 + y * 19) % 256, (x * y) % 256)
                for y in range(384)
                for x in range(384)
            ],
        )
        image.save(image_path, quality=95)
        original_size = image_path.stat().st_size

        owner = FakeCompressionOwner(CompressionOptions(min_savings_percent=1))
        compression_manager = CompressionManager(cast("Any", owner))
        media_item = _media_item(image_path)
        result = asyncio.run(compression_manager.compress_media_item(media_item))

        assert result is not None
        assert result.status == "compressed"
        assert image_path.suffix == ".jpg"
        assert image_path.stat().st_size < original_size
        assert not image_path.with_name("sample.compressed.jpg").exists()
        assert media_item.filesize == image_path.stat().st_size
        with Image.open(image_path) as compressed_image:
            compressed_image.verify()
        assert owner.progress_manager.results == [("compressed", original_size - image_path.stat().st_size)]
        assert owner.log_manager.rows[0]["media_type"] == "image"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_lifecycle_compresses_after_rename_before_history_and_hash() -> None:
    root = _reset_test_dir()
    try:
        partial_file = root / "download.jpg.part"
        complete_file = root / "download.jpg"
        partial_file.write_text("downloaded", encoding="utf8")
        order: list[str] = []

        class FakeHistoryTable:
            async def add_duration(self, domain: str, media_item: MediaItem) -> None:
                order.append("add_duration")

        class FakeClientManager:
            def __init__(self, manager: Any) -> None:
                self.manager = manager

            @contextlib.contextmanager
            def request_context(self, domain: str):
                yield

            async def check_file_duration(self, media_item: MediaItem) -> bool:
                order.append("duration")
                assert media_item.complete_file.exists()
                assert not media_item.partial_file.exists()
                return True

        class FakeCompressionManager:
            async def compress_media_item(self, media_item: MediaItem) -> None:
                order.append("compress")
                assert media_item.complete_file.read_text(encoding="utf8") == "downloaded"

        running = asyncio.Event()
        running.set()
        manager = SimpleNamespace(
            config=ConfigSettings(),
            config_manager=SimpleNamespace(settings_data=ConfigSettings()),
            states=SimpleNamespace(RUNNING=running),
            db_manager=SimpleNamespace(history_table=FakeHistoryTable()),
            compression_manager=FakeCompressionManager(),
        )
        client_manager = FakeClientManager(manager)
        client = DownloadClient(cast("Any", manager), cast("Any", client_manager))

        async def fake_download(domain: str, media_item: MediaItem) -> bool:
            order.append("download")
            return True

        async def fake_process_completed(media_item: MediaItem, domain: str) -> None:
            order.append("process_completed")

        async def fake_handle_completion(media_item: MediaItem, downloaded: bool = False) -> None:
            order.append("handle_completion")
            assert downloaded is True

        client._download = fake_download
        client.process_completed = fake_process_completed
        client.handle_media_item_completion = fake_handle_completion
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=complete_file,
                is_segment=False,
                partial_file=partial_file,
                url="https://example.com/download.jpg",
            ),
        )

        assert asyncio.run(client.download_file("example.com", media_item)) is True
        assert order == [
            "download",
            "duration",
            "add_duration",
            "compress",
            "process_completed",
            "handle_completion",
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
