from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from cyberdrop_dl.clients.download_client import DownloadClient
from cyberdrop_dl.config.config_model import CompressionOptions, ConfigSettings
from cyberdrop_dl.managers.compression_manager import CompressionManager
from cyberdrop_dl.utils import yaml
from cyberdrop_dl.utils.pynv_transcode_worker import (
    _candidate_outputs_for_cleanup,
    _optimize_mp4_for_streaming,
    _resolve_output,
    _retag_hevc_sample_entries,
    _stringify_config,
)
from cyberdrop_dl.utils.pynv_transcode_worker import (
    main as pynv_worker_main,
)

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
        assert options.ffmpeg_nvenc_fallback is True
        assert options.video_codec == "hevc"
        assert options.video_workers_per_gpu == 2
        assert options.hevc_cq == 23
        assert options.av1_cq == 26
        assert options.video_cq_retry_step == 4
        assert options.video_cq_max == 35
        assert options.bf == 3
        assert options.gop == 60
        assert options.idrperiod == 60
        assert options.image_min_savings_percent == 0

        assert CompressionOptions.model_validate({"video_workers_per_gpu": 99}).video_workers_per_gpu == 2
        assert CompressionOptions.model_validate({"video_workers_per_gpu": 0}).video_workers_per_gpu == 1
        assert CompressionOptions.model_validate({"video_codec": "AV1"}).video_codec == "av1"

        yaml.save(config_file, ConfigSettings())
        serialized_config = yaml.load(config_file)
        assert serialized_config["compression_options"]["video_codec"] == "hevc"
        assert serialized_config["compression_options"]["video_workers_per_gpu"] == 2
        assert serialized_config["compression_options"]["ffmpeg_nvenc_fallback"] is True
        assert serialized_config["compression_options"]["image_min_savings_percent"] == 0
        assert serialized_config["compression_options"]["video_cq_retry_step"] == 4
        assert serialized_config["compression_options"]["video_cq_max"] == 35
        assert ConfigSettings.model_validate(serialized_config).compression_options.hevc_cq == 23
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_video_compression_skips_when_pynv_is_unavailable() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"not a real video, but PyNv is checked before probing")
        owner = FakeCompressionOwner(CompressionOptions(ffmpeg_nvenc_fallback=False))
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


def test_video_compression_falls_back_to_ffmpeg_nvenc_after_pynv_failure() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"x" * 100)
        owner = FakeCompressionOwner()
        compression_manager = CompressionManager(cast("Any", owner))
        calls: list[str] = []

        async def fake_probe(source: Path) -> SimpleNamespace:
            return SimpleNamespace(video=SimpleNamespace(codec="h264"))

        async def fail_pynv(source: Path, output: Path, gpu_id: int, codec: str, cq: int) -> None:
            calls.append(f"pynv:{gpu_id}:{codec}:{cq}")
            raise RuntimeError("pynv crashed")

        async def has_encoder(encoder: str) -> bool:
            calls.append(f"encoder:{encoder}")
            return True

        async def transcode_ffmpeg(source: Path, output: Path, gpu_id: int, codec: str, cq: int) -> None:
            calls.append(f"ffmpeg:{gpu_id}:{codec}:{cq}")
            await asyncio.to_thread(output.write_bytes, b"y" * 50)

        async def validate_video(path: Path) -> None:
            calls.append(f"validate:{path.name}")

        compression_manager._probe_if_available = fake_probe
        compression_manager._import_pynv = lambda: object()
        compression_manager._transcode_with_pynv_subprocess = fail_pynv
        compression_manager._ffmpeg_has_encoder = has_encoder
        compression_manager._transcode_with_ffmpeg_nvenc = transcode_ffmpeg
        compression_manager._validate_video = validate_video

        result = asyncio.run(compression_manager.compress_media_item(_media_item(video)))

        assert result is not None
        assert result.status == "compressed"
        assert result.backend == "ffmpeg_nvenc"
        assert video.read_bytes() == b"y" * 50
        assert calls == [
            "pynv:0:hevc:23",
            "encoder:hevc_nvenc",
            "ffmpeg:0:hevc:23",
            "validate:video.compressed.mp4",
        ]
        assert owner.progress_manager.results == [("compressed", 50)]
        assert owner.log_manager.rows[0]["backend"] == "ffmpeg_nvenc"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pynv_retries_higher_cq_when_output_is_too_large() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"x" * 100)
        owner = FakeCompressionOwner(CompressionOptions(ffmpeg_nvenc_fallback=False))
        compression_manager = CompressionManager(cast("Any", owner))
        calls: list[int] = []

        async def fake_probe(source: Path) -> SimpleNamespace:
            return SimpleNamespace(video=SimpleNamespace(codec="h264"))

        async def transcode_pynv(source: Path, output: Path, gpu_id: int, codec: str, cq: int) -> None:
            calls.append(cq)
            if cq == 23:
                raise RuntimeError("PyNvVideoCodec output exceeded safe size limit")
            await asyncio.to_thread(output.write_bytes, b"y" * 50)

        async def validate_video(path: Path) -> None:
            return None

        compression_manager._probe_if_available = fake_probe
        compression_manager._import_pynv = lambda: object()
        compression_manager._transcode_with_pynv_subprocess = transcode_pynv
        compression_manager._validate_video = validate_video

        result = asyncio.run(compression_manager.compress_media_item(_media_item(video)))

        assert result is not None
        assert result.status == "compressed"
        assert result.backend == "pynv"
        assert result.cq == 27
        assert calls == [23, 27]
        assert video.read_bytes() == b"y" * 50
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
    assert hevc_kwargs["gop"] == 60
    assert hevc_kwargs["idrperiod"] == 60
    assert hevc_kwargs["usedevicememory"] is True
    assert hevc_kwargs["usecpuinputbuffer"] is False
    assert hevc_kwargs["format"] == "NV12"
    assert hevc_kwargs["preset"] == "P6"
    assert hevc_kwargs["tuning_info"] == "high_quality"
    assert "gpu_id" not in hevc_kwargs


def test_ffmpeg_nvenc_command_uses_cuda_constqp_audio_copy_and_faststart(monkeypatch) -> None:
    monkeypatch.setattr("cyberdrop_dl.managers.compression_manager.ffmpeg.which_ffmpeg", lambda: "ffmpeg")
    compression_manager = CompressionManager(cast("Any", FakeCompressionOwner()))

    command = compression_manager._ffmpeg_nvenc_command(
        Path("input.mp4"),
        Path("input.compressed.mp4"),
        1,
        "hevc",
        23,
    )

    assert command[:2] == ("ffmpeg", "-y")
    assert ("-hwaccel", "cuda") == command[6:8]
    assert ("-hwaccel_device", "1") == command[8:10]
    assert "hevc_nvenc" in command
    assert ("-rc", "constqp") == command[command.index("-rc") : command.index("-rc") + 2]
    assert ("-qp", "23") == command[command.index("-qp") : command.index("-qp") + 2]
    assert ("-bf", "3") == command[command.index("-bf") : command.index("-bf") + 2]
    assert ("-c:a", "copy") == command[command.index("-c:a") : command.index("-c:a") + 2]
    assert ("-movflags", "+faststart") == command[command.index("-movflags") : command.index("-movflags") + 2]
    assert ("-tag:v", "hvc1") == command[command.index("-tag:v") : command.index("-tag:v") + 2]


def test_pynv_worker_stringifies_config_and_resolves_segment_output() -> None:
    root = _reset_test_dir()
    try:
        expected_output = root / "video.compressed.mp4"
        segmented_output = root / "video.compressed_0.000000_4.404400.mp4"
        faststart_output = segmented_output.with_suffix(segmented_output.suffix + ".faststart")
        segmented_output.write_bytes(b"compressed")
        faststart_output.write_bytes(b"faststart")

        assert _stringify_config({"constqp": 23, "usedevicememory": True}) == {
            "constqp": "23",
            "usedevicememory": "true",
        }
        assert _resolve_output(str(expected_output)) == segmented_output
        assert faststart_output in _candidate_outputs_for_cleanup(expected_output)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pynv_worker_prefers_full_file_mux_transcode(monkeypatch) -> None:
    root = _reset_test_dir()
    try:
        source = root / "input.mkv"
        output = root / "output.mkv"
        source.write_bytes(b"source")
        calls: list[str] = []

        class FakeDecoder:
            def __init__(self, path: str, gpu_id: int = 0, use_device_memory: bool = False) -> None:
                calls.append(f"decode:{Path(path).name}:{gpu_id}:{use_device_memory}")

            def get_stream_metadata(self) -> SimpleNamespace:
                return SimpleNamespace(duration=1.0)

            def __getitem__(self, index: int) -> bytes:
                return b"frame"

        class FakeTranscoder:
            def __init__(
                self,
                enc_file_path: str,
                muxed_file_path: str,
                gpu_id: int,
                cuda_context: int,
                cuda_stream: int,
                **kwargs: Any,
            ) -> None:
                calls.append(f"init:{Path(enc_file_path).name}:{Path(muxed_file_path).name}:{gpu_id}")
                self.output = Path(muxed_file_path)

            def transcode_with_mux(self) -> None:
                calls.append("transcode_with_mux")
                self.output.write_bytes(b"compressed")

            def segmented_transcode(self, start: float, end: float) -> None:
                raise AssertionError("segmented_transcode should not be used for whole-file compression")

        fake_pynv = SimpleNamespace(Transcoder=FakeTranscoder, SimpleDecoder=FakeDecoder)
        monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_pynv)

        assert pynv_worker_main([str(source), str(output), "0", '{"codec": "hevc"}']) == 0
        assert output.read_bytes() == b"compressed"
        assert "transcode_with_mux" in calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mp4_faststart_moves_moov_before_mdat_and_patches_offsets() -> None:
    root = _reset_test_dir()
    try:
        video_path = root / "video.mp4"
        original_chunk_offset = 100
        stco = _mp4_atom(
            b"stco",
            b"\0\0\0\0" + (1).to_bytes(4, "big") + original_chunk_offset.to_bytes(4, "big"),
        )
        moov = _mp4_atom(b"moov", _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stco)))))
        video_path.write_bytes(_mp4_atom(b"ftyp", b"isom\0\0\0\0") + _mp4_atom(b"mdat", b"x" * 10) + moov)

        _optimize_mp4_for_streaming(video_path)

        optimized = video_path.read_bytes()
        assert optimized.find(b"moov") < optimized.find(b"mdat")
        stco_type_offset = optimized.find(b"stco")
        patched_offset = int.from_bytes(optimized[stco_type_offset + 12 : stco_type_offset + 16], "big")
        assert patched_offset == original_chunk_offset + len(moov)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _mp4_atom(atom_type: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + atom_type + payload


def test_retag_hevc_sample_entries_rewrites_hev1_to_hvc1_for_thumbnails() -> None:
    root = _reset_test_dir()
    try:
        video_path = root / "video.mp4"
        hvcC = _mp4_atom(b"hvcC", b"\x01" * 23)
        sample_entry_payload = (
            b"\x00" * 6
            + b"\x00\x01"
            + b"\x00" * 16
            + b"\x00\x80\x00\x80"
            + b"\x00" * 14
            + b"\x18\x00\xff\xff"
            + hvcC
        )
        hev1 = _mp4_atom(b"hev1", sample_entry_payload)
        stsd = _mp4_atom(b"stsd", b"\x00\x00\x00\x00" + (1).to_bytes(4, "big") + hev1)
        moov = _mp4_atom(
            b"moov",
            _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stsd)))),
        )
        ftyp = _mp4_atom(b"ftyp", b"isom" + b"\x00" * 4)
        mdat = _mp4_atom(b"mdat", b"x" * 32)
        original_bytes = ftyp + moov + mdat
        video_path.write_bytes(original_bytes)

        _retag_hevc_sample_entries(video_path)

        rewritten = video_path.read_bytes()
        assert rewritten.find(b"hvc1") != -1
        assert rewritten.find(b"hev1") == -1
        assert rewritten.find(b"hvcC") != -1
        assert len(rewritten) == len(original_bytes)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_retag_hevc_sample_entries_skips_non_hevc_streams_and_non_mp4_files() -> None:
    root = _reset_test_dir()
    try:
        avc_path = root / "avc.mp4"
        avc1 = _mp4_atom(b"avc1", b"\x00" * 86)
        stsd = _mp4_atom(b"stsd", b"\x00\x00\x00\x00" + (1).to_bytes(4, "big") + avc1)
        moov = _mp4_atom(
            b"moov",
            _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stsd)))),
        )
        ftyp = _mp4_atom(b"ftyp", b"isom" + b"\x00" * 4)
        mdat = _mp4_atom(b"mdat", b"x" * 16)
        avc_bytes = ftyp + moov + mdat
        avc_path.write_bytes(avc_bytes)
        _retag_hevc_sample_entries(avc_path)
        assert avc_path.read_bytes() == avc_bytes

        mkv_path = root / "video.mkv"
        mkv_bytes = b"hev1" + b"\x00" * 32
        mkv_path.write_bytes(mkv_bytes)
        _retag_hevc_sample_entries(mkv_path)
        assert mkv_path.read_bytes() == mkv_bytes
    finally:
        shutil.rmtree(root, ignore_errors=True)


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

    assert compression_manager._queue_worker_count() == 4

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


def test_video_temp_output_size_guard_stops_expanding_outputs() -> None:
    root = _reset_test_dir()
    try:
        source = root / "video.mov"
        temp_output = root / "video.compressed.mov"
        timestamped_output = root / "video.compressed_0.000000_101.031670.mov"
        faststart_output = timestamped_output.with_suffix(timestamped_output.suffix + ".faststart")
        source.write_bytes(b"x" * 100)
        temp_output.write_bytes(b"x" * 50)
        timestamped_output.write_bytes(b"x" * 96)
        faststart_output.write_bytes(b"x" * 120)

        compression_manager = CompressionManager(cast("Any", FakeCompressionOwner(CompressionOptions())))

        assert compression_manager._max_acceptable_output_size(source) == 95
        assert compression_manager._get_oversized_temp_output(temp_output, 95) == (timestamped_output.name, 96)

        timestamped_output.write_bytes(b"x" * 80)
        assert compression_manager._get_oversized_temp_output(temp_output, 95) == (faststart_output.name, 120)
    finally:
        shutil.rmtree(root, ignore_errors=True)


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


def test_image_threshold_accepts_any_smaller_file_while_video_keeps_minimum_savings() -> None:
    root = _reset_test_dir()
    try:
        source = root / "media.bin"
        temp_output = root / "media.compressed.bin"
        compression_manager = CompressionManager(cast("Any", FakeCompressionOwner(CompressionOptions())))

        source.write_bytes(b"x" * 100)
        temp_output.write_bytes(b"x" * 99)
        compression_manager._validate_image = lambda path: None

        image_result = asyncio.run(
            compression_manager._finalize_output(
                source,
                temp_output,
                "image",
                media_type="image",
                backend="pillow",
                path=source,
            )
        )

        assert image_result.status == "compressed"
        assert source.stat().st_size == 99

        source.write_bytes(b"x" * 100)
        temp_output.write_bytes(b"x" * 99)

        async def validate_video(path: Path) -> None:
            return None

        compression_manager._validate_video = validate_video
        video_result = asyncio.run(
            compression_manager._finalize_output(
                source,
                temp_output,
                "video",
                media_type="video",
                backend="pynv",
                path=source,
            )
        )

        assert video_result.status == "skipped"
        assert video_result.error == "Compressed output was not small enough"
        assert source.stat().st_size == 100
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_compression_queue_runs_workers_in_parallel_for_throughput() -> None:
    async def run_queue() -> tuple[int, list[str], bool]:
        owner = FakeCompressionOwner(CompressionOptions(gpu_ids=[0], video_workers_per_gpu=2))
        compression_manager = CompressionManager(cast("Any", owner))
        order: list[str] = []
        active = 0
        peak_active = 0
        both_workers_active = asyncio.Event()
        media_items = [
            cast("MediaItem", SimpleNamespace(complete_file=Path("one.jpg"), filename="one.jpg", is_segment=False)),
            cast("MediaItem", SimpleNamespace(complete_file=Path("two.jpg"), filename="two.jpg", is_segment=False)),
            cast("MediaItem", SimpleNamespace(complete_file=Path("three.jpg"), filename="three.jpg", is_segment=False)),
            cast("MediaItem", SimpleNamespace(complete_file=Path("four.jpg"), filename="four.jpg", is_segment=False)),
        ]

        async def fake_compress(media_item: MediaItem) -> None:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            if peak_active >= 2:
                both_workers_active.set()
            try:
                await asyncio.wait_for(both_workers_active.wait(), timeout=1)
                order.append(f"compress:{media_item.filename}")
            finally:
                active -= 1

        async def fake_process(media_item: MediaItem, domain: str) -> None:
            order.append(f"process:{domain}:{media_item.filename}")

        async def fake_handle(media_item: MediaItem, downloaded: bool = False) -> None:
            order.append(f"handle:{downloaded}:{media_item.filename}")

        async def fake_finalize(media_item: MediaItem, downloaded: bool) -> None:
            order.append(f"finalize:{downloaded}:{media_item.filename}")

        completion_lock = asyncio.Lock()
        await completion_lock.acquire()
        compression_manager.compress_media_item = fake_compress
        for index, media_item in enumerate(media_items):
            await compression_manager.enqueue_completed_download(
                "example.com",
                media_item,
                fake_process,
                fake_handle,
                finalize_download=fake_finalize,
                completion_lock=completion_lock if index == 0 else None,
            )

        await compression_manager.join()
        lock_released = not completion_lock.locked()
        await compression_manager.close()
        return peak_active, order, lock_released

    peak_active, order, lock_released = asyncio.run(run_queue())

    assert peak_active == 2
    assert lock_released
    for filename in ("one.jpg", "two.jpg", "three.jpg", "four.jpg"):
        assert f"compress:{filename}" in order
        assert f"process:example.com:{filename}" in order
        assert f"handle:True:{filename}" in order
        assert f"finalize:True:{filename}" in order


def test_download_lifecycle_enqueues_after_rename_and_duration_check() -> None:
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
            async def enqueue_completed_download(
                self,
                domain: str,
                media_item: MediaItem,
                process_completed: Any,
                handle_completion: Any,
                *,
                downloaded: bool = True,
                finalize_download: Any = None,
                completion_lock: asyncio.Lock | None = None,
            ) -> None:
                order.append("enqueue")
                assert domain == "example.com"
                assert downloaded is True
                assert process_completed == client.process_completed
                assert handle_completion == client.handle_media_item_completion
                assert finalize_download is None
                assert completion_lock is None
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
            "enqueue",
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
