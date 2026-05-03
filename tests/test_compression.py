from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from cyberdrop_dl.clients.download_client import DownloadClient
from cyberdrop_dl.config.config_model import CompressionOptions, ConfigSettings
from cyberdrop_dl.managers.compression_manager import (
    CompressionManager,
    CompressionResult,
    EffectiveVideoSettings,
    VideoCodecCapabilities,
    _PersistentWorkerDied,
    _format_pynv_exception,
)
from cyberdrop_dl.utils import yaml
from cyberdrop_dl.utils.pynv_transcode_worker import (
    _build_hvcc_body,
    _candidate_outputs_for_cleanup,
    _extract_inline_hevc_param_sets,
    _format_exception,
    _optimize_mp4_for_streaming,
    _repair_sample_entry_for_target_codec,
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
        assert options.video_profile == "hevc_balanced"
        assert options.video_backend == "pynv"
        assert options.ffmpeg_nvenc_fallback is False
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

        assert CompressionOptions.model_validate({"video_workers_per_gpu": 99}).video_workers_per_gpu == 99
        assert CompressionOptions.model_validate({"video_workers_per_gpu": 0}).video_workers_per_gpu == 1
        assert CompressionOptions.model_validate({"video_profile": "AV1_SAVINGS"}).video_profile == "av1_savings"
        assert CompressionOptions.model_validate({"video_codec": "AV1"}).video_codec == "av1"
        assert CompressionOptions().effective_video_profile() == "hevc_balanced"
        assert CompressionOptions(video_profile="custom").effective_video_profile() == "custom"
        assert CompressionOptions(preset="P4").effective_video_profile() == "custom"

        yaml.save(config_file, ConfigSettings())
        serialized_config = yaml.load(config_file)
        assert serialized_config["compression_options"]["video_profile"] == "hevc_balanced"
        assert serialized_config["compression_options"]["video_codec"] == "hevc"
        assert serialized_config["compression_options"]["video_workers_per_gpu"] == 2
        assert serialized_config["compression_options"]["ffmpeg_nvenc_fallback"] is False
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


def test_pynv_retries_higher_cq_when_output_is_too_large() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"x" * 100)
        owner = FakeCompressionOwner(CompressionOptions(ffmpeg_nvenc_fallback=False))
        compression_manager = CompressionManager(cast("Any", owner))
        calls: list[int] = []

        async def ensure_runtime() -> object:
            return object()

        async def transcode_pynv(source: Path, output: Path, gpu_id: int, settings: EffectiveVideoSettings) -> None:
            calls.append(settings.cq)
            if settings.cq == 23:
                raise RuntimeError("PyNvVideoCodec output exceeded safe size limit")
            await asyncio.to_thread(output.write_bytes, b"y" * 50)

        compression_manager._ensure_video_runtime = ensure_runtime
        compression_manager._transcode_with_pynv_subprocess = transcode_pynv

        result = asyncio.run(compression_manager.compress_media_item(_media_item(video)))

        assert result is not None
        assert result.status == "compressed"
        assert result.backend == "pynv"
        assert result.cq == 27
        assert calls == [23, 27]
        assert Path(result.path).read_bytes() == b"y" * 50
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pynv_invalid_input_errors_are_concise_and_non_retryable() -> None:
    traceback_output = """
Traceback (most recent call last):
  File "pynv_transcode_worker.py", line 44, in main
    transcoder = PyNvVideoCodec.Transcoder(...)
_PyNvVideoCodec.PyNvVCException: FFmpegDemuxer::CreateFormatContext :
Error code : -1094995529
Error Type : avformat_open_input(&ctx, szFilePath, NULL, NULL) returned error " Invalid data found when processing input"
    """
    compression_manager = CompressionManager(cast("Any", FakeCompressionOwner()))

    assert _format_pynv_exception(RuntimeError("Invalid data found when processing input")) == (
        "PyNvVideoCodec could not open the input video. "
        "The file is unsupported, corrupted, incomplete, or not a real video container."
    )
    assert _format_pynv_exception(RuntimeError("Invalid data found when processing input"), "output") == (
        "PyNvVideoCodec created an invalid output video at this CQ"
    )
    assert _format_exception(RuntimeError("Invalid data found when processing input")) == (
        "PyNvVideoCodec could not open the input video. "
        "The file is unsupported, corrupted, incomplete, or not a real video container."
    )
    assert _format_exception(RuntimeError("Invalid data found when processing input"), "output") == (
        "PyNvVideoCodec created an invalid output video at this CQ"
    )
    assert compression_manager._should_retry_with_higher_cq(traceback_output, 23) is False
    assert compression_manager._should_retry_with_higher_cq(
        "PyNvVideoCodec created an invalid output video at this CQ",
        23,
    )

    timescale_output = "[mov,mp4,m4a,3gp,3g2,mj2 @ 000001C6C0394040] stream 0, timescale not set"
    assert _format_pynv_exception(RuntimeError(timescale_output)) == (
        "PyNvVideoCodec could not read this MP4 stream timing metadata"
    )
    assert _format_exception(RuntimeError(timescale_output)) == (
        "PyNvVideoCodec could not read this MP4 stream timing metadata "
        "(timescale not set). The original file was kept and compression was skipped."
    )


def test_pynv_encoder_kwargs_use_gpu_buffers_constqp_and_b_frames() -> None:
    hevc_owner = FakeCompressionOwner()
    compression_manager = CompressionManager(cast("Any", hevc_owner))
    compression_manager._codec_capabilities = {
        0: {
            "hevc": VideoCodecCapabilities(
                codec="hevc",
                supported=True,
                num_encoder_engines=2,
                num_max_bframes=5,
                support_lookahead=True,
                support_temporal_aq=True,
                support_10bit_encode=True,
            ),
            "av1": VideoCodecCapabilities(codec="av1", supported=False),
        }
    }

    hevc_kwargs = compression_manager._pynv_transcode_kwargs(compression_manager._effective_video_settings(0))
    av1_manager = CompressionManager(
        cast("Any", FakeCompressionOwner(CompressionOptions(video_profile="av1_savings")))
    )
    av1_manager._codec_capabilities = {
        0: {
            "hevc": VideoCodecCapabilities(codec="hevc", supported=True, num_max_bframes=5),
            "av1": VideoCodecCapabilities(
                codec="av1",
                supported=True,
                num_encoder_engines=2,
                num_max_bframes=7,
                support_lookahead=True,
                support_temporal_aq=True,
                support_10bit_encode=True,
            ),
        }
    }
    av1_kwargs = av1_manager._pynv_transcode_kwargs(av1_manager._effective_video_settings(0))

    assert hevc_kwargs["codec"] == "hevc"
    assert hevc_kwargs["constqp"] == 23
    assert av1_kwargs["codec"] == "av1"
    assert av1_kwargs["constqp"] == 26
    assert hevc_kwargs["bf"] == 3
    assert hevc_kwargs["gop"] == 120
    assert hevc_kwargs["idrperiod"] == 120
    assert hevc_kwargs["usedevicememory"] is True
    assert hevc_kwargs["usecpuinputbuffer"] is False
    assert hevc_kwargs["format"] == "NV12"
    assert hevc_kwargs["preset"] == "P4"
    assert hevc_kwargs["tuning_info"] == "high_quality"
    assert hevc_kwargs["aq"] == 1
    assert "temporalaq" not in hevc_kwargs
    assert hevc_kwargs["lookahead"] == 10
    assert av1_kwargs["bf"] == 5
    assert av1_kwargs["gop"] == 240
    assert av1_kwargs["idrperiod"] == 240
    assert av1_kwargs["preset"] == "P5"
    assert av1_kwargs["aq"] == 1
    assert av1_kwargs["temporalaq"] == 1
    assert av1_kwargs["lookahead"] == 16
    assert "gpu_id" not in hevc_kwargs


def test_video_profile_selection_gates_features_from_encoder_caps() -> None:
    owner = FakeCompressionOwner(CompressionOptions(video_profile="hevc_balanced"))
    compression_manager = CompressionManager(cast("Any", owner))
    compression_manager._codec_capabilities = {
        0: {
            "hevc": VideoCodecCapabilities(
                codec="hevc",
                supported=True,
                num_encoder_engines=2,
                num_max_bframes=2,
                support_lookahead=False,
                support_temporal_aq=True,
                support_10bit_encode=True,
            ),
            "av1": VideoCodecCapabilities(codec="av1", supported=False),
        }
    }

    settings = compression_manager._effective_video_settings(0)

    assert settings.profile == "hevc_balanced"
    assert settings.codec == "hevc"
    assert settings.cq == 23
    assert settings.bf == 2
    assert settings.gop == 120
    assert settings.idrperiod == 120
    assert settings.preset == "P4"
    assert settings.aq is True
    assert settings.temporalaq is False
    assert settings.lookahead == 0
    assert settings.support_10bit_encode is True


def test_av1_profile_downgrades_to_hevc_when_gpu_lacks_av1_encode() -> None:
    owner = FakeCompressionOwner(CompressionOptions(video_profile="av1_savings"))
    compression_manager = CompressionManager(cast("Any", owner))
    compression_manager._codec_capabilities = {
        0: {
            "hevc": VideoCodecCapabilities(codec="hevc", supported=True, num_max_bframes=4),
            "av1": VideoCodecCapabilities(codec="av1", supported=False),
        }
    }

    settings = compression_manager._effective_video_settings(0)

    assert settings.requested_profile == "av1_savings"
    assert settings.profile == "hevc_balanced"
    assert settings.downgraded_from == "av1_savings"
    assert settings.codec == "hevc"
    assert settings.codec_supported is True


def test_custom_profile_preserves_advanced_nvidia_settings() -> None:
    owner = FakeCompressionOwner(
        CompressionOptions(
            video_profile="custom",
            video_codec="av1",
            av1_cq=29,
            bf=7,
            gop=48,
            idrperiod=96,
            preset="P4",
            tuning_info="ultra_low_latency",
        )
    )
    compression_manager = CompressionManager(cast("Any", owner))
    compression_manager._codec_capabilities = {
        0: {
            "hevc": VideoCodecCapabilities(codec="hevc", supported=True),
            "av1": VideoCodecCapabilities(codec="av1", supported=True, num_max_bframes=3),
        }
    }

    settings = compression_manager._effective_video_settings(0)

    assert settings.profile == "custom"
    assert settings.codec == "av1"
    assert settings.cq == 29
    assert settings.bf == 7
    assert settings.gop == 48
    assert settings.idrperiod == 96
    assert settings.preset == "P4"
    assert settings.tuning_info == "ultra_low_latency"
    assert settings.aq is False
    assert settings.temporalaq is False
    assert settings.lookahead == 0


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
        assert f"decode:{output.name}:0:True" in calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pynv_worker_rejects_outputs_that_cannot_be_decoded(monkeypatch) -> None:
    root = _reset_test_dir()
    try:
        source = root / "input.mkv"
        output = root / "output.mkv"
        source.write_bytes(b"source")

        class FakeDecoder:
            def __init__(self, path: str, gpu_id: int = 0, use_device_memory: bool = False) -> None:
                self.path = Path(path)

            def get_stream_metadata(self) -> SimpleNamespace:
                raise RuntimeError("FFmpegDemuxer::FFmpegDemuxer: invalid output")

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
                self.output = Path(muxed_file_path)

            def transcode_with_mux(self) -> None:
                self.output.write_bytes(b"broken")

        fake_pynv = SimpleNamespace(Transcoder=FakeTranscoder, SimpleDecoder=FakeDecoder)
        monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_pynv)

        assert pynv_worker_main([str(source), str(output), "0", '{"codec": "hevc"}']) == 1
        assert not output.exists()
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


def test_repair_sample_entry_converts_empty_avc1_to_hvc1_for_hevc_bitstream() -> None:
    root = _reset_test_dir()
    try:
        video_path = root / "video.mp4"
        empty_avcC = _mp4_atom(b"avcC", b"")
        sample_entry_payload = (
            b"\x00" * 6
            + b"\x00\x01"
            + b"\x00" * 16
            + b"\x00\x80"
            + b"\x00\x80"
            + b"\x00\x48\x00\x00"
            + b"\x00\x48\x00\x00"
            + b"\x00\x00\x00\x00"
            + b"\x00\x01"
            + b"\x00" * 32
            + b"\x00\x18"
            + b"\xff\xff"
            + empty_avcC
        )
        avc1 = _mp4_atom(b"avc1", sample_entry_payload)
        stsd = _mp4_atom(b"stsd", b"\x00\x00\x00\x00" + (1).to_bytes(4, "big") + avc1)
        moov = _mp4_atom(
            b"moov",
            _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stsd)))),
        )

        vps_nal = bytes([(32 << 1), 0x01]) + b"\x00" * 8
        sps_nal = bytes([(33 << 1), 0x01]) + b"\x00" * 15
        pps_nal = bytes([(34 << 1), 0x01]) + b"\x00" * 4
        mdat_body = b""
        for nal in (vps_nal, sps_nal, pps_nal):
            mdat_body += len(nal).to_bytes(4, "big") + nal
        mdat = _mp4_atom(b"mdat", mdat_body)
        ftyp = _mp4_atom(b"ftyp", b"isom\x00\x00\x00\x00")
        video_path.write_bytes(ftyp + moov + mdat)

        _repair_sample_entry_for_target_codec(video_path, "hevc")

        rewritten = video_path.read_bytes()
        assert b"hvc1" in rewritten
        assert b"avc1" not in rewritten
        assert b"hvcC" in rewritten
        assert b"avcC" not in rewritten
        hvcc_offset = rewritten.find(b"hvcC")
        hvcc_size = int.from_bytes(rewritten[hvcc_offset - 4 : hvcc_offset], "big")
        assert hvcc_size > 16
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_repair_sample_entry_skips_non_hevc_targets_and_real_avcC() -> None:
    root = _reset_test_dir()
    try:
        ftyp = _mp4_atom(b"ftyp", b"isom\x00\x00\x00\x00")
        mdat = _mp4_atom(b"mdat", b"x" * 16)

        empty_avcC = _mp4_atom(b"avcC", b"")
        avc1_empty = _mp4_atom(b"avc1", b"\x00" * 78 + empty_avcC)
        stsd_empty = _mp4_atom(b"stsd", b"\x00\x00\x00\x00" + (1).to_bytes(4, "big") + avc1_empty)
        moov_empty = _mp4_atom(
            b"moov",
            _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stsd_empty)))),
        )
        wrong_target = root / "wrong_target.mp4"
        original_wrong = ftyp + moov_empty + mdat
        wrong_target.write_bytes(original_wrong)
        _repair_sample_entry_for_target_codec(wrong_target, "av1")
        assert wrong_target.read_bytes() == original_wrong

        real_avcC = _mp4_atom(b"avcC", b"\x01" * 64)
        avc1_real = _mp4_atom(b"avc1", b"\x00" * 78 + real_avcC)
        stsd_real = _mp4_atom(b"stsd", b"\x00\x00\x00\x00" + (1).to_bytes(4, "big") + avc1_real)
        moov_real = _mp4_atom(
            b"moov",
            _mp4_atom(b"trak", _mp4_atom(b"mdia", _mp4_atom(b"minf", _mp4_atom(b"stbl", stsd_real)))),
        )
        real_path = root / "real_avcc.mp4"
        original_real = ftyp + moov_real + mdat
        real_path.write_bytes(original_real)
        _repair_sample_entry_for_target_codec(real_path, "hevc")
        assert real_path.read_bytes() == original_real
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_extract_inline_hevc_param_sets_finds_vps_sps_pps_in_length_prefixed_mdat() -> None:
    vps_nal = bytes([(32 << 1), 0x01]) + b"\xaa" * 6
    sps_nal = bytes([(33 << 1), 0x01]) + b"\xbb" * 13
    pps_nal = bytes([(34 << 1), 0x01]) + b"\xcc" * 4
    body = b""
    for nal in (vps_nal, sps_nal, pps_nal):
        body += len(nal).to_bytes(4, "big") + nal
    vps, sps, pps = _extract_inline_hevc_param_sets(body)
    assert vps == vps_nal
    assert sps == sps_nal
    assert pps == pps_nal


def test_build_hvcc_body_embeds_profile_tier_level_and_param_set_arrays() -> None:
    sps = bytes([(33 << 1), 0x01]) + b"\x60" + b"\xaa" * 12 + b"\x00" * 4
    vps = bytes([(32 << 1), 0x01]) + b"VPS"
    pps = bytes([(34 << 1), 0x01]) + b"PPS"
    body = _build_hvcc_body(vps, sps, pps)
    assert body[0] == 1
    assert body[1:13] == sps[3:15]
    assert body[22] == 3
    assert b"VPS" in body
    assert b"PPS" in body


def test_compressed_marker_renames_file_and_updates_media_item() -> None:
    root = _reset_test_dir()
    try:
        video_path = root / "my video.mp4"
        video_path.write_bytes(b"fake")
        options = CompressionOptions()
        owner = FakeCompressionOwner(options)
        manager = CompressionManager(cast("Any", owner))

        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=video_path,
                filename="my video.mp4",
                download_filename="my video.mp4",
                db_path="original/my video.mp4",
                is_segment=False,
                url="https://example.com/media",
                filesize=4,
            ),
        )

        new_path = asyncio.run(manager._apply_compressed_marker(media_item, video_path))

        assert new_path.name == "[COMPRESSED] my video.mp4"
        assert new_path.exists()
        assert not video_path.exists()
        assert media_item.complete_file == new_path
        assert media_item.download_filename == new_path.name
        assert media_item.filename == "my video.mp4"
        assert media_item.db_path == "original/my video.mp4"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_compressed_marker_is_idempotent_when_prefix_already_present() -> None:
    root = _reset_test_dir()
    try:
        already_marked = root / "[COMPRESSED] clip.mp4"
        already_marked.write_bytes(b"data")
        manager = CompressionManager(cast("Any", FakeCompressionOwner()))
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=already_marked,
                filename=already_marked.name,
                download_filename=already_marked.name,
                db_path="x",
                is_segment=False,
                url="https://example.com/media",
                filesize=4,
            ),
        )

        result_path = asyncio.run(manager._apply_compressed_marker(media_item, already_marked))
        assert result_path == already_marked
        assert already_marked.exists()
        assert media_item.complete_file == already_marked
        assert media_item.download_filename == already_marked.name
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_video_finalize_promotes_temp_to_compressed_marker_and_removes_original() -> None:
    root = _reset_test_dir()
    try:
        source = root / "clip.mp4"
        temp_output = root / "clip.compressed.mp4"
        source.write_bytes(b"x" * 100)
        temp_output.write_bytes(b"x" * 50)
        compression_manager = CompressionManager(cast("Any", FakeCompressionOwner(CompressionOptions())))

        async def validate_video(path: Path) -> None:
            assert path == temp_output

        compression_manager._validate_video = validate_video
        result = asyncio.run(
            compression_manager._finalize_output(
                source,
                temp_output,
                "video",
                media_type="video",
                backend="pynv",
                path=source,
            )
        )

        promoted = root / "[COMPRESSED] clip.mp4"
        assert result.status == "compressed"
        assert result.path == promoted
        assert promoted.read_bytes() == b"x" * 50
        assert not source.exists()
        assert not temp_output.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_compressed_marker_picks_counter_suffix_on_collision() -> None:
    root = _reset_test_dir()
    try:
        source = root / "clip.mp4"
        source.write_bytes(b"new")
        collision = root / "[COMPRESSED] clip.mp4"
        collision.write_bytes(b"old")
        manager = CompressionManager(cast("Any", FakeCompressionOwner()))
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=source,
                filename=source.name,
                download_filename=source.name,
                db_path="x",
                is_segment=False,
                url="https://example.com/media",
                filesize=3,
            ),
        )

        new_path = asyncio.run(manager._apply_compressed_marker(media_item, source))
        assert new_path.name == "[COMPRESSED] clip (1).mp4"
        assert new_path.exists()
        assert collision.exists()
        assert not source.exists()
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


def test_video_runtime_startup_uses_encoder_engine_count_for_worker_slots() -> None:
    owner = FakeCompressionOwner(CompressionOptions(gpu_ids=[0, 1], video_workers_per_gpu=4))
    compression_manager = CompressionManager(cast("Any", owner))
    started: list[tuple[int, int]] = []
    closed: list[int] = []

    class FakeWorker:
        def __init__(self, gpu_id: int, max_jobs: int) -> None:
            self.gpu_id = gpu_id
            self.max_jobs = max_jobs
            self.alive = False

        async def start(self) -> None:
            self.alive = True
            started.append((self.gpu_id, self.max_jobs))

        async def close(self) -> None:
            self.alive = False
            closed.append(self.gpu_id)

    def fake_probe(pynv: Any) -> dict[int, dict[str, VideoCodecCapabilities]]:
        return {
            0: {
                "hevc": VideoCodecCapabilities(codec="hevc", supported=True, num_encoder_engines=2),
                "av1": VideoCodecCapabilities(codec="av1", supported=True, num_encoder_engines=2),
            },
            1: {
                "hevc": VideoCodecCapabilities(codec="hevc", supported=True, num_encoder_engines=1),
                "av1": VideoCodecCapabilities(codec="av1", supported=False, num_encoder_engines=1),
            },
        }

    compression_manager._import_pynv = lambda: object()
    compression_manager._probe_encoder_capabilities = fake_probe
    compression_manager._create_persistent_worker = lambda gpu_id, max_jobs: cast(
        "Any",
        FakeWorker(gpu_id, max_jobs),
    )

    async def run() -> None:
        compression_manager.startup()
        await asyncio.wait_for(cast("Any", compression_manager._video_runtime_task), timeout=1)
        assert compression_manager._gpu_dispatch_order == [0, 0, 1]
        assert compression_manager._worker_slots_for_gpu(0) == 2
        assert compression_manager._worker_slots_for_gpu(1) == 1
        await compression_manager.close()

    asyncio.run(run())

    assert started == [(0, 2), (1, 1)]
    assert sorted(closed) == [0, 1]


def test_persistent_worker_restart_retries_transcode_once() -> None:
    root = _reset_test_dir()
    try:
        owner = FakeCompressionOwner()
        compression_manager = CompressionManager(cast("Any", owner))
        source = root / "input.mp4"
        output = root / "output.mp4"
        source.write_bytes(b"source")
        calls: list[str] = []

        class FakeWorker:
            def __init__(self, name: str, *, fail: bool = False) -> None:
                self.name = name
                self.fail = fail
                self.max_jobs = 1
                self.alive = True

            async def start(self) -> None:
                self.alive = True

            async def close(self) -> None:
                self.alive = False
                calls.append(f"close:{self.name}")

            async def transcode(self, source: Path, temp_output: Path, config: dict[str, Any]) -> None:
                calls.append(f"transcode:{self.name}:{config['codec']}:{config['constqp']}")
                if self.fail:
                    self.fail = False
                    self.alive = False
                    raise _PersistentWorkerDied("worker crashed")
                await asyncio.to_thread(temp_output.write_bytes, b"compressed")

        replacement = FakeWorker("replacement")
        compression_manager._pynv_workers = {0: FakeWorker("primary", fail=True)}
        compression_manager._ensure_video_runtime = lambda: asyncio.sleep(0, result=object())
        compression_manager._create_persistent_worker = lambda gpu_id, max_jobs: cast("Any", replacement)

        settings = EffectiveVideoSettings(
            profile="hevc_balanced",
            requested_profile="hevc_balanced",
            codec="hevc",
            cq=23,
            bf=3,
            gop=120,
            idrperiod=120,
            preset="P5",
            tuning_info="high_quality",
        )

        asyncio.run(compression_manager._transcode_with_pynv_subprocess(source, output, 0, settings))

        assert output.read_bytes() == b"compressed"
        assert calls == [
            "transcode:primary:hevc:23",
            "close:primary",
            "transcode:replacement:hevc:23",
        ]
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


def test_compression_report_write_failure_does_not_fail_compression() -> None:
    root = _reset_test_dir()
    try:
        video = root / "video.mp4"
        video.write_bytes(b"x" * 100)

        class FailingLogManager:
            async def write_compression_report(self, **kwargs: Any) -> None:
                raise PermissionError("locked")

        owner = FakeCompressionOwner()
        owner.log_manager = FailingLogManager()
        compression_manager = CompressionManager(cast("Any", owner))

        asyncio.run(
            compression_manager._record_result(
                _media_item(video),
                CompressionResult(
                    status="compressed",
                    media_type="video",
                    backend="pynv",
                    path=video,
                    original_size=100,
                    final_size=50,
                ),
            )
        )

        assert owner.progress_manager.results == [("compressed", 50)]
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
            database=SimpleNamespace(),
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
            "enqueue",
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_collision_is_treated_as_previously_downloaded() -> None:
    root = _reset_test_dir()
    try:
        partial_file = root / "download.jpg.part"
        complete_file = root / "download.jpg"
        partial_file.write_text("downloaded", encoding="utf8")
        complete_file.write_text("downloaded", encoding="utf8")
        calls: list[str] = []

        class FakeDownloadProgress:
            def __init__(self) -> None:
                self.args: list[bool] = []

            def add_previously_completed(self, increase_total: bool = True) -> None:
                self.args.append(increase_total)

        class FakeClientManager:
            def __init__(self, manager: Any) -> None:
                self.manager = manager

            @contextlib.contextmanager
            def request_context(self, domain: str):
                yield

            async def check_file_duration(self, media_item: MediaItem) -> bool:
                calls.append("duration")
                return True

        class FakeCompressionManager:
            async def enqueue_existing_file_if_needed(
                self,
                domain: str,
                media_item: MediaItem,
                process_completed: Any,
                handle_completion: Any,
                *,
                downloaded: bool = False,
                allow_images: bool = True,
            ) -> bool:
                calls.append("enqueue")
                assert domain == "example.com"
                assert downloaded is False
                assert media_item.complete_file == complete_file
                return True

        running = asyncio.Event()
        running.set()
        progress = SimpleNamespace(download_progress=FakeDownloadProgress())
        manager = SimpleNamespace(
            config=ConfigSettings(),
            config_manager=SimpleNamespace(settings_data=ConfigSettings()),
            states=SimpleNamespace(RUNNING=running),
            database=SimpleNamespace(),
            compression_manager=FakeCompressionManager(),
            progress_manager=progress,
        )
        client_manager = FakeClientManager(manager)
        client = DownloadClient(cast("Any", manager), cast("Any", client_manager))

        async def fake_download(domain: str, media_item: MediaItem) -> bool:
            calls.append("download")
            return True

        async def fake_promote_partial_to_complete(media_item: MediaItem) -> None:
            raise FileExistsError(
                183,
                "Cannot create a file when that file already exists",
                str(media_item.complete_file),
            )

        async def fake_process_completed(media_item: MediaItem, domain: str) -> None:
            calls.append("process_completed")

        async def fake_handle_completion(media_item: MediaItem, downloaded: bool = False) -> None:
            calls.append(f"handle_completion:{downloaded}")

        client._download = fake_download
        client._promote_partial_to_complete = fake_promote_partial_to_complete
        client.process_completed = fake_process_completed
        client.handle_media_item_completion = fake_handle_completion
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=complete_file,
                is_segment=False,
                partial_file=partial_file,
                url="https://example.com/download.jpg",
                referer="https://example.com/post",
                filesize=len("downloaded"),
            ),
        )

        assert asyncio.run(client.download_file("example.com", media_item)) is False
        assert calls == [
            "download",
            "enqueue",
        ]
        assert progress.download_progress.args == [False]
        assert complete_file.read_text(encoding="utf8") == "downloaded"
        assert not partial_file.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_collision_with_different_size_gets_unique_filename() -> None:
    root = _reset_test_dir()
    try:
        partial_file = root / "download.jpg.part"
        complete_file = root / "download.jpg"
        partial_file.write_text("fresh-download", encoding="utf8")
        complete_file.write_text("existing-file", encoding="utf8")
        calls: list[str] = []
        db_updates: list[str] = []

        class FakeDatabase:
            async def check_download_filename_exists(self, filename: str) -> bool:
                return False

            async def update_media_item(self, media_item: Any) -> None:
                db_updates.append(media_item.download_filename)

        class FakeClientManager:
            def __init__(self, manager: Any) -> None:
                self.manager = manager

            @contextlib.contextmanager
            def request_context(self, domain: str):
                yield

            async def check_file_duration(self, media_item: MediaItem) -> bool:
                calls.append("duration")
                assert media_item.complete_file.name == "download (1).jpg"
                assert media_item.complete_file.read_text(encoding="utf8") == "fresh-download"
                assert not partial_file.exists()
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
                calls.append("enqueue")
                assert domain == "example.com"
                assert downloaded is True
                assert media_item.complete_file.name == "download (1).jpg"

        running = asyncio.Event()
        running.set()
        manager = SimpleNamespace(
            config=ConfigSettings(),
            config_manager=SimpleNamespace(settings_data=ConfigSettings()),
            states=SimpleNamespace(RUNNING=running),
            database=FakeDatabase(),
            compression_manager=FakeCompressionManager(),
        )
        client_manager = FakeClientManager(manager)
        client = DownloadClient(cast("Any", manager), cast("Any", client_manager))

        async def fake_download(domain: str, media_item: MediaItem) -> bool:
            calls.append("download")
            return True

        async def fake_promote_partial_to_complete(media_item: MediaItem) -> None:
            raise FileExistsError(
                183,
                "Cannot create a file when that file already exists",
                str(media_item.complete_file),
            )

        client._download = fake_download
        client._promote_partial_to_complete = fake_promote_partial_to_complete
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=complete_file,
                download_folder=root,
                download_filename="download.jpg",
                filename="download.jpg",
                is_segment=False,
                partial_file=partial_file,
                referer="https://example.com/post",
                url="https://example.com/download.jpg",
                filesize=len("fresh-download"),
            ),
        )

        assert asyncio.run(client.download_file("example.com", media_item)) is True
        assert calls == [
            "download",
            "duration",
            "enqueue",
        ]
        assert db_updates == ["download (1).jpg"]
        assert media_item.complete_file.name == "download (1).jpg"
        assert (root / "download (1).jpg").read_text(encoding="utf8") == "fresh-download"
        assert complete_file.read_text(encoding="utf8") == "existing-file"
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeCompressionProgress:
    def __init__(self) -> None:
        self.pending_count = 0
        self.total = 0
        self.current: dict[int, Path] = {}
        self.results: list[str] = []
        self.started: list[tuple[int, Path, int]] = []
        self.updates: list[tuple[int, int, int | None]] = []
        self.finished: list[int] = []

    def set_pending_count(self, count: int) -> None:
        self.pending_count = count

    def increment_total(self, delta: int = 1) -> None:
        self.total = max(0, self.total + delta)

    def set_current(self, worker_id: int, path: Path | None) -> None:
        if path is None:
            self.current.pop(worker_id, None)
        else:
            self.current[worker_id] = path

    def add_result(self, status: str) -> None:
        self.results.append(status)

    def start_task(self, worker_id: int, path: Path, total: int) -> None:
        self.started.append((worker_id, path, total))

    def update_task(self, worker_id: int, completed: int, total: int | None = None) -> None:
        self.updates.append((worker_id, completed, total))

    def finish_task(self, worker_id: int) -> None:
        self.finished.append(worker_id)


class FakePathManager:
    def __init__(self, pending_file: Path) -> None:
        self.compression_pending_file = pending_file


def _owner_with_pending_file(pending_file: Path, options: CompressionOptions | None = None) -> FakeCompressionOwner:
    owner = FakeCompressionOwner(options)
    owner.path_manager = FakePathManager(pending_file)
    owner.progress_manager.compression_progress = FakeCompressionProgress()
    original_add = owner.progress_manager.add_compression_result

    def add_compression_result(status: str, bytes_saved: int = 0) -> None:
        original_add(status, bytes_saved)
        owner.progress_manager.compression_progress.add_result(status)

    owner.progress_manager.add_compression_result = add_compression_result
    return owner


def test_pending_compression_file_round_trip_and_pending_count() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        owner = _owner_with_pending_file(pending_file)
        compression_manager = CompressionManager(cast("Any", owner))

        first = root / "video1.mp4"
        second = root / "video2.mp4"

        async def run() -> None:
            await compression_manager._track_pending(first)
            await compression_manager._track_pending(second)
            await compression_manager._track_pending(first)

        asyncio.run(run())

        entries = compression_manager._read_pending_file()
        assert sorted(entries) == sorted([str(first), str(second)])
        assert owner.progress_manager.compression_progress.pending_count == 2
        assert json.loads(pending_file.read_text(encoding="utf-8")) == entries

        async def untrack() -> None:
            await compression_manager._untrack_pending(first)

        asyncio.run(untrack())

        assert compression_manager._read_pending_file() == [str(second)]
        assert owner.progress_manager.compression_progress.pending_count == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pending_compression_file_allows_utf8_bom() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        existing = root / "video.mp4"
        pending_file.write_text(f"\ufeff{json.dumps([str(existing)])}", encoding="utf-8")

        owner = _owner_with_pending_file(pending_file)
        compression_manager = CompressionManager(cast("Any", owner))

        assert compression_manager._read_pending_file() == [str(existing)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_skips_missing_files_and_requeues_existing_on_startup() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        existing = root / "existing.mp4"
        missing = root / "does_not_exist.mp4"
        existing.write_bytes(b"data")
        pending_file.write_text(
            json.dumps([str(existing), str(missing)]),
            encoding="utf-8",
        )

        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))

        processed: list[Path] = []

        async def fake_compress(media_item: MediaItem) -> None:
            processed.append(Path(media_item.complete_file))

        compression_manager.compress_media_item = fake_compress

        async def run() -> None:
            compression_manager.startup()
            await compression_manager.join()
            await compression_manager.close()

        asyncio.run(run())

        assert processed == [existing]
        assert compression_manager._read_pending_file() == []
        assert owner.progress_manager.compression_progress.pending_count == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_join_waits_for_startup_requeue_before_returning() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        existing = root / "existing.mp4"
        existing.write_bytes(b"data")
        pending_file.write_text(json.dumps([str(existing)]), encoding="utf-8")

        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))
        processed: list[Path] = []
        requeue_started = asyncio.Event()
        release_requeue = asyncio.Event()
        original_requeue = compression_manager._requeue_standalone_paths

        async def delayed_requeue(paths: list[str]) -> None:
            requeue_started.set()
            await release_requeue.wait()
            await original_requeue(paths)

        async def fake_compress(media_item: MediaItem) -> None:
            processed.append(Path(media_item.complete_file))

        compression_manager._requeue_standalone_paths = delayed_requeue
        compression_manager.compress_media_item = fake_compress

        async def run() -> None:
            compression_manager.startup()
            join_task = asyncio.create_task(compression_manager.join())
            await asyncio.wait_for(requeue_started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert not join_task.done()
            release_requeue.set()
            await asyncio.wait_for(join_task, timeout=1)
            await compression_manager.close()

        asyncio.run(run())

        assert processed == [existing]
        assert compression_manager._read_pending_file() == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_queue_worker_notifies_current_file_and_pending_count() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))

        media_path = root / "media.jpg"
        media_path.write_bytes(b"data")
        progress = owner.progress_manager.compression_progress
        snapshots: list[tuple[int, dict[int, Path]]] = []

        async def fake_compress(media_item: MediaItem) -> None:
            snapshots.append((progress.pending_count, dict(progress.current)))

        async def noop_process(media_item: MediaItem, domain: str) -> None:
            return None

        async def noop_handle(media_item: MediaItem, downloaded: bool = True) -> None:
            return None

        compression_manager.compress_media_item = fake_compress

        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=media_path,
                filename="media.jpg",
                is_segment=False,
            ),
        )

        async def run() -> None:
            await compression_manager.enqueue_completed_download(
                "example.com",
                media_item,
                noop_process,
                noop_handle,
            )
            await compression_manager.join()
            await compression_manager.close()

        asyncio.run(run())

        assert len(snapshots) == 1
        pending_during, current_during = snapshots[0]
        assert pending_during == 1
        assert current_during == {1: media_path}
        assert progress.pending_count == 0
        assert progress.current == {}
        assert compression_manager._read_pending_file() == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_queue_worker_reports_live_compression_progress() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))
        media_path = root / "media.mp4"
        media_path.write_bytes(b"x" * 100)
        progress = owner.progress_manager.compression_progress

        async def fake_compress(media_item: MediaItem) -> None:
            temp_output = media_path.with_name(f"{media_path.stem}.compressed{media_path.suffix}")
            await asyncio.to_thread(temp_output.write_bytes, b"x" * 25)
            await asyncio.sleep(0.6)
            await asyncio.to_thread(temp_output.write_bytes, b"x" * 50)
            await asyncio.sleep(0.6)

        async def noop_process(media_item: MediaItem, domain: str) -> None:
            return None

        async def noop_handle(media_item: MediaItem, downloaded: bool = True) -> None:
            return None

        compression_manager.compress_media_item = fake_compress
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=media_path,
                filename="media.mp4",
                is_segment=False,
            ),
        )

        async def run() -> None:
            await compression_manager.enqueue_completed_download(
                "example.com",
                media_item,
                noop_process,
                noop_handle,
            )
            await compression_manager.join()
            await compression_manager.close()

        asyncio.run(run())

        assert progress.started == [(1, media_path, 100)]
        assert any(update[0] == 1 and update[1] >= 25 and update[2] == 100 for update in progress.updates)
        assert 1 in progress.finished
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_join_waits_for_in_flight_enqueue_before_returning() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))
        media_path = root / "media.mp4"
        media_path.write_bytes(b"data")
        enqueue_started = asyncio.Event()
        release_enqueue = asyncio.Event()
        original_track_pending = compression_manager._track_pending
        processed: list[Path] = []

        async def delayed_track_pending(path: Path) -> None:
            enqueue_started.set()
            await release_enqueue.wait()
            await original_track_pending(path)

        async def fake_compress(media_item: MediaItem) -> None:
            processed.append(Path(media_item.complete_file))

        async def noop_process(media_item: MediaItem, domain: str) -> None:
            return None

        async def noop_handle(media_item: MediaItem, downloaded: bool = True) -> None:
            return None

        compression_manager._track_pending = delayed_track_pending
        compression_manager.compress_media_item = fake_compress
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=media_path,
                filename="media.mp4",
                is_segment=False,
            ),
        )

        async def run() -> None:
            enqueue_task = asyncio.create_task(
                compression_manager.enqueue_completed_download(
                    "example.com",
                    media_item,
                    noop_process,
                    noop_handle,
                )
            )
            await asyncio.wait_for(enqueue_started.wait(), timeout=1)
            join_task = asyncio.create_task(compression_manager.join())
            await asyncio.sleep(0)
            assert not join_task.done()
            release_enqueue.set()
            await asyncio.wait_for(enqueue_task, timeout=1)
            await asyncio.wait_for(join_task, timeout=1)
            await compression_manager.close()

        asyncio.run(run())

        assert processed == [media_path]
        assert compression_manager._read_pending_file() == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_worker_cancellation_waits_for_active_item_before_join_returns() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        owner = _owner_with_pending_file(pending_file, CompressionOptions(gpu_ids=[0], video_workers_per_gpu=1))
        compression_manager = CompressionManager(cast("Any", owner))
        media_path = root / "media.mp4"
        media_path.write_bytes(b"data")
        started = asyncio.Event()
        release = asyncio.Event()
        completed: list[str] = []

        async def fake_compress(media_item: MediaItem) -> None:
            started.set()
            await release.wait()
            completed.append("compress")

        async def noop_process(media_item: MediaItem, domain: str) -> None:
            completed.append("process")

        async def noop_handle(media_item: MediaItem, downloaded: bool = True) -> None:
            completed.append("handle")

        compression_manager.compress_media_item = fake_compress
        media_item = cast(
            "MediaItem",
            SimpleNamespace(
                complete_file=media_path,
                filename="media.mp4",
                is_segment=False,
            ),
        )

        async def run() -> None:
            await compression_manager.enqueue_completed_download(
                "example.com",
                media_item,
                noop_process,
                noop_handle,
            )
            worker = compression_manager._queue_tasks[0]
            await asyncio.wait_for(started.wait(), timeout=1)
            join_task = asyncio.create_task(compression_manager.join())
            worker.cancel()
            await asyncio.sleep(0)
            assert not join_task.done()
            assert compression_manager._read_pending_file() == [str(media_path)]
            release.set()
            await asyncio.wait_for(join_task, timeout=1)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=1)
            await compression_manager.close()

        asyncio.run(run())

        assert completed == ["compress", "process", "handle"]
        assert compression_manager._read_pending_file() == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resume_handles_empty_pending_file_without_errors() -> None:
    root = _reset_test_dir()
    try:
        pending_file = root / "compression_pending.json"
        pending_file.write_text("[]", encoding="utf-8")
        owner = _owner_with_pending_file(pending_file)
        compression_manager = CompressionManager(cast("Any", owner))

        async def run() -> None:
            compression_manager.startup()
            for _ in range(5):
                await asyncio.sleep(0)
            await compression_manager.close()

        asyncio.run(run())

        assert compression_manager._read_pending_file() == []
        assert owner.progress_manager.compression_progress.pending_count == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
