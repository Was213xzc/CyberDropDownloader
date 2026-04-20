from __future__ import annotations

import asyncio
import importlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cyberdrop_dl.constants import FILE_FORMATS
from cyberdrop_dl.utils import ffmpeg
from cyberdrop_dl.utils.logger import log

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from cyberdrop_dl.config.config_model import CompressionOptions
    from cyberdrop_dl.data_structures.url_objects import MediaItem
    from cyberdrop_dl.managers.manager import Manager


CompressionStatus = Literal["compressed", "skipped", "failed"]
MediaType = Literal["video", "image"]


@dataclass(slots=True, kw_only=True)
class CompressionResult:
    status: CompressionStatus
    media_type: MediaType
    backend: str
    path: Path
    original_size: int | None = None
    final_size: int | None = None
    gpu_id: int | None = None
    codec: str | None = None
    cq: int | None = None
    bf: int | None = None
    error: str = ""

    @property
    def bytes_saved(self) -> int:
        if self.status != "compressed" or self.original_size is None or self.final_size is None:
            return 0
        return max(self.original_size - self.final_size, 0)

    @property
    def savings_percent(self) -> float:
        if not self.original_size:
            return 0.0
        return round((self.bytes_saved / self.original_size) * 100, 2)


class CompressionManager:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        self._gpu_index = 0
        self._pynv_unavailable_logged = False
        self._video_semaphores: defaultdict[int, asyncio.BoundedSemaphore] = defaultdict(self._make_video_semaphore)

    @property
    def options(self) -> CompressionOptions:
        return self.manager.config.compression_options

    async def compress_media_item(self, media_item: MediaItem) -> CompressionResult | None:
        options = self.options
        if not options.enabled or media_item.is_segment:
            return None

        source = await asyncio.to_thread(media_item.complete_file.resolve)
        if not await asyncio.to_thread(source.is_file):
            return None

        ext = source.suffix.lower()
        if options.compress_videos and ext in FILE_FORMATS["Videos"]:
            result = await self._compress_video(media_item, source)
        elif options.compress_images and ext in FILE_FORMATS["Images"]:
            result = await self._compress_image(source)
        else:
            return None

        if result.status == "compressed" and result.final_size is not None:
            media_item.filesize = result.final_size
        await self._record_result(media_item, result)
        return result

    async def _compress_video(self, media_item: MediaItem, source: Path) -> CompressionResult:
        options = self.options
        codec = options.video_codec
        cq = self._codec_cq(codec)
        result_kwargs = {
            "media_type": "video",
            "backend": "pynv",
            "path": source,
            "codec": codec,
            "cq": cq,
            "bf": options.bf,
        }

        pynv = self._import_pynv()
        if pynv is None:
            return CompressionResult(status="skipped", error="PyNvVideoCodec is not installed", **result_kwargs)

        try:
            probe = await ffmpeg.probe(source)
            video = probe.video
            if not probe or video is None:
                return CompressionResult(status="skipped", error="No video stream found", **result_kwargs)
            if self._already_target_codec(video.codec, codec):
                return CompressionResult(status="skipped", error=f"Already encoded as {codec}", **result_kwargs)
            duration = float(video.duration or probe.format.duration or 0)
            if duration <= 0:
                return CompressionResult(status="skipped", error="Could not determine video duration", **result_kwargs)

            gpu_id = self._next_gpu_id()
            result_kwargs["gpu_id"] = gpu_id
            temp_output = source.with_name(f"{source.stem}.compressed{source.suffix}")
            await self._delete_temp(temp_output)
            try:
                async with self._video_slot(gpu_id):
                    await asyncio.to_thread(
                        self._transcode_with_pynv,
                        pynv,
                        source,
                        temp_output,
                        duration,
                        gpu_id,
                        codec,
                        cq,
                    )
            except Exception as e:
                await self._delete_temp(temp_output)
                return CompressionResult(status="skipped", error=str(e), **result_kwargs)
            temp_output = await asyncio.to_thread(self._resolve_pynv_output, source, temp_output)
            return await self._finalize_output(source, temp_output, "video", **result_kwargs)
        except Exception as e:
            await self._delete_temp(source.with_name(f"{source.stem}.compressed{source.suffix}"))
            log(f"Compression failed for {source}: {e}", 40, exc_info=True)
            return CompressionResult(status="failed", error=str(e), **result_kwargs)

    async def _compress_image(self, source: Path) -> CompressionResult:
        result_kwargs = {"media_type": "image", "backend": "pillow", "path": source}
        temp_output = source.with_name(f"{source.stem}.compressed{source.suffix}")
        try:
            await self._delete_temp(temp_output)
            await asyncio.to_thread(self._save_image_optimized, source, temp_output)
            return await self._finalize_output(source, temp_output, "image", **result_kwargs)
        except Exception as e:
            await self._delete_temp(temp_output)
            return CompressionResult(status="skipped", error=str(e), **result_kwargs)

    def _save_image_optimized(self, source: Path, temp_output: Path) -> None:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(source) as image:
                if getattr(image, "is_animated", False):
                    raise ValueError("Animated images are skipped")

                ext = source.suffix.lower()
                save_kwargs = self._image_save_kwargs(image.info, ext)
                output_image = image
                if ext in {".jpg", ".jpeg", ".jpe", ".jfif", ".jif"} and image.mode not in {"RGB", "L"}:
                    output_image = image.convert("RGB")
                output_image.save(temp_output, **save_kwargs)
        except UnidentifiedImageError as e:
            raise ValueError("Unsupported or corrupted image") from e

    def _image_save_kwargs(self, info: dict, ext: str) -> dict:
        options = self.options
        metadata = {
            key: value
            for key in ("exif", "icc_profile")
            if (value := info.get(key)) is not None
        }
        if ext in {".jpg", ".jpeg", ".jpe", ".jfif", ".jif"}:
            return metadata | {
                "quality": options.jpeg_quality,
                "optimize": True,
                "progressive": True,
            }
        if ext == ".png":
            return metadata | {"optimize": options.png_optimize}
        if ext == ".webp":
            return metadata | {
                "quality": options.webp_quality,
                "method": 6,
            }
        raise ValueError(f"Image format '{ext}' is not supported for compression")

    async def _finalize_output(
        self,
        source: Path,
        temp_output: Path,
        output_type: MediaType,
        **result_kwargs,
    ) -> CompressionResult:
        original_size = await asyncio.to_thread(lambda: source.stat().st_size)
        final_size = await asyncio.to_thread(lambda: temp_output.stat().st_size)
        result_kwargs |= {"original_size": original_size, "final_size": final_size}

        try:
            if output_type == "video":
                await self._validate_video(temp_output)
            else:
                await asyncio.to_thread(self._validate_image, temp_output)
        except Exception as e:
            await self._delete_temp(temp_output)
            return CompressionResult(status="skipped", error=f"Compressed output failed validation: {e}", **result_kwargs)

        if not self._meets_savings_threshold(original_size, final_size):
            await self._delete_temp(temp_output)
            return CompressionResult(status="skipped", error="Compressed output was not small enough", **result_kwargs)

        await asyncio.to_thread(temp_output.replace, source)
        return CompressionResult(status="compressed", **result_kwargs)

    async def _validate_video(self, path: Path) -> None:
        probe = await ffmpeg.probe(path)
        if not probe or probe.video is None:
            raise ValueError("Compressed video failed validation")

    def _validate_image(self, path: Path) -> None:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()

    def _transcode_with_pynv(
        self,
        pynv: ModuleType,
        source: Path,
        temp_output: Path,
        duration: float,
        gpu_id: int,
        codec: str,
        cq: int,
    ) -> None:
        transcode_kwargs = self._pynv_transcode_kwargs(codec, cq)
        factory = getattr(pynv, "Transcoder", None) or getattr(pynv, "CreateTranscoder", None)
        if factory is None:
            raise RuntimeError("PyNvVideoCodec Transcoder API is unavailable")

        try:
            transcoder = factory(
                enc_file_path=str(source),
                muxed_file_path=str(temp_output),
                gpu_id=gpu_id,
                **transcode_kwargs,
            )
        except TypeError:
            transcoder = factory(str(source), str(temp_output), gpu_id, **transcode_kwargs)

        if hasattr(transcoder, "transcode"):
            transcoder.transcode()
        elif hasattr(transcoder, "segmented_transcode"):
            transcoder.segmented_transcode(0.0, duration)
        else:
            raise RuntimeError("PyNvVideoCodec transcoder does not expose a transcode method")

    def _pynv_transcode_kwargs(self, codec: str, cq: int) -> dict:
        options = self.options
        return {
            "codec": codec,
            "format": "NV12",
            "usedevicememory": True,
            "usecpuinputbuffer": False,
            "rc": "constqp",
            "constqp": cq,
            "bf": options.bf,
            "preset": options.preset,
            "tuning_info": options.tuning_info,
        }

    def _resolve_pynv_output(self, source: Path, temp_output: Path) -> Path:
        if temp_output.is_file():
            return temp_output

        candidates = [
            path
            for path in temp_output.parent.glob(f"{temp_output.stem}*{temp_output.suffix}")
            if path.is_file() and path != source
        ]
        if not candidates:
            return temp_output
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _codec_cq(self, codec: str) -> int:
        return self.options.av1_cq if codec == "av1" else self.options.hevc_cq

    def _already_target_codec(self, input_codec: str, target_codec: str) -> bool:
        normalized = input_codec.casefold()
        if target_codec == "hevc":
            return normalized in {"hevc", "h265"}
        return normalized == target_codec

    def _meets_savings_threshold(self, original_size: int, final_size: int) -> bool:
        threshold = 1 - (self.options.min_savings_percent / 100)
        return final_size < original_size * threshold

    async def _record_result(self, media_item: MediaItem, result: CompressionResult) -> None:
        if hasattr(self.manager, "progress_manager"):
            self.manager.progress_manager.add_compression_result(result.status, result.bytes_saved)
        if hasattr(self.manager, "log_manager"):
            await self.manager.log_manager.write_compression_report(
                url=media_item.url,
                path=result.path,
                media_type=result.media_type,
                status=result.status,
                backend=result.backend,
                gpu_id=result.gpu_id,
                codec=result.codec,
                cq=result.cq,
                bf=result.bf,
                original_size=result.original_size,
                final_size=result.final_size,
                savings_percent=result.savings_percent,
                error=result.error,
            )

    def _import_pynv(self) -> ModuleType | None:
        try:
            return importlib.import_module("PyNvVideoCodec")
        except ImportError:
            if not self._pynv_unavailable_logged:
                log("PyNvVideoCodec is not installed; video compression will be skipped", 30)
                self._pynv_unavailable_logged = True
            return None

    def _next_gpu_id(self) -> int:
        gpu_ids = self.options.gpu_ids or [0]
        gpu_id = gpu_ids[self._gpu_index % len(gpu_ids)]
        self._gpu_index += 1
        return gpu_id

    def _make_video_semaphore(self) -> asyncio.BoundedSemaphore:
        workers = min(max(int(self.options.video_workers_per_gpu), 1), 2)
        return asyncio.BoundedSemaphore(workers)

    def _video_slot(self, gpu_id: int) -> asyncio.BoundedSemaphore:
        return self._video_semaphores[gpu_id]

    async def _delete_temp(self, temp_output: Path) -> None:
        await asyncio.to_thread(temp_output.unlink, missing_ok=True)
