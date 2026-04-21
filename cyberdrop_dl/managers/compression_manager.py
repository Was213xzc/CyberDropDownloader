from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cyberdrop_dl.constants import FILE_FORMATS
from cyberdrop_dl.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from types import ModuleType

    from cyberdrop_dl.config.config_model import CompressionOptions
    from cyberdrop_dl.data_structures.url_objects import MediaItem
    from cyberdrop_dl.managers.manager import Manager


CompressionStatus = Literal["compressed", "skipped", "failed"]
MediaType = Literal["video", "image"]

_COMPRESSED_MARKER = "[COMPRESSED] "


@dataclass(slots=True, kw_only=True)
class CompressionQueueItem:
    domain: str
    media_item: MediaItem
    process_completed: Callable[[MediaItem, str], Awaitable[None]]
    handle_completion: Callable[[MediaItem, bool], Awaitable[None]]
    downloaded: bool = True
    finalize_download: Callable[[MediaItem, bool], Awaitable[None]] | None = None
    completion_lock: asyncio.Lock | None = None


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
        self._queue: asyncio.Queue[CompressionQueueItem | None] = asyncio.Queue()
        self._queue_tasks: list[asyncio.Task[None]] = []
        self._video_semaphores: defaultdict[int, asyncio.BoundedSemaphore] = defaultdict(self._make_video_semaphore)

    @property
    def options(self) -> CompressionOptions:
        return self.manager.config.compression_options

    def startup(self) -> None:
        self._queue_tasks = [task for task in self._queue_tasks if not task.done()]
        for worker_id in range(len(self._queue_tasks), self._queue_worker_count()):
            self._queue_tasks.append(
                asyncio.create_task(self._run_queue_worker(), name=f"cyberdrop-compression-worker-{worker_id + 1}")
            )

    async def close(self) -> None:
        await self.join()
        if not self._queue_tasks:
            return

        for _ in self._queue_tasks:
            await self._queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._queue_tasks)
        self._queue_tasks.clear()

    async def join(self) -> None:
        if not self._queue_tasks:
            return
        await self._queue.join()

    async def enqueue_completed_download(
        self,
        domain: str,
        media_item: MediaItem,
        process_completed: Callable[[MediaItem, str], Awaitable[None]],
        handle_completion: Callable[[MediaItem, bool], Awaitable[None]],
        *,
        downloaded: bool = True,
        finalize_download: Callable[[MediaItem, bool], Awaitable[None]] | None = None,
        completion_lock: asyncio.Lock | None = None,
    ) -> None:
        self.startup()
        await self._queue.put(
            CompressionQueueItem(
                domain=domain,
                media_item=media_item,
                process_completed=process_completed,
                handle_completion=handle_completion,
                downloaded=downloaded,
                finalize_download=finalize_download,
                completion_lock=completion_lock,
            )
        )

    async def _run_queue_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await self._process_queue_item(item)
            finally:
                self._queue.task_done()

    async def _process_queue_item(self, item: CompressionQueueItem) -> None:
        try:
            try:
                await self.compress_media_item(item.media_item)
            except Exception as e:
                log(f"Compression queue failed for {item.media_item.complete_file}: {e}", 40, exc_info=True)

            try:
                await item.process_completed(item.media_item, item.domain)
                await item.handle_completion(item.media_item, downloaded=item.downloaded)
                if item.finalize_download is not None:
                    await item.finalize_download(item.media_item, downloaded=item.downloaded)
            except Exception as e:
                log(f"Post-download completion failed for {item.media_item.complete_file}: {e}", 40, exc_info=True)
        finally:
            if item.completion_lock is not None and item.completion_lock.locked():
                item.completion_lock.release()

    def _queue_worker_count(self) -> int:
        gpu_ids = self.options.gpu_ids or [0]
        workers_per_gpu = min(max(int(self.options.video_workers_per_gpu), 1), 2)
        return max(1, len(gpu_ids) * workers_per_gpu)

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
        if result.status == "compressed" and result.media_type == "video":
            result.path = await self._apply_compressed_marker(media_item, result.path)
        await self._record_result(media_item, result)
        return result

    async def _compress_video(self, media_item: MediaItem, source: Path) -> CompressionResult:
        options = self.options
        codec = options.video_codec
        cq = self._codec_cq(codec)
        gpu_id = self._next_gpu_id()
        result_kwargs = {
            "media_type": "video",
            "backend": "pynv",
            "path": source,
            "gpu_id": gpu_id,
            "codec": codec,
            "cq": cq,
            "bf": options.bf,
        }

        pynv = self._import_pynv()
        if pynv is None:
            return CompressionResult(status="skipped", error="PyNvVideoCodec is not installed", **result_kwargs)

        temp_output_template = source.with_name(f"{source.stem}.compressed{source.suffix}")
        errors: list[str] = []
        cq_attempts = self._video_cq_attempts(codec)
        for attempt_cq in cq_attempts:
            attempt_kwargs = result_kwargs | {"cq": attempt_cq}
            temp_output = temp_output_template
            await self._delete_temp(temp_output)
            try:
                async with self._video_slot(gpu_id):
                    await self._transcode_with_pynv_subprocess(source, temp_output, gpu_id, codec, attempt_cq)
            except Exception as e:
                await self._delete_temp(temp_output)
                error = _format_pynv_exception(e)
                errors.append(f"CQ {attempt_cq}: {error}")
                if self._should_retry_with_higher_cq(error, attempt_cq):
                    continue
                return CompressionResult(status="skipped", error=error, **attempt_kwargs)

            actual_output = await asyncio.to_thread(self._resolve_pynv_output, source, temp_output)
            result = await self._finalize_output(source, actual_output, "video", **attempt_kwargs)
            if result.status == "compressed":
                return result
            errors.append(f"CQ {attempt_cq}: {result.error}")
            if not self._should_retry_with_higher_cq(result.error, attempt_cq):
                return result

        return CompressionResult(
            status="skipped",
            error=_summarize_retry_errors("PyNvVideoCodec could not create a small enough output", errors),
            **(result_kwargs | {"cq": cq_attempts[-1]}),
        )

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

        if not self._meets_savings_threshold(original_size, final_size, output_type):
            await self._delete_temp(temp_output)
            return CompressionResult(status="skipped", error="Compressed output was not small enough", **result_kwargs)

        await self._replace_temp(temp_output, source)
        return CompressionResult(status="compressed", **result_kwargs)

    async def _validate_video(self, path: Path) -> None:
        if not await asyncio.to_thread(lambda: path.is_file() and path.stat().st_size > 0):
            raise ValueError("Compressed video output is empty")

    def _validate_image(self, path: Path) -> None:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()

    async def _transcode_with_pynv_subprocess(
        self,
        source: Path,
        temp_output: Path,
        gpu_id: int,
        codec: str,
        cq: int,
    ) -> None:
        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "cyberdrop_dl.utils.pynv_transcode_worker",
            str(source),
            str(temp_output),
            str(gpu_id),
            json.dumps(self._pynv_transcode_kwargs(codec, cq)),
            **kwargs,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            output = (stderr or stdout).decode("utf8", errors="replace").strip()
            raise RuntimeError(_format_pynv_worker_failure(process.returncode, output))

    def _transcode_with_pynv(
        self,
        pynv: ModuleType,
        source: Path,
        temp_output: Path,
        gpu_id: int,
        codec: str,
        cq: int,
    ) -> None:
        transcode_kwargs = self._stringify_pynv_kwargs(self._pynv_transcode_kwargs(codec, cq))
        factory = getattr(pynv, "Transcoder", None) or getattr(pynv, "CreateTranscoder", None)
        if factory is None:
            raise RuntimeError("PyNvVideoCodec Transcoder API is unavailable")

        transcoder = None
        try:
            try:
                transcoder = factory(
                    enc_file_path=str(source),
                    muxed_file_path=str(temp_output),
                    gpu_id=gpu_id,
                    cuda_context=0,
                    cuda_stream=0,
                    **transcode_kwargs,
                )
            except TypeError:
                if getattr(pynv, "Transcoder", None) is not None:
                    transcoder = factory(str(source), str(temp_output), gpu_id, 0, 0, **transcode_kwargs)
                else:
                    transcoder = factory(str(source), str(temp_output), gpu_id, 0, 0, transcode_kwargs)

            if hasattr(transcoder, "transcode_with_mux"):
                transcoder.transcode_with_mux()
            elif hasattr(transcoder, "transcode"):
                transcoder.transcode()
            elif hasattr(transcoder, "segmented_transcode"):
                raise RuntimeError("PyNvVideoCodec transcoder only exposes segmented_transcode")
            else:
                raise RuntimeError("PyNvVideoCodec transcoder does not expose a transcode method")
        finally:
            del transcoder
            gc.collect()

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
            "gop": options.gop,
            "idrperiod": options.idrperiod,
            "preset": options.preset,
            "tuning_info": options.tuning_info,
        }

    def _stringify_pynv_kwargs(self, kwargs: dict) -> dict[str, str]:
        return {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in kwargs.items()}

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

    def _video_cq_attempts(self, codec: str) -> list[int]:
        base_cq = self._codec_cq(codec)
        max_cq = max(base_cq, int(self.options.video_cq_max))
        step = max(1, int(self.options.video_cq_retry_step))
        attempts = list(range(base_cq, max_cq + 1, step))
        if attempts[-1] != max_cq:
            attempts.append(max_cq)
        return attempts

    def _should_retry_with_higher_cq(self, error: str, cq: int) -> bool:
        if cq >= int(self.options.video_cq_max):
            return False
        normalized = error.casefold()
        return any(
            marker in normalized
            for marker in (
                "output exceeded safe size limit",
                "compressed output was not small enough",
                "created an invalid output video",
                "error writing frame",
                "failed while writing encoded frames",
            )
        )

    def _already_target_codec(self, input_codec: str, target_codec: str) -> bool:
        normalized = input_codec.casefold()
        if target_codec == "hevc":
            return normalized in {"hevc", "h265"}
        return normalized == target_codec

    def _meets_savings_threshold(self, original_size: int, final_size: int, output_type: MediaType) -> bool:
        min_savings_percent = (
            self.options.image_min_savings_percent if output_type == "image" else self.options.min_savings_percent
        )
        if min_savings_percent <= 0:
            return final_size < original_size
        threshold = 1 - (min_savings_percent / 100)
        return final_size < original_size * threshold

    async def _record_result(self, media_item: MediaItem, result: CompressionResult) -> None:
        if hasattr(self.manager, "progress_manager"):
            self.manager.progress_manager.add_compression_result(result.status, result.bytes_saved)
        if hasattr(self.manager, "log_manager"):
            try:
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
            except OSError as e:
                log(f"Unable to write compression report row for {result.path}: {e}", 30)

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

    async def _apply_compressed_marker(self, media_item: MediaItem, source: Path) -> Path:
        if source.name.startswith(_COMPRESSED_MARKER):
            return source

        candidate = source.with_name(_COMPRESSED_MARKER + source.name)
        if await asyncio.to_thread(candidate.exists):
            candidate = await asyncio.to_thread(self._next_available_marked_name, source)
            if candidate is None:
                log(f"Unable to apply [COMPRESSED] marker to {source}: no free filename", 30)
                return source

        try:
            await asyncio.to_thread(source.rename, candidate)
        except OSError as e:
            log(f"Failed to apply [COMPRESSED] marker to {source}: {e}", 30)
            return source

        media_item.complete_file = candidate
        media_item.download_filename = candidate.name
        return candidate

    def _next_available_marked_name(self, source: Path) -> Path | None:
        stem, suffix = source.stem, source.suffix
        for counter in range(1, 1000):
            candidate = source.with_name(f"{_COMPRESSED_MARKER}{stem} ({counter}){suffix}")
            if not candidate.exists():
                return candidate
        return None

    async def _replace_temp(self, temp_output: Path, source: Path) -> None:
        for attempt in range(10):
            try:
                await asyncio.to_thread(temp_output.replace, source)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                gc.collect()
                await asyncio.sleep(0.25)

    async def _delete_temp(self, temp_output: Path) -> None:
        for attempt in range(10):
            try:
                await asyncio.to_thread(self._delete_temp_outputs, temp_output)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                gc.collect()
                await asyncio.sleep(0.25)

    def _delete_temp_outputs(self, temp_output: Path) -> None:
        for path in self._temp_output_cleanup_candidates(temp_output):
            path.unlink(missing_ok=True)

    def _temp_output_cleanup_candidates(self, temp_output: Path) -> list[Path]:
        candidates = []
        seen: set[Path] = set()
        timestamped_outputs = [temp_output]
        timestamped_outputs.extend(
            path for path in temp_output.parent.glob(f"{temp_output.stem}*{temp_output.suffix}") if path != temp_output
        )
        for path in timestamped_outputs:
            for candidate in (path, path.with_suffix(path.suffix + ".faststart")):
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
        return candidates


def _format_pynv_exception(error: Exception, stage: str = "input") -> str:
    message = str(error).strip()
    normalized = message.casefold()
    if "invalid data found when processing input" in normalized or "avformat_open_input" in normalized:
        if stage == "output":
            return "PyNvVideoCodec created an invalid output video at this CQ"
        return (
            "PyNvVideoCodec could not open the input video. "
            "The file is unsupported, corrupted, incomplete, or not a real video container."
        )
    if "timescale not set" in normalized:
        return "PyNvVideoCodec could not read this MP4 stream timing metadata"
    if "error writing frame" in normalized:
        return "PyNvVideoCodec failed while writing encoded frames"
    return message or error.__class__.__name__


def _format_pynv_worker_failure(returncode: int, output: str) -> str:
    if output:
        return output
    if sys.platform == "win32":
        return f"PyNvVideoCodec worker failed with exit code {returncode} (0x{returncode & 0xFFFFFFFF:08X})"
    return f"PyNvVideoCodec worker failed with exit code {returncode}"


def _summarize_retry_errors(prefix: str, errors: list[str], *, max_errors: int = 3) -> str:
    if not errors:
        return prefix
    return f"{prefix} after retries: {'; '.join(errors[-max_errors:])}"
