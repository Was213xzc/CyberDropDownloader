from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import json
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from cyberdrop_dl.constants import FILE_FORMATS
from cyberdrop_dl.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import ModuleType

    from cyberdrop_dl.config.config_model import CompressionOptions
    from cyberdrop_dl.data_structures.url_objects import MediaItem
    from cyberdrop_dl.managers.manager import Manager


CompressionStatus = Literal["compressed", "skipped", "failed"]
MediaType = Literal["video", "image"]
VideoProfile = Literal["hevc_balanced", "av1_savings", "custom"]

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


@dataclass(slots=True, kw_only=True)
class VideoCodecCapabilities:
    codec: str
    supported: bool = True
    num_encoder_engines: int = 1
    num_max_bframes: int = 0
    support_lookahead: bool = False
    support_temporal_aq: bool = False
    support_10bit_encode: bool = False


@dataclass(slots=True, kw_only=True)
class EffectiveVideoSettings:
    profile: VideoProfile
    requested_profile: VideoProfile
    codec: str
    cq: int
    bf: int
    gop: int
    idrperiod: int
    preset: str
    tuning_info: str
    aq: bool = False
    temporalaq: bool = False
    lookahead: int = 0
    support_10bit_encode: bool = False
    codec_supported: bool = True
    unsupported_reason: str = ""
    downgraded_from: VideoProfile | None = None


class _PersistentWorkerDied(RuntimeError):
    pass


class _PersistentPynvGpuWorker:
    def __init__(self, gpu_id: int, max_jobs: int) -> None:
        self.gpu_id = gpu_id
        self.max_jobs = max(max_jobs, 1)
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._job_counter = 0
        self._closed = False

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.alive:
            return
        if self._process is not None:
            await self.close()

        kwargs = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._closed = False
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "cyberdrop_dl.utils.pynv_persistent_worker",
            str(self.gpu_id),
            str(self.max_jobs),
            **kwargs,
        )
        self._stdout_task = asyncio.create_task(
            self._read_stdout(),
            name=f"cyberdrop-pynv-worker-{self.gpu_id}-stdout",
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(),
            name=f"cyberdrop-pynv-worker-{self.gpu_id}-stderr",
        )
        self._watch_task = asyncio.create_task(
            self._watch_process(),
            name=f"cyberdrop-pynv-worker-{self.gpu_id}-watch",
        )

    async def close(self) -> None:
        self._closed = True
        process = self._process
        if process is None:
            return

        try:
            if process.returncode is None and process.stdin is not None:
                payload = json.dumps({"command": "shutdown"}, ensure_ascii=False) + "\n"
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    async with self._write_lock:
                        process.stdin.write(payload.encode("utf-8"))
                        await process.stdin.drain()
                with contextlib.suppress(OSError):
                    process.stdin.close()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError, OSError):
                        process.kill()
                    with contextlib.suppress(ProcessLookupError, OSError, asyncio.TimeoutError):
                        await asyncio.wait_for(process.wait(), timeout=5)
        finally:
            self._fail_pending(_PersistentWorkerDied("Persistent PyNv worker closed"))
            tasks = [task for task in (self._stdout_task, self._stderr_task, self._watch_task) if task is not None]
            for task in tasks:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                if tasks:
                    await asyncio.gather(*tasks)
            self._process = None
            self._stdout_task = None
            self._stderr_task = None
            self._watch_task = None

    async def transcode(self, source: Path, temp_output: Path, config: dict[str, Any]) -> None:
        await self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise _PersistentWorkerDied("Persistent PyNv worker is unavailable")
        if process.returncode is not None:
            raise _PersistentWorkerDied(self._worker_exit_error())

        loop = asyncio.get_running_loop()
        self._job_counter += 1
        job_id = self._job_counter
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[job_id] = future

        payload = json.dumps(
            {
                "command": "transcode",
                "job_id": job_id,
                "source": str(source),
                "output": str(temp_output),
                "config": config,
            },
            ensure_ascii=False,
        ) + "\n"
        try:
            async with self._write_lock:
                process.stdin.write(payload.encode("utf-8"))
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._pending.pop(job_id, None)
            raise _PersistentWorkerDied("Persistent PyNv worker input pipe closed") from e

        response = await future
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error") or "Persistent PyNv worker failed"))

    async def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        while True:
            line = await process.stdout.readline()
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                self._stderr_lines.append(f"Invalid JSON from persistent worker: {line!r}")
                continue

            job_id = message.get("job_id")
            if not isinstance(job_id, int):
                continue
            future = self._pending.pop(job_id, None)
            if future is not None and not future.done():
                future.set_result(message)

    async def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        while True:
            line = await process.stderr.readline()
            if not line:
                return
            message = line.decode("utf-8", errors="replace").strip()
            if message:
                self._stderr_lines.append(message)

    async def _watch_process(self) -> None:
        process = self._process
        if process is None:
            return
        returncode = await process.wait()
        if self._closed:
            return
        self._fail_pending(_PersistentWorkerDied(self._worker_exit_error(returncode)))

    def _worker_exit_error(self, returncode: int | None = None) -> str:
        code = self._process.returncode if returncode is None and self._process is not None else returncode or 1
        output = "\n".join(self._stderr_lines).strip()
        return _format_pynv_worker_failure(code, output)

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if future.done():
                continue
            future.set_exception(error.__class__(str(error)))
        self._pending.clear()


class CompressionManager:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        self._gpu_index = 0
        self._pynv_unavailable_logged = False
        self._queue: asyncio.Queue[CompressionQueueItem | None] = asyncio.Queue()
        self._queue_tasks: list[asyncio.Task[None]] = []
        self._video_semaphores: dict[int, asyncio.BoundedSemaphore] = {}
        self._video_runtime_lock = asyncio.Lock()
        self._video_runtime_task: asyncio.Task[None] | None = None
        self._video_runtime_ready = False
        self._pynv_workers: dict[int, _PersistentPynvGpuWorker] = {}
        self._codec_capabilities: dict[int, dict[str, VideoCodecCapabilities]] = {}
        self._gpu_dispatch_order: list[int] = []
        self._logged_video_settings: set[tuple[int, VideoProfile, VideoProfile, str]] = set()
        self._pending_paths: set[str] = set()
        self._pending_lock = asyncio.Lock()
        # queue.join() only covers items that have already been put on the queue.
        # Track in-flight producers so shutdown also waits for late enqueues and resume requeues.
        self._active_producers = 0
        self._producers_idle = asyncio.Event()
        self._producers_idle.set()
        self._resumed = False

    @property
    def options(self) -> CompressionOptions:
        return self.manager.config.compression_options

    @property
    def _pending_file(self) -> Path | None:
        path_manager = getattr(self.manager, "path_manager", None)
        candidate = getattr(path_manager, "compression_pending_file", None)
        return candidate if isinstance(candidate, Path) else None

    def startup(self) -> None:
        self._queue_tasks = [task for task in self._queue_tasks if not task.done()]
        for worker_id in range(len(self._queue_tasks), self._queue_worker_count()):
            new_worker_id = worker_id + 1
            self._queue_tasks.append(
                asyncio.create_task(
                    self._run_queue_worker(new_worker_id),
                    name=f"cyberdrop-compression-worker-{new_worker_id}",
                )
            )
        self._start_video_runtime_if_needed()
        if not self._resumed:
            self._resumed = True
            self._resume_pending_from_disk()

    async def close(self) -> None:
        await self.join()
        self._queue_tasks = [task for task in self._queue_tasks if not task.done()]
        if not self._queue_tasks:
            await self._close_video_runtime()
            return

        for _ in self._queue_tasks:
            await self._queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._queue_tasks)
        self._queue_tasks.clear()
        await self._close_video_runtime()

    async def join(self) -> None:
        if not self._queue_tasks:
            return
        while True:
            await self._queue.join()
            if self._active_producers == 0 and self._queue.empty():
                return
            await self._producers_idle.wait()

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
        self._producer_started()
        try:
            await self._track_pending(media_item.complete_file)
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
        finally:
            self._producer_finished()

    async def enqueue_existing_file_if_needed(
        self,
        domain: str,
        media_item: MediaItem,
        process_completed: Callable[[MediaItem, str], Awaitable[None]],
        handle_completion: Callable[[MediaItem, bool], Awaitable[None]],
        *,
        downloaded: bool = False,
        allow_images: bool = True,
    ) -> bool:
        if not await self._should_enqueue_existing_file(media_item, allow_images=allow_images):
            return False

        await self.enqueue_completed_download(
            domain,
            media_item,
            process_completed,
            handle_completion,
            downloaded=downloaded,
        )
        return True

    async def _run_queue_worker(self, worker_id: int) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            process_task = asyncio.create_task(self._process_queue_item(item, worker_id))
            try:
                await asyncio.shield(process_task)
            except asyncio.CancelledError:
                log(
                    f"Compression worker #{worker_id} cancellation deferred until {item.media_item.complete_file} finishes",
                    30,
                )
                try:
                    await process_task
                finally:
                    self._queue.task_done()
                raise
            except BaseException:
                self._queue.task_done()
                raise
            else:
                self._queue.task_done()

    async def _process_queue_item(self, item: CompressionQueueItem, worker_id: int) -> None:
        source_path = Path(item.media_item.complete_file)
        self._notify_current(worker_id, source_path)
        progress_tracking = await self._start_progress_tracking(worker_id, source_path)
        try:
            try:
                await self.compress_media_item(item.media_item)
            except Exception as e:
                log(f"Compression queue failed for {source_path}: {e}", 40, exc_info=True)

            try:
                await item.process_completed(item.media_item, item.domain)
                await item.handle_completion(item.media_item, downloaded=item.downloaded)
                if item.finalize_download is not None:
                    await item.finalize_download(item.media_item, downloaded=item.downloaded)
            except Exception as e:
                log(f"Post-download completion failed for {source_path}: {e}", 40, exc_info=True)
        finally:
            await self._stop_progress_tracking(worker_id, progress_tracking)
            self._notify_current(worker_id, None)
            await self._untrack_pending(source_path)
            if item.completion_lock is not None and item.completion_lock.locked():
                item.completion_lock.release()

    async def _track_pending(self, path: Path) -> None:
        key = self._pending_key(path)
        async with self._pending_lock:
            if key in self._pending_paths:
                return
            self._pending_paths.add(key)
            await asyncio.to_thread(self._write_pending_file, sorted(self._pending_paths))
        self._notify_pending_count()
        self._notify_total_increment(1)

    async def _untrack_pending(self, path: Path) -> None:
        key = self._pending_key(path)
        async with self._pending_lock:
            if key not in self._pending_paths:
                return
            self._pending_paths.discard(key)
            await asyncio.to_thread(self._write_pending_file, sorted(self._pending_paths))
        self._notify_pending_count()

    @staticmethod
    def _pending_key(path: Path) -> str:
        return str(Path(path))

    def _write_pending_file(self, entries: list[str]) -> None:
        target = self._pending_file
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(target)
        except OSError as e:
            log(f"Unable to persist pending compression list: {e}", 30)

    def _read_pending_file(self) -> list[str]:
        target = self._pending_file
        if target is None or not target.is_file():
            return []
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log(f"Unable to read pending compression list: {e}", 30)
            return []
        return [entry for entry in data if isinstance(entry, str)]

    def _resume_pending_from_disk(self) -> None:
        entries = self._read_pending_file()
        resumable: list[str] = []
        for entry in entries:
            path = Path(entry)
            if path.is_file():
                resumable.append(self._pending_key(path))
        if not resumable:
            if entries:
                self._write_pending_file([])
            self._notify_pending_count()
            return

        self._pending_paths.update(resumable)
        self._write_pending_file(sorted(self._pending_paths))
        self._notify_pending_count()
        self._notify_total_increment(len(resumable))
        log(f"Resuming {len(resumable)} pending compression(s) from previous session", 20)
        self._producer_started()
        try:
            asyncio.get_running_loop().create_task(self._run_requeue_standalone_paths(resumable))
        except RuntimeError:
            self._producer_finished()
            return

    async def _run_requeue_standalone_paths(self, paths: list[str]) -> None:
        try:
            await self._requeue_standalone_paths(paths)
        finally:
            self._producer_finished()

    async def _requeue_standalone_paths(self, paths: list[str]) -> None:
        for entry in paths:
            media_item = self._build_standalone_media_item(Path(entry))
            await self._queue.put(
                CompressionQueueItem(
                    domain="",
                    media_item=media_item,
                    process_completed=_noop_process_completed,
                    handle_completion=_noop_handle_completion,
                )
            )

    def _build_standalone_media_item(self, path: Path) -> MediaItem:
        from yarl import URL

        absolute = path if path.is_absolute() else path.resolve()
        stand_in = SimpleNamespace(
            complete_file=path,
            partial_file=path,
            filename=path.name,
            original_filename=path.name,
            download_filename=path.name,
            db_path="",
            is_segment=False,
            filesize=path.stat().st_size if path.is_file() else None,
            url=URL(absolute.as_uri()),
            ext=path.suffix,
            hash=None,
        )
        return cast("MediaItem", stand_in)

    def _notify_current(self, worker_id: int, path: Path | None) -> None:
        progress = self._compression_progress()
        if progress is None:
            return
        progress.set_current(worker_id, path)

    def _notify_task_started(self, worker_id: int, path: Path, total: int) -> None:
        progress = self._compression_progress()
        if progress is None or not hasattr(progress, "start_task"):
            return
        progress.start_task(worker_id, path, total)

    def _notify_task_progress(self, worker_id: int, completed: int, total: int) -> None:
        progress = self._compression_progress()
        if progress is None or not hasattr(progress, "update_task"):
            return
        progress.update_task(worker_id, completed, total)

    def _notify_task_finished(self, worker_id: int) -> None:
        progress = self._compression_progress()
        if progress is None or not hasattr(progress, "finish_task"):
            return
        progress.finish_task(worker_id)

    def _notify_pending_count(self) -> None:
        progress = self._compression_progress()
        if progress is None:
            return
        progress.set_pending_count(len(self._pending_paths))

    def _notify_total_increment(self, delta: int) -> None:
        if delta <= 0:
            return
        progress = self._compression_progress()
        if progress is None or not hasattr(progress, "increment_total"):
            return
        progress.increment_total(delta)

    def _compression_progress(self):
        progress_manager = getattr(self.manager, "progress_manager", None)
        return getattr(progress_manager, "compression_progress", None)

    async def _start_progress_tracking(
        self, worker_id: int, source_path: Path
    ) -> tuple[asyncio.Event, asyncio.Task[None], int] | None:
        total = await asyncio.to_thread(self._compression_total_size, source_path)
        if total is None or not self._should_track_progress(source_path):
            return None

        self._notify_task_started(worker_id, source_path, total)
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(
            self._monitor_compression_progress(worker_id, source_path, total, stop_event),
            name=f"cyberdrop-compression-progress-{worker_id}",
        )
        return stop_event, monitor_task, total

    async def _stop_progress_tracking(
        self,
        worker_id: int,
        progress_tracking: tuple[asyncio.Event, asyncio.Task[None], int] | None,
    ) -> None:
        if progress_tracking is None:
            return

        stop_event, monitor_task, total = progress_tracking
        stop_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task
        self._notify_task_progress(worker_id, total, total)
        self._notify_task_finished(worker_id)

    async def _monitor_compression_progress(
        self,
        worker_id: int,
        source_path: Path,
        total: int,
        stop_event: asyncio.Event,
    ) -> None:
        temp_output = self._temp_output_template(source_path)
        while True:
            current_size = await asyncio.to_thread(self._current_compression_output_size, temp_output)
            self._notify_task_progress(worker_id, min(current_size, total), total)
            if stop_event.is_set():
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

    @staticmethod
    def _compression_total_size(source_path: Path) -> int | None:
        try:
            return source_path.stat().st_size
        except OSError:
            return None

    def _should_track_progress(self, source_path: Path) -> bool:
        options = self.options
        ext = source_path.suffix.lower()
        return (options.compress_videos and ext in FILE_FORMATS["Videos"]) or (
            options.compress_images and ext in FILE_FORMATS["Images"]
        )

    @staticmethod
    def _temp_output_template(source_path: Path) -> Path:
        return source_path.with_name(f"{source_path.stem}.compressed{source_path.suffix}")

    def _current_compression_output_size(self, temp_output: Path) -> int:
        largest = 0
        for path in self._temp_output_cleanup_candidates(temp_output):
            try:
                if path.is_file():
                    largest = max(largest, path.stat().st_size)
            except OSError:
                continue
        return largest

    async def _should_enqueue_existing_file(self, media_item: MediaItem, *, allow_images: bool) -> bool:
        options = self.options
        if not options.enabled or media_item.is_segment:
            return False

        source = getattr(media_item, "complete_file", None)
        if source is None:
            filename = media_item.download_filename or media_item.filename
            source = media_item.download_folder / filename
            media_item.complete_file = source
        source = Path(source)
        if not await asyncio.to_thread(source.is_file):
            return False

        ext = source.suffix.lower()
        if options.compress_videos and ext in FILE_FORMATS["Videos"]:
            return not source.name.startswith(_COMPRESSED_MARKER)
        if allow_images and options.compress_images and ext in FILE_FORMATS["Images"]:
            return True
        return False

    def _producer_started(self) -> None:
        self._active_producers += 1
        self._producers_idle.clear()

    def _producer_finished(self) -> None:
        self._active_producers = max(0, self._active_producers - 1)
        if self._active_producers == 0:
            self._producers_idle.set()

    def _queue_worker_count(self) -> int:
        gpu_ids = self.options.gpu_ids or [0]
        workers_per_gpu = max(int(self.options.video_workers_per_gpu), 1)
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
        gpu_id = self._next_gpu_id()
        pynv = await self._ensure_video_runtime()
        if pynv is None:
            return CompressionResult(
                status="skipped",
                media_type="video",
                backend="pynv",
                path=source,
                gpu_id=gpu_id,
                error="PyNvVideoCodec is not installed",
            )

        settings = self._effective_video_settings(gpu_id)
        self._log_effective_video_settings(gpu_id, settings)
        if not settings.codec_supported:
            return CompressionResult(
                status="skipped",
                media_type="video",
                backend="pynv",
                path=source,
                gpu_id=gpu_id,
                codec=settings.codec,
                cq=settings.cq,
                bf=settings.bf,
                error=settings.unsupported_reason or f"{settings.codec.upper()} encoding is not supported by this GPU",
            )

        result_kwargs = {
            "media_type": "video",
            "backend": "pynv",
            "path": source,
            "gpu_id": gpu_id,
            "codec": settings.codec,
            "cq": settings.cq,
            "bf": settings.bf,
        }

        temp_output_template = self._temp_output_template(source)
        errors: list[str] = []
        cq_attempts = self._video_cq_attempts(settings.codec)
        for attempt_cq in cq_attempts:
            attempt_settings = replace(settings, cq=attempt_cq)
            attempt_kwargs = result_kwargs | {"cq": attempt_cq, "bf": attempt_settings.bf}
            temp_output = temp_output_template
            await self._delete_temp(temp_output)
            try:
                async with self._video_slot(gpu_id):
                    await self._transcode_with_pynv_subprocess(source, temp_output, gpu_id, attempt_settings)
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
            **(result_kwargs | {"cq": cq_attempts[-1], "bf": settings.bf}),
        )

    async def _compress_image(self, source: Path) -> CompressionResult:
        result_kwargs = {"media_type": "image", "backend": "pillow", "path": source}
        temp_output = self._temp_output_template(source)
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
        settings: EffectiveVideoSettings,
    ) -> None:
        await self._ensure_video_runtime()
        worker = self._pynv_workers.get(gpu_id)
        if worker is None:
            raise RuntimeError(f"Persistent PyNv worker is unavailable for GPU {gpu_id}")
        try:
            await worker.transcode(source, temp_output, self._pynv_transcode_kwargs(settings))
        except _PersistentWorkerDied:
            await self._restart_pynv_worker(gpu_id)
            worker = self._pynv_workers.get(gpu_id)
            if worker is None:
                raise RuntimeError(f"Persistent PyNv worker restart failed for GPU {gpu_id}")
            await worker.transcode(source, temp_output, self._pynv_transcode_kwargs(settings))

    def _transcode_with_pynv(
        self,
        pynv: ModuleType,
        source: Path,
        temp_output: Path,
        gpu_id: int,
        settings: EffectiveVideoSettings | str,
        cq: int | None = None,
    ) -> None:
        effective_settings = self._coerce_effective_video_settings(settings, cq)
        transcode_kwargs = self._stringify_pynv_kwargs(self._pynv_transcode_kwargs(effective_settings))
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

    def _pynv_transcode_kwargs(self, settings: EffectiveVideoSettings | str, cq: int | None = None) -> dict:
        effective_settings = self._coerce_effective_video_settings(settings, cq)
        kwargs = {
            "codec": effective_settings.codec,
            "format": "NV12",
            "usedevicememory": True,
            "usecpuinputbuffer": False,
            "rc": "constqp",
            "constqp": effective_settings.cq,
            "bf": effective_settings.bf,
            "gop": effective_settings.gop,
            "idrperiod": effective_settings.idrperiod,
            "preset": effective_settings.preset,
            "tuning_info": effective_settings.tuning_info,
        }
        if effective_settings.aq:
            kwargs["aq"] = 1
        if effective_settings.temporalaq:
            kwargs["temporalaq"] = 1
        if effective_settings.lookahead > 0:
            kwargs["lookahead"] = effective_settings.lookahead
        return kwargs

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
        gpu_ids = self._gpu_dispatch_order or self.options.gpu_ids or [0]
        gpu_id = gpu_ids[self._gpu_index % len(gpu_ids)]
        self._gpu_index += 1
        return gpu_id

    def _make_video_semaphore(self, gpu_id: int) -> asyncio.BoundedSemaphore:
        return asyncio.BoundedSemaphore(self._worker_slots_for_gpu(gpu_id))

    def _video_slot(self, gpu_id: int) -> asyncio.BoundedSemaphore:
        semaphore = self._video_semaphores.get(gpu_id)
        if semaphore is None:
            semaphore = self._make_video_semaphore(gpu_id)
            self._video_semaphores[gpu_id] = semaphore
        return semaphore

    def _start_video_runtime_if_needed(self) -> None:
        if not self.options.enabled or not self.options.compress_videos:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._video_runtime_ready:
            return
        if self._video_runtime_task is not None and not self._video_runtime_task.done():
            return
        self._video_runtime_task = loop.create_task(
            self._ensure_video_runtime(),
            name="cyberdrop-compression-video-runtime",
        )
        self._video_runtime_task.add_done_callback(self._handle_video_runtime_task_done)

    def _handle_video_runtime_task_done(self, task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            exception = task.exception()
            if exception is not None:
                log(f"Compression video runtime startup failed: {exception}", 30)

    async def _close_video_runtime(self) -> None:
        task = self._video_runtime_task
        self._video_runtime_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        workers = list(self._pynv_workers.values())
        self._pynv_workers.clear()
        for worker in workers:
            with contextlib.suppress(Exception):
                await worker.close()
        self._video_runtime_ready = False
        self._codec_capabilities.clear()
        self._gpu_dispatch_order.clear()
        self._video_semaphores.clear()
        self._logged_video_settings.clear()

    async def _ensure_video_runtime(self) -> ModuleType | None:
        if self._video_runtime_ready and self._pynv_workers:
            return self._import_pynv()

        async with self._video_runtime_lock:
            if self._video_runtime_ready and self._pynv_workers:
                return self._import_pynv()

            pynv = self._import_pynv()
            if pynv is None:
                return None

            capabilities = await asyncio.to_thread(self._probe_encoder_capabilities, pynv)
            new_workers: dict[int, _PersistentPynvGpuWorker] = {}
            configured_gpu_ids = self.options.gpu_ids or [0]
            dispatch_order: list[int] = []
            try:
                for gpu_id in configured_gpu_ids:
                    slots = self._worker_slots_for_gpu(gpu_id, capabilities.get(gpu_id))
                    dispatch_order.extend([gpu_id] * slots)
                    worker = self._pynv_workers.get(gpu_id)
                    if worker is None or worker.max_jobs != slots or not worker.alive:
                        if worker is not None:
                            await worker.close()
                        worker = self._create_persistent_worker(gpu_id, slots)
                        await worker.start()
                    new_workers[gpu_id] = worker
                    self._video_semaphores[gpu_id] = asyncio.BoundedSemaphore(slots)

                for stale_gpu_id, worker in list(self._pynv_workers.items()):
                    if stale_gpu_id not in new_workers:
                        await worker.close()

                self._pynv_workers = new_workers
                self._codec_capabilities = capabilities
                self._gpu_dispatch_order = dispatch_order or configured_gpu_ids
                self._video_runtime_ready = True
                return pynv
            except Exception:
                for worker in new_workers.values():
                    with contextlib.suppress(Exception):
                        await worker.close()
                self._pynv_workers = {}
                self._codec_capabilities = {}
                self._gpu_dispatch_order = []
                self._video_runtime_ready = False
                raise

    async def _restart_pynv_worker(self, gpu_id: int) -> None:
        async with self._video_runtime_lock:
            worker = self._pynv_workers.pop(gpu_id, None)
            if worker is not None:
                await worker.close()
            replacement = self._create_persistent_worker(gpu_id, self._worker_slots_for_gpu(gpu_id))
            await replacement.start()
            self._pynv_workers[gpu_id] = replacement

    def _create_persistent_worker(self, gpu_id: int, max_jobs: int) -> _PersistentPynvGpuWorker:
        return _PersistentPynvGpuWorker(gpu_id, max_jobs)

    def _probe_encoder_capabilities(self, pynv: ModuleType) -> dict[int, dict[str, VideoCodecCapabilities]]:
        capabilities: dict[int, dict[str, VideoCodecCapabilities]] = {}
        for gpu_id in self.options.gpu_ids or [0]:
            capabilities[gpu_id] = {
                "hevc": self._probe_codec_capabilities(pynv, gpu_id, "hevc"),
                "av1": self._probe_codec_capabilities(pynv, gpu_id, "av1"),
            }
        return capabilities

    def _probe_codec_capabilities(self, pynv: ModuleType, gpu_id: int, codec: str) -> VideoCodecCapabilities:
        default_supported = codec == "hevc"
        try:
            raw = pynv.GetEncoderCaps(gpu_id, codec)
        except Exception as e:
            log(f"Unable to probe {codec.upper()} encoder caps for GPU {gpu_id}: {e}", 30)
            return VideoCodecCapabilities(
                codec=codec,
                supported=default_supported,
                num_encoder_engines=1,
                num_max_bframes=max(int(self.options.bf), 0),
            )

        data = raw if isinstance(raw, dict) else vars(raw)
        return VideoCodecCapabilities(
            codec=codec,
            supported=bool(data),
            num_encoder_engines=max(int(data.get("num_encoder_engines", 1)), 1),
            num_max_bframes=max(int(data.get("num_max_bframes", 0)), 0),
            support_lookahead=bool(data.get("support_lookahead", 0)),
            support_temporal_aq=bool(data.get("support_temporal_aq", 0)),
            support_10bit_encode=bool(data.get("support_10bit_encode", 0)),
        )

    def _worker_slots_for_gpu(
        self,
        gpu_id: int,
        capabilities: dict[str, VideoCodecCapabilities] | None = None,
    ) -> int:
        capabilities = capabilities or self._codec_capabilities.get(gpu_id, {})
        configured_workers = max(int(self.options.video_workers_per_gpu), 1)
        if not capabilities:
            return configured_workers
        encoder_engines = max(
            (cap.num_encoder_engines for cap in capabilities.values() if cap.supported),
            default=1,
        )
        return max(1, min(configured_workers, encoder_engines))

    def _capabilities_for(self, gpu_id: int, codec: str) -> VideoCodecCapabilities:
        capabilities = self._codec_capabilities.get(gpu_id, {})
        if codec in capabilities:
            return capabilities[codec]
        return VideoCodecCapabilities(
            codec=codec,
            supported=(codec == "hevc"),
            num_encoder_engines=max(int(self.options.video_workers_per_gpu), 1),
            num_max_bframes=max(int(self.options.bf), 0),
        )

    def _coerce_effective_video_settings(
        self,
        settings: EffectiveVideoSettings | str,
        cq: int | None = None,
    ) -> EffectiveVideoSettings:
        if isinstance(settings, EffectiveVideoSettings):
            return settings if cq is None else replace(settings, cq=cq)

        codec = settings.casefold()
        return EffectiveVideoSettings(
            profile="custom",
            requested_profile="custom",
            codec=codec,
            cq=self._codec_cq(codec) if cq is None else cq,
            bf=int(self.options.bf),
            gop=int(self.options.gop),
            idrperiod=int(self.options.idrperiod),
            preset=self.options.preset,
            tuning_info=self.options.tuning_info,
        )

    def _effective_video_settings(self, gpu_id: int) -> EffectiveVideoSettings:
        options = self.options
        requested_profile = options.effective_video_profile()
        hevc_caps = self._capabilities_for(gpu_id, "hevc")
        av1_caps = self._capabilities_for(gpu_id, "av1")

        if requested_profile == "custom":
            codec = options.video_codec
            caps = av1_caps if codec == "av1" else hevc_caps
            codec_supported = caps.supported or codec == "hevc"
            return EffectiveVideoSettings(
                profile="custom",
                requested_profile=requested_profile,
                codec=codec,
                cq=self._codec_cq(codec),
                bf=int(options.bf),
                gop=int(options.gop),
                idrperiod=int(options.idrperiod),
                preset=options.preset,
                tuning_info=options.tuning_info,
                support_10bit_encode=caps.support_10bit_encode,
                codec_supported=codec_supported,
                unsupported_reason="" if codec_supported else f"{codec.upper()} encoding is not supported on GPU {gpu_id}",
            )

        if requested_profile == "av1_savings" and av1_caps.supported:
            return EffectiveVideoSettings(
                profile="av1_savings",
                requested_profile=requested_profile,
                codec="av1",
                cq=int(options.av1_cq),
                bf=min(5, max(av1_caps.num_max_bframes, 0)),
                gop=max(int(options.gop), 240),
                idrperiod=max(int(options.idrperiod), 240),
                preset="P5",
                tuning_info="high_quality",
                aq=True,
                temporalaq=av1_caps.support_temporal_aq,
                lookahead=16 if av1_caps.support_lookahead else 0,
                support_10bit_encode=av1_caps.support_10bit_encode,
            )

        effective_profile: VideoProfile = "hevc_balanced"
        downgraded_from: VideoProfile | None = "av1_savings" if requested_profile == "av1_savings" else None
        return EffectiveVideoSettings(
            profile=effective_profile,
            requested_profile=requested_profile,
            codec="hevc",
            cq=int(options.hevc_cq),
            bf=min(3, max(hevc_caps.num_max_bframes, 0)),
            gop=max(int(options.gop), 120),
            idrperiod=max(int(options.idrperiod), 120),
            preset="P4",
            tuning_info="high_quality",
            aq=True,
            temporalaq=False,
            lookahead=10 if hevc_caps.support_lookahead else 0,
            support_10bit_encode=hevc_caps.support_10bit_encode,
            downgraded_from=downgraded_from,
        )

    def _log_effective_video_settings(self, gpu_id: int, settings: EffectiveVideoSettings) -> None:
        key = (gpu_id, settings.requested_profile, settings.profile, settings.codec)
        if key in self._logged_video_settings:
            return
        self._logged_video_settings.add(key)
        downgrade_note = ""
        if settings.downgraded_from is not None:
            downgrade_note = f", downgraded from {settings.downgraded_from}"
        log(
            "Compression profile "
            f"GPU {gpu_id}: requested={settings.requested_profile}, effective={settings.profile}{downgrade_note}, "
            f"codec={settings.codec}, cq={settings.cq}, bf={settings.bf}, preset={settings.preset}, "
            f"lookahead={settings.lookahead}, aq={int(settings.aq)}, temporalaq={int(settings.temporalaq)}, "
            f"support_10bit={int(settings.support_10bit_encode)}",
            20,
        )

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


async def _noop_process_completed(media_item: MediaItem, domain: str) -> None:
    return None


async def _noop_handle_completion(media_item: MediaItem, downloaded: bool = True) -> None:
    return None


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
