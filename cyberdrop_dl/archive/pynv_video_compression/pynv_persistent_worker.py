from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from cyberdrop_dl.archive.pynv_video_compression.pynv_transcode_worker import _stringify_config, _transcode_file


def _write_message(lock: threading.Lock, payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        sys.stderr.write("Usage: pynv_persistent_worker <gpu_id> <max_jobs>\n")
        return 2

    gpu_id = int(argv[0])
    max_jobs = max(int(argv[1]), 1)

    import PyNvVideoCodec

    write_lock = threading.Lock()
    job_slots = threading.BoundedSemaphore(max_jobs)
    active_jobs: set[threading.Thread] = set()
    active_jobs_lock = threading.Lock()
    shutting_down = False

    def run_job(job_id: int, source: str, output: str, config: dict) -> None:
        nonlocal active_jobs
        try:
            with job_slots:
                _transcode_file(PyNvVideoCodec, source, output, gpu_id, _stringify_config(config))
            _write_message(write_lock, {"job_id": job_id, "status": "ok"})
        except Exception as e:
            _write_message(write_lock, {"job_id": job_id, "status": "error", "error": str(e)})
        finally:
            current = threading.current_thread()
            with active_jobs_lock:
                active_jobs.discard(current)

    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        command = str(message.get("command", "")).casefold()
        if command == "shutdown":
            shutting_down = True
            break
        if command != "transcode":
            _write_message(
                write_lock,
                {
                    "job_id": message.get("job_id"),
                    "status": "error",
                    "error": f"Unsupported worker command: {command}",
                },
            )
            continue

        if shutting_down:
            _write_message(
                write_lock,
                {
                    "job_id": message.get("job_id"),
                    "status": "error",
                    "error": "Persistent worker is shutting down",
                },
            )
            continue

        job_id = int(message["job_id"])
        source = str(Path(message["source"]))
        output = str(Path(message["output"]))
        config = dict(message.get("config", {}))
        thread = threading.Thread(
            target=run_job,
            args=(job_id, source, output, config),
            name=f"cyberdrop-pynv-job-{gpu_id}-{job_id}",
            daemon=True,
        )
        with active_jobs_lock:
            active_jobs.add(thread)
        thread.start()

    while True:
        with active_jobs_lock:
            remaining = list(active_jobs)
        if not remaining:
            return 0
        for thread in remaining:
            thread.join(timeout=0.1)


if __name__ == "__main__":
    raise SystemExit(main())
