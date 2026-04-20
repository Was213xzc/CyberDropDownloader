from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 4:
        sys.stderr.write("Usage: pynv_transcode_worker <source> <output> <gpu_id> <config_json>\n")
        return 2

    source, output, gpu_id, config_json = argv
    config = _stringify_config(json.loads(config_json))

    import PyNvVideoCodec

    _delete_outputs(output)
    duration = _get_duration(PyNvVideoCodec, source, int(gpu_id))
    transcoder = None
    try:
        transcoder = PyNvVideoCodec.Transcoder(source, output, int(gpu_id), 0, 0, **config)
        if not hasattr(transcoder, "segmented_transcode"):
            raise RuntimeError("PyNvVideoCodec transcoder does not expose segmented_transcode")
        transcoder.segmented_transcode(0.0, duration)
    finally:
        del transcoder
        gc.collect()
    actual_output = _resolve_output(output)
    _validate_output(PyNvVideoCodec, str(actual_output), int(gpu_id))
    return 0


def _stringify_config(config: dict) -> dict[str, str]:
    return {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in config.items()}


def _delete_outputs(output: str) -> None:
    template = Path(output)
    for path in _candidate_outputs(template):
        path.unlink(missing_ok=True)


def _candidate_outputs(template: Path) -> list[Path]:
    candidates = [template]
    candidates.extend(path for path in template.parent.glob(f"{template.stem}*{template.suffix}") if path != template)
    return candidates


def _resolve_output(output: str) -> Path:
    candidates = [path for path in _candidate_outputs(Path(output)) if path.is_file()]
    if not candidates:
        raise RuntimeError("PyNvVideoCodec did not create an output file")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _get_duration(pynv_module, source: str, gpu_id: int) -> float:
    decoder = None
    try:
        decoder = pynv_module.SimpleDecoder(source, gpu_id=gpu_id, use_device_memory=True)
        metadata = decoder.get_stream_metadata()
        duration = float(getattr(metadata, "duration", 0) or getattr(metadata, "duration_in_seconds", 0) or 0)
        if duration <= 0:
            raise RuntimeError("PyNvVideoCodec input validation failed: zero duration")
        return duration
    finally:
        del decoder
        gc.collect()


def _validate_output(pynv_module, output: str, gpu_id: int) -> None:
    decoder = None
    try:
        decoder = pynv_module.SimpleDecoder(output, gpu_id=gpu_id, use_device_memory=True)
        metadata = decoder.get_stream_metadata()
        duration = float(getattr(metadata, "duration", 0) or 0)
        if duration <= 0:
            raise RuntimeError("PyNvVideoCodec output validation failed: zero duration")
        _ = decoder[0]
    finally:
        del decoder
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
