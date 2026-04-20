from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def _suppress_windows_error_dialogs() -> None:
    if sys.platform != "win32":
        return

    import ctypes

    sem_failcriticalerrors = 0x0001
    sem_nogpfault_errorbox = 0x0002
    sem_noopenfile_errorbox = 0x8000
    ctypes.windll.kernel32.SetErrorMode(
        sem_failcriticalerrors | sem_nogpfault_errorbox | sem_noopenfile_errorbox
    )


_suppress_windows_error_dialogs()


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 4:
        sys.stderr.write("Usage: pynv_transcode_worker <source> <output> <gpu_id> <config_json>\n")
        return 2

    source, output, gpu_id, config_json = argv
    config = _stringify_config(json.loads(config_json))

    import PyNvVideoCodec

    _delete_outputs(output)
    transcoder = None
    try:
        transcoder = PyNvVideoCodec.Transcoder(source, output, int(gpu_id), 0, 0, **config)
        if not hasattr(transcoder, "transcode_with_mux"):
            raise RuntimeError("PyNvVideoCodec transcoder does not expose transcode_with_mux")
        transcoder.transcode_with_mux()
    except Exception as e:
        _delete_outputs(output)
        sys.stderr.write(f"{_format_exception(e, 'input')}\n")
        return 1
    finally:
        del transcoder
        gc.collect()
    return 0


def _stringify_config(config: dict) -> dict[str, str]:
    return {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in config.items()}


def _format_exception(error: Exception, stage: str = "input") -> str:
    message = str(error).strip()
    normalized = message.casefold()
    if "timescale not set" in normalized:
        return (
            "PyNvVideoCodec could not read this MP4 stream timing metadata "
            "(timescale not set). The original file was kept and compression was skipped."
        )
    if "invalid data found when processing input" in normalized or "avformat_open_input" in normalized:
        if stage == "output":
            return "PyNvVideoCodec created an invalid output video at this CQ"
        return (
            "PyNvVideoCodec could not open the input video. "
            "The file is unsupported, corrupted, incomplete, or not a real video container."
        )
    if "error writing frame" in normalized:
        return "PyNvVideoCodec failed while writing encoded frames"
    return message or error.__class__.__name__


def _delete_outputs(output: str) -> None:
    template = Path(output)
    for path in _candidate_outputs_for_cleanup(template):
        path.unlink(missing_ok=True)


def _candidate_outputs(template: Path) -> list[Path]:
    candidates = [template]
    candidates.extend(path for path in template.parent.glob(f"{template.stem}*{template.suffix}") if path != template)
    return candidates


def _candidate_outputs_for_cleanup(template: Path) -> list[Path]:
    candidates = []
    seen: set[Path] = set()
    for path in _candidate_outputs(template):
        for candidate in (path, path.with_suffix(path.suffix + ".faststart")):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return candidates


def _resolve_output(output: str) -> Path:
    candidates = [path for path in _candidate_outputs(Path(output)) if path.is_file()]
    if not candidates:
        raise RuntimeError("PyNvVideoCodec did not create an output file")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _optimize_mp4_for_streaming(path: Path) -> Path:
    if path.suffix.casefold() not in {".mp4", ".m4v", ".mov"}:
        return path

    atoms = _read_top_level_atoms(path)
    moov = next((atom for atom in atoms if atom.type == b"moov"), None)
    first_mdat = next((atom for atom in atoms if atom.type == b"mdat"), None)
    if moov is None or first_mdat is None or moov.offset < first_mdat.offset:
        return path

    with path.open("rb") as input_file:
        input_file.seek(moov.offset)
        patched_moov = _patch_moov_offsets(input_file.read(moov.size), moov.size)

        faststart_path = path.with_suffix(path.suffix + ".faststart")
        with faststart_path.open("wb") as output_file:
            for atom in atoms:
                if atom == moov:
                    continue
                if atom == first_mdat:
                    output_file.write(patched_moov)
                input_file.seek(atom.offset)
                _copy_bytes(input_file, output_file, atom.size)

    faststart_path.replace(path)
    return path


def _retag_hevc_sample_entries(path: Path) -> None:
    if path.suffix.casefold() not in {".mp4", ".m4v", ".mov"}:
        return

    atoms = _read_top_level_atoms(path)
    moov = next((atom for atom in atoms if atom.type == b"moov"), None)
    if moov is None:
        return

    with path.open("rb") as input_file:
        input_file.seek(moov.offset)
        moov_bytes = input_file.read(moov.size)

    relative_offsets = list(_find_hev1_sample_entries(moov_bytes, 8, len(moov_bytes)))
    if not relative_offsets:
        return

    with path.open("r+b") as output_file:
        for relative_offset in relative_offsets:
            output_file.seek(moov.offset + relative_offset)
            output_file.write(b"hvc1")


def _find_hev1_sample_entries(data: bytes, start: int, end: int) -> Iterator[int]:
    position = start
    while position + 8 <= end:
        atom_size = int.from_bytes(data[position : position + 4], "big")
        atom_type = bytes(data[position + 4 : position + 8])
        header_size = 8
        if atom_size == 1:
            atom_size = int.from_bytes(data[position + 8 : position + 16], "big")
            header_size = 16
        elif atom_size == 0:
            atom_size = end - position
        if atom_size < header_size or position + atom_size > end:
            return

        content_start = position + header_size
        atom_end = position + atom_size
        if atom_type == b"stsd":
            yield from _walk_sample_entries_for_hev1(data, content_start + 8, atom_end)
        elif atom_type in _CONTAINER_ATOMS:
            yield from _find_hev1_sample_entries(data, content_start, atom_end)
        position = atom_end


def _walk_sample_entries_for_hev1(data: bytes, start: int, end: int) -> Iterator[int]:
    position = start
    while position + 8 <= end:
        entry_size = int.from_bytes(data[position : position + 4], "big")
        if entry_size < 8 or position + entry_size > end:
            return
        if bytes(data[position + 4 : position + 8]) == b"hev1":
            yield position + 4
        position += entry_size


class _Mp4Atom:
    def __init__(self, atom_type: bytes, offset: int, size: int) -> None:
        self.type = atom_type
        self.offset = offset
        self.size = size


def _read_top_level_atoms(path: Path) -> list[_Mp4Atom]:
    atoms = []
    file_size = path.stat().st_size
    with path.open("rb") as input_file:
        offset = 0
        while offset + 8 <= file_size:
            input_file.seek(offset)
            header = input_file.read(16)
            atom_size = int.from_bytes(header[0:4], "big")
            atom_type = header[4:8]
            header_size = 8
            if atom_size == 1:
                atom_size = int.from_bytes(header[8:16], "big")
                header_size = 16
            elif atom_size == 0:
                atom_size = file_size - offset
            if atom_size < header_size or offset + atom_size > file_size:
                raise RuntimeError(f"Invalid MP4 atom {atom_type!r} at offset {offset}")
            atoms.append(_Mp4Atom(atom_type, offset, atom_size))
            offset += atom_size
    return atoms


def _patch_moov_offsets(moov: bytes, offset_adjustment: int) -> bytes:
    patched = bytearray(moov)
    _patch_child_offsets(patched, 8, len(patched), offset_adjustment)
    return bytes(patched)


_CONTAINER_ATOMS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"dinf"}


def _patch_child_offsets(data: bytearray, start: int, end: int, offset_adjustment: int) -> None:
    position = start
    while position + 8 <= end:
        atom_size = int.from_bytes(data[position : position + 4], "big")
        atom_type = bytes(data[position + 4 : position + 8])
        header_size = 8
        if atom_size == 1:
            atom_size = int.from_bytes(data[position + 8 : position + 16], "big")
            header_size = 16
        elif atom_size == 0:
            atom_size = end - position
        if atom_size < header_size or position + atom_size > end:
            return

        content_start = position + header_size
        atom_end = position + atom_size
        if atom_type == b"stco":
            _patch_stco(data, content_start, atom_end, offset_adjustment)
        elif atom_type == b"co64":
            _patch_co64(data, content_start, atom_end, offset_adjustment)
        elif atom_type in _CONTAINER_ATOMS:
            _patch_child_offsets(data, content_start, atom_end, offset_adjustment)
        position = atom_end


def _patch_stco(data: bytearray, content_start: int, atom_end: int, offset_adjustment: int) -> None:
    entry_count_offset = content_start + 4
    entries_start = content_start + 8
    if entries_start > atom_end:
        return
    entry_count = int.from_bytes(data[entry_count_offset:entries_start], "big")
    for index in range(entry_count):
        entry_offset = entries_start + index * 4
        if entry_offset + 4 > atom_end:
            return
        new_offset = int.from_bytes(data[entry_offset : entry_offset + 4], "big") + offset_adjustment
        if new_offset > 0xFFFFFFFF:
            raise RuntimeError("Cannot faststart MP4 because stco offsets overflow 32-bit range")
        data[entry_offset : entry_offset + 4] = new_offset.to_bytes(4, "big")


def _patch_co64(data: bytearray, content_start: int, atom_end: int, offset_adjustment: int) -> None:
    entry_count_offset = content_start + 4
    entries_start = content_start + 8
    if entries_start > atom_end:
        return
    entry_count = int.from_bytes(data[entry_count_offset:entries_start], "big")
    for index in range(entry_count):
        entry_offset = entries_start + index * 8
        if entry_offset + 8 > atom_end:
            return
        new_offset = int.from_bytes(data[entry_offset : entry_offset + 8], "big") + offset_adjustment
        data[entry_offset : entry_offset + 8] = new_offset.to_bytes(8, "big")


def _copy_bytes(input_file, output_file, count: int) -> None:
    remaining = count
    while remaining:
        chunk = input_file.read(min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("Unexpected EOF while optimizing MP4 metadata")
        output_file.write(chunk)
        remaining -= len(chunk)


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
