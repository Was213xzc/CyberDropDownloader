from __future__ import annotations

import contextlib
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

    try:
        _transcode_file(PyNvVideoCodec, source, output, int(gpu_id), config)
    except Exception as e:
        sys.stderr.write(f"{e}\n")
        return 1

    return 0


def _transcode_file(pynv_module, source: str, output: str, gpu_id: int, config: dict[str, str]) -> Path:
    _delete_outputs(output)
    transcoder = None
    try:
        transcoder = pynv_module.Transcoder(source, output, gpu_id, 0, 0, **config)
        if not hasattr(transcoder, "transcode_with_mux"):
            raise RuntimeError("PyNvVideoCodec transcoder does not expose transcode_with_mux")
        transcoder.transcode_with_mux()
    except Exception as e:
        _delete_outputs(output)
        raise RuntimeError(_format_exception(e, "input")) from e
    finally:
        del transcoder
        gc.collect()

    try:
        actual_output = _resolve_output(output)
        actual_output = _optimize_mp4_for_streaming(actual_output)
        _repair_sample_entry_for_target_codec(actual_output, str(config.get("codec", "")))
        _retag_hevc_sample_entries(actual_output)
        _validate_output(pynv_module, str(actual_output), gpu_id)
        return actual_output
    except Exception as e:
        _delete_outputs(output)
        raise RuntimeError(_format_exception(e, "output")) from e


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
    if (
        "invalid data found when processing input" in normalized
        or "avformat_open_input" in normalized
        or (stage == "output" and "ffmpegdemuxer" in normalized)
    ):
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
        with contextlib.suppress(OSError):
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


_HEVC_CODEC_NAMES = {"hevc", "h265"}
_AVC1_CONTAINER_ATOMS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _repair_sample_entry_for_target_codec(path: Path, target_codec: str) -> None:
    if _normalize_codec_name(target_codec) not in _HEVC_CODEC_NAMES:
        return
    if path.suffix.casefold() not in {".mp4", ".m4v", ".mov"}:
        return

    atoms = _read_top_level_atoms(path)
    moov = next((atom for atom in atoms if atom.type == b"moov"), None)
    mdat = next((atom for atom in atoms if atom.type == b"mdat"), None)
    if moov is None or mdat is None:
        return

    with path.open("rb") as input_file:
        input_file.seek(moov.offset)
        moov_bytes = bytearray(input_file.read(moov.size))
        input_file.seek(mdat.offset + 8)
        mdat_head = input_file.read(min(mdat.size - 8, 1024 * 1024))

    info = _find_avc1_sample_entry(moov_bytes)
    if info is None:
        return
    sample_entry_offset, _entry_size, avcc_offset, avcc_size, ancestor_offsets = info

    if avcc_size > 16:
        return

    vps, sps, pps = _extract_inline_hevc_param_sets(mdat_head)
    if not (vps and sps and pps):
        return

    hvcc_body = _build_hvcc_body(vps, sps, pps)
    hvcc_atom = (8 + len(hvcc_body)).to_bytes(4, "big") + b"hvcC" + hvcc_body
    size_delta = len(hvcc_atom) - avcc_size

    moov_bytes[sample_entry_offset + 4 : sample_entry_offset + 8] = b"hvc1"
    new_moov = bytearray(
        bytes(moov_bytes[:avcc_offset]) + hvcc_atom + bytes(moov_bytes[avcc_offset + avcc_size :])
    )

    for bump_offset in [*ancestor_offsets, sample_entry_offset]:
        current = int.from_bytes(new_moov[bump_offset : bump_offset + 4], "big")
        new_moov[bump_offset : bump_offset + 4] = (current + size_delta).to_bytes(4, "big")

    if moov.offset < mdat.offset:
        _patch_child_offsets(new_moov, 8, len(new_moov), size_delta)

    repair_path = path.with_suffix(path.suffix + ".repair")
    try:
        with path.open("rb") as input_file, repair_path.open("wb") as output_file:
            for atom in atoms:
                if atom is moov:
                    output_file.write(bytes(new_moov))
                else:
                    input_file.seek(atom.offset)
                    _copy_bytes(input_file, output_file, atom.size)
        repair_path.replace(path)
    except Exception:
        repair_path.unlink(missing_ok=True)
        raise


def _normalize_codec_name(codec: str) -> str:
    return codec.casefold().replace(".", "").replace("-", "").replace("_", "")


def _find_avc1_sample_entry(
    moov_bytes: bytes,
) -> tuple[int, int, int, int, list[int]] | None:
    return _walk_for_avc1(moov_bytes, 8, len(moov_bytes), [0])


def _walk_for_avc1(
    data: bytes, start: int, end: int, ancestor_offsets: list[int]
) -> tuple[int, int, int, int, list[int]] | None:
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
            return None
        content_start = position + header_size
        atom_end = position + atom_size
        if atom_type == b"stsd":
            result = _find_avc1_in_stsd(data, content_start + 8, atom_end)
            if result is not None:
                entry_offset, entry_size, avcc_offset, avcc_size = result
                return entry_offset, entry_size, avcc_offset, avcc_size, [*ancestor_offsets, position]
        elif atom_type in _AVC1_CONTAINER_ATOMS:
            nested = _walk_for_avc1(data, content_start, atom_end, [*ancestor_offsets, position])
            if nested is not None:
                return nested
        position = atom_end
    return None


def _find_avc1_in_stsd(
    data: bytes, start: int, end: int
) -> tuple[int, int, int, int] | None:
    position = start
    while position + 8 <= end:
        entry_size = int.from_bytes(data[position : position + 4], "big")
        if entry_size < 8 or position + entry_size > end:
            return None
        entry_type = bytes(data[position + 4 : position + 8])
        entry_end = position + entry_size
        if entry_type == b"avc1":
            avcc = _find_child_atom(data, position + 8 + 78, entry_end, b"avcC")
            if avcc is not None:
                avcc_offset, avcc_size = avcc
                return position, entry_size, avcc_offset, avcc_size
            return position, entry_size, entry_end, 0
        position += entry_size
    return None


def _find_child_atom(
    data: bytes, start: int, end: int, target: bytes
) -> tuple[int, int] | None:
    position = start
    while position + 8 <= end:
        atom_size = int.from_bytes(data[position : position + 4], "big")
        atom_type = bytes(data[position + 4 : position + 8])
        if atom_size < 8 or position + atom_size > end:
            return None
        if atom_type == target:
            return position, atom_size
        position += atom_size
    return None


def _extract_inline_hevc_param_sets(
    data: bytes,
) -> tuple[bytes | None, bytes | None, bytes | None]:
    vps: bytes | None = None
    sps: bytes | None = None
    pps: bytes | None = None
    position = 0
    for _ in range(32):
        if position + 4 > len(data):
            break
        nal_size = int.from_bytes(data[position : position + 4], "big")
        position += 4
        if nal_size <= 0 or position + nal_size > len(data):
            break
        nal = data[position : position + nal_size]
        position += nal_size
        if len(nal) < 2:
            continue
        nal_type = (nal[0] >> 1) & 0x3F
        if nal_type == 32 and vps is None:
            vps = bytes(nal)
        elif nal_type == 33 and sps is None:
            sps = bytes(nal)
        elif nal_type == 34 and pps is None:
            pps = bytes(nal)
        elif nal_type < 32:
            break
        if vps and sps and pps:
            break
    return vps, sps, pps


def _build_hvcc_body(vps: bytes, sps: bytes, pps: bytes) -> bytes:
    if len(sps) < 15:
        raise RuntimeError("HEVC SPS too short to derive profile_tier_level")

    num_temporal_layers = ((sps[2] >> 1) & 0x07) + 1
    temporal_id_nested = sps[2] & 0x01

    body = bytearray()
    body.append(1)
    body.extend(sps[3:15])
    body.extend(b"\xf0\x00")
    body.append(0xFC)
    body.append(0xFD)
    body.append(0xF8)
    body.append(0xF8)
    body.extend(b"\x00\x00")
    body.append(((num_temporal_layers & 0x07) << 3) | ((temporal_id_nested & 0x01) << 2) | 0x03)
    body.append(3)
    for nal_type, nal in ((32, vps), (33, sps), (34, pps)):
        body.append(nal_type)
        body.extend(b"\x00\x01")
        body.extend(len(nal).to_bytes(2, "big"))
        body.extend(nal)
    return bytes(body)


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
        duration = float(getattr(metadata, "duration", 0) or getattr(metadata, "duration_in_seconds", 0) or 0)
        if duration <= 0:
            raise RuntimeError("PyNvVideoCodec output validation failed: zero duration")
        _ = decoder[0]
    finally:
        del decoder
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
