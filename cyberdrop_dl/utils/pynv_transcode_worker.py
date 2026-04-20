from __future__ import annotations

import gc
import json
import sys
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

    transcoder = None
    try:
        transcoder = PyNvVideoCodec.Transcoder(source, output, int(gpu_id), 0, 0, **config)
        if hasattr(transcoder, "transcode_with_mux"):
            transcoder.transcode_with_mux()
        elif hasattr(transcoder, "transcode"):
            transcoder.transcode()
        else:
            raise RuntimeError("PyNvVideoCodec transcoder does not expose a whole-file transcode method")
    finally:
        del transcoder
        gc.collect()
    return 0


def _stringify_config(config: dict) -> dict[str, str]:
    return {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in config.items()}


if __name__ == "__main__":
    raise SystemExit(main())
