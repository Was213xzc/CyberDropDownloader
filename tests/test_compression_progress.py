from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.layout import Layout

from cyberdrop_dl.ui.progress.compression_progress import CompressionProgress


def _make_manager(*, portrait: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_args=SimpleNamespace(cli_only_args=SimpleNamespace(portrait=portrait)),
        config_manager=SimpleNamespace(loaded_config="Default"),
    )


def _render_text(progress: CompressionProgress) -> str:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=20),
        Layout(progress.get_renderable(), name="Compression", ratio=8, minimum_size=5),
    )
    stream = StringIO()
    console = Console(file=stream, width=180, height=18, force_terminal=False, legacy_windows=False)
    console.print(layout)
    return stream.getvalue()


def test_compression_progress_renders_active_bars_and_summary_in_constrained_layout() -> None:
    progress = CompressionProgress(_make_manager())

    progress.set_current(1, Path("foo.mp4"))
    progress.start_task(1, Path("foo.mp4"), 100)
    progress.update_task(1, 25, 100)

    progress.set_current(2, Path("bar.mp4"))
    progress.start_task(2, Path("bar.mp4"), 200)
    progress.update_task(2, 40, 200)

    progress.set_pending_count(11)

    text = _render_text(progress)

    assert "foo.mp4" in text
    assert "25.00%" in text
    assert "bar.mp4" in text
    assert "20.00%" in text
    assert "Active: 2" in text
    assert "Pending: 11" in text


def test_compression_progress_shows_result_totals_in_summary() -> None:
    progress = CompressionProgress(_make_manager())

    progress.set_pending_count(3)
    progress.add_result("compressed")
    progress.add_result("skipped")
    progress.add_result("failed")

    text = _render_text(progress)

    assert "Compressed: 1" in text
    assert "Skipped: 1" in text
    assert "Failed: 1" in text
    assert "Pending: 3" in text
