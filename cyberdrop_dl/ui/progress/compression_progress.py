from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress

if TYPE_CHECKING:
    from pathlib import Path

    from cyberdrop_dl.managers.manager import Manager


class CompressionProgress:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        self._queue_progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "{task.completed:,}",
        )
        self._file_info = Progress("{task.description}")
        self._header_task = self._file_info.add_task("")
        self._current_task = self._file_info.add_task("")
        self._pending_task = self._queue_progress.add_task("[cyan]Pending", total=None)
        self._compressed_task = self._queue_progress.add_task("[green]Compressed", total=None)
        self._skipped_task = self._queue_progress.add_task("[yellow]Skipped", total=None)
        self._failed_task = self._queue_progress.add_task("[red]Failed", total=None)
        self._current_files: dict[int, str] = {}
        self._pending = 0

    def get_renderable(self) -> Panel:
        return Panel(
            Group(self._file_info, self._queue_progress),
            title=f"Compression (config: {self.manager.config_manager.loaded_config})",
            border_style="magenta",
            padding=(1, 1),
        )

    def set_pending_count(self, count: int) -> None:
        self._pending = max(0, count)
        self._queue_progress.update(self._pending_task, completed=self._pending)

    def increment_pending(self, delta: int = 1) -> None:
        self.set_pending_count(self._pending + delta)

    def set_current(self, worker_id: int, path: Path | None) -> None:
        if path is None:
            self._current_files.pop(worker_id, None)
        else:
            self._current_files[worker_id] = path.name
        self._refresh_current_display()

    def add_result(self, status: str) -> None:
        task = {
            "compressed": self._compressed_task,
            "skipped": self._skipped_task,
            "failed": self._failed_task,
        }.get(status)
        if task is not None:
            self._queue_progress.advance(task, 1)

    def _refresh_current_display(self) -> None:
        if not self._current_files:
            self._file_info.update(self._header_task, description="")
            self._file_info.update(self._current_task, description="[dim]Idle")
            return
        self._file_info.update(self._header_task, description="[green]Currently compressing:")
        lines = [
            f"  [cyan]#{wid}[/cyan] [blue]{escape(name)}"
            for wid, name in sorted(self._current_files.items())
        ]
        self._file_info.update(self._current_task, description="\n".join(lines))
