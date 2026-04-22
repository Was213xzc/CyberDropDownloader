from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from cyberdrop_dl.ui.progress.deque_progress import adjust_title

if TYPE_CHECKING:
    from pathlib import Path

    from cyberdrop_dl.managers.manager import Manager


class CompressionProgress:
    def __init__(self, manager: Manager) -> None:
        self.manager = manager
        progress_columns = (SpinnerColumn(), "[progress.description]{task.description}", BarColumn(bar_width=None))
        horizontal_columns = (
            *progress_columns,
            "[progress.percentage]{task.percentage:>6.2f}%",
            "|",
            DownloadColumn(),
            "|",
            TransferSpeedColumn(),
            "|",
            TimeRemainingColumn(),
        )
        vertical_columns = (*progress_columns, DownloadColumn(), "|", TransferSpeedColumn())
        use_columns = horizontal_columns
        if manager.parsed_args.cli_only_args.portrait:
            use_columns = vertical_columns
        self._active_progress = Progress(*use_columns)
        self._queue_progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "{task.completed:,}",
        )
        self._status = Progress("{task.description}")
        self._status_task = self._status.add_task("[dim]Idle")
        self._pending_task = self._queue_progress.add_task("[cyan]Pending", total=None)
        self._compressed_task = self._queue_progress.add_task("[green]Compressed", total=None)
        self._skipped_task = self._queue_progress.add_task("[yellow]Skipped", total=None)
        self._failed_task = self._queue_progress.add_task("[red]Failed", total=None)
        self._current_files: dict[int, str] = {}
        self._worker_tasks: dict[int, TaskID] = {}
        self._pending = 0

    def get_renderable(self) -> Panel:
        return Panel(
            Group(self._status, self._active_progress, self._queue_progress),
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
            task_id = self._worker_tasks.pop(worker_id, None)
            if task_id is not None:
                self._active_progress.remove_task(task_id)
        else:
            self._current_files[worker_id] = path.name
            self._ensure_worker_task(worker_id, path)
        self._refresh_current_display()

    def add_result(self, status: str) -> None:
        task = {
            "compressed": self._compressed_task,
            "skipped": self._skipped_task,
            "failed": self._failed_task,
        }.get(status)
        if task is not None:
            self._queue_progress.advance(task, 1)

    def start_task(self, worker_id: int, path: Path, total: int) -> None:
        task_id = self._ensure_worker_task(worker_id, path)
        self._active_progress.update(task_id, completed=0, total=max(total, 1), visible=True)

    def update_task(self, worker_id: int, completed: int, total: int | None = None) -> None:
        task_id = self._worker_tasks.get(worker_id)
        if task_id is None:
            return

        kwargs = {"completed": max(0, completed), "visible": True}
        if total is not None:
            kwargs["total"] = max(total, 1)
        self._active_progress.update(task_id, **kwargs)

    def finish_task(self, worker_id: int) -> None:
        task_id = self._worker_tasks.get(worker_id)
        if task_id is None:
            return
        task = self._active_progress._tasks[task_id]
        total = int(task.total or 1)
        self._active_progress.update(task_id, completed=total, total=total, visible=True)

    def _ensure_worker_task(self, worker_id: int, path: Path) -> TaskID:
        task_id = self._worker_tasks.get(worker_id)
        description = self._task_description(worker_id, path.name)
        if task_id is None:
            task_id = self._active_progress.add_task(description, total=1, completed=0)
            self._worker_tasks[worker_id] = task_id
        else:
            self._active_progress.update(task_id, description=description, visible=True)
        return task_id

    def _task_description(self, worker_id: int, name: str) -> str:
        clean_name = name.encode("ascii", "ignore").decode().strip()
        return f"[cyan]#{worker_id}[/cyan] [blue]{escape(adjust_title(clean_name, length=40))}"

    def _refresh_current_display(self) -> None:
        if not self._current_files:
            self._status.update(self._status_task, description="[dim]Idle")
            return
        self._status.update(self._status_task, description="[green]Currently compressing:")
