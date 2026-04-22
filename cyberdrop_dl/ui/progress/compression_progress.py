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
        self._active_progress = Progress(*use_columns, expand=True)
        self._queue_progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>6.2f}%",
            "━",
            "{task.completed:,}",
            expand=True,
        )
        self._pending_task = self._queue_progress.add_task("[cyan]Pending", total=0, completed=0)
        self._compressed_task = self._queue_progress.add_task("[green]Compressed", total=0, completed=0)
        self._skipped_task = self._queue_progress.add_task("[yellow]Skipped", total=0, completed=0)
        self._failed_task = self._queue_progress.add_task("[red]Failed", total=0, completed=0)
        self._worker_tasks: dict[int, TaskID] = {}
        self._pending = 0
        self._compressed = 0
        self._skipped = 0
        self._failed = 0
        self._total = 0
        self._panel = Panel(
            Group(self._active_progress, self._queue_progress),
            title=f"Compression (config: {self.manager.config_manager.loaded_config})",
            border_style="magenta",
            padding=(0, 1),
            subtitle=f"Total: [white]{self._total:,}",
        )

    def get_renderable(self) -> Panel:
        return self._panel

    def set_pending_count(self, count: int) -> None:
        self._pending = max(0, count)
        self._refresh_queue_stats()

    def increment_pending(self, delta: int = 1) -> None:
        self.set_pending_count(self._pending + delta)

    def increment_total(self, delta: int = 1) -> None:
        self._total = max(0, self._total + delta)
        self._refresh_queue_stats()

    def set_current(self, worker_id: int, path: Path | None) -> None:
        if path is None:
            task_id = self._worker_tasks.pop(worker_id, None)
            if task_id is not None:
                self._active_progress.remove_task(task_id)
        else:
            self._ensure_worker_task(worker_id, path)

    def add_result(self, status: str) -> None:
        if status == "compressed":
            self._compressed += 1
        elif status == "skipped":
            self._skipped += 1
        elif status == "failed":
            self._failed += 1
        else:
            return
        self._refresh_queue_stats()

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
        return f"[cyan]#{worker_id}[/cyan] [blue]{escape(adjust_title(clean_name, length=40))}[/blue]"

    def _refresh_queue_stats(self) -> None:
        # Total can lag briefly during transitions — grow it to match whatever is
        # currently accounted for so bar percentages never exceed 100%.
        accounted = self._pending + self._compressed + self._skipped + self._failed
        if accounted > self._total:
            self._total = accounted
        total = self._total
        self._queue_progress.update(self._pending_task, total=total, completed=self._pending)
        self._queue_progress.update(self._compressed_task, total=total, completed=self._compressed)
        self._queue_progress.update(self._skipped_task, total=total, completed=self._skipped)
        self._queue_progress.update(self._failed_task, total=total, completed=self._failed)
        self._panel.subtitle = f"Total: [white]{total:,}"
