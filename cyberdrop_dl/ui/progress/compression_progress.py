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
        self._summary_progress = Progress("{task.description}", expand=True)
        self._summary_task = self._summary_progress.add_task("[dim]Idle")
        self._worker_tasks: dict[int, TaskID] = {}
        self._pending = 0
        self._results = {"compressed": 0, "skipped": 0, "failed": 0}

    def get_renderable(self) -> Panel:
        return Panel(
            Group(self._active_progress, self._summary_progress),
            title=f"Compression (config: {self.manager.config_manager.loaded_config})",
            border_style="magenta",
            padding=(0, 1),
        )

    def set_pending_count(self, count: int) -> None:
        self._pending = max(0, count)
        self._refresh_summary()

    def increment_pending(self, delta: int = 1) -> None:
        self.set_pending_count(self._pending + delta)

    def set_current(self, worker_id: int, path: Path | None) -> None:
        if path is None:
            task_id = self._worker_tasks.pop(worker_id, None)
            if task_id is not None:
                self._active_progress.remove_task(task_id)
        else:
            self._ensure_worker_task(worker_id, path)
        self._refresh_summary()

    def add_result(self, status: str) -> None:
        if status in self._results:
            self._results[status] += 1
            self._refresh_summary()

    def start_task(self, worker_id: int, path: Path, total: int) -> None:
        task_id = self._ensure_worker_task(worker_id, path)
        self._active_progress.update(task_id, completed=0, total=max(total, 1), visible=True)
        self._refresh_summary()

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

    def _refresh_summary(self) -> None:
        active = len(self._worker_tasks)
        if active == 0 and self._pending == 0 and not any(self._results.values()):
            self._summary_progress.update(self._summary_task, description="[dim]Idle")
            return
        self._summary_progress.update(
            self._summary_task,
            description=(
                f"[cyan]Active:[/cyan] {active}  "
                f"[cyan]Pending:[/cyan] {self._pending}  "
                f"[green]Compressed:[/green] {self._results['compressed']}  "
                f"[yellow]Skipped:[/yellow] {self._results['skipped']}  "
                f"[red]Failed:[/red] {self._results['failed']}"
            ),
        )
