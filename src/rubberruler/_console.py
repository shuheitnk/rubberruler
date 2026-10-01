"""What rubberruler writes on stderr while it runs: the plan, the stages, and their
progress.

In a terminal or a notebook, rich draws each stage with a spinner while it runs, a bar
if it counts, and a check mark when it ends. Elsewhere, such as in a log file, a stage
writes plain lines: its name, or its progress when it starts, at most every
LOG_SECONDS, and when it ends.
"""

import contextlib
import os
import sys
import time
from collections.abc import Callable, Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

# How often a stage writes its progress when stderr is not a terminal, such as a log.
LOG_SECONDS = 30.0
# Any value but 0 hides the progress bars; the names of the stages are still written.
NO_PROGRESS = "RUBBERRULER_NO_PROGRESS"
CHECK_MARK = "\u2714"
PREFIX = "rubberruler:"


def stderr_console() -> Console:
    """A console on sys.stderr as it is now: pytest and users may replace it."""
    return Console(file=sys.stderr, highlight=False, soft_wrap=True)


def drawn(console: Console) -> bool:
    """Whether rich draws on the console: a terminal or a notebook."""
    return console.is_jupyter or console.is_interactive


def _unicode(console: Console) -> bool:
    return console.encoding.lower().startswith("utf")


def _spinner(console: Console) -> str:
    return "dots" if _unicode(console) else "line"


def _check_mark(console: Console) -> Text:
    return Text(CHECK_MARK if _unicode(console) else " ", style="green")


def say(text: str, style: str = "") -> None:
    """Write a message on stderr; in a terminal, its "rubberruler:" is in bold."""
    line = Text(text, style=style)
    line.highlight_words([PREFIX], style="bold")
    stderr_console().print(line)


@contextlib.contextmanager
def stage(name: str) -> Generator[None, None, None]:
    """A stage without a bar. Drawn: a spinner while it runs, then a check mark.
    Otherwise: its name when it starts."""
    console = stderr_console()
    if not drawn(console):
        console.print(Text(name))
        yield
        return
    with console.status(Text(name, style="bold"), spinner=_spinner(console)):
        yield
    console.print(Text.assemble(_check_mark(console), " ", (name, "bold")))


@contextlib.contextmanager
def progress(
    description: str, total: int
) -> Generator[Callable[[int], object], None, None]:
    """Show the progress of a stage on stderr and give its update function.

    With nothing to count, or with RUBBERRULER_NO_PROGRESS set, it is a stage without
    a bar.
    """
    if total == 0 or os.environ.get(NO_PROGRESS, "0") != "0":
        with stage(description):
            yield lambda count: None
        return
    console = stderr_console()
    if not drawn(console):
        with _logged_progress(console, description, total) as advance:
            yield advance
        return
    with Progress(*_columns(console), console=console) as bar:
        task = bar.add_task(description, total=total)
        yield lambda count: bar.advance(task, count)


def _columns(console: Console) -> tuple[ProgressColumn, ...]:
    """The columns of a bar, which fit in 80 characters: the time left while the
    stage runs, and the time it took when it ends."""
    return (
        SpinnerColumn(_spinner(console), finished_text=_check_mark(console)),
        TextColumn("{task.description}", style="bold", markup=False),
        BarColumn(bar_width=20),
        TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f}", markup=False),
        TimeRemainingColumn(elapsed_when_finished=True),
    )


@contextlib.contextmanager
def _logged_progress(
    console: Console, description: str, total: int
) -> Generator[Callable[[int], object], None, None]:
    """Plain lines of progress: when the stage starts, after an update at least
    LOG_SECONDS after the last line, and when it ends (also on an exception) unless
    the last line shows the same count."""
    start = time.monotonic()
    last = start
    done = 0
    shown = 0

    def write(now: float) -> None:
        nonlocal shown
        shown = done
        console.print(Text(progress_line(description, done, total, now - start)))

    def advance(count: int) -> None:
        nonlocal done, last
        done += count
        now = time.monotonic()
        if now - last >= LOG_SECONDS:
            last = now
            write(now)

    write(start)
    try:
        yield advance
    finally:
        if done != shown:
            write(time.monotonic())


def progress_line(description: str, done: int, total: int, elapsed: float) -> str:
    """ "description  40% 1,152/2,880 [00:05<00:07]"; the time left is ? before the
    first update."""
    left = "?" if done == 0 else _clock(elapsed / done * (total - done))
    percentage = 100 * done / total
    return (
        f"{description} {percentage:3.0f}% {done:,}/{total:,} "
        f"[{_clock(elapsed)}<{left}]"
    )


def _clock(seconds: float) -> str:
    minutes, whole = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{whole:02d}"
    return f"{minutes:02d}:{whole:02d}"
