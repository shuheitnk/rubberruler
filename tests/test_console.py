import io

import pytest
from rich.console import Console

from rubberruler import _console
from rubberruler._console import NO_PROGRESS, progress, progress_line, say, stage


class Screen:
    """A terminal in the given encoding that keeps what rich draws on it. It writes
    strictly, so a character that the encoding lacks fails the test."""

    def __init__(self, encoding: str = "utf-8") -> None:
        self.bytes = io.BytesIO()
        self.file = io.TextIOWrapper(
            self.bytes, encoding=encoding, errors="strict", write_through=True
        )

    def console(self) -> Console:
        return Console(
            file=self.file,
            force_terminal=True,
            force_interactive=True,
            color_system=None,
            width=120,
            highlight=False,
            soft_wrap=True,
        )

    def text(self) -> str:
        return self.bytes.getvalue().decode(self.file.encoding)


def run_two_stages() -> None:
    with stage("[1/4] Preparing instances"):
        pass
    with progress("[2/4] Measuring performance", 3) as advance:
        advance(3)


def counts(err: str) -> list[str]:
    """The "done/total" of each line of progress in a log."""
    return [line.split()[2] for line in err.splitlines()]


def test_in_a_terminal_each_stage_ends_with_a_check_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a terminal, rich draws the stages: a spinner and a bar while they run, and a
    check mark when they end."""
    screen = Screen()
    monkeypatch.setattr(_console, "stderr_console", screen.console)
    run_two_stages()
    text = screen.text()
    assert "\u2714 [1/4] Preparing instances" in text
    assert "\u2714 [2/4] Measuring performance" in text
    assert "3/3" in text


def test_a_terminal_without_unicode_gets_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal in cp932 receives an ASCII spinner and bar, and no check mark."""
    screen = Screen("cp932")
    monkeypatch.setattr(_console, "stderr_console", screen.console)
    run_two_stages()
    text = screen.text()
    assert text.isascii()
    assert "[2/4] Measuring performance" in text


def test_in_a_log_progress_is_written_at_most_every_30_seconds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When stderr is not a terminal, such as a log file, a stage writes its progress
    when it starts, after an update at least 30 seconds after the last line, and when
    it ends."""
    with progress("stage", 3) as advance:
        for _ in range(3):
            advance(1)
    assert counts(capsys.readouterr().err) == ["0/3", "3/3"]
    monkeypatch.setattr(_console, "LOG_SECONDS", 0.0)
    with progress("stage", 3) as advance:
        for _ in range(3):
            advance(1)
    assert counts(capsys.readouterr().err) == ["0/3", "1/3", "2/3", "3/3"]


def test_in_a_log_the_count_so_far_is_written_on_an_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On Ctrl-C, the log ends with how far the stage came."""
    with pytest.raises(KeyboardInterrupt), progress("stage", 3) as advance:
        advance(1)
        raise KeyboardInterrupt
    assert counts(capsys.readouterr().err) == ["0/3", "1/3"]


@pytest.mark.parametrize(
    ("done", "elapsed", "line"),
    [
        (0, 0.0, "stage   0% 0/2,880 [00:00<?]"),
        (1152, 5.0, "stage  40% 1,152/2,880 [00:05<00:08]"),
        (1440, 3725.0, "stage  50% 1,440/2,880 [1:02:05<1:02:05]"),
    ],
)
def test_the_progress_line(done: int, elapsed: float, line: str) -> None:
    """The percentage, the count, the time so far, and the time left."""
    assert progress_line("stage", done, 2880, elapsed) == line


def test_rubberruler_no_progress_hides_the_bars(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With RUBBERRULER_NO_PROGRESS set, a stage writes only its name; 0 keeps the
    bars."""
    monkeypatch.setenv(NO_PROGRESS, "1")
    with progress("stage", 3) as advance:
        advance(3)
    assert capsys.readouterr().err == "stage\n"
    monkeypatch.setenv(NO_PROGRESS, "0")
    with progress("stage", 3) as advance:
        advance(3)
    assert counts(capsys.readouterr().err) == ["0/3", "3/3"]


def test_a_message_is_one_line_as_it_is(capsys: pytest.CaptureFixture[str]) -> None:
    """A message is not wrapped at the width of the console, and brackets in it, such
    as in a path, are not taken as rich's markup."""
    text = f"rubberruler: figure saved to [1/4]/{'x' * 200}/rubber_ruler_test.png"
    say(text)
    assert capsys.readouterr().err == text + "\n"
