"""The one exception that rubberruler raises."""


class RubberRulerError(Exception):
    """Raised when rubberruler cannot go on.

    The message has three lines: what happened and where, ``Why:`` it is a
    problem, and ``Fix:`` how to fix it.
    """


RubberRulerError.__module__ = "rubberruler"


def three_lines(what: str, *, why: str, fix: str) -> str:
    """The three-line message: what happened and where, Why:, and Fix:."""
    return f"{what}\nWhy: {why}\nFix: {fix}"


def error(what: str, *, why: str, fix: str) -> RubberRulerError:
    """Return a RubberRulerError with the message of ``three_lines``."""
    return RubberRulerError(three_lines(what, why=why, fix=fix))


def check_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
    example: str,
) -> int:
    """Return ``value`` if it is an int in [minimum, maximum]; otherwise stop."""
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
        and (maximum is None or value <= maximum)
    ):
        return value
    if maximum is None:
        allowed = f"an int of at least {minimum}"
    else:
        allowed = f"an int from {minimum} to {maximum}"
    raise error(
        f"{name}={value!r} is not allowed.",
        why=f"{name} must be {allowed}.",
        fix=f"pass {example}.",
    )


def detail(exception: BaseException) -> str:
    """The first line of the message of an exception, for the first line of a
    three-line message: at most 120 characters, in ASCII, or the name of its
    type if the message is empty. The error may be printed without its traceback."""
    lines = str(exception).strip().splitlines()
    text = lines[0] if lines else type(exception).__name__
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return text if len(text) <= 120 else text[:117] + "..."
