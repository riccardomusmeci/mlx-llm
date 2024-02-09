import contextlib
import datetime
import time
from typing import Callable, Optional

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"


def now() -> str:
    """Return current time in the format: %Y-%m-%d-%H-%M-%S.

    Returns:
        str: current time
    """
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)


class Timing(contextlib.ContextDecorator):
    """Timing Context Decorator.

    Args:
        prefix (str, optional): name of activity to time. Defaults to "".
        on_exit (Optional[Callable]): function to call on exit. Defaults to None.
        enabled (bool, optional): whether to enable timing. Defaults to True.
    """

    def __init__(self, prefix: str = "", on_exit: Optional[Callable] = None, enabled: bool = True) -> None:
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled

    def __enter__(self) -> None:
        """Start timing."""
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc) -> None:
        """End timing and print elapsed time."""
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            print(f"{self.prefix}: {self.et*1e-9:.2f} s" + (self.on_exit(self.et) if self.on_exit else ""))
