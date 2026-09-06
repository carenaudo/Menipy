"""Cooperative cancellation, scoped to an execution thread, never serialized."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class AnalysisCancelled(BaseException):
    """Control flow, like asyncio.CancelledError; never an algorithm failure."""


class CancellationToken:
    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    def __deepcopy__(self, memo):
        # A copied Context belongs to the same job, including its stop request.
        return self

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise AnalysisCancelled()


_current: ContextVar[CancellationToken | None] = ContextVar(
    "analysis_cancellation", default=None
)


def check_cancelled():
    """No-op for ordinary headless calls outside a cancellation scope."""
    token = _current.get()
    if token is not None:
        token.check()


@contextmanager
def cancellation_scope(token):
    handle = _current.set(token)
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        _current.reset(handle)
