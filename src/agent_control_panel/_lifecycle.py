"""Background threads for heartbeats and flushes, a time-bounded stop (also at exit), and fork safety."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
import weakref

from ._control import ControlState
from ._reporter import Reporter

_instances: weakref.WeakSet[Background] = weakref.WeakSet()


class Background:
    """Daemon threads never keep the process alive; ``stop()`` and interpreter exit flush with a deadline."""

    _control: ControlState
    _reporter: Reporter
    _log: logging.Logger

    def __init__(self, heartbeat_seconds: float, flush_seconds: float) -> None:
        self._heartbeat_seconds = heartbeat_seconds
        self._flush_seconds = flush_seconds
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._started = False
        self._restart_after_fork = False
        self._threads: list[threading.Thread] = []
        # Set once a heartbeat has completed since start(); the first runs wait for it (see AcpClient).
        self._first_heartbeat = threading.Event()
        self._startup_wait_pending = False
        _instances.add(self)

    def _heartbeat(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        """Start heartbeats and periodic flushes on daemon threads (they never keep the process alive)."""
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._closing.clear()
            self._wake.clear()
            self._first_heartbeat.clear()
            self._startup_wait_pending = True
            self._threads = [
                threading.Thread(target=self._heartbeat_loop, name="acp-heartbeat", daemon=True),
                threading.Thread(target=self._flush_loop, name="acp-flush", daemon=True),
            ]
            for thread in self._threads:
                thread.start()
            atexit.register(self._atexit)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background threads and send what is buffered, giving up after ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        with self._state_lock:
            self._started = False
            threads, self._threads = self._threads, []
        self._closing.set()
        self._wake.set()
        self._first_heartbeat.set()  # nothing waits for a heartbeat that will not come
        atexit.unregister(self._atexit)
        # On a helper thread so that nothing (a hung connection, a slow DNS lookup) holds stop() or the
        # interpreter's exit past the timeout.
        flushed: list[bool] = []
        finisher = threading.Thread(
            target=lambda: flushed.append(self._reporter.flush(deadline)), daemon=True
        )
        try:
            finisher.start()
        except RuntimeError:  # threads can no longer be started (interpreter shutting down)
            flushed.append(self._reporter.flush(deadline))
        else:
            finisher.join(max(0.0, deadline - time.monotonic()))
        if flushed != [True]:
            self._log.warning("stop: buffered runs could not be sent within %.1f s", timeout)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, min(1.0, deadline - time.monotonic())))

    def flush(self) -> None:
        """Send buffered runs now (blocking). Never raises."""
        self._reporter.flush()

    def _atexit(self) -> None:
        self.stop(timeout=5.0)

    def _after_fork_in_child(self) -> None:
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._first_heartbeat = threading.Event()
        self._control.reset_lock_after_fork()
        self._reporter.reset_after_fork()
        self._restart_after_fork = self._started
        self._started = False
        self._threads = []

    def _resume_after_fork(self) -> None:
        if self._restart_after_fork:
            self._restart_after_fork = False
            self.start()

    def _heartbeat_loop(self) -> None:
        while True:
            try:
                self._heartbeat()
            finally:
                self._first_heartbeat.set()
            if self._closing.wait(self._heartbeat_seconds):
                return

    def _flush_loop(self) -> None:
        while not self._closing.is_set():
            self._wake.wait(self._flush_seconds)
            self._wake.clear()
            if self._closing.is_set():
                return
            self._reporter.flush()

    def _wake_flusher(self) -> None:
        if any(t.is_alive() for t in self._threads):
            self._wake.set()
        else:
            threading.Thread(target=self._reporter.flush, name="acp-flush-once", daemon=True).start()


def _after_fork_in_child() -> None:
    for instance in list(_instances):
        instance._after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
