"""Run buffer, heartbeat counters and batched, retried flushes. Never raises into the caller."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

MAX_RETRIES = 4
MAX_BATCH_BYTES = 4_000_000
"""Keep each report request well under the ingest body limit (10 MB), as the Node SDK does."""

Post = Callable[[str, bytes, float], bool]
"""(path, body, timeout) -> accepted. Never raises."""


@dataclass(frozen=True)
class Counters:
    runs: int = 0
    tokens: int = 0
    errors: int = 0


def chunk_by_size(rows: list[str], max_bytes: int) -> list[list[str]]:
    """Consecutive groups of serialised rows under ``max_bytes`` each (a single larger row goes alone)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for row in rows:
        row_bytes = len(row.encode("utf-8")) + 1
        if current and size + row_bytes > max_bytes:
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += row_bytes
    if current:
        chunks.append(current)
    return chunks


def jitter_backoff(attempt: int) -> float:
    base: float = min(8.0, 0.25 * 2**attempt)
    return base + random.random() * base * 0.5  # noqa: S311 (retry jitter, not security)


class Reporter:
    def __init__(
        self,
        *,
        api_key: str,
        post: Post,
        flush_at: int,
        max_buffer: int,
        logger: logging.Logger,
        wake_flusher: Callable[[], None],
        wait: Callable[[float], bool],
    ) -> None:
        self._api_key = api_key
        self._post = post
        self._flush_at = flush_at
        self._max_buffer = max_buffer
        self._log = logger
        self._wake_flusher = wake_flusher
        self._wait = wait
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        # Runs are stored serialised: the batch is immutable once reported, and one run that cannot be
        # serialised is dropped on its own instead of failing its whole batch.
        self._buffer: deque[str] = deque()
        self._counters = Counters()
        self.dropped = 0

    def reset_after_fork(self) -> None:
        # The parent process still owns (and sends) what it had buffered.
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._buffer.clear()
        self._counters = Counters()

    @property
    def buffered(self) -> int:
        with self._lock:
            return len(self._buffer)

    def add(self, run: dict[str, object], tokens: int, is_error: bool) -> None:
        encoded = json.dumps(run, default=str, separators=(",", ":"))
        with self._lock:
            if len(self._buffer) >= self._max_buffer:
                self._buffer.popleft()
                self.dropped += 1
                self._log.warning("report buffer full; dropped oldest (total dropped=%d)", self.dropped)
            self._buffer.append(encoded)
            c = self._counters
            self._counters = Counters(c.runs + 1, c.tokens + tokens, c.errors + (1 if is_error else 0))
            full = len(self._buffer) >= self._flush_at
        if full:
            self._wake_flusher()

    def counters(self) -> Counters:
        with self._lock:
            return self._counters

    def commit_counters(self, sent: Counters) -> None:
        """Subtract what a successful heartbeat carried; runs reported meanwhile stay for the next one."""
        with self._lock:
            c = self._counters
            self._counters = Counters(c.runs - sent.runs, c.tokens - sent.tokens, c.errors - sent.errors)

    def flush(self, deadline: float | None = None) -> bool:
        """Send everything buffered. Flushes are serialised, so a flush in flight finishes first.

        With a ``deadline`` (``time.monotonic()``), gives up once it passes. Returns False when the flush
        could not start before the deadline.
        """
        if deadline is None:
            self._flush_lock.acquire()
        elif not self._flush_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return False
        try:
            with self._lock:
                batch = list(self._buffer)
                self._buffer.clear()
            if batch:
                self._send(batch, deadline)
            return True
        finally:
            self._flush_lock.release()

    def _drop(self, count: int) -> int:
        with self._lock:
            self.dropped += count
            return self.dropped

    def _requeue(self, batch: list[str]) -> None:
        with self._lock:
            self._buffer.extendleft(reversed(batch))
            while len(self._buffer) > self._max_buffer:
                self._buffer.popleft()
                self.dropped += 1

    def _send(self, batch: list[str], deadline: float | None) -> None:
        pending = len(batch)
        try:
            chunks = chunk_by_size(batch, MAX_BATCH_BYTES)
            for index, chunk in enumerate(chunks):
                if not self._send_chunk(chunk, deadline):
                    # The client is stopping: hand the rest to stop()'s own, time-bounded flush.
                    self._requeue([row for rest in chunks[index:] for row in rest])
                    return
                pending -= len(chunk)
        except Exception as err:
            self._drop(pending)
            self._log.warning("flush error (fail-open, dropped batch): %s", err)

    def _send_chunk(self, chunk: list[str], deadline: float | None) -> bool:
        """One report request with retries; a chunk that still fails is dropped. False: requeue (stopping)."""
        body = b'{"api_key":%s,"runs":[%s]}' % (json.dumps(self._api_key).encode(), ",".join(chunk).encode())
        for attempt in range(MAX_RETRIES + 1):
            timeout = 10.0
            if deadline is not None:
                timeout = min(timeout, deadline - time.monotonic())
                if timeout <= 0:
                    break
            if self._post("/api/report", body, timeout):
                return True
            if attempt == MAX_RETRIES:
                break
            pause = jitter_backoff(attempt)
            if deadline is not None and time.monotonic() + pause >= deadline:
                break
            if self._wait(pause) and deadline is None:
                return False
        total = self._drop(len(chunk))
        self._log.warning(
            "flush failed after retries; dropping %d runs (total dropped=%d)", len(chunk), total
        )
        return True
