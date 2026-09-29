"""Webhook signature v2 (same contract as the Node SDK and the ACP server).

The signed payload is ``f"{timestamp}.{nonce}.{raw_body}"`` (HMAC-SHA256, hex). The timestamp bounds the
window (300 s either way) and the nonce is single-use, so a captured request cannot be replayed.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

SIG_HEADER = "x-acp-signature"
TS_HEADER = "x-acp-timestamp"
NONCE_HEADER = "x-acp-nonce"
PROTOCOL_HEADER = "x-acp-protocol"
PROTOCOL_VERSION = "1"

MAX_SKEW_SECONDS = 300
"""Reject anything older or newer than this (seconds)."""

VerifyFailure = Literal[
    "missing_headers", "bad_timestamp", "stale_timestamp", "replayed_nonce", "bad_signature"
]

_TIMESTAMP_RE = re.compile(r"[0-9]{1,15}")
_SIGNATURE_RE = re.compile(r"[0-9a-fA-F]{64}")


class HeadersLike(Protocol):
    """Anything with ``get`` and ``keys``: a dict, or the request headers of FastAPI, Flask or Django."""

    def get(self, key: str, /) -> Any: ...

    def keys(self) -> Iterable[Any]: ...


@dataclass(frozen=True)
class SignedPayload:
    headers: dict[str, str]
    timestamp: int
    nonce: str
    signature: str


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: VerifyFailure | None = None
    nonce: str | None = None


def _as_bytes(raw_body: str | bytes) -> bytes:
    return raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)


def _mac(secret: str, timestamp: int, nonce: str, body: bytes) -> bytes:
    message = f"{timestamp}.{nonce}.".encode() + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()


def sign_payload(
    secret: str,
    raw_body: str | bytes,
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> SignedPayload:
    """Sign a body the way ACP does. Useful in tests of your webhook endpoint."""
    ts = int(time.time()) if timestamp is None else timestamp
    n = nonce if nonce is not None else str(uuid.uuid4())
    signature = _mac(secret, ts, n, _as_bytes(raw_body)).hex()
    return SignedPayload(
        headers={
            SIG_HEADER: signature,
            TS_HEADER: str(ts),
            NONCE_HEADER: n,
            PROTOCOL_HEADER: PROTOCOL_VERSION,
            "content-type": "application/json",
        },
        timestamp=ts,
        nonce=n,
        signature=signature,
    )


def _first_text(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        return _first_text(value[0]) if value else None
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return value if isinstance(value, str) else None


def get_header(headers: HeadersLike, name: str) -> str | None:
    """Case-insensitive header lookup; list values use the first entry."""
    value = _first_text(headers.get(name))
    if value is not None:
        return value
    # keys(), not iteration: Werkzeug (Flask) headers iterate as (name, value) pairs.
    for key in headers.keys():  # noqa: SIM118
        key_text = key.decode("latin-1") if isinstance(key, bytes) else key
        if isinstance(key_text, str) and key_text.lower() == name:
            return _first_text(headers.get(key))
    return None


def verify_signature(
    *,
    secrets: Iterable[str | None],
    raw_body: str | bytes,
    headers: HeadersLike,
    now: int | None = None,
    seen_nonce: Callable[[str], bool] | None = None,
) -> VerifyResult:
    """Verify an inbound ACP webhook. Checks run in the same order as the Node SDK, so the reasons match."""
    signature = get_header(headers, SIG_HEADER)
    ts_raw = get_header(headers, TS_HEADER)
    nonce = get_header(headers, NONCE_HEADER)
    if not signature or not ts_raw or not nonce:
        return VerifyResult(ok=False, reason="missing_headers")

    # ACP always sends a whole number of seconds; anything else is not an ACP request.
    if not _TIMESTAMP_RE.fullmatch(ts_raw):
        return VerifyResult(ok=False, reason="bad_timestamp")
    ts = int(ts_raw)

    current = int(time.time()) if now is None else now
    if abs(current - ts) > MAX_SKEW_SECONDS:
        return VerifyResult(ok=False, reason="stale_timestamp")

    if seen_nonce is not None and seen_nonce(nonce):
        return VerifyResult(ok=False, reason="replayed_nonce")

    if not _SIGNATURE_RE.fullmatch(signature):
        return VerifyResult(ok=False, reason="bad_signature")
    expected = bytes.fromhex(signature)
    body = _as_bytes(raw_body)
    for secret in secrets:
        if not secret:
            continue
        if hmac.compare_digest(_mac(secret, ts, nonce, body), expected):
            return VerifyResult(ok=True, nonce=nonce)
    return VerifyResult(ok=False, reason="bad_signature")


class NonceCache:
    """Bounded LRU of recently seen nonces. Not thread-safe on its own; the client guards it."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max = max_size
        self._seen: OrderedDict[str, None] = OrderedDict()

    def has(self, nonce: str) -> bool:
        return nonce in self._seen

    def add(self, nonce: str) -> None:
        if nonce in self._seen:
            return
        self._seen[nonce] = None
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)
