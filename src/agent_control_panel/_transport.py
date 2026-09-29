"""HTTP transport on the standard library.

Never follows redirects: the API key must not reach another host.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import Message
from typing import IO, Any

MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Request:
    method: str
    url: str
    body: bytes | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 10.0


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = b""
    location: str | None = None
    """The ``Location`` header of a 3xx reply (not followed)."""


Transport = Callable[[Request], Response]
"""Sends one request. Returns the reply for any HTTP status; raises only when no reply arrived."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def _read_capped(stream: Any) -> bytes:
    data: bytes = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("response too large")
    return data


def build_urllib_transport(user_agent: str) -> Transport:
    opener = urllib.request.build_opener(
        _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )

    def send(request: Request) -> Response:
        headers = {"User-Agent": user_agent, "Accept": "application/json", **request.headers}
        # create_client only accepts http(s) URLs, so no file: or custom schemes reach this.
        req = urllib.request.Request(  # noqa: S310
            request.url, data=request.body, headers=headers, method=request.method
        )
        try:
            with opener.open(req, timeout=request.timeout) as res:
                return Response(status=res.status, body=_read_capped(res))
        except urllib.error.HTTPError as err:
            try:
                location = err.headers.get("Location") if err.headers is not None else None
                try:
                    body = _read_capped(err) if err.fp is not None else b""
                except Exception:
                    body = b""
                return Response(status=err.code, body=body, location=location)
            finally:
                err.close()

    return send
