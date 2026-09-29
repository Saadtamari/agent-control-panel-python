"""Talking to ACP: URL rules, fail-open requests and the warnings that tell users what to fix."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from ._transport import Request, Response, Transport

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def assert_secure_url(url: str, allow_insecure_http: bool = False) -> None:
    """Raise unless ``url`` is https, http to a loopback host, or http with the explicit dev opt-out."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc or not parts.hostname:
        raise ValueError(f"create_client: url is not a valid URL: {url}")
    if parts.scheme == "https":
        return
    if parts.scheme != "http":
        raise ValueError(f"create_client: url must use https: (got {parts.scheme}:)")
    if parts.hostname.lower() in LOOPBACK_HOSTS or allow_insecure_http:
        return
    raise ValueError(
        f'create_client: refusing insecure http:// url for host "{parts.hostname}" - the API key would be '
        "sent in cleartext. Use https://, or set allow_insecure_http=True for local development only."
    )


def clean(text: str, limit: int = 200) -> str:
    """Safe to log: no control characters (no forged log lines), bounded length."""
    return _CONTROL_CHARS.sub(" ", text)[:limit]


class Api:
    def __init__(self, url: str, api_key: str, transport: Transport, logger: logging.Logger) -> None:
        self.url = url
        self._api_key = api_key
        self._transport = transport
        self._log = logger
        parts = urlsplit(url)
        self._display_url = urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))

    def warn_status(self, label: str, path: str, res: Response, suffix: str = "") -> None:
        if 300 <= res.status < 400:
            if res.location:
                target = clean(urljoin(self.url + path, res.location))
                moved_to = target[: -len(path)] if target.endswith(path) else target
                self._log.warning(
                    "%s -> %d: ACP moved to %s; update your url (currently %s). Redirects are not "
                    "followed, so the API key is never sent to another host.%s",
                    label, res.status, moved_to, self._display_url, suffix,
                )  # fmt: skip
            else:
                self._log.warning(
                    "%s -> %d: redirect without a Location; check your url.%s", label, res.status, suffix
                )
        elif res.status == 401:
            self._log.warning(
                "%s -> 401: ACP rejected the API key. Check ACP_API_KEY is the full key shown once when it "
                "was created (not revoked); create a new key on the app page if unsure.%s",
                label, suffix,
            )  # fmt: skip
        else:
            self._log.warning("%s -> %d%s", label, res.status, suffix)

    def post(self, path: str, body: bytes, timeout: float) -> tuple[bool, Any]:
        """Fail-open POST: (accepted, parsed JSON reply or None). Never raises."""
        try:
            res = self._transport(
                Request("POST", self.url + path, body, {"Content-Type": "application/json"}, timeout)
            )
        except Exception as err:
            self._log.warning("%s failed (continuing anyway): %s", path, err)
            return False, None
        if not 200 <= res.status < 300:
            self.warn_status(path, path, res)
            return False, None
        try:
            return True, json.loads(res.body) if res.body else None
        except ValueError:
            return True, None

    def get_json(self, label: str, path: str, timeout: float, suffix: str) -> Any:
        """GET with the API key header; the parsed JSON reply, or None (warned). Never raises."""
        try:
            res = self._transport(
                Request("GET", self.url + path, None, {"x-api-key": self._api_key}, timeout)
            )
            if 200 <= res.status < 300:
                return json.loads(res.body)
            self.warn_status(label, path, res, suffix)
        except Exception as err:
            self._log.warning("%s failed%s: %s", label, suffix, err)
        return None
