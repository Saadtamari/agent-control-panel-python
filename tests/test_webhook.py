from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest

from agent_control_panel import MAX_SKEW_SECONDS, sign_payload

from .conftest import SECRET, ClientFactory

PAUSE = json.dumps({"command": "pause", "agent_id": "agent-1", "payload": {}})


class WerkzeugLikeHeaders:
    """Flask/Werkzeug headers: case-insensitive get, iteration yields (name, value) pairs."""

    def __init__(self, items: dict[str, str]) -> None:
        self._items = items

    def get(self, key: str, default: Any = None) -> Any:
        for name, value in self._items.items():
            if name.lower() == key.lower():
                return value
        return default

    def keys(self) -> list[str]:
        return list(self._items)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._items.items())


def test_webhook_response_codes(make_client: ClientFactory) -> None:
    acp = make_client()
    code, body = acp.webhook_response(sign_payload(SECRET, PAUSE).headers, PAUSE)
    assert (code, body) == (200, {"ok": True, "result": "paused", "status": "paused"})

    code, body = acp.webhook_response(sign_payload("wrong", PAUSE).headers, PAUSE)
    assert (code, body) == (401, {"ok": False, "reason": "bad_signature"})

    other = json.dumps({"command": "pause", "agent_id": "agent-2", "payload": {}})
    code, body = acp.webhook_response(sign_payload(SECRET, other).headers, other)
    assert (code, body) == (409, {"ok": False, "reason": "wrong_agent"})

    for not_an_object in ("[1, 2]", "null", '"pause"'):
        code, body = acp.webhook_response(sign_payload(SECRET, not_an_object).headers, not_an_object)
        assert (code, body) == (401, {"ok": False, "reason": "bad_body"})


def test_headers_from_any_framework(make_client: ClientFactory) -> None:
    acp = make_client()
    for i, shape in enumerate(("mixed", "werkzeug", "bytes", "lists")):
        body = json.dumps({"command": "ping", "agent_id": "agent-1", "payload": {}, "n": i})
        signed = sign_payload(SECRET, body).headers
        headers: Any
        if shape == "mixed":
            headers = {k.upper() if k.startswith("x-acp-s") else k.title(): v for k, v in signed.items()}
        elif shape == "werkzeug":
            headers = WerkzeugLikeHeaders({k.title(): v for k, v in signed.items()})
        elif shape == "bytes":
            headers = {k.encode(): v.encode() for k, v in signed.items()}
        else:
            headers = {k: [v] for k, v in signed.items()}
        assert acp.handle_webhook(headers, body.encode()).ok, shape


def test_bytes_and_str_bodies_are_the_same(make_client: ClientFactory) -> None:
    acp = make_client()
    body = json.dumps({"command": "update_prompt", "agent_id": "agent-1", "payload": {"prompt": "héllo ✓"}})
    assert acp.handle_webhook(sign_payload(SECRET, body).headers, body.encode("utf-8")).ok
    assert acp.handle_webhook(sign_payload(SECRET, body).headers, body).ok


def test_rotation_accepts_the_previous_secret_only_when_configured(make_client: ClientFactory) -> None:
    without = make_client()
    rotating = make_client(webhook_secret_prev="whsec_prev")
    headers = sign_payload("whsec_prev", PAUSE).headers
    assert without.handle_webhook(headers, PAUSE).reason == "bad_signature"
    assert rotating.handle_webhook(headers, PAUSE).ok


def test_replays_and_stale_requests_are_refused(make_client: ClientFactory) -> None:
    acp = make_client()
    headers = sign_payload(SECRET, PAUSE).headers
    assert acp.handle_webhook(headers, PAUSE).ok
    assert acp.handle_webhook(headers, PAUSE).reason == "replayed_nonce"
    stale = sign_payload(SECRET, PAUSE, timestamp=int(time.time()) - MAX_SKEW_SECONDS - 10)
    assert acp.handle_webhook(stale.headers, PAUSE).reason == "stale_timestamp"
    assert acp.handle_webhook({}, PAUSE).reason == "missing_headers"


def test_a_re_serialised_body_fails_verification(make_client: ClientFactory) -> None:
    acp = make_client()
    headers = sign_payload(SECRET, PAUSE).headers
    reserialised = json.dumps(json.loads(PAUSE), indent=2)
    assert acp.handle_webhook(headers, reserialised).reason == "bad_signature"


def test_the_claimed_agent_id_is_logged_safely(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    acp = make_client()
    body = json.dumps({"command": "stop", "agent_id": "x\nFAKE LINE" + "y" * 200, "payload": {}})
    assert acp.handle_webhook(sign_payload(SECRET, body).headers, body).reason == "wrong_agent"
    message = caplog.records[-1].getMessage()
    assert "\n" not in message
    assert "y" * 100 not in message
    assert acp.status == "idle"
