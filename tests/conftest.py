from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from agent_control_panel import AcpClient, Request, Response, create_client

SECRET = "whsec_test"


class FakeTransport:
    """Records requests; answers through ``handler`` (default: 200 with ``{}``)."""

    def __init__(self, handler: Callable[[Request], Response] | None = None) -> None:
        self.requests: list[Request] = []
        self.handler = handler or (lambda _req: Response(200, b"{}"))
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> Response:
        with self._lock:
            self.requests.append(request)
        return self.handler(request)

    def to(self, path: str) -> list[Request]:
        with self._lock:
            return [r for r in self.requests if r.url.endswith(path) or f"{path}/" in r.url]

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.body or b"null") for r in self.to(path)]


def json_response(data: Any, status: int = 200) -> Response:
    return Response(status, json.dumps(data).encode())


ClientFactory = Callable[..., AcpClient]


@pytest.fixture
def make_client() -> Iterator[ClientFactory]:
    clients: list[AcpClient] = []

    def factory(**overrides: Any) -> AcpClient:
        options: dict[str, Any] = {
            "url": "https://panel.test",
            "api_key": "acp_test",
            "agent_id": "agent-1",
            "webhook_secret": SECRET,
            "transport": FakeTransport(),
            "logger": logging.getLogger("acp-test"),
        }
        options.update(overrides)
        client = create_client(**options)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.stop(timeout=2)


@pytest.fixture(autouse=True)
def _no_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACP_DISABLE_CONTROL", raising=False)
