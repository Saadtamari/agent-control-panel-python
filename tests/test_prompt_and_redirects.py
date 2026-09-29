from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from agent_control_panel import Request, Response
from agent_control_panel._transport import build_urllib_transport

from .conftest import ClientFactory, FakeTransport, json_response


def test_fetch_prompt_caches_for_a_minute(make_client: ClientFactory) -> None:
    transport = FakeTransport(
        lambda _r: json_response({"prompt": "base + skills", "version": 7, "compositionHash": "h7"})
    )
    acp = make_client(transport=transport)
    assert acp.fetch_prompt() == "base + skills"
    assert acp.fetch_prompt() == "base + skills"
    assert len(transport.requests) == 1
    assert transport.requests[0].headers["x-api-key"] == "acp_test"
    assert transport.requests[0].url == "https://panel.test/api/prompts/agent-1"
    assert (acp.cached_prompt_version, acp.cached_composition_hash) == (7, "h7")

    cached_at = acp._control._prompt_cached_at
    assert cached_at is not None
    acp._control._prompt_cached_at = cached_at - 61
    acp.fetch_prompt()
    assert len(transport.requests) == 2


def test_a_pushed_prompt_marks_the_cache_stale(make_client: ClientFactory) -> None:
    transport = FakeTransport(lambda _r: json_response({"prompt": "composed", "version": 2}))
    acp = make_client(transport=transport)
    acp.fetch_prompt()
    acp.apply_command("update_prompt", {"prompt": "pushed", "version": 3})
    assert acp.cached_prompt_version == 3
    assert acp.fetch_prompt() == "composed"
    assert len(transport.requests) == 2


def test_prompt_version_defaults_to_1(make_client: ClientFactory) -> None:
    acp = make_client(transport=FakeTransport(lambda _r: json_response({"prompt": "p", "version": 0})))
    acp.fetch_prompt()
    assert acp.cached_prompt_version == 1


@pytest.mark.parametrize(
    "answer",
    [
        Response(401),
        Response(500),
        Response(200, b"not json"),
        Response(200, b'{"prompt": 5}'),
        Response(200, b"[]"),
    ],
)
def test_prompt_fetch_falls_back_to_the_cache(make_client: ClientFactory, answer: Response) -> None:
    replies = iter([json_response({"prompt": "cached", "version": 1})])
    acp = make_client(transport=FakeTransport(lambda _r: next(replies, answer)))
    assert acp.fetch_prompt() == "cached"
    acp.apply_command("update_prompt", {"prompt": "pushed"})
    assert acp.fetch_prompt() == "pushed"


def test_a_rejected_prompt_fetch_is_logged(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    acp = make_client(transport=FakeTransport(lambda _r: Response(401)))
    assert acp.fetch_prompt() == ""
    assert "prompt fetch -> 401" in caplog.text
    assert "ACP_API_KEY" in caplog.text


def test_the_agent_id_is_url_encoded(make_client: ClientFactory) -> None:
    transport = FakeTransport(lambda _r: Response(404))
    make_client(agent_id="a/../b?x", transport=transport).fetch_prompt()
    assert transport.requests[0].url == "https://panel.test/api/prompts/a%2F..%2Fb%3Fx"


def test_a_redirect_says_where_acp_moved(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    def moved(req: Request) -> Response:
        path = req.url.removeprefix("https://ops:pw@old.example.com")
        return Response(308, location=f"https://acp.example.com{path}")

    acp = make_client(url="https://ops:pw@old.example.com", transport=FakeTransport(moved))
    acp._heartbeat()
    acp.fetch_prompt()
    acp.report_run("success")
    acp.stop()
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("/api/heartbeat -> 308: ACP moved to https://acp.example.com; update your url")
        for m in lines
    )
    assert any(m.startswith("prompt fetch -> 308: ACP moved to https://acp.example.com;") for m in lines)
    assert any(m.startswith("/api/report -> 308: ACP moved to https://acp.example.com;") for m in lines)
    assert "(currently https://old.example.com)" in caplog.text
    assert "pw" not in caplog.text


def test_a_relative_or_missing_location(make_client: ClientFactory, caplog: pytest.LogCaptureFixture) -> None:
    answers = iter([Response(301, location="/v2/api/heartbeat"), Response(302)])
    acp = make_client(transport=FakeTransport(lambda _r: next(answers)))
    acp._heartbeat()
    acp._heartbeat()
    assert "ACP moved to https://panel.test/v2; update your url" in caplog.text
    assert "redirect without a Location" in caplog.text


def test_a_location_cannot_forge_log_lines(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    acp = make_client(
        transport=FakeTransport(lambda _r: Response(302, location="https://x.test/\nFAKE LOG LINE"))
    )
    acp._heartbeat()
    assert "\n" not in caplog.records[-1].getMessage()


class _Handler(http.server.BaseHTTPRequestHandler):
    hits: list[str]
    redirect_to: str

    def _answer(self) -> None:
        self.hits.append(f"{self.command} {self.path} key={self.headers.get('x-api-key')}")
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        if self.path.startswith("/moved"):
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
        elif self.path.startswith("/fail"):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"oops")
        else:
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, *_args: Any) -> None:
        pass


def _serve(hits: list[str], redirect_to: str = "") -> http.server.ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"hits": hits, "redirect_to": redirect_to})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def servers() -> Iterator[tuple[str, list[str], list[str]]]:
    first_hits: list[str] = []
    other_hits: list[str] = []
    other = _serve(other_hits)
    first = _serve(first_hits, f"http://127.0.0.1:{other.server_address[1]}/stolen")
    yield f"http://127.0.0.1:{first.server_address[1]}", first_hits, other_hits
    for server in (first, other):
        server.shutdown()
        server.server_close()


def test_the_real_transport_never_follows_a_redirect(servers: tuple[str, list[str], list[str]]) -> None:
    base, first_hits, other_hits = servers
    send = build_urllib_transport("test-agent")
    for method in ("GET", "POST"):
        res = send(
            Request(method, f"{base}/moved", b"{}" if method == "POST" else None, {"x-api-key": "k"}, 5)
        )
        assert res.status == 302
        assert res.location is not None
        assert res.location.endswith("/stolen")
    assert other_hits == []
    assert len(first_hits) == 2

    ok = send(Request("GET", f"{base}/fine", None, {}, 5))
    assert (ok.status, ok.body) == (200, b'{"ok": true}')
    failed = send(Request("POST", f"{base}/fail", b"{}", {}, 5))
    assert (failed.status, failed.body) == (500, b"oops")
