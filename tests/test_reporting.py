from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import pytest

from agent_control_panel import Request, Response

from .conftest import ClientFactory, FakeTransport


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_control_panel._reporter.jitter_backoff", lambda _attempt: 0.0)


def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_runs_are_batched_into_one_request_and_flushed_on_stop(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, flush_seconds=60)
    acp.report_run("success", tokens_total=1, dedupe_key="k1")
    acp.report_run("success", tokens_input=2, tokens_output=3, dedupe_key="k2", model_used="gpt-4o")
    acp.report_run("error", error_message="boom", dedupe_key="k3")
    acp.stop()

    [body] = transport.bodies("/api/report")
    assert body["api_key"] == "acp_test"
    assert [r["dedupe_key"] for r in body["runs"]] == ["k1", "k2", "k3"]
    assert body["runs"][1] == {
        "agent_id": "agent-1",
        "dedupe_key": "k2",
        "status": "success",
        "tokens_input": 2,
        "tokens_output": 3,
        "tokens_total": 5,
        "model_used": "gpt-4o",
    }
    assert body["runs"][2]["error_message"] == "boom"


def test_generated_dedupe_keys_are_unique_hex(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    for _ in range(20):
        acp.report_run("success")
    acp.stop()
    keys = [r["dedupe_key"] for r in transport.bodies("/api/report")[0]["runs"]]
    assert len(set(keys)) == 20
    assert all(len(k) == 32 and int(k, 16) >= 0 for k in keys)


def test_flush_at_sends_without_waiting_for_the_timer(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, flush_at=3, flush_seconds=60)
    for i in range(3):
        acp.report_run("success", dedupe_key=f"f{i}")
    _wait_for(lambda: transport.to("/api/report"))
    assert len(transport.bodies("/api/report")[0]["runs"]) == 3

    acp.start()
    for i in range(3):
        acp.report_run("success", dedupe_key=f"g{i}")
    _wait_for(lambda: len(transport.to("/api/report")) == 2)


def test_max_buffer_drops_the_oldest(make_client: ClientFactory, caplog: pytest.LogCaptureFixture) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, max_buffer=3, flush_at=100)
    for i in range(5):
        acp.report_run("success", dedupe_key=f"b{i}")
    assert "buffer full" in caplog.text
    assert acp.dropped_runs == 2
    acp.stop()
    assert [r["dedupe_key"] for r in transport.bodies("/api/report")[0]["runs"]] == ["b2", "b3", "b4"]


def test_failed_flushes_are_retried_then_dropped(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    transport = FakeTransport(lambda _req: Response(500))
    acp = make_client(transport=transport)
    acp.report_run("success", dedupe_key="r1")
    acp.report_run("success", dedupe_key="r2")
    acp.stop()
    assert len(transport.to("/api/report")) == 5
    assert "dropping 2 runs" in caplog.text
    assert acp.dropped_runs == 2


def test_a_retry_succeeds_without_dropping(make_client: ClientFactory) -> None:
    answers = iter([Response(503), Response(429), Response(200, b"{}")])
    transport = FakeTransport(lambda _req: next(answers))
    acp = make_client(transport=transport)
    acp.report_run("success")
    acp.stop()
    assert len(transport.to("/api/report")) == 3
    assert acp.dropped_runs == 0


@pytest.mark.parametrize("error", [OSError("network down"), TimeoutError("timed out"), ValueError("bad")])
def test_an_unreachable_panel_never_raises_into_the_caller(
    make_client: ClientFactory, error: Exception
) -> None:
    def fail(_req: Request) -> Response:
        raise error

    acp = make_client(transport=FakeTransport(fail), heartbeat_seconds=0.02, flush_seconds=0.02)
    acp.start()
    acp.report_run("success")
    assert acp.wrap_agent_run(lambda: "ok") == "ok"
    assert acp.fetch_prompt() == ""
    acp.flush()
    acp.stop()


def test_content_that_cannot_be_serialised_is_left_out_of_its_run(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    circular: dict[str, Any] = {}
    circular["self"] = circular
    acp.report_run("success", full_input=circular, full_output="kept", dedupe_key="circular")
    acp.report_run("success", full_input={"x": float("nan")}, dedupe_key="nan")
    acp.report_run("success", full_output={"when": object()}, dedupe_key="object")
    acp.stop()
    circular_run, nan_run, object_run = transport.bodies("/api/report")[0]["runs"]
    assert "full_input" not in circular_run
    assert circular_run["full_output"] == "kept"
    assert "full_input" not in nan_run
    assert json.loads(object_run["full_output"])["when"].startswith("<object")


def test_runs_are_snapshotted_when_reported(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    payload = {"step": 1}
    acp.report_run("success", full_input=payload)
    payload["step"] = 2
    acp.stop()
    assert transport.bodies("/api/report")[0]["runs"][0]["full_input"] == '{"step":1}'


def test_stop_also_sends_runs_reported_while_a_flush_was_in_flight(make_client: ClientFactory) -> None:
    release = threading.Event()
    reported: list[str] = []
    first = threading.Event()

    def handler(req: Request) -> Response:
        if req.url.endswith("/api/report"):
            reported.extend(r["dedupe_key"] for r in json.loads(req.body or b"{}")["runs"])
            if not first.is_set():
                first.set()
                release.wait(3)
        return Response(200, b"{}")

    acp = make_client(transport=FakeTransport(handler))
    acp.report_run("success", dedupe_key="run-one")
    in_flight = threading.Thread(target=acp.flush)
    in_flight.start()
    first.wait(3)
    acp.report_run("success", dedupe_key="run-two")
    stopping = threading.Thread(target=acp.stop)
    stopping.start()
    release.set()
    stopping.join(5)
    in_flight.join(5)
    assert reported == ["run-one", "run-two"]


def test_stop_gives_up_after_its_timeout(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture
) -> None:
    release = threading.Event()

    def hang(req: Request) -> Response:
        if req.url.endswith("/api/report"):
            release.wait(3)
        return Response(200, b"{}")

    acp = make_client(transport=FakeTransport(hang))
    acp.report_run("success")
    in_flight = threading.Thread(target=acp.flush)
    in_flight.start()
    time.sleep(0.05)
    acp.report_run("success")
    started = time.monotonic()
    acp.stop(timeout=0.3)
    assert time.monotonic() - started < 1.5
    assert "could not be sent within 0.3 s" in caplog.text
    release.set()
    in_flight.join(5)
    caplog.clear()
    acp.stop(timeout=3)
    assert "could not be sent" not in caplog.text


def test_prompt_version_and_hash_default_to_the_cached_prompt(make_client: ClientFactory) -> None:
    transport = FakeTransport(
        lambda req: (
            Response(200, b'{"prompt": "p", "version": 4, "compositionHash": "h4"}')
            if "/api/prompts/" in req.url
            else Response(200, b"{}")
        )
    )
    acp = make_client(transport=transport)
    acp.fetch_prompt()
    acp.report_run("success")
    acp.report_run("success", prompt_version=9, composition_hash="own")
    acp.stop()
    runs = transport.bodies("/api/report")[0]["runs"]
    assert (runs[0]["prompt_version"], runs[0]["composition_hash"]) == (4, "h4")
    assert (runs[1]["prompt_version"], runs[1]["composition_hash"]) == (9, "own")


def test_logs_never_contain_the_api_key(make_client: ClientFactory, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    acp = make_client(api_key="acp_secret_value", transport=FakeTransport(lambda _r: Response(401)))
    acp.report_run("success")
    acp._heartbeat()
    acp.fetch_prompt()
    acp.stop()
    assert "401" in caplog.text
    assert "acp_secret_value" not in caplog.text
