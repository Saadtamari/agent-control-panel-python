from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_control_panel import AgentPausedError, Request, Response, sign_payload

from .conftest import SECRET, ClientFactory, FakeTransport, json_response


def _hb(status: str, **extra: Any) -> Response:
    return json_response({"success": True, "control": {"agents": {"agent-1": {"status": status, **extra}}}})


def _signed(acp: Any, command: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps({"command": command, "agent_id": "agent-1", "payload": payload or {}})
    return acp.handle_webhook(sign_payload(SECRET, body).headers, body)


def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_start_sends_a_heartbeat_at_once_and_a_fresh_process_adopts_a_pause(
    make_client: ClientFactory,
) -> None:
    transport = FakeTransport(
        lambda req: _hb("paused") if req.url.endswith("/api/heartbeat") else Response(200)
    )
    acp = make_client(transport=transport, heartbeat_seconds=60)
    acp.start()
    _wait_for(lambda: acp.status == "paused")
    called = []
    with pytest.raises(AgentPausedError) as err:
        acp.wrap_agent_run(lambda: called.append(1))
    assert err.value.status == "paused"
    assert called == []


def test_threads_are_daemons_and_stop_ends_them(make_client: ClientFactory) -> None:
    acp = make_client(heartbeat_seconds=0.05, flush_seconds=0.05)
    acp.start()
    acp.start()  # idempotent
    threads = [t for t in threading.enumerate() if t.name in ("acp-heartbeat", "acp-flush")]
    assert len(threads) == 2
    assert all(t.daemon for t in threads)
    acp.stop()
    assert not any(t.is_alive() for t in threads)


def test_the_heartbeat_loop_survives_errors(make_client: ClientFactory) -> None:
    calls: list[int] = []

    def flaky(req: Request) -> Response:
        if req.url.endswith("/api/heartbeat"):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionResetError("reset")
        return Response(200, b"{}")

    acp = make_client(transport=FakeTransport(flaky), heartbeat_seconds=0.02)
    acp.start()
    _wait_for(lambda: len(calls) >= 3)


def test_runs_reported_during_a_heartbeat_count_in_the_next_one(make_client: ClientFactory) -> None:
    def handler(req: Request) -> Response:
        if req.url.endswith("/api/heartbeat") and len(transport.to("/api/heartbeat")) == 1:
            acp.report_run("error", tokens_total=5)
        return Response(200, b"{}")

    transport = FakeTransport(handler)
    acp = make_client(transport=transport, flush_at=1000)
    acp.report_run("success", tokens_total=10)
    acp._heartbeat()
    acp._heartbeat()
    first, second = (b["agents"][0] for b in transport.bodies("/api/heartbeat"))
    assert (first["runs_since_last"], first["tokens_since_last"], first["errors_since_last"]) == (1, 10, 0)
    assert (second["runs_since_last"], second["tokens_since_last"], second["errors_since_last"]) == (1, 5, 1)


def test_the_model_override_is_reported_in_the_heartbeat(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    acp._heartbeat()
    _signed(acp, "update_model", {"model": "gpt-4o-mini"})
    acp._heartbeat()
    first, second = (b["agents"][0] for b in transport.bodies("/api/heartbeat"))
    assert "model" not in first
    assert second["model"] == "gpt-4o-mini"
    assert acp.get_effective_model("gpt-4o") == "gpt-4o-mini"


def test_a_stale_reply_after_a_webhook_command_is_ignored(make_client: ClientFactory) -> None:
    release = threading.Event()

    def handler(req: Request) -> Response:
        if req.url.endswith("/api/heartbeat"):
            release.wait(3)
            return _hb("paused")
        return Response(200)

    acp = make_client(transport=FakeTransport(handler))
    _signed(acp, "pause")
    beat = threading.Thread(target=acp._heartbeat)
    beat.start()
    time.sleep(0.05)
    assert _signed(acp, "resume").status == "idle"
    release.set()
    beat.join(3)
    assert acp.status == "idle"


def test_a_local_hold_is_never_released_by_acp(make_client: ClientFactory) -> None:
    acp = make_client(transport=FakeTransport(lambda _r: _hb("running")))
    acp.apply_command("stop")
    acp._heartbeat()
    assert acp.status == "stopped"
    acp.apply_command("restart")
    assert acp.status == "idle"


def test_pause_and_stop_refuse_sync_and_async_runs(make_client: ClientFactory) -> None:
    acp = make_client()

    async def work() -> str:
        return "ok"

    for command in ("pause", "stop"):
        acp.apply_command(command)
        with pytest.raises(AgentPausedError):
            acp.wrap_agent_run(lambda: "ok")
        with pytest.raises(AgentPausedError):
            asyncio.run(acp.wrap_agent_run_async(work))
        acp.apply_command("resume")
        assert acp.wrap_agent_run(lambda: "ok") == "ok"
        assert asyncio.run(acp.wrap_agent_run_async(work)) == "ok"


@pytest.mark.parametrize("use_async", [False, True])
def test_the_callers_exception_is_re_raised_unchanged(make_client: ClientFactory, use_async: bool) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    boom = RuntimeError("llm exploded")

    def fail() -> None:
        raise boom

    async def afail() -> None:
        raise boom

    def run() -> None:
        if use_async:
            asyncio.run(acp.wrap_agent_run_async(afail, input_summary="q"))
        else:
            acp.wrap_agent_run(fail, input_summary="q")

    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is boom
    acp.stop()
    [reported] = transport.bodies("/api/report")[0]["runs"]
    assert reported["status"] == "error"
    assert reported["error_message"] == "llm exploded"
    assert reported["input_summary"] == "q"
    assert acp.status == "idle"


def test_an_exception_without_a_message_reports_its_type(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)

    def fail() -> None:
        raise ValueError

    with pytest.raises(ValueError, match=r"^$"):
        acp.wrap_agent_run(fail)
    acp.stop()
    assert transport.bodies("/api/report")[0]["runs"][0]["error_message"] == "ValueError"


def test_keyboard_interrupt_passes_through_without_a_report(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)

    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        acp.wrap_agent_run(interrupted)
    assert acp.status == "idle"
    acp.stop()
    assert transport.to("/api/report") == []


def test_wrapped_runs_report_tokens_timing_and_the_swapped_model(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport)
    _signed(acp, "update_model", {"model": "gpt-4o-mini"})
    openai_like = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
    )
    assert acp.wrap_agent_run(lambda: openai_like) is openai_like
    acp.wrap_agent_run(lambda: {"usage": {"input_tokens": 1, "output_tokens": 1}}, model_used="claude-own")
    acp.stop()
    first, second = transport.bodies("/api/report")[0]["runs"]
    assert (first["tokens_input"], first["tokens_output"], first["tokens_total"]) == (12, 3, 15)
    assert first["model_used"] == "gpt-4o-mini"
    assert first["execution_time_ms"] >= 0
    assert second["model_used"] == "claude-own"


def test_status_is_running_while_any_run_is_active(make_client: ClientFactory) -> None:
    acp = make_client()
    inside = threading.Barrier(3)
    leave = threading.Event()

    def work() -> None:
        inside.wait(3)
        leave.wait(3)

    threads = [threading.Thread(target=acp.wrap_agent_run, args=(work,)) for _ in range(2)]
    for t in threads:
        t.start()
    inside.wait(3)
    assert acp.status == "running"
    leave.set()
    for t in threads:
        t.join(3)
    assert acp.status == "idle"


def test_a_pause_during_a_run_holds_after_it(make_client: ClientFactory) -> None:
    acp = make_client()
    acp.wrap_agent_run(lambda: _signed(acp, "pause"))
    assert acp.status == "paused"


def test_update_prompt_and_unknown_commands(make_client: ClientFactory) -> None:
    acp = make_client()
    assert acp.apply_command("update_prompt", {"prompt": "hello", "version": 3}) == "prompt updated"
    assert acp.cached_prompt_version == 3
    assert acp.apply_command("update_prompt", {"prompt": ""}) == "prompt updated"
    assert acp.cached_prompt_version == 3
    assert acp.apply_command("update_model") == "model now null"
    assert acp.apply_command("nonsense") is None
    assert acp.status == "idle"
