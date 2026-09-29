from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agent_control_panel import Request, Response, sign_payload

from .conftest import SECRET, ClientFactory, FakeTransport

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_concurrent_reports_heartbeats_and_webhooks(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(transport=transport, flush_at=25, max_buffer=100_000, heartbeat_seconds=0.01)
    acp.start()
    body = json.dumps({"command": "ping", "agent_id": "agent-1", "payload": {}})
    replayed = sign_payload(SECRET, body).headers
    accepted: list[bool] = []
    go = threading.Barrier(9)

    def reporter(n: int) -> None:
        go.wait()
        for i in range(250):
            acp.report_run("error" if i % 10 == 0 else "success", tokens_total=2, dedupe_key=f"t{n}-{i}")

    def replayer() -> None:
        go.wait()
        for _ in range(50):
            accepted.append(acp.handle_webhook(replayed, body).ok)

    threads = [threading.Thread(target=reporter, args=(n,)) for n in range(8)]
    threads.append(threading.Thread(target=replayer))
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    acp.stop()
    acp._heartbeat()
    acp._heartbeat()

    runs = [r["dedupe_key"] for b in transport.bodies("/api/report") for r in b["runs"]]
    assert len(runs) == len(set(runs)) == 2000
    beats = [b["agents"][0] for b in transport.bodies("/api/heartbeat")]
    assert sum(b["runs_since_last"] for b in beats) == 2000
    assert sum(b["tokens_since_last"] for b in beats) == 4000
    assert sum(b["errors_since_last"] for b in beats) == 200
    assert beats[-1]["runs_since_last"] == 0
    assert accepted.count(True) == 1


def test_one_nonce_is_accepted_once_across_threads(make_client: ClientFactory) -> None:
    acp = make_client()
    body = json.dumps({"command": "pause", "agent_id": "agent-1", "payload": {}})
    headers = sign_payload(SECRET, body).headers
    go = threading.Barrier(16)
    results: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        go.wait()
        ok = acp.handle_webhook(headers, body).ok
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results.count(True) == 1


def _run_script(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(  # noqa: S603 (fixed script, this interpreter)
        [sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=30
    )


def test_pending_runs_are_flushed_at_exit() -> None:
    started = time.monotonic()
    done = _run_script(
        """
        import json, sys
        from agent_control_panel import Response, create_client

        def transport(req):
            if req.url.endswith("/api/report"):
                runs = json.loads(req.body)["runs"]
                print("REPORTED", ",".join(r["dedupe_key"] for r in runs), flush=True)
            return Response(200, b"{}")

        acp = create_client(url="https://panel.test", api_key="acp_x", agent_id="a", webhook_secret="s",
                            transport=transport, flush_seconds=3600, heartbeat_seconds=3600)
        acp.start()
        acp.report_run("success", dedupe_key="at-exit-1")
        acp.report_run("success", dedupe_key="at-exit-2")
        # no stop(): the interpreter exits with the daemon threads still running
        """
    )
    assert done.returncode == 0, done.stderr
    assert "REPORTED at-exit-1,at-exit-2" in done.stdout
    assert time.monotonic() - started < 20


def test_exit_is_bounded_when_the_panel_hangs() -> None:
    done = _run_script(
        """
        import time
        from agent_control_panel import Response, create_client

        def transport(req):
            time.sleep(60)
            return Response(200, b"{}")

        acp = create_client(url="https://panel.test", api_key="acp_x", agent_id="a", webhook_secret="s",
                            transport=transport, heartbeat_seconds=3600)
        acp.start()
        acp.report_run("success")
        """
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork is POSIX-only")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # 3.12+ warns about fork() with threads running
def test_a_forked_child_starts_clean(make_client: ClientFactory) -> None:
    transport = FakeTransport(lambda _r: Response(200, b"{}"))
    acp = make_client(transport=transport, flush_seconds=3600, heartbeat_seconds=3600)
    acp.start()
    acp.report_run("success", dedupe_key="parent-run")
    read_end, write_end = os.pipe()
    pid = os.fork()  # type: ignore[attr-defined,unused-ignore]
    if pid == 0:  # child
        try:
            buffered_after_fork = acp._reporter.buffered
            acp.report_run("success", dedupe_key="child-run")
            threads = sorted(t.name for t in threading.enumerate() if t.name.startswith("acp-"))
            os.write(write_end, json.dumps({"buffered": buffered_after_fork, "threads": threads}).encode())
        finally:
            os._exit(0)
    os.close(write_end)
    with os.fdopen(read_end) as pipe:
        child = json.loads(pipe.read())
    os.waitpid(pid, 0)
    assert child == {"buffered": 0, "threads": ["acp-flush", "acp-heartbeat"]}
    acp.stop()
    assert [r["dedupe_key"] for r in transport.bodies("/api/report")[0]["runs"]] == ["parent-run"]


def test_request_timeouts_are_passed_to_the_transport(make_client: ClientFactory) -> None:
    seen: dict[str, float] = {}

    def record(req: Request) -> Response:
        seen[req.url.rsplit("/", 1)[-1]] = req.timeout
        return Response(200, b"{}")

    acp = make_client(transport=FakeTransport(record))
    acp._heartbeat()
    acp.fetch_prompt()
    acp.report_run("success")
    acp.flush()
    assert seen == {"heartbeat": 10.0, "agent-1": 5.0, "report": 10.0}
