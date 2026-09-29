from __future__ import annotations

from typing import Any

import pytest

from agent_control_panel import create_client, sign_payload

from .conftest import ClientFactory, FakeTransport

BASE: dict[str, Any] = {"api_key": "acp_x", "agent_id": "a1", "webhook_secret": "whsec_x"}


def test_webhook_secret_is_required() -> None:
    with pytest.raises(ValueError, match="webhook_secret"):
        create_client(url="https://panel.test", api_key="acp_x", agent_id="a1", webhook_secret="")
    with pytest.raises(ValueError, match="webhook_secret"):
        create_client(url="https://panel.test", api_key="acp_x", agent_id="a1", webhook_secret="  \n")


@pytest.mark.parametrize("field", ["api_key", "agent_id"])
def test_credentials_are_required(field: str) -> None:
    for blank in ("", "   "):
        with pytest.raises(ValueError, match=field):
            create_client(url="https://panel.test", **{**BASE, field: blank})
    with pytest.raises(ValueError, match="url is required"):
        create_client(url="", **BASE)


def test_plain_http_is_refused_for_a_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="insecure http"):
        create_client(url="http://panel.example.com", **BASE)


def test_other_schemes_and_invalid_urls_are_refused() -> None:
    with pytest.raises(ValueError, match="https"):
        create_client(url="ftp://panel.example.com", **BASE)
    with pytest.raises(ValueError, match="valid URL"):
        create_client(url="not a url", **BASE)
    with pytest.raises(ValueError, match="valid URL"):
        create_client(url="https://", **BASE)


@pytest.mark.parametrize(
    "url",
    ["https://panel.example.com", "http://localhost:5000", "http://127.0.0.1:5000", "http://[::1]:5000"],
)
def test_https_and_loopback_http_are_allowed(url: str) -> None:
    create_client(url=url, **BASE)


def test_http_elsewhere_only_with_the_explicit_dev_opt_out() -> None:
    create_client(url="http://panel.internal", allow_insecure_http=True, **BASE)


@pytest.mark.parametrize(
    "option", [{"heartbeat_seconds": 0}, {"flush_seconds": -1}, {"flush_at": 0}, {"max_buffer": 0}]
)
def test_invalid_timings_and_sizes_are_refused(option: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        create_client(url="https://panel.test", **BASE, **option)


def test_credentials_with_pasted_whitespace_are_trimmed(make_client: ClientFactory) -> None:
    transport = FakeTransport()
    acp = make_client(
        url="https://panel.test/",
        api_key="  acp_test\n",
        agent_id=" agent-1 \n",
        webhook_secret="whsec_trim\n",
        webhook_secret_prev="  ",
        transport=transport,
    )
    assert acp.url == "https://panel.test"
    assert acp.api_key == "acp_test"
    assert acp.agent_id == "agent-1"
    assert acp.webhook_secret == "whsec_trim"
    assert acp.webhook_secret_prev is None
    body = '{"command":"ping","agent_id":"agent-1","payload":{}}'
    assert acp.handle_webhook(sign_payload("whsec_trim", body).headers, body).ok
    acp._heartbeat()
    assert transport.bodies("/api/heartbeat")[0]["api_key"] == "acp_test"
    assert transport.to("/api/heartbeat")[0].url == "https://panel.test/api/heartbeat"


def test_kill_switch_ignores_commands_but_still_reports(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACP_DISABLE_CONTROL", "1")
    transport = FakeTransport()
    acp = make_client(transport=transport)
    assert acp.disable_control
    assert acp.apply_command("pause") == "ignored"
    assert acp.status == "idle"
    acp.report_run("success", dedupe_key="killswitch-1")
    acp.stop()
    assert [r["dedupe_key"] for r in transport.bodies("/api/report")[0]["runs"]] == ["killswitch-1"]


def test_kill_switch_only_for_the_value_1(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACP_DISABLE_CONTROL", "true")
    assert not make_client().disable_control
