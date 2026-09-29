"""Wire-contract vectors shared with the ACP server and the Node SDK.

They are generated from the server's real signing code (server/__tests__/sdk-vectors.ts in the main
repository); do not edit contract/vectors.json by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_control_panel import NonceCache, Request, Response, extract_tokens, sign_payload, verify_signature

from .conftest import ClientFactory, FakeTransport

VECTORS: dict[str, Any] = json.loads((Path(__file__).parent / "contract" / "vectors.json").read_text("utf-8"))


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["name"] for item in items]


@pytest.mark.parametrize("case", VECTORS["webhooks"]["cases"], ids=_ids(VECTORS["webhooks"]["cases"]))
def test_webhook_signature(case: dict[str, Any]) -> None:
    cache = NonceCache()
    secrets = case.get("secrets", [VECTORS["secret"], VECTORS["secret_prev"]])
    for req in case["requests"]:
        out = verify_signature(
            secrets=secrets,
            raw_body=req["raw_body"],
            headers=req["headers"],
            now=VECTORS["webhooks"]["now"],
            seen_nonce=cache.has,
        )
        if out.ok and out.nonce:
            cache.add(out.nonce)
        got: dict[str, Any] = {"ok": out.ok}
        if out.reason:
            got["reason"] = out.reason
        assert got == req["expect"]


@pytest.mark.parametrize("scenario", VECTORS["scenarios"], ids=_ids(VECTORS["scenarios"]))
def test_client_scenario(scenario: dict[str, Any], make_client: ClientFactory) -> None:
    sent: list[dict[str, Any]] = []
    reply: dict[str, Any] = {}

    def sign(raw: str) -> dict[str, str]:
        return sign_payload(VECTORS["secret"], raw).headers

    def handler(req: Request) -> Response:
        if req.url.endswith("/api/heartbeat"):
            sent.append(json.loads(req.body or b"{}")["agents"][0])
            for raw in reply["while_in_flight"]:
                client.handle_webhook(sign(raw), raw)
            return Response(reply["status"], reply["body"])
        return Response(200, b"{}")

    client = make_client(
        agent_id=VECTORS["agent_id"],
        webhook_secret=VECTORS["secret"],
        webhook_secret_prev=VECTORS["secret_prev"],
        disable_control=scenario.get("client", {}).get("disable_control", False),
        transport=FakeTransport(handler),
    )

    for step in scenario["steps"]:
        if "webhook" in step:
            raw = step["webhook"]["raw_body"]
            assert client.handle_webhook(sign(raw), raw).to_dict() == step["expect"]
        elif "apply" in step:
            applied = client.apply_command(step["apply"]["command"], step["apply"].get("payload"))
            assert applied == step["expect"]["result"]
        elif "report" in step:
            r = step["report"]
            client.report_run(
                r["status"],
                tokens_input=r.get("tokens_input"),
                tokens_output=r.get("tokens_output"),
                tokens_total=r.get("tokens_total"),
            )
        elif "heartbeat" in step:
            hb = step["heartbeat"]
            body = hb["reply_raw"] if "reply_raw" in hb else json.dumps(hb["reply"])
            reply.update(
                status=hb["http_status"], body=body.encode(), while_in_flight=hb.get("while_in_flight", [])
            )
            client._heartbeat()
            assert sent[-1] == hb["sent"]
        else:
            raise AssertionError(f"unknown step: {step}")
        state = {
            "status": client.status,
            "model": client.get_effective_model(),
            "prompt_version": client.cached_prompt_version,
        }
        assert state == step["state"]


def expand_repeats(value: Any) -> Any:
    """Replace every ``{"$repeat": [text, n]}`` with ``text`` repeated n times."""
    if isinstance(value, list):
        return [expand_repeats(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$repeat"}:
            text, times = value["$repeat"]
            return str(text) * int(times)
        return {k: expand_repeats(v) for k, v in value.items()}
    return value


@pytest.mark.parametrize("case", VECTORS["report"], ids=_ids(VECTORS["report"]))
def test_run_reporting(case: dict[str, Any], make_client: ClientFactory) -> None:
    transport = FakeTransport()
    client = make_client(
        agent_id=VECTORS["agent_id"],
        capture_content=case.get("client", {}).get("capture_content", True),
        transport=transport,
    )
    call = expand_repeats(case["call"])
    if "wrap" in call:
        w = call["wrap"]
        options = {k: w.get(k) for k in ("input_summary", "user_input", "full_input", "model_used")}

        def run() -> Any:
            if "raises" in w:
                raise RuntimeError(w["raises"])
            return w["result"]

        if "raises" in w:
            with pytest.raises(RuntimeError, match=w["raises"]):
                client.wrap_agent_run(run, **options)
        else:
            client.wrap_agent_run(run, **options)
    else:
        client.report_run(**call["report"])
    client.stop()

    row = transport.bodies("/api/report")[0]["runs"][0]
    assert row.pop("agent_id") == VECTORS["agent_id"]
    assert isinstance(row.pop("dedupe_key"), str)
    assert isinstance(row.pop("execution_time_ms", 0), int)
    assert row == expand_repeats(case["row"])


@pytest.mark.parametrize("case", VECTORS["tokens"], ids=_ids(VECTORS["tokens"]))
def test_token_extraction(case: dict[str, Any]) -> None:
    out = extract_tokens(case["input"])
    got = out and {
        "tokens_input": out.tokens_input,
        "tokens_output": out.tokens_output,
        "tokens_total": out.tokens_total,
    }
    assert got == case["expect"]
