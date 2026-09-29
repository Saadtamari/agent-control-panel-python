"""Inbound ACP commands: verify, check the addressee, apply."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ._control import ControlState
from ._http import clean
from .errors import AgentStatus
from .signature import HeadersLike


@dataclass(frozen=True)
class WebhookResult:
    ok: bool
    result: str | None = None
    status: AgentStatus | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "result": self.result, "status": self.status}
        return {"ok": False, "reason": self.reason}

    @property
    def http_status(self) -> int:
        """200 when applied; 409 for another agent's command (ACP records it as failed); 401 otherwise."""
        if self.ok:
            return 200
        return 409 if self.reason == "wrong_agent" else 401


def handle_webhook(
    control: ControlState,
    secrets: tuple[str, str | None],
    headers: HeadersLike,
    raw_body: bytes | str,
    log: logging.Logger,
) -> WebhookResult:
    body = raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)
    verified = control.verify_webhook(headers, body, secrets)
    if not verified.ok:
        return WebhookResult(ok=False, reason=verified.reason or "bad_signature")
    try:
        parsed = json.loads(body)
    except ValueError:
        return WebhookResult(ok=False, reason="bad_body")
    if not isinstance(parsed, dict):
        return WebhookResult(ok=False, reason="bad_body")
    # One webhook URL can front several agents; a command addressed to another agent must never change this
    # one. ACP always signs agent_id into the body.
    if parsed.get("agent_id") != control.agent_id:
        claimed = clean(str(parsed.get("agent_id")), 80)
        log.warning("control rejected: command for agent %s reached agent %s", claimed, control.agent_id)
        return WebhookResult(ok=False, reason="wrong_agent")
    payload = parsed.get("payload")
    command = str(parsed.get("command") or "")
    result = control.apply(command, payload if isinstance(payload, dict) else None, "acp")
    return WebhookResult(ok=True, result=result, status=control.status)
