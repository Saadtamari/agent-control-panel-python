"""Agent Control Panel SDK: fail-open reporting and remote control for AI agents."""

from __future__ import annotations

import logging

from ._content import MAX_FULL_CHARS, extract_text
from ._http import assert_secure_url
from ._transport import Request, Response, Transport
from ._version import __version__
from ._webhook import WebhookResult
from .client import AcpClient
from .errors import AgentPausedError, AgentStatus
from .signature import (
    MAX_SKEW_SECONDS,
    NONCE_HEADER,
    PROTOCOL_HEADER,
    SIG_HEADER,
    TS_HEADER,
    HeadersLike,
    NonceCache,
    SignedPayload,
    VerifyResult,
    sign_payload,
    verify_signature,
)
from .tokens import TokenUsage, extract_tokens

PACKAGE_NAME = "agent-control-panel"


def create_client(
    *,
    url: str,
    api_key: str,
    agent_id: str,
    webhook_secret: str,
    webhook_secret_prev: str | None = None,
    heartbeat_seconds: float = 30.0,
    flush_seconds: float = 5.0,
    flush_at: int = 50,
    max_buffer: int = 200,
    logger: logging.Logger | None = None,
    disable_control: bool = False,
    allow_insecure_http: bool = False,
    capture_content: bool = True,
    transport: Transport | None = None,
) -> AcpClient:
    """Create a client for one agent. Call ``start()`` (or use ``with``) for heartbeats and flushes.

    ``capture_content=False`` sends metadata only (status, tokens, timings, model, errors): no user input,
    summaries, full input or answer text, whatever ``report_run`` or ``wrap_agent_run`` is given.

    Raises ``ValueError`` for a missing credential or an insecure URL; nothing else in the client raises
    into your code except ``AgentPausedError`` from ``wrap_agent_run``.
    """
    return AcpClient(
        url=url,
        api_key=api_key,
        agent_id=agent_id,
        webhook_secret=webhook_secret,
        webhook_secret_prev=webhook_secret_prev,
        heartbeat_seconds=heartbeat_seconds,
        flush_seconds=flush_seconds,
        flush_at=flush_at,
        max_buffer=max_buffer,
        logger=logger,
        disable_control=disable_control,
        allow_insecure_http=allow_insecure_http,
        capture_content=capture_content,
        transport=transport,
    )


__all__ = [
    "MAX_FULL_CHARS",
    "MAX_SKEW_SECONDS",
    "NONCE_HEADER",
    "PACKAGE_NAME",
    "PROTOCOL_HEADER",
    "SIG_HEADER",
    "TS_HEADER",
    "AcpClient",
    "AgentPausedError",
    "AgentStatus",
    "HeadersLike",
    "NonceCache",
    "Request",
    "Response",
    "SignedPayload",
    "TokenUsage",
    "Transport",
    "VerifyResult",
    "WebhookResult",
    "__version__",
    "assert_secure_url",
    "create_client",
    "extract_text",
    "extract_tokens",
    "sign_payload",
    "verify_signature",
]
