"""The ACP client: fail-open reporting, heartbeats and remote control for one agent."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar
from urllib.parse import quote

from ._content import clamp_summary, extract_text, run_row
from ._control import CachedPrompt, ControlState
from ._http import Api, assert_secure_url
from ._lifecycle import Background
from ._reporter import Reporter
from ._transport import Transport, build_urllib_transport
from ._version import __version__
from ._webhook import WebhookResult, handle_webhook
from .errors import AgentStatus
from .signature import HeadersLike
from .tokens import extract_tokens

T = TypeVar("T")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class AcpClient(Background):
    """Use ``create_client(...)``. Every method is fail-open except the documented ``AgentPausedError``."""

    def __init__(
        self,
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
    ) -> None:
        if not url:
            raise ValueError("create_client: url is required")
        assert_secure_url(url, allow_insecure_http)
        # Credentials never contain whitespace; trim what env files and dashboards often add.
        for name, value in (("api_key", api_key), ("agent_id", agent_id)):
            if not (value or "").strip():
                raise ValueError(f"create_client: {name} is required")
        if not (webhook_secret or "").strip():
            raise ValueError(
                "create_client: webhook_secret is required (refusing to accept unsigned control commands)"
            )
        if heartbeat_seconds <= 0 or flush_seconds <= 0 or flush_at < 1 or max_buffer < 1:
            raise ValueError(
                "create_client: heartbeat_seconds, flush_seconds, flush_at, max_buffer must be > 0"
            )

        self.url = url.rstrip("/")
        self.api_key = api_key.strip()
        self.agent_id = agent_id.strip()
        self.webhook_secret = webhook_secret.strip()
        self.webhook_secret_prev = (webhook_secret_prev or "").strip() or None
        self.disable_control = disable_control or os.environ.get("ACP_DISABLE_CONTROL") == "1"
        self.capture_content = capture_content
        self._log = logger or logging.getLogger("agent_control_panel")
        super().__init__(heartbeat_seconds, flush_seconds)
        self._api = Api(
            self.url,
            self.api_key,
            transport or build_urllib_transport(f"agent-control-panel-python/{__version__}"),
            self._log,
        )
        self._control = ControlState(self.agent_id, self.disable_control, self._log)
        self._reporter = Reporter(
            api_key=self.api_key,
            post=lambda path, body, timeout: self._api.post(path, body, timeout)[0],
            flush_at=flush_at,
            max_buffer=max_buffer,
            logger=self._log,
            wake_flusher=self._wake_flusher,
            wait=lambda seconds: self._closing.wait(seconds),
        )

    # --- state ------------------------------------------------------------------------------------------

    @property
    def status(self) -> AgentStatus:
        return self._control.status

    def get_effective_model(self, fallback: str | None = None) -> str | None:
        """The model ACP swapped this agent to, else ``fallback``."""
        return self._control.model_override or fallback

    @property
    def cached_prompt_version(self) -> int | None:
        prompt = self._control.prompt
        return prompt.version if prompt else None

    @property
    def cached_composition_hash(self) -> str | None:
        prompt = self._control.prompt
        return prompt.composition_hash if prompt else None

    @property
    def dropped_runs(self) -> int:
        return self._reporter.dropped

    # --- heartbeat --------------------------------------------------------------------------------------

    def __enter__(self) -> AcpClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _heartbeat(self) -> None:
        try:
            snap = self._control.heartbeat_snapshot()
            # Counters sent in this heartbeat; runs reported while it is in flight stay for the next one.
            sent = self._reporter.counters()
            entry: dict[str, Any] = {
                "agent_id": self.agent_id,
                "status": snap.status,
                "runs_since_last": sent.runs,
                "tokens_since_last": sent.tokens,
                "errors_since_last": sent.errors,
            }
            if snap.model:
                # The model override in effect, so ACP can confirm a model swap.
                entry["model"] = snap.model
            body = json.dumps({"api_key": self.api_key, "agents": [entry]}).encode()
            ok, reply = self._api.post("/api/heartbeat", body, 10.0)
            if ok:
                self._reporter.commit_counters(sent)
                self._control.adopt(reply, snap.seq)
        except Exception as err:
            self._log.warning("heartbeat error (continuing anyway): %s", err)

    # --- reporting --------------------------------------------------------------------------------------

    def report_run(
        self,
        status: str,
        *,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error_message: str | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        tokens_total: int | None = None,
        execution_time_ms: int | None = None,
        model_used: str | None = None,
        prompt_version: int | None = None,
        composition_hash: str | None = None,
        trigger: str | None = None,
        user_input: str | None = None,
        full_input: Any = None,
        full_output: Any = None,
        dedupe_key: str | None = None,
    ) -> None:
        """Buffer one run for the next flush. Never raises (ACP being down must never break your agent).

        Full fields that are not strings are sent as JSON; each is cut to 100,000 characters (flagged as
        truncated). With ``capture_content=False`` only metadata is sent: no summaries, inputs or outputs.
        """
        try:
            self._resume_after_fork()
            prompt = self._control.prompt
            total = tokens_total or (tokens_input or 0) + (tokens_output or 0)
            fields: dict[str, Any] = {
                "agent_id": self.agent_id,
                "dedupe_key": dedupe_key or uuid.uuid4().hex,
                "status": status,
                "input_summary": input_summary,
                "output_summary": output_summary,
                "error_message": error_message,
                "tokens_input": tokens_input or 0,
                "tokens_output": tokens_output or 0,
                "tokens_total": total,
                "execution_time_ms": execution_time_ms,
                "model_used": model_used,
                "prompt_version": prompt.version if prompt_version is None and prompt else prompt_version,
                "composition_hash": composition_hash or (prompt.composition_hash if prompt else None),
                "trigger": trigger,
                "user_input": user_input,
                "full_input": full_input,
                "full_output": full_output,
            }
            self._reporter.add(run_row(fields, self.capture_content), total, status == "error")
        except Exception as err:
            self._log.warning("report_run failed (fail-open): %s", err)

    def _record(
        self, started: float, options: dict[str, Any], result: Any = None, error: Exception | None = None
    ) -> None:
        fields: dict[str, Any] = {"status": "success"}
        if error is not None:
            fields = {"status": "error", "error_message": str(error) or type(error).__name__}
        else:
            if (tokens := extract_tokens(result)) is not None:
                fields.update(
                    tokens_input=tokens.tokens_input,
                    tokens_output=tokens.tokens_output,
                    tokens_total=tokens.tokens_total,
                )
            if self.capture_content and (text := extract_text(result)) is not None:
                fields.update(output_summary=clamp_summary(text), full_output=text)
        self.report_run(
            input_summary=options["input_summary"],
            user_input=options["user_input"],
            full_input=options["full_input"],
            execution_time_ms=int((time.monotonic() - started) * 1000),
            model_used=options["model_used"] or self._control.model_override,
            **fields,
        )

    def wrap_agent_run(
        self,
        fn: Callable[[], T],
        *,
        input_summary: str | None = None,
        user_input: str | None = None,
        full_input: Any = None,
        model_used: str | None = None,
    ) -> T:
        """Run ``fn()`` and report it: tokens and the answer text when the result is a recognised LLM
        response. Pass ``user_input`` (the end user's request) and ``full_input`` (what was sent to the
        model, e.g. the messages list) so quality reviews and Test Lab have the whole context.

        Raises ``AgentPausedError`` without calling ``fn`` while paused or stopped; re-raises ``fn``'s own
        exception unchanged.
        """
        options = dict(
            input_summary=input_summary, user_input=user_input, full_input=full_input, model_used=model_used
        )
        self._resume_after_fork()
        self._control.begin_run()
        started = time.monotonic()
        try:
            result = fn()
        except Exception as err:
            self._record(started, options, error=err)
            raise
        finally:
            self._control.end_run()
        self._record(started, options, result=result)
        return result

    async def wrap_agent_run_async(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        input_summary: str | None = None,
        user_input: str | None = None,
        full_input: Any = None,
        model_used: str | None = None,
    ) -> T:
        """``wrap_agent_run`` for coroutines: ``await acp.wrap_agent_run_async(lambda: llm.create(...))``."""
        options = dict(
            input_summary=input_summary, user_input=user_input, full_input=full_input, model_used=model_used
        )
        self._resume_after_fork()
        self._control.begin_run()
        started = time.monotonic()
        try:
            result = await fn()
        except Exception as err:
            self._record(started, options, error=err)
            raise
        finally:
            self._control.end_run()
        self._record(started, options, result=result)
        return result

    # --- prompts ----------------------------------------------------------------------------------------

    def fetch_prompt(self) -> str:
        """The composed system prompt from ACP (cached 60 s), else the last one, else "". Never raises.

        Blocks up to 5 s; from async code use ``await asyncio.to_thread(acp.fetch_prompt)``.
        """
        fresh = self._control.fresh_prompt()
        if fresh is not None:
            return fresh.prompt
        path = f"/api/prompts/{quote(self.agent_id, safe='')}"
        data = self._api.get_json("prompt fetch", path, 5.0, " (using the cached prompt)")
        if isinstance(data, dict) and isinstance(data.get("prompt"), str):
            prompt: str = data["prompt"]
            version, composition_hash = data.get("version"), data.get("compositionHash")
            self._control.store_prompt(
                CachedPrompt(
                    prompt=prompt,
                    version=version if _is_int(version) and version else 1,
                    composition_hash=composition_hash if isinstance(composition_hash, str) else None,
                )
            )
            return prompt
        cached = self._control.prompt
        return cached.prompt if cached else ""

    # --- control ----------------------------------------------------------------------------------------

    def apply_command(self, command: str, payload: Mapping[str, Any] | None = None) -> str | None:
        """Apply a control command locally. A pause/stop set this way is never released by ACP."""
        return self._control.apply(command, payload, "local")

    def handle_webhook(self, headers: HeadersLike, raw_body: bytes | str) -> WebhookResult:
        """Verify and apply a signed ACP command. Pass the raw request body, not re-serialised JSON."""
        secrets = (self.webhook_secret, self.webhook_secret_prev)
        return handle_webhook(self._control, secrets, headers, raw_body, self._log)

    def webhook_response(self, headers: HeadersLike, raw_body: bytes | str) -> tuple[int, dict[str, Any]]:
        """``handle_webhook`` as an HTTP answer for any framework: (200, 401 or 409; JSON body)."""
        out = self.handle_webhook(headers, raw_body)
        return out.http_status, out.to_dict()
