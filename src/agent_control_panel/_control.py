"""Control state: status, holds, model override, prompt cache and the webhook nonce cache.

Same rules as the Node SDK (``adoptControlState`` / ``applyCommandFrom``):
- only an explicit resume/restart clears paused/stopped;
- a hold ACP set is adopted from, and released by, the heartbeat reply; a hold this app set locally never is;
- a heartbeat reply sent before the last applied command is ignored (``command_seq``);
- a model in the heartbeat reply is adopted unless the app set its own; a missing model is no instruction.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import AgentPausedError, AgentStatus
from .signature import HeadersLike, NonceCache, VerifyResult, verify_signature

Origin = Literal["acp", "local"]

PROMPT_CACHE_SECONDS = 60.0
MAX_MODEL_LENGTH = 120


@dataclass(frozen=True)
class CachedPrompt:
    prompt: str
    version: int
    composition_hash: str | None = None


@dataclass(frozen=True)
class HeartbeatSnapshot:
    seq: int
    status: AgentStatus
    model: str | None


class ControlState:
    def __init__(self, agent_id: str, disable_control: bool, logger: logging.Logger) -> None:
        self.agent_id = agent_id
        self.disable_control = disable_control
        self._log = logger
        self._lock = threading.Lock()
        self._status: AgentStatus = "idle"
        self._hold_origin: Origin | None = None
        self._command_seq = 0
        self._model_override: str | None = None
        self._model_origin: Origin | None = None
        self._prompt: CachedPrompt | None = None
        self._prompt_cached_at: float | None = None
        self._active_runs = 0
        self._nonces = NonceCache()

    def reset_lock_after_fork(self) -> None:
        self._lock = threading.Lock()

    @property
    def status(self) -> AgentStatus:
        with self._lock:
            return self._status

    @property
    def model_override(self) -> str | None:
        with self._lock:
            return self._model_override

    @property
    def prompt(self) -> CachedPrompt | None:
        with self._lock:
            return self._prompt

    # --- runs -------------------------------------------------------------------------------------------

    def begin_run(self) -> None:
        with self._lock:
            if self._status in ("paused", "stopped"):
                raise AgentPausedError(self._status)
            self._active_runs += 1
            self._status = "running"

    def end_run(self) -> None:
        with self._lock:
            self._active_runs = max(0, self._active_runs - 1)
            if self._active_runs == 0 and self._status == "running":
                self._status = "idle"

    # --- commands ---------------------------------------------------------------------------------------

    def apply(self, command: str, payload: Mapping[str, Any] | None, origin: Origin) -> str | None:
        if self.disable_control:
            self._log.info("control ignored (disable_control): %s", command[:40])
            return "ignored"
        with self._lock:
            if command not in ("ping", "verify"):
                self._command_seq += 1
            if command == "pause":
                self._status, self._hold_origin = "paused", origin
                return "paused"
            if command == "resume":
                self._status, self._hold_origin = "idle", None
                return "resumed"
            if command == "stop":
                self._status, self._hold_origin = "stopped", origin
                return "stopped"
            if command == "restart":
                self._status, self._hold_origin = "idle", None
                return "restarted"
            if command == "update_model":
                model = payload.get("model") if payload else None
                if isinstance(model, str) and model:
                    self._model_override, self._model_origin = model, origin
                return f"model now {self._model_override if self._model_override is not None else 'null'}"
            if command == "update_prompt":
                prompt = payload.get("prompt") if payload else None
                if isinstance(prompt, str) and prompt:
                    # The pushed text lacks the agent's skills (ACP composes them into GET /api/prompts), so
                    # it is only the fallback: the cache is marked stale and the next fetch_prompt() pulls
                    # the composed prompt.
                    sent_version = payload.get("version") if payload else None
                    version = (self._prompt.version if self._prompt else 0) + 1
                    if isinstance(sent_version, int) and not isinstance(sent_version, bool):
                        version = sent_version
                    self._prompt = CachedPrompt(prompt=prompt, version=version)
                    self._prompt_cached_at = None
                return "prompt updated"
            if command in ("ping", "verify"):
                # Handshake / no-op: never touches agent status.
                return "pong"
            return None

    # --- heartbeat --------------------------------------------------------------------------------------

    def heartbeat_snapshot(self) -> HeartbeatSnapshot:
        with self._lock:
            return HeartbeatSnapshot(self._command_seq, self._status, self._model_override)

    def adopt(self, reply: Any, seq_at_send: int) -> None:
        if self.disable_control:
            return
        control = reply.get("control") if isinstance(reply, dict) else None
        agents = control.get("agents") if isinstance(control, dict) else None
        entry = agents.get(self.agent_id) if isinstance(agents, dict) else None
        if not isinstance(entry, dict):
            return
        with self._lock:
            if seq_at_send != self._command_seq:
                return
            model = entry.get("model")
            if (
                isinstance(model, str)
                and model
                and len(model) <= MAX_MODEL_LENGTH
                and self._model_origin != "local"
                and model != self._model_override
            ):
                self._model_override, self._model_origin = model, "acp"
                self._log.info("model from ACP: %s", model)
            status = entry.get("status")
            if not isinstance(status, str):
                return
            if status in ("paused", "stopped"):
                if self._hold_origin == "local" or self._status == status:
                    return
                self._status = "paused" if status == "paused" else "stopped"
                self._hold_origin = "acp"
                self._log.info("control state from ACP: %s", status)
            elif self._hold_origin == "acp":
                self._status, self._hold_origin = "idle", None
                self._log.info("control state from ACP: released")

    # --- webhooks ---------------------------------------------------------------------------------------

    def verify_webhook(
        self, headers: HeadersLike, raw_body: bytes, secrets: tuple[str, str | None]
    ) -> VerifyResult:
        # Checking and recording the nonce under one lock: two server threads must not both accept a replay.
        with self._lock:
            result = verify_signature(
                secrets=secrets, raw_body=raw_body, headers=headers, seen_nonce=self._nonces.has
            )
            if result.ok and result.nonce:
                self._nonces.add(result.nonce)
            return result

    # --- prompt cache -----------------------------------------------------------------------------------

    def fresh_prompt(self) -> CachedPrompt | None:
        with self._lock:
            if self._prompt is None or self._prompt_cached_at is None:
                return None
            if time.monotonic() - self._prompt_cached_at >= PROMPT_CACHE_SECONDS:
                return None
            return self._prompt

    def store_prompt(self, prompt: CachedPrompt) -> None:
        with self._lock:
            self._prompt = prompt
            self._prompt_cached_at = time.monotonic()
