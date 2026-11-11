# Changelog

All notable changes to the Python SDK. Versions stay `0.x` until the wire contract is proven with real
customers; after the first public release, contract changes are additive only (clients live inside customer
apps). The wire contract is the same as the Node SDK (`agent-control-panel` on npm).

## 0.1.4 (2026-11-11)

Agent framework results are read as they are; no adapter code. `wrap_agent_run(lambda: ...)` now records the
answer and tokens of:

- an OpenAI Agents SDK `RunResult` (`Runner.run_sync()` / `await Runner.run()`): `final_output` (structured
  output as JSON) and `context_wrapper.usage`, every model call of the run;
- a LangGraph state (`graph.invoke()`, `{"messages": [...]}`): the last AI message as the answer and the sum of
  every AI message's `usage_metadata` as tokens; a single LangChain message (`chat_model.invoke()`) too;
- a CrewAI `CrewOutput` (`crew.kickoff()`): `raw` and `token_usage`.

Additive: shapes that were read before read the same.

## 0.1.3 (2026-11-07)

A pause now holds across restarts. A restarted process starts `idle` until ACP's first heartbeat reply, and
`wrap_agent_run` used to run straight away: a paused agent ran once after every restart or deploy. After
`start()`, the first `wrap_agent_run` / `wrap_agent_run_async` now waits for that reply, at most
`startup_wait_seconds` (new option, default 3; 0 turns it off; the async wrapper waits off the event loop).
Fail-open: if ACP does not answer in time the run goes ahead, and later runs never wait. No wait with
`disable_control`, or when `start()` was not called.

## 0.1.2 (2026-11-03)

Package metadata only; no code changes. The website moved to https://agentcontrolpanel.si: PyPI (`Homepage`,
`Documentation`), the README and SECURITY.md now link it. The old links redirect there. The API host your app
reports to is unchanged.

## 0.1.1 (2026-10-03)

Package metadata only; no code changes. PyPI links the website and the documentation (`Homepage`,
`Documentation`), the summary says what the SDK does, more search keywords, and the README links the docs and the
quickstart.

## 0.1.0 (2026-09-29)

First release, with the same behaviour and wire contract as the Node SDK 0.3.0:

- `create_client(...)` with the same validation: `https` only (plain `http` for loopback hosts, or with
  `allow_insecure_http=True` for local development), credentials trimmed, refuses to start without a
  webhook secret, `ACP_DISABLE_CONTROL=1` honoured.
- Fail-open reporting: `report_run` buffers and never raises; batched flushes with retry and jitter; the
  oldest runs are dropped past `max_buffer`. Heartbeats and flushes run on daemon threads; pending runs are
  flushed on interpreter exit.
- `wrap_agent_run` / `wrap_agent_run_async`: refuse to run while paused or stopped (`AgentPausedError`),
  report tokens and the answer text (`extract_text`: OpenAI Chat and Responses, Anthropic, Gemini, `text`;
  reads keys and attributes only, never calls methods), accept `user_input` and `full_input`, and re-raise
  your exception unchanged.
- Run content: full fields are sent as JSON text cut to 100,000 characters (counted in UTF-16 units, as ACP
  does) and flagged `full_input_truncated` / `full_output_truncated`; large flushes are split into requests
  under 4 MB. `capture_content=False` sends metadata only, from `wrap_agent_run` and `report_run` alike.
- Remote control: `handle_webhook` / `webhook_response` verify signature v2 (timestamp + nonce, 300 s skew,
  single-use nonces, previous secret during rotation) and apply pause, resume, stop, restart,
  `update_model` and `update_prompt`. Durable pause and model swaps through the heartbeat reply, with the
  same rules: a hold set locally is never released by ACP, and replies older than the last applied command
  are ignored.
- `fetch_prompt()` with a 60 s cache and fallback to the last prompt.
- Zero runtime dependencies; typed (`py.typed`); Python 3.10+.
- Redirects are never followed (the API key is never sent to another host); a 3xx is logged with its target
  so you know which URL to update.
