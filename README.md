# Agent Control Panel SDK for Python

Connect an AI agent that already runs in your app to [Agent Control Panel](https://acp.skaigroup.tech)
(ACP): report every run (tokens, model, latency, errors, the answer) and let your team **pause, resume, stop,
restart, swap the model or push an approved prompt** without redeploying.

- **Fail-open.** If ACP is slow or down, your agent keeps working. Reporting is buffered and retried on a
  background thread; nothing the SDK does raises into your code (except `AgentPausedError`, on purpose).
- **Signed control.** Commands are HMAC-SHA256 signed with a timestamp and a single-use nonce; unsigned,
  replayed, stale or wrong-agent commands are rejected.
- **Durable pause and model swaps.** A pause or a model swap set in ACP survives restarts, extra workers and
  missed webhooks: every heartbeat reply carries the state ACP holds, and the client adopts it.
- **Zero runtime dependencies.** Python 3.10+, standard library only, fully typed (`py.typed`).

Same behaviour and wire contract as the Node SDK (`agent-control-panel` on npm); both are tested against
the same contract vectors.

```bash
pip install agent-control-panel
```

## Quickstart

```python
import os
from agent_control_panel import AgentPausedError, create_client

acp = create_client(
    url=os.environ["ACP_URL"],                      # your control panel URL
    api_key=os.environ["ACP_API_KEY"],              # shown once when the key is created
    agent_id=os.environ["ACP_AGENT_ID"],            # from the app page in ACP
    webhook_secret=os.environ["ACP_WEBHOOK_SECRET"],
)
acp.start()  # heartbeats every 30 s, flushes every 5 s, on daemon threads

def handle(question: str) -> str:
    try:
        return acp.wrap_agent_run(
            lambda: my_agent(question, system_prompt=acp.fetch_prompt() or DEFAULT_PROMPT),
            user_input=question,
            model_used=acp.get_effective_model("gpt-4o-mini"),
        )
    except AgentPausedError:
        return "This assistant is paused. Please try again later."
```

`wrap_agent_run` records duration, success or error, token usage and the answer text, read from the
response objects of the OpenAI, Anthropic and Google Gemini SDKs (and plain dicts of the same shape). For
anything else, return a string or call `report_run(...)` yourself. While the agent is paused or stopped it
raises `AgentPausedError` **without calling your function**, so a pause really stops model calls.

ACP's quality reviews, Test Lab and error explanations read the user input, the full input and the output,
so pass `user_input` and `full_input` for accurate results (each field is kept up to 100,000 characters). To
send metadata only, create the client with `capture_content=False`.

### OpenAI

```python
from openai import OpenAI

openai = OpenAI()

def answer(question: str) -> str:
    model = acp.get_effective_model("gpt-4o-mini")
    messages = [
        {"role": "system", "content": acp.fetch_prompt() or "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    completion = acp.wrap_agent_run(
        lambda: openai.chat.completions.create(model=model, messages=messages),
        user_input=question,
        full_input=messages,
        model_used=model,
    )
    return completion.choices[0].message.content or ""
```

### Anthropic (async)

```python
import asyncio
from anthropic import AsyncAnthropic

claude = AsyncAnthropic()

async def answer(question: str) -> str:
    model = acp.get_effective_model("claude-sonnet-4-5")
    system = await asyncio.to_thread(acp.fetch_prompt)  # fetch_prompt blocks up to 5 s
    messages = [{"role": "user", "content": question}]
    message = await acp.wrap_agent_run_async(
        lambda: claude.messages.create(model=model, max_tokens=1024, system=system, messages=messages),
        user_input=question,
        full_input=messages,
        model_used=model,
    )
    return "".join(block.text for block in message.content if block.type == "text")
```

Anthropic prompt-cache tokens are counted as input tokens.

## Receive control commands

ACP sends signed commands to `<Deployed URL>/api/acp/webhook` (set the Deployed URL on the app page).
Pass the **raw request body** (bytes), not re-serialised JSON: the signature covers the exact bytes.
`webhook_response` returns the HTTP status (200 applied, 409 for another agent's command, 401 otherwise) and
a JSON body.

**FastAPI**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/api/acp/webhook")
async def acp_webhook(request: Request) -> JSONResponse:
    status, body = acp.webhook_response(request.headers, await request.body())
    return JSONResponse(body, status_code=status)
```

**Flask**

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.post("/api/acp/webhook")
def acp_webhook():
    status, body = acp.webhook_response(request.headers, request.get_data())
    return jsonify(body), status
```

**Django**

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

@csrf_exempt  # ACP requests are authenticated by their HMAC signature, not a CSRF token
@require_POST
def acp_webhook(request):
    status, body = acp.webhook_response(request.headers, request.body)
    return JsonResponse(body, status=status)
```

No public URL (local development, a worker, a cron job)? Control still works: pause, stop, resume and model
swaps arrive with the next heartbeat (within 30 s), and prompt changes are fetched before each run.

## API

| Member | What it does |
|---|---|
| `create_client(...)` | `url`, `api_key`, `agent_id`, `webhook_secret` (required); `webhook_secret_prev` during rotation; `heartbeat_seconds` (30), `flush_seconds` (5), `flush_at` (50), `max_buffer` (200); `logger`; `disable_control`; `capture_content` (default `True`; `False` = metadata only); `allow_insecure_http` (development only); `transport` (for tests). `url` must be https (or http to localhost). Raises `ValueError` for a missing credential or an insecure URL. |
| `start()` / `stop(timeout=10)` | Start heartbeats and background flushing / flush everything and stop, giving up after `timeout`. The threads never keep your process alive; pending runs are flushed at interpreter exit. `with create_client(...) as acp:` does both. |
| `wrap_agent_run(fn, *, user_input, full_input, input_summary, model_used)` | Run `fn()`, report it (with the answer text from the result), re-raise its exception unchanged. Raises `AgentPausedError` while paused or stopped. |
| `wrap_agent_run_async(fn, ...)` | The same for coroutines: `await acp.wrap_agent_run_async(lambda: client.create(...))`. |
| `report_run(status, **fields)` | Report a run yourself (tokens, model, timings, summaries, `user_input`, `full_input`, `full_output`, `error_message`, `trigger`, optional `dedupe_key`). Buffered; never raises. |
| `extract_text(result)` / `extract_tokens(result)` | The answer text / token counts of an OpenAI, Anthropic or Gemini response, or `None`. |
| `fetch_prompt()` | The agent's current prompt from ACP (composed with its skills), cached 60 s; falls back to the last prompt on any error. |
| `get_effective_model(fallback)` | The model ACP asked for (after a model swap), else `fallback`. |
| `status` | `idle`, `running`, `paused` or `stopped`. |
| `handle_webhook(headers, raw_body)` / `webhook_response(...)` | Verify and apply a signed command / the same as an HTTP status and JSON body. |
| `apply_command(command, payload=None)` | Apply a command locally (e.g. your own kill switch). A pause set this way is never lifted by ACP. |

## Security

- Never commit `ACP_API_KEY` or `ACP_WEBHOOK_SECRET`; keep them in your platform's secrets.
- `ACP_DISABLE_CONTROL=1` (or `disable_control=True`) ignores every control command while reporting keeps
  working: a local kill switch that does not depend on ACP.
- Rotate the webhook secret in ACP, then set the old one as `webhook_secret_prev` until every instance has
  the new one.
- Commands are accepted only within 5 minutes of their timestamp and only once (nonce cache, safe across
  threads).
- Run content (user input, full input, the answer) is sent to ACP by default, cut to 100,000 characters per
  field. Leave out anything you must not share, or use `capture_content=False` for metadata only.
- Redirects are never followed: requests carry the API key, so a redirect is logged with where ACP moved
  ("ACP moved to ...; update your url") and the request fails open.
- TLS certificates are verified with the system's default trust store.

## Troubleshooting

| Log line | Meaning | Fix |
|---|---|---|
| `-> 401: ACP rejected the API key` | The key is wrong, revoked or expired. | Rotate the key on the app page, copy the full key from the dialog, set `ACP_API_KEY`. |
| `ACP moved to https://...; update your url` | ACP answered with a redirect. | Set `ACP_URL` to the address in the message. |
| `reason: bad_signature` in the webhook answer | `ACP_WEBHOOK_SECRET` differs from the app's secret in ACP, or the body was re-serialised before verification. | Reveal the secret on the app page and set it again; pass the raw body. |
| `control rejected: command for agent ... reached agent ...` | Several agents share one webhook URL but this client is configured for another agent. | One client (and `agent_id`) per agent. |
| `CERTIFICATE_VERIFY_FAILED` | Python cannot find root certificates (common with python.org installers on macOS). | Run "Install Certificates.command" from the Python folder, or install your OS's CA bundle. |

Warnings are logged to the `agent_control_panel` logger and reach stderr unless you configure logging.

**Forking servers (gunicorn with `--preload`, multiprocessing).** Threads do not survive `fork()`. The SDK
resets itself in the child and restarts its threads on the next report, and runs buffered before the fork
stay with the parent. Creating the client in each worker (e.g. gunicorn's `post_fork` hook) is simplest.

**Serverless and short scripts.** Call `acp.stop()` (or use `with create_client(...) as acp:`) before the
function returns so buffered runs are sent; frozen containers do not run exit hooks.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements-dev.txt
ruff check src tests && ruff format --check src tests && mypy && pytest
python -m build
```

The SDK is developed in the Agent Control Panel repository, next to the server it talks to, so contract
changes are tested on both sides together (`tests/contract/vectors.json` is generated there). This public
repository receives one commit per release. Issues and pull requests are welcome here; accepted changes are
applied upstream and ship in the next release.

## License

[Apache License 2.0](LICENSE). It includes an explicit patent grant from contributors, and ends that grant
for anyone who sues over patents in the SDK.
