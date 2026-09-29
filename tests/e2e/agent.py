"""A tiny agent driven over stdin by server/__tests__/sdk-python-e2e.test.ts (in the main repository).

It runs the real SDK against a real ACP test server: reports runs, sends heartbeats and receives signed
commands on a local webhook endpoint. One JSON reply line per command line.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
from typing import Any

from agent_control_panel import AgentPausedError, create_client

acp = create_client(
    url=os.environ["ACP_URL"],
    api_key=os.environ["ACP_API_KEY"],
    agent_id=os.environ["ACP_AGENT_ID"],
    webhook_secret=os.environ["ACP_WEBHOOK_SECRET"],
    heartbeat_seconds=3600,
    flush_seconds=3600,
)


class Webhook(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        code, body = acp.webhook_response(self.headers, raw)
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args: Any) -> None:
        pass


def main() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Webhook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    acp.start()
    print(json.dumps({"ready": True, "webhook_port": server.server_address[1]}), flush=True)

    for line in sys.stdin:
        command = line.strip()
        reply: dict[str, Any] = {"ok": True}
        if command == "run":
            try:
                acp.wrap_agent_run(
                    lambda: {
                        "choices": [{"message": {"content": "Refund approved ✓"}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    },
                    input_summary="e2e run",
                    user_input="Can I get a refund?",
                    full_input=[{"role": "user", "content": "Can I get a refund?"}],
                    model_used=acp.get_effective_model("e2e-model"),
                )
            except AgentPausedError as err:
                reply = {"ok": False, "paused": err.status}
        elif command == "flush":
            acp.flush()
        elif command == "heartbeat":
            acp._heartbeat()
        elif command == "prompt":
            reply["prompt"] = acp.fetch_prompt()
            reply["version"] = acp.cached_prompt_version
        elif command == "quit":
            acp.stop()
            server.shutdown()
            print(json.dumps(reply), flush=True)
            return
        reply.update(status=acp.status, model=acp.get_effective_model())
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
