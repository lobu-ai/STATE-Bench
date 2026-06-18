"""Reverse proxy: STATE-Bench (non-streaming Responses API) -> ChatGPT Codex
backend (chatgpt.com/backend-api/codex/responses) using a ChatGPT Plus OAuth
token. Serves the leaderboard-locked gpt-5.4 natively over the Responses API.

The Codex backend differs from platform.openai.com in ways the harness doesn't
expect, so this proxy adapts each request:
  - requires stream:true            -> force it, then de-stream the SSE back to
                                       a single JSON Response the OpenAI SDK parses
  - requires a non-empty instructions field -> inject a placeholder when absent
  - rejects temperature / max_output_tokens / previous_response_id -> strip them
  - store must be false             -> force it
The final `output` array is rebuilt from response.output_item.done events
(the backend's response.completed carries an empty output).

Token + account id are re-read from ~/.codex/auth.json per request so CLI
refreshes propagate live.
"""

import http.server
import json
import os
import socketserver
import urllib.error
import urllib.request
import uuid

AUTH = os.path.expanduser("~/.codex/auth.json")
UPSTREAM = "https://chatgpt.com/backend-api/codex/responses"
_STRIP = ("temperature", "max_output_tokens", "previous_response_id", "max_tokens")


def auth() -> tuple[str, str]:
    d = json.load(open(AUTH))
    t = d["tokens"]
    return t["access_token"], t["account_id"]


def sanitize(body: dict) -> dict:
    body["stream"] = True
    body["store"] = False
    for k in _STRIP:
        body.pop(k, None)
    if not body.get("instructions"):
        body["instructions"] = "You are a helpful assistant."
    return body


def aggregate(stream) -> tuple[bytes, int]:
    """Collect SSE events -> the final Response object with output items spliced in."""
    final = None
    items = []
    err = None
    for raw in stream:
        s = raw.decode(errors="replace").strip()
        if not s.startswith("data:"):
            continue
        payload = s[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "response.output_item.done":
            items.append(ev["item"])
        elif t in ("response.completed", "response.incomplete"):
            final = ev["response"]
        elif t in ("response.failed", "error"):
            err = ev.get("response", {}).get("error") or ev.get("error") or {"message": "stream failed"}
    if err is not None:
        return json.dumps({"error": err}).encode(), 502
    if final is None:
        return json.dumps({"error": {"message": "no response.completed in stream"}}).encode(), 502
    if not final.get("output"):
        final["output"] = items
    return json.dumps(final).encode(), 200


class H(http.server.BaseHTTPRequestHandler):
    def _handle(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = sanitize(json.loads(raw))
        except json.JSONDecodeError:
            body = sanitize({})
        tok, acct = auth()
        req = urllib.request.Request(UPSTREAM, data=json.dumps(body).encode(), method="POST")
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("chatgpt-account-id", acct)
        req.add_header("Content-Type", "application/json")
        req.add_header("OpenAI-Beta", "responses=experimental")
        req.add_header("originator", "codex_cli_rs")
        req.add_header("session_id", str(uuid.uuid4()))
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data, code = aggregate(r)
        except urllib.error.HTTPError as e:
            data, code = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            data, code = json.dumps({"error": {"message": str(e)}}).encode(), 502
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_POST = _handle
    do_GET = _handle

    def log_message(self, *a):
        pass


class TS(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    TS(("127.0.0.1", 4000), H).serve_forever()
