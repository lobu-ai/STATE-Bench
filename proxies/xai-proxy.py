"""Thin reverse proxy: STATE-Bench (/openai/v1/*) -> api.x.ai/v1/* with the
SuperGrok OAuth bearer injected. xAI natively supports the Responses API, so this
is a passthrough (no Responses->ChatCompletions bridge needed). The token is
re-read from ~/.grok/auth.json on every request, so CLI refreshes propagate live.
"""

import http.server
import json
import os
import socketserver
import urllib.error
import urllib.request

AUTH = os.path.expanduser("~/.grok/auth.json")
UPSTREAM = "https://api.x.ai/v1"


def token() -> str:
    d = json.load(open(AUTH))
    return list(d.values())[0]["key"]


class H(http.server.BaseHTTPRequestHandler):
    def _proxy(self):
        path = self.path
        for pre in ("/openai/v1", "/v1"):
            if path.startswith(pre):
                path = path[len(pre):]
                break
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(UPSTREAM + path, data=body, method=self.command)
        req.add_header("Authorization", f"Bearer {token()}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data, code = r.read(), r.status
        except urllib.error.HTTPError as e:
            data, code = e.read(), e.code
        except Exception as e:  # noqa: BLE001
            data, code = json.dumps({"error": str(e)}).encode(), 502
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_POST = _proxy
    do_GET = _proxy

    def log_message(self, *a):
        pass


class TS(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    TS(("127.0.0.1", 4000), H).serve_forever()
