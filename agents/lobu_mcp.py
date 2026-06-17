"""Minimal Lobu MCP memory client (stdlib only) for the Agent Learning Track.

Talks to a local Lobu server's /mcp endpoint via JSON-RPC over Streamable HTTP:
initialize -> notifications/initialized -> tools/call. Used to save procedural
learnings (offline) and retrieve them (at eval time) via save_memory / search_memory.

Config via env: LOBU_MCP_URL (default http://127.0.0.1:8787/mcp), LOBU_PAT (Bearer token).
Import-safe: no network at import; the MCP session is established lazily.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

LEARNING_MARKER = "PROCEDURAL LEARNING"


class LobuMemoryClient:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = url or os.environ.get("LOBU_MCP_URL", "http://127.0.0.1:8787/mcp")
        self.token = token or os.environ.get("LOBU_PAT", "")
        self._session_id: str | None = None
        self._rpc_id = 0

    def _post(self, body: dict, extra_headers: dict | None = None) -> tuple[dict, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.token}",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            sid = resp.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
            raw = resp.read().decode()
        return self._parse(raw), raw

    @staticmethod
    def _parse(raw: str) -> dict:
        # Streamable-HTTP returns SSE ("event: message\ndata: {json}") or plain JSON.
        m = re.search(r"data:\s*(\{.*\})", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raw = raw.strip()
        return json.loads(raw) if raw.startswith("{") else {}

    def _ensure_session(self) -> None:
        if self._session_id:
            return
        self._rpc_id += 1
        self._post(
            {
                "jsonrpc": "2.0",
                "id": self._rpc_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "statebench-lobu", "version": "1.0"},
                },
            }
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _call_tool(self, name: str, arguments: dict) -> str:
        self._ensure_session()
        self._rpc_id += 1
        result, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._rpc_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        content = (result.get("result") or {}).get("content") or []
        return "".join(c.get("text", "") for c in content if c.get("type") == "text")

    def save_learning(self, content: str) -> None:
        self._call_tool("save_memory", {"semantic_type": "note", "content": content, "metadata": {}})

    def retrieve(self, query: str, top_k: int = 3, candidate_limit: int = 10) -> list[str]:
        text = self._call_tool("search_memory", {"query": query, "limit": candidate_limit})
        # "Related Content" entries render as `> <content>` blockquotes.
        learnings: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                body = line.lstrip("> ").strip()
                if LEARNING_MARKER in body and body not in learnings:
                    learnings.append(body)
        return learnings[:top_k]
