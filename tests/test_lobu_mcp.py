"""Exercise the memory client's HTTP boundary without model/provider credentials."""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))
import lobu_memory_agent
from lobu_mcp import LobuMemoryClient

from state_bench.client import LLMClient


class MemoryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.envelope = {"result": {"content": []}}
        self.require_content_limit = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                result = {"result": {}} if body["method"] != "tools/call" else owner.envelope
                if body["method"] == "tools/call" and owner.require_content_limit is not None:
                    if body["params"]["arguments"].get("content_limit") != owner.require_content_limit:
                        result = {"result": {"content": []}}
                data = json.dumps({"jsonrpc": "2.0", "id": body.get("id"), **result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("mcp-session-id", "synthetic-session")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = LobuMemoryClient(f"http://127.0.0.1:{self.server.server_port}/mcp", "synthetic-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_empty_memory_is_a_valid_result(self):
        self.assertEqual(self.client.retrieve("flight changes"), [])

    def test_missing_content_is_not_empty_memory(self):
        self.envelope = {"result": {}}
        with self.assertRaisesRegex(RuntimeError, "invalid content"):
            self.client.retrieve("flight changes")

    def test_non_array_content_is_not_empty_memory(self):
        for value in (None, {}, "", False, 0):
            with self.subTest(content=value):
                self.envelope = {"result": {"content": value}}
                with self.assertRaisesRegex(RuntimeError, "invalid content"):
                    self.client.retrieve("flight changes")

    def test_malformed_content_blocks_do_not_become_empty_memory(self):
        blocks = [
            None,
            [],
            "",
            {},
            {"type": None},
            {"type": ""},
            {"type": "text"},
            {"type": "text", "text": None},
            {"type": "text", "text": 0},
            {"type": "text", "text": []},
        ]
        for block in blocks:
            with self.subTest(block=block):
                self.envelope = {"result": {"content": [block]}}
                with self.assertRaises(RuntimeError):
                    self.client.retrieve("flight changes")

    def test_explicit_empty_text_is_valid(self):
        self.envelope = {"result": {"content": [{"type": "text", "text": ""}]}}
        self.assertEqual(self.client.retrieve("flight changes"), [])

    def test_json_rpc_failure_is_not_empty_memory(self):
        self.envelope = {"error": {"code": -32603, "message": "synthetic upstream outage"}}
        with self.assertRaisesRegex(RuntimeError, "MCP RPC error -32603"):
            self.client.retrieve("flight changes")

    def test_tool_failure_is_not_empty_memory(self):
        self.envelope = {"result": {"isError": True, "content": [{"type": "text", "text": "synthetic failure"}]}}
        with self.assertRaisesRegex(RuntimeError, "MCP tool search_memory failed"):
            self.client.retrieve("flight changes")

    def test_successful_learning_survives_the_http_boundary(self):
        self.envelope = {
            "result": {
                "content": [{"type": "text", "text": "> PROCEDURAL LEARNING: Confirm consent before cancellation."}]
            }
        }
        self.assertEqual(
            self.client.retrieve("flight changes"), ["PROCEDURAL LEARNING: Confirm consent before cancellation."]
        )

    def test_agent_tool_propagates_outage_and_can_retry_without_caching_failure(self):
        variables = {
            "SYNTHETIC_MEMORY_TEST_KEY": "synthetic-key",
            "LOBU_MCP_URL": self.client.url,
            "LOBU_PAT": "synthetic-token",
            "STATE_BENCH_AGENT_BASE_URL": "",
            "OPENAI_BASE_URL": "",
        }
        with patch.dict(os.environ, variables):
            client = LLMClient(provider="openai", deployment="synthetic-model", api_key_var="SYNTHETIC_MEMORY_TEST_KEY")
            agent = lobu_memory_agent.LobuMemoryAgent(client, "Synthetic task", [], {})
            lobu_memory_agent._CLIENT = None
            try:
                handler = agent.tool_handlers["retrieve_learnings"]
                for result in (
                    {"isError": True, "content": []},
                    {},
                    {"content": None},
                    {"content": [{"type": "text"}]},
                ):
                    self.envelope = {"result": result}
                    with self.assertRaises(RuntimeError):
                        handler({"query": "flight changes"})
                self.envelope = {
                    "result": {"content": [{"type": "text", "text": "> PROCEDURAL LEARNING: Retry proof."}]}
                }
                self.assertEqual(
                    handler({"query": "flight changes"}), {"learnings": ["PROCEDURAL LEARNING: Retry proof."]}
                )
            finally:
                lobu_memory_agent._CLIENT = None
                client._client.close()

    def test_candidate_budget_applies_to_content_retrieval(self):
        self.require_content_limit = 10
        self.envelope = {
            "result": {"content": [{"type": "text", "text": "> PROCEDURAL LEARNING: Content budget proof."}]}
        }
        self.assertEqual(
            self.client.retrieve("flight changes", candidate_limit=10), ["PROCEDURAL LEARNING: Content budget proof."]
        )


if __name__ == "__main__":
    unittest.main()
