"""FullHistoryAgent — StateBenchAgent that sends full conversation history each
turn instead of chaining on previous_response_id.

Why: the locked simulator/judge are stateless (complete_chat / complete_json), but
the built-in StateBenchAgent.act() inner tool loop chains on previous_response_id.
When the Responses API is reached through a LiteLLM /responses -> /chat/completions
bridge (required for any non-OpenAI backend, e.g. z.ai GLM), the bridge drops the
prior user turn on previous_response_id continuations, so the provider rejects the
reconstructed conversation. Sending the full input array each call (the portable,
provider-agnostic pattern) sidesteps the bridge's stateful path entirely.

Behaviour is otherwise identical to StateBenchAgent, including the optional
retrieve_learnings memory hook (enabled when a subclass defines retrieve_learnings).
"""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.state_bench import StateBenchAgent

_MAX_TOOL_ROUNDS = 12


class FullHistoryAgent(StateBenchAgent):
    """No-memory baseline agent compatible with bridged (non-OpenAI) backends."""

    def act(self, conversation: list[Any]) -> tuple[str, list[dict[str, Any]], list[Any]]:
        all_tool_calls: list[dict[str, Any]] = []
        raw_items: list[Any] = []

        # Growing Responses-API input array. No previous_response_id is ever sent.
        items: list[Any] = list(self.prepare_conversation(conversation))
        final_text = ""

        for _ in range(_MAX_TOOL_ROUNDS):
            response = self.client.complete_with_tools(
                instructions=self.system_prompt,
                input=items,
                tools=self.tools,
                reasoning_effort=self.agent_reasoning_effort,
            )
            if response.usage:
                self.total_output_tokens += response.usage.output_tokens
                self.add_response_usage(response.usage, category="agent_turn")

            raw_items.extend(response.output)
            tool_calls = [it for it in response.output if getattr(it, "type", None) == "function_call"]
            if not tool_calls:
                final_text = response.output_text or ""
                break

            for tc in tool_calls:
                args = json.loads(tc.arguments)
                handler = self.tool_handlers.get(tc.name)
                if handler is None:
                    result: Any = {"error": f"Unknown tool: {tc.name}"}
                else:
                    result = handler(args)
                    all_tool_calls.append({"name": tc.name, "arguments": args, "result": result})

                # Re-emit clean function_call + function_call_output items into the
                # growing input so the next call carries the entire turn explicitly.
                items.append(
                    {"type": "function_call", "call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments}
                )
                output_item = {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
                items.append(output_item)
                raw_items.append(output_item)

        return final_text, all_tool_calls, raw_items
