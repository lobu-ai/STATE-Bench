# Lobu adapter for STATE-Bench

Run STATE-Bench against **Lobu memory** as the Agent Learning Track memory system,
using **only z.ai (GLM)** — no OpenAI/Azure key required.

This adapter adds three things to upstream STATE-Bench, with no changes to the
harness itself (all files live in the repo-root `agents/` and `scripts/` extension
dirs that the loader already scans):

| File | Role |
| --- | --- |
| `agents/full_history_agent.py` | No-memory baseline agent. Same loop as `StateBenchAgent` but sends full conversation history each turn instead of chaining on `previous_response_id`. |
| `agents/lobu_memory_agent.py` | Agent Learning Track agent: `FullHistoryAgent` + a read-only `retrieve_learnings` hook backed by Lobu `search_memory`. |
| `agents/lobu_mcp.py` | Minimal stdlib MCP client for Lobu (`/mcp` JSON-RPC: initialize → tools/call). |
| `scripts/build_lobu_learnings.py` | Offline learning extraction: distills train trajectories into procedural rules via the agent model and saves them with Lobu `save_memory`. |
| `litellm.config.yaml` | LiteLLM proxy config that bridges the OpenAI Responses API → Chat Completions for a non-OpenAI backend (GLM via z.ai, or real `gpt-5.4` via OpenRouter). |

## Why the bridge + `FullHistoryAgent`

STATE-Bench's harness uses the OpenAI **Responses API** (`responses.create` +
`previous_response_id`), which only OpenAI and Azure serve natively. To run on any
other backend you put a **LiteLLM proxy** in front that translates Responses →
Chat Completions (`use_chat_completions_api: true`).

Two gotchas this adapter handles:

1. **z.ai key = GLM Coding Plan** → base URL must be `https://api.z.ai/api/coding/paas/v4`
   (not `/api/paas/v4`, which reports a misleading "insufficient balance").
2. **The bridge drops the prior user turn on `previous_response_id` continuations**,
   so GLM rejects the reconstructed conversation. `FullHistoryAgent` sends the full
   input array every turn and never chains — the portable pattern. The locked
   simulator/judge are stateless (`complete_chat`/`complete_json`) so they bridge cleanly.

## Run it (z.ai only)

```bash
# 0) deps
uv sync && cp .env.lobu.example .env   # fill in / point eval+agent at the proxy

# 1) LiteLLM bridge: Responses API -> Chat Completions, backed by GLM coding endpoint.
#    Run under SERVER_ROOT_PATH=/openai so it matches STATE-Bench's azure URL munging.
export Z_AI_API_KEY=...           # GLM Coding Plan key
SERVER_ROOT_PATH=/openai litellm --config litellm.config.yaml --port 4000

# 2) (memory track) start a LOCAL Lobu, mint a token, build learnings
#    Lobu memory embeds in-process (bge-base-en-v1.5) — no embeddings key needed.
PAT=$(curl -s -X POST http://127.0.0.1:8787/api/local-init -H 'X-Lobu-Client: cli' | python -c 'import json,sys;print(json.load(sys.stdin)["device_token"])')
export LOBU_PAT="$PAT" LOBU_MCP_URL="http://127.0.0.1:8787/mcp"
uv run python scripts/build_lobu_learnings.py --domain travel --limit 30

# 3a) baseline (no memory)
uv run python -m state_bench.scripts.run_batch --domain travel \
  --agent-class FullHistoryAgent --agent-model-name glm-4.6 --num-runs 5 --num-workers 2

# 3b) Lobu memory
uv run python -m state_bench.scripts.run_batch --domain travel \
  --agent-class LobuMemoryAgent --agent-model-name glm-4.6 \
  --retrieve-learnings-top-k 3 --num-runs 5 --num-workers 2

# 4) metrics
uv run python -m state_bench.scripts.compute_metrics --domain travel \
  --results-dir outputs/travel --num-runs 5
```

For leaderboard-**comparable** scores, point `litellm.config.yaml` at the real
`openai/gpt-5.4` via OpenRouter instead of GLM (the simulator/judge are protocol-locked
to GPT-5.4). GLM gives a directional, internal signal only.

> Note: z.ai's GLM Coding Plan rate-limits aggressively; keep `--num-workers` at 1–2.
