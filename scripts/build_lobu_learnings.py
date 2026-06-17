"""Offline learning extraction for the Agent Learning Track (Lobu memory).

Reads N train trajectories for a domain, distills each into one generalizable
procedural rule via the agent model (GLM through the LiteLLM bridge), and saves
it into Lobu memory via save_memory. Only train trajectories are used (no test
oracle), per the official Agent Learning Track rules.

Usage:
  uv run python scripts/build_lobu_learnings.py --domain travel --limit 25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
from lobu_mcp import LobuMemoryClient  # noqa: E402

DISTILL_INSTRUCTIONS = (
    "You analyze a successful customer-service agent transcript and extract ONE concise, "
    "generalizable procedural rule that would help an agent on similar future tasks: tool-use "
    "order, policy checks, consent steps, or facts to proactively verify before acting. "
    "Output exactly one imperative sentence. No preamble, no markdown, no quotes."
)


def compact_conversation(conv: list, max_chars: int = 6000) -> str:
    parts: list[str] = []
    for msg in conv:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict)) or str(content)
        line = f"{role}: {content}"
        for tc in msg.get("tool_calls", []) or []:
            line += f" [tool {tc.get('name')}({json.dumps(tc.get('arguments', {}))[:200]})]"
        parts.append(line[:800])
    text = "\n".join(parts)
    return text[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="travel")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--base-url", default="http://localhost:4000/openai/v1/")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="sk-statebench-local")
    mem = LobuMemoryClient()

    # Draw from the official train split order (diverse scenario coverage:
    # cancel/book/change/policy/cascade/...), restricted to train trajectories that exist.
    split = json.load(open(f"state_bench/domains/{args.domain}/splits/train_test.json"))
    train_ids = split["splits"]["train"]
    files = []
    for tid in train_ids:
        p = f"datasets/train_task_trajectories/{args.domain}/{tid}.json"
        if os.path.exists(p):
            files.append(p)
        if len(files) >= args.limit:
            break
    print(f"Distilling {len(files)} {args.domain} train trajectories -> Lobu memory")
    saved = 0
    for i, f in enumerate(files, 1):
        try:
            conv = json.load(open(f)).get("conversation", [])
            transcript = compact_conversation(conv)
            resp = client.responses.create(
                model=args.model,
                instructions=DISTILL_INSTRUCTIONS,
                input=[{"role": "user", "content": transcript}],
            )
            rule = (resp.output_text or "").strip().strip('"')
            if not rule:
                print(f"  [{i}/{len(files)}] {os.path.basename(f)}: empty, skipped")
                continue
            mem.save_learning(f"PROCEDURAL LEARNING ({args.domain}): {rule}")
            saved += 1
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}: {rule[:90]}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}: ERROR {str(e)[:120]}")
    print(f"Done: saved {saved} learnings into Lobu memory.")


if __name__ == "__main__":
    main()
