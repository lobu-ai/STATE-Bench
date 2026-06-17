"""LobuMemoryAgent — Agent Learning Track agent backed by Lobu memory.

Bridge-safe agent loop (FullHistoryAgent) + a read-only `retrieve_learnings`
hook that queries Lobu's search_memory over MCP. Offline learning extraction
(save_memory) is done by scripts/build_lobu_learnings.py, not here.
"""

from __future__ import annotations

import os
import sys

# Loader imports agent files by path; add this dir so sibling modules resolve.
sys.path.insert(0, os.path.dirname(__file__))

# Alias so the loader doesn't see a second "FullHistoryAgent" definition here.
from full_history_agent import FullHistoryAgent as _FullHistoryAgent  # noqa: E402
from lobu_mcp import LobuMemoryClient  # noqa: E402

_CLIENT: LobuMemoryClient | None = None


def _client() -> LobuMemoryClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LobuMemoryClient()
    return _CLIENT


class LobuMemoryAgent(_FullHistoryAgent):
    """FullHistoryAgent + Lobu-backed procedural-learning retrieval."""

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        try:
            return _client().retrieve(query, top_k=top_k)
        except Exception:
            # Retrieval is best-effort; never fail a task on a memory error.
            return []
