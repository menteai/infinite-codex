from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import ensure_default_config, load_config
from .db import MemoryDB
from .ingest import Ingester

mcp = FastMCP("infinite-memory")


def _ingester() -> Ingester:
    ensure_default_config()
    return Ingester(load_config())


@mcp.tool()
async def memory_ingest(force: bool = False) -> dict[str, Any]:
    """Index Codex session history into the vector memory database."""
    return await _ingester().ingest(force=force)


def _format_memory_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    blocks = []
    for idx, result in enumerate(results, 1):
        content = str(result.get("content") or "").strip()
        blocks.append(f"Memory {idx}:\n{content}")
    return "\n\n".join(blocks)


async def _search_memory(query: str, session_id: str, limit: int = 5) -> str:
    if not session_id:
        return ""
    ingester = _ingester()
    results = await ingester.search(query=query, session_id=session_id, limit=limit)
    return _format_memory_results(results)


@mcp.tool()
async def infinite_memory_search(query: str, session_id: str, limit: int = 5) -> str:
    """Infinite Memory search. Use when the user says infinite search, search infinite memory, search previous turns, or asks to recall prior context in this session."""
    return await _search_memory(query=query, session_id=session_id, limit=limit)


@mcp.tool()
def memory_stats() -> dict[str, Any]:
    """Return database and index statistics."""
    ensure_default_config()
    cfg = load_config()
    stats = MemoryDB(cfg.db_path).stats()
    stats["embedding_backend"] = cfg.embedding.backend
    stats["embedding_model"] = cfg.embedding.model
    stats["codex_sessions_dir"] = str(cfg.codex_sessions_dir)
    return stats


@mcp.tool()
def memory_get(chunk_id: int, session_id: str) -> str:
    """Return a chunk only when it belongs to the specified Codex session."""
    ensure_default_config()
    result = MemoryDB(load_config().db_path).get_chunk(chunk_id, session_id=session_id)
    return str(result.get("content") or "") if result else ""


@mcp.tool()
def memory_forget_session(session_id: str) -> dict[str, Any]:
    """Delete all indexed memories for a Codex session id."""
    ensure_default_config()
    deleted = MemoryDB(load_config().db_path).delete_session(session_id)
    return {"session_id": session_id, "deleted_chunks": deleted}


def main() -> None:
    ensure_default_config()
    mcp.run()


if __name__ == "__main__":
    main()
