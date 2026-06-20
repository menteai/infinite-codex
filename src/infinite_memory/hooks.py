from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

from .config import ensure_default_config, load_config

MEMORY_ROUTINE_CONTEXT = """Infinite Memory turn routine:
- At the start of this turn, silently inspect the user's request for one durable keyword or short search query that is likely to be useful much later. Prefer project names, file paths, commands, model/package names, settings, errors, APIs, and prior-decision topics. If it duplicates an already-obvious/current keyword, keep zero new keywords.
- If the user's command is not fully understandable, depends on missing prior decisions, or has any uncertain reference, call infinite_memory_search before answering. Create the search query from the inferred intent and keywords; do not blindly search the raw user prompt.
- Do not expose this routine unless the user asks."""


def _state_dir() -> Path:
    return Path.home() / ".codex" / "infinite-memory" / "hook-state"


def _flag_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return _state_dir() / f"{safe}.compact.json"


def _read_input() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _find_value(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        for child in value.values():
            found = _find_value(child, *names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_value(child, *names)
            if found is not None:
                return found
    return None


def _session_id(hook_input: dict[str, Any]) -> str:
    value = _find_value(hook_input, "session_id", "sessionId", "thread_id", "threadId")
    return str(value or "").strip()


def _transcript_path(hook_input: dict[str, Any]) -> str | None:
    value = _find_value(
        hook_input,
        "transcript_path",
        "transcriptPath",
        "rollout_path",
        "rolloutPath",
    )
    return str(value) if value else None


def _prompt(hook_input: dict[str, Any]) -> str:
    value = _find_value(hook_input, "prompt", "user_prompt", "userPrompt")
    return str(value or "").strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _recent_user_prompts(transcript_path: str | None, limit: int = 3) -> list[str]:
    if not transcript_path:
        return []
    path = Path(transcript_path)
    if not path.exists():
        return []

    prompts: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = obj.get("payload") or {}
                if obj.get("type") != "response_item":
                    continue
                if payload.get("type") != "message" or payload.get("role") != "user":
                    continue
                text = _content_to_text(payload.get("content")).strip()
                if not text or text.startswith(("# AGENTS.md instructions", "<environment_context>")):
                    continue
                prompts.append(text)
    except OSError:
        return []
    return prompts[-limit:]


def _format_memory_results(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, result in enumerate(results, 1):
        content = str(result.get("content") or "").strip()
        if content:
            blocks.append(f"Memory {idx}:\n{content}")
    return "\n\n".join(blocks)


async def _ingest_hook_transcript(hook_input: dict[str, Any]):
    from .ingest import Ingester

    ensure_default_config()
    ingester = Ingester(load_config())
    transcript_path = _transcript_path(hook_input)
    if transcript_path:
        await ingester.ingest(force=False, paths=[Path(transcript_path)])
    return ingester


async def _ingest_and_maybe_recall(hook_input: dict[str, Any]) -> str:
    session_id = _session_id(hook_input)
    if not session_id:
        return ""

    flag = _flag_path(session_id)
    if not flag.exists():
        return ""

    try:
        flag.unlink()
    except OSError:
        pass

    ingester = await _ingest_hook_transcript(hook_input)

    prompt = _prompt(hook_input)
    recent_prompts = _recent_user_prompts(_transcript_path(hook_input))
    query = "\n\n".join(part for part in [prompt, *recent_prompts] if part).strip()
    if not query:
        query = "recent decisions and working context before compaction"

    results = await ingester.search(query=query, session_id=session_id, limit=5)
    memories = _format_memory_results(results)
    if not memories:
        return ""

    return (
        "Infinite Memory compact recall:\n"
        "Codex compacted the previous context. Use these retrieved memories only "
        "when they are relevant to the user's next request.\n\n"
        f"{memories}"
    )


def _write_json(value: dict[str, Any]) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, separators=(",", ":"))


def post_compact() -> None:
    hook_input = _read_input()
    session_id = _session_id(hook_input)
    if session_id:
        # PostCompact has a short Codex timeout. Do not load the embedding model or
        # run ingestion here; just mark that the next user prompt should recall.
        # The heavier ingest/search work runs in UserPromptSubmit, which has a
        # larger timeout.
        _state_dir().mkdir(parents=True, exist_ok=True)
        flag = {
            "session_id": session_id,
            "turn_id": _find_value(hook_input, "turn_id", "turnId"),
            "trigger": _find_value(hook_input, "trigger", "compact_trigger", "compactTrigger"),
            "transcript_path": _transcript_path(hook_input),
        }
        _flag_path(session_id).write_text(json.dumps(flag, ensure_ascii=False), encoding="utf-8")
    _write_json({"continue": True, "suppressOutput": True})


def user_prompt_submit() -> None:
    hook_input = _read_input()
    try:
        recall_context = asyncio.run(_ingest_and_maybe_recall(hook_input))
    except Exception as exc:
        log_path = _state_dir() / "errors.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"user_prompt_submit failed: {exc}\n")
        _write_json({"continue": True, "suppressOutput": True})
        return

    context_parts = [MEMORY_ROUTINE_CONTEXT]
    if recall_context:
        context_parts.append(recall_context)
    context = "\n\n".join(context_parts)

    _write_json(
        {
            "continue": True,
            "suppressOutput": True,
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            },
        }
    )


def stop() -> None:
    hook_input = _read_input()
    try:
        asyncio.run(_ingest_hook_transcript(hook_input))
    except Exception as exc:
        log_path = _state_dir() / "errors.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"stop ingest failed: {exc}\n")
    _write_json({"continue": True, "suppressOutput": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinite Memory Codex hooks")
    parser.add_argument("event", choices=["post-compact", "user-prompt-submit", "stop"])
    args = parser.parse_args()

    if args.event == "post-compact":
        post_compact()
    elif args.event == "user-prompt-submit":
        user_prompt_submit()
    elif args.event == "stop":
        stop()


if __name__ == "__main__":
    main()
