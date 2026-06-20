from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

from .config import ensure_default_config, load_config


def _state_dir() -> Path:
    return Path.home() / ".codex" / "infinite-memory" / "hook-state"


def _flag_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return _state_dir() / f"{safe}.compact.json"


def _keywords_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
    return _state_dir() / f"{safe}.keywords.json"


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


def _normalize_keyword(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _load_keywords(session_id: str) -> list[str]:
    path = _keywords_path(session_id)
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = _normalize_keyword(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            keywords.append(item.strip())
    return keywords


def _save_keywords(session_id: str, keywords: list[str]) -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    # Keep the trigger dictionary small. Oldest keywords fall out first.
    _keywords_path(session_id).write_text(
        json.dumps(keywords[-200:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_durable_keyword(text: str) -> str | None:
    """Mechanically extract one likely reusable keyword/search anchor."""
    stripped = text.strip()
    if not stripped:
        return None

    patterns = [
        # Paths and config files.
        r"(?:~|\.{1,2})?/[A-Za-z0-9._~@%+=:,/-]+",
        r"\b[A-Za-z0-9_.-]+\.(?:py|js|mjs|ts|tsx|json|toml|md|txt|sqlite3?|db)\b",
        # Scoped packages, model ids, MCP/tool-ish names.
        r"@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*(?:[A-Z][A-Za-z0-9_]*)+\b",
        r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b",
        r"\b[A-Za-z0-9_.-]*(?:hook|mcp|qwen|codex|sqlite|npm|batch|vector|embedding|compact)[A-Za-z0-9_.-]*\b",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, stripped, flags=re.IGNORECASE))

    # Korean technical-ish compounds near common memory/project terms.
    candidates.extend(
        re.findall(
            r"[가-힣A-Za-z0-9_.-]*(?:훅|검색|기억|압축|임베딩|모델|설정|키워드|벡터|세션)[가-힣A-Za-z0-9_.-]*",
            stripped,
        )
    )

    stopwords = {
        "그냥", "이제", "좋아", "아니", "근데", "그리고", "사용자", "요청",
        "검색", "기억", "설정", "hook", "mcp", "codex", "memory", "infinite",
    }
    best: str | None = None
    best_score = -1
    for raw in candidates:
        candidate = raw.strip("`'\".,:;()[]{}<>")
        normalized = _normalize_keyword(candidate)
        if len(normalized) < 4 or normalized in stopwords:
            continue
        score = len(candidate)
        if any(ch in candidate for ch in "/._-@"):
            score += 10
        if any(ch.isupper() for ch in candidate):
            score += 4
        if re.search(r"\d", candidate):
            score += 3
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _explicit_or_uncertain_request(text: str) -> bool:
    lowered = text.lower()
    explicit = (
        "기억" in text or "검색" in text or "찾아" in text or "전에" in text
        or "이전" in text or "아까" in text or "뭐였" in text or "뭐지" in text
        or "remember" in lowered or "recall" in lowered or "search" in lowered
        or "find" in lowered or "previous" in lowered or "before" in lowered
    )
    vague = (
        "그거" in text or "그걸" in text or "그건" in text or "이거" in text
        or "그대로" in text or "계속" in text or "아무튼" in text
        or "it" == lowered.strip() or "that" in lowered
    )
    return explicit or vague


def _matched_keywords(prompt: str, keywords: list[str]) -> list[str]:
    normalized_prompt = _normalize_keyword(prompt)
    matches: list[str] = []
    for keyword in keywords:
        normalized = _normalize_keyword(keyword)
        if len(normalized) >= 4 and normalized in normalized_prompt:
            matches.append(keyword)
    return matches[:5]


def _update_keyword_store(session_id: str, prompt: str) -> tuple[str | None, list[str]]:
    keywords = _load_keywords(session_id)
    matches = _matched_keywords(prompt, keywords)

    keyword = _extract_durable_keyword(prompt)
    if keyword:
        existing = {_normalize_keyword(item) for item in keywords}
        if _normalize_keyword(keyword) not in existing:
            keywords.append(keyword)
            _save_keywords(session_id, keywords)
        else:
            keyword = None

    return keyword, matches


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


async def _keyword_trigger_recall(hook_input: dict[str, Any], matched: list[str]) -> str:
    session_id = _session_id(hook_input)
    prompt = _prompt(hook_input)
    if not session_id or not prompt or not matched:
        return ""
    if not _explicit_or_uncertain_request(prompt):
        return ""

    ingester = await _ingest_hook_transcript(hook_input)
    query = " ".join([*matched, prompt]).strip()
    results = await ingester.search(query=query, session_id=session_id, limit=5)
    memories = _format_memory_results(results)
    if not memories:
        return ""
    return (
        "Infinite Memory keyword recall:\n"
        f"Matched remembered keywords: {', '.join(matched)}\n"
        "Use these retrieved memories only when they are relevant to the user's request.\n\n"
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
        session_id = _session_id(hook_input)
        prompt = _prompt(hook_input)
        matched: list[str] = []
        if session_id and prompt:
            _, matched = _update_keyword_store(session_id, prompt)
        recall_context = asyncio.run(_ingest_and_maybe_recall(hook_input))
        keyword_context = asyncio.run(_keyword_trigger_recall(hook_input, matched))
    except Exception as exc:
        log_path = _state_dir() / "errors.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"user_prompt_submit failed: {exc}\n")
        _write_json({"continue": True, "suppressOutput": True})
        return

    context_parts = []
    if recall_context:
        context_parts.append(recall_context)
    if keyword_context:
        context_parts.append(keyword_context)
    if not context_parts:
        _write_json({"continue": True, "suppressOutput": True})
        return
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
