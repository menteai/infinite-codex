from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PARSER_VERSION = 2


@dataclass(frozen=True)
class SessionMessage:
    session_id: str
    source_file: str
    timestamp: str | None
    role: str
    content: str
    cwd: str | None = None
    turn_id: str | None = None


def iter_session_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in {"input_text", "output_text"} and isinstance(
                    item.get("content"), str
                ):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _skip_user_message(text: str) -> bool:
    stripped = text.lstrip()
    return (
        not stripped
        or stripped.startswith("# AGENTS.md instructions")
        or stripped.startswith("<turn_aborted>")
        or stripped.startswith("<environment_context>")
    )


def _format_turn(user_text: str, assistant_text: str) -> str:
    return f"User:\n{user_text.strip()}\n\nAssistant:\n{assistant_text.strip()}"


def parse_session_file(path: Path) -> list[SessionMessage]:
    """Parse complete user/final-answer turns from one Codex rollout.

    Only response_item records are used because event_msg mirrors the same text.
    Commentary, tool calls, tool output, reasoning, and incomplete/aborted turns
    are deliberately excluded.
    """
    session_id = path.stem
    cwd: str | None = None
    turns: list[SessionMessage] = []
    pending_user: tuple[str, str | None, str | None] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = obj.get("payload") or {}
            typ = obj.get("type")
            timestamp = obj.get("timestamp") or payload.get("timestamp")

            if typ == "session_meta":
                session_id = payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                continue

            if typ == "turn_context":
                cwd = payload.get("cwd") or cwd
                continue

            if typ != "response_item" or payload.get("type") != "message":
                continue

            role = payload.get("role")
            text = _content_to_text(payload.get("content")).strip()
            if not text:
                continue

            if role == "user":
                if _skip_user_message(text):
                    continue
                # A new user prompt supersedes an incomplete/aborted pending turn.
                pending_user = (text, timestamp, payload.get("turn_id"))
                continue

            if role != "assistant":
                continue
            # phase may be absent for active/live sessions; only skip explicitly
            # non-final phases such as commentary.
            phase = payload.get("phase")
            if phase is not None and phase != "final_answer":
                continue
            if pending_user is None:
                continue

            user_text, user_timestamp, turn_id = pending_user
            turns.append(
                SessionMessage(
                    session_id=session_id,
                    source_file=str(path),
                    timestamp=user_timestamp or timestamp,
                    role="turn",
                    content=_format_turn(user_text, text),
                    cwd=cwd,
                    turn_id=turn_id or payload.get("turn_id"),
                )
            )
            pending_user = None

    return turns
