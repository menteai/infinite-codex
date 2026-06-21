import json

from infinite_memory.codex_sessions import parse_session_file, parse_session_metadata


def test_parse_complete_turn_and_ignore_duplicates_and_commentary(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "sid", "cwd": "/x"}},
        {"type": "response_item", "timestamp": "t1", "payload": {"type": "message", "role": "user", "content": [{"text": "build it"}]}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "build it"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"text": "working..."}]}},
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "large log"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "phase": "final_answer", "message": "done"}},
        {"type": "response_item", "timestamp": "t2", "payload": {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"text": "done"}]}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    turns = parse_session_file(p)
    assert len(turns) == 1
    assert turns[0].role == "turn"
    assert turns[0].content == "User:\nbuild it\n\nAssistant:\ndone"
    assert turns[0].session_id == "sid"
    assert turns[0].cwd == "/x"


def test_ignore_incomplete_aborted_and_startup_context(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "sid", "cwd": "/x"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": "# AGENTS.md instructions for /x"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": "unfinished request"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"text": "working"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": "<turn_aborted> stopped"}]}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert parse_session_file(p) == []


def test_parse_fork_metadata(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "child",
                    "forked_from_id": "parent",
                    "cwd": "/x",
                },
            }
        )
    )

    metadata = parse_session_metadata(p)

    assert metadata.session_id == "child"
    assert metadata.forked_from_id == "parent"
    assert metadata.cwd == "/x"
