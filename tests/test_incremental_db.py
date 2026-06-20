from infinite_memory.codex_sessions import SessionMessage
from infinite_memory.db import MemoryDB


def _msg(path, session_id, ordinal, content):
    return SessionMessage(
        session_id=session_id,
        source_file=str(path),
        timestamp=f"t{ordinal}",
        role="turn",
        content=content,
        cwd="/x",
        turn_id=f"turn-{ordinal}",
    )


def test_incremental_upsert_appends_only_new_messages(tmp_path):
    db = MemoryDB(tmp_path / "memory.sqlite3")
    session_file = tmp_path / "s.jsonl"
    session_file.write_text("first")

    first = [_msg(session_file, "sid", 0, "one")]
    inserted, replaced = db.upsert_session_messages_incremental(session_file, first)
    assert (inserted, replaced) == (1, False)

    # Existing chunk must survive append-only ingest.
    row = db.conn.execute("SELECT id FROM messages WHERE session_id = 'sid'").fetchone()
    db.insert_chunk(
        session_id="sid",
        message_id=row["id"],
        chunk_index=0,
        embedding_model="test-model",
        content="one",
        search_text="one",
        embedding=[1.0, 0.0],
        metadata={},
    )
    db.conn.commit()

    session_file.write_text("first\nsecond")
    second = [_msg(session_file, "sid", i, text) for i, text in enumerate(["one", "two"])]
    inserted, replaced = db.upsert_session_messages_incremental(session_file, second)

    assert (inserted, replaced) == (1, False)
    assert db.conn.execute("SELECT COUNT(*) c FROM messages WHERE session_id = 'sid'").fetchone()["c"] == 2
    assert db.conn.execute("SELECT COUNT(*) c FROM chunks WHERE session_id = 'sid'").fetchone()["c"] == 1


def test_incremental_upsert_replaces_when_existing_prefix_changes(tmp_path):
    db = MemoryDB(tmp_path / "memory.sqlite3")
    session_file = tmp_path / "s.jsonl"
    session_file.write_text("first")

    db.upsert_session_messages_incremental(session_file, [_msg(session_file, "sid", 0, "one")])
    row = db.conn.execute("SELECT id FROM messages WHERE session_id = 'sid'").fetchone()
    db.insert_chunk(
        session_id="sid",
        message_id=row["id"],
        chunk_index=0,
        embedding_model="test-model",
        content="one",
        search_text="one",
        embedding=[1.0, 0.0],
        metadata={},
    )
    db.conn.commit()

    session_file.write_text("changed")
    inserted, replaced = db.upsert_session_messages_incremental(
        session_file,
        [_msg(session_file, "sid", 0, "changed")],
    )

    assert (inserted, replaced) == (1, True)
    assert db.conn.execute("SELECT content FROM messages WHERE session_id = 'sid'").fetchone()["content"] == "changed"
    assert db.conn.execute("SELECT COUNT(*) c FROM chunks WHERE session_id = 'sid'").fetchone()["c"] == 0
