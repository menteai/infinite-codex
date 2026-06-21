import asyncio
from types import SimpleNamespace

from infinite_memory.codex_sessions import SessionMessage
from infinite_memory.config import EmbeddingConfig, MemoryConfig
from infinite_memory.db import MemoryDB
from infinite_memory.ingest import Ingester


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


def test_reembed_existing_messages_does_not_import_session_logs(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    session_file = tmp_path / "captured.jsonl"
    session_file.write_text("captured")
    db = MemoryDB(db_path)
    db.upsert_session_messages_incremental(session_file, [_msg(session_file, "captured", 0, "stored")])

    historical_dir = tmp_path / "sessions"
    historical_dir.mkdir()
    historical = historical_dir / "historical.jsonl"
    historical.write_text(
        '{"type":"session_meta","payload":{"id":"historical","cwd":"/x"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"text":"old"}]}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"text":"done"}]}}\n'
    )

    cfg = MemoryConfig(
        home=tmp_path,
        db_path=db_path,
        codex_sessions_dir=historical_dir,
        embedding=EmbeddingConfig(model="fake"),
    )
    ingester = Ingester(cfg)

    class FakeEmbedder:
        model_key = "local:fake"
        config = SimpleNamespace(batch_size=4, max_length=1024)

        async def embed(self, texts, input_type="document"):
            return [[1.0, 0.0] for _ in texts]

    ingester.embedder = FakeEmbedder()
    stats = asyncio.run(ingester.reembed_existing_messages())

    assert stats["files_total"] == 0
    assert db.conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 1
    assert db.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 1


def test_fork_child_searches_parent_chain(tmp_path):
    db = MemoryDB(tmp_path / "memory.sqlite3")
    model = "test-model"

    for session_id, content, vector in [
        ("A", "root memory", [1.0, 0.0]),
        ("B", "child memory", [0.9, 0.1]),
        ("C", "grandchild memory", [0.8, 0.2]),
    ]:
        session_file = tmp_path / f"{session_id}.jsonl"
        session_file.write_text(content)
        db.upsert_session_messages_incremental(
            session_file,
            [_msg(session_file, session_id, 0, content)],
        )
        row = db.conn.execute(
            "SELECT id FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        db.insert_chunk(
            session_id=session_id,
            message_id=row["id"],
            chunk_index=0,
            embedding_model=model,
            content=content,
            search_text=content,
            embedding=vector,
            metadata={},
        )

    db.conn.commit()
    db.upsert_session_parent("B", "A")
    db.upsert_session_parent("C", "B")

    results = db.search_bruteforce([1.0, 0.0], model, limit=10, session_id="C")
    assert {result["session_id"] for result in results} == {"A", "B", "C"}

    child_results = db.search_bruteforce([1.0, 0.0], model, limit=10, session_id="B")
    assert {result["session_id"] for result in child_results} == {"A", "B"}

    ancestor_chunk_id = next(result["chunk_id"] for result in results if result["session_id"] == "A")
    assert db.get_chunk(ancestor_chunk_id, session_id="C")["content"] == "root memory"
    assert db.get_chunk(ancestor_chunk_id, session_id="B")["content"] == "root memory"
