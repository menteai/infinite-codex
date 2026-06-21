from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .codex_sessions import PARSER_VERSION, SessionMessage

SCHEMA_VERSION = 2


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        aa += x * x
        bb += y * y
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return dot / ((aa ** 0.5) * (bb ** 0.5))


class MemoryDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.vec_enabled = self._load_vec()
        self.init_schema()

    def _load_vec(self) -> bool:
        try:
            import sqlite_vec

            sqlite_vec.load(self.conn)
            return True
        except Exception:
            return False

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_parents (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                timestamp TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                cwd TEXT,
                turn_id TEXT,
                ordinal INTEGER NOT NULL,
                UNIQUE(session_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                content TEXT NOT NULL,
                search_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(message_id, chunk_index, embedding_model)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embedding_model);
            CREATE INDEX IF NOT EXISTS idx_session_parents_parent
                ON session_parents(parent_session_id);
            """
        )
        parser_row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'parser_version'"
        ).fetchone()
        previous_parser_version = int(parser_row["value"]) if parser_row else 0
        if previous_parser_version != PARSER_VERSION:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM messages")
            self.conn.execute("DELETE FROM sessions")

        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('parser_version', ?)",
            (str(PARSER_VERSION),),
        )
        self.conn.commit()

    def session_current(self, path: Path) -> bool:
        stat = path.stat()
        row = self.conn.execute(
            "SELECT mtime, size FROM sessions WHERE source_file = ?", (str(path),)
        ).fetchone()
        return bool(row and float(row["mtime"]) == stat.st_mtime and int(row["size"]) == stat.st_size)

    def session_id_for_path(self, path: Path) -> str | None:
        row = self.conn.execute(
            "SELECT session_id FROM sessions WHERE source_file = ?", (str(path),)
        ).fetchone()
        return str(row["session_id"]) if row else None

    def _update_session_record(self, path: Path, session_id: str) -> None:
        stat = path.stat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sessions(session_id, source_file, mtime, size, indexed_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (session_id, str(path), stat.st_mtime, stat.st_size),
        )

    def upsert_session_parent(self, session_id: str, parent_session_id: str | None) -> None:
        session_id = (session_id or "").strip()
        parent_session_id = (parent_session_id or "").strip()
        if not session_id or not parent_session_id or session_id == parent_session_id:
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO session_parents(session_id, parent_session_id, created_at)
            VALUES (?, ?, datetime('now'))
            """,
            (session_id, parent_session_id),
        )
        self.conn.commit()

    def session_ancestor_ids(self, session_id: str, max_depth: int = 32) -> list[str]:
        current = (session_id or "").strip()
        if not current:
            return []

        ancestors: list[str] = []
        seen = {current}
        for _ in range(max(0, max_depth)):
            row = self.conn.execute(
                "SELECT parent_session_id FROM session_parents WHERE session_id = ?",
                (current,),
            ).fetchone()
            if not row:
                break
            parent = str(row["parent_session_id"] or "").strip()
            if not parent or parent in seen:
                break
            ancestors.append(parent)
            seen.add(parent)
            current = parent
        return ancestors

    def session_search_scope(self, session_id: str, max_depth: int = 32) -> list[str]:
        session_id = (session_id or "").strip()
        if not session_id:
            return []
        return [session_id, *self.session_ancestor_ids(session_id, max_depth=max_depth)]

    def _insert_message(self, msg: SessionMessage, ordinal: int) -> None:
        self.conn.execute(
            """
            INSERT INTO messages(session_id, source_file, timestamp, role, content, cwd, turn_id, ordinal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.session_id,
                msg.source_file,
                msg.timestamp,
                msg.role,
                msg.content,
                msg.cwd,
                msg.turn_id,
                ordinal,
            ),
        )

    def _existing_messages_match_prefix(self, existing: list[sqlite3.Row], messages: list[SessionMessage]) -> bool:
        if len(messages) < len(existing):
            return False
        for row, msg in zip(existing, messages):
            if row["role"] != msg.role or row["content"] != msg.content:
                return False
            if (row["turn_id"] or None) != (msg.turn_id or None):
                return False
        return True

    def upsert_session_messages_incremental(self, path: Path, messages: list[SessionMessage]) -> tuple[int, bool]:
        """Insert only appended turns when a session log grew monotonically.

        Returns ``(inserted_messages, replaced)``. If any already-indexed turn no
        longer matches the parsed prefix, the session is replaced to keep the DB
        correct.
        """
        if not messages:
            return (0, False)

        session_id = messages[0].session_id
        existing = self.conn.execute(
            """
            SELECT id, role, content, turn_id, ordinal
            FROM messages
            WHERE session_id = ?
            ORDER BY ordinal
            """,
            (session_id,),
        ).fetchall()

        if existing and not self._existing_messages_match_prefix(existing, messages):
            self.replace_session_messages(path, messages)
            return (len(messages), True)

        inserted = 0
        for ordinal, msg in enumerate(messages[len(existing):], start=len(existing)):
            self._insert_message(msg, ordinal)
            inserted += 1

        self._update_session_record(path, session_id)
        self.conn.commit()
        return (inserted, False)

    def delete_session(self, session_id: str) -> int:
        chunk_count = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE session_id = ?", (session_id,),
        ).fetchone()["c"]
        self.conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM session_parents WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return int(chunk_count)

    def replace_session_messages(self, path: Path, messages: list[SessionMessage]) -> None:
        if not messages:
            return
        session_id = messages[0].session_id
        self.delete_session(session_id)
        for ordinal, msg in enumerate(messages):
            self._insert_message(msg, ordinal)
        self._update_session_record(path, session_id)
        self.conn.commit()

    def ensure_index_settings(
        self,
        embedding_model: str,
        *,
        chunk_chars: int,
        chunk_overlap_chars: int,
        max_length: int,
    ) -> bool:
        key = f"index_settings:{embedding_model}"
        value = json.dumps(
            {
                "chunk_chars": chunk_chars,
                "chunk_overlap_chars": chunk_overlap_chars,
                "max_length": max_length,
            },
            sort_keys=True,
        )
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row and row["value"] == value:
            return False
        self.conn.execute("DELETE FROM chunks WHERE embedding_model = ?", (embedding_model,))
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
        self.conn.commit()
        return True

    def messages_without_chunks(
        self,
        embedding_model: str,
        session_ids: set[str] | None = None,
    ) -> list[sqlite3.Row]:
        params: list[Any] = [embedding_model]
        session_filter = ""
        if session_ids is not None:
            if not session_ids:
                return []
            placeholders = ",".join("?" for _ in session_ids)
            session_filter = f"AND m.session_id IN ({placeholders})"
            params.extend(sorted(session_ids))

        return self.conn.execute(
            f"""
            SELECT m.* FROM messages m
            WHERE NOT EXISTS (
              SELECT 1 FROM chunks c WHERE c.message_id = m.id AND c.embedding_model = ?
            )
            {session_filter}
            ORDER BY m.id
            """,
            params,
        ).fetchall()

    def insert_chunk(
        self,
        *,
        session_id: str,
        message_id: int,
        chunk_index: int,
        embedding_model: str,
        content: str,
        search_text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO chunks(
                session_id, message_id, chunk_index, embedding_model, content, search_text,
                embedding_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                message_id,
                chunk_index,
                embedding_model,
                content,
                search_text,
                json.dumps(embedding),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def search_bruteforce(
        self,
        query_embedding: list[float],
        embedding_model: str,
        limit: int,
        session_id: str,
        include_ancestors: bool = True,
    ) -> list[dict[str, Any]]:
        if not session_id:
            return []
        session_ids = (
            self.session_search_scope(session_id)
            if include_ancestors
            else [session_id]
        )
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        rows = self.conn.execute(
            f"""
            SELECT * FROM chunks
            WHERE embedding_model = ? AND session_id IN ({placeholders})
            """,
            (embedding_model, *session_ids),
        ).fetchall()
        scored = []
        for row in rows:
            emb = json.loads(row["embedding_json"])
            score = _cosine(query_embedding, emb)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_result(row, score) for score, row in scored[:limit]]

    def get_chunk(self, chunk_id: int, session_id: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        session_ids = self.session_search_scope(session_id)
        if not session_ids:
            return None
        placeholders = ",".join("?" for _ in session_ids)
        row = self.conn.execute(
            f"SELECT * FROM chunks WHERE id = ? AND session_id IN ({placeholders})",
            (chunk_id, *session_ids),
        ).fetchone()
        if not row:
            return None
        return self._row_result(row, None)

    def session_turn_count(self, session_id: str) -> int:
        if not session_id:
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["c"])

    def stats(self) -> dict[str, Any]:
        return {
            "db_path": str(self.path),
            "sqlite_vec_loaded": self.vec_enabled,
            "sessions": self.conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"],
            "session_parent_links": self.conn.execute("SELECT COUNT(*) c FROM session_parents").fetchone()["c"],
            "messages": self.conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"],
            "chunks": self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"],
            "models": [r["embedding_model"] for r in self.conn.execute("SELECT DISTINCT embedding_model FROM chunks ORDER BY 1")],
        }

    def _row_result(self, row: sqlite3.Row, score: float | None) -> dict[str, Any]:
        metadata = json.loads(row["metadata_json"])
        return {
            "chunk_id": row["id"],
            "score": score,
            "session_id": row["session_id"],
            "content": row["content"],
            "metadata": metadata,
        }
