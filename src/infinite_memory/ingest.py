from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .codex_sessions import iter_session_files, parse_session_file
from .config import MemoryConfig
from .db import MemoryDB
from .embeddings import EmbeddingClient, normalize


class Ingester:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db = MemoryDB(config.db_path)
        self.embedder = EmbeddingClient(config.embedding)

    async def ingest(self, force: bool = False, paths: list[Path] | None = None) -> dict:
        explicit_paths = paths is not None
        files = list(paths) if explicit_paths else list(iter_session_files(self.config.codex_sessions_dir))
        target_session_ids: set[str] | None = set() if explicit_paths else None
        if target_session_ids is not None:
            for path in files:
                session_id = self.db.session_id_for_path(path)
                if session_id:
                    target_session_ids.add(session_id)

        parsed = 0
        skipped = 0
        appended = 0
        replaced = 0
        for path in files:
            if not path.exists():
                skipped += 1
                continue
            if not force and self.db.session_current(path):
                skipped += 1
                continue
            messages = parse_session_file(path)
            if not messages:
                skipped += 1
                continue
            if target_session_ids is not None:
                target_session_ids.add(messages[0].session_id)
            if force:
                self.db.replace_session_messages(path, messages)
                parsed += 1
                replaced += 1
            else:
                inserted, did_replace = self.db.upsert_session_messages_incremental(path, messages)
                parsed += 1
                appended += inserted
                replaced += int(did_replace)

        reset_chunks = self.db.ensure_index_settings(
            self.embedder.model_key,
            chunk_chars=self.config.chunk_chars,
            chunk_overlap_chars=self.config.chunk_overlap_chars,
            max_length=self.embedder.config.max_length,
        )
        if reset_chunks and explicit_paths:
            target_session_ids = None

        messages = self.db.messages_without_chunks(self.embedder.model_key, target_session_ids)
        chunk_jobs: list[tuple] = []
        texts: list[str] = []
        for msg in messages:
            chunks = chunk_text(
                msg["content"],
                self.config.chunk_chars,
                self.config.chunk_overlap_chars,
            )
            for idx, content in enumerate(chunks):
                prefix = f"role: {msg['role']}\ncwd: {msg['cwd'] or ''}\ntime: {msg['timestamp'] or ''}\n\n"
                search_text = prefix + content
                texts.append(search_text)
                chunk_jobs.append((msg, idx, content, search_text))

        embedded = 0
        batch_size = max(1, int(self.embedder.config.batch_size))
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            vectors = await self.embedder.embed(batch_texts, input_type="document")
            for (msg, idx, content, search_text), vector in zip(chunk_jobs[i : i + batch_size], vectors):
                self.db.insert_chunk(
                    session_id=msg["session_id"],
                    message_id=msg["id"],
                    chunk_index=idx,
                    embedding_model=self.embedder.model_key,
                    content=content,
                    search_text=search_text,
                    embedding=normalize(vector),
                    metadata={
                        "role": msg["role"],
                        "timestamp": msg["timestamp"],
                        "cwd": msg["cwd"],
                        "source_file": msg["source_file"],
                        "turn_id": msg["turn_id"],
                    },
                )
                embedded += 1
            self.db.conn.commit()

        stats = self.db.stats()
        stats.update({
            "files_total": len(files),
            "files_parsed": parsed,
            "files_skipped": skipped,
            "messages_appended": appended,
            "sessions_replaced": replaced,
            "chunks_embedded": embedded,
            "chunks_reset_for_settings_change": reset_chunks,
        })
        return stats

    async def search(self, query: str, session_id: str, limit: int = 8) -> list[dict]:
        if not session_id:
            return []
        vector = normalize((await self.embedder.embed([query], input_type="query"))[0])
        return self.db.search_bruteforce(
            vector,
            self.embedder.model_key,
            limit,
            session_id=session_id,
        )
