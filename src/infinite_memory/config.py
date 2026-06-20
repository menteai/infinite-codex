from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOME = Path(os.environ.get("INFINITE_MEMORY_HOME", Path.home() / ".codex" / "infinite-memory"))
DEFAULT_CONFIG_PATH = Path(os.environ.get("INFINITE_MEMORY_CONFIG", DEFAULT_HOME / "config.toml"))


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "local"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    api_key_env: str = "CUSTOM_EMBEDDINGS_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    max_length: int = 1024
    batch_size: int = 4


@dataclass(frozen=True)
class MemoryConfig:
    home: Path = DEFAULT_HOME
    db_path: Path = DEFAULT_HOME / "memory.sqlite3"
    codex_sessions_dir: Path = Path.home() / ".codex" / "sessions"
    chunk_chars: int = 800
    chunk_overlap_chars: int = 100
    embedding: EmbeddingConfig = EmbeddingConfig()


def default_model_for_backend(backend: str) -> str:
    return {
        "voyage": "voyage-4-large",
        "local": "Qwen/Qwen3-Embedding-0.6B",
        "custom": "text-embedding-3-small",
    }.get(backend, "Qwen/Qwen3-Embedding-0.6B")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> MemoryConfig:
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text())

    home = Path(os.environ.get("INFINITE_MEMORY_HOME", data.get("home", str(DEFAULT_HOME)))).expanduser()
    db_path = Path(data.get("db_path", str(home / "memory.sqlite3"))).expanduser()
    sessions_dir = Path(data.get("codex_sessions_dir", str(Path.home() / ".codex" / "sessions"))).expanduser()

    emb_data = data.get("embedding", {})
    backend = os.environ.get("INFINITE_MEMORY_BACKEND", emb_data.get("backend", "local"))
    default_model = default_model_for_backend(backend)

    embedding = EmbeddingConfig(
        backend=backend,
        model=os.environ.get("INFINITE_MEMORY_MODEL", emb_data.get("model", default_model)),
        api_key_env=emb_data.get(
            "api_key_env",
            "VOYAGE_API_KEY" if backend == "voyage" else "CUSTOM_EMBEDDINGS_API_KEY",
        ),
        api_key=emb_data.get("api_key"),
        base_url=emb_data.get("base_url"),
        dimensions=emb_data.get("dimensions"),
        max_length=int(emb_data.get("max_length", 1024)),
        batch_size=int(emb_data.get("batch_size", 4)),
    )

    return MemoryConfig(
        home=home,
        db_path=db_path,
        codex_sessions_dir=sessions_dir,
        chunk_chars=int(data.get("chunk_chars", 800)),
        chunk_overlap_chars=int(data.get("chunk_overlap_chars", 100)),
        embedding=embedding,
    )


def write_config(config: MemoryConfig, path: Path = DEFAULT_CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = config.embedding
    lines = [
        f'home = "{config.home}"',
        f'db_path = "{config.db_path}"',
        f'codex_sessions_dir = "{config.codex_sessions_dir}"',
        f"chunk_chars = {config.chunk_chars}",
        f"chunk_overlap_chars = {config.chunk_overlap_chars}",
        "",
        "[embedding]",
        f'backend = "{emb.backend}"',
        f'model = "{emb.model}"',
        f'api_key_env = "{emb.api_key_env}"',
    ]
    if emb.api_key:
        lines.append(f'api_key = "{emb.api_key}"')
    if emb.base_url:
        lines.append(f'base_url = "{emb.base_url}"')
    if emb.dimensions:
        lines.append(f"dimensions = {emb.dimensions}")
    lines.append(f"max_length = {emb.max_length}")
    lines.append(f"batch_size = {emb.batch_size}")
    path.write_text("\n".join(lines) + "\n")
    return path


def ensure_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    if not path.exists():
        write_config(MemoryConfig(), path)
    return path
