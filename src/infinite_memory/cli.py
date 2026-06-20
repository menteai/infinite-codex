from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .config import (
    DEFAULT_CONFIG_PATH,
    EmbeddingConfig,
    MemoryConfig,
    ensure_default_config,
    load_config,
    write_config,
)
from .db import MemoryDB

CODEX_CONFIG = Path.home() / ".codex" / "config.toml"
CODEX_HOOKS = Path.home() / ".codex" / "hooks.json"
MCP_SECTION_RE = re.compile(r"(?ms)^\[mcp_servers\.infinite_memory\]\n.*?(?=^\[|\Z)")
LOCAL_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _runtime_python_path() -> Path:
    return Path.home() / ".codex" / "infinite-memory" / "runtime_python"


def _python_command() -> str:
    override = os.environ.get("INFINITE_MEMORY_SERVER_PYTHON")
    if override:
        return override
    runtime_file = _runtime_python_path()
    if runtime_file.exists():
        value = runtime_file.read_text().strip()
        if value:
            return value
    return sys.executable


def _shell_quote(value: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _toml_string(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')


def _mcp_section() -> str:
    python = _toml_string(_python_command())
    return f'''[mcp_servers.infinite_memory]
command = "{python}"
args = ["-m", "infinite_memory.server"]
env_vars = ["CUSTOM_EMBEDDINGS_API_KEY", "INFINITE_MEMORY_BACKEND", "INFINITE_MEMORY_MODEL", "INFINITE_MEMORY_CONFIG", "INFINITE_MEMORY_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"]
'''


def register_codex_mcp(config_path: Path = CODEX_CONFIG) -> str:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    old = config_path.read_text() if config_path.exists() else ""
    section = _mcp_section()

    if MCP_SECTION_RE.search(old):
        new = MCP_SECTION_RE.sub(section.rstrip() + "\n", old)
        action = "updated"
    else:
        sep = "" if old.endswith("\n") or not old else "\n"
        new = old + sep + "\n" + section
        action = "added"

    if new != old:
        config_path.write_text(new)
    return action


def register_codex_hooks(hooks_path: Path = CODEX_HOOKS) -> str:
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    if hooks_path.exists():
        try:
            data = json.loads(hooks_path.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        data["hooks"] = hooks

    python = _shell_quote(_python_command())
    desired = {
        "PostCompact": {
            "matcher": "manual|auto",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{python} -m infinite_memory.hooks post-compact",
                    "statusMessage": "Marking Infinite Memory compact recall",
                    "timeout": 30,
                }
            ],
        },
        "UserPromptSubmit": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{python} -m infinite_memory.hooks user-prompt-submit",
                    "statusMessage": "Checking Infinite Memory compact recall",
                    "timeout": 600,
                }
            ],
        },
        "Stop": {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{python} -m infinite_memory.hooks stop",
                    "statusMessage": "Saving Infinite Memory turn",
                    "timeout": 120,
                }
            ],
        },
    }

    changed = False
    for event, group in desired.items():
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            groups = []
            hooks[event] = groups

        target_module = "infinite_memory.hooks"
        replacement_index = None
        for idx, existing_group in enumerate(groups):
            if not isinstance(existing_group, dict):
                continue
            for hook in existing_group.get("hooks", []):
                if isinstance(hook, dict) and target_module in str(hook.get("command", "")):
                    replacement_index = idx
                    break
            if replacement_index is not None:
                break

        if replacement_index is None:
            groups.append(group)
            changed = True
        elif groups[replacement_index] != group:
            groups[replacement_index] = group
            changed = True

    old_text = hooks_path.read_text() if hooks_path.exists() else ""
    new_text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if new_text != old_text:
        hooks_path.write_text(new_text)
        changed = True
    return "updated" if changed else "unchanged"


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(text: str, default: int, minimum: int = 1) -> int:
    raw = _prompt(text, str(default))
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid number. Using {default}.")
        return default
    if value < minimum:
        print(f"Value must be >= {minimum}. Using {default}.")
        return default
    return value


def _embedding_batch_size(current: EmbeddingConfig | None = None) -> int:
    default = current.batch_size if current and current.batch_size else 4
    return _prompt_int("Embedding batch size", default, minimum=1)


def _local_model_home() -> Path:
    return Path.home() / ".codex" / "infinite-memory" / "models" / "huggingface"


def _local_model_env() -> dict[str, str]:
    env = os.environ.copy()
    model_home = str(_local_model_home())
    env["HF_HOME"] = model_home
    env["TRANSFORMERS_CACHE"] = model_home
    env["SENTENCE_TRANSFORMERS_HOME"] = model_home
    return env


def _install_local_dependencies(device: str = "cpu") -> None:
    model_home = _local_model_home()
    model_home.mkdir(parents=True, exist_ok=True)
    env = _local_model_env()
    temp_home = Path.home() / ".codex" / "infinite-memory" / "tmp"
    temp_home.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temp_home)

    device_label = device.upper()
    print(f"Installing local embedding dependencies ({device_label})...")
    print(f"Model cache: {model_home}")

    torch_cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
                 "--upgrade", "--force-reinstall", "torch"]
    if device == "cpu":
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]

    subprocess.check_call(torch_cmd, env=env)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "sentence-transformers>=3.0",
    ], env=env)


def choose_embedding_config(current: EmbeddingConfig | None = None) -> EmbeddingConfig:
    print("Select embedding model:")
    print(f"1) {LOCAL_EMBEDDING_MODEL} (GPU) (DEFAULT)")
    print(f"2) {LOCAL_EMBEDDING_MODEL} (CPU)")
    print("3) Custom OpenAI-compatible embeddings endpoint")

    choice = _prompt("Enter choice", "1")

    if choice == "1":
        _install_local_dependencies(device="gpu")
        batch_size = _embedding_batch_size(current)
        return EmbeddingConfig(
            backend="local",
            model=LOCAL_EMBEDDING_MODEL,
            api_key_env="CUSTOM_EMBEDDINGS_API_KEY",
            batch_size=batch_size,
        )

    if choice == "2":
        _install_local_dependencies(device="cpu")
        batch_size = _embedding_batch_size(current)
        return EmbeddingConfig(
            backend="local",
            model=LOCAL_EMBEDDING_MODEL,
            api_key_env="CUSTOM_EMBEDDINGS_API_KEY",
            batch_size=batch_size,
        )

    if choice == "3":
        print("Enter custom OpenAI-compatible embeddings settings.")
        base_url = _prompt("Base URL", current.base_url if current and current.base_url else "http://localhost:8000/v1")
        model = _prompt("Model name", current.model if current and current.backend == "custom" else "text-embedding-3-small")
        key = _prompt("API key (leave blank if not required)")
        batch_size = _embedding_batch_size(current)
        return EmbeddingConfig(
            backend="custom",
            model=model,
            api_key_env="CUSTOM_EMBEDDINGS_API_KEY",
            api_key=key or None,
            base_url=base_url,
            batch_size=batch_size,
        )

    print("Invalid choice. Using GPU Qwen3-Embedding.")
    _install_local_dependencies(device="gpu")
    batch_size = _embedding_batch_size(current)
    return EmbeddingConfig(
        backend="local",
        model=LOCAL_EMBEDDING_MODEL,
        api_key_env="CUSTOM_EMBEDDINGS_API_KEY",
        batch_size=batch_size,
    )


async def _ingest(force: bool) -> None:
    ensure_default_config()
    from .ingest import Ingester

    result = await Ingester(load_config()).ingest(force=force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


async def _search(query: str, session_id: str, limit: int) -> None:
    ensure_default_config()
    from .ingest import Ingester

    results = await Ingester(load_config()).search(
        query=query,
        session_id=session_id,
        limit=limit,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def setup(
    register: bool = True,
    write_agent_hint: bool = True,
    interactive: bool = True,
    register_hooks: bool = True,
) -> None:
    if DEFAULT_CONFIG_PATH.exists():
        cfg = load_config(DEFAULT_CONFIG_PATH)
    else:
        cfg = MemoryConfig()

    embedding = choose_embedding_config(cfg.embedding) if interactive else cfg.embedding
    cfg = replace(cfg, embedding=embedding)
    config_path = write_config(cfg, DEFAULT_CONFIG_PATH)

    cfg.home.mkdir(parents=True, exist_ok=True)
    db = MemoryDB(cfg.db_path)
    stats = db.stats()

    from .embeddings import EmbeddingClient
    new_model_key = EmbeddingClient(embedding).model_key
    existing_models = stats.get("models", [])
    if existing_models and new_model_key not in existing_models:
        print(f"Embedding model changed to {new_model_key}. Clearing old embeddings and re-embedding stored memories...")
        db.conn.execute("DELETE FROM chunks")
        db.conn.commit()
        from .ingest import Ingester
        asyncio.run(Ingester(cfg).reembed_existing_messages())
        print("Re-embedding complete.")

    mcp_action = register_codex_mcp() if register else "skipped"
    hooks_action = register_codex_hooks() if register_hooks else "skipped"

    if write_agent_hint:
        agents = Path.home() / ".codex" / "AGENTS.md"
        agents.parent.mkdir(parents=True, exist_ok=True)
        hint = (
            "\n"
            "- At the start of every turn, silently inspect the user's request for one durable keyword or short search query that is likely to be useful much later. Prefer project names, file paths, commands, model/package names, settings, errors, APIs, and prior-decision topics. If it duplicates an already-obvious/current keyword, keep zero new keywords.\n"
            "- Use the `infinite_memory.infinite_memory_search` MCP tool when prior conversation context, earlier decisions, previous setup, post-compaction continuity, a new task, or a conversation topic change may matter. Use it only when retrieved context is likely to improve the answer.\n"
            "- If the user asks to remember, recall, find, search, look up, retrieve, or check previous memory/conversation/context, treat that as an explicit request to use `infinite_memory.infinite_memory_search` before answering.\n"
            "- If the user's command is not fully understandable, depends on missing prior decisions, or has any uncertain reference, call `infinite_memory.infinite_memory_search` before answering.\n"
            "- Do not blindly search the user's exact wording. First infer the user's intent, the relevant entities, decisions, constraints, and time/context clues, then write a concise semantic search query yourself and pass that query to `infinite_memory.infinite_memory_search`.\n"
            "- Do not expose the keyword/search routine unless the user asks.\n"
        )
        old = agents.read_text() if agents.exists() else ""
        if "At the start of every turn, silently inspect the user's request" not in old:
            filtered = "\n".join(
                line for line in old.splitlines()
                if "infinite_memory.infinite_memory_search" not in line
                and "Do not blindly search the user's exact wording" not in line
                and "If the user asks to remember, recall" not in line
                and "At the start of every turn, silently inspect the user's request" not in line
                and "Do not expose the keyword/search routine" not in line
            ).strip()
            agents.write_text((filtered + "\n" if filtered else "") + hint)

    print("Infinite Memory setup complete")
    print(f"config: {config_path}")
    print(f"database: {cfg.db_path}")
    print(f"embedding: {cfg.embedding.backend}:{cfg.embedding.model}")
    print(f"codex mcp: {mcp_action}")
    print(f"codex hooks: {hooks_action}")
    print("Restart Codex to load MCP changes.")


def ingest_main() -> None:
    parser = argparse.ArgumentParser(description="Index Codex sessions into Infinite Memory")
    parser.add_argument("--force", action="store_true", help="re-index all sessions")
    args = parser.parse_args()
    asyncio.run(_ingest(args.force))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="infinite",
        description="Set up and use Infinite Memory for Codex",
    )
    sub = parser.add_subparsers(dest="command")

    setup_p = sub.add_parser("setup", help="create config/db and register Codex MCP")
    setup_p.add_argument("--no-register", action="store_true", help="do not edit ~/.codex/config.toml")
    setup_p.add_argument("--no-agent-hint", action="store_true", help="do not append a usage hint to ~/.codex/AGENTS.md")
    setup_p.add_argument("--no-hooks", action="store_true", help="do not edit ~/.codex/hooks.json")
    setup_p.add_argument("--no-interactive", action="store_true", help="use existing config without prompts")

    ingest_p = sub.add_parser("ingest", help="index Codex sessions")
    ingest_p.add_argument("--force", action="store_true", help="re-index all sessions")

    search_p = sub.add_parser("search", help="search indexed memory")
    search_p.add_argument("query")
    search_p.add_argument("--session-id", required=True)
    search_p.add_argument("--limit", type=int, default=8)

    sub.add_parser("stats", help="show database stats")

    args = parser.parse_args()

    if args.command in {None, "setup"}:
        setup(
            register=not getattr(args, "no_register", False),
            write_agent_hint=not getattr(args, "no_agent_hint", False),
            interactive=not getattr(args, "no_interactive", False),
            register_hooks=not getattr(args, "no_hooks", False),
        )
    elif args.command == "ingest":
        asyncio.run(_ingest(args.force))
    elif args.command == "search":
        asyncio.run(_search(args.query, args.session_id, args.limit))
    elif args.command == "stats":
        ensure_default_config()
        cfg = load_config()
        stats = MemoryDB(cfg.db_path).stats()
        stats["embedding_backend"] = cfg.embedding.backend
        stats["embedding_model"] = cfg.embedding.model
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
