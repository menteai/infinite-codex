# Infinite Memory

Infinite Memory is a Codex MCP server that adds vector search over Codex session history.

Infinite Memory stores Codex turns captured by its lifecycle hooks. It combines each user request with its final assistant answer into one turn, embeds those turns, stores them in SQLite, and exposes memory search tools to Codex. Commentary, tool calls, tool output, incomplete turns, and duplicated event records are excluded.

## Quickstart

With npm:

```bash
cd /home/flqbh/Downloads/infinite-memory
npm install -g @menteai/infinite-memory
imemory
```

With Python directly:

```bash
cd /home/flqbh/Downloads/infinite-memory
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
imemory
```

`imemory` prompts for the embedding backend and embedding batch size, then does the setup:

```text
Select embedding model:
1) Qwen/Qwen3-Embedding-0.6B (GPU) (DEFAULT)
2) Qwen/Qwen3-Embedding-0.6B (CPU)
3) Custom OpenAI-compatible embeddings endpoint
Embedding batch size [4]:
```

Then it:

- writes `~/.codex/infinite-memory/config.toml`
- creates `~/.codex/infinite-memory/memory.sqlite3`
- registers the MCP server in `~/.codex/config.toml`
- registers compact-recall hooks in `~/.codex/hooks.json`
- adds a model-level usage hint to `~/.codex/AGENTS.md`
- installs the selected local dependencies
- re-indexes the database if the embedding model changed

Restart Codex after setup. If Codex asks you to review hooks, open `/hooks` and trust the Infinite Memory hooks.


## Platform support

Infinite Memory is packaged for Linux, macOS, and Windows. The npm installer creates a local Python virtual environment using Python 3.11, 3.12, 3.13, or 3.14.

Device behavior:

- GPU option installs PyTorch and uses the best accelerator PyTorch exposes on the system.
- macOS Apple Silicon/Metal uses PyTorch MPS when available.
- Other systems use CUDA when available.
- If no supported accelerator is available, runtime falls back to CPU.

Codex CLI, IDE, and Codex app share the same `~/.codex/config.toml` and `~/.codex/hooks.json` setup. Run `imemory` once from a terminal, then restart Codex. In the Codex app, open `/mcp` to check the MCP server and `/hooks` to trust the lifecycle hooks if prompted.

## Embeddings

Default backend is selected during `imemory` setup.

Config file:

```text
~/.codex/infinite-memory/config.toml
```

Default config:

```toml
home = "~/.codex/infinite-memory"
db_path = "~/.codex/infinite-memory/memory.sqlite3"
codex_sessions_dir = "~/.codex/sessions"
chunk_chars = 800
chunk_overlap_chars = 100

[embedding]
backend = "local"
model = "Qwen/Qwen3-Embedding-0.6B"
api_key_env = "CUSTOM_EMBEDDINGS_API_KEY"
max_length = 1024
batch_size = 4
```

GPU Qwen3-Embedding is installed automatically when option 1 is selected. Runtime device is detected automatically (`cuda` or `mps` if available, otherwise `cpu`). Model files are stored under `~/.codex/infinite-memory/models/huggingface`.

CPU Qwen3-Embedding is installed when option 2 is selected. The installer uses the CPU-only PyTorch build.

Local embedding options:

- `max_length`: tokenizer limit per chunk, in tokens. Default: `1024`.
- `batch_size`: number of chunks embedded per model call. Default: `4`; increase on larger GPUs, decrease if VRAM runs out.

Custom OpenAI-compatible endpoint:

```toml
[embedding]
backend = "custom"
base_url = "http://localhost:8000/v1"
model = "text-embedding-3-small"
api_key_env = "CUSTOM_EMBEDDINGS_API_KEY"
```

## CLI

```bash
imemory              # setup config, database, and Codex MCP
imemory setup        # same as above
imemory ingest       # index Codex sessions
imemory ingest --force
imemory search "query text" --session-id SESSION_ID
imemory stats
```

## MCP tools

- `memory_ingest(force=false)`
- `infinite_memory_search(query, session_id, limit=8)`
- `memory_stats()`
- `memory_get(chunk_id, session_id)`
- `memory_forget_session(session_id)`

## Usage

Codex is instructed through `~/.codex/AGENTS.md` to decide for itself whether prior conversation context would help. When it judges memory is useful, it can call `infinite_memory.infinite_memory_search` with the current `session_id`. You can still force a manual lookup by saying something like “infinite memory search ...”. Results are formatted without metadata.

Automatic hook capture is installed. `Stop` indexes the current transcript after each completed turn. `PostCompact` marks the current session after Codex compacts context. The next `UserPromptSubmit` searches Infinite Memory once for that same session and injects the retrieved snippets as temporary context. After that one prompt, the flag is cleared. Existing historical Codex sessions are not automatically imported.

## License

Apache-2.0. See [LICENSE](LICENSE).
