# 🔥 INFINITE MEMORY FOR CODEX — YOUR CODEX NOW REMEMBERS EVERYTHING 🔥

> Give Codex a permanent, unlimited memory.  
> No compaction loss. Preserve memories from 500 turns ago. 🧠⚡

**Infinite Memory** is a vector-memory MCP + hook system for Codex. It captures completed Codex turns, embeds them locally or through a custom embeddings endpoint, stores them in SQLite, and exposes semantic recall back to Codex through MCP.

No more “what did we decide again?”  
No more losing the plot after compaction.  
No more manually pasting old context like it is 1999. 🚀

---

## ✨ What It Does

- 🧠 **Permanent Codex memory** using vector embeddings
- 🔎 **Semantic search** over remembered turns
- 🪝 **Lifecycle hooks** that save completed turns automatically
- 🧩 **MCP tool integration** for model-controlled recall
- 💥 **Post-compaction recall** so Codex can recover context after compression
- 🗃️ **SQLite storage** — simple, local, portable
- ⚙️ **Qwen3-Embedding-0.6B by default**
- 🖥️ **Linux / macOS / Windows support**
- 🧑‍💻 **Codex CLI, IDE, and Codex app compatible**

Infinite Memory stores Codex turns captured by lifecycle hooks. It combines each user request with the final assistant answer into one turn, embeds that turn, and stores it for later recall. Commentary, tool calls, tool output, incomplete turns, and duplicated event records are excluded.

---

## 🚀 Quickstart

### Install from npm

```bash
npm install -g @menteai/infinite-memory
imemory
```

### Or install from source

```bash
git clone git@github.com:menteai/codex-infinite-memory.git
cd codex-infinite-memory
npm install -g .
imemory
```

### Python direct mode

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
imemory
```

`imemory` asks for the embedding backend and batch size:

```text
Select embedding model:
1) Qwen/Qwen3-Embedding-0.6B (GPU) (DEFAULT)
2) Qwen/Qwen3-Embedding-0.6B (CPU)
3) Custom OpenAI-compatible embeddings endpoint
Embedding batch size [4]:
```

Then it automatically:

- 📝 writes `~/.codex/infinite-memory/config.toml`
- 🗄️ creates `~/.codex/infinite-memory/memory.sqlite3`
- 🧩 registers the MCP server in `~/.codex/config.toml`
- 🪝 registers lifecycle hooks in `~/.codex/hooks.json`
- 🧠 adds a model-level usage hint to `~/.codex/AGENTS.md`
- 📦 installs selected local dependencies

Restart Codex after setup. If Codex asks you to review hooks, open `/hooks` and trust the Infinite Memory hooks.

---

## 🧨 How The Magic Works

```text
Codex turn finishes
  ↓
Stop hook captures the completed turn
  ↓
Turn gets chunked + embedded
  ↓
Vector goes into SQLite
  ↓
Codex later calls infinite_memory_search
  ↓
Relevant memories come back as clean context
```

After context compaction:

```text
PostCompact hook marks the session
  ↓
Next UserPromptSubmit hook fires
  ↓
Infinite Memory searches the current session once
  ↓
Retrieved snippets are injected as temporary context
  ↓
Flag is cleared
```

Existing historical Codex sessions are **not automatically imported**. Infinite Memory remembers turns captured after setup through hooks.

---

## 🖥️ Platform Support

Infinite Memory is packaged for:

- 🐧 Linux
- 🍎 macOS
- 🪟 Windows

The npm installer creates a local Python virtual environment using Python `3.11`, `3.12`, `3.13`, or `3.14`.

Device behavior:

- ⚡ GPU option installs PyTorch and uses the best accelerator PyTorch exposes
- 🍎 macOS Apple Silicon/Metal uses PyTorch MPS when available
- 🟩 CUDA is used when available
- 🧊 If no supported accelerator is available, runtime falls back to CPU

Codex CLI, IDE, and Codex app share the same setup:

```text
~/.codex/config.toml
~/.codex/hooks.json
```

Run `imemory` once from a terminal, restart Codex, then use:

```text
/mcp    # check the MCP server
/hooks  # review/trust hooks
```

---

## 🧠 Embeddings

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

### Local Qwen mode

- 🔥 Model: `Qwen/Qwen3-Embedding-0.6B`
- 📏 `max_length`: tokenizer limit per chunk, in tokens. Default: `1024`
- 📦 `batch_size`: chunks embedded per model call. Default: `4`
- 🧊 Increase batch size on larger GPUs; decrease it if VRAM runs out
- 📁 Model cache: `~/.codex/infinite-memory/models/huggingface`

### Custom OpenAI-compatible endpoint

```toml
[embedding]
backend = "custom"
base_url = "http://localhost:8000/v1"
model = "text-embedding-3-small"
api_key_env = "CUSTOM_EMBEDDINGS_API_KEY"
```

---

## ⚔️ CLI Commands

```bash
imemory              # setup config, database, hooks, and Codex MCP
imemory setup        # same as above
imemory ingest       # manually index Codex sessions
imemory ingest --force
imemory search "query text" --session-id SESSION_ID
imemory stats
```

Manual `imemory ingest` exists for explicit imports. The normal memory path is hook-based capture after setup.

---

## 🧩 MCP Tools

- `memory_ingest(force=false)`
- `infinite_memory_search(query, session_id, limit=8)`
- `memory_stats()`
- `memory_get(chunk_id, session_id)`
- `memory_forget_session(session_id)`

The main tool is the one that matters:

```text
infinite_memory_search
```

When prior decisions, setup details, new task context, topic changes, or post-compaction continuity matter, Codex can call it and pull the right memory back into the conversation.

---

## 🕹️ Usage

You can nudge Codex directly:

```text
infinite memory search what did we decide about batch size?
```

Or let Codex decide through the installed `AGENTS.md` hint:

```text
When prior conversation context may matter, use infinite_memory_search.
```

Search results are formatted without metadata, so the model gets the memory without noisy IDs, timestamps, or scores.

---

## 📦 Package

npm:

```bash
npm install -g @menteai/infinite-memory
```

GitHub:

```text
git@github.com:menteai/codex-infinite-memory.git
```

---

## 📜 License

Apache-2.0. See [LICENSE](LICENSE).
