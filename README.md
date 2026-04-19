# 📖 信望愛 AI 聖經助手 — Bible RAG with Tool Calling

An agentic Bible study assistant that answers questions using real-time data from the [信望愛 (FHL) Bible API](https://bible.fhl.net/). The AI doesn't guess — it calls 13 specialized Bible tools to fetch verses, parse Greek/Hebrew originals, look up Strong's numbers, and pull commentary, then synthesizes a cited answer.

**Two backends:**
- **Claude (Anthropic API)** — recommended, production-ready, large context window
- **Gemma 4 E4B (local GPU)** — for offline use or tool-calling verification

---

## Architecture

```
User question
    │
    ▼
┌─────────────────────────────┐
│  Claude / Gemma 4           │  LLM decides which tools to call
│  (agentic orchestration)    │  and chains multiple rounds
└──────────┬──────────────────┘
           │ tool calls (up to 10 rounds)
           ▼
┌─────────────────────────────┐
│  fhl_tools.py               │  13 Bible tools
│  ├─ get_bible_verse          │  Fetch verse text (和合本, KJV, etc.)
│  ├─ get_bible_chapter        │  Fetch full chapter
│  ├─ get_word_analysis        │  Greek/Hebrew word-by-word parsing
│  ├─ lookup_strongs           │  Strong's dictionary lookup
│  ├─ search_strongs_occurrences│  Find all uses of a Greek/Hebrew word
│  ├─ search_bible_advanced    │  Keyword search across books
│  ├─ get_commentary           │  Verse commentary
│  ├─ search_commentary        │  Search commentary by keyword
│  ├─ get_topic_study          │  Topic-based study
│  ├─ list_bible_versions      │  Available Bible versions
│  ├─ list_commentaries        │  Available commentary sets
│  ├─ get_book_list            │  Book names for a version
│  └─ query_verse_citation     │  Parse references like "約3:16"
└──────────┬──────────────────┘
           │ HTTP GET
           ▼
    bible.fhl.net/json/*.php
    (信望愛 public REST API)
```

The LLM autonomously decides the tool chain based on the question type:
- **Verse lookup** → `get_bible_verse` → `get_word_analysis` → `get_commentary`
- **Word study** → `get_word_analysis` → `lookup_strongs` → `search_strongs_occurrences`
- **Theological question** → `search_bible_advanced` → `get_bible_verse` → `get_commentary`

---

## Quick Start (Claude backend — recommended)

No GPU required. Runs entirely via the Anthropic API.

### 1. Clone the repo

```bash
git clone https://github.com/Peter-Tan/Gemma4-E4b-zh-tw.git
cd Gemma4
```

### 2. Install dependencies

**Option A — using [uv](https://docs.astral.sh/uv/) (recommended)**

uv is a fast Python package manager. If you don't have it:

```bash
# Install uv (one-time)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all dependencies:

```bash
uv sync
```

This creates a `.venv`, installs all packages from `pyproject.toml` + `uv.lock` (including PyTorch for local Gemma if you need it later), and ensures reproducible versions.

```bash
# Activate the venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install additional Claude dependencies (not yet in pyproject.toml)
uv pip install anthropic python-dotenv gradio
```

**Option B — using pip (Claude-only, lightweight)**

If you only need the Claude backend and don't plan to run local Gemma:

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install anthropic python-dotenv gradio requests
```

### 3. Set your API key

Get a key from [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
```

> The `.env` file is gitignored — your key stays local.

### 4. Run the Web UI

```bash
python scripts/app.py
```

Opens a browser at `http://127.0.0.1:7860` with:
- **Collapsible sidebar** — chat session history (switch between conversations)
- **Chat area** — live tool-call logs stream in real-time, then the final answer appears below
- **Persistent history** — all chats auto-save to `logs/chat_history.json`

### 5. Run the CLI (for debugging)

```bash
python scripts/ask_bible.py --backend claude
```

Add `--quiet` to hide tool-call traces, or `--no-history` for single-turn mode.

---

## File Guide

```
Gemma4/
├── .env                      # Your API key (create this, gitignored)
├── pyproject.toml            # Dependency spec (uv sync reads this)
├── uv.lock                   # Exact lockfile for reproducibility
├── README.md                 # This file
├── SETUP.md                  # Local Gemma 4 E4B setup guide (GPU required)
├── SETUP-RTX6000PRO.md       # Server GPU (96 GB) setup notes
│
├── scripts/
│   ├── fhl_tools.py          # 13 FHL Bible API tools (model-agnostic)
│   ├── claude_bible_rag.py   # Claude agentic RAG engine (Anthropic API)
│   ├── bible_rag.py          # Gemma 4 E4B agentic RAG engine (local GPU)
│   ├── app.py                # Gradio web UI
│   └── ask_bible.py          # CLI REPL (supports both backends)
│
└── logs/
    └── chat_history.json     # Auto-saved chat sessions (gitignored)
```

### `fhl_tools.py` — Bible Tool Layer

13 tools that wrap the 信望愛 REST API. Each tool is a plain Python function with type hints and docstrings. The tool schemas are auto-generated from these functions for both Claude (Anthropic tool_use) and Gemma (HuggingFace chat template).

Book name normalization is built in: `'約翰福音'`, `'約'`, `'John'`, `'Jn'` all resolve to the same book.

### `claude_bible_rag.py` — Claude RAG Engine

The main engine. Sends the user's question to Claude with all 13 tools available. Claude autonomously decides which tools to call, executes them against the FHL API, and chains up to 10 rounds of tool calls before generating a final answer.

- Auto-generates Anthropic tool schemas from `fhl_tools.py` function signatures
- Supports `log_callback` for streaming tool-call logs to the UI
- Uses `claude-opus-4-7` by default (configurable via `MODEL_ID`)

### `app.py` — Gradio Web UI

Chat interface with:
- **Live tool-call streaming** — see each API call as it happens
- **Multi-turn conversation** — full chat history sent to Claude's context
- **Session management** — create, switch, delete chat sessions
- **Persistent storage** — auto-saves to `logs/chat_history.json` after every turn and on shutdown (`Ctrl+C`)

### `ask_bible.py` — CLI REPL

Lightweight terminal interface for debugging. Supports both backends:

```bash
python scripts/ask_bible.py --backend claude    # Anthropic API
python scripts/ask_bible.py --backend gemma     # Local Gemma 4 (requires GPU setup)
```

Useful flags: `--quiet` (hide tool traces), `--no-history` (stateless), `--log path/to/file.log` (transcript).

### `bible_rag.py` — Local Gemma 4 Engine

Runs Gemma 4 E4B locally on an NVIDIA GPU with 4-bit NF4 quantization (~8.7 GB VRAM). Same tool set as the Claude engine, but with a smaller context window (~8K tokens) and local inference.

This is primarily for:
- Verifying tool-calling behavior without API costs
- Offline usage
- Comparing local vs. API model quality

See [SETUP.md](SETUP.md) for GPU environment setup and model download.

---

## Chat History

All conversations are persisted to `logs/chat_history.json`:

- **Auto-saves** after every assistant response
- **Auto-saves on shutdown** — press `Ctrl+C` in terminal to stop cleanly
- **Loads on startup** — previous sessions appear in the sidebar
- **Per-session isolation** — each "New Chat" creates a separate session with its own message history

The file stores both display messages (with tool-call logs) and clean API history (without logs), so multi-turn conversations work correctly across sessions.

> `logs/` is gitignored — your chat history stays local.

---

## Example Questions

```
約翰福音3:16是什麼意思？
聖經中「愛」有幾種？原文有什麼區別？
羅馬書如何論述因信稱義？
創世記1章的內容是什麼？
What does John 1:1 say in Greek?
```

---

## Configuration

| Setting | File | Default |
|---------|------|---------|
| Claude model | `claude_bible_rag.py` line 46 | `claude-opus-4-7` |
| Max tool rounds | `claude_bible_rag.py` line 47 | `10` |
| Max output tokens | `claude_bible_rag.py` line 253 | `16384` |
| Chat history path | `app.py` line 36 | `logs/chat_history.json` |
| System prompt | `claude_bible_rag.py` lines 50–112 | Chinese Bible expert persona |

---

## Local Gemma 4 Setup (Optional)

Only needed if you want to run the local GPU backend (`--backend gemma`). Requires an NVIDIA GPU with 11+ GB VRAM.

See [SETUP.md](SETUP.md) for:
- Environment setup with `uv sync`
- Model weight download (~16 GB)
- CUDA/GPU verification

For server deployment with Gemma 4 31B on 96 GB VRAM, see [SETUP-RTX6000PRO.md](SETUP-RTX6000PRO.md).

---

## License

Scripts in this repo are MIT. The Gemma 4 model weights are under [Google's Gemma license](https://ai.google.dev/gemma/docs/gemma_4_license). This project uses the [信望愛 (FHL)](https://bible.fhl.net/) public Bible API — please respect their terms of service.
