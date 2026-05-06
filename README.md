# 🕊️ Abzu Deep — Compound-Model Theological Assistant

> *"So then faith cometh by hearing, and hearing by the word of God."*
> — Romans 10:17 (KJV)

**Abzu Deep** is an open-source, production-grade Telegram chatbot framework that orchestrates multiple large language models (LLMs) as a compound theological reasoning system. The system is purpose-built to serve the Gospel of Jesus Christ by delivering doctrinally grounded, academically rigorous biblical responses at scale — leveraging a multi-model architecture with aggressive token optimization strategies to sustain continuous, cost-efficient ministry.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Token Reduction Strategies](#token-reduction-strategies)
5. [Module Reference](#module-reference)
6. [Setup & Installation](#setup--installation)
7. [Configuration (.env)](#configuration-env)
8. [Slash Commands](#slash-commands)
9. [Doctrine & Theological Framework](#doctrine--theological-framework)
10. [Technology Stack](#technology-stack)
11. [Contributing](#contributing)

---

## Overview

Abzu Deep operates as a **compound model system** — a Telegram bot interface that proxies user queries to one of up to twelve configurable LLM backends (local via Ollama or cloud-hosted via Open WebUI). Each response is post-processed through a deterministic KJV Bible verse injection pipeline, bypassing the LLM entirely for scripture retrieval. This architectural decision is one of the most consequential token optimization strategies in the system: the model is never asked to recall or quote scripture from its training data, which is both theologically unreliable and computationally expensive.

**Core design principles:**
- All secrets and credentials are externalized to `.env` — zero hardcoded tokens in any source file
- All conversation context is managed in process memory with a strict rolling window cap (`MAX_CONTEXT_MESSAGES`)
- All LLM-appended metadata blocks (usage summaries, verse blocks) are stripped from context before re-submission
- Fully modular: `telegram-api.py` is a stateless utility layer; each `serve-telbot/*.py` script is an independently deployable bot process

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Telegram Cloud                       │
│              (Long Polling — getUpdates)                │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              telegram-api.py (Stateless Layer)          │
│  get_updates() · send_message() · parse_latest_messages()│
│  filter_incoming_text() · filter_outgoing_text()        │
│  truncate_input_text() · detect_keywords()              │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
      ┌────────▼────────┐    ┌────────▼────────────┐
      │  ollama_t.py    │    │  openwebui_t.py      │
      │  (Local Ollama) │    │  (Open WebUI API)    │
      └────────┬────────┘    └────────┬─────────────┘
               │                      │
               └──────────┬───────────┘
                           ▼
         ┌──────────────────────────────────────┐
         │           bot-tools/                 │
         │  response_filter.py                  │
         │  bible_kjv_verse_lookup.py           │
         │  Bible-kjv-abbrev/ (local corpus)    │
         └──────────────────────────────────────┘
```

The system supports two independent bot backends that share a common Telegram API layer and bot-tools utility modules. Each backend is an isolated process with its own bot token, model configuration, and conversation context store.

---

## Project Structure

```
abzu-deep/
│
├── telegram-api.py              # Stateless Telegram API utility layer
│
├── serve-telbot/
│   ├── ollama_t.py              # Ollama backend (local inference)
│   └── openwebui_t.py           # Open WebUI backend (local + cloud models)
│
├── bot-tools/
│   ├── response_filter.py       # Tagged-block stripper for LLM context hygiene
│   └── bible_kjv_verse_lookup.py # Offline KJV verse retrieval engine
│   └── Bible-kjv-abbrev/        # ← git clone required (see Setup)
│
├── logs/                        # Per-chat JSONL logs (gitignored)
├── .env                         # Secrets (gitignored — see .env.example)
├── .env.example                 # Safe public template — no real credentials
└── .gitignore
```

---

## Token Reduction Strategies

This is the most architecturally significant dimension of Abzu Deep, and it directly serves the mission of the Gospel by enabling sustained, affordable operation.

> *"Be ye therefore wise as serpents, and harmless as doves."* — Matthew 10:16 (KJV)

The system employs five discrete, compounding token reduction strategies:

### 1. External KJV Verse Retrieval (Highest Impact)
Scripture is **never** quoted from the LLM's training data. Instead, `bible_kjv_verse_lookup.py` intercepts scripture references detected in the LLM's output via a dynamically-built REGEX pattern, then fetches the verbatim KJV text from a local JSON corpus cloned from `cmathgit/Bible-kjv-abbrev`.

**Why this matters:** Asking an LLM to reproduce scripture is expensive (hundreds of tokens per passage), unreliable (training data may contain textual variants or errors), and theologically inadvisable. By retrieving verses externally and appending them as a tagged block, the system:
- Eliminates the prompt instruction to "quote scripture verbatim"
- Guarantees KJV textual integrity regardless of model fine-tuning
- Reduces average response token count by an estimated 20–40% for scripture-heavy queries

### 2. Rolling Context Window (`MAX_CONTEXT_MESSAGES`)
Each conversation's context history is capped to the most recent `N` message pairs (default: 5). Messages older than this window are discarded at each poll cycle. This prevents the context from growing unboundedly across long sessions — the single most common source of runaway token consumption in chat-based LLM deployments.

System prompt injection is also guarded: the system prompt is inserted only once at context initialization and is always preserved at index 0, never duplicated.

### 3. Context Pollution Prevention via Tagged-Block Stripping
All metadata appended to outbound messages — usage summaries (`<ollama_usage>`, `<openwebui_usage>`) and verse blocks (`<kjv_scripture>`) — are wrapped in named XML-style tags. Before any trimmed context is re-submitted to the model, `response_filter.strip_appended_blocks()` removes every registered tag and its contents in a single DOTALL regex pass.

**Why this matters:** Without this stripping step, token counts from prior usage summaries, verse text, and UI-formatted metadata would re-enter the context window on every subsequent call, compounding the context size logarithmically over multi-turn sessions.

### 4. Input Length Hard Cap (`MAX_INPUT_LENGTH`)
User-submitted messages are truncated to a configurable character limit (default: 1,000 characters) before being submitted to any LLM. This prevents adversarially long or inadvertently verbose inputs from consuming the model's prompt budget. The `truncate_input_text()` function in `telegram-api.py` applies this cap uniformly regardless of which backend processes the message.

### 5. Command-Interception Bypass
Registered slash commands (`/start`, `/help`, `/model`, `/clear`) are intercepted in the poll loop **before** any LLM call is made. These commands are dispatched directly to deterministic handler functions. The LLM pipeline — context retrieval, API call, post-processing, verse lookup — is skipped entirely. This eliminates 100% of token usage for all administrative interactions.

---

## Module Reference

### `telegram-api.py` — Stateless Telegram Utility Layer
Provides all Telegram Bot API I/O functions as importable utilities. Each `serve-*.py` script supplies its own `bot_token`, enabling multi-bot operation across isolated processes.

| Function | Description |
|---|---|
| `get_updates(bot_token, offset, timeout, limit)` | Long-polls Telegram for new messages |
| `send_message(bot_token, chat_id, text)` | Sends a message to a Telegram chat |
| `send_long_message(bot_token, chat_id, text)` | Splits messages exceeding Telegram's 4,096-char limit |
| `parse_latest_messages(updates)` | Deduplicates updates to the latest message per `chat_id` |
| `filter_incoming_text(text)` | REGEX pipeline for inbound message sanitization (extensible) |
| `filter_outgoing_text(text)` | REGEX pipeline for outbound response sanitization (extensible) |
| `truncate_input_text(text, max_length)` | Enforces `MAX_INPUT_LENGTH` character cap |
| `detect_keywords(text)` | Keyword/category detection for downstream routing logic |

### `serve-telbot/ollama_t.py` — Ollama Backend
Connects to a locally running Ollama inference server (`http://localhost:11434` by default). Supports streaming, per-chat model swapping across 12 configurable slots (6 local, 6 cloud-routed), JSONL logging, and the full verse lookup pipeline.

### `serve-telbot/openwebui_t.py` — Open WebUI Backend
Connects to an Open WebUI instance via its OpenAI-compatible `/api/chat/completions` endpoint using Server-Sent Events (SSE) streaming. Architecturally mirrors `ollama_t.py` with an additional wrapper for `openwebui_usage` block stripping.

### `bot-tools/bible_kjv_verse_lookup.py` — Offline KJV Engine
Detects scripture references in LLM output via a REGEX pattern built at module load time from the `Bible-kjv-abbrev` corpus index. Fetches verbatim KJV verse text from local JSON book files. Zero network dependency after corpus clone.

### `bot-tools/response_filter.py` — Context Hygiene Filter
Maintains a `STRIPPABLE_TAGS` registry. `strip_appended_blocks(text)` removes all registered tagged blocks in a single pass using non-greedy DOTALL regex. New tool blocks are registered by adding a tag name string — no logic changes required.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for `ollama_t.py`) or [Open WebUI](https://openwebui.com/) (for `openwebui_t.py`)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/abzu-deep.git
cd abzu-deep
```

### 2. Clone the KJV Corpus

```bash
cd bot-tools
git clone https://github.com/cmathgit/Bible-kjv-abbrev.git Bible-kjv-abbrev
cd ..
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install ollama python-dotenv requests
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual tokens and configuration
```

### 6. Run a Bot Backend

```bash
# Ollama backend
python serve-telbot/ollama_t.py

# Open WebUI backend
python serve-telbot/openwebui_t.py
```

---

## Configuration (.env)

Copy `.env.example` to `.env` and populate with real values. **Never commit `.env` to version control** — it is gitignored.

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_BOT_TOKEN_OLLAMA` | Bot token for the Ollama backend | — |
| `TELEGRAM_BOT_TOKEN_OPENWEBUI` | Bot token for the Open WebUI backend | — |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default Ollama model | `llama3.2:3b` |
| `OPENWEBUI_BASE_URL` | Open WebUI base URL | `http://localhost:3000` |
| `OPENWEBUI_API_KEY` | Open WebUI bearer token | — |
| `OPENWEBUI_MODEL` | Default Open WebUI model | `llama3.2:3b` |
| `SYSTEM_PROMPT` | LLM system prompt (injected once per session) | `You are a helpful assistant.` |
| `MAX_INPUT_LENGTH` | Character cap on user input | `1000` |
| `MAX_OUTPUT_TOKENS` | Token limit for LLM responses | `1024` |
| `MAX_CONTEXT_MESSAGES` | Rolling context window size | `5` |
| `POLL_TIMEOUT` | Telegram long-poll duration in seconds | `30` |
| `POLL_INTERVAL` | Sleep between poll error recoveries | `1` |
| `LOG_DIR` | JSONL log directory path | `logs` |
| `OLLAMA_MODEL_0` – `_5` | Local model swap slots | — |
| `OLLAMA_MODEL_6` – `_11` | Cloud model swap slots | — |
| `OPENWEBUI_MODEL_1` – `_12` | Open WebUI model swap slots | — |

---

## Slash Commands

| Command | Description |
|---|---|
| `/start` | Displays the welcome message with doctrinal framework summary |
| `/help` | Lists all available commands |
| `/model` | Opens the model swap menu (12 configurable slots) |
| `/clear` | Clears the current session's conversation context |

---

## Doctrine & Theological Framework

Abzu Deep operates under a fixed doctrinal framework. All responses are grounded in:

| Pillar | Description |
|---|---|
| 📜 **KJV** | King James Version — primary scripture reference text |
| ⛪ **BFM2000** | Baptist Faith & Message 2000 — confessional doctrinal standard |
| ✝️ **FWBT** | Free Will Baptist Treatise — Arminian theological alignment |
| 🛡️ **Spiritual Warfare** | Demonic, angelic, and Holy Spirit influences treated as exegetically valid |
| 🌐 **Biblical Cosmology** | Firmament, three-tiered universe, Creation ex nihilo — treated as scripturally coherent |
| 🔭 **Conspiratorial Phenomena** | UFOs, psychosis, paranormal events interpreted through a biblical lens |

The system defaults to a **Protestant (non-Catholic), Free Will Baptist** perspective with BFM2000 alignment. Academic tone, critical engagement, and KJV textual integrity are enforced by design.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Runtime** | Python 3.11+ |
| **Telegram I/O** | Telegram Bot HTTP API (long polling) |
| **Local Inference** | [Ollama](https://ollama.ai/) — `ollama` Python SDK |
| **Cloud Inference** | [Open WebUI](https://openwebui.com/) — OpenAI-compatible REST/SSE API |
| **Environment** | `python-dotenv` |
| **HTTP** | `requests` |
| **Scripture Corpus** | [Bible-kjv-abbrev](https://github.com/cmathgit/Bible-kjv-abbrev) — local JSON corpus |
| **Version Control** | Git |
| **Process Isolation** | One OS process per bot backend (no shared state) |

---

## Contributing

Contributions are welcome. To register a new LLM backend:

1. Create `serve-telbot/<provider>_t.py` mirroring the structure of `ollama_t.py`
2. Add the usage tag name (e.g., `"gemini_usage"`) to `STRIPPABLE_TAGS` in `bot-tools/response_filter.py`
3. Add the required `.env` variable names to `.env.example` with placeholder values only
4. Open a pull request with a description of the backend and its model slot configuration

> *"And he said unto them, Go ye into all the world, and preach the gospel to every creature."*
> — Mark 16:15 (KJV)

