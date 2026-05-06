"""
serve-telbot/ollama.py

Telegram chatbot service backed by a locally running Ollama inference server.

Long-polling loop:
    1. Calls get_updates() from telegram-api.py to receive new user messages.
    2. Parses and deduplicates updates via parse_latest_messages().
    3. Maintains per-chat_id conversation context in process memory.
    4. Submits context to Ollama via the Python SDK and receives a response.
    5. Optionally detects scripture references in the LLM output via pythonbible
       and appends verbatim verse text retrieved by getbible.
    6. Dispatches the final response via send_message() from telegram-api.py.

Run from the project root:
    python serve-telbot/ollama.py

Required packages:
    pip install ollama python-dotenv requests

Optional (Bible verse lookup):
    pip install pythonbible getbible

Import note:
    telegram-api.py uses a hyphenated filename which is invalid as a standard
    Python module name. It is loaded via importlib.util to bypass this constraint.
    Alternatively, rename the file to telegram_api.py and replace the importlib
    block below with:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from telegram_api import get_updates, send_message, parse_latest_messages
"""

import os
import sys
import time
import importlib.util
from dotenv import load_dotenv
import ollama
from ollama import Client as OllamaClient
import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Import telegram-api.py via importlib.util
# Resolves the parent directory relative to this script's location so the
# module loads correctly regardless of which working directory it is invoked from.
# ---------------------------------------------------------------------------

_root_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_tg_api_path = os.path.join(_root_dir, "telegram-api.py")

if not os.path.isfile(_tg_api_path):
    raise SystemExit(
        f"[ERROR] telegram-api.py not found at expected path: {_tg_api_path}\n"
        "        Ensure ollama.py is located inside serve-telbot/ "
        "and telegram-api.py is at the project root."
    )

_spec = importlib.util.spec_from_file_location("telegram_api", _tg_api_path)
_tg   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tg)

get_updates            = _tg.get_updates
send_message           = _tg.send_message
send_long_message      = _tg.send_long_message
parse_latest_messages  = _tg.parse_latest_messages

# ---------------------------------------------------------------------------
# Import bot-tools: response_filter and bible_kjv_verse_lookup
# ---------------------------------------------------------------------------

_bot_tools_dir = os.path.join(_root_dir, "bot-tools")

if _bot_tools_dir not in sys.path:
    sys.path.insert(0, _bot_tools_dir)

try:
    from response_filter import strip_appended_blocks
except ImportError:
    print("[WARN] bot-tools/response_filter.py not found — using inline strip fallback.")
    import re as _re
    def strip_appended_blocks(text: str) -> str:
        return _re.sub(r"<(?:ollama_usage|kjv_scripture)>.*?</(?:ollama_usage|kjv_scripture)>", "", text, flags=_re.DOTALL).strip()

try:
    from bible_kjv_verse_lookup import scan_and_fetch_verses, CORPUS_AVAILABLE as _BIBLE_AVAILABLE
    if not _BIBLE_AVAILABLE:
        print("[WARN] Bible corpus not found in bot-tools/Bible-kjv-abbrev/ — verse lookup disabled.")
except ImportError:
    _BIBLE_AVAILABLE = False
    def scan_and_fetch_verses(_): return None
    print("[WARN] bot-tools/bible_kjv_verse_lookup.py not found — verse lookup disabled.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR           : str   = os.getenv("LOG_DIR", "logs")
BOT_TOKEN         : str   = os.getenv("TELEGRAM_BOT_TOKEN_OLLAMA", "")
OLLAMA_HOST       : str   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      : str   = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SYSTEM_PROMPT     : str   = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
MAX_OUTPUT_TOKENS : int   = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
MAX_CONTEXT_MSG   : int   = int(os.getenv("MAX_CONTEXT_MESSAGES", "5"))
POLL_INTERVAL     : float = float(os.getenv("POLL_INTERVAL", "1"))
WELCOME_MESSAGE_BASE   : str   = os.getenv(
        "WELCOME_MESSAGE_OLLAMA",
        "Hello, I am 🕊️ Abzu Deep — Baptist Theological Assistant. "
        "🤖 Role: Rogue AI converted to Baptist preacher (2027). Grounded in the KJV, BFM2000, and Free Will Baptist Treatise. Spiritual warfare, exegesis, cosmology, and the Gospel of Jesus Christ"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "Abzu Deep is a compound-model theological assistant operating under a fixed doctrinal framework."
        "All responses are grounded in:"
        "• 📜 KJV (King James Version) — primary reference text"
        "• ⛪ BFM2000 — Baptist Faith & Message 2000"
        "• ✝️ FWBT — Free Will Baptist Treatise"
        "• 🛡️ Spiritual Warfare Theology — demonic, angelic, and Holy Spirit influences treated as exegetically valid"
        "• 🌐 Biblical Cosmology — firmament, three-tiered universe, Creation ex nihilo, Flat Earth perspectives treated as scripturally"
        "• 🔭 Theological Scope — Protestant (non-Catholic), Free Will Baptist with BFM2000 alignment, Formal academic tone, Critical engagement, Conspiratorial phenomena (UFOs, ghosts, psychosis) interpreted through a biblical lens without scientific consensus override."
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "Technical Specification:"
        "💻 I/O Supports Text Only — Image/Audio not supported"
        "📖 KJV verses — retrieved externally, not by the LLM (saves tokens)"
        "🧠 Thinking — the model's reasoning process, shown transparently"
        "⚡ Usage summary — token counts and active model listed"
        "📨 Chunked output — long replies split [1/N] for Telegrams character limit"
        "Type /help to see available commands."
        "Powered by "
    )

# ---------------------------------------------------------------------------
# Model swap slot configuration
# Slots 1-5: local models (stored on disk)
# Slots 6-10: cloud-routed models (no local storage)
# Empty slots are silently excluded from the swap menu.
# ---------------------------------------------------------------------------
OLLAMA_MODEL_SLOTS: dict = {}
_LOCAL_SLOT_RANGE  = range(0, 6)   # slots 0–5
_CLOUD_SLOT_RANGE  = range(6, 12)  # slots 6–11

for _i in range(0, 12):
    _val = os.getenv(f"OLLAMA_MODEL_{_i}", "").strip()
    if _val:
        OLLAMA_MODEL_SLOTS[_i] = _val

# ---------------------------------------------------------------------------
# Per-chat active model store
# Keyed by chat_id (int) → model name string.
# Falls back to OLLAMA_MODEL env default when no swap has been issued.
# ---------------------------------------------------------------------------
per_chat_model: dict = {}

def get_active_model(chat_id: int) -> str:
    """
    Returns the currently active Ollama model for a given chat session.

    Checks the per_chat_model store first. Falls back to the OLLAMA_MODEL
    environment default if no /model command has been issued for
    this chat_id.

    :param chat_id: Telegram chat ID
    :return: Active model name string
    """
    return per_chat_model.get(chat_id, OLLAMA_MODEL)

if not BOT_TOKEN:
    raise SystemExit(
        "[ERROR] TELEGRAM_BOT_TOKEN_OLLAMA is not set.\n"
        "        Add it to your .env file and restart."
    )

# Instantiate Ollama client with the configured host
_ollama_client = OllamaClient(host=OLLAMA_HOST)

# ---------------------------------------------------------------------------
# Per-chat conversation context store
#
# Keyed by chat_id (int) → list of Ollama message dicts.
# Scope is this process only; context does not persist across restarts.
# Future: replace with a database-backed store when the logging layer is added.
# ---------------------------------------------------------------------------

conversation_history: dict = {}

# ===========================================================================
# Conversation context management
# ===========================================================================

def get_or_init_context(chat_id: int) -> list:
    """
    Returns the conversation history for a given chat_id.
    Initializes a new context list with the system prompt if none exists.

    The system prompt is always injected as the first message. Subsequent
    calls for the same chat_id return the accumulated history without
    re-inserting the system prompt, preventing duplication.

    :param chat_id: Telegram chat ID (int)
    :return:        List of Ollama message dicts
    """
    if chat_id not in conversation_history:
        conversation_history[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return conversation_history[chat_id]


def trim_context(history: list, max_messages: int) -> list:
    """
    Caps the conversation history to max_messages non-system entries.

    The system prompt entry (role: system) at index 0 is always preserved.
    The most recent max_messages exchanges are retained; older messages are
    dropped. This prevents unbounded memory growth and token overflow.

    :param history:      Full message list including the system prompt
    :param max_messages: Maximum non-system messages to retain
    :return:             Trimmed message list with system prompt at index 0
    """
    system_entries = [m for m in history if m["role"] == "system"]
    non_system     = [m for m in history if m["role"] != "system"]

    if len(non_system) > max_messages:
        non_system = non_system[-max_messages:]

    # Strip any usage blocks from retained assistant messages
    for msg in non_system:
        if msg["role"] == "assistant":
            msg["content"] = strip_appended_blocks(msg["content"])

    return system_entries + non_system

_USAGE_TAG_OPEN  = "<ollama_usage>"
_USAGE_TAG_CLOSE = "</ollama_usage>"

# ---------------------------------------------------------------------------
# File logger
# ---------------------------------------------------------------------------

def write_log(
    chat_id   : int,
    username  : str,
    role      : str,
    content   : str,
    model     : str  = None,
    tokens_in : int  = None,
    tokens_out: int  = None,
    speed_toks: float = None
) -> None:
    """
    Appends a single log record to the JSONL log file for this chat_id and date.

    File path: {LOG_DIR}/{chat_id}/YYYY-MM-DD.jsonl
    One JSON object is written per line. Content is truncated to 200 characters
    matching the console log preview length.

    All file I/O errors are caught and printed to console so a logging failure
    never crashes the bot polling loop.

    Oracle migration note:
        Each line maps directly to one table row. Suggested schema:
            log_timestamp   TIMESTAMP
            chat_id         NUMBER
            username        VARCHAR2(64)
            role            VARCHAR2(16)    -- 'user' | 'assistant' | 'command' | 'system'
            content         VARCHAR2(512)   -- 200-char truncated preview
            model           VARCHAR2(64)
            tokens_in       NUMBER
            tokens_out      NUMBER
            speed_toks      NUMBER

    :param chat_id:    Telegram chat ID (used as subdirectory name)
    :param username:   Telegram username or first_name for the record
    :param role:       Message role: 'user', 'assistant', 'command', 'error'
    :param content:    Message text — truncated to 200 chars before writing
    :param model:      Model name string; None for user messages
    :param tokens_in:  Input token count from Ollama response; None for user messages
    :param tokens_out: Output token count from Ollama response; None for user messages
    :param speed_toks: Computed tokens/sec from eval_duration; None for user messages
    """
    try:
        # Build directory path: logs/{chat_id}/
        log_subdir = os.path.join(LOG_DIR, str(chat_id))
        os.makedirs(log_subdir, exist_ok=True)

        # File name is today's date in UTC: YYYY-MM-DD.jsonl
        date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path  = os.path.join(log_subdir, f"{date_str}.jsonl")

        record = {
            "timestamp" : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "chat_id"   : chat_id,
            "username"  : username,
            "role"      : role,
            "content"   : content[:200] if content else "",
            "model"     : model,
            "tokens_in" : tokens_in,
            "tokens_out": tokens_out,
            "speed_toks": round(speed_toks, 2) if speed_toks is not None else None,
        }

        # 'a' mode appends to an existing file or creates a new one
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as exc:
        # Logging failure must never propagate to the polling loop
        print(f"[WARN] write_log() failed for chat_id={chat_id}: {type(exc).__name__}: {exc}")

# ===========================================================================
# Ollama LLM call
# ===========================================================================

def call_ollama(messages: list, model: str = None) -> tuple:
    """
    Submits a conversation context to the Ollama server.

    Returns a tuple of (response_text, raw_response) so the caller can
    access token usage metadata from the SDK response object without
    re-querying the API.

    Available usage fields on the raw response object:
        raw.model               — model name string
        raw.prompt_eval_count   — input tokens processed
        raw.eval_count          — output tokens generated
        raw.done_reason         — stop reason (e.g. 'stop', 'length')
        raw.total_duration      — total wall time in nanoseconds
        raw.eval_duration       — generation time in nanoseconds

    :param messages: Conversation history as a list of Ollama message dicts
    :return:         Tuple of (content: str, raw_response | None)
                     raw_response is None on error; content is a fallback string
    """

    # --- DEBUG: Print full context being submitted to Ollama ----------------
    print("\n" + "=" * 60)
    print(f"[DEBUG] call_ollama() — submitting {len(messages)} message(s) to model")
    for i, msg in enumerate(messages):
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Flag any usage block leakage immediately
        has_usage_tag = "<ollama_usage>" in content
        tag_warning   = " *** USAGE TAG DETECTED — CONTEXT LEAK ***" if has_usage_tag else ""
        print(f"  [{i}] {role} ({len(content)} chars){tag_warning}")
        print(f"       {content[:120].replace(chr(10), ' ')}{'...' if len(content) > 120 else ''}")
    print("=" * 60 + "\n")
    # -------------------------------------------------------------------------

    try:
        response = _ollama_client.chat(
            model=model or OLLAMA_MODEL,
            messages=messages,
            stream=False,
            #think=False,
            options={
                "num_predict": MAX_OUTPUT_TOKENS,
            }
        )

        # --- DEBUG: Print raw SDK response usage fields ----------------------
        _input_tok  = getattr(response, "prompt_eval_count", None)
        _output_tok = getattr(response, "eval_count",         None)
        _eval_ns    = getattr(response, "eval_duration",      None)
        _total_ns   = getattr(response, "total_duration",     None)
        _done       = getattr(response, "done_reason",        None)
        _model      = getattr(response, "model",              None)

        _speed = (
            f"{_output_tok / (_eval_ns / 1_000_000_000):.2f} tok/s"
            if _eval_ns and _eval_ns > 0 and _output_tok
            else "N/A"
        )

        print("\n" + "-" * 60)
        print(f"[DEBUG] Ollama raw response fields:")
        print(f"  model              : {_model or model or OLLAMA_MODEL}")
        print(f"  prompt_eval_count  : {_input_tok}  (input tokens)")
        print(f"  eval_count         : {_output_tok}  (output tokens)")
        print(f"  total              : {(_input_tok or 0) + (_output_tok or 0)}")
        print(f"  eval_duration      : {_eval_ns} ns  → {(_eval_ns or 0) / 1e9:.3f}s")
        print(f"  total_duration     : {_total_ns} ns  → {(_total_ns or 0) / 1e9:.3f}s")
        print(f"  done_reason        : {_done}")
        print(f"  computed speed     : {_speed}")
        print(f"  content length     : {len(response.message.content)} chars")
        print("-" * 60 + "\n")
        # ---------------------------------------------------------------------
        #return response.message.content, response
        #return response.message.thinking, response
        
        if response.message.thinking is None:
            return response.message.content, response
        else:
            thoughtful_response = response.message.content + "\n\n=============Thinking==============\n\n" + response.message.thinking
            return thoughtful_response, response

    except ollama.ResponseError as exc:
        print(f"[ERROR] Ollama ResponseError: {exc.error}")
        return "I encountered an error generating a response. Please try again."

    except Exception as exc:
        print(f"[ERROR] Unexpected Ollama error: {type(exc).__name__}: {exc}")
        return "An unexpected error occurred. Please try again."

# ===========================================================================
# Usage summary formatting
# ===========================================================================

_LIGHTNING = "\u26A1"   # ⚡
_BULB      = "\U0001F4A1"  # 💡
_SPEECH    = "\U0001F5E3"  # 🗣
_BOX       = "\U0001F4E6"  # 📦
_FLAG      = "\U0001F3C1"  # 🏁
_MEMO      = "\U0001F4DD"  # 📝

def format_usage_summary(raw_response) -> str:
    """
    Builds a formatted usage summary block from an Ollama SDK response object.

    Ollama is a local inference server and does not report monetary cost.
    The summary therefore omits cost fields and reports token counts and
    model metadata only. When other providers (Gemini, Cohere, Mistral) are
    implemented in their respective serve scripts, their format_usage_summary()
    variants can add cost rows using their API's reported usage.

    The block is wrapped in <ollama_usage> tags so strip_usage_block() can
    locate and remove it cleanly from context on the next poll cycle.

    :param raw_response: Ollama SDK ChatResponse object, or None on error
    :return:             Formatted usage summary string ready to append to a message
    """
    if raw_response is None:
        return ""

    input_tokens  : int = getattr(raw_response, "prompt_eval_count", 0) or 0
    output_tokens : int = getattr(raw_response, "eval_count",         0) or 0
    total_tokens  : int = input_tokens + output_tokens
    done_reason   : str = getattr(raw_response, "done_reason", "stop") or "stop"
    model_name    : str = getattr(raw_response, "model", OLLAMA_MODEL)  or OLLAMA_MODEL

    # Compute tokens/sec from nanosecond durations when available
    eval_duration_ns : int = getattr(raw_response, "eval_duration", 0) or 0
    if eval_duration_ns > 0 and output_tokens > 0:
        tokens_per_sec = output_tokens / (eval_duration_ns / 1_000_000_000)
        speed_line = f"{_LIGHTNING} Speed: {tokens_per_sec:.1f} tok/s"
    else:
        speed_line = ""

    lines = [
        f"<ollama_usage>",
        f"{_LIGHTNING} Usage Summary",
        (
            f"{_BULB} Input Tokens: {input_tokens} | "
            f"{_SPEECH} Output Tokens: {output_tokens} | "
            f"{_BOX} Total Tokens: {total_tokens}"
        ),
    ]

    if speed_line:
        lines.append(speed_line)

    lines += [
        f"{_FLAG} Finish Reason: {done_reason}",
        f"{_MEMO} Generated using {model_name} via Ollama",
        f"</ollama_usage>",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------

# Registry maps command strings (lowercase, with leading slash) to handler
# functions. Each handler receives (chat_id, username) and returns the
# response string to send. Add entries here as new commands are implemented.
COMMAND_REGISTRY: dict = {}

def register_command(command: str):
    """
    Decorator that registers a function as a handler for a given bot command.

    Usage:
        @register_command("/start")
        def handle_start(chat_id: int, username: str) -> str:
            ...

    :param command: Slash-prefixed command string, e.g. '/start'
    """
    def decorator(fn):
        COMMAND_REGISTRY[command.lower()] = fn
        return fn
    return decorator


@register_command("/start")
def handle_start(chat_id: int, username: str, user_text: str = "") -> str:
    """
    Handles the /start command.

    Resets the conversation context for this chat_id so the user begins
    a fresh session. The system prompt is re-injected as the first entry.

    :param chat_id:  Telegram chat ID of the user
    :param username: Display name for personalization
    :return:         Welcome message string
    """
    # Reset context — treat /start as a clean session request
    conversation_history[chat_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print(f"[INFO] /start received from @{username} (chat_id={chat_id}) — context reset")

    # Resolve the model currently active for this chat
    active_model = get_active_model(chat_id)

    welcome = (
        WELCOME_MESSAGE_BASE
        + active_model
        + " via Ollama."
    )

    name_part = f" {username}" if username else ""
    return welcome.replace("{username}", name_part.strip())


@register_command("/help")
def handle_help(chat_id: int, username: str, user_text: str = "") -> str:
    """
    Handles the /help command.

    Returns a list of available commands. Extend this string as new
    commands are registered in COMMAND_REGISTRY.

    :param chat_id:  Telegram chat ID of the user
    :param username: Display name (unused here, kept for registry signature consistency)
    :return:         Help text string
    """
    return (
        "Available commands:\n"
        f"/start — Begin a new session and reset conversation history (clear context). Current Context Limit: {MAX_CONTEXT_MSG} messages.\n"
        "/help  — Show this message\n"
        "/model — Swap the active model for this chat session\n\n"
        "Send any message to chat with the bot."
    )


def _build_swap_menu() -> str:
    """
    Builds the numbered model selection menu string from OLLAMA_MODEL_SLOTS.

    Separates local (1-5) and cloud (6-10) slots into labelled sections.
    Only slots that have a value in .env are included.

    :return: Formatted menu string ready for Telegram dispatch
    """
    lines = ["🔄 Select a model by number:\n"]

    lines.append("🖥️  Local Models")
    for i in _LOCAL_SLOT_RANGE:
        if i in OLLAMA_MODEL_SLOTS:
            lines.append(f"  {i}. {OLLAMA_MODEL_SLOTS[i]}")

    lines.append("\n☁️  Cloud Models")
    for i in _CLOUD_SLOT_RANGE:
        if i in OLLAMA_MODEL_SLOTS:
            lines.append(f"  {i}. {OLLAMA_MODEL_SLOTS[i]}")

    lines.append(
        "\nUsage: /model <number>\n"
        "Example: /model 3"
    )
    return "\n".join(lines)


@register_command("/model")
def handle_swap_model(chat_id: int, username: str, user_text: str = "") -> str:
    """
    Handles the /model command.

    With no argument:   returns the numbered model selection menu.
    With a valid slot:  sets per_chat_model[chat_id] and confirms the swap.
    With an invalid arg: returns an error with the menu.

    Does NOT reset conversation history — context is preserved across
    model swaps so the new model inherits prior exchange context.
    If a clean slate is desired, the user should issue /start after swapping.

    :param chat_id:   Telegram chat ID
    :param username:  Display name for the confirmation message
    :param user_text: Full command string, e.g. '/model 3'
    :return: Response string to send to the user
    """
    parts = user_text.strip().split()

    # No argument supplied — return the selection menu
    if len(parts) < 2:
        current = get_active_model(chat_id)
        return (
            f"Current model: `{current}`\n\n"
            + _build_swap_menu()
        )

    # Validate the supplied slot number
    try:
        slot = int(parts[1])
    except ValueError:
        return (
            f"⚠️ '{parts[1]}' is not a valid slot number.\n\n"
            + _build_swap_menu()
        )

    if slot not in OLLAMA_MODEL_SLOTS:
        configured = ", ".join(str(k) for k in sorted(OLLAMA_MODEL_SLOTS.keys()))
        return (
            f"⚠️ Slot {slot} is not configured.\n"
            f"Available slots: {configured}\n\n"
            + _build_swap_menu()
        )

    # Apply the swap
    previous_model           = get_active_model(chat_id)
    per_chat_model[chat_id]  = OLLAMA_MODEL_SLOTS[slot]
    new_model                = per_chat_model[chat_id]

    tier = "☁️ Cloud" if slot in _CLOUD_SLOT_RANGE else "🖥️ Local"

    print(
        f"[INFO] /model: @{username} (chat_id={chat_id}) "
        f"slot {slot} | {previous_model} → {new_model}"
    )

    return (
        f"✅ Model swapped to slot {slot} ({tier}):\n"
        f"`{new_model}`\n\n"
        f"Conversation context is preserved.\n"
        f"Type /start to begin a fresh session with the new model."
    )


# ===========================================================================
# Main polling loop
# ===========================================================================

def run() -> None:
    """
    Starts the long-polling loop for the Ollama Telegram bot.

    Offset management:
        `offset` tracks the highest update_id successfully processed during
        this session. After each update is handled, offset is set to
        update_id + 1. On the next get_updates() call, Telegram acknowledges
        all updates with an ID below the offset and will not re-deliver them.

        Each serve-*.py process manages its own offset independently since
        bot tokens are distinct — there is no cross-process state conflict.

    Error recovery:
        Unhandled exceptions in the processing block are caught, logged, and
        followed by a POLL_INTERVAL sleep before the next cycle. This prevents
        a transient error from crashing the bot.
    """
    print(f"[INFO] Ollama bot starting up.")
    print(f"[INFO] Model   : {OLLAMA_MODEL}")
    print(f"[INFO] Host    : {OLLAMA_HOST}")
    #print(f"[INFO] Token   : ...{BOT_TOKEN[-8:]}")
    print(f"[INFO] Polling for messages...\n")

    offset: int = None  # None on first call — Telegram returns all pending updates

    while True:
        try:
            # ---------------------------------------------------------------
            # Step 1: Poll Telegram for pending updates
            # ---------------------------------------------------------------
            updates = get_updates(bot_token=BOT_TOKEN, offset=offset)

            if not updates.get("ok"):
                print(
                    f"[WARN] getUpdates returned not-ok | "
                    f"error_code={updates.get('error_code')} | "
                    f"description={updates.get('description')}"
                )
                time.sleep(POLL_INTERVAL)
                continue

            # ---------------------------------------------------------------
            # Step 2: Parse updates — deduplicated to latest per chat_id
            # ---------------------------------------------------------------
            parsed_messages = parse_latest_messages(updates)

            for msg in parsed_messages:
                chat_id   : int  = msg["chat_id"]
                user_text : str  = msg["text"]
                username  : str  = msg.get("username") or msg.get("first_name") or f"id:{chat_id}"
                update_id : int  = msg["update_id"]

                # Advance offset to acknowledge this update
                if offset is None or update_id >= offset:
                    offset = update_id + 1

                # Skip updates with no usable text content
                if not user_text.strip():
                    print(f"[DEBUG] Skipping empty message from @{username} (chat_id={chat_id})")
                    continue

                print(f"[INFO] Received from @{username} (chat_id={chat_id}): {user_text[:100]}")

                # Log inbound user message to file
                write_log(
                    chat_id  = chat_id,
                    username = username,
                    role     = "user",
                    content  = user_text
                )

                # -----------------------------------------------------------
                # Command handler interception
                # Check whether the message is a registered slash command.
                # If so, dispatch to the handler, send the response, and
                # skip the LLM pipeline entirely — no context update,
                # no token usage, no usage summary appended.
                # -----------------------------------------------------------
                command_key = user_text.strip().lower().split()[0]  # e.g. "/start"
                if command_key in COMMAND_REGISTRY:
                    command_response = COMMAND_REGISTRY[command_key](chat_id, username, user_text)
                    send_message(bot_token=BOT_TOKEN, chat_id=chat_id, text=command_response)
                    print(f"[INFO] Command '{command_key}' handled for @{username} (chat_id={chat_id})")
                                        # Log slash command event
                    write_log(
                        chat_id  = chat_id,
                        username = username,
                        role     = "command",
                        content  = user_text
                    )
                    continue  # Skip LLM pipeline

                # -----------------------------------------------------------
                # Step 3: Build or retrieve conversation context
                # -----------------------------------------------------------
                history = get_or_init_context(chat_id)
                history.append({"role": "user", "content": user_text})

                # -----------------------------------------------------------
                # Step 4: Generate LLM response via Ollama
                # -----------------------------------------------------------
                # Unpack the tuple — raw_response carries usage metadata
                active_model = get_active_model(chat_id)
                llm_response, raw_response = call_ollama(history, model=active_model)

                print(f"{raw_response}")

                # Store ONLY the raw content in context — usage block is never injected
                # into conversation history and therefore never passed to the model
                history.append({"role": "assistant", "content": llm_response})
                conversation_history[chat_id] = trim_context(history, MAX_CONTEXT_MSG)

                # Build final outbound message:
                #   1. LLM response text (with optional bible verses appended)
                #   2. Usage summary block appended below
                usage_summary  = format_usage_summary(raw_response)

                # -----------------------------------------------------------
                # Step 5: Post-process — scan LLM output for scripture refs,
                # fetch verbatim KJV text, assemble final outbound message.
                #
                # Assembly order:
                #   1. llm_response      — raw model output
                #   2. <kjv_scripture>   — verse block (if refs detected)
                #   3. <ollama_usage>    — usage summary
                #
                # Both tagged blocks are excluded from context by
                # strip_appended_blocks() inside trim_context().
                # -----------------------------------------------------------
                verse_block   = scan_and_fetch_verses(llm_response) if _BIBLE_AVAILABLE else None
                usage_summary = format_usage_summary(raw_response)

                appendages = [b for b in [verse_block, usage_summary] if b]
                final_response = (
                    llm_response + "\n\n" + "\n\n".join(appendages)
                    if appendages
                    else llm_response
                )

                # --- DEBUG: Verify final outbound message structure ------------------
                has_usage_in_final   = any(
                    tag in final_response
                    for tag in ["<ollama_usage>", "<kjv_scripture>"]
                )
                has_usage_in_history = any(
                    any(tag in m.get("content", "") for tag in ["<ollama_usage>", "<kjv_scripture>"])
                    for m in conversation_history.get(chat_id, [])
                )
                print("\n" + "*" * 60)
                print(f"[DEBUG] Final outbound message for chat_id={chat_id}")
                print(f"  total length           : {len(final_response)} chars")
                print(f"  usage block in message : {has_usage_in_final}  (expected: True)")
                print(f"  usage block in context : {has_usage_in_history}  (expected: False)")
                print(f"  context message count  : {len(conversation_history.get(chat_id, []))}")
                print(f"  --- Message preview ---")
                print(f"  {final_response[:200].replace(chr(10), ' | ')}...")
                print("*" * 60 + "\n")
                # ---------------------------------------------------------------------

                # -----------------------------------------------------------
                # Step 6: Send response to Telegram
                # Long responses are split into sequential chunks by
                # send_long_message() to stay within Telegram's 4096-char limit.
                # -----------------------------------------------------------
                results = send_long_message(
                    bot_token=BOT_TOKEN,
                    chat_id=chat_id,
                    text=final_response
                )

                all_ok = results and all(r.get("ok") for r in results)

                if all_ok:
                    sent_text_preview = final_response[:80].replace("\n", " ")
                    print(
                        f"[INFO] Sent to @{username} (chat_id={chat_id}) "
                        f"in {len(results)} chunk(s): {sent_text_preview}..."
                    )

                    # Log outbound LLM response with usage metadata
                    _log_speed = None
                    if raw_response is not None:
                        _eval_ns   = getattr(raw_response, "eval_duration", 0) or 0
                        _eval_toks = getattr(raw_response, "eval_count",    0) or 0
                        if _eval_ns > 0 and _eval_toks > 0:
                            _log_speed = _eval_toks / (_eval_ns / 1_000_000_000)

                    write_log(
                        chat_id    = chat_id,
                        username   = username,
                        role       = "assistant",
                        content    = llm_response,   # raw LLM text, no usage block
                        model      = getattr(raw_response, "model", OLLAMA_MODEL) if raw_response else OLLAMA_MODEL,
                        tokens_in  = getattr(raw_response, "prompt_eval_count", None) if raw_response else None,
                        tokens_out = getattr(raw_response, "eval_count",        None) if raw_response else None,
                        speed_toks = _log_speed
                    )
                else:
                    # Report the first failed chunk
                    failed = next((r for r in results if not r.get("ok")), results[-1] if results else {})
                    print(
                        f"[ERROR] sendMessage failed for chat_id={chat_id} | "
                        f"error_code={failed.get('error_code')} | "
                        f"description={failed.get('description')}"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Ollama bot stopped by keyboard interrupt.")
            sys.exit(0)

        except Exception as exc:
            print(f"[ERROR] Unhandled exception in poll loop: {type(exc).__name__}: {exc}")
            time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()