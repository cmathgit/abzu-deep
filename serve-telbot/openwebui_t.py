"""
serve-telbot/openwebui_t.py

Telegram chatbot service backed by Open WebUI's OpenAI-compatible API.
Mirrors ollama_t.py exactly:
  - Long-polling loop via telegram-api.py
  - Per-chat conversation context in process memory
  - KJV Bible verse lookup via bot-tools/bible_kjv_verse_lookup.py
  - Tagged block stripping via bot-tools/response_filter.py (wrapped — see note)
  - /start, /help, /model command registry
  - JSONL file logging per chat_id
  - 12 configurable model slots via .env
  - Streaming SSE via requests (distributes bandwidth, mirrors stream=True in ollama_t.py)

Open WebUI API reference: https://docs.openwebui.com/reference/api-endpoints/

NOTE — response_filter.py compatibility:
  response_filter.py strips <ollama_usage> and <kjv_scripture> tags.
  This script uses <openwebui_usage> for its usage block.
  strip_appended_blocks() is wrapped locally below to handle <openwebui_usage>
  without modifying response_filter.py.

Reused from .env (no new entries needed):
  SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, MAX_CONTEXT_MESSAGES,
  POLL_INTERVAL, MAX_INPUT_LENGTH, POLL_TIMEOUT, LOG_DIR

New .env vars required:
  TELEGRAM_BOT_TOKEN_OPENWEBUI   — bot token from @BotFather
  OPENWEBUI_BASE_URL             — e.g. http://localhost:3000
  OPENWEBUI_API_KEY              — Bearer token from Open WebUI settings
  OPENWEBUI_MODEL                — default model name (fallback)
  OPENWEBUI_MODEL_1 .. _12       — swap slot model names
  WELCOME_MESSAGE_OPENWEBUI      — optional custom welcome message base

Run from the project root:
  python serve-telbot/openwebui_t.py
"""

import os
import sys
import re
import time
import json
import importlib.util
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import telegram-api.py via importlib (hyphenated filename workaround)
# ---------------------------------------------------------------------------
_root_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_tg_api_path = os.path.join(_root_dir, "telegram-api.py")

if not os.path.isfile(_tg_api_path):
    raise SystemExit(
        f"[ERROR] telegram-api.py not found at: {_tg_api_path}\n"
        "        Ensure openwebui_t.py is inside serve-telbot/ "
        "and telegram-api.py is at the project root."
    )

_spec = importlib.util.spec_from_file_location("telegram_api", _tg_api_path)
_tg   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tg)

get_updates           = _tg.get_updates
send_message          = _tg.send_message
send_long_message     = _tg.send_long_message
parse_latest_messages = _tg.parse_latest_messages

# ---------------------------------------------------------------------------
# Import bot-tools
# ---------------------------------------------------------------------------
_bot_tools_dir = os.path.join(_root_dir, "bot-tools")
if _bot_tools_dir not in sys.path:
    sys.path.insert(0, _bot_tools_dir)

# Wrap strip_appended_blocks to also handle <openwebui_usage> without
# modifying response_filter.py. The imported base handles ollama_usage
# and kjv_scripture; the wrapper adds openwebui_usage on top.
try:
    from response_filter import strip_appended_blocks as _strip_base

    def strip_appended_blocks(text: str) -> str:
        text = _strip_base(text)
        return re.sub(
            r"<openwebui_usage>.*?</openwebui_usage>",
            "", text, flags=re.DOTALL
        ).strip()

except ImportError:
    print("[WARN] bot-tools/response_filter.py not found — using inline fallback.")

    def strip_appended_blocks(text: str) -> str:
        return re.sub(
            r"<(?:ollama_usage|openwebui_usage|kjv_scripture)>.*?"
            r"</(?:ollama_usage|openwebui_usage|kjv_scripture)>",
            "", text, flags=re.DOTALL
        ).strip()

try:
    from bible_kjv_verse_lookup import scan_and_fetch_verses, CORPUS_AVAILABLE as _BIBLE_AVAILABLE
    if not _BIBLE_AVAILABLE:
        print("[WARN] Bible corpus not found in bot-tools/Bible-kjv-abbrev/ — verse lookup disabled.")
except ImportError:
    _BIBLE_AVAILABLE = False
    def scan_and_fetch_verses(_): return None
    print("[WARN] bot-tools/bible_kjv_verse_lookup.py not found — verse lookup disabled.")

# ---------------------------------------------------------------------------
# Configuration — shared vars reused directly, new vars prefixed OPENWEBUI_
# ---------------------------------------------------------------------------

# --- New vars (required) ---
BOT_TOKEN          : str   = os.getenv("TELEGRAM_BOT_TOKEN_OPENWEBUI", "")
OPENWEBUI_BASE_URL : str   = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:3000").rstrip("/")
OPENWEBUI_API_KEY  : str   = os.getenv("OPENWEBUI_API_KEY",  "")
OPENWEBUI_MODEL    : str   = os.getenv("OPENWEBUI_MODEL",    "llama3.2:3b")

# --- Reused vars (identical names to ollama_t.py) ---
SYSTEM_PROMPT     : str   = os.getenv("SYSTEM_PROMPT",         "You are a helpful assistant.")
MAX_OUTPUT_TOKENS : int   = int(os.getenv("MAX_OUTPUT_TOKENS",    "512"))
MAX_CONTEXT_MSG   : int   = int(os.getenv("MAX_CONTEXT_MESSAGES", "5"))
POLL_INTERVAL     : float = float(os.getenv("POLL_INTERVAL",      "1"))
LOG_DIR           : str   = os.getenv("LOG_DIR",                "logs")
WELCOME_MESSAGE_BASE   : str   = os.getenv(
        "WELCOME_MESSAGE_OPENWEBUI",
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
# Model swap slots — 12 slots, all accessed identically via Open WebUI API
# ---------------------------------------------------------------------------

OPENWEBUI_MODEL_SLOTS : dict = {}
_SLOT_RANGE = range(1, 13)  # slots 1–12

for _i in _SLOT_RANGE:
    _val = os.getenv(f"OPENWEBUI_MODEL_{_i}", "").strip()
    if _val:
        OPENWEBUI_MODEL_SLOTS[_i] = _val

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

if not BOT_TOKEN:
    raise SystemExit(
        "[ERROR] TELEGRAM_BOT_TOKEN_OPENWEBUI is not set.\n"
        "        Add it to your .env file and restart."
    )

if not OPENWEBUI_API_KEY:
    print("[WARN] OPENWEBUI_API_KEY is not set — requests will likely be rejected (401).")

# ---------------------------------------------------------------------------
# Per-chat active model store
# ---------------------------------------------------------------------------

_per_chat_model : dict = {}

def get_active_model(chat_id: int) -> str:
    return _per_chat_model.get(chat_id, OPENWEBUI_MODEL)

# ---------------------------------------------------------------------------
# Per-chat conversation context store
# ---------------------------------------------------------------------------

conversation_history : dict = {}

def get_or_init_context(chat_id: int) -> list:
    """
    Returns the conversation history for a given chat_id.
    Initializes a new context list with the system prompt if none exists.
    """
    if chat_id not in conversation_history:
        conversation_history[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return conversation_history[chat_id]


def trim_context(history: list, max_messages: int) -> list:
    """
    Caps conversation history to max_messages non-system entries.
    System prompt at index 0 is always preserved.
    Strips tagged appendage blocks from retained assistant messages
    so they are never re-submitted to the model.
    """
    system_entries = [m for m in history if m["role"] == "system"]
    non_system     = [m for m in history if m["role"] != "system"]

    if len(non_system) > max_messages:
        non_system = non_system[-max_messages:]

    for msg in non_system:
        if msg["role"] == "assistant":
            msg["content"] = strip_appended_blocks(msg["content"])

    return system_entries + non_system

# ---------------------------------------------------------------------------
# File logger — identical schema to ollama_t.py (Oracle-migration compatible)
# ---------------------------------------------------------------------------

def write_log(
    chat_id   : int,
    username  : str,
    role      : str,
    content   : str,
    model     : str   = None,
    tokens_in : int   = None,
    tokens_out: int   = None,
    speed_toks: float = None
) -> None:
    """
    Appends one JSONL record to logs/<chat_id>/YYYY-MM-DD.jsonl.
    Content is truncated to 200 chars. Logging failures never crash the poll loop.

    Schema (mirrors ollama_t.py for shared Oracle table compatibility):
      timestamp, chat_id, username, role, content (200-char preview),
      model, tokens_in, tokens_out, speed_toks
    """
    try:
        log_subdir = os.path.join(LOG_DIR, str(chat_id))
        os.makedirs(log_subdir, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = os.path.join(log_subdir, f"{date_str}.jsonl")

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

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as exc:
        print(f"[WARN] write_log() failed for chat_id={chat_id}: {type(exc).__name__}: {exc}")

# ===========================================================================
# Open WebUI API call
# ===========================================================================

def call_openwebui(messages: list, model: str = None) -> tuple:
    """
    Submits a conversation context to Open WebUI via its OpenAI-compatible
    /api/chat/completions endpoint using SSE streaming.

    Streaming distributes response bandwidth over the generation duration
    rather than delivering a single burst, mirroring stream=True in ollama_t.py.

    Usage fields sourced from:
      - chunk["usage"] in the final SSE chunk (if stream_options.include_usage=True
        is honoured by the Open WebUI version in use)
      - Wall-clock elapsed time for tokens/sec (always available regardless of
        whether the backend reports timing metadata)

    Reasoning/thinking content is captured from delta.reasoning_content
    (supported by DeepSeek-R1 and similar models routed through Open WebUI).

    :param messages: Conversation history as OpenAI message dicts
    :param model:    Model name override; falls back to OPENWEBUI_MODEL
    :return:         Tuple of (response_text: str, usage_meta: dict | None)
                     usage_meta keys: model, prompt_tokens, completion_tokens,
                                      finish_reason, elapsed_sec, tokens_per_sec
    """
    active_model = model or OPENWEBUI_MODEL
    url          = f"{OPENWEBUI_BASE_URL}/api/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
        "Content-Type" : "application/json",
    }

    payload = {
        "model"          : active_model,
        "messages"       : messages,
        "max_tokens"     : MAX_OUTPUT_TOKENS,
        "stream"         : True,
        "stream_options" : {"include_usage": True},  # final chunk carries usage if supported
    }

    # --- DEBUG: Print full context being submitted -----------------------
    print("\n" + "=" * 60)
    print(f"[DEBUG] call_openwebui() — submitting {len(messages)} message(s)")
    print(f"  model  : {active_model}")
    print(f"  url    : {url}")
    for i, msg in enumerate(messages):
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        has_tag = "<openwebui_usage>" in content or "<kjv_scripture>" in content
        tag_warn = "  *** TAG LEAK ***" if has_tag else ""
        print(f"  [{i}] {role} ({len(content)} chars){tag_warn}")
        print(f"       {content[:120].replace(chr(10), ' ')}{'...' if len(content) > 120 else ''}")
    print("=" * 60 + "\n")
    # ---------------------------------------------------------------------

    content_parts   = []
    reasoning_parts = []
    usage_data      = {}
    finish_reason   = None
    response_model  = active_model

    start_time = time.time()

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=300  # 5-minute hard ceiling for large cloud models
        )
        resp.raise_for_status()

        # SSE parsing — accumulate split chunks into a buffer to handle
        # cases where a data: line is delivered across multiple TCP segments.
        buffer = ""
        for raw_chunk in resp.iter_content(chunk_size=None):
            if not raw_chunk:
                continue
            buffer += raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # --- DEBUG: Print every raw SSE line as it arrives ----------
                #if line:
                #    print(f"[SSE RAW] {line}")
                # -------------------------------------------------------------

                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # --- DEBUG: Print parsed chunk dict -------------------------
                #print(f"[SSE CHUNK] {json.dumps(chunk, ensure_ascii=False)}")
                # -------------------------------------------------------------

                choices = chunk.get("choices", [])
                if choices:
                    delta         = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])

                # Usage arrives in the final chunk when stream_options honoured
                if chunk.get("usage"):
                    usage_data = chunk["usage"]
                if chunk.get("model"):
                    response_model = chunk["model"]

    except requests.exceptions.Timeout:
        print("[ERROR] call_openwebui(): request timed out after 300s")
        return "The request timed out. The model may be overloaded — please try again.", None

    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        print(f"[ERROR] call_openwebui(): HTTP {code} — {exc}")
        if code == 401:
            return "Authentication failed. Check OPENWEBUI_API_KEY in your .env.", None
        return f"API error (HTTP {code}). Please try again.", None

    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] call_openwebui(): {type(exc).__name__}: {exc}")
        return "A connection error occurred. Please try again.", None

    elapsed_sec   = time.time() - start_time
    llm_content   = "".join(content_parts)
    llm_reasoning = "".join(reasoning_parts) if reasoning_parts else None

    input_tokens  = usage_data.get("prompt_tokens",     0) or 0
    output_tokens = usage_data.get("completion_tokens", 0) or 0
    tokens_per_sec = (output_tokens / elapsed_sec) if elapsed_sec > 0 and output_tokens > 0 else None

    # --- DEBUG: Print response metadata ----------------------------------
    _speed_str = f"{tokens_per_sec:.2f} tok/s (wall clock)" if tokens_per_sec else "N/A"
    print("\n" + "-" * 60)
    print(f"[DEBUG] Open WebUI response fields:")
    print(f"  model             : {response_model}")
    print(f"  prompt_tokens     : {input_tokens}  (input tokens)")
    print(f"  completion_tokens : {output_tokens}  (output tokens)")
    print(f"  total             : {input_tokens + output_tokens}")
    print(f"  elapsed_sec       : {elapsed_sec:.3f}s")
    print(f"  finish_reason     : {finish_reason}")
    print(f"  computed speed    : {_speed_str}")
    print(f"  content length    : {len(llm_content)} chars")
    print(f"  reasoning length  : {len(llm_reasoning) if llm_reasoning else 0} chars")
    print("-" * 60 + "\n")
    # ---------------------------------------------------------------------

    usage_meta = {
        "model"            : response_model,
        "prompt_tokens"    : input_tokens,
        "completion_tokens": output_tokens,
        "finish_reason"    : finish_reason or "stop",
        "elapsed_sec"      : elapsed_sec,
        "tokens_per_sec"   : tokens_per_sec,
    }

    if llm_reasoning:
        thoughtful_response = (
            llm_content
            + "\n\n=============Thinking==============\n\n"
            + llm_reasoning
        )
        return thoughtful_response, usage_meta

    return llm_content, usage_meta

# ===========================================================================
# Usage summary formatting
# ===========================================================================

_LIGHTNING = "\u26A1"
_BULB      = "\U0001F4A1"
_SPEECH    = "\U0001F5E3"
_BOX       = "\U0001F4E6"
_FLAG      = "\U0001F3C1"
_MEMO      = "\U0001F4DD"

def format_usage_summary(usage_meta: dict) -> str:
    """
    Builds a formatted usage summary block from the usage_meta dict
    returned by call_openwebui(). Wrapped in <openwebui_usage> tags so
    strip_appended_blocks() (wrapped above) removes it from LLM context
    on the next poll cycle.

    Speed is computed from wall-clock elapsed time since Open WebUI does
    not expose nanosecond eval_duration metadata. Token counts require
    stream_options.include_usage=True to be honoured by the server;
    if not, counts will be 0 and only the speed line will carry meaning.

    :param usage_meta: Dict from call_openwebui(), or None on error
    :return:           Formatted summary string
    """
    if not usage_meta:
        return ""

    input_tokens  : int   = usage_meta.get("prompt_tokens",      0) or 0
    output_tokens : int   = usage_meta.get("completion_tokens",  0) or 0
    total_tokens  : int   = input_tokens + output_tokens
    finish_reason : str   = usage_meta.get("finish_reason",   "stop") or "stop"
    model_name    : str   = usage_meta.get("model", OPENWEBUI_MODEL) or OPENWEBUI_MODEL
    elapsed_sec   : float = usage_meta.get("elapsed_sec", 0.0) or 0.0
    tokens_per_sec: float = usage_meta.get("tokens_per_sec")

    speed_line = (
        f"{_LIGHTNING} Speed: {tokens_per_sec:.1f} tok/s  |  "
        f"Elapsed: {elapsed_sec:.1f}s (wall clock)"
        if tokens_per_sec
        else f"{_LIGHTNING} Elapsed: {elapsed_sec:.1f}s (wall clock)"
    )

    lines = [
        "<openwebui_usage>",
        f"{_LIGHTNING} Usage Summary",
        (
            f"{_BULB} Input: {input_tokens} | "
            f"{_SPEECH} Output: {output_tokens} | "
            f"{_BOX} Total: {total_tokens} tokens"
        ),
        speed_line,
        f"{_FLAG} Finish Reason: {finish_reason}",
        f"{_MEMO} Generated using {model_name} via Open WebUI",
        "</openwebui_usage>",
    ]

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Telegram command registry
# ---------------------------------------------------------------------------

COMMAND_REGISTRY : dict = {}

def register_command(command: str):
    def decorator(fn):
        COMMAND_REGISTRY[command.lower()] = fn
        return fn
    return decorator


@register_command("/start")
def handle_start(chat_id: int, username: str, user_text: str = "") -> str:
    """Resets conversation context and returns the welcome message."""
    conversation_history[chat_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print(f"[INFO] /start received from @{username} (chat_id={chat_id}) — context reset")
    active_model = get_active_model(chat_id)
    welcome      = WELCOME_MESSAGE_BASE + active_model + " via Open WebUI."
    name_part    = f" {username}" if username else ""
    return welcome.replace("{username}", name_part.strip())


@register_command("/help")
def handle_help(chat_id: int, username: str, user_text: str = "") -> str:
    return (
        "Available commands:\n"
        "/start       — Begin a new session and reset conversation history.\n"
        f"               Current context limit: {MAX_CONTEXT_MSG} messages.\n"
        "/help        — Show this message.\n"
        "/model  — Display the model selection menu.\n\n"
        "Send any message to chat with the bot."
    )


def _build_swap_menu() -> str:
    lines = ["Select a model by number:\n"]
    for i in _SLOT_RANGE:
        if i in OPENWEBUI_MODEL_SLOTS:
            lines.append(f"  {i}. {OPENWEBUI_MODEL_SLOTS[i]}")
    lines.append("\nUsage: /model <number>")
    lines.append("Example: /model 4")
    return "\n".join(lines)


@register_command("/model")
def handle_swap_model(chat_id: int, username: str, user_text: str = "") -> str:
    """
    No argument  — returns the numbered model selection menu.
    Valid slot    — swaps the active model; preserves conversation context.
    Invalid arg   — returns an error with the menu.
    """
    parts = user_text.strip().split()

    if len(parts) < 2:
        current = get_active_model(chat_id)
        return f"Current model: {current}\n\n" + _build_swap_menu()

    try:
        slot = int(parts[1])
    except ValueError:
        return f"'{parts[1]}' is not a valid slot number.\n\n" + _build_swap_menu()

    if slot not in OPENWEBUI_MODEL_SLOTS:
        configured = ", ".join(str(k) for k in sorted(OPENWEBUI_MODEL_SLOTS.keys()))
        return (
            f"Slot {slot} is not configured.\n"
            f"Available slots: {configured}\n\n"
            + _build_swap_menu()
        )

    previous_model            = get_active_model(chat_id)
    _per_chat_model[chat_id]  = OPENWEBUI_MODEL_SLOTS[slot]
    new_model                 = _per_chat_model[chat_id]

    print(
        f"[INFO] /model: @{username} (chat_id={chat_id}) "
        f"slot {slot} | {previous_model} -> {new_model}"
    )

    return (
        f"Model swapped to slot {slot}:\n"
        f"{new_model}\n\n"
        f"Conversation context is preserved.\n"
        f"Type /start to begin a fresh session with the new model."
    )

# ===========================================================================
# Main polling loop
# ===========================================================================

def run() -> None:
    """
    Starts the long-polling loop for the Open WebUI Telegram bot.

    Offset management mirrors ollama_t.py exactly:
      offset is set to update_id + 1 after each processed update so
      Telegram does not re-deliver acknowledged messages. Each serve-*.py
      process manages its own offset independently.

    Error recovery: unhandled exceptions are caught, printed, and followed
    by a POLL_INTERVAL sleep before the next cycle.
    """
    print(f"[INFO] Open WebUI bot starting up.")
    print(f"[INFO] Model    : {OPENWEBUI_MODEL}")
    print(f"[INFO] Base URL : {OPENWEBUI_BASE_URL}")
    #print(f"[INFO] Token    : ...{BOT_TOKEN[-8:]}")
    print(f"[INFO] Polling for messages...\n")

    offset : int = None  # None on first call — Telegram returns all pending updates

    while True:
        try:
            # Step 1: Poll Telegram for new updates
            updates = get_updates(bot_token=BOT_TOKEN, offset=offset)

            if not updates.get("ok"):
                print(
                    f"[WARN] getUpdates returned not-ok | "
                    f"error_code={updates.get('error_code')} | "
                    f"description={updates.get('description')}"
                )
                time.sleep(POLL_INTERVAL)
                continue

            # Step 2: Parse and deduplicate updates
            parsed_messages = parse_latest_messages(updates)

            for msg in parsed_messages:
                chat_id   : int = msg["chat_id"]
                user_text : str = msg["text"]
                username  : str = (
                    msg.get("username") or msg.get("first_name") or f"id:{chat_id}"
                )
                update_id : int = msg["update_id"]

                # Advance offset to acknowledge this update
                if offset is None or update_id >= offset:
                    offset = update_id + 1

                if not user_text.strip():
                    print(f"[DEBUG] Skipping empty message from @{username} (chat_id={chat_id})")
                    continue

                print(f"[INFO] Received from @{username} (chat_id={chat_id}): {user_text[:100]}")

                write_log(chat_id=chat_id, username=username, role="user", content=user_text)

                # Step 3: Command interception
                command_key = user_text.strip().lower().split()[0]
                if command_key in COMMAND_REGISTRY:
                    command_response = COMMAND_REGISTRY[command_key](chat_id, username, user_text)
                    send_message(bot_token=BOT_TOKEN, chat_id=chat_id, text=command_response)
                    print(f"[INFO] Command '{command_key}' handled for @{username} (chat_id={chat_id})")
                    write_log(chat_id=chat_id, username=username, role="command", content=user_text)
                    continue

                # Step 4: Build context and call the model
                history = get_or_init_context(chat_id)
                history.append({"role": "user", "content": user_text})

                active_model             = get_active_model(chat_id)
                llm_response, usage_meta = call_openwebui(history, model=active_model)

                print(f"[RAW] usage_meta={usage_meta}")

                # Step 5: Store raw content — no tagged blocks in context
                history.append({"role": "assistant", "content": llm_response})
                conversation_history[chat_id] = trim_context(history, MAX_CONTEXT_MSG)

                # Step 6: Verse lookup + usage summary
                verse_block   = scan_and_fetch_verses(llm_response) if _BIBLE_AVAILABLE else None
                usage_summary = format_usage_summary(usage_meta)

                appendages     = [b for b in [verse_block, usage_summary] if b]
                final_response = (
                    llm_response + "\n\n" + "\n\n".join(appendages)
                    if appendages
                    else llm_response
                )

                # DEBUG: Verify outbound message structure
                has_tag_in_final   = any(
                    t in final_response for t in ["<openwebui_usage>", "<kjv_scripture>"]
                )
                has_tag_in_history = any(
                    any(t in m.get("content", "") for t in ["<openwebui_usage>", "<kjv_scripture>"])
                    for m in conversation_history.get(chat_id, [])
                )
                print("\n" + "*" * 60)
                print(f"[DEBUG] Final outbound message for chat_id={chat_id}")
                print(f"  total length           : {len(final_response)} chars")
                print(f"  usage block in message : {has_tag_in_final}  (expected: True)")
                print(f"  usage block in context : {has_tag_in_history}  (expected: False)")
                print(f"  context message count  : {len(conversation_history.get(chat_id, []))}")
                print(f"  --- Message preview ---")
                print(f"  {final_response[:200].replace(chr(10), ' | ')}...")
                print("*" * 60 + "\n")

                # Step 7: Dispatch to Telegram
                results = send_long_message(
                    bot_token=BOT_TOKEN,
                    chat_id=chat_id,
                    text=final_response
                )

                all_ok = results and all(r.get("ok") for r in results)
                if all_ok:
                    sent_preview = final_response[:80].replace("\n", " ")
                    print(
                        f"[INFO] Sent to @{username} (chat_id={chat_id}) "
                        f"in {len(results)} chunk(s): {sent_preview}..."
                    )
                    write_log(
                        chat_id    = chat_id,
                        username   = username,
                        role       = "assistant",
                        content    = llm_response,
                        model      = usage_meta.get("model", OPENWEBUI_MODEL) if usage_meta else OPENWEBUI_MODEL,
                        tokens_in  = usage_meta.get("prompt_tokens")    if usage_meta else None,
                        tokens_out = usage_meta.get("completion_tokens") if usage_meta else None,
                        speed_toks = usage_meta.get("tokens_per_sec")   if usage_meta else None,
                    )
                else:
                    failed = next(
                        (r for r in results if not r.get("ok")),
                        results[-1] if results else {}
                    )
                    print(
                        f"[ERROR] sendMessage failed for chat_id={chat_id} | "
                        f"error_code={failed.get('error_code')} | "
                        f"description={failed.get('description')}"
                    )

        except KeyboardInterrupt:
            print("\n[INFO] Open WebUI bot stopped by keyboard interrupt.")
            sys.exit(0)

        except Exception as exc:
            print(f"[ERROR] Unhandled exception in poll loop: {type(exc).__name__}: {exc}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()