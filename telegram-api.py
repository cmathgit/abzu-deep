"""
telegram-api.py

Telegram Bot API interface module — stateless utility layer.

Provides reusable functions for polling updates and dispatching messages
via the Telegram Bot HTTP API. Intended to be imported by serve-telbot/*.py
scripts; not executed directly.

    from telegram_api import get_updates, send_message, parse_latest_messages

Each serve-*.py script supplies its own bot_token, enabling simultaneous
multi-bot operation across isolated processes without shared state.

Telegram Bot API reference: https://core.telegram.org/bots/api
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level configuration derived from environment variables
# ---------------------------------------------------------------------------

TELEGRAM_API_BASE_URL: str = "https://api.telegram.org"
MAX_INPUT_LENGTH: int      = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
POLL_TIMEOUT: int          = int(os.getenv("POLL_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Internal URL builder
# ---------------------------------------------------------------------------

def _build_bot_url(bot_token: str, method: str) -> str:
    """
    Constructs the full Telegram Bot API endpoint URL.

    :param bot_token: Telegram bot token (e.g., '7123456789:AAF...')
    :param method:    Telegram API method name (e.g., 'getUpdates', 'sendMessage')
    :return:          Fully qualified URL string
    """
    return f"{TELEGRAM_API_BASE_URL}/bot{bot_token}/{method}"


# ---------------------------------------------------------------------------
# Text filter functions — REGEX placeholders
# All filter functions accept a string and return a string.
# Filtering logic is commented out pending specification.
# ---------------------------------------------------------------------------

def filter_incoming_text(text: str) -> str:
    """
    Applies REGEX filters to incoming user message text prior to LLM processing.

    This function is called inside parse_latest_messages() on every inbound
    message. Extend the commented block below as filtering requirements are
    defined. The unmodified input is returned until patterns are activated.

    :param text: Raw incoming message text from a Telegram update payload
    :return:     Filtered text string
    """
    # --- INCOMING TEXT REGEX FILTERS -------------------------------------------
    # Uncomment and extend patterns as requirements are defined.
    #
    # Strip URLs from user input before passing to the LLM:
    #   text = re.sub(r'https?://\S+', '[link]', text)
    #
    # Redact phone numbers:
    #   text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[phone]', text)
    #
    # Block specific phrases and substitute a sentinel value:
    #   BLOCKED = [r'phrase one', r'phrase two']
    #   for pattern in BLOCKED:
    #       if re.search(pattern, text, re.IGNORECASE):
    #           return "[Message blocked by content filter]"
    #
    # Normalize excessive whitespace:
    #   text = re.sub(r'\s{2,}', ' ', text).strip()
    # ---------------------------------------------------------------------------
    return text


def filter_outgoing_text(text: str) -> str:
    """
    Applies REGEX filters to LLM-generated response text before transmission
    to Telegram via send_message().

    This function is called inside send_message() on every outbound message.
    Extend the commented block below as filtering requirements are defined.
    The unmodified input is returned until patterns are activated.

    :param text: LLM-generated response string
    :return:     Filtered text string
    """
    # --- OUTGOING TEXT REGEX FILTERS -------------------------------------------
    # Uncomment and extend patterns as requirements are defined.
    #
    # Strip common LLM disclaimer prefixes:
    #   text = re.sub(r'(?i)^as an ai( language model)?,?\s*', '', text).strip()
    #
    # Remove any URLs injected by the LLM:
    #   text = re.sub(r'https?://\S+', '[link removed]', text)
    #
    # Replace prohibited terms with a redaction placeholder:
    #   PROHIBITED = {'term_a': '[redacted]', 'term_b': '[redacted]'}
    #   for term, replacement in PROHIBITED.items():
    #       text = re.sub(rf'(?i)\b{re.escape(term)}\b', replacement, text)
    #
    # Normalize excessive line breaks in LLM output:
    #   text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # ---------------------------------------------------------------------------
    return text


def detect_keywords(text: str) -> dict:
    """
    Scans input text for predefined keyword categories using REGEX.

    Returns a dict of boolean category flags for downstream routing logic
    in serve-*.py scripts. For example, a True value on "bible_verse_request"
    can trigger a lookup function rather than a generative LLM call.

    All detection patterns are commented out pending specification.

    :param text: Input text to scan (typically the processed user message)
    :return:     Dict mapping category name strings to boolean detection results
    """
    results: dict = {
        "bible_verse_request": False,
        "greeting":            False,
        "question":            False,
        # Extend with additional detection categories as needed
    }

    # --- KEYWORD / PHRASE DETECTION REGEX --------------------------------------
    # Uncomment and extend patterns as detection requirements are defined.
    #
    # Detect requests for Bible verses — triggers verse lookup in serve-*.py:
    #   bible_pattern = r'(?i)\b(verse|scripture|bible|proverbs?|psalms?|genesis|matthew|john)\b'
    #   results["bible_verse_request"] = bool(re.search(bible_pattern, text))
    #
    # Detect conversational greetings:
    #   greeting_pattern = r'(?i)^\s*(hi|hello|hey|good\s+(morning|evening|afternoon))\b'
    #   results["greeting"] = bool(re.search(greeting_pattern, text))
    #
    # Detect questions by terminal punctuation:
    #   results["question"] = bool(text.strip().endswith('?'))
    # ---------------------------------------------------------------------------

    return results


# ---------------------------------------------------------------------------
# Input length management
# ---------------------------------------------------------------------------

def truncate_input_text(text: str, max_length: int = None) -> str:
    """
    Truncates incoming user text to the configured maximum character length.

    Prevents excessively long messages from consuming LLM context budget.
    Called inside parse_latest_messages() after filter_incoming_text().

    :param text:       Input text string
    :param max_length: Override character limit; defaults to MAX_INPUT_LENGTH from .env
    :return:           Truncated string, or the original string if within limit
    """
    limit: int = max_length if max_length is not None else MAX_INPUT_LENGTH
    if len(text) <= limit:
        return text
    return text[:limit]


# ---------------------------------------------------------------------------
# Core Telegram API: get_updates
# ---------------------------------------------------------------------------

def get_updates(
    bot_token: str,
    offset: int = None,
    timeout: int = None,
    limit: int = 100,
    allowed_updates: list = None
) -> dict:
    """
    Polls the Telegram Bot API getUpdates endpoint using long polling.

    OFFSET MANAGEMENT (critical):
        After processing a batch of updates, the calling serve-*.py script
        must track the highest update_id from the result and pass:
            offset = last_update_id + 1
        on the next call. This acknowledges all prior updates and prevents
        Telegram from re-delivering them. Each serve-*.py process maintains
        its own offset independently since bot tokens are distinct.

    LONG POLLING:
        Setting timeout > 0 instructs Telegram's server to hold the connection
        open for up to `timeout` seconds before returning an empty result.
        The HTTP client timeout is set to `timeout + 10` to prevent premature
        disconnection by the requests library.

    :param bot_token:       Telegram bot token for this specific bot instance
    :param offset:          Update ID offset; pass last_update_id + 1 after each cycle
    :param timeout:         Long-poll hold duration in seconds; defaults to POLL_TIMEOUT
    :param limit:           Maximum updates to retrieve per call (1–100)
    :param allowed_updates: Update types to receive; defaults to ["message"] only
    :return:                Parsed JSON response dict from Telegram API,
                            or error dict with keys: ok, error_code, description
    """
    if timeout is None:
        timeout = POLL_TIMEOUT

    if allowed_updates is None:
        allowed_updates = ["message"]

    url: str = _build_bot_url(bot_token, "getUpdates")

    params: dict = {
        "timeout":         timeout,
        "limit":           limit,
        "allowed_updates": allowed_updates,
    }

    if offset is not None:
        params["offset"] = offset

    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout + 10  # Must exceed server-side hold duration
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        return {
            "ok":          False,
            "error_code":  408,
            "description": "Request timed out during long poll cycle"
        }

    except requests.exceptions.ConnectionError as exc:
        return {
            "ok":          False,
            "error_code":  503,
            "description": f"Connection error: {exc}"
        }

    except requests.exceptions.HTTPError as exc:
        return {
            "ok":          False,
            "error_code":  exc.response.status_code,
            "description": str(exc)
        }

    except requests.exceptions.RequestException as exc:
        return {
            "ok":          False,
            "error_code":  0,
            "description": f"Unexpected request error: {exc}"
        }


# ---------------------------------------------------------------------------
# Core Telegram API: send_message
# ---------------------------------------------------------------------------

def send_message(
    bot_token: str,
    chat_id: int,
    text: str,
    parse_mode: str = None,
    max_retries: int = 3
) -> dict:
    """
    Sends a message to a Telegram chat via the sendMessage endpoint.

    Uses HTTP POST with a JSON request body. This is intentional: URL query
    parameters (as used in the GET-based Postman example) impose a practical
    length ceiling of ~2048 characters on many servers and proxies, which LLM
    responses can easily exceed. The POST body carries no such restriction.

    filter_outgoing_text() is applied to `text` before transmission.

    RATE LIMITING:
        Telegram enforces 30 messages/second globally and 1 message/second
        per chat_id. HTTP 429 responses include a retry_after integer.
        This function sleeps for that duration and retries up to max_retries
        times before returning a failure dict.

    :param bot_token:   Telegram bot token for this specific bot instance
    :param chat_id:     Target chat ID (sourced from the inbound update payload)
    :param text:        Message text to transmit (LLM-generated response)
    :param parse_mode:  Optional Telegram formatting: 'HTML', 'Markdown', 'MarkdownV2'
                        Note: 'MarkdownV2' requires escaping many characters in LLM output.
                        'HTML' is the more forgiving choice for generative text.
    :param max_retries: Retry attempts on rate-limit (429) responses
    :return:            Parsed JSON response dict from Telegram API,
                        or error dict with keys: ok, error_code, description
    """
    filtered_text: str = filter_outgoing_text(text)

    url: str = _build_bot_url(bot_token, "sendMessage")

    payload: dict = {
        "chat_id": chat_id,
        "text":    filtered_text,
    }

    if parse_mode:
        payload["parse_mode"] = parse_mode

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=15)

            # Telegram rate limit: sleep for retry_after seconds then retry
            if response.status_code == 429:
                retry_after: int = (
                    response.json()
                    .get("parameters", {})
                    .get("retry_after", 5)
                )
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return {
                    "ok":          False,
                    "error_code":  408,
                    "description": "sendMessage request timed out after all retries"
                }
            time.sleep(2 ** attempt)  # Exponential backoff on timeout

        except requests.exceptions.HTTPError as exc:
            return {
                "ok":          False,
                "error_code":  exc.response.status_code,
                "description": str(exc)
            }

        except requests.exceptions.RequestException as exc:
            return {
                "ok":          False,
                "error_code":  0,
                "description": f"Unexpected request error: {exc}"
            }

    return {
        "ok":          False,
        "error_code":  429,
        "description": "sendMessage failed: max retries exceeded due to rate limiting"
    }


def send_long_message(
    bot_token   : str,
    chat_id     : int,
    text        : str,
    parse_mode  : str  = None,
    max_length  : int  = 4096
) -> list:
    """
    Splits text exceeding Telegram's 4096-character limit and sends each
    chunk as a sequential message.

    Splitting strategy (priority order):
        1. Split on double newline (paragraph boundary) — cleanest break
        2. Split on single newline (line boundary)
        3. Hard split at max_length — last resort for pathological content

    Tagged appendage blocks (<ollama_usage>, <kjv_scripture>) are detected
    and always attached to the FINAL chunk so they appear at the end of the
    full response rather than mid-conversation.

    :param bot_token:  Telegram bot token
    :param chat_id:    Target chat ID
    :param text:       Full message text, potentially exceeding 4096 chars
    :param parse_mode: Optional Telegram parse mode: 'HTML', 'Markdown', etc.
    :param max_length: Character cap per chunk; default is Telegram's hard limit
    :return:           List of send_message() response dicts, one per chunk
    """
    import re as _re

    if len(text) <= max_length:
        return [send_message(bot_token, chat_id, text, parse_mode)]

    # ---------------------------------------------------------------------------
    # Separate tagged appendage blocks from the main body before chunking.
    # Appendages are re-attached to the last chunk only.
    # ---------------------------------------------------------------------------
    TAG_PATTERN = r"(<(?:ollama_usage|kjv_scripture)>.*?</(?:ollama_usage|kjv_scripture)>)"
    parts       = _re.split(TAG_PATTERN, text, flags=_re.DOTALL)

    body_parts  = [p for p in parts if not _re.match(TAG_PATTERN, p, _re.DOTALL)]
    tag_parts   = [p for p in parts if     _re.match(TAG_PATTERN, p, _re.DOTALL)]

    body        = "".join(body_parts).strip()
    appendages  = "\n\n".join(tag_parts)

    # ---------------------------------------------------------------------------
    # Chunk the body text
    # ---------------------------------------------------------------------------
    chunks      = []
    remaining   = body

    while len(remaining) > max_length:
        # Attempt 1: split at last double newline within max_length window
        split_pos = remaining.rfind("\n\n", 0, max_length)

        # Attempt 2: split at last single newline within window
        if split_pos == -1:
            split_pos = remaining.rfind("\n", 0, max_length)

        # Attempt 3: hard split at max_length
        if split_pos == -1:
            split_pos = max_length

        chunks.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()

    if remaining:
        chunks.append(remaining)

    # Attach appendage blocks to the last chunk.
    # If appending overflows max_length, chunk the appendages separately
    # using the same splitting strategy and extend the chunks list.
    if appendages and chunks:
        combined_last = chunks[-1] + "\n\n" + appendages
        if len(combined_last) <= max_length:
            # Fits cleanly — merge into last body chunk as before
            chunks[-1] = combined_last
        else:
            # Appendages overflow the last chunk — chunk them independently
            # and append those chunks after the body chunks.
            appendage_chunks = []
            remaining = appendages
            while len(remaining) > max_length:
                split_pos = remaining.rfind("\n\n", 0, max_length)
                if split_pos == -1:
                    split_pos = remaining.rfind("\n", 0, max_length)
                if split_pos == -1:
                    split_pos = max_length
                appendage_chunks.append(remaining[:split_pos].strip())
                remaining = remaining[split_pos:].strip()
            if remaining:
                appendage_chunks.append(remaining)
            chunks.extend(appendage_chunks)

    # ---------------------------------------------------------------------------
    # Dispatch each chunk sequentially
    # ---------------------------------------------------------------------------
    results     = []
    total       = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue

        # Prefix multi-part messages so the user knows there is more coming
        if total > 1:
            chunk = f"[{i}/{total}]\n{chunk}"

        result = send_message(bot_token, chat_id, chunk, parse_mode)
        results.append(result)

        # Abort remaining chunks if one fails
        if not result.get("ok"):
            print(
                f"[ERROR] send_long_message(): chunk {i}/{total} failed for "
                f"chat_id={chat_id} | error_code={result.get('error_code')} | "
                f"{result.get('description')}"
            )
            break

        # Brief pause between chunks to respect per-chat rate limit (1 msg/sec)
        if i < total:
            import time
            time.sleep(1.1)

    return results

# ---------------------------------------------------------------------------
# Update parsing utility
# ---------------------------------------------------------------------------

def parse_latest_messages(updates: dict) -> list:
    """
    Processes a raw getUpdates response payload and returns the single latest
    message per unique chat_id, sorted by message date descending.

    Processing pipeline applied per message:
        1. Skip updates with no 'message' field (edited messages, channel posts, etc.)
        2. Skip messages from other bots (is_bot=True in the 'from' field)
        3. Apply filter_incoming_text() to the message text
        4. Apply truncate_input_text() to enforce MAX_INPUT_LENGTH
        5. Run detect_keywords() and attach results to the output dict

    The calling serve-*.py script is responsible for:
        - Tracking the highest update_id from the returned list
        - Passing offset = highest_update_id + 1 on the next get_updates() call
        - Maintaining per-chat_id conversation context in its own scope

    :param updates: Raw dict returned by get_updates()
    :return:        List of parsed message dicts, one entry per unique chat_id.
                    Each dict contains the following keys:
                        update_id   (int)        — use as offset base: update_id + 1
                        message_id  (int)
                        chat_id     (int)        — pass to send_message() as chat_id
                        username    (str | None)
                        first_name  (str | None)
                        last_name   (str | None)
                        text        (str)        — filtered and truncated user text
                        date        (int)        — Unix timestamp
                        is_bot      (bool)
                        keywords    (dict)       — output of detect_keywords()
    """
    if not updates.get("ok") or not updates.get("result"):
        return []

    # Sort all result entries descending by message date to isolate the latest
    # message per chat_id in a single forward pass
    sorted_updates: list = sorted(
        updates["result"],
        key=lambda u: u.get("message", {}).get("date", 0),
        reverse=True
    )

    seen_chat_ids: set = set()
    latest_messages: list = []

    for update in sorted_updates:
        message: dict = update.get("message")
        if not message:
            continue  # Skip non-message update types

        # Exclude messages from other bots
        if message.get("from", {}).get("is_bot", False):
            continue

        chat_id: int = message.get("chat", {}).get("id")
        if chat_id in seen_chat_ids:
            continue  # Already captured the latest message for this chat_id

        seen_chat_ids.add(chat_id)

        raw_text: str = message.get("text", "")

        # Apply incoming content pipeline: filter → truncate
        processed_text: str = filter_incoming_text(raw_text)
        processed_text = truncate_input_text(processed_text)

        latest_messages.append({
            "update_id":  update.get("update_id"),
            "message_id": message.get("message_id"),
            "chat_id":    chat_id,
            "username":   message.get("chat", {}).get("username"),
            "first_name": message.get("from", {}).get("first_name"),
            "last_name":  message.get("from", {}).get("last_name"),
            "text":       processed_text,
            "date":       message.get("date"),
            "is_bot":     message.get("from", {}).get("is_bot", False),
            "keywords":   detect_keywords(processed_text),
        })

    return latest_messages