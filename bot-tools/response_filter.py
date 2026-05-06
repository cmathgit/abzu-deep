"""
bot-tools/response_filter.py

Generic tagged-block filter for outbound bot message appendages.

Any block wrapped in <tag>...</tag> that was appended to an outbound message
(e.g. <ollama_usage>, <kjv_scripture>) must be stripped from the assistant
entry before it re-enters the LLM context window. This module provides a
single stripping function that handles all registered tags in one pass.

To register a new appendage type in the future:
    1. Define its open/close tag string in its source module
    2. Add the tag name (without angle brackets) to STRIPPABLE_TAGS below
    The stripping logic requires no further changes.
"""

import re

# ---------------------------------------------------------------------------
# Registry of all tags that must be stripped from LLM context.
#
# Add new tag name strings here (without angle brackets) as new tool blocks
# and usage summaries are introduced across serve-*.py scripts.
# ---------------------------------------------------------------------------

STRIPPABLE_TAGS: list = [
    "ollama_usage",
    "kjv_scripture",
    # "gemini_usage",   # register when serve-telbot/gemini.py is implemented
    # "cohere_usage",   # register when serve-telbot/cohere.py is implemented
    # "mistral_usage",  # register when serve-telbot/mistral.py is implemented
]


def strip_appended_blocks(text: str) -> str:
    """
    Removes all registered tagged blocks from a string in a single pass.

    Iterates over STRIPPABLE_TAGS and applies a non-greedy DOTALL REGEX for
    each tag. Non-greedy matching ensures that two adjacent blocks of the same
    type are each removed independently without consuming content between them.

    Called inside trim_context() in each serve-*.py script on every assistant
    message entry before it is submitted to the LLM on the next poll cycle.

    :param text: String that may contain one or more tagged appendage blocks
    :return:     Cleaned string with all registered blocks removed, whitespace trimmed
    """
    cleaned = text

    for tag in STRIPPABLE_TAGS:
        pattern = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    return cleaned.strip()