"""
bot-tools/bible_verse_lookup.py

KJV Bible verse lookup tool using the local Bible-kjv-abbrev corpus.

Detects scripture references in LLM response text via a REGEX pattern built
dynamically from abbreviations.json and Books.json, then fetches verbatim KJV
verse text from the individual book JSON files in the cloned corpus directory.

No pip dependencies required — fully offline after cloning the corpus.

Setup:
    cd bot-tools
    git clone https://github.com/cmathgit/Bible-kjv-abbrev.git Bible-kjv-abbrev

Filename resolution rule:
    Books.json canonical name  →  strip all spaces  →  + ".json"
    "1 Samuel"                 →  "1Samuel"          →  "1Samuel.json"
    "Song of Solomon"          →  "SongofSolomon"    →  "SongofSolomon.json"
"""

import os
import re
import json

# ---------------------------------------------------------------------------
# Corpus path — resolved relative to this file so imports work regardless
# of the working directory the serve script is launched from.
# ---------------------------------------------------------------------------

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_CORPUS_DIR = os.path.join(_THIS_DIR, "Bible-kjv-abbrev")

# Block tag — must match the tag registered in response_filter.py
KJV_BLOCK_TAG = "kjv_scripture"

# Unicode constants — defined outside f-strings to avoid escape parse errors
_BOOK_EMOJI = "\U0001F4D6"  # 📖
_RULE_CHAR  = "\u2500"      # ─

# ---------------------------------------------------------------------------
# Corpus loader — executes once at module import
# ---------------------------------------------------------------------------

def _load_corpus_index() -> tuple:
    """
    Loads Books.json and abbreviations.json from the cloned corpus directory.

    :return: Tuple of (abbrev_to_canonical, canonical_to_filename) dicts.
             Both are None if the corpus directory is not found.

             abbrev_to_canonical  : {"jn": "John", "1 sam": "1 Samuel", ...}
             canonical_to_filename: {"John": "John.json", "1 Samuel": "1Samuel.json", ...}
    """
    books_path  = os.path.join(_CORPUS_DIR, "Books.json")
    abbrev_path = os.path.join(_CORPUS_DIR, "abbreviations.json")

    if not os.path.isfile(books_path) or not os.path.isfile(abbrev_path):
        print(
            f"[WARN] bible_verse_lookup: Corpus not found at: {_CORPUS_DIR}\n"
            f"       Run from bot-tools/:\n"
            f"       git clone https://github.com/cmathgit/Bible-kjv-abbrev.git Bible-kjv-abbrev"
        )
        return None, None

    try:
        with open(books_path, "r", encoding="utf-8") as f:
            books: list = json.load(f)

        with open(abbrev_path, "r", encoding="utf-8") as f:
            abbrev_to_canonical: dict = json.load(f)

        # Build filename map: strip all spaces from canonical name
        # "1 Samuel" → "1Samuel.json"
        # "Song of Solomon" → "SongofSolomon.json"
        canonical_to_filename: dict = {
            book: book.replace(" ", "") + ".json"
            for book in books
        }

        print(
            f"[INFO] bible_verse_lookup: Corpus loaded — "
            f"{len(books)} books, {len(abbrev_to_canonical)} abbreviations"
        )
        return abbrev_to_canonical, canonical_to_filename

    except Exception as exc:
        print(f"[WARN] bible_verse_lookup: Failed to load corpus index: {exc}")
        return None, None


_ABBREV_TO_CANONICAL, _CANONICAL_TO_FILENAME = _load_corpus_index()

# Public flag — checked by ollama_t.py to guard the scan call
CORPUS_AVAILABLE: bool = _ABBREV_TO_CANONICAL is not None


# ---------------------------------------------------------------------------
# REGEX pattern builder — constructed once from the loaded index
# ---------------------------------------------------------------------------

def _build_reference_pattern() -> re.Pattern | None:
    """
    Builds a compiled REGEX that matches scripture references in free text.

    The book identifier alternation is assembled from:
        - All canonical names from Books.json  (e.g. "1 Samuel", "Song of Solomon")
        - All keys from abbreviations.json      (e.g. "1 sam", "jn", "gen.")

    Identifiers are sorted longest-first to prevent shorter patterns from
    consuming the prefix of a longer match (e.g. "jn" must not shadow "john").

    Pattern anatomy:
        (?<![a-zA-Z0-9])           not preceded by alphanumeric (word boundary)
        (book_identifier)          book name or abbreviation — captured as group 1
        [\s.]*                     optional whitespace / trailing period
        (\d+)                      chapter number — captured as group 2
        :                          chapter:verse separator
        (\d+)                      start verse — captured as group 3
        (?:-(\d+))?                optional end verse for ranges — group 4
        (?![a-zA-Z0-9])            not followed by alphanumeric (word boundary)

    Limitation: very short abbreviations ("is", "am", "re") may produce false
    positives in prose without a chapter:verse pattern directly following them.
    The strict `\d+:\d+` suffix provides the primary false-positive guard.
    """
    if not CORPUS_AVAILABLE:
        return None

    all_ids    = set(_ABBREV_TO_CANONICAL.keys()) | set(_CANONICAL_TO_FILENAME.keys())
    sorted_ids = sorted(all_ids, key=len, reverse=True)
    escaped    = [re.escape(identifier) for identifier in sorted_ids]
    book_group = "|".join(escaped)

    pattern = (
        rf"(?<![a-zA-Z0-9])"
        rf"({book_group})"
        rf"[\s.]*"
        rf"(\d+)"
        rf":"
        rf"(\d+)"
        rf"(?:-(\d+))?"
        rf"(?![a-zA-Z0-9])"
    )

    return re.compile(pattern, re.IGNORECASE)


_REFERENCE_PATTERN: re.Pattern | None = _build_reference_pattern()


# ---------------------------------------------------------------------------
# Book name resolver
# ---------------------------------------------------------------------------

def _resolve_canonical_name(matched_book: str) -> str | None:
    """
    Resolves a REGEX-matched book string to its canonical Books.json name.

    Resolution order:
        1. Case-insensitive direct match against canonical names
        2. Lowercase lookup in abbreviations index

    :param matched_book: Raw book string captured from the REGEX match
    :return:             Canonical name string, e.g. "1 Samuel", or None
    """
    if not CORPUS_AVAILABLE:
        return None

    normalized = matched_book.strip().lower()

    for canonical in _CANONICAL_TO_FILENAME:
        if canonical.lower() == normalized:
            return canonical

    return _ABBREV_TO_CANONICAL.get(normalized)


# ---------------------------------------------------------------------------
# Verse fetcher
# ---------------------------------------------------------------------------

def _fetch_verses(
    canonical_name: str,
    chapter       : int,
    start_verse   : int,
    end_verse     : int
) -> str | None:
    """
    Loads the book JSON file and extracts the requested verse range verbatim.

    File path:    {CORPUS_DIR}/{canonical_name_no_spaces}.json
    JSON schema:  {"book": str,
                   "chapters": [{"chapter": str,
                                 "verses": [{"verse": str, "text": str}]}]}

    :param canonical_name: Canonical book name, e.g. "1 Samuel"
    :param chapter:        Chapter number (1-indexed integer)
    :param start_verse:    First verse to retrieve (inclusive)
    :param end_verse:      Last verse to retrieve (inclusive); equals start_verse for single
    :return:               Verse lines as a single string, or None on failure
    """
    filename = _CANONICAL_TO_FILENAME.get(canonical_name)
    if not filename:
        print(f"[WARN] bible_verse_lookup: No filename mapping for '{canonical_name}'")
        return None

    filepath = os.path.join(_CORPUS_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"[WARN] bible_verse_lookup: File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            book_data: dict = json.load(f)

        # Locate target chapter by matching the "chapter" field value
        target_chapter = None
        for ch in book_data.get("chapters", []):
            if int(ch.get("chapter", 0)) == chapter:
                target_chapter = ch
                break

        if target_chapter is None:
            print(
                f"[WARN] bible_verse_lookup: "
                f"Chapter {chapter} not found in {canonical_name}"
            )
            return None

        lines = []
        for v in target_chapter.get("verses", []):
            v_num = int(v.get("verse", 0))
            if start_verse <= v_num <= end_verse:
                lines.append(f"{v_num} {v.get('text', '').strip()}")

        return "\n".join(lines) if lines else None

    except Exception as exc:
        print(f"[WARN] bible_verse_lookup: Error reading {filepath}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def scan_and_fetch_verses(text: str) -> str | None:
    """
    Scans text for scripture references and returns a formatted <kjv_scripture> block.

    Full pipeline:
        REGEX scan (built from abbreviations.json + Books.json)
          → _resolve_canonical_name()   map matched string → canonical name
            → _CANONICAL_TO_FILENAME    canonical name → filename (spaces stripped)
              → _fetch_verses()         load JSON, slice chapter/verses
                → format and wrap in <kjv_scripture> block

    Deduplicates identical references within a single response (e.g. if the
    LLM cites John 3:16 twice, only one verse block is appended).

    Returns None when:
        - Corpus was not found at startup (CORPUS_AVAILABLE is False)
        - No scripture references detected in text
        - All detected references fail verse resolution

    The <kjv_scripture>...</kjv_scripture> wrapper allows
    response_filter.strip_appended_blocks() to locate and remove the block
    from LLM context before the next poll cycle.

    :param text: LLM response content string to scan
    :return:     Formatted <kjv_scripture> block string, or None
    """
    if not CORPUS_AVAILABLE or _REFERENCE_PATTERN is None:
        return None

    verse_blocks = []
    seen_refs    = set()

    for match in _REFERENCE_PATTERN.finditer(text):
        raw_book    = match.group(1)
        chapter     = int(match.group(2))
        start_verse = int(match.group(3))
        end_verse   = int(match.group(4)) if match.group(4) else start_verse

        canonical = _resolve_canonical_name(raw_book)
        if not canonical:
            print(f"[WARN] bible_verse_lookup: Unresolved book name '{raw_book}'")
            continue

        # Deduplicate by (book, chapter, start, end)
        ref_key = (canonical, chapter, start_verse, end_verse)
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)

        ref_label = (
            f"{canonical} {chapter}:{start_verse}"
            if start_verse == end_verse
            else f"{canonical} {chapter}:{start_verse}-{end_verse}"
        )

        verse_text = _fetch_verses(canonical, chapter, start_verse, end_verse)
        if verse_text:
            verse_blocks.append(f"{_BOOK_EMOJI} {ref_label} (KJV)\n{verse_text}")
            print(f"[INFO] bible_verse_lookup: Fetched {ref_label}")

    if not verse_blocks:
        return None

    separator = f"\n{_RULE_CHAR * 21}\n"
    body      = separator.join(verse_blocks)

    return f"<{KJV_BLOCK_TAG}>\n{body}\n</{KJV_BLOCK_TAG}>"