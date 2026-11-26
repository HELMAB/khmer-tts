"""Pronunciation lexicon: English words → Khmer phonetic spelling, applied only at synthesis time.

VoxPocket is a Khmer model and reads Latin letters poorly. Text files and captions keep English in
Latin script (the user's rule); `speakable()` swaps each known English word for how it is said in
Khmer just before the text reaches the model. Entries live in lexicon.txt next to this file.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

LEXICON_FILE = Path(__file__).with_name("lexicon.txt")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*[A-Za-z0-9]|[A-Za-z]")


@lru_cache(maxsize=1)
def _load() -> tuple[re.Pattern | None, dict[str, str]]:
    entries: dict[str, str] = {}
    for raw in LEXICON_FILE.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or "\t" not in raw:
            continue
        word, khmer = raw.split("\t", 1)
        entries[word.strip().lower()] = khmer.strip()
    if not entries:
        return None, {}
    keys = sorted(entries, key=len, reverse=True)  # "background music" before "background"
    # (?<![A-Za-z]) / (?![A-Za-z]): whole words, but Khmer letters may touch either side
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in keys)
                         + r")(?![A-Za-z])", re.IGNORECASE)
    return pattern, entries


def speakable(text: str) -> str:
    """The line as the model should read it: known English words replaced, spaced off from Khmer."""
    pattern, entries = _load()
    if pattern is None:
        return text
    out = pattern.sub(lambda m: f" {entries[re.sub(r'\s+', ' ', m.group(1).lower())]} ", text)
    out = re.sub(r" +([។!?៕,])", r"\1", out)
    return re.sub(r" {2,}", " ", out).strip()


def unknown_words(text: str) -> list[str]:
    """Latin words still left after `speakable` (no lexicon entry yet)."""
    return LATIN_WORD.findall(speakable(text))
