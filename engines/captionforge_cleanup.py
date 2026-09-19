"""Shared forbidden-phrase matching helpers for CaptionForge cleanup paths."""

from __future__ import annotations

import re
from collections.abc import Iterable


def phrase_boundary_pattern(phrase: str) -> re.Pattern[str] | None:
    """Compile a case-insensitive phrase matcher that respects token boundaries.

    Boundary checks are added only when the corresponding phrase edge is a
    word character. This preserves literal punctuation in configured phrases
    while preventing tokens such as old from matching inside holding or bold.
    """
    text = str(phrase or "").strip()
    if not text:
        return None

    pattern = re.escape(text)
    if re.match(r"\w", text[0], flags=re.UNICODE):
        pattern = r"(?<!\w)" + pattern
    if re.match(r"\w", text[-1], flags=re.UNICODE):
        pattern = pattern + r"(?!\w)"

    return re.compile(pattern, flags=re.IGNORECASE)


def contains_forbidden_phrase(text: str, forbidden_phrases: Iterable[str]) -> bool:
    """Return True when any configured forbidden phrase matches at boundaries."""
    haystack = str(text or "")
    for phrase in forbidden_phrases:
        pattern = phrase_boundary_pattern(phrase)
        if pattern is not None and pattern.search(haystack):
            return True
    return False


def remove_forbidden_phrases(text: str, forbidden_phrases: Iterable[str]) -> str:
    """Remove configured forbidden phrases without corrupting containing words."""
    result = str(text or "")
    for phrase in forbidden_phrases:
        pattern = phrase_boundary_pattern(phrase)
        if pattern is not None:
            result = pattern.sub("", result)
    return result
