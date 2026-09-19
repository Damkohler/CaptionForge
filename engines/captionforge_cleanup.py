"""Shared forbidden-phrase matching helpers for CaptionForge cleanup paths."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def phrase_boundary_pattern(
    phrase: str,
    *,
    case_insensitive: bool = True,
) -> re.Pattern[str] | None:
    """Compile a phrase matcher that respects token boundaries.

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

    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(pattern, flags=flags)


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



def replace_phrases(
    text: str,
    replacement_rules: Iterable[tuple[str, str]],
    *,
    case_insensitive: bool = True,
) -> str:
    """Apply replacement rules only at whole-word/phrase boundaries."""
    result = str(text or "")
    for old, new in replacement_rules:
        pattern = phrase_boundary_pattern(old, case_insensitive=case_insensitive)
        if pattern is not None:
            result = pattern.sub(str(new or ""), result)
    return result


def normalize_forbidden_phrases(value: Any) -> list[str]:
    """Normalize UI or plan values into an ordered forbidden-phrase list."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.splitlines()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    return [text for item in items if (text := str(item or "").strip())]


def normalize_replace_pairs(value: Any) -> list[tuple[str, str]]:
    """Normalize ``old=>new`` UI text or serialized plan replacement pairs."""
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.splitlines()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]

    pairs: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            old, new = item.get("old", ""), item.get("new", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            old, new = item[0], item[1]
        else:
            line = str(item or "").strip()
            if not line or line.startswith("#") or "=>" not in line:
                continue
            old, new = line.split("=>", 1)
        old_text = str(old or "").strip()
        if old_text:
            pairs.append((old_text, str(new or "").strip()))
    return pairs


def resolve_cleanup_settings(
    pipeline_plan: Any,
    standalone_forbidden_phrases: Any,
    standalone_replace_pairs: Any,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Resolve Planner-owned cleanup values, preserving standalone node use."""
    plan = pipeline_plan if isinstance(pipeline_plan, dict) else {}
    cleanup = plan.get("cleanup") if isinstance(plan.get("cleanup"), dict) else None
    if cleanup is not None:
        forbidden_value = cleanup.get("forbidden_phrases", [])
        replace_value = cleanup.get("replace_pairs", [])
    else:
        forbidden_value = standalone_forbidden_phrases
        replace_value = standalone_replace_pairs
    return normalize_forbidden_phrases(forbidden_value), normalize_replace_pairs(replace_value)


def apply_cleanup_contract(
    text: str,
    forbidden_phrases: Iterable[str],
    replace_pairs: Iterable[tuple[str, str]],
) -> str:
    """Apply the shared boundary-safe cleanup contract and repair separators."""
    result = replace_phrases(text, replace_pairs)
    result = remove_forbidden_phrases(result, forbidden_phrases)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r",\s*,+", ",", result)
    result = re.sub(r"([.;:!?])(?:\s*[,.;:!?])+", r"\1", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip(" ,")
