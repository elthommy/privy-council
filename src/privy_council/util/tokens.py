"""Rough token estimation (chars/4): good enough for budgets and caps."""

from __future__ import annotations

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def tokens_to_chars(tokens: int) -> int:
    """Convert a token budget to an approximate character budget."""
    return tokens * CHARS_PER_TOKEN
