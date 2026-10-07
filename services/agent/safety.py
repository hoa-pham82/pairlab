"""Input guard: detect personal data in a prompt before it reaches the model."""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"(?<![\w.])\+?\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)"),
}


def find_pii(text: str) -> list[str]:
    """Return the kinds of personal data found in ``text`` (empty if none).

    Each match is removed before the next kind is checked, so a card number
    is not also reported as a phone number.
    """
    found: list[str] = []
    for kind, pattern in _PATTERNS.items():
        text, count = pattern.subn(" ", text)
        if count:
            found.append(kind)
    return found
