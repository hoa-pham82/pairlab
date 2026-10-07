"""A/B routing: send a stable share of keys (pairs, sessions) to a challenger."""

from __future__ import annotations

import hashlib

CHAMPION = "champion"
CHALLENGER = "challenger"
_BUCKETS = 10_000


def assign_variant(key: str, challenger_share: float) -> str:
    """Return the variant for a key; the same key always gets the same answer.

    The key is hashed into one of 10,000 buckets; the lowest
    ``challenger_share`` of buckets go to the challenger.
    """
    if not 0.0 <= challenger_share <= 1.0:
        raise ValueError("challenger_share must be between 0 and 1")
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % _BUCKETS
    return CHALLENGER if bucket < challenger_share * _BUCKETS else CHAMPION
