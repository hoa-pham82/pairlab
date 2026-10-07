"""Stable A/B variant assignment for the agent LLM."""

from __future__ import annotations

import hashlib

CHAMPION = "champion"
CHALLENGER = "challenger"


def assign_llm_variant(question: str, challenger_share: float = 0.2) -> str:
    """Assign a question to champion or challenger deterministically.

    The same question always lands in the same bucket, so retries are
    consistent and metrics are comparable between variants.
    """
    bucket = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], "big") % 10_000
    return CHALLENGER if bucket < challenger_share * 10_000 else CHAMPION
