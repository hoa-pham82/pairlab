"""Compare the agent API's champion and challenger on the same questions.

Each question runs once per variant (temperature 0, fixed seed), scored with
deterministic checks on the tool calls, a grounding check of every number in
the answer against the signal and regime APIs, plus latency and token counts.

Usage:
    uv run python -m services.agent.ab_eval --base-url http://localhost:8003 \
        --signal-url http://localhost:8000 --regime-url http://localhost:8001 \
        --out docs/pngs/phase4_llm_ab.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from services.ab import CHALLENGER, CHAMPION
from services.agent.analysts import VERDICT_PREFIXES

COORDINATOR_TOOLS = frozenset({"ask_features_analyst", "ask_regime_analyst"})
PAIRS = ("KO__PEP", "XOM__CVX", "GS__MS")
QUESTIONS = tuple(
    template.format(pair=pair)
    for pair in PAIRS
    for template in ("Is {pair} still tradeable?", "Should we keep watching {pair}?")
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.(\d+))?")


@dataclass(frozen=True)
class Score:
    """Deterministic checks and cost of one answer."""

    pair_id: str
    finished: bool
    called_all_tools: bool
    right_pair: bool
    no_failed_tools: bool
    grounded: bool
    verdict_format: bool
    seconds: float
    output_tokens: int

    @property
    def passed(self) -> bool:
        return (
            self.finished
            and self.called_all_tools
            and self.right_pair
            and self.no_failed_tools
            and self.grounded
        )


def truth_values(payload: object) -> list[float]:
    """Every number in an API response, including numbers inside strings like ``v7``."""
    if isinstance(payload, bool):
        return []
    if isinstance(payload, int | float):
        return [float(payload)]
    if isinstance(payload, str):
        return [float(match.group()) for match in re.finditer(r"\d+(?:\.\d+)?", payload)]
    if isinstance(payload, dict):
        payload = list(payload.values())
    if isinstance(payload, list):
        return [value for item in payload for value in truth_values(item)]
    return []


def ungrounded_numbers(answer: str, truth: list[float]) -> list[str]:
    """Numbers in ``answer`` that match no true value at the precision written.

    A number also matches a true value written as a percentage (0.688 as 68.8).
    """
    missing = []
    for match in _NUMBER.finditer(answer):
        written = float(match.group())
        tolerance = 0.5 * 10 ** -len(match.group(1) or "") + 1e-9
        if not any(
            abs(written - t) <= tolerance or abs(written - 100 * t) <= tolerance for t in truth
        ):
            missing.append(match.group())
    return missing


def score(pair_id: str, body: dict, seconds: float, truth: list[float]) -> Score:
    """Check that the coordinator asked both analysts about the right pair and invented nothing."""
    calls = body.get("tool_calls") or []
    answer = body.get("answer", "")
    return Score(
        pair_id=pair_id,
        finished=bool(body.get("finished")) and not body.get("blocked"),
        called_all_tools={call["name"] for call in calls} >= COORDINATOR_TOOLS,
        right_pair=bool(calls)
        and all(call["arguments"].get("pair_id") == pair_id for call in calls),
        no_failed_tools=not any(call["failed"] for call in calls),
        grounded=not ungrounded_numbers(answer, truth),
        verdict_format=answer.lstrip().startswith(VERDICT_PREFIXES),
        seconds=seconds,
        output_tokens=int(body.get("output_tokens", 0)),
    )


def summarise(scores: list[Score]) -> dict:
    """Pass rates and median cost for one variant."""
    if not scores:
        raise ValueError("no scores")
    n = len(scores)

    def rate(field: str) -> float:
        return round(sum(getattr(s, field) for s in scores) / n, 3)

    return {
        "questions": n,
        "pass_rate": rate("passed"),
        "grounded_rate": rate("grounded"),
        "verdict_format_rate": rate("verdict_format"),
        "right_pair_rate": rate("right_pair"),
        "latency_seconds_median": round(statistics.median(s.seconds for s in scores), 2),
        "output_tokens_median": statistics.median(s.output_tokens for s in scores),
    }


async def fetch_truth(
    client: httpx.AsyncClient, signal_url: str, regime_url: str, pair_id: str
) -> list[float]:
    """The numbers the agents' tools can see for a pair."""
    values: list[float] = []
    for url in (f"{signal_url}/signal", f"{regime_url}/regime"):
        response = await client.post(url, json={"pair_id": pair_id}, timeout=60)
        response.raise_for_status()
        values += truth_values(response.json())
    return values


async def ask(client: httpx.AsyncClient, base_url: str, question: str, variant: str) -> tuple:
    """Run one question on one variant; return the response body and wall time."""
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/ask", json={"question": question, "variant": variant}, timeout=300
    )
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


async def run(
    base_url: str,
    signal_url: str,
    regime_url: str,
    questions: tuple[str, ...] = QUESTIONS,
) -> dict:
    """Ask every question to both variants, one at a time so latencies are comparable."""
    results: dict[str, list[Score]] = {CHAMPION: [], CHALLENGER: []}
    answers = []
    async with httpx.AsyncClient() as client:
        truth = {pair: await fetch_truth(client, signal_url, regime_url, pair) for pair in PAIRS}
        for question in questions:
            pair_id = next(pair for pair in PAIRS if pair in question)
            for variant in (CHAMPION, CHALLENGER):
                body, seconds = await ask(client, base_url, question, variant)
                results[variant].append(score(pair_id, body, seconds, truth[pair_id]))
                answers.append(
                    {
                        "variant": variant,
                        "question": question,
                        "answer": body["answer"],
                        "ungrounded": ungrounded_numbers(body["answer"], truth[pair_id]),
                    }
                )
    return {
        "variants": {variant: summarise(scores) for variant, scores in results.items()},
        "answers": answers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("--signal-url", default="http://localhost:8000")
    parser.add_argument("--regime-url", default="http://localhost:8001")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = asyncio.run(run(args.base_url, args.signal_url, args.regime_url))
    print(json.dumps(report["variants"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
