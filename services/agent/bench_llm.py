"""Benchmark an OpenAI-compatible LLM server: TTFT, latency, and tokens per second.

Usage:
    uv run python -m services.agent.bench_llm --base-url http://localhost:8085 \
        --requests 12 --concurrency 2 --label baseline --out docs/pngs/llm_bench_baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

PROMPT = (
    "You are a regime analyst. A drift check for pair KO__PEP returned: regime stable, "
    "cointegration p-value 0.0095, PSI 0.12. Explain in two sentences what this means."
)


@dataclass(frozen=True)
class Sample:
    """Timing of one streamed generation."""

    ttft_seconds: float
    total_seconds: float
    output_tokens: int

    @property
    def tokens_per_second(self) -> float:
        generating = self.total_seconds - self.ttft_seconds
        return self.output_tokens / generating if generating > 0 else 0.0


async def probe(client: httpx.AsyncClient, base_url: str, max_tokens: int = 96) -> Sample:
    """Stream one completion and time the first token and the whole response."""
    body = {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token: float | None = None
    chunks = 0
    output_tokens = 0
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=300
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: ") :])
            usage = event.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens", output_tokens))
            for choice in event.get("choices") or []:
                if (choice.get("delta") or {}).get("content"):
                    chunks += 1
                    if first_token is None:
                        first_token = time.perf_counter()
    finished = time.perf_counter()
    return Sample(
        ttft_seconds=(first_token or finished) - started,
        total_seconds=finished - started,
        output_tokens=output_tokens or chunks,
    )


def summarise(samples: list[Sample], wall_seconds: float, label: str, concurrency: int) -> dict:
    """Aggregate samples into the numbers reported in the docs."""
    if not samples:
        raise ValueError("no samples")
    totals = sorted(s.total_seconds for s in samples)
    p95_index = max(0, -(-95 * len(totals) // 100) - 1)
    return {
        "label": label,
        "requests": len(samples),
        "concurrency": concurrency,
        "ttft_seconds_median": round(statistics.median(s.ttft_seconds for s in samples), 3),
        "latency_seconds_median": round(statistics.median(totals), 3),
        "latency_seconds_p95": round(totals[p95_index], 3),
        "tokens_per_second_median": round(
            statistics.median(s.tokens_per_second for s in samples), 1
        ),
        "throughput_tokens_per_second": round(
            sum(s.output_tokens for s in samples) / wall_seconds, 1
        ),
        "requests_per_minute": round(60 * len(samples) / wall_seconds, 1),
    }


async def run(base_url: str, requests: int, concurrency: int, label: str) -> dict:
    """Fire ``requests`` probes with at most ``concurrency`` in flight."""
    gate = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient) -> Sample:
        async with gate:
            return await probe(client, base_url)

    async with httpx.AsyncClient() as client:
        await probe(client, base_url, max_tokens=8)  # warm-up, not measured
        started = time.perf_counter()
        samples = await asyncio.gather(*(one(client) for _ in range(requests)))
        wall = time.perf_counter() - started
    return {
        **summarise(list(samples), wall, label, concurrency),
        "samples": [asdict(s) for s in samples],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8085")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = asyncio.run(run(args.base_url, args.requests, args.concurrency, args.label))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
