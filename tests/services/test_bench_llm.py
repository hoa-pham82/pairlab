"""Tests for the LLM benchmark probe and its summary statistics."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from services.agent.bench_llm import Sample, probe, run, summarise


def _sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _stream(with_usage: bool = True) -> bytes:
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "The"}}]},
        {"choices": [{"delta": {"content": " pair"}}]},
    ]
    if with_usage:
        events.append({"choices": [], "usage": {"completion_tokens": 7}})
    return _sse(*events)


def _client(body: bytes, seen: dict | None = None, status: int = 200) -> httpx.AsyncClient:
    def handler(request):
        if seen is not None:
            seen.update(json.loads(request.content))
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_probe_reads_usage_and_times_the_stream():
    seen: dict = {}
    sample = asyncio.run(probe(_client(_stream(), seen), "http://llm"))
    assert sample.output_tokens == 7
    assert 0 <= sample.ttft_seconds <= sample.total_seconds
    assert (seen["stream"], seen["temperature"], seen["seed"]) == (True, 0.0, 42)


def test_probe_counts_chunks_when_the_server_sends_no_usage():
    assert asyncio.run(probe(_client(_stream(with_usage=False)), "http://llm")).output_tokens == 2


def test_probe_raises_on_server_error():
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(probe(_client(b"", status=500), "http://llm"))


def test_tokens_per_second():
    assert Sample(1.0, 3.0, 40).tokens_per_second == 20.0
    assert Sample(2.0, 2.0, 40).tokens_per_second == 0.0


def test_summarise_known_values():
    samples = [Sample(0.5, float(total), 20) for total in range(1, 21)]
    report = summarise(samples, wall_seconds=100.0, label="x", concurrency=2)
    assert report["requests"] == 20
    assert report["ttft_seconds_median"] == 0.5
    assert report["latency_seconds_median"] == 10.5
    assert report["latency_seconds_p95"] == 19.0
    assert report["throughput_tokens_per_second"] == 4.0
    assert report["requests_per_minute"] == 12.0


# Boundary: p95 of a single sample is that sample; of an empty list is an error.
def test_summarise_boundaries():
    assert summarise([Sample(0.1, 2.0, 5)], 2.0, "x", 1)["latency_seconds_p95"] == 2.0
    with pytest.raises(ValueError):
        summarise([], 1.0, "x", 1)


def test_run_measures_every_request_after_a_warm_up(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=_stream(), headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "services.agent.bench_llm.httpx.AsyncClient",
        lambda: real(transport=httpx.MockTransport(handler)),
    )
    report = asyncio.run(run("http://llm", requests=5, concurrency=2, label="t"))
    assert (report["requests"], len(report["samples"]), calls["n"]) == (5, 5, 6)
