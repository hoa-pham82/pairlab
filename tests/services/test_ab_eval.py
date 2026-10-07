"""Tests for the LLM A/B evaluation script."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from services.agent import ab_eval
from services.agent.ab_eval import QUESTIONS, Score, score, summarise


def _call(name: str, pair: str = "KO__PEP", failed: bool = False) -> dict:
    return {"name": name, "arguments": {"pair_id": pair}, "failed": failed}


GOOD = {
    "answer": "Tradeable: p-value 0.01, probability 0.7.",
    "tool_calls": [_call("ask_features_analyst"), _call("ask_regime_analyst")],
    "finished": True,
    "blocked": False,
    "output_tokens": 40,
}


def test_good_answer_passes_every_check():
    result = score("KO__PEP", GOOD, 3.0)
    assert result.passed and result.verdict_format
    assert (result.seconds, result.output_tokens) == (3.0, 40)


# One partition per check that can fail; each breaks exactly one rule.
@pytest.mark.parametrize(
    ("change", "failed_check"),
    [
        ({"finished": False}, "finished"),
        ({"blocked": True}, "finished"),
        ({"tool_calls": [_call("ask_regime_analyst")]}, "called_all_tools"),
        ({"tool_calls": []}, "called_all_tools"),
        (
            {"tool_calls": [_call("ask_features_analyst"), _call("ask_regime_analyst", "GS__MS")]},
            "right_pair",
        ),
        (
            {
                "tool_calls": [
                    _call("ask_features_analyst"),
                    _call("ask_regime_analyst", failed=True),
                ]
            },
            "no_failed_tools",
        ),
    ],
)
def test_each_broken_rule_fails_the_answer(change, failed_check):
    result = score("KO__PEP", {**GOOD, **change}, 1.0)
    assert not result.passed
    assert not getattr(result, failed_check)


# Verdict format: both prefixes, leading whitespace allowed, anything else fails.
@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("Tradeable: yes", True),
        ("Not tradeable: broken regime", True),
        ("  Tradeable: yes", True),
        ("tradeable: yes", False),
        ("The pair is tradeable.", False),
        ("", False),
    ],
)
def test_verdict_format(answer, ok):
    assert score("KO__PEP", {**GOOD, "answer": answer}, 1.0).verdict_format is ok


def test_summarise_rates_and_medians():
    scores = [
        score("KO__PEP", GOOD, 2.0),
        score("KO__PEP", {**GOOD, "answer": "plain"}, 4.0),
        score("KO__PEP", {**GOOD, "finished": False, "output_tokens": 10}, 9.0),
    ]
    assert summarise(scores) == {
        "questions": 3,
        "pass_rate": 0.667,
        "verdict_format_rate": 0.667,
        "right_pair_rate": 1.0,
        "latency_seconds_median": 4.0,
        "output_tokens_median": 40,
    }


def test_summarise_rejects_empty_input():
    with pytest.raises(ValueError):
        summarise([])


def test_questions_cover_every_pair_twice():
    assert len(QUESTIONS) == 2 * len(ab_eval.PAIRS)
    for pair in ab_eval.PAIRS:
        assert sum(pair in q for q in QUESTIONS) == 2


def test_run_asks_every_question_on_both_variants(monkeypatch):
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((body["question"], body["variant"]))
        pair = next(p for p in ab_eval.PAIRS if p in body["question"])
        calls = [_call("ask_features_analyst", pair), _call("ask_regime_analyst", pair)]
        answer = "Tradeable: ok" if body["variant"] == "challenger" else "It is fine."
        return httpx.Response(200, json={**GOOD, "tool_calls": calls, "answer": answer})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        ab_eval.httpx,
        "AsyncClient",
        lambda: real_client(transport=httpx.MockTransport(handler)),
    )
    report = asyncio.run(ab_eval.run("http://agent", QUESTIONS[:2]))

    assert seen == [(q, v) for q in QUESTIONS[:2] for v in ("champion", "challenger")]
    assert report["variants"]["champion"]["pass_rate"] == 1.0
    assert report["variants"]["champion"]["verdict_format_rate"] == 0.0
    assert report["variants"]["challenger"]["verdict_format_rate"] == 1.0
    assert len(report["answers"]) == 4


def test_score_is_a_plain_value():
    assert score("KO__PEP", GOOD, 1.0) == Score("KO__PEP", True, True, True, True, True, 1.0, 40)
