"""Tests for the LLM A/B evaluation script."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from services.agent import ab_eval
from services.agent.ab_eval import (
    QUESTIONS,
    Score,
    score,
    summarise,
    truth_values,
    ungrounded_numbers,
)

SIGNAL = {
    "pair_id": "KO__PEP",
    "prob": 0.688019,
    "take_trade": True,
    "threshold": 0.5,
    "model_version": "meta_label-v7",
    "features": {"zscore": -0.7221, "hedge_ratio": 0.3393},
}
REGIME = {"pair_id": "KO__PEP", "regime": "stable", "coint_pvalue": 0.009517, "bars_used": 315}
TRUTH = truth_values(SIGNAL) + truth_values(REGIME)


def _call(name: str, pair: str = "KO__PEP", failed: bool = False) -> dict:
    return {"name": name, "arguments": {"pair_id": pair}, "failed": failed}


GOOD = {
    "answer": "Tradeable: p-value 0.0095, probability 0.69.",
    "tool_calls": [_call("ask_features_analyst"), _call("ask_regime_analyst")],
    "finished": True,
    "blocked": False,
    "output_tokens": 40,
}


def test_good_answer_passes_every_check():
    result = score("KO__PEP", GOOD, 3.0, TRUTH)
    assert result.passed and result.grounded and result.verdict_format
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
        ({"answer": "Tradeable: 0.69, 100"}, "grounded"),
    ],
)
def test_each_broken_rule_fails_the_answer(change, failed_check):
    result = score("KO__PEP", {**GOOD, **change}, 1.0, TRUTH)
    assert not result.passed
    assert not getattr(result, failed_check)


def test_truth_values_flatten_nested_numbers_and_skip_booleans():
    assert truth_values(SIGNAL) == [0.688019, 0.5, 7.0, -0.7221, 0.3393]
    assert truth_values(None) == []


# Grounding partitions: exact, rounded at the written precision (boundary: one
# unit past half a step), percentage form, negative, integer, numbers inside
# tickers or versions are not numbers, invented values.
@pytest.mark.parametrize(
    ("answer", "missing"),
    [
        ("probability 0.688019", []),
        ("probability 0.69", []),
        ("probability 0.7", []),
        ("probability 0.68", ["0.68"]),
        ("probability 68.8%", []),
        ("probability 69%", []),
        ("z-score -0.72", []),
        ("z-score 0.72", ["0.72"]),
        ("315 bars", []),
        ("model meta_label-v7", []),
        ("Tradeable: 0.69, 100", ["100"]),
        ("p-value 0.0095 and PSI 0.25", ["0.25"]),
        ("no numbers here.", []),
    ],
)
def test_ungrounded_numbers(answer, missing):
    assert ungrounded_numbers(answer, TRUTH) == missing


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
    assert score("KO__PEP", {**GOOD, "answer": answer}, 1.0, TRUTH).verdict_format is ok


def test_summarise_rates_and_medians():
    scores = [
        score("KO__PEP", GOOD, 2.0, TRUTH),
        score("KO__PEP", {**GOOD, "answer": "plain 42"}, 4.0, TRUTH),
        score("KO__PEP", {**GOOD, "finished": False, "output_tokens": 10}, 9.0, TRUTH),
    ]
    assert summarise(scores) == {
        "questions": 3,
        "pass_rate": 0.333,
        "grounded_rate": 0.667,
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
    asked: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/signal":
            return httpx.Response(200, json=SIGNAL)
        if request.url.path == "/regime":
            return httpx.Response(200, json=REGIME)
        asked.append((body["question"], body["variant"]))
        pair = next(p for p in ab_eval.PAIRS if p in body["question"])
        calls = [_call("ask_features_analyst", pair), _call("ask_regime_analyst", pair)]
        answer = "Tradeable: 0.69, 100" if body["variant"] == "challenger" else "It is 0.69."
        return httpx.Response(200, json={**GOOD, "tool_calls": calls, "answer": answer})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        ab_eval.httpx,
        "AsyncClient",
        lambda: real_client(transport=httpx.MockTransport(handler)),
    )
    report = asyncio.run(ab_eval.run("http://agent", "http://sig", "http://reg", QUESTIONS[:2]))

    assert asked == [(q, v) for q in QUESTIONS[:2] for v in ("champion", "challenger")]
    champion, challenger = report["variants"]["champion"], report["variants"]["challenger"]
    assert (champion["pass_rate"], champion["verdict_format_rate"]) == (1.0, 0.0)
    assert (challenger["pass_rate"], challenger["verdict_format_rate"]) == (0.0, 1.0)
    assert report["answers"][1]["ungrounded"] == ["100"]


def test_score_is_a_plain_value():
    expected = Score("KO__PEP", True, True, True, True, True, True, 1.0, 40)
    assert score("KO__PEP", GOOD, 1.0, TRUTH) == expected
