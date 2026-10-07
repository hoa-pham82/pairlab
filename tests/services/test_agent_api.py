"""Tests for the agent chat API."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from services.agent.agent import AgentAnswer, ToolAgent, ToolCall, ToolSpec
from services.agent.app import build_default_app, create_app, outcome
from services.agent.llm import ChatResult
from services.agent.telemetry import AgentTelemetry
from services.common import HttpMetrics

REGIME = {"regime": "stable", "psi": 0.05}


class FakeLLM:
    """Asks for check_regime on the first turn of each run, then answers."""

    def __init__(self, error: Exception | None = None, model: str = "fake"):
        self.error = error
        self.model = model

    async def chat(self, messages, tools):
        if self.error:
            raise self.error
        if messages[-1]["role"] == "user":
            call = {
                "id": "c",
                "function": {"name": "check_regime", "arguments": '{"pair_id": "KO__PEP"}'},
            }
            return ChatResult(
                {"role": "assistant", "tool_calls": [call]}, self.model, 10, 4, 0.1, 0.05
            )
        return ChatResult(
            {"role": "assistant", "content": f"KO__PEP is stable ({self.model})."},
            self.model,
            15,
            6,
            0.1,
        )


def _registry(llm, telemetry: AgentTelemetry) -> dict[str, ToolAgent]:
    async def check_regime(pair_id: str) -> dict:
        return REGIME

    spec = ToolSpec("check_regime", "drift check", check_regime)
    return {
        name: ToolAgent(name, "prompt", llm, [spec], telemetry)
        for name in ("coordinator", "regime_analyst", "features_analyst")
    }


def _client(
    llm=None, with_agents: bool = True, with_challenger: bool = False, share: float = 0.5
) -> TestClient:
    metrics = HttpMetrics("agent_api")
    telemetry = AgentTelemetry(metrics.registry)
    agents = _registry(llm or FakeLLM(), telemetry) if with_agents else {}
    challenger = _registry(FakeLLM(model="fake-b"), telemetry) if with_challenger else None
    return TestClient(create_app(agents, metrics, challenger=challenger, challenger_share=share))


def test_ask_returns_answer_and_tool_trace():
    body = _client().post("/ask", json={"question": "Is KO__PEP still tradeable?"}).json()
    assert body == {
        "agent": "coordinator",
        "variant": "champion",
        "answer": "KO__PEP is stable (fake).",
        "tool_calls": [
            {"name": "check_regime", "arguments": {"pair_id": "KO__PEP"}, "failed": False}
        ],
        "steps": 2,
        "input_tokens": 25,
        "output_tokens": 10,
        "blocked": False,
        "finished": True,
    }


def test_agent_can_be_chosen():
    body = _client().post("/ask", json={"question": "q", "agent": "regime_analyst"}).json()
    assert body["agent"] == "regime_analyst"


# Equivalence partitions and boundaries for the request: question length 0 /
# 1 / 2000 / 2001, unknown agent, wrong type, missing field.
@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"question": "q"}, 200),
        ({"question": "q" * 2000}, 200),
        ({"question": ""}, 422),
        ({"question": "q" * 2001}, 422),
        ({"question": "q", "agent": "trader"}, 422),
        ({"question": 5}, 422),
        ({}, 422),
        ({"question": "q", "variant": "champion"}, 200),
        ({"question": "q", "variant": "control"}, 422),
        ({"question": "q", "session_id": "s" * 128}, 200),
        ({"question": "q", "session_id": "s" * 129}, 422),
    ],
)
def test_request_validation(payload, code):
    assert _client().post("/ask", json=payload).status_code == code


def test_prompt_with_personal_data_is_blocked_with_200():
    body = _client().post("/ask", json={"question": "mail hoa@example.com about KO__PEP"}).json()
    assert body["blocked"] is True
    assert body["tool_calls"] == []


def test_llm_server_failure_is_502():
    client = _client(FakeLLM(httpx.ConnectError("refused")))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "ConnectError" in response.json()["detail"]


def test_health_and_readiness():
    ready, empty = _client(), _client(with_agents=False)
    assert ready.get("/healthz").json() == {"status": "ok"}
    assert ready.get("/readyz").status_code == 200
    assert empty.get("/readyz").status_code == 503
    assert empty.post("/ask", json={"question": "q"}).status_code == 503


def test_metrics_expose_http_llm_agent_and_tool_series():
    client = _client()
    client.post("/ask", json={"question": "Is KO__PEP ok?"})
    client.post("/ask", json={"question": "card 4111 1111 1111 1111"})
    text = client.get("/metrics").text
    for expected in (
        'agent_api_requests_total{route="/ask",status="200"} 2.0',
        'agent_calls_total{agent="coordinator"} 2.0',
        'agent_tool_calls_total{tool="check_regime"} 1.0',
        'llm_tokens_total{kind="input",model="fake",variant="champion"} 25.0',
        'llm_tokens_total{kind="output",model="fake",variant="champion"} 10.0',
        'llm_round_trip_seconds_count{model="fake",variant="champion"} 2.0',
        'llm_time_to_first_token_seconds_count{model="fake",variant="champion"} 1.0',
        'agent_pii_blocked_total{kind="card_number"} 1.0',
    ):
        assert expected in text


class TestABRouting:
    SESSIONS = [f"session-{i}" for i in range(200)]

    # Share boundaries: 0 (all champion), 1 (all challenger).
    @pytest.mark.parametrize(("share", "variant"), [(0.0, "champion"), (1.0, "challenger")])
    def test_share_boundaries(self, share, variant):
        client = _client(with_challenger=True, share=share)
        for session in self.SESSIONS[:20]:
            body = client.post("/ask", json={"question": "q", "session_id": session}).json()
            assert body["variant"] == variant

    def test_same_session_always_gets_the_same_variant_and_model(self):
        client = _client(with_challenger=True)
        bodies = [
            client.post("/ask", json={"question": f"q{i}", "session_id": "hoa"}).json()
            for i in range(5)
        ]
        assert len({(b["variant"], b["answer"]) for b in bodies}) == 1

    def test_half_share_splits_sessions_between_both_variants(self):
        client = _client(with_challenger=True)
        variants = [
            client.post("/ask", json={"question": "q", "session_id": s}).json()["variant"]
            for s in self.SESSIONS
        ]
        assert 0.35 < variants.count("challenger") / len(variants) < 0.65

    def test_question_is_the_routing_key_without_a_session(self):
        client = _client(with_challenger=True)
        first = client.post("/ask", json={"question": "Is GS__MS ok?"}).json()["variant"]
        again = client.post("/ask", json={"question": "Is GS__MS ok?"}).json()["variant"]
        assert first == again

    def test_forced_variant_overrides_the_hash(self):
        client = _client(with_challenger=True, share=0.0)
        body = client.post("/ask", json={"question": "q", "variant": "challenger"}).json()
        assert body["variant"] == "challenger"
        assert body["answer"] == "KO__PEP is stable (fake-b)."

    def test_without_a_challenger_everything_is_champion(self):
        client = _client(share=1.0)
        assert client.post("/ask", json={"question": "q"}).json()["variant"] == "champion"

    def test_forcing_a_missing_challenger_is_503(self):
        response = _client().post("/ask", json={"question": "q", "variant": "challenger"})
        assert response.status_code == 503
        assert "challenger" in response.json()["detail"]

    @pytest.mark.parametrize("share", [-0.01, 1.01])
    def test_share_outside_0_1_is_rejected(self, share):
        with pytest.raises(ValueError):
            _client(share=share)

    def test_metrics_split_by_variant(self):
        client = _client(with_challenger=True)
        client.post("/ask", json={"question": "q", "variant": "champion"})
        client.post("/ask", json={"question": "q", "variant": "challenger"})
        client.post("/ask", json={"question": "mail a@b.co", "variant": "challenger"})
        text = client.get("/metrics").text
        for expected in (
            'agent_ab_requests_total{agent="coordinator",outcome="answered",variant="champion"} 1.0',
            'agent_ab_requests_total{agent="coordinator",outcome="answered",variant="challenger"} 1.0',
            'agent_ab_requests_total{agent="coordinator",outcome="blocked",variant="challenger"} 1.0',
            'agent_ab_seconds_count{agent="coordinator",variant="challenger"} 2.0',
            'llm_tokens_total{kind="output",model="fake-b",variant="challenger"} 10.0',
        ):
            assert expected in text


def _call(failed: bool) -> ToolCall:
    return ToolCall("check_regime", {}, {"error": "x"} if failed else {})


# One partition per outcome; "blocked" and "unfinished" win over a failed tool.
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (AgentAnswer("a", [_call(False)], 2), "answered"),
        (AgentAnswer("a", [], 1), "answered"),
        (AgentAnswer("a", [_call(False), _call(True)], 3), "tool_failed"),
        (AgentAnswer("a", [_call(True)], 4, finished=False), "unfinished"),
        (AgentAnswer("refused", [], 0, blocked=True), "blocked"),
    ],
)
def test_outcome(answer, expected):
    assert outcome(answer) == expected


def test_default_app_starts_not_ready_when_the_mcp_server_is_down(monkeypatch):
    monkeypatch.setenv("MCP_URL", "http://127.0.0.1:1/mcp")
    with TestClient(build_default_app()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
