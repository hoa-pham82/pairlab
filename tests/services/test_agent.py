"""Tests for the MCP tool functions, the LLM client, the agent loop, and the PII guard."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from services.agent.ab import CHAMPION, assign_llm_variant
from services.agent.agent import AgentAnswer, ToolAgent, ToolSpec, agent_as_tool
from services.agent.analysts import BRIEF_PROMPTS, DEFAULT_PROMPTS, PROMPT_SETS, build_agents
from services.agent.llm import ChatResult, OpenAICompatClient
from services.agent.safety import find_pii
from services.agent.telemetry import AgentTelemetry
from services.mcp_server.tools import ApiConfig, PairTools

SIGNAL = {"pair_id": "KO__PEP", "prob": 0.7, "take_trade": True, "features": {"zscore": 2.3}}
REGIME = {"pair_id": "KO__PEP", "regime": "stable", "coint_pvalue": 0.01, "psi": 0.05}


def run(coro):
    return asyncio.run(coro)


def _metric(telemetry: AgentTelemetry, name: str, **labels) -> float:
    return telemetry.registry.get_sample_value(name, labels) or 0.0


# ── MCP tool functions ───────────────────────────────────────────────────────


def _tools(handler) -> PairTools:
    return PairTools(httpx.AsyncClient(transport=httpx.MockTransport(handler)), ApiConfig())


class TestPairTools:
    def test_get_pair_features_posts_to_signal_api(self):
        seen = {}

        def handler(request):
            seen.update(url=str(request.url), body=json.loads(request.content))
            return httpx.Response(200, json=SIGNAL)

        assert run(_tools(handler).get_pair_features("KO__PEP")) == SIGNAL
        assert seen == {"url": "http://localhost:8000/signal", "body": {"pair_id": "KO__PEP"}}

    def test_check_regime_posts_to_regime_api(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=REGIME)

        assert run(_tools(handler).check_regime("KO__PEP")) == REGIME
        assert seen["url"] == "http://localhost:8001/regime"

    # Equivalence partitions for pair_id: valid form reaches the API; every
    # malformed form is rejected locally with a 422-style error.
    @pytest.mark.parametrize("pair_id", ["KO_PEP", "ko__pep", "", "KO__", "KO__PEP__X", None, 42])
    def test_invalid_pair_id_never_reaches_the_api(self, pair_id):
        def handler(request):
            raise AssertionError("API must not be called")

        result = run(_tools(handler).get_pair_features(pair_id))
        assert result["status"] == 422 and "error" in result

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (404, {"detail": "no online features for X__Y"}, "no online features for X__Y"),
            (503, {"detail": "model or features not loaded"}, "model or features not loaded"),
            (500, None, "boom"),
        ],
    )
    def test_api_errors_become_error_results(self, status, body, expected):
        def handler(request):
            if body is None:
                return httpx.Response(status, text="boom")
            return httpx.Response(status, json=body)

        assert run(_tools(handler).check_regime("X__Y")) == {"error": expected, "status": status}

    def test_unreachable_service_is_a_503_error_result(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        result = run(_tools(handler).get_pair_features("KO__PEP"))
        assert result == {"error": "service unreachable: ConnectError", "status": 503}


# ── LLM client ───────────────────────────────────────────────────────────────


class TestOpenAICompatClient:
    def _client(self, payload: dict, seen: dict) -> OpenAICompatClient:
        def handler(request):
            seen.update(url=str(request.url), body=json.loads(request.content))
            return httpx.Response(200, json=payload)

        return OpenAICompatClient(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)), base_url="http://llm"
        )

    def test_sends_deterministic_settings_and_parses_usage(self):
        seen: dict = {}
        payload = {
            "model": "qwen",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "timings": {"prompt_ms": 250.0},
        }
        result = run(self._client(payload, seen).chat([{"role": "user", "content": "x"}], []))
        assert seen["url"] == "http://llm/v1/chat/completions"
        assert (seen["body"]["temperature"], seen["body"]["seed"]) == (0.0, 42)
        assert "tools" not in seen["body"]
        assert (result.content, result.model) == ("hi", "qwen")
        assert (result.input_tokens, result.output_tokens) == (12, 3)
        assert result.ttft_seconds == 0.25
        assert result.round_trip_seconds >= 0

    def test_tools_are_forwarded_and_tool_calls_parsed(self):
        seen: dict = {}
        call = {"id": "c1", "function": {"name": "check_regime", "arguments": "{}"}}
        payload = {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}
        tools = [{"type": "function", "function": {"name": "check_regime"}}]
        result = run(self._client(payload, seen).chat([], tools))
        assert seen["body"]["tools"] == tools
        assert result.tool_calls == [call]
        assert (result.content, result.ttft_seconds, result.input_tokens) == ("", None, 0)

    def test_http_error_is_raised(self):
        client = OpenAICompatClient(
            httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        )
        with pytest.raises(httpx.HTTPStatusError):
            run(client.chat([], []))


# ── Agent loop ───────────────────────────────────────────────────────────────


def _tool_request(name: str, arguments, call_id: str = "c1") -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def _asks(*requests: dict) -> ChatResult:
    return ChatResult(
        {"role": "assistant", "content": None, "tool_calls": list(requests)},
        model="fake", input_tokens=10, output_tokens=5, round_trip_seconds=0.2, ttft_seconds=0.1,
    )


def _says(text: str) -> ChatResult:
    return ChatResult(
        {"role": "assistant", "content": text},
        model="fake", input_tokens=20, output_tokens=7, round_trip_seconds=0.3,
    )


class ScriptedLLM:
    """Replays fixed assistant turns and records what it was sent."""

    def __init__(self, *turns: ChatResult):
        self.turns = list(turns)
        self.requests: list[tuple[list[dict], list[dict]]] = []

    async def chat(self, messages, tools):
        self.requests.append(([dict(m) for m in messages], tools))
        return self.turns.pop(0)


def _spec(name: str, result: dict | Exception, seen: list | None = None) -> ToolSpec:
    async def call(pair_id: str) -> dict:
        if seen is not None:
            seen.append(pair_id)
        if isinstance(result, Exception):
            raise result
        return result

    return ToolSpec(name, f"{name} tool", call)


def _agent(llm, *tools, telemetry=None, max_steps=4) -> ToolAgent:
    return ToolAgent("analyst", "system prompt", llm, list(tools), telemetry, max_steps)


class TestToolAgent:
    def test_calls_the_tool_then_answers(self):
        seen: list[str] = []
        llm = ScriptedLLM(
            _asks(_tool_request("check_regime", {"pair_id": "KO__PEP"})), _says("stable")
        )
        answer = run(_agent(llm, _spec("check_regime", REGIME, seen)).run("Is KO__PEP ok?"))

        assert (answer.text, answer.steps, answer.finished) == ("stable", 2, True)
        assert seen == ["KO__PEP"]
        assert [(c.name, c.arguments, c.failed) for c in answer.tool_calls] == [
            ("check_regime", {"pair_id": "KO__PEP"}, False)
        ]
        assert (answer.input_tokens, answer.output_tokens) == (30, 12)

    def test_tool_result_is_fed_back_to_the_model(self):
        llm = ScriptedLLM(
            _asks(_tool_request("check_regime", {"pair_id": "KO__PEP"}, "id-7")), _says("ok")
        )
        run(_agent(llm, _spec("check_regime", REGIME)).run("q"))
        first, second = llm.requests
        assert [m["role"] for m in first[0]] == ["system", "user"]
        assert first[1][0]["function"]["name"] == "check_regime"
        tool_message = second[0][-1]
        assert (tool_message["role"], tool_message["tool_call_id"]) == ("tool", "id-7")
        assert json.loads(tool_message["content"]) == REGIME

    def test_answers_directly_when_no_tool_is_needed(self):
        answer = run(_agent(ScriptedLLM(_says("hello")), _spec("check_regime", REGIME)).run("hi"))
        assert (answer.text, answer.steps, answer.tool_calls) == ("hello", 1, [])

    # Equivalence partitions for a tool request the loop must survive:
    #   unknown tool, malformed JSON arguments, wrong argument names, tool
    #   raising, tool returning an error dict. Each becomes a failed ToolCall
    #   and the conversation continues.
    @pytest.mark.parametrize(
        ("request_", "tool_result", "error_part"),
        [
            (_tool_request("nope", {"pair_id": "KO__PEP"}), REGIME, "unknown tool"),
            (_tool_request("check_regime", "{not json"), REGIME, "bad arguments"),
            (_tool_request("check_regime", {"ticker": "KO"}), REGIME, "bad arguments"),
            (_tool_request("check_regime", {"pair_id": "KO__PEP"}), RuntimeError("x"),
             "tool failed: RuntimeError"),
            (_tool_request("check_regime", {"pair_id": "KO__PEP"}),
             {"error": "no prices", "status": 404}, "no prices"),
        ],
    )
    def test_bad_tool_requests_become_failed_calls_and_the_loop_continues(
        self, request_, tool_result, error_part
    ):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM(_asks(request_), _says("could not check"))
        agent = _agent(llm, _spec("check_regime", tool_result), telemetry=telemetry)
        answer = run(agent.run("q"))

        assert answer.text == "could not check"
        assert answer.tool_calls[0].failed
        assert error_part in answer.tool_calls[0].result["error"]
        name = request_["function"]["name"]
        assert _metric(telemetry, "agent_tool_failures_total", tool=name) == 1

    def test_dict_arguments_are_accepted(self):
        request = {"id": "c", "function": {"name": "check_regime", "arguments": {"pair_id": "A__B"}}}
        seen: list[str] = []
        llm = ScriptedLLM(_asks(request), _says("ok"))
        run(_agent(llm, _spec("check_regime", REGIME, seen)).run("q"))
        assert seen == ["A__B"]

    # Boundary values for max_steps: the model answers on the last allowed
    # step vs still asking for tools when the limit is hit.
    @pytest.mark.parametrize(("tool_turns", "finished"), [(1, True), (2, False), (3, False)])
    def test_step_limit_boundary(self, tool_turns, finished):
        request = _tool_request("check_regime", {"pair_id": "KO__PEP"})
        llm = ScriptedLLM(*[_asks(request)] * tool_turns, _says("done"))
        answer = run(_agent(llm, _spec("check_regime", REGIME), max_steps=2).run("q"))
        assert answer.finished is finished
        assert answer.steps == 2
        assert ("step limit" in answer.text) is (not finished)

    def test_telemetry_counts_agent_tools_tokens_and_latency(self):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM(
            _asks(_tool_request("check_regime", {"pair_id": "KO__PEP"})), _says("ok")
        )
        run(_agent(llm, _spec("check_regime", REGIME), telemetry=telemetry).run("q"))

        assert _metric(telemetry, "agent_calls_total", agent="analyst") == 1
        assert _metric(telemetry, "agent_tool_calls_total", tool="check_regime") == 1
        assert _metric(telemetry, "agent_tool_failures_total", tool="check_regime") == 0
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="input", variant="champion") == 30
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="output", variant="champion") == 12
        assert _metric(telemetry, "llm_round_trip_seconds_count", model="fake", variant="champion") == 2
        assert _metric(telemetry, "llm_time_to_first_token_seconds_count", model="fake", variant="champion") == 1


class TestPiiGuard:
    # Equivalence partitions: each kind of personal data, clean prompts, and
    # look-alikes that must not be flagged (pair IDs, prices, dates).
    @pytest.mark.parametrize(
        ("text", "kinds"),
        [
            ("mail me at hoa@example.com", ["email"]),
            ("card 4111 1111 1111 1111", ["card_number"]),
            ("call +84 912 345 678", ["phone"]),
            ("hoa@example.com or +1 415 555 2671", ["email", "phone"]),
            ("Is KO__PEP still tradeable?", []),
            ("price was 123.45 on 2024-03-01 with z-score -2.31", []),
            ("", []),
        ],
    )
    def test_find_pii(self, text, kinds):
        assert find_pii(text) == kinds

    def test_prompt_with_pii_is_refused_before_the_model_sees_it(self):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM()
        answer = run(_agent(llm, telemetry=telemetry).run("my email is hoa@example.com"))
        assert answer.blocked and answer.steps == 0 and "email" in answer.text
        assert llm.requests == []
        assert _metric(telemetry, "agent_pii_blocked_total", kind="email") == 1
        assert _metric(telemetry, "agent_calls_total", agent="analyst") == 1


class TestCoordinator:
    def _agents(self, llm, regime=REGIME, telemetry=None):
        return build_agents(
            llm, _spec("get_pair_features", SIGNAL), _spec("check_regime", regime), telemetry
        )

    def test_coordinator_delegates_to_both_analysts_and_combines(self):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM(
            _asks(
                _tool_request("ask_features_analyst", {"pair_id": "KO__PEP"}, "a"),
                _tool_request("ask_regime_analyst", {"pair_id": "KO__PEP"}, "b"),
            ),
            _asks(_tool_request("get_pair_features", {"pair_id": "KO__PEP"})),
            _says("z-score 2.3, probability 0.7, take"),
            _asks(_tool_request("check_regime", {"pair_id": "KO__PEP"})),
            _says("regime stable"),
            _says("KO__PEP is tradeable: stable regime, model would take it."),
        )
        answer = run(self._agents(llm, telemetry=telemetry)["coordinator"].run("Is KO__PEP ok?"))

        assert answer.text.startswith("KO__PEP is tradeable")
        assert [c.name for c in answer.tool_calls] == [
            "ask_features_analyst", "ask_regime_analyst",
        ]
        assert answer.tool_calls[0].result == {
            "agent": "features_analyst", "answer": "z-score 2.3, probability 0.7, take",
        }
        for agent in ("coordinator", "features_analyst", "regime_analyst"):
            assert _metric(telemetry, "agent_calls_total", agent=agent) == 1
        assert _metric(telemetry, "agent_tool_calls_total", tool="check_regime") == 1

    def test_sub_agent_tool_failure_is_reported_to_the_coordinator(self):
        llm = ScriptedLLM(
            _asks(_tool_request("ask_regime_analyst", {"pair_id": "X__Y"})),
            _asks(_tool_request("check_regime", {"pair_id": "X__Y"})),
            _says("the regime check failed"),
            _says("Cannot assess X__Y: the regime check failed."),
        )
        agents = self._agents(llm, regime={"error": "no prices for X__Y", "status": 404})
        answer = run(agents["coordinator"].run("Is X__Y ok?"))
        assert answer.tool_calls[0].failed
        assert "check_regime" in answer.tool_calls[0].result["error"]

    def test_sub_agent_hitting_its_step_limit_is_an_error(self):
        request = _tool_request("check_regime", {"pair_id": "KO__PEP"})
        sub = _agent(ScriptedLLM(_asks(request)), _spec("check_regime", REGIME), max_steps=1)
        report = run(agent_as_tool(sub, "ask", "d").call("KO__PEP"))
        assert report["error"] == "step limit"

    def test_prompt_set_reaches_every_agent(self):
        llm = ScriptedLLM(_says("Tradeable: p 0.01, prob 0.7."))
        agents = build_agents(
            llm, _spec("get_pair_features", SIGNAL), _spec("check_regime", REGIME),
            prompts=BRIEF_PROMPTS,
        )
        run(agents["coordinator"].run("Is KO__PEP ok?"))
        assert llm.requests[0][0][0]["content"] == BRIEF_PROMPTS.coordinator
        assert "Tradeable:" in BRIEF_PROMPTS.coordinator
        assert "Tradeable:" not in DEFAULT_PROMPTS.coordinator
        assert set(PROMPT_SETS) == {"default", "brief"}


class TestIdempotency:
    """Property-based: the same question with the same model turns gives the same answer."""

    @settings(
        max_examples=50, deadline=None,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    @given(
        pair=st.from_regex(r"[A-Z]{1,5}__[A-Z]{1,5}", fullmatch=True),
        psi=st.floats(0, 5, allow_nan=False),
        repeats=st.integers(2, 4),
    )
    def test_repeated_runs_give_identical_answers(self, pair, psi, repeats):
        def one_run() -> AgentAnswer:
            llm = ScriptedLLM(
                _asks(_tool_request("check_regime", {"pair_id": pair})), _says(f"psi {psi}")
            )
            return run(_agent(llm, _spec("check_regime", {"psi": psi})).run(f"check {pair}"))

        answers = [one_run() for _ in range(repeats)]
        assert all(answer == answers[0] for answer in answers)
        assert answers[0].tool_calls[0].arguments == {"pair_id": pair}


# ── A/B variant routing ───────────────────────────────────────────────────────


class TestAssignLlmVariant:
    # Equivalence partitions: always champion when share=0, always challenger
    # when share=1, and stable (same result on repeated calls) in between.
    def test_share_zero_always_champion(self):
        for q in ("Is KO__PEP ok?", "check XOM__CVX", "tell me about GS__MS"):
            assert assign_llm_variant(q, 0.0) == CHAMPION

    def test_share_one_always_challenger(self):
        for q in ("Is KO__PEP ok?", "check XOM__CVX", "tell me about GS__MS"):
            assert assign_llm_variant(q, 1.0) == "challenger"

    def test_same_question_gives_same_variant(self):
        q = "Is KO__PEP still tradeable?"
        first = assign_llm_variant(q, 0.5)
        assert all(assign_llm_variant(q, 0.5) == first for _ in range(20))

    def test_50_percent_split_is_roughly_even(self):
        # Property: with 1000 distinct questions and share=0.5, between 40% and 60%
        # should land in each bucket (exact rate depends on SHA-256 distribution).
        questions = [f"Is PAIR{i}__PAIR{i + 1} ok?" for i in range(1000)]
        challengers = sum(1 for q in questions if assign_llm_variant(q, 0.5) == "challenger")
        assert 400 <= challengers <= 600


class TestAgentVariantTelemetry:
    """Variant label flows all the way from run() to Prometheus counters."""

    def test_champion_variant_labels_telemetry(self):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM(_says("ok"))
        run(_agent(llm, telemetry=telemetry).run("q", variant="champion"))
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="input", variant="champion") > 0
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="input", variant="challenger") == 0

    def test_challenger_variant_labels_telemetry(self):
        telemetry = AgentTelemetry()
        llm = ScriptedLLM(_says("ok"))
        run(_agent(llm, telemetry=telemetry).run("q", variant="challenger"))
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="input", variant="challenger") > 0
        assert _metric(telemetry, "llm_tokens_total", model="fake", kind="input", variant="champion") == 0

    def test_variant_flows_to_answer(self):
        llm = ScriptedLLM(_says("ok"))
        answer = run(_agent(llm).run("q", variant="challenger"))
        assert answer.variant == "challenger"

    def test_pii_block_preserves_variant(self):
        answer = run(_agent(ScriptedLLM()).run("email me at x@example.com", variant="challenger"))
        assert answer.blocked and answer.variant == "challenger"
