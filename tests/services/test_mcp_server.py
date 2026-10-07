"""Tests for the MCP server and the agent-side MCP adapter, over an in-memory connection."""

from __future__ import annotations

import asyncio
import json

import httpx
from fastmcp import Client
from services.agent.agent import ToolAgent
from services.agent.llm import ChatResult
from services.agent.mcp_tools import load_mcp_tools
from services.mcp_server.server import build_server, config_from_env
from services.mcp_server.tools import ApiConfig, PairTools
from starlette.testclient import TestClient

SIGNAL = {"pair_id": "KO__PEP", "prob": 0.7, "take_trade": True}
REGIME = {"pair_id": "KO__PEP", "regime": "stable", "psi": 0.05}


def _server(calls: list[str] | None = None):
    def handler(request):
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path == "/signal":
            return httpx.Response(200, json=SIGNAL)
        if json.loads(request.content)["pair_id"] == "NOPE__X":
            return httpx.Response(404, json={"detail": "no prices for NOPE__X"})
        return httpx.Response(200, json=REGIME)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return build_server(PairTools(client, ApiConfig()))


def test_server_lists_both_tools_with_descriptions_and_schema():
    async def scenario():
        async with Client(_server()) as client:
            return {tool.name: tool for tool in await client.list_tools()}

    tools = asyncio.run(scenario())
    assert set(tools) == {"get_pair_features", "check_regime"}
    assert "KO__PEP" in tools["check_regime"].description
    assert tools["check_regime"].inputSchema["required"] == ["pair_id"]


def test_tools_return_the_api_payload_over_mcp():
    calls: list[str] = []

    async def scenario():
        async with Client(_server(calls)) as client:
            features = await client.call_tool("get_pair_features", {"pair_id": "KO__PEP"})
            regime = await client.call_tool("check_regime", {"pair_id": "KO__PEP"})
            return features.data, regime.data

    assert asyncio.run(scenario()) == (SIGNAL, REGIME)
    assert calls == ["/signal", "/regime"]


def test_api_error_reaches_the_client_as_an_error_result_not_an_exception():
    async def scenario():
        async with Client(_server()) as client:
            return (await client.call_tool("check_regime", {"pair_id": "NOPE__X"})).data

    assert asyncio.run(scenario()) == {"error": "no prices for NOPE__X", "status": 404}


def test_healthz_route():
    app = _server().http_app()
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("SIGNAL_API_URL", "http://signal_api:8000")
    monkeypatch.delenv("REGIME_API_URL", raising=False)
    config = config_from_env()
    assert config.signal_url == "http://signal_api:8000"
    assert config.regime_url == "http://localhost:8001"


class OneToolThenAnswer:
    """Fake model: asks for check_regime once, then answers with the tool's regime."""

    def __init__(self, pair_id: str):
        self.pair_id = pair_id
        self.turn = 0

    async def chat(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            call = {
                "id": "c1",
                "function": {
                    "name": "check_regime",
                    "arguments": json.dumps({"pair_id": self.pair_id}),
                },
            }
            return ChatResult({"role": "assistant", "tool_calls": [call]})
        tool_result = json.loads(messages[-1]["content"])
        return ChatResult({"role": "assistant", "content": json.dumps(tool_result)})


def _run_agent(pair_id: str, wrong_argument: bool = False):
    async def scenario():
        async with Client(_server()) as client:
            specs = await load_mcp_tools(client)
            if wrong_argument:
                regime = next(spec for spec in specs if spec.name == "check_regime")
                return specs, await regime.call(ticker="KO")
            agent = ToolAgent("regime_analyst", "prompt", OneToolThenAnswer(pair_id), specs)
            return specs, await agent.run(f"check {pair_id}")

    return asyncio.run(scenario())


def test_agent_calls_tools_through_mcp():
    specs, answer = _run_agent("KO__PEP")
    assert {spec.name for spec in specs} == {"get_pair_features", "check_regime"}
    assert specs[0].schema()["function"]["parameters"]["required"] == ["pair_id"]
    assert json.loads(answer.text) == REGIME
    assert not answer.tool_calls[0].failed


def test_api_error_over_mcp_is_a_failed_tool_call_for_the_agent():
    _, answer = _run_agent("NOPE__X")
    assert answer.tool_calls[0].failed
    assert answer.tool_calls[0].result["status"] == 404


def test_schema_violation_over_mcp_becomes_an_error_result():
    _, result = _run_agent("KO__PEP", wrong_argument=True)
    assert "error" in result
