"""FastAPI app for chatting with the agents, with an A/B split between two variants.

Usage:
    uv run uvicorn services.agent.app:build_default_app --factory --port 8003
Env: LLM_BASE_URL (default http://localhost:8085), LLM_MODEL, MCP_URL
(default http://localhost:8002/mcp), CHALLENGER_SHARE (default 0.5),
CHALLENGER_LLM_BASE_URL, CHALLENGER_LLM_MODEL (default: the champion's),
CHAMPION_PROMPTS (default "default"), CHALLENGER_PROMPTS (default "brief").
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Response, status
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from services.ab import CHALLENGER, CHAMPION, assign_variant
from services.agent.agent import AgentAnswer, ToolAgent
from services.agent.telemetry import AgentTelemetry
from services.common import HttpMetrics

log = logging.getLogger(__name__)
AgentName = Literal["coordinator", "features_analyst", "regime_analyst"]
Variant = Literal["champion", "challenger"]
_SECONDS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120)


class AskRequest(BaseModel):
    """A question for one of the agents."""

    question: str = Field(min_length=1, max_length=2000, examples=["Is KO__PEP still tradeable?"])
    agent: AgentName = "coordinator"
    session_id: str | None = Field(default=None, max_length=128)
    variant: Variant | None = None


class ToolCallOut(BaseModel):
    """One tool call made while answering."""

    name: str
    arguments: dict
    failed: bool


class AskResponse(BaseModel):
    """The agent's answer and a trace of how it was produced."""

    agent: str
    variant: Variant
    answer: str
    tool_calls: list[ToolCallOut]
    steps: int
    input_tokens: int
    output_tokens: int
    blocked: bool
    finished: bool


class Health(BaseModel):
    """Liveness or readiness status."""

    status: str


def create_app(
    agents: dict[str, ToolAgent],
    metrics: HttpMetrics,
    lifespan=None,
    challenger: dict[str, ToolAgent] | None = None,
    challenger_share: float = 0.5,
) -> FastAPI:
    """Build the app around (possibly still empty) champion and challenger registries.

    ``metrics`` must own the registry the agents' telemetry writes to, so
    ``/metrics`` shows HTTP, LLM, agent and tool metrics together. When
    ``challenger`` has agents, ``challenger_share`` of sessions go to it.
    """
    assign_variant("", challenger_share)  # validates the share
    challenger = challenger if challenger is not None else {}
    app = FastAPI(title="pairlab agent API", version="0.2.0", lifespan=lifespan)
    metrics.install(app)
    outcomes = Counter(
        "agent_ab_requests_total",
        "Answered requests by A/B variant, agent and outcome",
        ["variant", "agent", "outcome"],
        registry=metrics.registry,
    )
    seconds = Histogram(
        "agent_ab_seconds",
        "Time to answer one request, by A/B variant",
        ["variant", "agent"],
        buckets=_SECONDS,
        registry=metrics.registry,
    )

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        return Health(status="ok")

    @app.get("/readyz", response_model=Health)
    async def readyz(response: Response) -> Health:
        if not agents:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="not ready")
        return Health(status="ready")

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        variant = request.variant or _route(request, bool(challenger), challenger_share)
        agent = (challenger if variant == CHALLENGER else agents).get(request.agent)
        if agent is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{variant} agents not loaded")
        started = time.perf_counter()
        try:
            answer = await agent.run(request.question)
        except httpx.HTTPError as error:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, f"LLM server error: {type(error).__name__}"
            ) from error
        seconds.labels(variant, request.agent).observe(time.perf_counter() - started)
        outcomes.labels(variant, request.agent, outcome(answer)).inc()
        return _to_response(request.agent, variant, answer)

    return app


def _route(request: AskRequest, has_challenger: bool, share: float) -> str:
    if not has_challenger:
        return CHAMPION
    return assign_variant(request.session_id or request.question, share)


def outcome(answer: AgentAnswer) -> str:
    """Classify a run as blocked, unfinished, tool_failed or answered."""
    if answer.blocked:
        return "blocked"
    if not answer.finished:
        return "unfinished"
    if any(call.failed for call in answer.tool_calls):
        return "tool_failed"
    return "answered"


def _to_response(agent: str, variant: str, answer: AgentAnswer) -> AskResponse:
    return AskResponse(
        agent=agent,
        variant=variant,
        answer=answer.text,
        tool_calls=[
            ToolCallOut(name=c.name, arguments=c.arguments, failed=c.failed)
            for c in answer.tool_calls
        ],
        steps=answer.steps,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        blocked=answer.blocked,
        finished=answer.finished,
    )


def build_default_app() -> FastAPI:
    """App wired to the llama.cpp server(s) and the MCP server from environment variables."""
    from fastmcp import Client

    from services.agent.analysts import PROMPT_SETS, build_agents
    from services.agent.llm import OpenAICompatClient
    from services.agent.mcp_tools import load_mcp_tools

    champion_prompts = PROMPT_SETS[os.environ.get("CHAMPION_PROMPTS", "default")]
    challenger_prompts = PROMPT_SETS[os.environ.get("CHALLENGER_PROMPTS", "brief")]
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8085")
    model = os.environ.get("LLM_MODEL", "qwen2.5-3b-instruct")
    agents: dict[str, ToolAgent] = {}
    challenger: dict[str, ToolAgent] = {}
    metrics = HttpMetrics("agent_api")
    telemetry = AgentTelemetry(metrics.registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient() as http:
            champion_llm = OpenAICompatClient(http, base_url=base_url, model=model)
            challenger_llm = OpenAICompatClient(
                http,
                base_url=os.environ.get("CHALLENGER_LLM_BASE_URL", base_url),
                model=os.environ.get("CHALLENGER_LLM_MODEL", model),
            )
            try:
                async with Client(os.environ.get("MCP_URL", "http://localhost:8002/mcp")) as mcp:
                    tools = {tool.name: tool for tool in await load_mcp_tools(mcp)}
                    features, regime = tools["get_pair_features"], tools["check_regime"]
                    agents.update(
                        build_agents(champion_llm, features, regime, telemetry, champion_prompts)
                    )
                    challenger.update(
                        build_agents(
                            challenger_llm, features, regime, telemetry, challenger_prompts
                        )
                    )
                    yield
            except Exception:
                if agents:
                    raise
                log.exception("MCP server unavailable; /readyz will report not ready")
                yield
            finally:
                agents.clear()
                challenger.clear()

    return create_app(
        agents,
        metrics,
        lifespan,
        challenger=challenger,
        challenger_share=float(os.environ.get("CHALLENGER_SHARE", "0.5")),
    )
