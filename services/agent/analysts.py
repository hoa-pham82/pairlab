"""The three agents: features analyst, regime analyst, and their coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from services.agent.agent import ToolAgent, ToolSpec, agent_as_tool
from services.agent.llm import ChatClient
from services.agent.telemetry import AgentTelemetry

_RULES = (
    "Use only numbers returned by tools; never invent values. "
    "If a tool returns an error, say what failed. "
    "You do not make trading decisions: report what the tools say, in plain English."
)

FEATURES_PROMPT = (
    "You are a features analyst for a pairs-trading desk. For the pair in the question, call "
    "get_pair_features once, then summarise the z-score, the model probability and whether the "
    "model would take the trade. " + _RULES
)
REGIME_PROMPT = (
    "You are a regime analyst for a pairs-trading desk. For the pair in the question, call "
    "check_regime once, then state the regime (stable, shifting or broken) with the "
    "cointegration p-value and PSI behind it. " + _RULES
)
COORDINATOR_PROMPT = (
    "You coordinate two analysts to answer whether a pair is still tradeable. Call "
    "ask_features_analyst and ask_regime_analyst for the pair, then give one short answer that "
    "combines both reports. A broken regime means not tradeable, whatever the model says. " + _RULES
)

VERDICT_PREFIXES = ("Tradeable:", "Not tradeable:")
_BRIEF = "Answer in at most two sentences. "


@dataclass(frozen=True)
class PromptSet:
    """System prompts for the three agents."""

    name: str
    features: str
    regime: str
    coordinator: str


DEFAULT_PROMPTS = PromptSet("default", FEATURES_PROMPT, REGIME_PROMPT, COORDINATOR_PROMPT)
BRIEF_PROMPTS = PromptSet(
    "brief",
    FEATURES_PROMPT + " " + _BRIEF,
    REGIME_PROMPT + " " + _BRIEF,
    COORDINATOR_PROMPT
    + " "
    + _BRIEF
    + "Start with 'Tradeable:' or 'Not tradeable:', then the two numbers that decide it.",
)
PROMPT_SETS = {prompts.name: prompts for prompts in (DEFAULT_PROMPTS, BRIEF_PROMPTS)}


def build_agents(
    llm: ChatClient,
    get_pair_features: ToolSpec,
    check_regime: ToolSpec,
    telemetry: AgentTelemetry | None = None,
    prompts: PromptSet = DEFAULT_PROMPTS,
) -> dict[str, ToolAgent]:
    """Create the features analyst, regime analyst and coordinator on shared telemetry."""
    telemetry = telemetry or AgentTelemetry()
    features = ToolAgent("features_analyst", prompts.features, llm, [get_pair_features], telemetry)
    regime = ToolAgent("regime_analyst", prompts.regime, llm, [check_regime], telemetry)
    coordinator = ToolAgent(
        "coordinator",
        prompts.coordinator,
        llm,
        [
            agent_as_tool(
                features, "ask_features_analyst", "Features and model decision for a pair"
            ),
            agent_as_tool(regime, "ask_regime_analyst", "Regime (drift) status for a pair"),
        ],
        telemetry,
        max_steps=5,
    )
    return {agent.name: agent for agent in (features, regime, coordinator)}
