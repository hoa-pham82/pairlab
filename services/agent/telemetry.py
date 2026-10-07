"""Prometheus metrics for LLM calls, agents, and tools."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

_SECONDS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)


class AgentTelemetry:
    """All agent-side metrics on one registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.agent_calls = Counter(
            "agent_calls_total", "Times each agent was run", ["agent"], registry=self.registry
        )
        self.tool_calls = Counter(
            "agent_tool_calls_total",
            "Times each tool was called",
            ["tool"],
            registry=self.registry,
        )
        self.tool_failures = Counter(
            "agent_tool_failures_total",
            "Tool calls that returned an error",
            ["tool"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "llm_tokens_total",
            "Tokens by kind (input, output)",
            ["model", "kind"],
            registry=self.registry,
        )
        self.round_trip = Histogram(
            "llm_round_trip_seconds",
            "Time for one LLM generation",
            ["model"],
            buckets=_SECONDS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "llm_time_to_first_token_seconds",
            "Time until the first generated token",
            ["model"],
            buckets=_SECONDS,
            registry=self.registry,
        )
        self.pii_blocked = Counter(
            "agent_pii_blocked_total",
            "Prompts refused for containing personal data",
            ["kind"],
            registry=self.registry,
        )
