"""Tool-calling agent loop, and a coordinator that uses other agents as tools."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from services.agent.llm import ChatClient, ChatResult
from services.agent.safety import find_pii
from services.agent.telemetry import AgentTelemetry

Tool = Callable[..., Awaitable[dict]]

PAIR_ID_PARAMETERS = {
    "type": "object",
    "properties": {
        "pair_id": {"type": "string", "description": "Two tickers joined by __, e.g. KO__PEP"}
    },
    "required": ["pair_id"],
}


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool with the description the model sees."""

    name: str
    description: str
    call: Tool
    parameters: dict = field(default_factory=lambda: PAIR_ID_PARAMETERS)

    def schema(self) -> dict:
        """OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation made during a run."""

    name: str
    arguments: dict
    result: dict

    @property
    def failed(self) -> bool:
        return "error" in self.result


@dataclass(frozen=True)
class AgentAnswer:
    """The agent's final text and how it got there."""

    text: str
    tool_calls: list[ToolCall]
    steps: int
    input_tokens: int = 0
    output_tokens: int = 0
    blocked: bool = False
    finished: bool = True


class ToolAgent:
    """Asks the model, runs the tools it requests, repeats until it answers in text."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: ChatClient,
        tools: list[ToolSpec],
        telemetry: AgentTelemetry | None = None,
        max_steps: int = 4,
    ) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._telemetry = telemetry or AgentTelemetry()
        self._max_steps = max_steps

    async def run(self, question: str) -> AgentAnswer:
        """Answer ``question``, refusing prompts that contain personal data."""
        self._telemetry.agent_calls.labels(self.name).inc()
        pii = find_pii(question)
        if pii:
            for kind in pii:
                self._telemetry.pii_blocked.labels(kind).inc()
            return AgentAnswer(
                f"Request refused: it contains personal data ({', '.join(pii)}).",
                tool_calls=[],
                steps=0,
                blocked=True,
            )

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        schemas = [tool.schema() for tool in self._tools.values()]
        calls: list[ToolCall] = []
        tokens_in = tokens_out = 0

        for step in range(1, self._max_steps + 1):
            result = await self._llm.chat(messages, schemas)
            self._record(result)
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            if not result.tool_calls:
                return AgentAnswer(result.content, calls, step, tokens_in, tokens_out)

            messages.append(result.message)
            for request in result.tool_calls:
                call = await self._run_tool(request)
                calls.append(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": request.get("id", ""),
                        "content": json.dumps(call.result),
                    }
                )

        return AgentAnswer(
            "No answer: the step limit was reached before the model finished.",
            calls,
            self._max_steps,
            tokens_in,
            tokens_out,
            finished=False,
        )

    async def _run_tool(self, request: dict) -> ToolCall:
        """Run one requested tool; bad names, arguments, and crashes become error results."""
        function = request.get("function") or {}
        name = function.get("name", "")
        self._telemetry.tool_calls.labels(name or "unknown").inc()
        arguments: dict = {}
        tool = self._tools.get(name)
        if tool is None:
            result = {"error": f"unknown tool: {name}"}
        else:
            try:
                raw = function.get("arguments") or "{}"
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                result = await tool.call(**arguments)
            except (json.JSONDecodeError, TypeError) as error:
                result = {"error": f"bad arguments: {error}"}
            except Exception as error:  # a failing tool must not end the conversation
                result = {"error": f"tool failed: {type(error).__name__}"}
        call = ToolCall(name, arguments, result)
        if call.failed:
            self._telemetry.tool_failures.labels(name or "unknown").inc()
        return call

    def _record(self, result: ChatResult) -> None:
        self._telemetry.tokens.labels(result.model, "input").inc(result.input_tokens)
        self._telemetry.tokens.labels(result.model, "output").inc(result.output_tokens)
        self._telemetry.round_trip.labels(result.model).observe(result.round_trip_seconds)
        if result.ttft_seconds is not None:
            self._telemetry.ttft.labels(result.model).observe(result.ttft_seconds)


def agent_as_tool(agent: ToolAgent, name: str, description: str) -> ToolSpec:
    """Expose an agent as a tool, so a coordinator can delegate a pair to it."""

    async def ask(pair_id: str) -> dict:
        answer = await agent.run(f"Report on pair {pair_id}.")
        failed = [call.name for call in answer.tool_calls if call.failed]
        report = {"agent": agent.name, "answer": answer.text}
        if failed or not answer.finished:
            report["error"] = f"incomplete: failed tools {failed}" if failed else "step limit"
        return report

    return ToolSpec(name, description, ask)
