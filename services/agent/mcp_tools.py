"""Adapter: turn the tools of a connected MCP client into agent ToolSpecs."""

from __future__ import annotations

from services.agent.agent import ToolSpec


async def load_mcp_tools(client) -> list[ToolSpec]:
    """List the server's tools and wrap each as a ToolSpec that calls it over MCP.

    Args:
        client: a connected ``fastmcp.Client``.
    """
    return [_wrap(client, tool) for tool in await client.list_tools()]


def _wrap(client, tool) -> ToolSpec:
    async def call(**arguments) -> dict:
        result = await client.call_tool(tool.name, arguments, raise_on_error=False)
        if result.is_error:
            text = " ".join(getattr(part, "text", "") for part in result.content).strip()
            return {"error": text or "tool call failed"}
        data = result.data
        return data if isinstance(data, dict) else {"result": data}

    return ToolSpec(tool.name, (tool.description or "").strip(), call, tool.inputSchema)
