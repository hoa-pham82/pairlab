"""MCP server: registers the pair tools and serves them over streamable HTTP.

Usage:
    uv run python -m services.mcp_server.server
Env: SIGNAL_API_URL, REGIME_API_URL, MCP_HOST (default 0.0.0.0), MCP_PORT (default 8002).
"""

from __future__ import annotations

import os

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.mcp_server.tools import ApiConfig, PairTools


def build_server(tools: PairTools) -> FastMCP:
    """Create an MCP server exposing ``get_pair_features`` and ``check_regime``."""
    server = FastMCP(
        "pairlab",
        instructions="Tools for inspecting pairs: online features with the model's decision, "
        "and a drift (regime) check. Pair IDs look like KO__PEP.",
    )

    @server.tool
    async def get_pair_features(pair_id: str) -> dict:
        """Latest features for a pair and the model's take/skip decision.

        Returns zscore, hedge_ratio, spread_vol, correlation_60d, prob, take_trade
        and model_version; or an error. pair_id example: KO__PEP.
        """
        return await tools.get_pair_features(pair_id)

    @server.tool
    async def check_regime(pair_id: str) -> dict:
        """Drift check for a pair: regime (stable, shifting, broken), coint_pvalue and psi.

        Returns an error for unknown pairs. pair_id example: KO__PEP.
        """
        return await tools.check_regime(pair_id)

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return server


def config_from_env() -> ApiConfig:
    """Read the API addresses from the environment."""
    defaults = ApiConfig()
    return ApiConfig(
        signal_url=os.environ.get("SIGNAL_API_URL", defaults.signal_url),
        regime_url=os.environ.get("REGIME_API_URL", defaults.regime_url),
    )


if __name__ == "__main__":
    build_server(PairTools(httpx.AsyncClient(), config_from_env())).run(
        transport="http",
        host=os.environ.get("MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("MCP_PORT", "8002")),
    )
