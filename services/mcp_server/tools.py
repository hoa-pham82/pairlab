"""Tool functions behind the MCP server: thin, validated calls to the two web APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from services.common import PAIR_ID_PATTERN

_PAIR_ID = re.compile(PAIR_ID_PATTERN)


@dataclass(frozen=True)
class ApiConfig:
    """Where the web APIs live."""

    signal_url: str = "http://localhost:8000"
    regime_url: str = "http://localhost:8001"
    timeout_seconds: float = 10.0


class PairTools:
    """The tools an agent can call. Each returns a JSON-ready dict.

    Failures come back as ``{"error": ..., "status": ...}`` rather than
    exceptions, so the model can read what went wrong and explain it.
    """

    def __init__(self, client: httpx.AsyncClient, config: ApiConfig | None = None) -> None:
        self._client = client
        self._config = config or ApiConfig()

    async def get_pair_features(self, pair_id: str) -> dict:
        """Latest online features for a pair and the model's take/skip decision.

        Args:
            pair_id: two tickers joined by a double underscore, e.g. ``KO__PEP``.
        """
        return await self._post(f"{self._config.signal_url}/signal", pair_id)

    async def check_regime(self, pair_id: str) -> dict:
        """Drift check for a pair: regime (stable, shifting, broken) with its statistics.

        Args:
            pair_id: two tickers joined by a double underscore, e.g. ``KO__PEP``.
        """
        return await self._post(f"{self._config.regime_url}/regime", pair_id)

    async def _post(self, url: str, pair_id: str) -> dict:
        if not isinstance(pair_id, str) or not _PAIR_ID.fullmatch(pair_id):
            return {"error": "pair_id must look like KO__PEP", "status": 422}
        try:
            response = await self._client.post(
                url, json={"pair_id": pair_id}, timeout=self._config.timeout_seconds
            )
        except httpx.HTTPError as error:
            return {"error": f"service unreachable: {type(error).__name__}", "status": 503}
        if response.status_code != 200:
            return {"error": _detail(response), "status": response.status_code}
        return response.json()


def _detail(response: httpx.Response) -> str:
    """Pull the API's error message out of a non-200 response."""
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text
