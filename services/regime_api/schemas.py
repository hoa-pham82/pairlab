"""Request and response models for the regime API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.common import PairId
from services.regime_api.drift import Regime


class RegimeRequest(BaseModel):
    """Which pair to assess."""

    pair_id: PairId = Field(examples=["KO__PEP"])


class RegimeResponse(BaseModel):
    """Drift statistics and the resulting regime."""

    pair_id: str
    regime: Regime
    coint_pvalue: float = Field(ge=0.0, le=1.0)
    psi: float = Field(ge=0.0)
    bars_used: int


class Health(BaseModel):
    """Liveness or readiness status."""

    status: str
