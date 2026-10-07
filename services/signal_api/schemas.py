"""Request and response models for the signal API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.common import PairId


class SignalRequest(BaseModel):
    """Which pair to score."""

    pair_id: PairId = Field(examples=["KO__PEP"])


class SignalResponse(BaseModel):
    """The model's view of the pair's current entry signal."""

    pair_id: str
    prob: float = Field(ge=0.0, le=1.0, description="Probability the spread reverts")
    take_trade: bool
    threshold: float
    model_version: str
    features: dict[str, float]


class Health(BaseModel):
    """Liveness or readiness status."""

    status: str
