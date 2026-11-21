"""Pydantic request/response schemas: they validate input and generate the OpenAPI (Swagger) documentation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXAMPLE_TX = "TXN_IND_202608_001768"


class ScoreRequest(BaseModel):
    transaction_id: str | None = Field(None, description="Score a stored transaction (point-in-time features).",
                                       examples=[EXAMPLE_TX])
    features: dict[str, Any] | None = Field(None, description="Or: a complete Exp C feature row (49 columns + amount_inr).")
    overrides: dict[str, Any] = Field(default_factory=dict,
                                      description="What-if changes applied on top, e.g. {'amount_inr': 95000, 'is_new_device': 1}.")
    counterparty_name: str | None = Field(None, description="Extra name to screen against OFAC + UN (besides the sender).")
    explain: bool = Field(True, description="Return the top feature contributions (native TreeSHAP).")

    @model_validator(mode="after")
    def one_source(self):
        if (self.transaction_id is None) == (self.features is None):
            raise ValueError("provide exactly one of transaction_id or features")
        return self


class Component(BaseModel):
    model_probability: float
    anomaly_pct: float
    graph_risk: float


class ScoreResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    transaction_id: str
    risk_score: float = Field(description="0–100 after overrides")
    tier: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    base_score: float
    components: Component
    weights: dict[str, float]
    overrides_applied: list[str]
    decision: dict[str, Any]
    reasons: list[dict[str, Any]] = []
    network: dict[str, Any]
    rules: dict[str, Any]
    cluster: dict[str, Any]
    anomaly: dict[str, Any]
    sanctions: list[dict[str, Any]]
    context: dict[str, Any]
    timings_ms: dict[str, float]
    mode: str
    model_version: str


class AnomalyRequest(BaseModel):
    transaction_id: str | None = Field(None, examples=[EXAMPLE_TX])
    features: dict[str, Any] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_source(self):
        if (self.transaction_id is None) == (self.features is None):
            raise ValueError("provide exactly one of transaction_id or features")
        return self


class RowsRequest(BaseModel):
    """Internal service-to-service payload: feature rows (JSON objects)."""
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
    explain_top: int = Field(6, ge=0, le=20)


class NamesRequest(BaseModel):
    """Internal: distinct names to screen in one pass (batch analysis)."""
    names: list[str] = Field(max_length=30000)
    threshold: float = Field(90.0, ge=50, le=100)


class DemoRequest(BaseModel):
    sample: int | None = Field(None, ge=100, le=10000, description="Analyse a stratified sample of this many rows (default: all 10,000).")


class SimulateRequest(BaseModel):
    attacks: list[str] = Field(min_length=1, max_length=5, examples=[["fraud_ring", "mule_chain"]],
                               description="Any of fraud_ring, mule_chain, burst, structuring, shared_device.")
    size: int = Field(10, ge=3, le=40, description="Intensity: senders in the ring, payments in the burst, …")
    camouflage: bool = Field(False, description="Make each payment look normal on behaviour (only the network gives it away).")
    background: int = Field(1500, ge=200, le=5000, description="Normal payments in the simulated day.")
    seed: int | None = Field(None, description="Random seed (fixed = reproducible attack).")
