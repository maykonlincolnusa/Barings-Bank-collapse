from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ScenarioLiteral = Literal["healthy_desk", "mild_anomaly", "rogue_trader", "collapse"]


class AuthRequest(BaseModel):
    username: str
    password: str


class SimulationRequest(BaseModel):
    scenario: ScenarioLiteral = "rogue_trader"
    days: int = Field(default=90, ge=30, le=365)
    seed: int = Field(default=42, ge=0, le=100000)
    start_date: str = "1994-12-01"
    exposure_growth_rate: float | None = None
    concealment_strength: float | None = None
    control_weakness_level: float | None = None
    market_shock_sensitivity: float | None = None
    detection_delay: float | None = None


class TrainRequest(BaseModel):
    scenarios: list[ScenarioLiteral] = Field(default_factory=lambda: ["healthy_desk", "mild_anomaly", "rogue_trader", "collapse"])
    days_per_scenario: int = Field(default=120, ge=60, le=365)
    runs_per_scenario: int = Field(default=3, ge=1, le=10)


class PredictRequest(BaseModel):
    scenario: ScenarioLiteral | None = None
    days: int = Field(default=90, ge=30, le=365)
    seed: int = Field(default=42, ge=0, le=100000)
    observations: list[dict[str, Any]] | None = None

    @field_validator("observations")
    @classmethod
    def validate_observations(cls, value: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if value is not None and len(value) == 0:
            raise ValueError("observations cannot be empty")
        return value


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]

