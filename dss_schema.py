"""
DSS Schema — Pydantic models for input validation and output serialization.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class TeamProfile(BaseModel):
    """Team macro-attribute vector (A1-A14), all values in [0.0, 1.0]."""
    name: str | None = None
    A1:  float = Field(..., ge=0.0, le=1.0, description="Offensive Strength")
    A2:  float = Field(..., ge=0.0, le=1.0, description="Defensive Strength")
    A3:  float = Field(..., ge=0.0, le=1.0, description="Midfield Control")
    A4:  float = Field(..., ge=0.0, le=1.0, description="Transition Speed")
    A5:  float = Field(..., ge=0.0, le=1.0, description="High Press Capability")
    A6:  float = Field(..., ge=0.0, le=1.0, description="Width Utilization")
    A7:  float = Field(..., ge=0.0, le=1.0, description="Psychological Resilience")
    A8:  float = Field(..., ge=0.0, le=1.0, description="Residual Energy")
    A9:  float = Field(..., ge=0.0, le=1.0, description="Team Morale")
    A10: float = Field(..., ge=0.0, le=1.0, description="Time Management")
    A11: float = Field(..., ge=0.0, le=1.0, description="Tactical Cohesion")
    A12: float = Field(..., ge=0.0, le=1.0, description="Technical Base")
    A13: float = Field(..., ge=0.0, le=1.0, description="Physical Base")
    A14: float = Field(..., ge=0.0, le=1.0, description="Relational Cohesion")

    def to_dict(self) -> dict[str, float]:
        """Return only A1-A14 as dict (excludes name)."""
        return {f"A{i}": getattr(self, f"A{i}") for i in range(1, 15)}


class MatchConditions(BaseModel):
    time_remaining: float = Field(..., ge=0, le=90, description="Minutes remaining")
    score_diff: int = Field(..., description="Goal difference (positive = ahead)")
    fatigue_level: float = Field(..., ge=0.0, le=1.0, description="0=fresh, 1=exhausted")
    morale: float = Field(..., ge=0.0, le=1.0, description="0=demoralized, 1=euphoric")


class Scenario(BaseModel):
    id: str | None = None
    label: str | None = None
    match_conditions: MatchConditions


class DSSConfig(BaseModel):
    opponent_penalty_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    top_n: int = Field(default=5, ge=1, le=20)


class DSSInput(BaseModel):
    input_mode: Literal["macro", "raw"] = "macro"
    team: TeamProfile
    opponent: TeamProfile
    scenarios: list[Scenario] = Field(..., min_length=1)
    config: DSSConfig = DSSConfig()

    @model_validator(mode="after")
    def check_input_mode(self):
        if self.input_mode != "macro":
            raise ValueError(f"input_mode '{self.input_mode}' not yet supported. Use 'macro'.")
        return self


# ---------------------------------------------------------------------------
# Output models (for serialization, not strictly required but useful)
# ---------------------------------------------------------------------------

class StrategyScore(BaseModel):
    strategy: str
    adjusted_distance: float
    raw_distance: float
    category: str


class ScenarioResult(BaseModel):
    scenario_id: str | None
    scenario_label: str | None
    match_conditions: MatchConditions
    best_strategy: StrategyScore
    baseline_strategy: StrategyScore
    ranking: list[StrategyScore]


class DSSMeta(BaseModel):
    version: str
    timestamp: str
    config_used: DSSConfig
    team_name: str | None
    opponent_name: str | None
    total_scenarios: int


class DSSOutput(BaseModel):
    meta: DSSMeta
    results: list[ScenarioResult]
