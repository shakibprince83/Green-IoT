from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Classification(str, Enum):
    smoke = "SMOKE"
    pilot = "PILOT"
    official = "OFFICIAL"
    extension = "EXTENSION"


class RunStatus(str, Enum):
    success = "success"
    failed = "failed"
    aborted = "aborted"
    excluded = "excluded"


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    experiment_id: str
    experiment_family: str
    classification: Classification
    protocol_version: str
    protocol_hash: str
    config_hash: str
    timestamp_start: datetime
    timestamp_end: datetime | None = None
    machine_id: str
    hardware_hash: str
    os: str
    python_version: str
    git_commit: str
    git_dirty: bool
    framework: str
    framework_version: str | None = None
    model: str
    architecture_hash: str
    initialization_hash: str
    dataset: str
    dataset_source: str
    dataset_checksum: str | None = None
    split_hash: str
    seed: int
    execution_order: int
    device: str
    precision: str
    measurement_tier: int = Field(ge=1, le=3)
    measurement_backend: str
    measurement_method: str
    measurement_scope: str
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    optimizer: str
    optimizer_hyperparameters: dict[str, Any]
    training_samples: int = Field(ge=0)
    validation_samples: int = Field(ge=0)
    test_samples: int = Field(ge=0)
    train_time_s: float | None = Field(default=None, ge=0)
    time_per_epoch_s: list[float] = Field(default_factory=list)
    inference: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, float | None] = Field(default_factory=dict)
    average_power_w: float | None = Field(default=None, ge=0)
    peak_power_w: float | None = Field(default=None, ge=0)
    training_energy_j: float | None = Field(default=None, ge=0)
    carbon_estimate_kg: float | None = Field(default=None, ge=0)
    carbon_intensity_g_per_kwh: float | None = Field(default=None, ge=0)
    carbon_intensity_source: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    run_status: RunStatus
    warnings: list[str] = Field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def enforce_integrity(self) -> RunRecord:
        if self.measurement_tier == 3 and any(
            x is not None for x in (self.average_power_w, self.peak_power_w, self.training_energy_j)
        ):
            raise ValueError("Tier 3 proxies cannot be stored as power or energy")
        if self.classification is Classification.official and self.dataset == "fixture":
            raise ValueError("Fixture data cannot be official")
        if self.excluded and not self.exclusion_reason:
            raise ValueError("Excluded runs require a reason")
        return self
